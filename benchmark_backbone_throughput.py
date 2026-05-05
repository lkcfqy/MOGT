import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark_synthetic_last_value import value_gate_input_dim
from model_baseline_transformer import TransformerForCausalLM, choose_num_heads
from model_mogt import MOGTForCausalLM


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def autocast_context(device: torch.device, dtype_name: str):
    if device.type == "cuda" and dtype_name == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and dtype_name == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.no_grad()


def build_mogt(args, device: torch.device):
    model = MOGTForCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        r=args.rank,
    ).to(device)
    for block in model.mogt.blocks:
        block.scan_impl = args.scan_impl
        block.connection_impl = args.connection_impl
        block.connection_damping = args.connection_damping
        block.scan_block_size = args.scan_block_size
        if args.value_gate:
            value_gate_dim = args.rank if args.value_gate_width == "rank" else 1
            block.phi_value_gate = torch.nn.Linear(
                value_gate_input_dim(args.d_model, args.value_gate_input),
                value_gate_dim,
                bias=True,
            ).to(device)
            torch.nn.init.zeros_(block.phi_value_gate.weight)
            torch.nn.init.constant_(block.phi_value_gate.bias, args.value_gate_bias)
            block.value_gate_input_mode = args.value_gate_input
            block.couple_forget_to_value_gate = bool(args.couple_forget_to_value_gate)
    if args.prefix_condition_position is not None:
        model.mogt.prefix_condition_position = args.prefix_condition_position
    model.eval()
    return model


def build_transformer(args, device: torch.device):
    model = TransformerForCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=choose_num_heads(args.d_model, args.num_heads),
        rope_theta=args.rope_theta,
    ).to(device)
    model.eval()
    return model


@torch.inference_mode()
def time_backbone(model, model_type: str, input_ids: torch.Tensor, args, device: torch.device):
    def run_once():
        with autocast_context(device, args.dtype):
            if model_type == "mogt":
                return model.mogt(input_ids)
            return model.transformer(input_ids)

    for _ in range(args.warmup):
        _ = run_once()
    sync(device)
    started = time.perf_counter()
    for _ in range(args.iters):
        _ = run_once()
    sync(device)
    return (time.perf_counter() - started) * 1000.0 / args.iters


def benchmark_model(model_type: str, args, device: torch.device):
    model = build_mogt(args, device) if model_type == "mogt" else build_transformer(args, device)
    parameter_count = sum(param.numel() for param in model.parameters())
    rows = []
    for length in args.lengths:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, length),
            device=device,
            dtype=torch.long,
        )
        try:
            elapsed_ms = time_backbone(model, model_type, input_ids, args, device)
            status = "ok"
            error = None
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elapsed_ms = None
            status = "oom"
            error = str(exc)
        rows.append(
            {
                "length": int(length),
                "status": status,
                "elapsed_ms": elapsed_ms,
                "tokens_per_second": (
                    args.batch_size * length * 1000.0 / elapsed_ms
                    if elapsed_ms and elapsed_ms > 0
                    else None
                ),
                "peak_memory_mb": peak_memory_mb(device),
                "error": error,
            }
        )
    return parameter_count, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Backbone-level MOGT vs Transformer throughput.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--rope-theta", type=float, default=0.0)
    parser.add_argument("--scan-impl", default="triton_hybrid")
    parser.add_argument("--connection-impl", default="identity")
    parser.add_argument("--connection-damping", type=float, default=1.0)
    parser.add_argument("--scan-block-size", type=int, default=256)
    parser.add_argument("--value-gate", action="store_true")
    parser.add_argument("--value-gate-width", choices=["scalar", "rank"], default="rank")
    parser.add_argument(
        "--value-gate-input",
        choices=["current", "current_prev", "current_prev_prefix"],
        default="current_prev_prefix",
    )
    parser.add_argument("--value-gate-bias", type=float, default=-2.0)
    parser.add_argument("--couple-forget-to-value-gate", action="store_true")
    parser.add_argument("--prefix-condition-position", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--output-json",
        default="benchmark_runs/backbone_throughput_identity_coupled_d768_l2_20260504.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    records = []
    for model_type in ("mogt", "transformer"):
        parameter_count, rows = benchmark_model(model_type, args, device)
        records.append(
            {
                "model": model_type,
                "parameter_count": parameter_count,
                "results": rows,
            }
        )
        for row in rows:
            elapsed = "oom" if row["elapsed_ms"] is None else f"{row['elapsed_ms']:.2f}ms"
            print(f"{model_type} L={row['length']} {elapsed}", flush=True)

    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    payload = {
        "schema_version": "mogt-backbone-throughput-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "backbone_throughput",
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "config": vars(args),
        "records": records,
        "notes": (
            "Backbone hidden-state forward only: embeddings + sequence blocks + final norm. "
            "No LM head, no loss, no backward pass."
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
