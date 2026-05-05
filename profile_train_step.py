import argparse
import json
from pathlib import Path
import time
import os

DEFAULT_ALLOC_CONF = os.environ.get("MOGT_ALLOC_CONF", "expandable_segments:True")
if DEFAULT_ALLOC_CONF:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", DEFAULT_ALLOC_CONF)

import torch
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, profile

from model_mogt import MOGTForCausalLM


INTERESTING_EVENTS = [
    "train_step.total",
    "train_step.forward",
    "train_step.backward",
    "train_step.optimizer",
    "mogt.connection",
    "mogt.connection_map",
    "mogt.matrix_exp",
    "mogt.cayley_solve",
    "mogt.value_projection",
    "mogt.affine_scan",
    "affine_backward.recompute_forward",
    "affine_backward.reverse_scan",
    "affine_backward.recompute_prefix_hybrid",
    "affine_backward.reverse_scan_hybrid",
    "affine_backward.grad_u_matmul",
    "triton_hybrid.forward",
    "triton_hybrid.local_scan",
    "triton_hybrid.carry_scan",
    "triton_hybrid.carry_apply",
    "triton_hybrid.backward_recompute",
    "mogt.lm_head_loss",
    "mogt.readout",
    "mogt.ffn",
]


def resolve_block_carry_scan(impl: str, seq_len: int, requested: str) -> str:
    if requested not in {"auto", "sequential", "doubling"}:
        raise ValueError("carry_scan must be one of: auto, sequential, doubling")
    if requested != "auto":
        return requested
    if impl == "triton_hybrid" and seq_len >= 32768:
        return "doubling"
    return "sequential"


def parse_args():
    parser = argparse.ArgumentParser(description="Profile a single MOGT training step.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--impls", nargs="+", default=["sequential", "triton_hybrid"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--block-c", type=int, default=16)
    parser.add_argument("--carry-scan", choices=["auto", "sequential", "doubling"], default="auto")
    parser.add_argument("--connection-impl", choices=["matrix_exp", "cayley"], default="matrix_exp")
    parser.add_argument("--connection-damping", type=float, default=0.999)
    parser.add_argument("--loss-chunk-size", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="profile_runs")
    args = parser.parse_args()
    if args.loss_chunk_size <= 0:
        args.loss_chunk_size = 256 if args.seq_len >= 32768 else 4096
    args.gradient_checkpointing = (
        args.gradient_checkpointing == "on"
        or (args.gradient_checkpointing == "auto" and args.seq_len >= 32768)
    )
    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_scan_impl(model: MOGTForCausalLM, impl: str, args):
    resolved_carry_scan = resolve_block_carry_scan(impl, args.seq_len, args.carry_scan)
    for block in model.mogt.blocks:
        block.scan_impl = impl
        block.scan_block_size = args.block_size
        block.scan_block_c = args.block_c
        block.block_carry_scan = resolved_carry_scan
        block.connection_impl = args.connection_impl
        block.connection_damping = args.connection_damping
    return resolved_carry_scan


def get_amp_dtype(device: str):
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def make_batch(args, device: str):
    input_ids = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
    return input_ids, labels


def maybe_synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def event_total_ms(evt, use_cuda: bool):
    if use_cuda:
        total_us = getattr(evt, "cuda_time_total", 0.0)
        if not total_us:
            total_us = getattr(evt, "device_time_total", 0.0)
        return float(total_us) / 1000.0
    return float(evt.cpu_time_total) / 1000.0


def event_self_ms(evt, use_cuda: bool):
    if use_cuda:
        self_us = getattr(evt, "self_cuda_time_total", 0.0)
        if not self_us:
            self_us = getattr(evt, "self_device_time_total", 0.0)
        return float(self_us) / 1000.0
    return float(evt.self_cpu_time_total) / 1000.0


def collect_event_summary(prof, use_cuda: bool):
    summary = {}
    for evt in prof.key_averages():
        if evt.key in INTERESTING_EVENTS:
            summary[evt.key] = {
                "count": int(evt.count),
                "total_ms": event_total_ms(evt, use_cuda),
                "self_ms": event_self_ms(evt, use_cuda),
            }
    return summary


def run_train_step(
    model,
    optimizer,
    scaler,
    input_ids,
    labels,
    device: str,
    amp_dtype,
    grad_accum_steps: int,
    loss_chunk_size: int,
    *,
    measure_wall: bool = False,
):
    optimizer.zero_grad(set_to_none=True)
    phase_wall_ms = {}
    step_start = time.perf_counter()

    with torch.autograd.profiler.record_function("train_step.total"):
        if measure_wall:
            maybe_synchronize(device)
            forward_start = time.perf_counter()
        with torch.autograd.profiler.record_function("train_step.forward"):
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=amp_dtype):
                _, loss = model(
                    input_ids,
                    labels=labels,
                    return_logits=False,
                    loss_chunk_size=loss_chunk_size,
                )
                loss = loss / grad_accum_steps
        if measure_wall:
            maybe_synchronize(device)
            phase_wall_ms["forward_wall_ms"] = (time.perf_counter() - forward_start) * 1000.0

        if measure_wall:
            backward_start = time.perf_counter()
        with torch.autograd.profiler.record_function("train_step.backward"):
            scaler.scale(loss).backward()
        if measure_wall:
            maybe_synchronize(device)
            phase_wall_ms["backward_wall_ms"] = (time.perf_counter() - backward_start) * 1000.0

        if measure_wall:
            optimizer_start = time.perf_counter()
        with torch.autograd.profiler.record_function("train_step.optimizer"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        if measure_wall:
            maybe_synchronize(device)
            phase_wall_ms["optimizer_wall_ms"] = (time.perf_counter() - optimizer_start) * 1000.0

    if measure_wall:
        phase_wall_ms["total_wall_ms"] = (time.perf_counter() - step_start) * 1000.0

    return loss.detach(), phase_wall_ms


def profile_impl(impl: str, args):
    device = args.device
    use_cuda = device.startswith("cuda")
    amp_dtype = get_amp_dtype(device)

    set_seed(args.seed)
    model = MOGTForCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        r=args.rank,
    ).to(device)
    model.train()
    model.mogt.gradient_checkpointing = args.gradient_checkpointing
    resolved_carry_scan = configure_scan_impl(model, impl, args)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    input_ids, labels = make_batch(args, device)

    # Warm up once so the profiled step is less polluted by one-time setup.
    warmup_loss, _ = run_train_step(
        model,
        optimizer,
        scaler,
        input_ids,
        labels,
        device,
        amp_dtype,
        args.grad_accum_steps,
        args.loss_chunk_size,
    )
    maybe_synchronize(device)

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loss, phase_wall_ms = run_train_step(
            model,
            optimizer,
            scaler,
            input_ids,
            labels,
            device,
            amp_dtype,
            args.grad_accum_steps,
            args.loss_chunk_size,
            measure_wall=True,
        )
        maybe_synchronize(device)

    peak_memory_mb = None
    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    sort_by = "self_cuda_time_total" if use_cuda else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=30)
    event_summary = collect_event_summary(prof, use_cuda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"train_step_{impl}.trace.json"
    table_path = output_dir / f"train_step_{impl}.table.txt"
    json_path = output_dir / f"train_step_{impl}.summary.json"
    prof.export_chrome_trace(str(trace_path))
    table_path.write_text(table)

    summary = {
        "impl": impl,
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "rank": args.rank,
            "grad_accum_steps": args.grad_accum_steps,
            "block_size": args.block_size,
            "block_c": args.block_c,
            "carry_scan": resolved_carry_scan,
            "connection_impl": args.connection_impl,
            "connection_damping": args.connection_damping,
            "loss_chunk_size": args.loss_chunk_size,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "device": args.device,
        },
        "warmup_loss": float(warmup_loss),
        "profiled_loss": float(loss),
        "peak_memory_mb": peak_memory_mb,
        "phase_wall_ms": phase_wall_ms,
        "events": event_summary,
        "artifacts": {
            "trace_json": str(trace_path),
            "table_txt": str(table_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))

    del model, optimizer, scaler, input_ids, labels
    if use_cuda:
        torch.cuda.empty_cache()

    return summary, table_path


def main():
    args = parse_args()
    results = []

    print("==================================================")
    print("📊 MOGT Single-Step Training Profiler")
    print("==================================================")
    for impl in args.impls:
        print(f"\n--- Profiling {impl} ---")
        summary, table_path = profile_impl(impl, args)
        results.append(summary)
        print(json.dumps(
            {
                "impl": summary["impl"],
                "profiled_loss": summary["profiled_loss"],
                "peak_memory_mb": summary["peak_memory_mb"],
                "phase_wall_ms": summary["phase_wall_ms"],
                "events": summary["events"],
                "table_txt": str(table_path),
            },
            indent=2,
        ))

    output_dir = Path(args.output_dir)
    compare_path = output_dir / "train_step_compare.json"
    compare_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ profiling summaries written to {compare_path}")


if __name__ == "__main__":
    main()
