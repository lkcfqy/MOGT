import argparse
import json
import math
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch

from dataset import get_dataloaders
from model_mogt import MOGTForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved MOGT checkpoints on WikiText-103 validation.")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Checkpoint files or directories containing mogt_best.pt.")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scan-impl", type=str, default="triton_hybrid")
    parser.add_argument("--scan-block-size", type=int, default=256)
    parser.add_argument("--scan-block-c", type=int, default=32)
    parser.add_argument("--block-carry-scan", type=str, default="auto", choices=["auto", "sequential", "doubling"])
    parser.add_argument("--connection-impl", type=str, default="cayley")
    parser.add_argument("--connection-damping", type=float, default=0.999)
    parser.add_argument("--loss-chunk-size", type=int, default=0)
    parser.add_argument("--output", type=str, default="benchmark_runs/checkpoint_eval.json")
    return parser.parse_args()


def resolve_checkpoint_paths(inputs):
    paths = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            path = path / "mogt_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        paths.append(path)
    return paths


def resolve_block_carry_scan(scan_impl: str, context_length: int, requested: str) -> str:
    if requested != "auto":
        return requested
    if scan_impl == "triton_hybrid" and context_length >= 32768:
        return "doubling"
    return "sequential"


def configure_model(model, args, context_length: int):
    carry_scan = resolve_block_carry_scan(args.scan_impl, context_length, args.block_carry_scan)
    model.mogt.gradient_checkpointing = False
    for block in model.mogt.blocks:
        block.scan_impl = args.scan_impl
        block.scan_block_size = args.scan_block_size
        block.scan_block_c = args.scan_block_c
        block.block_carry_scan = carry_scan
        block.connection_impl = args.connection_impl
        block.connection_damping = args.connection_damping
    return carry_scan


def load_model(checkpoint, vocab_size: int, args, device):
    model = MOGTForCausalLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        r=args.rank,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, val_dl, device, *, max_batches: int, loss_chunk_size: int):
    total_loss = 0.0
    total_batches = 0
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_dl):
            if batch_idx >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)
            amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
            with amp_ctx:
                _, loss = model(x, labels=y, return_logits=False, loss_chunk_size=loss_chunk_size)
            total_loss += float(loss.item())
            total_batches += 1

    if total_batches == 0:
        return None

    avg_loss = total_loss / total_batches
    return {
        "loss": avg_loss,
        "ppl": math.exp(avg_loss),
        "num_batches": total_batches,
    }


def build_aggregate(results):
    by_context = {}
    for item in results:
        metrics = item.get("metrics")
        if metrics is None:
            continue
        context_length = str(item["context_length"])
        by_context.setdefault(context_length, []).append(metrics)

    aggregate = {}
    for context_length, metrics_list in by_context.items():
        losses = [item["loss"] for item in metrics_list]
        ppls = [item["ppl"] for item in metrics_list]
        batches = [item["num_batches"] for item in metrics_list]
        aggregate[context_length] = {
            "num_checkpoints": len(metrics_list),
            "num_batches_per_checkpoint": batches,
            "loss_mean": statistics.mean(losses),
            "loss_sample_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "loss_min": min(losses),
            "loss_max": max(losses),
            "ppl_mean": statistics.mean(ppls),
            "ppl_sample_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "ppl_min": min(ppls),
            "ppl_max": max(ppls),
        }
    return aggregate


def main():
    args = parse_args()
    checkpoint_paths = resolve_checkpoint_paths(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    report = {
        "config": {
            "context_lengths": args.context_lengths,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_batches": args.max_batches,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "rank": args.rank,
            "scan_impl": args.scan_impl,
            "connection_impl": args.connection_impl,
            "connection_damping": args.connection_damping,
            "device": str(device),
        },
        "results": [],
    }

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = None

        try:
            for context_length in args.context_lengths:
                _, val_dl, vocab_size = get_dataloaders(
                    context_length=context_length,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                if model is None:
                    model = load_model(checkpoint, vocab_size, args, device)

                carry_scan = configure_model(model, args, context_length)
                loss_chunk_size = args.loss_chunk_size or (256 if context_length >= 32768 else 4096)
                metrics = evaluate_model(
                    model,
                    val_dl,
                    device,
                    max_batches=args.max_batches,
                    loss_chunk_size=loss_chunk_size,
                )

                report["results"].append({
                    "checkpoint": str(checkpoint_path),
                    "context_length": context_length,
                    "loss_chunk_size": loss_chunk_size,
                    "carry_scan": carry_scan,
                    "metrics": metrics,
                    "checkpoint_meta": {
                        "epoch": checkpoint.get("epoch"),
                        "step": checkpoint.get("step"),
                        "global_accum_step": checkpoint.get("global_accum_step"),
                        "best_val_loss": checkpoint.get("best_val_loss"),
                        "best_val_ppl": checkpoint.get("best_val_ppl"),
                    },
                })
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_path = Path(args.output)
    report["aggregate_by_context"] = build_aggregate(report["results"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to {output_path}")


if __name__ == "__main__":
    main()
