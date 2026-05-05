import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small MOGT/Transformer layer-ratio LM sweep."
    )
    parser.add_argument("--run-prefix", default="hybrid_ratio_sweep")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-max-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_runs")
    parser.add_argument("--checkpoint-root", default="baseline_checkpoints")
    parser.add_argument("--scan-impl", default="triton_hybrid")
    parser.add_argument("--connection-impl", choices=["matrix_exp", "cayley", "identity"], default="cayley")
    parser.add_argument("--connection-damping", type=float, default=0.999)
    parser.add_argument("--zero-init-attention-out", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fraction_tag(fraction: float) -> str:
    return f"{fraction:g}".replace(".", "p")


def build_command(args, *, fraction: float, seed: int) -> list[str]:
    tag = fraction_tag(fraction)
    run_name = (
        f"{args.run_prefix}_frac{tag}_ctx{args.context_length}_"
        f"d{args.d_model}_l{args.num_layers}_seed{seed}_steps{args.max_steps}"
    )
    output_path = Path(args.output_dir) / f"{run_name}.json"
    checkpoint_dir = Path(args.checkpoint_root) / run_name
    cmd = [
        sys.executable,
        "train_budget_hybrid.py",
        "--run-name",
        run_name,
        "--context-length",
        str(args.context_length),
        "--d-model",
        str(args.d_model),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--rank",
        str(args.rank),
        "--seed",
        str(seed),
        "--hybrid-pattern",
        "ratio_even",
        "--mogt-layer-fraction",
        str(fraction),
        "--scan-impl",
        args.scan_impl,
        "--connection-impl",
        args.connection_impl,
        "--connection-damping",
        str(args.connection_damping),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--max-steps",
        str(args.max_steps),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-max-batches",
        str(args.eval_max_batches),
        "--num-workers",
        str(args.num_workers),
        "--latest-checkpoint-interval",
        "0",
        "--report-output",
        str(output_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if not args.save_checkpoints:
        cmd.extend(["--no-save-best", "--no-save-last"])
    if args.zero_init_attention_out:
        cmd.append("--zero-init-attention-out")
    return cmd


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for fraction in args.fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    for seed in args.seeds:
        for fraction in args.fractions:
            cmd = build_command(args, fraction=fraction, seed=seed)
            print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
