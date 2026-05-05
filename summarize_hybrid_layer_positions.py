import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize single-MOGT-layer position ablations for the hybrid LM."
    )
    parser.add_argument("--glob", default="benchmark_runs/*.json")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--mogt-lr-mult",
        type=float,
        default=None,
        help="Filter by MOGT block learning-rate multiplier; default keeps baseline multiplier 1.0.",
    )
    parser.add_argument("--mogt-residual-scale", type=float, default=None)
    parser.add_argument("--mogt-ffn-residual-scale", type=float, default=None)
    parser.add_argument(
        "--mogt-residual-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter by learned MOGT residual gate usage; default excludes gated runs.",
    )
    parser.add_argument(
        "--include-all-mogt-scales",
        action="store_true",
        help="Do not apply the default residual-scale=1.0 filters.",
    )
    parser.add_argument("--require-zero-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-md",
        default="benchmark_runs/hybrid_layer_position_summary_20260504.md",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def infer_position(block_types: list[str]) -> str | None:
    mogt_indices = [index for index, block_type in enumerate(block_types) if block_type == "mogt"]
    if not mogt_indices:
        return "attention-only"
    if len(mogt_indices) == 1:
        return f"layer-{mogt_indices[0]}"
    return "layers-" + "-".join(str(index) for index in mogt_indices)


def scale_matches(value: Any, target: float) -> bool:
    if value is None:
        value = 1.0
    return abs(float(value) - target) <= 1e-12


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for raw_path in sorted(glob.glob(args.glob)):
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        standard = payload.get("standard_report")
        if not standard or standard.get("status") != "ok":
            continue
        if standard.get("task") != "language_modeling":
            continue
        data = standard.get("data", {})
        training = standard.get("training", {})
        config = standard.get("model_config", {})
        if data.get("train_context") != args.context_length:
            continue
        if training.get("steps") != args.steps:
            continue
        if config.get("d_model") != args.d_model or config.get("num_layers") != args.num_layers:
            continue
        lr = training.get("lr")
        if args.lr is not None and (lr is None or abs(float(lr) - args.lr) > 1e-12):
            continue
        block_types = config.get("block_types") or []
        position = infer_position(block_types)
        if position is None:
            continue
        has_mogt = position != "attention-only"
        target_mogt_lr_mult = 1.0 if args.mogt_lr_mult is None else args.mogt_lr_mult
        if has_mogt and abs(float(config.get("mogt_lr_mult", 1.0)) - target_mogt_lr_mult) > 1e-12:
            continue
        if has_mogt and not args.include_all_mogt_scales:
            target_scale = args.mogt_residual_scale
            if target_scale is None:
                target_scale = 1.0
            if not scale_matches(config.get("mogt_residual_scale"), target_scale):
                continue

            target_ffn_scale = args.mogt_ffn_residual_scale
            if target_ffn_scale is None:
                target_ffn_scale = 1.0
            if not scale_matches(config.get("mogt_ffn_residual_scale"), target_ffn_scale):
                continue

            residual_schedule = config.get("mogt_residual_scale_schedule", "constant")
            if residual_schedule != "constant":
                continue
        gate_enabled = bool(config.get("mogt_residual_gate", False))
        if args.mogt_residual_gate is None:
            if gate_enabled:
                continue
        elif gate_enabled is not bool(args.mogt_residual_gate):
            continue
        if args.require_zero_init and not config.get("zero_init_attention_out"):
            continue
        metrics = standard.get("metrics", {})
        environment = standard.get("environment", {})
        rows.append(
            {
                "position": position,
                "seed": training.get("seed"),
                "loss": metrics.get("loss"),
                "ppl": metrics.get("ppl"),
                "train_loss": metrics.get("train_loss_final"),
                "elapsed_seconds": environment.get("elapsed_seconds"),
                "peak_memory_mb": environment.get("peak_memory_mb"),
                "lr": lr,
                "mogt_residual_scale": config.get("mogt_residual_scale"),
                "mogt_ffn_residual_scale": config.get("mogt_ffn_residual_scale"),
                "mogt_residual_gate": config.get("mogt_residual_gate"),
                "mogt_residual_gate_init": config.get("mogt_residual_gate_init"),
                "run_name": standard.get("run_name"),
                "path": str(path),
            }
        )
    return rows


def sort_position(position: str) -> tuple[int, int]:
    if position == "attention-only":
        return (0, -1)
    if position.startswith("layer-"):
        return (1, int(position.rsplit("-", 1)[1]))
    if position.startswith("layers-"):
        first_layer = int(position.split("-", 2)[1])
        return (2, first_layer)
    return (3, 0)


def fmt(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def render(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["position"]].append(row)

    baseline_losses = [
        float(row["loss"])
        for row in grouped.get("attention-only", [])
        if row.get("loss") is not None
    ]
    baseline_loss = mean(baseline_losses) if baseline_losses else None

    lines = [
        "# Hybrid LM Single-Layer Position Summary",
        "",
        f"Context {args.context_length}, `d_model={args.d_model}`, "
        f"{args.num_layers} layers, {args.steps} optimizer steps,",
        "WikiText-103/GPT-2 token stream.",
        f"Learning-rate filter: `{args.lr}`." if args.lr is not None else "Learning-rate filter: none.",
        (
            "MOGT residual-scale filter: all."
            if args.include_all_mogt_scales
            else (
                f"MOGT residual-scale filter: `{args.mogt_residual_scale}`."
                if args.mogt_residual_scale is not None
                else "MOGT residual-scale filter: default `1.0`."
            )
        ),
        (
            "MOGT FFN residual-scale filter: all."
            if args.include_all_mogt_scales
            else (
                f"MOGT FFN residual-scale filter: `{args.mogt_ffn_residual_scale}`."
                if args.mogt_ffn_residual_scale is not None
                else "MOGT FFN residual-scale filter: default `1.0`."
            )
        ),
        (
            f"MOGT residual-gate filter: `{args.mogt_residual_gate}`."
            if args.mogt_residual_gate is not None
            else "MOGT residual-gate filter: default `False`."
        ),
        (
            f"MOGT LR multiplier filter: `{args.mogt_lr_mult}`."
            if args.mogt_lr_mult is not None
            else "MOGT LR multiplier filter: default `1.0`."
        ),
        (
            "All rows use `--zero-init-attention-out`."
            if args.require_zero_init
            else "Rows are not filtered by `--zero-init-attention-out`."
        ),
        "",
        "## Aggregate",
        "",
        "| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for position in sorted(grouped, key=sort_position):
        group_rows = grouped[position]
        losses = [float(row["loss"]) for row in group_rows if row.get("loss") is not None]
        ppls = [float(row["ppl"]) for row in group_rows if row.get("ppl") is not None]
        elapsed = [
            float(row["elapsed_seconds"])
            for row in group_rows
            if row.get("elapsed_seconds") is not None
        ]
        seeds = sorted({int(row["seed"]) for row in group_rows if row.get("seed") is not None})
        delta = "-"
        if baseline_loss is not None and losses:
            delta = f"{mean(losses) - baseline_loss:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    position,
                    ",".join(str(seed) for seed in seeds),
                    fmt(mean(losses), precision=4) if losses else "-",
                    fmt(std(losses), precision=4) if losses else "-",
                    delta,
                    fmt(mean(ppls), precision=2) if ppls else "-",
                    fmt(mean(elapsed), precision=2) if elapsed else "-",
                ]
            )
            + " |"
        )

    layer_positions = sorted(
        [
            position
            for position in grouped
            if position.startswith("layer-") or position.startswith("layers-")
        ],
        key=sort_position,
    )
    lines.append("")
    if len(layer_positions) < args.num_layers:
        present_positions = sorted(grouped, key=sort_position)
        lines.append(
            "This filtered summary contains "
            + ", ".join(present_positions)
            + "; absent layers were not part of this run set."
        )
    else:
        lines.append(
            "Layer 0 currently has only one seed in the 200-step table; layer 1 is "
            "the original 25% ratio run, and layers 2/3 are explicit-index follow-ups."
        )
    lines.extend(
        [
            "Treat the layer-order trend as a scaling target, not a final LM claim.",
            "",
            "## Runs",
            "",
            "| Position | Seed | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (sort_position(item["position"]), item["seed"] or 0)):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["position"],
                    fmt(row["seed"], precision=0),
                    fmt(row["loss"], precision=4),
                    fmt(row["ppl"], precision=2),
                    fmt(row["train_loss"], precision=4),
                    fmt(row["peak_memory_mb"], precision=1),
                    fmt(row["elapsed_seconds"], precision=2),
                    f"`{row['path']}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render(rows, args), encoding="utf-8")
    print(f"Wrote {output_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
