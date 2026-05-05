import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize hybrid LM ratio sweep reports.")
    parser.add_argument("--glob", default="benchmark_runs/*hybrid_ratio_sweep*.json")
    parser.add_argument("--output-md", default="benchmark_runs/hybrid_lm_ratio_sweep_summary_20260504.md")
    return parser.parse_args()


def load_rows(pattern: str):
    rows = []
    for raw_path in sorted(glob.glob(pattern)):
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        standard = payload.get("standard_report")
        if not standard or standard.get("status") != "ok":
            continue
        config = standard.get("model_config", {})
        metrics = standard.get("metrics", {})
        env = standard.get("environment", {})
        training = standard.get("training", {})
        rows.append(
            {
                "run_name": standard.get("run_name"),
                "variant": standard.get("variant"),
                "seed": training.get("seed"),
                "steps": training.get("steps"),
                "context": standard.get("data", {}).get("train_context"),
                "fraction": config.get("mogt_layer_fraction"),
                "mogt_layers": config.get("mogt_layer_count"),
                "zero_init_attention_out": config.get("zero_init_attention_out"),
                "block_types": config.get("block_types"),
                "loss": metrics.get("loss"),
                "ppl": metrics.get("ppl"),
                "train_loss": metrics.get("train_loss_final"),
                "peak_memory_mb": env.get("peak_memory_mb"),
                "elapsed_seconds": env.get("elapsed_seconds"),
                "path": str(path),
            }
        )
    return rows


def fmt(value, precision=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def mean(values):
    return sum(values) / max(1, len(values))


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def family_name(run_name):
    if not run_name:
        return "-"
    return str(run_name).split("_frac", 1)[0]


def render(rows):
    lines = [
        "# Hybrid LM Ratio Sweep Summary",
        "",
        "Generated from `mogt-experiment-v1` language-modeling reports.",
        "These are pilot runs unless the command manifest says otherwise.",
        "",
        "## Aggregate",
        "",
        "| Family | Fraction | Zero-attn init | Seeds | Steps | Mean val loss | Std | Mean PPL | Mean elapsed s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in rows:
        key = (
            family_name(row["run_name"]),
            row["fraction"],
            bool(row["zero_init_attention_out"]),
            row["steps"],
        )
        grouped[key].append(row)

    for (family, fraction, zero_init, steps), group_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][3] or 0, item[0][2], item[0][1] or 0),
    ):
        losses = [row["loss"] for row in group_rows if row["loss"] is not None]
        ppls = [row["ppl"] for row in group_rows if row["ppl"] is not None]
        elapsed = [row["elapsed_seconds"] for row in group_rows if row["elapsed_seconds"] is not None]
        seeds = sorted({row["seed"] for row in group_rows if row["seed"] is not None})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{family}`",
                    fmt(fraction, precision=2),
                    "yes" if zero_init else "no",
                    ",".join(str(seed) for seed in seeds) or "-",
                    fmt(steps, precision=0),
                    fmt(mean(losses), precision=4) if losses else "-",
                    fmt(std(losses), precision=4) if losses else "-",
                    fmt(mean(ppls), precision=2) if ppls else "-",
                    fmt(mean(elapsed), precision=2) if elapsed else "-",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Fraction | MOGT layers | Zero-attn init | Seed | Steps | Val loss | PPL | Train loss | Peak MB | Elapsed s | Artifact |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["context"] or 0,
            item["steps"] or 0,
            item["seed"] or 0,
            bool(item["zero_init_attention_out"]),
            item["fraction"] or 0,
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["fraction"], precision=2),
                    fmt(row["mogt_layers"], precision=0),
                    "yes" if row["zero_init_attention_out"] else "no",
                    fmt(row["seed"], precision=0),
                    fmt(row["steps"], precision=0),
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


def main():
    args = parse_args()
    rows = load_rows(args.glob)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {output_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
