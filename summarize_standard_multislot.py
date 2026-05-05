import glob
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def load_reports() -> list[dict[str, Any]]:
    reports = []
    for raw_path in sorted(glob.glob("benchmark_runs/*.json")):
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        standard = payload.get("standard_report")
        if standard is None or standard.get("task") != "tracked_multislot_last_value":
            continue
        reports.append({"path": str(path), "payload": payload, "standard": standard})
    return reports


def group_key(report: dict[str, Any]) -> tuple[Any, ...]:
    standard = report["standard"]
    data = standard.get("data", {})
    training = standard.get("training", {})
    return (
        standard.get("model"),
        standard.get("variant"),
        training.get("steps"),
        data.get("train_context"),
        data.get("num_slots"),
        data.get("min_train_slots"),
        training.get("slot_curriculum_steps"),
        training.get("dense_loss"),
    )


def summarize(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[group_key(report)].append(report)

    rows = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        model, variant, steps, train_context, num_slots, min_train_slots, slot_curriculum_steps, dense_loss = key
        by_context: dict[str, list[float]] = defaultdict(list)
        for item in items:
            metrics = item["standard"].get("metrics", {})
            for context, accuracy in metrics.get("accuracy_by_context", {}).items():
                by_context[str(context)].append(float(accuracy))
        rows.append(
            {
                "model": model,
                "variant": variant,
                "steps": steps,
                "train_context": train_context,
                "num_slots": num_slots,
                "min_train_slots": min_train_slots,
                "slot_curriculum_steps": slot_curriculum_steps,
                "dense_loss": dense_loss,
                "seeds": sorted(
                    item["standard"].get("training", {}).get("seed") for item in items
                ),
                "artifacts": [item["path"] for item in items],
                "accuracy_by_context": {
                    context: {
                        "mean": mean(values),
                        "std": std(values),
                        "n": len(values),
                    }
                    for context, values in sorted(by_context.items(), key=lambda pair: int(pair[0]))
                },
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_json = Path("benchmark_runs/synthetic_multislot_standard_summary_20260504.json")
    output_md = Path("benchmark_runs/synthetic_multislot_standard_summary_20260504.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "groups": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Synthetic Multi-Slot Standard Summary",
        "",
        "Date: 2026-05-04",
        "",
        "This summary is generated only from benchmark artifacts that contain the",
        "`mogt-experiment-v1` standard report schema.",
        "",
    ]
    for row in rows:
        contexts = list(row["accuracy_by_context"].keys())
        lines.extend(
            [
                f"## {row['variant']}",
                "",
                f"Model: `{row['model']}`. Seeds: {', '.join(str(seed) for seed in row['seeds'])}. "
                f"Steps: {row['steps']}. Train context: {row['train_context']}. "
                f"Slots: {row['num_slots']}. Dense loss: {row['dense_loss']}.",
                "",
                "| Context | Accuracy | Seeds |",
                "|---:|---:|---:|",
            ]
        )
        for context in contexts:
            stats = row["accuracy_by_context"][context]
            lines.append(
                f"| {context} | {pct(stats['mean'])} +/- {pct(stats['std'])} | {stats['n']} |"
            )
        lines.extend(["", "Artifacts:"])
        lines.extend(f"- `{path}`" for path in row["artifacts"])
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    return output_json, output_md


def main() -> None:
    rows = summarize(load_reports())
    output_json, output_md = write_outputs(rows)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
