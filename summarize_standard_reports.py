import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment_report import validate_standard_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an index of benchmark JSON files that contain standard_report."
    )
    parser.add_argument(
        "--reports",
        nargs="*",
        default=None,
        help="Reports to scan. Defaults to benchmark_runs/*.json.",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_runs/standard_report_index.json",
    )
    parser.add_argument(
        "--output-md",
        default="benchmark_runs/standard_report_index.md",
    )
    return parser.parse_args()


def load_standard_reports(paths):
    rows = []
    skipped = 0
    invalid = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append({"path": str(path), "errors": [f"could not parse JSON: {exc}"]})
            continue
        standard_report = report.get("standard_report")
        if standard_report is None:
            skipped += 1
            continue
        errors = validate_standard_report(standard_report)
        if errors:
            invalid.append({"path": str(path), "errors": errors})
            continue
        rows.append(
            {
                "path": str(path),
                "run_name": standard_report.get("run_name"),
                "task": standard_report.get("task"),
                "model": standard_report.get("model"),
                "variant": standard_report.get("variant"),
                "status": standard_report.get("status"),
                "seed": standard_report.get("training", {}).get("seed"),
                "steps": standard_report.get("training", {}).get("steps"),
                "train_context": standard_report.get("data", {}).get("train_context"),
                "eval_contexts": standard_report.get("data", {}).get("eval_contexts"),
                "num_params": standard_report.get("model_config", {}).get("num_params"),
                "peak_memory_mb": standard_report.get("environment", {}).get("peak_memory_mb"),
                "elapsed_seconds": standard_report.get("environment", {}).get("elapsed_seconds"),
            }
        )
    return rows, skipped, invalid


def build_summary(rows, skipped, invalid):
    by_task = Counter(row["task"] for row in rows)
    by_status = Counter(row["status"] for row in rows)
    by_task_model = defaultdict(Counter)
    for row in rows:
        by_task_model[row["task"]][row["model"]] += 1
    return {
        "num_standard_reports": len(rows),
        "num_skipped_legacy_reports": skipped,
        "num_invalid_reports": len(invalid),
        "by_task": dict(sorted(by_task.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_task_model": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(by_task_model.items())
        },
    }


def render_markdown(summary, rows, invalid):
    lines = [
        "# Standard Report Index",
        "",
        f"Standard reports: {summary['num_standard_reports']}",
        f"Skipped legacy reports: {summary['num_skipped_legacy_reports']}",
        f"Invalid reports: {summary['num_invalid_reports']}",
        "",
        "## By Task",
        "",
        "| Task | Count |",
        "|---|---:|",
    ]
    for task, count in summary["by_task"].items():
        lines.append(f"| {task} | {count} |")

    lines.extend(
        [
            "",
            "## Reports",
            "",
            "| Task | Model | Variant | Status | Seed | Steps | Train ctx | Eval ctx | Artifact |",
            "|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["task"] or "", item["model"] or "", str(item["variant"] or ""), str(item["seed"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task") or "-"),
                    str(row.get("model") or "-"),
                    str(row.get("variant") or "-"),
                    str(row.get("status") or "-"),
                    str(row.get("seed") if row.get("seed") is not None else "-"),
                    str(row.get("steps") if row.get("steps") is not None else "-"),
                    str(row.get("train_context") if row.get("train_context") is not None else "-"),
                    ", ".join(str(value) for value in (row.get("eval_contexts") or [])) or "-",
                    f"`{row['path']}`",
                ]
            )
            + " |"
        )

    if invalid:
        lines.extend(["", "## Invalid Reports", ""])
        for item in invalid:
            lines.append(f"- `{item['path']}`: {'; '.join(item['errors'])}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    paths = args.reports or sorted(glob.glob("benchmark_runs/*.json"))
    rows, skipped, invalid = load_standard_reports(paths)
    summary = build_summary(rows, skipped, invalid)
    payload = {
        "summary": summary,
        "records": rows,
        "invalid": invalid,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary, rows, invalid), encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
