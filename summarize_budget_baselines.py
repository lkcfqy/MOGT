import argparse
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MOGT and budget-matched scratch baselines.")
    parser.add_argument(
        "--reports",
        nargs="*",
        default=None,
        help="Budget baseline JSON reports. Defaults to benchmark_runs/*scratch_budget*.json.",
    )
    parser.add_argument(
        "--mogt-eval",
        default="benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json",
    )
    parser.add_argument(
        "--mogt-run",
        default="benchmark_runs/baseline_v1_cayley_ctx32768_multiseed_20260428.json",
    )
    parser.add_argument("--mogt-context", type=int, default=32768)
    parser.add_argument("--output-md", default="benchmark_runs/budget_matched_baseline_summary.md")
    parser.add_argument("--output-json", default="benchmark_runs/budget_matched_baseline_summary.json")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--include-probe", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_float(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_ppl(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def fmt_params(value):
    if value is None:
        return "-"
    return f"{float(value) / 1e6:.2f}M"


def fmt_mb(value):
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def fmt_elapsed(value):
    if value is None:
        return "-"
    return f"{float(value):.1f}s"


def mogt_record(eval_path, run_path, context):
    eval_data = load_json(eval_path)
    run_data = load_json(run_path) if Path(run_path).exists() else {}
    context_key = str(context)
    if context_key not in eval_data.get("aggregate_by_context", {}):
        return None

    aggregate = eval_data["aggregate_by_context"][context_key]
    run_config = run_data.get("config", {})
    eval_config = eval_data.get("config", {})
    return {
        "model": "MOGT baseline_v1 Cayley",
        "run_name": "baseline_v1_cayley",
        "status": "ok",
        "context_length": context,
        "parameter_count": None,
        "config": (
            f"{eval_config.get('num_layers', '-') }L d{eval_config.get('d_model', '-')} "
            f"r{eval_config.get('rank', '-')}, {eval_config.get('scan_impl', '-')}, "
            f"{eval_config.get('connection_impl', '-')}"
        ),
        "seeds": aggregate.get("num_checkpoints"),
        "steps": run_config.get("max_global_steps", 200),
        "validation_batches": "/".join(str(v) for v in aggregate.get("num_batches_per_checkpoint", [])),
        "best_loss_mean": aggregate.get("loss_mean"),
        "best_loss_std": aggregate.get("loss_sample_std"),
        "best_ppl_mean": aggregate.get("ppl_mean"),
        "best_ppl_std": aggregate.get("ppl_sample_std"),
        "peak_memory_mb": None,
        "elapsed_s": None,
        "notes": "Checkpoint-only eval of three 200-step seeds.",
        "source": eval_path,
    }


def aggregate_mamba_record(path, data):
    protocol = data.get("protocol", {})
    aggregates = data.get("aggregates", {})
    per_seed = data.get("per_seed", [])
    return {
        "model": data.get("model", "Scratch Mamba SSM"),
        "run_name": Path(path).stem,
        "status": "ok",
        "context_length": protocol.get("context_length"),
        "parameter_count": protocol.get("parameter_count"),
        "config": f"{protocol.get('num_layers', '-') }L d{protocol.get('d_model', '-')}",
        "seeds": len(per_seed) or None,
        "steps": protocol.get("max_steps"),
        "validation_batches": "/".join(str(seed.get("best_val_batches", "-")) for seed in per_seed),
        "best_loss_mean": aggregates.get("best_val_loss", {}).get("mean"),
        "best_loss_std": aggregates.get("best_val_loss", {}).get("std"),
        "best_ppl_mean": aggregates.get("best_val_ppl", {}).get("mean"),
        "best_ppl_std": aggregates.get("best_val_ppl", {}).get("std"),
        "peak_memory_mb": None,
        "elapsed_s": None,
        "notes": "Aggregate scratch baseline report.",
        "source": path,
    }


def single_budget_record(path, data):
    config = data.get("config", {})
    result = data.get("result", {})
    best_val = result.get("best_val") or {}
    status = result.get("status", "ok")

    if config.get("model_type") == "scratch_transformer":
        model = "Scratch Transformer"
        model_config = f"{config.get('num_layers', '-') }L d{config.get('d_model', '-')} h{config.get('num_heads', '-')}"
    elif config.get("model_type") == "mamba_ssm":
        model = "Scratch Mamba SSM"
        model_config = f"{config.get('num_layers', '-') }L d{config.get('d_model', '-')}"
    else:
        model = config.get("model_type", "Scratch baseline")
        model_config = f"{config.get('num_layers', '-') }L d{config.get('d_model', '-')}"

    return {
        "model": model,
        "run_name": config.get("run_name", Path(path).stem),
        "status": status,
        "context_length": config.get("context_length"),
        "parameter_count": config.get("parameter_count"),
        "config": model_config,
        "seeds": 1,
        "steps": config.get("max_steps"),
        "validation_batches": best_val.get("num_batches"),
        "best_loss_mean": best_val.get("loss"),
        "best_loss_std": None,
        "best_ppl_mean": best_val.get("ppl"),
        "best_ppl_std": None,
        "peak_memory_mb": result.get("peak_memory_mb"),
        "elapsed_s": result.get("elapsed_s"),
        "notes": result.get("failed_phase", "") if status != "ok" else "Single-seed scratch report.",
        "source": path,
    }


def collect_budget_records(paths):
    records = []
    seen = set()
    for path in paths:
        path = str(path)
        if path in seen or not Path(path).exists():
            continue
        seen.add(path)
        data = load_json(path)
        if "aggregates" in data and "per_seed" in data:
            records.append(aggregate_mamba_record(path, data))
        elif "config" in data and "result" in data:
            records.append(single_budget_record(path, data))
    return records


def default_report_paths(include_smoke: bool, include_probe: bool):
    paths = sorted(
        set(
            glob.glob("benchmark_runs/*scratch_budget*.json")
            + glob.glob("benchmark_runs/*transformer_scratch*.json")
        )
    )
    if not include_smoke:
        paths = [path for path in paths if "smoke" not in Path(path).stem]
    if not include_probe:
        paths = [path for path in paths if "probe" not in Path(path).stem]

    # Prefer the curated multiseed Mamba aggregate over the single-seed source reports.
    has_mamba_multiseed = any("mamba_scratch_budget_v1_ctx32768_multiseed" in Path(path).stem for path in paths)
    if has_mamba_multiseed:
        paths = [
            path
            for path in paths
            if "mamba_scratch_budget_v1_ctx32768_seed" not in Path(path).stem
        ]
    return paths


def render_markdown(records):
    lines = [
        "# Budget-Matched Baseline Summary",
        "",
        "| Model | Context | Params | Config | Seeds | Steps | Status | Val batches | Best loss | Best PPL | Peak MB | Elapsed | Notes |",
        "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("model", "-")),
                    str(record.get("context_length", "-")),
                    fmt_params(record.get("parameter_count")),
                    str(record.get("config", "-")),
                    str(record.get("seeds", "-")),
                    str(record.get("steps", "-")),
                    str(record.get("status", "-")),
                    str(record.get("validation_batches", "-")),
                    fmt_float(record.get("best_loss_mean")),
                    fmt_ppl(record.get("best_ppl_mean")),
                    fmt_mb(record.get("peak_memory_mb")),
                    fmt_elapsed(record.get("elapsed_s")),
                    str(record.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- MOGT values come from checkpoint-only evaluation of the three baseline_v1 seeds.",
            "- Scratch baselines use the repo GPT-2 token stream and WikiText-103 data protocol.",
            "- OOM rows should be kept as long-context systems evidence rather than omitted.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.reports is None:
        reports = default_report_paths(args.include_smoke, args.include_probe)
    else:
        reports = args.reports

    records = []
    mogt = mogt_record(args.mogt_eval, args.mogt_run, args.mogt_context)
    if mogt is not None:
        records.append(mogt)
    records.extend(collect_budget_records(reports))

    records.sort(key=lambda item: (item.get("context_length") or 0, item.get("model") or "", item.get("run_name") or ""))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(records), encoding="utf-8")

    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
