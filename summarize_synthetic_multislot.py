import glob
import json
import math
from collections import defaultdict


GROUPS = [
    (
        "Slot-addressed 2-slot dense",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot2_mogt_slotaddr_coupled_rank_value_forget_dense_"
                "ctx512_seed*_steps3000.json"
            ),
            "Transformer NoPE": (
                "benchmark_runs/synthetic_multislot2_transformer_nope_dense_"
                "ctx512_seed*_steps3000.json"
            ),
        },
    ),
    (
        "Slot-addressed 4-slot dense",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot4_mogt_slotaddr_coupled_rank_value_forget_dense_"
                "ctx512_seed*_steps3000.json"
            ),
            "Transformer NoPE": (
                "benchmark_runs/synthetic_multislot4_transformer_nope_dense_"
                "ctx512_seed*_steps3000.json"
            ),
        },
    ),
    (
        "Slot-addressed 2-slot final-only",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot2_mogt_slotaddr_coupled_rank_finalonly_"
                "ctx512_seed*_steps3000.json"
            ),
            "Transformer NoPE": (
                "benchmark_runs/synthetic_multislot2_transformer_nope_finalonly_"
                "ctx512_seed*_steps3000.json"
            ),
        },
    ),
    (
        "Slot-addressed 4-slot final-only slot curriculum",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot4_mogt_slotaddr_coupled_rank_slotcurr_finalonly_"
                "ctx512_seed*_steps3000.json"
            ),
            "Transformer NoPE": (
                "benchmark_runs/synthetic_multislot4_nope_slotcurr_finalonly_"
                "ctx512_seed*_steps3000.json"
            ),
        },
    ),
    (
        "Slot-addressed 6-slot final-only slot curriculum",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot6_mogt_slotaddr_coupled_rank_slotcurr_finalonly_"
                "ctx512_seed*_steps4000.json"
            ),
            "Transformer NoPE": (
                "benchmark_runs/synthetic_multislot6_nope_slotcurr_finalonly_"
                "ctx512_seed*_steps4000.json"
            ),
        },
    ),
    (
        "Slot-addressed 8-slot final-only slot curriculum",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot8_mogt_slotaddr_coupled_rank_slotcurr_finalonly_"
                "ctx512_seed*_steps5000.json"
            ),
            "Transformer NoPE": (
                "benchmark_runs/synthetic_multislot8_nope_slotcurr_finalonly_"
                "ctx512_seed*_steps5000.json"
            ),
            "HF Mamba": (
                "benchmark_runs/synthetic_multislot8_mamba_slotcurr_finalonly_"
                "ctx512_seed*_steps5000.json"
            ),
            "HF Mamba d192": (
                "benchmark_runs/synthetic_multislot8_mamba_d192_slotcurr_finalonly_"
                "ctx512_seed*_steps5000.json"
            ),
        },
    ),
    (
        "Slot-addressed 6-slot direct final-only seed42 ablation",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot6_mogt_slotaddr_coupled_rank_direct_finalonly_"
                "ctx512_seed42_steps4000.json"
            ),
        },
    ),
    (
        "Slot-addressed 8-slot direct final-only seed42 ablation",
        {
            "Slot-addressed MOGT": (
                "benchmark_runs/"
                "synthetic_multislot8_mogt_slotaddr_coupled_rank_direct_finalonly_"
                "ctx512_seed42_steps5000.json"
            ),
        },
    ),
    (
        "Scratch GRU 6-slot final-only slot curriculum early probe",
        {
            "GRU": (
                "benchmark_runs/"
                "synthetic_multislot6_gru_slotcurr2000_finalonly_"
                "ctx512_seed42_steps500.json"
            ),
        },
    ),
]


def mean(values):
    return sum(values) / max(1, len(values))


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def load_reports(pattern):
    reports = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        report["_path"] = path
        reports.append(report)
    return reports


def summarize_pattern(pattern):
    by_context = defaultdict(list)
    paths = []
    for report in load_reports(pattern):
        paths.append(report["_path"])
        for result in report["eval_results"]:
            by_context[result["context_length"]].append(result["accuracy"])
    return paths, {
        context: {
            "mean": mean(values),
            "std": std(values),
            "n": len(values),
        }
        for context, values in sorted(by_context.items())
    }


def format_percent(value):
    return f"{100.0 * value:.2f}%"


def format_cell(stats):
    if stats["n"] == 0:
        return "missing"
    return f"{format_percent(stats['mean'])} +/- {format_percent(stats['std'])}"


def print_group(title, specs):
    print(f"\n## {title}\n")
    contexts = set()
    rows = []
    for label, pattern in specs.items():
        paths, summary = summarize_pattern(pattern)
        contexts.update(summary)
        rows.append((label, paths, summary))
    contexts = sorted(contexts)
    print("| Model | Seeds | " + " | ".join(str(context) for context in contexts) + " |")
    print("|---|---:|" + "|".join("---:" for _ in contexts) + "|")
    for label, paths, summary in rows:
        cells = [format_cell(summary.get(context, {"n": 0})) for context in contexts]
        print(f"| {label} | {len(paths)} | " + " | ".join(cells) + " |")
    for label, paths, _summary in rows:
        print(f"- {label}: " + ", ".join(f"`{path}`" for path in paths))


def main():
    for title, specs in GROUPS:
        print_group(title, specs)


if __name__ == "__main__":
    main()
