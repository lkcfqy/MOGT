import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone


def load_reports():
    reports = []
    for path in glob.glob("benchmark_runs/synthetic_last_value_*.json"):
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        if "config" not in report:
            continue
        report["_path"] = path
        reports.append(report)
    return reports


def mean(values):
    return sum(values) / max(1, len(values))


def std(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def model_label(report):
    config = report["config"]
    if report["model_type"] == "transformer":
        rope_theta = float(config.get("rope_theta", 10000.0))
        if rope_theta <= 0:
            return "Transformer NoPE"
        if rope_theta != 10000.0:
            return f"Transformer RoPE theta={rope_theta:g}"
        return "Transformer RoPE"
    if config.get("couple_forget_to_value_gate"):
        return "Coupled MOGT"
    if config.get("transport_gate"):
        return "Gated MOGT"
    if config.get("value_gate"):
        return "Value-gated MOGT"
    return "MOGT"


def is_main_run(report):
    config = report["config"]
    eval_contexts = config.get("eval_contexts")
    return (
        report.get("task") == "last_value_tracking"
        and config.get("steps") == 2000
        and config.get("train_context") == 128
        and (config.get("eval_batch_size", 0) or 0) == 0
        and eval_contexts == [128, 256, 512, 1024]
        and config.get("num_values") == 16
        and config.get("max_updates") == 4
        and model_label(report) in {"Gated MOGT", "Transformer RoPE"}
        and config.get("seed") in {7, 42, 123}
    )


def is_nope_run(report):
    config = report["config"]
    return (
        report.get("task") == "last_value_tracking"
        and config.get("steps") == 2000
        and config.get("train_context") == 128
        and config.get("eval_batch_size") == 4
        and config.get("num_values") == 16
        and config.get("max_updates") == 4
        and model_label(report) == "Transformer NoPE"
        and config.get("seed") in {7, 42, 123}
    )


def is_matched_nope_run(report):
    config = report["config"]
    return (
        report.get("task") == "last_value_tracking"
        and config.get("steps") == 2000
        and config.get("train_context") == 128
        and (config.get("eval_batch_size", 0) or 0) == 0
        and config.get("eval_batches") == 32
        and config.get("eval_contexts") == [128, 256, 512, 1024]
        and config.get("num_values") == 16
        and config.get("max_updates") == 4
        and model_label(report) == "Transformer NoPE"
        and config.get("seed") in {7, 42, 123}
    )


def summarize_main(reports):
    grouped = defaultdict(list)
    for report in reports:
        if not is_main_run(report):
            continue
        label = model_label(report)
        for result in report["eval_results"]:
            grouped[(label, result["context_length"])].append(result["accuracy"])
    rows = []
    for (label, context_length), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append(
            {
                "model": label,
                "context_length": context_length,
                "mean_accuracy": mean(values),
                "std_accuracy": std(values),
                "seeds": len(values),
            }
        )
    return rows


def summarize_individual_1024(reports):
    rows = []
    for report in reports:
        if not is_main_run(report):
            continue
        context_1024 = next(
            result for result in report["eval_results"] if result["context_length"] == 1024
        )
        rows.append(
            {
                "model": model_label(report),
                "seed": report["config"]["seed"],
                "accuracy_1024": context_1024["accuracy"],
                "loss_1024": context_1024["loss"],
                "path": report["_path"],
            }
        )
    return sorted(rows, key=lambda row: (row["model"], row["seed"]))


def summarize_nope(reports):
    grouped = defaultdict(list)
    for report in reports:
        if not is_nope_run(report):
            continue
        for result in report["eval_results"]:
            grouped[result["context_length"]].append(result["accuracy"])
    rows = []
    for context_length, values in sorted(grouped.items()):
        rows.append(
            {
                "context_length": context_length,
                "mean_accuracy": mean(values),
                "std_accuracy": std(values),
                "seeds": len(values),
            }
        )
    return rows


def summarize_matched_nope(reports):
    grouped = defaultdict(list)
    examples = defaultdict(list)
    for report in reports:
        if not is_matched_nope_run(report):
            continue
        for result in report["eval_results"]:
            grouped[result["context_length"]].append(result["accuracy"])
            examples[result["context_length"]].append(result.get("examples"))
    rows = []
    for context_length, values in sorted(grouped.items()):
        rows.append(
            {
                "context_length": context_length,
                "mean_accuracy": mean(values),
                "std_accuracy": std(values),
                "seeds": len(values),
                "examples_per_seed": examples[context_length],
            }
        )
    return rows


def summarize_extended_seed42(reports):
    wanted_paths = {
        "benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps2000_eval8192.json": "Gated MOGT",
        "benchmark_runs/synthetic_last_value_transformer_ctx128_seed42_steps2000_eval8192.json": "Transformer RoPE",
        "benchmark_runs/synthetic_last_value_transformer_rope1e6_ctx128_seed42_steps2000.json": "Transformer RoPE theta=1e6",
        "benchmark_runs/synthetic_last_value_transformer_nope_ctx128_seed42_steps2000.json": "Transformer NoPE",
    }
    by_path = {report["_path"]: report for report in reports}
    rows = []
    for path, label in wanted_paths.items():
        report = by_path.get(path)
        if report is None:
            continue
        row = {
            "model": label,
            "path": path,
            "examples": {
                result["context_length"]: result["examples"]
                for result in report["eval_results"]
            },
            "accuracy": {
                result["context_length"]: result["accuracy"]
                for result in report["eval_results"]
            },
        }
        rows.append(row)
    return rows


def summarize_ablations(reports):
    wanted_paths = {
        "benchmark_runs/synthetic_last_value_mogt_ctx128_seed42_steps500.json": "MOGT, hybrid, 500 steps",
        "benchmark_runs/synthetic_last_value_mogt_sequential_ctx128_seed42_steps500.json": "MOGT, sequential, 500 steps",
        "benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps500.json": "Gated MOGT, 500 steps",
        "benchmark_runs/synthetic_last_value_mogt_gated_ctx128_seed42_steps2000.json": "Gated MOGT, 2000 steps",
        "benchmark_runs/synthetic_last_value_transformer_ctx128_seed42_steps2000.json": "Transformer RoPE, 2000 steps",
        "benchmark_runs/synthetic_last_value_transformer_nope_ctx128_seed42_steps2000.json": "Transformer NoPE, 2000 steps",
    }
    by_path = {report["_path"]: report for report in reports}
    rows = []
    for path, label in wanted_paths.items():
        report = by_path.get(path)
        if report is None:
            continue
        by_context = {result["context_length"]: result for result in report["eval_results"]}
        rows.append(
            {
                "variant": label,
                "params": report["params"],
                "train_elapsed_s": report["train_elapsed_s"],
                "peak_memory_mb": report["peak_memory_mb"],
                "acc_128": by_context.get(128, {}).get("accuracy"),
                "acc_256": by_context.get(256, {}).get("accuracy"),
                "acc_512": by_context.get(512, {}).get("accuracy"),
                "acc_1024": by_context.get(1024, {}).get("accuracy"),
                "path": path,
            }
        )
    return rows


def summarize_scale512(reports):
    wanted_paths = {
        "benchmark_runs/synthetic_last_value_mogt_gated_ctx512_seed42_steps2000.json": "Gated MOGT direct train512",
        "benchmark_runs/synthetic_last_value_mogt_gated_curriculum_ctx512_seed42_steps3000.json": "Gated MOGT curriculum to 512",
        "benchmark_runs/synthetic_last_value_mogt_dualgate_dense_ctx512_seed42_steps2000.json": "MOGT Cayley dual-gate dense train512",
        "benchmark_runs/synthetic_last_value_mogt_identity_dualgate_dense_ctx512_seed42_steps400.json": "MOGT identity dual-gate dense train512, 400 steps",
        "benchmark_runs/synthetic_last_value_mogt_identity_dualgate_dense_ctx512_seed42_steps2000.json": "MOGT identity dual-gate dense train512, 2000 steps",
        "benchmark_runs/synthetic_last_value_mogt_identity_dualgate_keep12_dense_ctx512_seed42_steps1000.json": "MOGT identity dual-gate keep12 dense train512",
        "benchmark_runs/synthetic_last_value_mogt_identity_dualgate_damp10002_dense_ctx512_seed42_steps1000.json": "MOGT identity dual-gate damp1.0002 dense train512",
        "benchmark_runs/synthetic_last_value_mogt_identity_valuegate_only_dense_ctx512_seed42_steps2000.json": "MOGT identity value-gate-only dense train512",
        "benchmark_runs/synthetic_last_value_mogt_identity_coupled_value_forget_dense_ctx512_seed42_steps2000.json": "MOGT identity coupled write-forget dense train512",
        "benchmark_runs/synthetic_last_value_mogt_identity_forgetrelu_dualgate_dense_ctx512_seed42_steps1000.json": "MOGT identity forget-ReLU dual-gate dense train512",
        "benchmark_runs/synthetic_last_value_mogt_identity_resgate_dualgate_dense_ctx512_seed42_steps1000.json": "MOGT identity residual transport-gate dense train512",
        "benchmark_runs/synthetic_last_value_transformer_nope_dense_ctx512_seed42_steps400.json": "Transformer NoPE dense train512, 400 steps",
        "benchmark_runs/synthetic_last_value_transformer_nope_dense_ctx512_seed42_steps2000.json": "Transformer NoPE dense train512, 2000 steps",
        "benchmark_runs/synthetic_last_value_transformer_nope_ctx512_seed42_steps2000.json": "Transformer NoPE direct train512",
    }
    by_path = {report["_path"]: report for report in reports}
    rows = []
    for path, label in wanted_paths.items():
        report = by_path.get(path)
        if report is None:
            continue
        rows.append(
            {
                "variant": label,
                "path": path,
                "train_elapsed_s": report["train_elapsed_s"],
                "peak_memory_mb": report["peak_memory_mb"],
                "accuracy": {
                    result["context_length"]: result["accuracy"]
                    for result in report["eval_results"]
                },
            }
        )
    return rows


def summarize_coupled_long_probe(reports):
    path = (
        "benchmark_runs/"
        "synthetic_last_value_mogt_identity_coupled_value_forget_dense_ctx512_seed42_steps2000_eval65536.json"
    )
    by_path = {report["_path"]: report for report in reports}
    report = by_path.get(path)
    if report is None:
        return None
    return {
        "path": path,
        "examples": {
            result["context_length"]: result["examples"]
            for result in report["eval_results"]
        },
        "accuracy": {
            result["context_length"]: result["accuracy"]
            for result in report["eval_results"]
        },
        "peak_memory_mb": report["peak_memory_mb"],
        "train_elapsed_s": report["train_elapsed_s"],
    }


def summarize_gate_diagnostics(reports):
    path = (
        "benchmark_runs/"
        "synthetic_last_value_mogt_identity_coupled_value_forget_dense_ctx512_seed42_steps2000.json"
    )
    by_path = {report["_path"]: report for report in reports}
    report = by_path.get(path)
    if report is None:
        return None
    diagnostics = report.get("gate_diagnostics")
    if not diagnostics:
        return None
    return {
        "path": path,
        "diagnostics": diagnostics,
    }


def is_train512_dense_run(report):
    config = report["config"]
    label = model_label(report)
    return (
        report.get("task") == "last_value_tracking"
        and config.get("train_context") == 512
        and config.get("steps") == 2000
        and config.get("dense_loss") is True
        and config.get("batch_size") == 16
        and config.get("eval_batch_size") == 4
        and config.get("num_values") == 16
        and config.get("max_updates") == 4
        and config.get("seed") in {7, 42, 123}
        and (
            (
                label == "Gated MOGT"
                and config.get("connection_impl") == "identity"
                and config.get("transport_gate") is True
                and config.get("value_gate") is True
            )
            or (
                label == "Coupled MOGT"
                and config.get("connection_impl") == "identity"
                and config.get("value_gate") is True
                and config.get("couple_forget_to_value_gate") is True
            )
            or label == "Transformer NoPE"
        )
    )


def train512_dense_label(report):
    label = model_label(report)
    if label == "Gated MOGT":
        return "MOGT identity dual-gate dense"
    if label == "Coupled MOGT":
        return "MOGT coupled write-forget dense"
    return label + " dense"


def summarize_train512_dense(reports):
    preferred_paths = {
        "MOGT coupled write-forget dense": [
            f"benchmark_runs/synthetic_last_value_mogt_identity_coupled_value_forget_dense_stdreport_biasm2_ctx512_seed{seed}_steps2000.json"
            for seed in (7, 42, 123)
        ],
        "MOGT identity dual-gate dense": [
            f"benchmark_runs/synthetic_last_value_mogt_identity_dualgate_dense_ctx512_seed{seed}_steps2000.json"
            for seed in (7, 42, 123)
        ],
        "Transformer NoPE dense": [
            f"benchmark_runs/synthetic_last_value_transformer_nope_dense_stdreport_ctx512_seed{seed}_steps2000.json"
            for seed in (7, 42, 123)
        ],
    }
    by_path = {report["_path"]: report for report in reports}
    grouped = defaultdict(list)
    for label, paths in preferred_paths.items():
        for path in paths:
            report = by_path.get(path)
            if report is None:
                continue
            if not is_train512_dense_run(report):
                continue
            if label == "MOGT coupled write-forget dense":
                bias = report["config"].get("value_gate_bias")
                if bias != -2.0:
                    continue
            for result in report["eval_results"]:
                grouped[(label, result["context_length"])].append(result["accuracy"])
    rows = []
    for (label, context_length), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        rows.append(
            {
                "model": label,
                "context_length": context_length,
                "mean_accuracy": mean(values),
                "std_accuracy": std(values),
                "seeds": len(values),
            }
        )
    return rows


def pct(value):
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def gate_pct(value):
    if value is None:
        return "-"
    if abs(value) < 0.001:
        return f"{100.0 * value:.4f}%"
    return pct(value)


def write_outputs(summary):
    json_path = "benchmark_runs/synthetic_last_value_summary_20260503.json"
    md_path = "benchmark_runs/synthetic_last_value_summary_20260503.md"
    os.makedirs("benchmark_runs", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    lines = [
        "# Synthetic Last-Value Tracking Summary",
        "",
        "Date: 2026-05-03",
        "",
        "Task: train at context 128, then evaluate last written value at longer contexts.",
        "All main runs use 2 layers, d_model 128, vocab 128, 16 values, 1-4 writes, batch 64, 2000 steps, BF16 on one NVIDIA L4.",
        "",
        "## Main 3-Seed Results",
        "",
        "| Context | Gated MOGT acc | Transformer RoPE acc | Gap |",
        "|---:|---:|---:|---:|",
    ]

    by_key = {(row["model"], row["context_length"]): row for row in summary["main"]}
    contexts = sorted({row["context_length"] for row in summary["main"]})
    for context_length in contexts:
        mogt = by_key.get(("Gated MOGT", context_length))
        transformer = by_key.get(("Transformer RoPE", context_length))
        mogt_text = "-"
        transformer_text = "-"
        gap_text = "-"
        if mogt:
            mogt_text = f"{pct(mogt['mean_accuracy'])} +/- {pct(mogt['std_accuracy'])}"
        if transformer:
            transformer_text = f"{pct(transformer['mean_accuracy'])} +/- {pct(transformer['std_accuracy'])}"
        if mogt and transformer:
            gap_text = f"{100.0 * (mogt['mean_accuracy'] - transformer['mean_accuracy']):.2f} pp"
        lines.append(f"| {context_length} | {mogt_text} | {transformer_text} | {gap_text} |")

    lines.extend(
        [
            "",
            "## NoPE Transformer Matched-Eval 3-Seed Baseline",
            "",
            "These runs remove RoPE and use the same 2048 eval examples per context as the main table.",
            "",
            "| Context | Transformer NoPE acc | Eval examples / seed |",
            "|---:|---:|---:|",
        ]
    )
    for row in summary["matched_nope"]:
        examples = "/".join(str(value) for value in row.get("examples_per_seed", []))
        lines.append(
            f"| {row['context_length']} | {pct(row['mean_accuracy'])} +/- {pct(row['std_accuracy'])} | {examples} |"
        )

    lines.extend(
        [
            "",
            "## NoPE Transformer Earlier Light-Eval Baseline",
            "",
            "These runs remove RoPE and use 128 eval examples per context. They are not directly mixed into the main table because the eval protocol is lighter, but they answer whether the RoPE baseline was artificially weak.",
            "",
            "| Context | Transformer NoPE acc |",
            "|---:|---:|",
        ]
    )
    for row in summary["nope"]:
        lines.append(
            f"| {row['context_length']} | {pct(row['mean_accuracy'])} +/- {pct(row['std_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Seed-42 Long Extrapolation",
            "",
            "These runs train at context 128 and evaluate farther out with 64-128 examples per context.",
            "",
            "| Model | 128 | 1024 | 2048 | 4096 | 8192 | Artifact |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["extended_seed42"]:
        acc = row["accuracy"]
        lines.append(
            f"| {row['model']} | {pct(acc.get(128))} | {pct(acc.get(1024))} | "
            f"{pct(acc.get(2048))} | {pct(acc.get(4096))} | {pct(acc.get(8192))} | "
            f"`{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## 1024-Context Individual Seeds",
            "",
            "| Model | Seed | Accuracy | Loss | Artifact |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["individual_1024"]:
        lines.append(
            f"| {row['model']} | {row['seed']} | {pct(row['accuracy_1024'])} | "
            f"{row['loss_1024']:.4f} | `{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Train-512 Dense 3-Seed Results",
            "",
            "Both models use dense state supervision, train context 512, 2000 steps, batch 16, and 64 eval examples per context.",
            "",
            "| Context | MOGT coupled write-forget dense | MOGT identity dual-gate dense | Transformer NoPE dense | Coupled gap vs NoPE |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    dense_by_key = {
        (row["model"], row["context_length"]): row for row in summary["train512_dense"]
    }
    dense_contexts = sorted({row["context_length"] for row in summary["train512_dense"]})
    for context_length in dense_contexts:
        coupled = dense_by_key.get(("MOGT coupled write-forget dense", context_length))
        mogt = dense_by_key.get(("MOGT identity dual-gate dense", context_length))
        transformer = dense_by_key.get(("Transformer NoPE dense", context_length))
        coupled_text = "-"
        mogt_text = "-"
        transformer_text = "-"
        gap_text = "-"
        if coupled:
            coupled_text = f"{pct(coupled['mean_accuracy'])} +/- {pct(coupled['std_accuracy'])}"
        if mogt:
            mogt_text = f"{pct(mogt['mean_accuracy'])} +/- {pct(mogt['std_accuracy'])}"
        if transformer:
            transformer_text = f"{pct(transformer['mean_accuracy'])} +/- {pct(transformer['std_accuracy'])}"
        if coupled and transformer:
            gap_text = f"{100.0 * (coupled['mean_accuracy'] - transformer['mean_accuracy']):.2f} pp"
        lines.append(
            f"| {context_length} | {coupled_text} | {mogt_text} | "
            f"{transformer_text} | {gap_text} |"
        )

    if summary["coupled_long_probe"] is not None:
        row = summary["coupled_long_probe"]
        acc = row["accuracy"]
        examples = row["examples"]
        lines.extend(
            [
                "",
                "## Coupled Write-Forget Long Probe",
                "",
                "Seed 42 only, trained at context 512. This probe uses fewer eval examples than the main table and is a stress test rather than a final multi-seed result.",
                "",
                "| Context | Accuracy | Eval examples |",
                "|---:|---:|---:|",
            ]
        )
        for context_length in sorted(acc):
            lines.append(
                f"| {context_length} | {pct(acc[context_length])} | "
                f"{examples.get(context_length, '-')} |"
            )
        lines.extend(
            [
                "",
                f"Artifact: `{row['path']}`. Peak memory {row['peak_memory_mb']:.1f} MB; train elapsed {row['train_elapsed_s']:.1f}s.",
            ]
        )

    if summary["gate_diagnostics"] is not None:
        lines.extend(
            [
                "",
                "## Coupled Gate Diagnostics",
                "",
                "Seed 42, train context 512. Gate values are means on a fresh training-context batch after training.",
                "",
                "| Block | SET | value token | filler | QUERY |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["gate_diagnostics"]["diagnostics"]:
            stats = row["mean_gate"]
            lines.append(
                f"| {row['block']} | {gate_pct(stats.get('set'))} | "
                f"{gate_pct(stats.get('value'))} | {gate_pct(stats.get('filler'))} | "
                f"{gate_pct(stats.get('query'))} |"
            )
        lines.append("")
        lines.append(f"Artifact: `{summary['gate_diagnostics']['path']}`.")

    lines.extend(
        [
            "",
            "## Train-512 Scaling Probe",
            "",
            "Seed 42 only. This is a scale stress test, not a final result.",
            "",
            "| Variant | 512 | 1024 | 2048 | 4096 | 8192 | Peak MB | Train elapsed | Artifact |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["scale512"]:
        acc = row["accuracy"]
        lines.append(
            f"| {row['variant']} | {pct(acc.get(512))} | {pct(acc.get(1024))} | "
            f"{pct(acc.get(2048))} | {pct(acc.get(4096))} | {pct(acc.get(8192))} | "
            f"{row['peak_memory_mb']:.1f} | {row['train_elapsed_s']:.1f}s | `{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Seed-42 Ablation",
            "",
            "| Variant | 128 | 256 | 512 | 1024 | Peak MB | Train elapsed | Artifact |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["ablations"]:
        lines.append(
            f"| {row['variant']} | {pct(row['acc_128'])} | {pct(row['acc_256'])} | "
            f"{pct(row['acc_512'])} | {pct(row['acc_1024'])} | "
            f"{row['peak_memory_mb']:.1f} | {row['train_elapsed_s']:.1f}s | `{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Original MOGT stays near chance on this overwrite-style memory task.",
            "- Adding a token-dependent transport gate makes the recurrence learnable and gives much stronger length extrapolation than the matched Transformer in this small setting.",
            "- Removing RoPE makes Transformer a much stronger extrapolation baseline. Under the matched 2048-example eval protocol, NoPE Transformer reaches 68.64% +/- 15.51% at 1024, close to the gated MOGT main-table mean of 69.71% +/- 18.69%. Future claims must treat NoPE/position-robust attention as a first-class baseline rather than a caveat.",
            "- Train512 exposed two separate issues: Cayley/Magnus transport makes overwrite memory hard to optimize, while identity transport plus dual gates and dense state supervision learns quickly.",
            "- Coupling the value/write gate to transport forgetting is the strongest current mechanism: across seeds 7/42/123 it reaches 100.00% +/- 0.00% from 512 through 8192 after training only at context 512.",
            "- The value-gate-only ablation learns in-distribution but drops to 48.44% at 4096 and 25.00% at 8192, so explicit transport/forget gating is still needed for state preservation.",
            "- The residual transport-gate mode was numerically unstable in the seed-42 probe, while forget-ReLU was stable but much worse at far extrapolation.",
            "- Under dense train512 supervision, independently gated identity MOGT learns faster early but trails NoPE at far context; coupled write-forget MOGT fixes the single-slot far-extrapolation gap while still using more memory and time than the small NoPE Transformer implementation.",
            "- This is not yet a language-modeling win. It is a scoped synthetic result that identifies a plausible paper direction: gated affine transport for recurrent state tracking and length extrapolation.",
            "- Next evidence needed: larger contexts, more tasks, training-token matched curves, throughput/memory tradeoffs, and LM hybrid experiments.",
        ]
    )

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return json_path, md_path


def main():
    reports = load_reports()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "main": summarize_main(reports),
        "individual_1024": summarize_individual_1024(reports),
        "nope": summarize_nope(reports),
        "matched_nope": summarize_matched_nope(reports),
        "extended_seed42": summarize_extended_seed42(reports),
        "ablations": summarize_ablations(reports),
        "scale512": summarize_scale512(reports),
        "coupled_long_probe": summarize_coupled_long_probe(reports),
        "gate_diagnostics": summarize_gate_diagnostics(reports),
        "train512_dense": summarize_train512_dense(reports),
    }
    json_path, md_path = write_outputs(summary)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
