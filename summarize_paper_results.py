import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def cell(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{pct(mean(values))} +/- {pct(std(values))}"


def load_standard_reports() -> list[dict[str, Any]]:
    rows = []
    for raw_path in sorted(glob.glob("benchmark_runs/*.json")):
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        standard = payload.get("standard_report")
        if standard is None or standard.get("status") != "ok":
            continue
        rows.append({"path": str(path), "payload": payload, "standard": standard})
    return rows


def accuracies(
    reports: list[dict[str, Any]],
    *,
    task: str,
    model: str | None = None,
    variant_contains: str | None = None,
    path_contains: str | None = None,
    train_context: int | None = None,
    steps: int | None = None,
    num_slots: int | None = None,
    dense_loss: bool | None = None,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        standard = report["standard"]
        data = standard.get("data", {})
        training = standard.get("training", {})
        if standard.get("task") != task:
            continue
        if model is not None and standard.get("model") != model:
            continue
        if variant_contains is not None and variant_contains not in str(standard.get("variant")):
            continue
        if path_contains is not None and path_contains not in report["path"]:
            continue
        if train_context is not None and data.get("train_context") != train_context:
            continue
        if steps is not None and training.get("steps") != steps:
            continue
        if num_slots is not None and data.get("num_slots") != num_slots:
            continue
        if dense_loss is not None and training.get("dense_loss") is not dense_loss:
            continue
        for context, accuracy in standard.get("metrics", {}).get("accuracy_by_context", {}).items():
            grouped[str(context)].append(float(accuracy))
    return dict(grouped)


def contexts(*groups: dict[str, list[float]]) -> list[str]:
    keys = set()
    for group in groups:
        keys.update(group)
    return sorted(keys, key=int)


def lm_pilot_rows(
    reports: list[dict[str, Any]],
    *,
    train_context: int,
    steps: int,
    d_model: int,
    num_layers: int,
) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        standard = report["standard"]
        data = standard.get("data", {})
        training = standard.get("training", {})
        model_config = standard.get("model_config", {})
        if standard.get("task") != "language_modeling":
            continue
        if data.get("train_context") != train_context:
            continue
        if training.get("steps") != steps:
            continue
        if model_config.get("d_model") != d_model:
            continue
        if model_config.get("num_layers") != num_layers:
            continue
        metrics = standard.get("metrics", {})
        environment = standard.get("environment", {})
        rows.append(
            {
                "run_name": standard.get("run_name"),
                "training_steps": training.get("steps"),
                "training_seed": training.get("seed"),
                "lr": training.get("lr"),
                "model": standard.get("model"),
                "variant": standard.get("variant"),
                "loss": metrics.get("loss"),
                "ppl": metrics.get("ppl"),
                "peak_memory_mb": environment.get("peak_memory_mb"),
                "elapsed_seconds": environment.get("elapsed_seconds"),
                "model_config": model_config,
                "path": report["path"],
            }
        )
    return sorted(rows, key=lambda row: (str(row["model"]), str(row["variant"])))


def single_mogt_layer_position(block_types: list[str]) -> str | None:
    mogt_indices = [index for index, block_type in enumerate(block_types) if block_type == "mogt"]
    if not mogt_indices:
        return "attention-only"
    if len(mogt_indices) == 1:
        return f"layer {mogt_indices[0]}"
    return "layers " + "+".join(str(index) for index in mogt_indices)


def position_sort_key(position: str) -> tuple[int, int]:
    if position == "attention-only":
        return (0, -1)
    if position.startswith("layer "):
        return (1, int(position.rsplit(" ", 1)[1]))
    if position.startswith("layers "):
        return (2, int(position.split(" ", 1)[1].split("+", 1)[0]))
    return (3, 0)


def lr_matches(row: dict[str, Any], value: float) -> bool:
    lr = row.get("lr")
    return lr is not None and abs(float(lr) - value) <= 1e-12


def mogt_scale_matches(
    row: dict[str, Any],
    *,
    residual_scale: float = 1.0,
    ffn_residual_scale: float = 1.0,
    schedule: str = "constant",
) -> bool:
    config = row.get("model_config", {})
    residual = config.get("mogt_residual_scale", 1.0)
    ffn_residual = config.get("mogt_ffn_residual_scale", 1.0)
    residual_schedule = config.get("mogt_residual_scale_schedule", "constant")
    return (
        abs(float(residual) - residual_scale) <= 1e-12
        and abs(float(ffn_residual) - ffn_residual_scale) <= 1e-12
        and residual_schedule == schedule
    )


def mogt_residual_gate_matches(row: dict[str, Any], enabled: bool = False) -> bool:
    config = row.get("model_config", {})
    return bool(config.get("mogt_residual_gate", False)) is bool(enabled)


def mogt_lr_mult_matches(row: dict[str, Any], value: float = 1.0) -> bool:
    config = row.get("model_config", {})
    return abs(float(config.get("mogt_lr_mult", 1.0)) - value) <= 1e-12


def write_snapshot(path: Path, reports: list[dict[str, Any]]) -> None:
    single_mogt = accuracies(
        reports,
        task="last_value_tracking",
        model="mogt",
        path_contains="stdreport_biasm2_ctx512",
        train_context=512,
        steps=2000,
        dense_loss=True,
    )
    single_nope = accuracies(
        reports,
        task="last_value_tracking",
        model="transformer",
        path_contains="transformer_nope_dense_stdreport_ctx512",
        train_context=512,
        steps=2000,
        dense_loss=True,
    )

    multi_mogt_curr = accuracies(
        reports,
        task="tracked_multislot_last_value",
        model="mogt",
        variant_contains="slots4_final_curriculum",
        train_context=512,
        steps=3000,
        num_slots=4,
        dense_loss=False,
    )
    multi_nope_curr = accuracies(
        reports,
        task="tracked_multislot_last_value",
        model="transformer",
        variant_contains="slots4_final_curriculum",
        train_context=512,
        steps=3000,
        num_slots=4,
        dense_loss=False,
    )
    multi_mamba_curr = accuracies(
        reports,
        task="tracked_multislot_last_value",
        model="mamba",
        variant_contains="mamba_d192_slots4_final_curriculum",
        train_context=512,
        steps=3000,
        num_slots=4,
        dense_loss=False,
    )
    multi_mogt_direct = accuracies(
        reports,
        task="tracked_multislot_last_value",
        model="mogt",
        variant_contains="slots4_final",
        path_contains="rank_finalonly_stdreport",
        train_context=512,
        steps=3000,
        num_slots=4,
        dense_loss=False,
    )
    multi_nope_direct = accuracies(
        reports,
        task="tracked_multislot_last_value",
        model="transformer",
        variant_contains="transformer_nope_slots4_final",
        path_contains="transformer_nope_finalonly_stdreport",
        train_context=512,
        steps=3000,
        num_slots=4,
        dense_loss=False,
    )

    lines = [
        "# Results Snapshot",
        "",
        "Last updated: 2026-05-05",
        "",
        "This file is generated from `mogt-experiment-v1` standard reports by",
        "`summarize_paper_results.py`.",
        "",
        "## Standard Evidence Status",
        "",
        "- Standard report index: `benchmark_runs/standard_report_index.md`",
        "- Last-value summary: `benchmark_runs/synthetic_last_value_summary_20260503.md`",
        "- Multi-slot summary: `benchmark_runs/synthetic_multislot_standard_summary_20260504.md`",
        f"- Standard reports loaded into this snapshot: {len(reports)} ok reports.",
        "",
        "## Main Table A: Single-Slot Overwrite State Tracking",
        "",
        "| Context | Coupled MOGT | Transformer NoPE | Gap |",
        "|---:|---:|---:|---:|",
    ]
    for context in contexts(single_mogt, single_nope):
        mogt_values = single_mogt.get(context, [])
        nope_values = single_nope.get(context, [])
        gap = "-"
        if mogt_values and nope_values:
            gap = f"{100.0 * (mean(mogt_values) - mean(nope_values)):.2f} pp"
        lines.append(f"| {context} | {cell(mogt_values)} | {cell(nope_values)} | {gap} |")

    lines.extend(
        [
            "",
            "## Main Table B: Tracked 4-Slot Final-Query Routing",
            "",
            "| Context | Slot-addressed MOGT | Transformer NoPE | HF-Mamba d192 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for context in contexts(multi_mogt_curr, multi_nope_curr, multi_mamba_curr):
        lines.append(
            f"| {context} | {cell(multi_mogt_curr.get(context, []))} | "
            f"{cell(multi_nope_curr.get(context, []))} | "
            f"{cell(multi_mamba_curr.get(context, []))} |"
        )

    lines.extend(
        [
            "",
            "## Ablation: Curriculum Is An Optimization Condition",
            "",
            "| Context | MOGT direct | NoPE direct | MOGT curriculum |",
            "|---:|---:|---:|---:|",
        ]
    )
    for context in contexts(multi_mogt_direct, multi_nope_direct, multi_mogt_curr):
        lines.append(
            f"| {context} | {cell(multi_mogt_direct.get(context, []))} | "
            f"{cell(multi_nope_direct.get(context, []))} | "
            f"{cell(multi_mogt_curr.get(context, []))} |"
        )

    throughput_path = Path(
        "benchmark_runs/throughput_core_operator_d768_len8192_16384_32768_20260504.json"
    )
    if throughput_path.exists():
        throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
        results = throughput.get("results_ms", {})
        affine = {int(length): float(value) for length, value in results.get("affine_triton_hybrid_ms", [])}
        attention = {int(length): float(value) for length, value in results.get("attention_ms", [])}
        lines.extend(
            [
                "",
                "## Systems Snapshot: Core Operator Timing",
                "",
                "Core timing only: affine scan excludes connection/value projection,",
                "matrix exponential/Cayley construction, normalization, FFN, and LM head;",
                "attention timing measures FlashAttention/SDPA core only.",
                "",
                "| Length | Affine Triton hybrid ms | Attention core ms | Attention / affine |",
                "|---:|---:|---:|---:|",
            ]
        )
        for length in sorted({*affine, *attention}):
            aff = affine.get(length)
            attn = attention.get(length)
            speedup = "-"
            if aff is not None and attn is not None and aff != 0:
                speedup = f"{attn / aff:.2f}x"
            lines.append(
                f"| {length} | {aff:.2f} | {attn:.2f} | {speedup} |"
            )

    backbone_path = Path(
        "benchmark_runs/backbone_throughput_identity_coupled_d768_l2_len8192_16384_32768_20260504.json"
    )
    if backbone_path.exists():
        backbone = json.loads(backbone_path.read_text(encoding="utf-8"))
        by_model = {record["model"]: record for record in backbone.get("records", [])}
        mogt_rows = {
            int(row["length"]): row
            for row in by_model.get("mogt", {}).get("results", [])
        }
        transformer_rows = {
            int(row["length"]): row
            for row in by_model.get("transformer", {}).get("results", [])
        }
        lines.extend(
            [
                "",
                "## Systems Snapshot: Backbone Forward Timing",
                "",
                "Backbone hidden-state forward only: embeddings, sequence blocks, and",
                "final normalization. This excludes LM head, loss, backward pass,",
                "optimizer, and KV-cache decode behavior.",
                "",
                "| Length | MOGT ms | Transformer NoPE ms | Transformer / MOGT |",
                "|---:|---:|---:|---:|",
            ]
        )
        for length in sorted({*mogt_rows, *transformer_rows}):
            mogt_ms = mogt_rows.get(length, {}).get("elapsed_ms")
            transformer_ms = transformer_rows.get(length, {}).get("elapsed_ms")
            ratio = "-"
            if mogt_ms and transformer_ms and mogt_ms != 0:
                ratio = f"{transformer_ms / mogt_ms:.2f}x"
            lines.append(
                f"| {length} | {mogt_ms:.2f} | {transformer_ms:.2f} | {ratio} |"
            )

    lm_rows = lm_pilot_rows(
        reports,
        train_context=8192,
        steps=10,
        d_model=128,
        num_layers=2,
    )
    if lm_rows:
        lines.extend(
            [
                "",
                "## Language Modeling Pilot: Hybrid Wiring Sanity",
                "",
                "This is a 10-step, one-seed WikiText-103 pilot at context 8192,",
                "`d_model=128`, and two layers. It is only a wiring and optimization",
                "sanity check, not a language-modeling quality claim.",
                "",
                "| Run | Model | Val loss | PPL | Peak MB | Elapsed s |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in lm_rows:
            loss = row.get("loss")
            ppl_value = row.get("ppl")
            peak_memory = row.get("peak_memory_mb")
            elapsed = row.get("elapsed_seconds")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['run_name']}`",
                        str(row["variant"]),
                        f"{loss:.4f}" if loss is not None else "-",
                        f"{ppl_value:.2f}" if ppl_value is not None else "-",
                        f"{peak_memory:.1f}" if peak_memory is not None else "-",
                        f"{elapsed:.2f}" if elapsed is not None else "-",
                    ]
                )
                + " |"
            )

    ratio_rows = []
    for ratio_steps in (5, 50, 200):
        ratio_rows.extend(
            [
                row
                for row in lm_pilot_rows(
                    reports,
                    train_context=8192,
                    steps=ratio_steps,
                    d_model=128,
                    num_layers=4,
                )
                if row.get("run_name") and "hybrid_ratio_sweep" in str(row["run_name"])
            ]
        )
    if ratio_rows:
        ratio_5_rows = [row for row in ratio_rows if row.get("training_steps") == 5]
        ratio_50_3seed = [
            row
            for row in ratio_rows
            if row.get("training_steps") == 50 and "3seed" in str(row.get("run_name"))
        ]
        ratio_50_seed42_100 = [
            row
            for row in ratio_rows
            if row.get("training_steps") == 50
            and "3seed" not in str(row.get("run_name"))
            and row["model_config"].get("mogt_layer_fraction") == 1.0
        ]
        ratio_200_rows = [row for row in ratio_rows if row.get("training_steps") == 200]
        lines.extend(
            [
                "",
                "## Hybrid Ratio Pilot",
                "",
                "These are context-8192, `d_model=128`, four-layer pilot runs.",
                "The 5-step rows are one-seed queueing signals. The 50-step",
                "aggregate uses seeds 7/42/123 and two validation batches.",
                "",
                "### 5-Step Ratio Probe",
                "",
                "| Steps | MOGT fraction | MOGT layers | Zero-attn init | Val loss | PPL | Elapsed s |",
                "|---:|---:|---:|---|---:|---:|---:|",
            ]
        )
        for row in sorted(
            ratio_5_rows,
            key=lambda item: (
                item.get("training_steps") or 0,
                bool(item["model_config"].get("zero_init_attention_out")),
                item["model_config"].get("mogt_layer_fraction", 0.0),
            ),
        ):
            config = row["model_config"]
            steps = row["training_steps"]
            fraction = config.get("mogt_layer_fraction")
            mogt_layers = config.get("mogt_layer_count")
            zero_attn = "yes" if config.get("zero_init_attention_out") else "no"
            loss = row.get("loss")
            ppl_value = row.get("ppl")
            elapsed = row.get("elapsed_seconds")
            lines.append(
                f"| {steps} | {fraction:.2f} | {mogt_layers} | "
                f"{zero_attn} | {loss:.4f} | {ppl_value:.2f} | {elapsed:.2f} |"
            )

        if ratio_50_3seed:
            grouped_ratio_50: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for row in ratio_50_3seed:
                grouped_ratio_50[float(row["model_config"].get("mogt_layer_fraction"))].append(row)
            lines.extend(
                [
                    "",
                    "### 50-Step Zero-Init Aggregate",
                    "",
                    "| MOGT fraction | Seeds | Mean val loss | Std | Mean PPL | Mean elapsed s |",
                    "|---:|---|---:|---:|---:|---:|",
                ]
            )
            for fraction, rows_for_fraction in sorted(grouped_ratio_50.items()):
                losses = [float(row["loss"]) for row in rows_for_fraction if row.get("loss") is not None]
                ppls = [float(row["ppl"]) for row in rows_for_fraction if row.get("ppl") is not None]
                elapsed = [
                    float(row["elapsed_seconds"])
                    for row in rows_for_fraction
                    if row.get("elapsed_seconds") is not None
                ]
                seeds = sorted({int(row["training_seed"]) for row in rows_for_fraction})
                lines.append(
                    f"| {fraction:.2f} | {','.join(str(seed) for seed in seeds)} | "
                    f"{mean(losses):.4f} | {std(losses):.4f} | "
                    f"{mean(ppls):.2f} | {mean(elapsed):.2f} |"
                )
        if ratio_50_seed42_100:
            row = ratio_50_seed42_100[0]
            lines.extend(
                [
                    "",
                    f"Seed-42 100% MOGT zero-init control at 50 steps: val loss "
                    f"{row['loss']:.4f}, PPL {row['ppl']:.2f}.",
                ]
            )

        if ratio_200_rows:
            grouped_ratio_200: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for row in ratio_200_rows:
                grouped_ratio_200[float(row["model_config"].get("mogt_layer_fraction"))].append(row)
            lines.extend(
                [
                    "",
                    "### 200-Step Zero-Init Aggregate",
                    "",
                    "| MOGT fraction | Seeds | Mean val loss | Std | Mean PPL | Mean elapsed s |",
                    "|---:|---|---:|---:|---:|---:|",
                ]
            )
            for fraction, rows_for_fraction in sorted(grouped_ratio_200.items()):
                losses = [float(row["loss"]) for row in rows_for_fraction if row.get("loss") is not None]
                ppls = [float(row["ppl"]) for row in rows_for_fraction if row.get("ppl") is not None]
                elapsed = [
                    float(row["elapsed_seconds"])
                    for row in rows_for_fraction
                    if row.get("elapsed_seconds") is not None
                ]
                seeds = sorted({int(row["training_seed"]) for row in rows_for_fraction})
                lines.append(
                    f"| {fraction:.2f} | {','.join(str(seed) for seed in seeds)} | "
                    f"{mean(losses):.4f} | {std(losses):.4f} | "
                    f"{mean(ppls):.2f} | {mean(elapsed):.2f} |"
                )

    position_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=200,
            d_model=128,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0003)
        and mogt_scale_matches(row)
        and mogt_residual_gate_matches(row, False)
        and mogt_lr_mult_matches(row)
    ]
    grouped_positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        position = single_mogt_layer_position(row["model_config"].get("block_types") or [])
        if position is not None:
            grouped_positions[position].append(row)
    if grouped_positions:
        attention_losses = [
            float(row["loss"])
            for row in grouped_positions.get("attention-only", [])
            if row.get("loss") is not None
        ]
        attention_mean = mean(attention_losses) if attention_losses else None
        lines.extend(
            [
                "",
                "### 200-Step Single-Layer Position Ablation",
                "",
                "Same setup as the 200-step ratio follow-up. The layer-1 row is the",
                "original 25% ratio run; layers 2 and 3 are explicit-index follow-ups.",
                "Layer 0 currently has only seed 42, so its row is a provisional",
                "diagnostic rather than a full aggregate.",
                "",
                "| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for position, rows_for_position in sorted(
            grouped_positions.items(), key=lambda item: position_sort_key(item[0])
        ):
            losses = [float(row["loss"]) for row in rows_for_position if row.get("loss") is not None]
            ppls = [float(row["ppl"]) for row in rows_for_position if row.get("ppl") is not None]
            seeds = sorted({int(row["training_seed"]) for row in rows_for_position})
            delta = "-"
            if attention_mean is not None and losses:
                delta = f"{mean(losses) - attention_mean:.4f}"
            lines.append(
                f"| {position} | {','.join(str(seed) for seed in seeds)} | "
                f"{mean(losses):.4f} | {std(losses):.4f} | {delta} | {mean(ppls):.2f} |"
            )

    for scaleup_steps in (500, 1000):
        scaleup_rows = [
            row
            for row in lm_pilot_rows(
                reports,
                train_context=8192,
                steps=scaleup_steps,
                d_model=128,
                num_layers=4,
            )
                if row["model_config"].get("zero_init_attention_out")
                and lr_matches(row, 0.0003)
                and mogt_scale_matches(row)
                and mogt_residual_gate_matches(row, False)
                and mogt_lr_mult_matches(row)
        ]
        grouped_scaleup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scaleup_rows:
            position = single_mogt_layer_position(row["model_config"].get("block_types") or [])
            if position is not None:
                grouped_scaleup[position].append(row)
        if grouped_scaleup:
            attention_losses = [
                float(row["loss"])
                for row in grouped_scaleup.get("attention-only", [])
                if row.get("loss") is not None
            ]
            attention_mean = mean(attention_losses) if attention_losses else None
            lines.extend(
                [
                    "",
                    f"### {scaleup_steps}-Step Late-Layer Scale-Up",
                    "",
                    f"Same context/model size as above, but {scaleup_steps} optimizer steps and eight",
                    "validation batches. This targets the best 200-step position rather",
                    "than sweeping every layer.",
                    "",
                    "| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for position, rows_for_position in sorted(
                grouped_scaleup.items(), key=lambda item: position_sort_key(item[0])
            ):
                losses = [float(row["loss"]) for row in rows_for_position if row.get("loss") is not None]
                ppls = [float(row["ppl"]) for row in rows_for_position if row.get("ppl") is not None]
                elapsed = [
                    float(row["elapsed_seconds"])
                    for row in rows_for_position
                    if row.get("elapsed_seconds") is not None
                ]
                seeds = sorted({int(row["training_seed"]) for row in rows_for_position})
                delta = "-"
                if attention_mean is not None and losses:
                    delta = f"{mean(losses) - attention_mean:.4f}"
                lines.append(
                    f"| {position} | {','.join(str(seed) for seed in seeds)} | "
                    f"{mean(losses):.4f} | {std(losses):.4f} | {delta} | "
                    f"{mean(ppls):.2f} | {mean(elapsed):.2f} |"
                )

    for d192_steps in (500, 1000):
        d192_rows = [
            row
            for row in lm_pilot_rows(
                reports,
                train_context=8192,
                steps=d192_steps,
                d_model=192,
                num_layers=4,
            )
                if row["model_config"].get("zero_init_attention_out")
                and lr_matches(row, 0.0003)
                and mogt_scale_matches(row)
                and mogt_residual_gate_matches(row, False)
                and mogt_lr_mult_matches(row)
        ]
        grouped_d192: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in d192_rows:
            position = single_mogt_layer_position(row["model_config"].get("block_types") or [])
            if position is not None:
                grouped_d192[position].append(row)
        if grouped_d192:
            attention_losses = [
                float(row["loss"])
                for row in grouped_d192.get("attention-only", [])
                if row.get("loss") is not None
            ]
            attention_mean = mean(attention_losses) if attention_losses else None
            lines.extend(
                [
                    "",
                    f"### d_model=192 Width Scale Probe ({d192_steps} Steps)",
                    "",
                    f"Context 8192, four layers, {d192_steps} optimizer steps, eight validation",
                    "batches, and the same late-layer target as the d_model=128 runs.",
                    "",
                    "| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |",
                    "|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for position, rows_for_position in sorted(
                grouped_d192.items(), key=lambda item: position_sort_key(item[0])
            ):
                losses = [float(row["loss"]) for row in rows_for_position if row.get("loss") is not None]
                ppls = [float(row["ppl"]) for row in rows_for_position if row.get("ppl") is not None]
                elapsed = [
                    float(row["elapsed_seconds"])
                    for row in rows_for_position
                    if row.get("elapsed_seconds") is not None
                ]
                params = [
                    int(row["model_config"].get("num_params"))
                    for row in rows_for_position
                    if row["model_config"].get("num_params") is not None
                ]
                seeds = sorted({int(row["training_seed"]) for row in rows_for_position})
                delta = "-"
                if attention_mean is not None and losses:
                    delta = f"{mean(losses) - attention_mean:.4f}"
                lines.append(
                    f"| {position} | {','.join(str(seed) for seed in seeds)} | "
                    f"{mean(losses):.4f} | {std(losses):.4f} | {delta} | "
                    f"{mean(ppls):.2f} | {mean(elapsed):.2f} | "
                    f"{int(mean(params)) if params else '-'} |"
                )

    d192_lr5_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_scale_matches(row)
        and mogt_residual_gate_matches(row, False)
        and mogt_lr_mult_matches(row)
    ]
    grouped_d192_lr5: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in d192_lr5_rows:
        position = single_mogt_layer_position(row["model_config"].get("block_types") or [])
        if position is not None:
            grouped_d192_lr5[position].append(row)
    if grouped_d192_lr5:
        attention_losses = [
            float(row["loss"])
            for row in grouped_d192_lr5.get("attention-only", [])
            if row.get("loss") is not None
        ]
        attention_mean = mean(attention_losses) if attention_losses else None
        lines.extend(
            [
                "",
                "### d_model=192 Learning-Rate Probe (lr=5e-4)",
                "",
                "Context 8192, four layers, 1000 optimizer steps, eight validation",
                "batches. This is a fairness check after lr=5e-4 improved the",
                "layer-2 hybrid seed-42 diagnostic.",
                "",
                "| Position | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for position, rows_for_position in sorted(
            grouped_d192_lr5.items(), key=lambda item: position_sort_key(item[0])
        ):
            losses = [float(row["loss"]) for row in rows_for_position if row.get("loss") is not None]
            ppls = [float(row["ppl"]) for row in rows_for_position if row.get("ppl") is not None]
            elapsed = [
                float(row["elapsed_seconds"])
                for row in rows_for_position
                if row.get("elapsed_seconds") is not None
            ]
            params = [
                int(row["model_config"].get("num_params"))
                for row in rows_for_position
                if row["model_config"].get("num_params") is not None
            ]
            seeds = sorted({int(row["training_seed"]) for row in rows_for_position})
            delta = "-"
            if attention_mean is not None and losses:
                delta = f"{mean(losses) - attention_mean:.4f}"
            lines.append(
                f"| {position} | {','.join(str(seed) for seed in seeds)} | "
                f"{mean(losses):.4f} | {std(losses):.4f} | {delta} | "
                f"{mean(ppls):.2f} | {mean(elapsed):.2f} | "
                f"{int(mean(params)) if params else '-'} |"
            )

    d192_scale05_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_residual_gate_matches(row, False)
        and mogt_lr_mult_matches(row)
        and (
            (
                single_mogt_layer_position(row["model_config"].get("block_types") or [])
                == "attention-only"
                and mogt_scale_matches(row)
            )
            or (
                single_mogt_layer_position(row["model_config"].get("block_types") or [])
                == "layer 2"
                and mogt_scale_matches(row, residual_scale=0.5, ffn_residual_scale=1.0)
            )
        )
    ]
    grouped_d192_scale05: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in d192_scale05_rows:
        position = single_mogt_layer_position(row["model_config"].get("block_types") or [])
        if position is not None:
            grouped_d192_scale05[position].append(row)
    if {"attention-only", "layer 2"}.issubset(grouped_d192_scale05):
        attention_losses = [
            float(row["loss"])
            for row in grouped_d192_scale05.get("attention-only", [])
            if row.get("loss") is not None
        ]
        attention_mean = mean(attention_losses) if attention_losses else None
        lines.extend(
            [
                "",
                "### d_model=192 Residual-Scale 0.5 Confirmation",
                "",
                "Context 8192, four layers, 1000 optimizer steps, lr=5e-4,",
                "and eight validation batches. This compares the tuned",
                "attention-only control against the fixed residual-scale 0.5",
                "layer-2 hybrid on the same three seeds.",
                "",
                "| Position | Residual scale | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for position, rows_for_position in sorted(
            grouped_d192_scale05.items(), key=lambda item: position_sort_key(item[0])
        ):
            losses = [float(row["loss"]) for row in rows_for_position if row.get("loss") is not None]
            ppls = [float(row["ppl"]) for row in rows_for_position if row.get("ppl") is not None]
            elapsed = [
                float(row["elapsed_seconds"])
                for row in rows_for_position
                if row.get("elapsed_seconds") is not None
            ]
            params = [
                int(row["model_config"].get("num_params"))
                for row in rows_for_position
                if row["model_config"].get("num_params") is not None
            ]
            seeds = sorted({int(row["training_seed"]) for row in rows_for_position})
            residual_scale = (
                1.0
                if position == "attention-only"
                else float(rows_for_position[0]["model_config"].get("mogt_residual_scale", 1.0))
            )
            delta = "-"
            if attention_mean is not None and losses:
                delta = f"{mean(losses) - attention_mean:.4f}"
            lines.append(
                f"| {position} | {residual_scale:.2f} | "
                f"{','.join(str(seed) for seed in seeds)} | "
                f"{mean(losses):.4f} | {std(losses):.4f} | {delta} | "
                f"{mean(ppls):.2f} | {mean(elapsed):.2f} | "
                f"{int(mean(params)) if params else '-'} |"
            )
        attention_by_seed = {
            int(row["training_seed"]): float(row["loss"])
            for row in grouped_d192_scale05["attention-only"]
            if row.get("loss") is not None and row.get("training_seed") is not None
        }
        layer2_by_seed = {
            int(row["training_seed"]): float(row["loss"])
            for row in grouped_d192_scale05["layer 2"]
            if row.get("loss") is not None and row.get("training_seed") is not None
        }
        paired_seeds = sorted(set(attention_by_seed) & set(layer2_by_seed))
        wins = [seed for seed in paired_seeds if layer2_by_seed[seed] < attention_by_seed[seed]]
        if paired_seeds:
            lines.extend(
                [
                    "",
                    "Paired-seed diagnostic: fixed scale 0.5 beats attention on "
                    f"{len(wins)}/{len(paired_seeds)} seeds"
                    + (f" ({','.join(str(seed) for seed in wins)})" if wins else "")
                    + ", but its aggregate mean is still the claim boundary.",
                ]
            )

    d192_mogt_lr05_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_residual_gate_matches(row, False)
        and row["model_config"].get("mogt_residual_scale_schedule", "constant") == "constant"
        and (
            (
                single_mogt_layer_position(row["model_config"].get("block_types") or [])
                == "attention-only"
                and mogt_scale_matches(row)
                and mogt_lr_mult_matches(row)
            )
            or (
                single_mogt_layer_position(row["model_config"].get("block_types") or [])
                == "layer 2"
                and mogt_scale_matches(row, residual_scale=0.5, ffn_residual_scale=1.0)
                and mogt_lr_mult_matches(row, 0.5)
            )
        )
    ]
    grouped_d192_mogt_lr05: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in d192_mogt_lr05_rows:
        position = single_mogt_layer_position(row["model_config"].get("block_types") or [])
        if position is not None:
            grouped_d192_mogt_lr05[position].append(row)
    if {"attention-only", "layer 2"}.issubset(grouped_d192_mogt_lr05):
        attention_losses = [
            float(row["loss"])
            for row in grouped_d192_mogt_lr05.get("attention-only", [])
            if row.get("loss") is not None
        ]
        attention_mean = mean(attention_losses) if attention_losses else None
        lines.extend(
            [
                "",
                "### d_model=192 MOGT LR Multiplier 0.5 Confirmation",
                "",
                "Context 8192, four layers, 1000 optimizer steps, lr=5e-4,",
                "fixed residual scale 0.5, and eight validation batches. This",
                "compares tuned attention-only against the layer-2 hybrid with",
                "a 0.5 learning-rate multiplier on MOGT block parameters.",
                "",
                "| Position | Residual scale | MOGT LR mult | Seeds | Mean val loss | Std | Delta vs attention-only | Mean PPL | Mean elapsed s | Params |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for position, rows_for_position in sorted(
            grouped_d192_mogt_lr05.items(), key=lambda item: position_sort_key(item[0])
        ):
            losses = [float(row["loss"]) for row in rows_for_position if row.get("loss") is not None]
            ppls = [float(row["ppl"]) for row in rows_for_position if row.get("ppl") is not None]
            elapsed = [
                float(row["elapsed_seconds"])
                for row in rows_for_position
                if row.get("elapsed_seconds") is not None
            ]
            params = [
                int(row["model_config"].get("num_params"))
                for row in rows_for_position
                if row["model_config"].get("num_params") is not None
            ]
            seeds = sorted({int(row["training_seed"]) for row in rows_for_position})
            residual_scale = (
                1.0
                if position == "attention-only"
                else float(rows_for_position[0]["model_config"].get("mogt_residual_scale", 1.0))
            )
            mogt_lr_mult = (
                1.0
                if position == "attention-only"
                else float(rows_for_position[0]["model_config"].get("mogt_lr_mult", 1.0))
            )
            delta = "-"
            if attention_mean is not None and losses:
                delta = f"{mean(losses) - attention_mean:.4f}"
            lines.append(
                f"| {position} | {residual_scale:.2f} | {mogt_lr_mult:.2f} | "
                f"{','.join(str(seed) for seed in seeds)} | "
                f"{mean(losses):.4f} | {std(losses):.4f} | {delta} | "
                f"{mean(ppls):.2f} | {mean(elapsed):.2f} | "
                f"{int(mean(params)) if params else '-'} |"
            )
        attention_by_seed = {
            int(row["training_seed"]): float(row["loss"])
            for row in grouped_d192_mogt_lr05["attention-only"]
            if row.get("loss") is not None and row.get("training_seed") is not None
        }
        layer2_by_seed = {
            int(row["training_seed"]): float(row["loss"])
            for row in grouped_d192_mogt_lr05["layer 2"]
            if row.get("loss") is not None and row.get("training_seed") is not None
        }
        paired_seeds = sorted(set(attention_by_seed) & set(layer2_by_seed))
        wins = [seed for seed in paired_seeds if layer2_by_seed[seed] < attention_by_seed[seed]]
        if paired_seeds:
            lines.extend(
                [
                    "",
                    "Paired-seed diagnostic: layer-2 MOGT with residual scale 0.5 "
                    f"and MOGT LR multiplier 0.5 beats attention on {len(wins)}/{len(paired_seeds)} "
                    "paired seeds"
                    + (f" ({','.join(str(seed) for seed in wins)})" if wins else "")
                    + ".",
                ]
            )

    d256_seed = 42
    d256_width_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=256,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_residual_gate_matches(row, False)
        and row["model_config"].get("mogt_residual_scale_schedule", "constant") == "constant"
        and int(row.get("training_seed") or -1) == d256_seed
        and (
            (
                single_mogt_layer_position(row["model_config"].get("block_types") or [])
                == "attention-only"
                and mogt_scale_matches(row)
                and mogt_lr_mult_matches(row)
            )
            or (
                single_mogt_layer_position(row["model_config"].get("block_types") or [])
                == "layer 2"
                and mogt_scale_matches(row, residual_scale=0.5, ffn_residual_scale=1.0)
                and mogt_lr_mult_matches(row, 0.5)
            )
        )
    ]
    if d256_width_rows:
        attention_rows = [
            row
            for row in d256_width_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "attention-only"
        ]
        attention_loss = float(attention_rows[0]["loss"]) if attention_rows else None
        lines.extend(
            [
                "",
                "### d_model=256 Width Migration Diagnostic (seed 42)",
                "",
                "Context 8192, four layers, 1000 optimizer steps, lr=5e-4,",
                "and eight validation batches. This tests whether the d_model=192",
                "MOGT LR multiplier recipe immediately transfers to a wider model.",
                "",
                "| Position | Residual scale | MOGT LR mult | Val loss | Delta vs attention seed42 | PPL | Params | Run |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(
            d256_width_rows,
            key=lambda item: position_sort_key(single_mogt_layer_position(item["model_config"].get("block_types") or []) or ""),
        ):
            config = row["model_config"]
            position = single_mogt_layer_position(config.get("block_types") or []) or "-"
            residual_scale = float(config.get("mogt_residual_scale", 1.0))
            mogt_lr_mult = float(config.get("mogt_lr_mult", 1.0))
            loss = float(row["loss"])
            delta = "-"
            if attention_loss is not None:
                delta = f"{loss - attention_loss:.4f}"
            params = config.get("num_params")
            lines.append(
                f"| {position} | {residual_scale:.2f} | {mogt_lr_mult:.2f} | "
                f"{loss:.4f} | {delta} | {float(row['ppl']):.2f} | "
                f"{int(params) if params is not None else '-'} | `{row['run_name']}` |"
            )

    d192_scale_seed = 42
    d192_scale_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_residual_gate_matches(row, False)
        and mogt_lr_mult_matches(row)
        and int(row.get("training_seed") or -1) == d192_scale_seed
    ]
    d192_default_seed_rows = [row for row in d192_scale_rows if mogt_scale_matches(row)]
    d192_scaled_seed_rows = [
        row
        for row in d192_scale_rows
        if row["model_config"].get("mogt_residual_scale_schedule", "constant") == "constant"
        and not mogt_scale_matches(row)
        and abs(float(row["model_config"].get("mogt_ffn_residual_scale", 1.0)) - 1.0) <= 1e-12
    ]
    if d192_scaled_seed_rows:
        seed_attention = [
            row
            for row in d192_default_seed_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "attention-only"
        ]
        seed_attention_loss = float(seed_attention[0]["loss"]) if seed_attention else None
        diagnostic_rows = [
            row
            for row in d192_default_seed_rows + d192_scaled_seed_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            in {"attention-only", "layer 2"}
        ]
        lines.extend(
            [
                "",
                "### d_model=192 Residual-Scale Sweep (seed 42)",
                "",
                "Single-seed diagnostic at context 8192, four layers, 1000 optimizer",
                "steps, lr=5e-4, and eight validation batches. This table is not",
                "mixed into the aggregate above.",
                "",
                "| Position | Residual scale | Val loss | Delta vs attention seed42 | PPL | Run |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(
            diagnostic_rows,
            key=lambda item: (
                position_sort_key(single_mogt_layer_position(item["model_config"].get("block_types") or []) or ""),
                float(item["model_config"].get("mogt_residual_scale", 1.0)),
            ),
        ):
            config = row["model_config"]
            position = single_mogt_layer_position(config.get("block_types") or []) or "-"
            residual_scale = float(config.get("mogt_residual_scale", 1.0))
            loss = float(row["loss"])
            delta = "-"
            if seed_attention_loss is not None:
                delta = f"{loss - seed_attention_loss:.4f}"
            lines.append(
                f"| {position} | {residual_scale:.2f} | {loss:.4f} | "
                f"{delta} | {float(row['ppl']):.2f} | `{row['run_name']}` |"
            )

    d192_schedule_seed_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_lr_mult_matches(row)
        and int(row.get("training_seed") or -1) == d192_scale_seed
    ]
    d192_schedule_rows = [
        row
        for row in d192_schedule_seed_rows
        if row["model_config"].get("mogt_residual_scale_schedule", "constant") != "constant"
        and mogt_residual_gate_matches(row, False)
    ]
    if d192_schedule_rows:
        seed_attention = [
            row
            for row in d192_schedule_seed_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "attention-only"
            and mogt_scale_matches(row)
            and mogt_residual_gate_matches(row, False)
        ]
        seed_attention_loss = float(seed_attention[0]["loss"]) if seed_attention else None
        fixed_scale_05 = [
            row
            for row in d192_schedule_seed_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "layer 2"
            and mogt_scale_matches(row, residual_scale=0.5, ffn_residual_scale=1.0)
            and mogt_residual_gate_matches(row, False)
        ]
        diagnostic_rows = [*seed_attention, *fixed_scale_05, *d192_schedule_rows]
        lines.extend(
            [
                "",
                "### d_model=192 Residual-Scale Schedule Diagnostic (seed 42)",
                "",
                "Single-seed diagnostic at context 8192, four layers, 1000 optimizer",
                "steps, lr=5e-4, and eight validation batches. Scheduled runs are",
                "reported separately from fixed-scale aggregates.",
                "",
                "| Position | Schedule | Scale path | Val loss | Delta vs attention seed42 | PPL | Run |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for row in sorted(
            diagnostic_rows,
            key=lambda item: (
                position_sort_key(single_mogt_layer_position(item["model_config"].get("block_types") or []) or ""),
                item["model_config"].get("mogt_residual_scale_schedule", "constant"),
                float(item["model_config"].get("mogt_residual_scale", 1.0)),
            ),
        ):
            config = row["model_config"]
            position = single_mogt_layer_position(config.get("block_types") or []) or "-"
            schedule = config.get("mogt_residual_scale_schedule", "constant")
            if schedule == "linear_warmup":
                scale_path = (
                    f"{float(config.get('mogt_residual_scale_start', 0.0)):.2f} -> "
                    f"{float(config.get('mogt_residual_scale', 1.0)):.2f} / "
                    f"{int(config.get('mogt_residual_scale_warmup_steps', 0))} steps"
                )
            else:
                scale_path = f"{float(config.get('mogt_residual_scale', 1.0)):.2f}"
            loss = float(row["loss"])
            delta = "-"
            if seed_attention_loss is not None:
                delta = f"{loss - seed_attention_loss:.4f}"
            lines.append(
                f"| {position} | {schedule} | {scale_path} | {loss:.4f} | "
                f"{delta} | {float(row['ppl']):.2f} | `{row['run_name']}` |"
            )

    d192_mogt_lr_seed_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and int(row.get("training_seed") or -1) == d192_scale_seed
        and row["model_config"].get("mogt_residual_scale_schedule", "constant") == "constant"
        and mogt_residual_gate_matches(row, False)
    ]
    d192_mogt_lr_rows = [
        row
        for row in d192_mogt_lr_seed_rows
        if not mogt_lr_mult_matches(row)
    ]
    if d192_mogt_lr_rows:
        seed_attention = [
            row
            for row in d192_mogt_lr_seed_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "attention-only"
            and mogt_scale_matches(row)
            and mogt_lr_mult_matches(row)
        ]
        seed_attention_loss = float(seed_attention[0]["loss"]) if seed_attention else None
        fixed_scale_05 = [
            row
            for row in d192_mogt_lr_seed_rows
            if single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "layer 2"
            and mogt_scale_matches(row, residual_scale=0.5, ffn_residual_scale=1.0)
            and mogt_lr_mult_matches(row)
        ]
        diagnostic_rows = [*seed_attention, *fixed_scale_05, *d192_mogt_lr_rows]
        lines.extend(
            [
                "",
                "### d_model=192 MOGT Learning-Rate Multiplier Diagnostic (seed 42)",
                "",
                "Single-seed diagnostic at context 8192, four layers, 1000 optimizer",
                "steps, lr=5e-4, fixed residual scale 0.5, and eight validation",
                "batches. Non-default MOGT optimizer groups are reported separately",
                "from the fixed-scale aggregate.",
                "",
                "| Position | Residual scale | MOGT LR mult | Val loss | Delta vs attention seed42 | PPL | Run |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(
            diagnostic_rows,
            key=lambda item: (
                position_sort_key(single_mogt_layer_position(item["model_config"].get("block_types") or []) or ""),
                float(item["model_config"].get("mogt_lr_mult", 1.0)),
            ),
        ):
            config = row["model_config"]
            position = single_mogt_layer_position(config.get("block_types") or []) or "-"
            residual_scale = float(config.get("mogt_residual_scale", 1.0))
            mogt_lr_mult = float(config.get("mogt_lr_mult", 1.0))
            loss = float(row["loss"])
            delta = "-"
            if seed_attention_loss is not None:
                delta = f"{loss - seed_attention_loss:.4f}"
            lines.append(
                f"| {position} | {residual_scale:.2f} | {mogt_lr_mult:.2f} | "
                f"{loss:.4f} | {delta} | {float(row['ppl']):.2f} | `{row['run_name']}` |"
            )

    d192_gate_seed_rows = [
        row
        for row in lm_pilot_rows(
            reports,
            train_context=8192,
            steps=1000,
            d_model=192,
            num_layers=4,
        )
        if row["model_config"].get("zero_init_attention_out")
        and lr_matches(row, 0.0005)
        and mogt_lr_mult_matches(row)
        and int(row.get("training_seed") or -1) == d192_scale_seed
    ]
    d192_gate_rows = [
        row
        for row in d192_gate_seed_rows
        if mogt_residual_gate_matches(row, True)
    ]
    if d192_gate_rows:
        seed_attention = [
            row
            for row in d192_gate_seed_rows
            if mogt_scale_matches(row)
            and mogt_residual_gate_matches(row, False)
            and single_mogt_layer_position(row["model_config"].get("block_types") or [])
            == "attention-only"
        ]
        seed_attention_loss = float(seed_attention[0]["loss"]) if seed_attention else None
        fixed_scale_05 = [
            row
            for row in d192_gate_seed_rows
            if mogt_scale_matches(row, residual_scale=0.5, ffn_residual_scale=1.0)
            and mogt_residual_gate_matches(row, False)
        ]
        diagnostic_rows = [*fixed_scale_05, *d192_gate_rows]
        lines.extend(
            [
                "",
                "### d_model=192 Learned Residual-Gate Diagnostic (seed 42)",
                "",
                "Single-seed diagnostic at context 8192, four layers, 1000 optimizer",
                "steps, lr=5e-4, and eight validation batches. The learned gate is",
                "reported separately because it changes the optimization dynamics.",
                "",
                "| Position | Gate | Init / fixed scale | Val loss | Delta vs attention seed42 | PPL | Run |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(
            diagnostic_rows,
            key=lambda item: (
                not mogt_residual_gate_matches(item, False),
                float(item["model_config"].get("mogt_residual_scale", 1.0)),
            ),
        ):
            config = row["model_config"]
            position = single_mogt_layer_position(config.get("block_types") or []) or "-"
            has_gate = bool(config.get("mogt_residual_gate", False))
            gate_label = "learned" if has_gate else "fixed"
            init_or_scale = (
                float(config.get("mogt_residual_gate_init", 0.0))
                if has_gate
                else float(config.get("mogt_residual_scale", 1.0))
            )
            loss = float(row["loss"])
            delta = "-"
            if seed_attention_loss is not None:
                delta = f"{loss - seed_attention_loss:.4f}"
            lines.append(
                f"| {position} | {gate_label} | {init_or_scale:.2f} | "
                f"{loss:.4f} | {delta} | {float(row['ppl']):.2f} | `{row['run_name']}` |"
            )

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Safe current wording:",
            "",
            "> Coupling token-dependent writing with forgetting gives a scan-compatible",
            "> matrix-valued recurrent operator that solves overwrite state tracking.",
            "> Adding prefix-conditioned slot addressing extends this mechanism to tracked",
            "> multi-slot routing under controlled synthetic settings.",
            "",
            "Unsafe current wording:",
            "",
            "> MOGT generally beats Transformer, solves language modeling, or replaces",
            "> attention across tasks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    write_snapshot(Path("paper/results_snapshot.md"), load_standard_reports())
    print("Wrote paper/results_snapshot.md")


if __name__ == "__main__":
    main()
