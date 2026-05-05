import json
import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "mogt-experiment-v1"


def command_line() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def torch_environment(
    device: torch.device | str,
    *,
    amp_dtype: str | None = None,
    peak_memory_mb: float | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "amp_dtype": amp_dtype,
        "peak_memory_mb": peak_memory_mb,
        "elapsed_seconds": elapsed_seconds,
    }


def _args_dict(args: Namespace | dict[str, Any]) -> dict[str, Any]:
    if isinstance(args, Namespace):
        return vars(args)
    return dict(args)


def _sum_examples(eval_results: list[dict[str, Any]]) -> int | None:
    if not eval_results:
        return None
    total = 0
    found = False
    for result in eval_results:
        if result.get("examples") is not None:
            total += int(result["examples"])
            found = True
    return total if found else None


def _by_context(eval_results: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = {}
    for result in eval_results:
        context = result.get("context_length")
        value = result.get(key)
        if context is not None and value is not None:
            values[str(int(context))] = float(value)
    return values


def _synthetic_task_suffixes(config: dict[str, Any]) -> list[str]:
    parts = []
    if config.get("num_slots") is not None:
        parts.append(f"slots{config.get('num_slots')}")
    if config.get("dense_loss"):
        parts.append("dense")
    elif config.get("num_slots") is not None:
        parts.append("final")
    if config.get("curriculum_steps", 0) or config.get("slot_curriculum_steps", 0):
        parts.append("curriculum")
    return parts


def synthetic_variant(args: Namespace | dict[str, Any]) -> str:
    config = _args_dict(args)
    model_type = config.get("model_type", "")
    if model_type == "transformer":
        rope_theta = float(config.get("rope_theta", 10000.0))
        if rope_theta <= 0:
            base = "transformer_nope"
        elif rope_theta != 10000.0:
            base = f"transformer_rope_theta_{rope_theta:g}"
        else:
            base = "transformer_rope"
        suffixes = _synthetic_task_suffixes(config)
        return "_".join([base, *suffixes])
    if model_type == "mogt":
        parts = ["mogt"]
        connection_impl = config.get("connection_impl")
        if connection_impl:
            parts.append(str(connection_impl))
        if config.get("couple_forget_to_value_gate"):
            parts.append("coupled_write_forget")
        elif config.get("transport_gate") and config.get("value_gate"):
            parts.append("dual_gate")
        elif config.get("transport_gate"):
            parts.append("transport_gate")
        elif config.get("value_gate"):
            parts.append("value_gate")
        else:
            parts.append("ungated")
        gate_input = config.get("value_gate_input")
        if gate_input and gate_input != "current":
            parts.append(str(gate_input))
        value_gate_bias = config.get("value_gate_bias")
        if value_gate_bias not in (None, 0, 0.0):
            bias_text = str(value_gate_bias).replace("-", "m").replace(".", "p")
            parts.append(f"value_bias_{bias_text}")
        parts.extend(_synthetic_task_suffixes(config))
        return "_".join(parts)
    suffixes = _synthetic_task_suffixes(config)
    if suffixes:
        base = str(model_type or "unknown")
        if model_type in {"mamba", "gru"} and config.get("d_model") is not None:
            base = f"{base}_d{config.get('d_model')}"
        return "_".join([base, *suffixes])
    return str(model_type or "unknown")


def synthetic_gate_config(args: Namespace | dict[str, Any]) -> dict[str, Any]:
    config = _args_dict(args)
    return {
        "transport_gate": bool(config.get("transport_gate", False)),
        "transport_gate_mode": config.get("transport_gate_mode"),
        "transport_gate_width": config.get("transport_gate_width"),
        "transport_gate_scale": config.get("transport_gate_scale"),
        "transport_gate_bias": config.get("transport_gate_bias"),
        "value_gate": bool(config.get("value_gate", False)),
        "value_gate_width": config.get("value_gate_width"),
        "value_gate_input": config.get("value_gate_input"),
        "value_gate_bias": config.get("value_gate_bias"),
        "couple_forget_to_value_gate": bool(config.get("couple_forget_to_value_gate", False)),
        "prefix_condition_position": config.get("prefix_condition_position"),
    }


def build_synthetic_standard_report(
    *,
    task: str,
    args: Namespace | dict[str, Any],
    parameter_count: int | None,
    eval_results: list[dict[str, Any]],
    device: torch.device | str,
    elapsed_seconds: float | None,
    peak_memory_mb: float | None,
    status: str = "ok",
    failure_reason: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    config = _args_dict(args)
    eval_contexts = [int(value) for value in config.get("eval_contexts", [])]
    train_context = config.get("train_context")
    eval_examples = _sum_examples(eval_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": str(config.get("run_name") or config.get("output") or ""),
        "task": task,
        "model": str(config.get("model_type", "")),
        "variant": synthetic_variant(config),
        "status": status,
        "failure_reason": failure_reason,
        "command": command_line(),
        "git_commit": git_commit(),
        "environment": torch_environment(
            device,
            amp_dtype=config.get("dtype"),
            peak_memory_mb=peak_memory_mb,
            elapsed_seconds=elapsed_seconds,
        ),
        "data": {
            "dataset": "synthetic",
            "tokenizer": None,
            "train_context": int(train_context) if train_context is not None else None,
            "eval_contexts": eval_contexts,
            "train_examples": None,
            "eval_examples": eval_examples,
            "tokens_seen": None,
            "num_values": config.get("num_values"),
            "min_updates": config.get("min_updates"),
            "max_updates": config.get("max_updates"),
            "num_slots": config.get("num_slots"),
            "min_train_slots": config.get("min_train_slots"),
        },
        "model_config": {
            "num_params": parameter_count,
            "d_model": config.get("d_model"),
            "num_layers": config.get("num_layers"),
            "rank": config.get("rank"),
            "num_heads": config.get("num_heads"),
            "scan_impl": config.get("scan_impl"),
            "connection_impl": config.get("connection_impl"),
            "gate_config": synthetic_gate_config(config),
        },
        "training": {
            "seed": config.get("seed"),
            "steps": config.get("steps"),
            "batch_size": config.get("batch_size"),
            "grad_accum_steps": None,
            "optimizer": "AdamW",
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),
            "scheduler": None,
            "dense_loss": bool(config.get("dense_loss", False)),
            "curriculum_steps": config.get("curriculum_steps"),
            "slot_curriculum_steps": config.get("slot_curriculum_steps"),
            "state_label_mode": config.get("state_label_mode"),
            "dense_loss_steps": config.get("dense_loss_steps"),
        },
        "metrics": {
            "loss": None,
            "ppl": None,
            "accuracy_by_context": _by_context(eval_results, "accuracy"),
            "loss_by_context": _by_context(eval_results, "loss"),
        },
        "notes": notes,
    }


def build_lm_standard_report(
    *,
    task: str,
    config: dict[str, Any],
    result: dict[str, Any],
    device: torch.device | str,
    validation_trace: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    status = result.get("status", "ok")
    best_val = result.get("best_val") or {}
    context_length = config.get("context_length")
    tokens_seen = None
    if all(
        config.get(key) is not None
        for key in ("max_steps", "batch_size", "grad_accum_steps", "context_length")
    ):
        tokens_seen = (
            int(config["max_steps"])
            * int(config["batch_size"])
            * int(config["grad_accum_steps"])
            * int(config["context_length"])
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": str(config.get("run_name") or ""),
        "task": task,
        "model": str(config.get("model_type", "")),
        "variant": str(config.get("variant") or config.get("model_type", "")),
        "status": status,
        "failure_reason": result.get("error") or result.get("failed_phase"),
        "command": command_line(),
        "git_commit": git_commit(),
        "environment": torch_environment(
            device,
            amp_dtype=config.get("amp_dtype"),
            peak_memory_mb=result.get("peak_memory_mb"),
            elapsed_seconds=result.get("elapsed_s"),
        ),
        "data": {
            "dataset": "wikitext-103-raw-v1",
            "tokenizer": "gpt2",
            "train_context": int(context_length) if context_length is not None else None,
            "eval_contexts": [int(context_length)] if context_length is not None else [],
            "train_examples": None,
            "eval_examples": None,
            "tokens_seen": tokens_seen,
            "eval_max_batches_requested": config.get("eval_max_batches_requested"),
            "loss_chunk_size": config.get("loss_chunk_size"),
        },
        "model_config": {
            "num_params": config.get("parameter_count"),
            "d_model": config.get("d_model"),
            "num_layers": config.get("num_layers"),
            "rank": config.get("rank"),
            "num_heads": config.get("num_heads"),
            "scan_impl": config.get("scan_impl"),
            "connection_impl": config.get("connection_impl"),
            "gate_config": None,
            "attention": config.get("attention"),
            "gradient_checkpointing": config.get("gradient_checkpointing"),
            "hybrid_pattern": config.get("hybrid_pattern"),
            "mogt_layer_fraction": config.get("mogt_layer_fraction"),
            "mogt_layer_count": config.get("mogt_layer_count"),
            "mogt_layer_indices_requested": config.get("mogt_layer_indices_requested"),
            "mogt_layer_indices": config.get("mogt_layer_indices"),
            "block_types": config.get("block_types"),
            "mogt_residual_scale": config.get("mogt_residual_scale"),
            "mogt_residual_scale_start": config.get("mogt_residual_scale_start"),
            "mogt_residual_scale_warmup_steps": config.get("mogt_residual_scale_warmup_steps"),
            "mogt_residual_scale_schedule": config.get("mogt_residual_scale_schedule"),
            "mogt_ffn_residual_scale": config.get("mogt_ffn_residual_scale"),
            "mogt_residual_gate": config.get("mogt_residual_gate"),
            "mogt_residual_gate_init": config.get("mogt_residual_gate_init"),
            "mogt_lr_mult": config.get("mogt_lr_mult"),
            "zero_init_attention_out": config.get("zero_init_attention_out"),
        },
        "training": {
            "seed": config.get("seed"),
            "steps": config.get("max_steps"),
            "batch_size": config.get("batch_size"),
            "grad_accum_steps": config.get("grad_accum_steps"),
            "optimizer": "AdamW",
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),
            "scheduler": "cosine_with_warmup",
        },
        "metrics": {
            "loss": best_val.get("loss"),
            "ppl": best_val.get("ppl"),
            "accuracy_by_context": {},
            "loss_by_context": (
                {str(int(context_length)): float(best_val["loss"])}
                if context_length is not None and best_val.get("loss") is not None
                else {}
            ),
            "validation_trace": validation_trace or [],
            "train_loss_final": result.get("train_loss_final"),
        },
        "notes": notes,
    }


def validate_standard_report(report: dict[str, Any]) -> list[str]:
    errors = []
    required = [
        "schema_version",
        "run_name",
        "task",
        "model",
        "variant",
        "status",
        "command",
        "environment",
        "data",
        "model_config",
        "training",
        "metrics",
    ]
    for key in required:
        if key not in report:
            errors.append(f"missing top-level key: {key}")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if report.get("status") not in {"ok", "failed", "oom", "skipped", "partial"}:
        errors.append("status must be one of ok/failed/oom/skipped/partial")
    return errors


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
