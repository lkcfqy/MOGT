import json
from pathlib import Path


def _series(results_ms: dict, key: str) -> dict[int, float]:
    return {int(length): float(value) for length, value in results_ms.get(key, [])}


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _speedup(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return "-"
    return f"{numerator / denominator:.2f}x"


def write_summary(input_json: Path, output_md: Path) -> None:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    results = payload.get("results_ms", {})
    affine = _series(results, "affine_triton_hybrid_ms")
    attention = _series(results, "attention_ms")
    transport_only = _series(results, "transport_only_triton_ms")
    parallel_ref = _series(results, "affine_parallel_ref_ms")
    lengths = sorted({*affine.keys(), *attention.keys(), *transport_only.keys()})

    lines = [
        "# Core Operator Throughput Summary",
        "",
        f"Source: `{input_json}`",
        "",
        "This is a core-operator timing summary, not an end-to-end model throughput",
        "claim. Affine scan timings exclude connection/value projection, matrix",
        "exponential/Cayley construction, normalization, FFN, and LM head. Attention",
        "timings measure FlashAttention/SDPA core only.",
        "",
        f"Device: `{payload.get('gpu_name') or payload.get('device')}`. "
        f"Batch size: {payload.get('batch_size')}. "
        f"`d_model={payload.get('d_model')}`, `rank={payload.get('rank')}`. "
        f"Warmup/iters: {payload.get('warmup')}/{payload.get('iters')}.",
        "",
        "| Length | Affine Triton hybrid ms | Attention core ms | Attention / affine | Transport-only ms | Parallel ref ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for length in lengths:
        affine_ms = affine.get(length)
        attention_ms = attention.get(length)
        lines.append(
            f"| {length} | {_fmt(affine_ms)} | {_fmt(attention_ms)} | "
            f"{_speedup(attention_ms, affine_ms)} | {_fmt(transport_only.get(length))} | "
            f"{_fmt(parallel_ref.get(length))} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- At 8k, attention core remains faster in this measurement.",
            "- At 16k and 32k, the affine scan core is faster than attention core.",
            "- This supports a systems hypothesis, not a full-model speed claim.",
            "- The next systems step is to profile full MOGT blocks and fused/near-fused",
            "  connection + scan + readout paths.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_json = Path(
        "benchmark_runs/throughput_core_operator_d768_len8192_16384_32768_20260504.json"
    )
    output_md = Path("benchmark_runs/throughput_core_operator_summary_20260504.md")
    write_summary(input_json, output_md)
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
