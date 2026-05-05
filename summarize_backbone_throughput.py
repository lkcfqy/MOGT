import json
from pathlib import Path


def _record_by_model(payload):
    return {record["model"]: record for record in payload.get("records", [])}


def _rows(record):
    return {int(row["length"]): row for row in record.get("results", [])}


def _fmt(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def write_summary(input_json: Path, output_md: Path) -> None:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    by_model = _record_by_model(payload)
    mogt = _rows(by_model.get("mogt", {}))
    transformer = _rows(by_model.get("transformer", {}))
    lengths = sorted({*mogt, *transformer})

    lines = [
        "# Backbone Throughput Summary",
        "",
        f"Source: `{input_json}`",
        "",
        "Backbone hidden-state forward only: embeddings, sequence blocks, and final",
        "normalization. This excludes LM head, loss, backward pass, optimizer, and",
        "KV-cache decode behavior.",
        "",
        f"Device: `{payload.get('gpu_name') or payload.get('device')}`. "
        f"`d_model={payload.get('config', {}).get('d_model')}`, "
        f"`num_layers={payload.get('config', {}).get('num_layers')}`, "
        f"batch size {payload.get('config', {}).get('batch_size')}.",
        "",
        "| Length | MOGT ms | Transformer NoPE ms | Transformer / MOGT | MOGT peak MB | Transformer peak MB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for length in lengths:
        mogt_row = mogt.get(length, {})
        transformer_row = transformer.get(length, {})
        mogt_ms = mogt_row.get("elapsed_ms")
        transformer_ms = transformer_row.get("elapsed_ms")
        ratio = "-"
        if mogt_ms and transformer_ms:
            ratio = f"{transformer_ms / mogt_ms:.2f}x"
        lines.append(
            f"| {length} | {_fmt(mogt_ms)} | {_fmt(transformer_ms)} | {ratio} | "
            f"{_fmt(mogt_row.get('peak_memory_mb'))} | {_fmt(transformer_row.get('peak_memory_mb'))} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- At 8k and 16k, the small NoPE Transformer backbone remains competitive.",
            "- At 32k, the identity coupled MOGT backbone is faster in this measurement.",
            "- This is still not an end-to-end training or generation throughput claim.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_json = Path(
        "benchmark_runs/backbone_throughput_identity_coupled_d768_l2_len8192_16384_32768_20260504.json"
    )
    output_md = Path("benchmark_runs/backbone_throughput_summary_20260504.md")
    write_summary(input_json, output_md)
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
