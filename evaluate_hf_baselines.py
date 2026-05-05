import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace CausalLM baselines on WikiText-103.")
    parser.add_argument("--models", nargs="+", default=["gpt2"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[1023])
    parser.add_argument("--tokenization-mode", choices=["gpt2_stream", "native_text"], default="gpt2_stream")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    parser.add_argument("--output", default="benchmark_runs/hf_baseline_eval.json")
    return parser.parse_args()


def resolve_dtype(device, requested: str):
    if requested == "float32" or device.type != "cuda":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def model_context_limit(config, tokenizer):
    candidates = [
        getattr(config, "max_position_embeddings", None),
        getattr(config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    valid = [int(value) for value in candidates if isinstance(value, int) and 0 < value < 10**9]
    return min(valid) if valid else None


def load_hf_model(model_name: str, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer, config


def autocast_context(device, dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def build_validation_tokens(tokenizer):
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(item for item in raw["text"] if item.strip())
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False, verbose=False)
    return encoded.input_ids[0]


def evaluate_token_stream(model, tokens, device, dtype, *, context_length: int, max_batches: int):
    seq_len = context_length + 1
    available_batches = tokens.numel() // seq_len
    num_batches = min(max_batches, available_batches)
    if num_batches <= 0:
        return None
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * seq_len
            full = tokens[start : start + seq_len].unsqueeze(0).to(device)
            with autocast_context(device, dtype):
                outputs = model(input_ids=full, labels=full)
            loss_tokens = full.numel() - 1
            total_nll += float(outputs.loss.item()) * loss_tokens
            total_tokens += loss_tokens

    if total_tokens == 0:
        return None
    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "num_batches": num_batches,
        "num_loss_tokens": total_tokens,
    }


def evaluate_native_text(model, tokenizer, device, dtype, *, context_length: int, max_batches: int):
    tokens = build_validation_tokens(tokenizer)
    return evaluate_token_stream(
        model,
        tokens,
        device,
        dtype,
        context_length=context_length,
        max_batches=max_batches,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = resolve_dtype(device, args.dtype)

    report = {
        "config": {
            "models": args.models,
            "context_lengths": args.context_lengths,
            "tokenization_mode": args.tokenization_mode,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_batches": args.max_batches,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
        },
        "interpretation": {
            "gpt2_stream": "Uses this repo's GPT-2 tokenized WikiText-103 chunks and is tokenization-compatible with MOGT.",
            "native_text": "Uses each HF model's own tokenizer on WikiText-103 raw validation text; useful as a pretrained anchor, not a tokenization-matched comparison.",
        },
        "results": [],
    }

    for model_name in args.models:
        model = None
        try:
            model, tokenizer, config = load_hf_model(model_name, device, dtype)
            limit = model_context_limit(config, tokenizer)
            tokenizer_matches_gpt2 = len(tokenizer) == 50257 and getattr(config, "vocab_size", None) == 50257
            validation_tokens = None

            for context_length in args.context_lengths:
                full_len = context_length + 1
                if limit is not None and full_len > limit:
                    report["results"].append({
                        "model": model_name,
                        "context_length": context_length,
                        "status": "skipped",
                        "reason": f"full sequence length {full_len} exceeds model/tokenizer limit {limit}",
                    })
                    continue

                if args.tokenization_mode == "gpt2_stream":
                    if not tokenizer_matches_gpt2 and not args.allow_tokenizer_mismatch:
                        report["results"].append({
                            "model": model_name,
                            "context_length": context_length,
                            "status": "skipped",
                            "reason": "model tokenizer/vocab is not GPT-2 compatible; use --tokenization-mode native_text for a native-tokenizer anchor",
                        })
                        continue
                    if validation_tokens is None:
                        validation_tokens = build_validation_tokens(tokenizer)
                    metrics = evaluate_token_stream(
                        model,
                        validation_tokens,
                        device,
                        dtype,
                        context_length=context_length,
                        max_batches=args.max_batches,
                    )
                else:
                    if validation_tokens is None:
                        validation_tokens = build_validation_tokens(tokenizer)
                    metrics = evaluate_token_stream(
                        model,
                        validation_tokens,
                        device,
                        dtype,
                        context_length=context_length,
                        max_batches=args.max_batches,
                    )

                report["results"].append({
                    "model": model_name,
                    "context_length": context_length,
                    "tokenization_mode": args.tokenization_mode,
                    "tokenizer": getattr(tokenizer, "name_or_path", model_name),
                    "model_type": getattr(config, "model_type", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                    "status": "ok" if metrics is not None else "empty",
                    "metrics": metrics,
                })
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            report["results"].append({
                "model": model_name,
                "status": "oom",
                "reason": str(exc),
            })
        except Exception as exc:
            report["results"].append({
                "model": model_name,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            })
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote HF baseline report to {output_path}")


if __name__ == "__main__":
    main()
