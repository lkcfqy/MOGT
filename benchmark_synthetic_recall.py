import argparse
import contextlib
import json
import os
import random
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from model_baseline_transformer import TransformerForCausalLM, choose_num_heads
from model_mogt import MOGTForCausalLM


KEY_MARK = 1
VALUE_MARK = 2
QUERY_MARK = 3
RESERVED_TOKENS = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def autocast_context(device: torch.device, dtype_name: str):
    if device.type == "cuda" and dtype_name == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and dtype_name == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class KVRecallGenerator:
    def __init__(self, vocab_size: int, num_pairs: int):
        if vocab_size < 128:
            raise ValueError("vocab_size must be at least 128")
        if num_pairs < 1:
            raise ValueError("num_pairs must be positive")

        usable = vocab_size - RESERVED_TOKENS
        self.vocab_size = vocab_size
        self.num_pairs = num_pairs
        self.key_low = RESERVED_TOKENS
        self.key_high = RESERVED_TOKENS + usable // 4
        self.value_low = RESERVED_TOKENS + usable // 4
        self.value_high = RESERVED_TOKENS + usable // 2
        self.filler_low = RESERVED_TOKENS + usable // 2
        self.filler_high = vocab_size

        if self.key_high - self.key_low < num_pairs:
            raise ValueError("vocab_size is too small for unique key tokens")
        if self.value_high - self.value_low < num_pairs:
            raise ValueError("vocab_size is too small for unique value tokens")

    def _random_non_overlapping_start(self, intervals, max_start: int, width: int) -> int:
        for _ in range(256):
            start = random.randint(0, max_start)
            if all(start + width <= left or right <= start for left, right in intervals):
                return start
        for start in range(max_start + 1):
            if all(start + width <= left or right <= start for left, right in intervals):
                return start
        raise RuntimeError("could not place synthetic key/value pair without overlap")

    def generate_batch(
        self,
        batch_size: int,
        context_length: int,
        depth_ratio: float,
        device: torch.device,
    ):
        pair_width = 4
        query_width = 2
        min_length = self.num_pairs * pair_width + query_width + 4
        if context_length < min_length:
            raise ValueError(
                f"context_length={context_length} is too short for "
                f"num_pairs={self.num_pairs}; need at least {min_length}"
            )

        depth_ratio = min(1.0, max(0.0, float(depth_ratio)))
        input_ids = torch.randint(
            self.filler_low,
            self.filler_high,
            (batch_size, context_length),
            device=device,
            dtype=torch.long,
        )
        targets = torch.empty(batch_size, device=device, dtype=torch.long)
        query_pos = context_length - 1
        query_start = context_length - query_width
        max_start = query_start - pair_width - 1

        for batch_idx in range(batch_size):
            key_perm = torch.randperm(self.key_high - self.key_low, device=device)
            value_perm = torch.randperm(self.value_high - self.value_low, device=device)
            keys = key_perm[: self.num_pairs] + self.key_low
            values = value_perm[: self.num_pairs] + self.value_low

            target_pair = random.randrange(self.num_pairs)
            target_start = int(round(depth_ratio * max_start))
            intervals = [(target_start, target_start + pair_width)]
            starts = [None] * self.num_pairs
            starts[target_pair] = target_start

            for pair_idx in range(self.num_pairs):
                if pair_idx == target_pair:
                    continue
                start = self._random_non_overlapping_start(intervals, max_start, pair_width)
                intervals.append((start, start + pair_width))
                starts[pair_idx] = start

            for pair_idx, start in enumerate(starts):
                input_ids[batch_idx, start] = KEY_MARK
                input_ids[batch_idx, start + 1] = keys[pair_idx]
                input_ids[batch_idx, start + 2] = VALUE_MARK
                input_ids[batch_idx, start + 3] = values[pair_idx]

            input_ids[batch_idx, query_start] = QUERY_MARK
            input_ids[batch_idx, query_start + 1] = keys[target_pair]
            targets[batch_idx] = values[target_pair]

        return input_ids, targets, query_pos


def build_model(args, device: torch.device):
    if args.model_type == "mogt":
        model = MOGTForCausalLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            r=args.rank,
        )
        for block in model.mogt.blocks:
            block.scan_impl = args.scan_impl
            block.connection_impl = args.connection_impl
            block.connection_damping = args.connection_damping
            block.scan_block_size = args.scan_block_size
            if args.transport_gate:
                block.phi_transport_gate = torch.nn.Linear(args.d_model, 1, bias=True)
                torch.nn.init.zeros_(block.phi_transport_gate.weight)
                torch.nn.init.constant_(
                    block.phi_transport_gate.bias,
                    args.transport_gate_bias,
                )
        model.mogt.gradient_checkpointing = bool(args.gradient_checkpointing)
        return model.to(device)

    num_heads = choose_num_heads(args.d_model, args.num_heads)
    model = TransformerForCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=num_heads,
        rope_theta=args.rope_theta,
    )
    model.transformer.gradient_checkpointing = bool(args.gradient_checkpointing)
    return model.to(device)


def query_logits(model, model_type: str, input_ids: torch.Tensor, query_pos: int) -> torch.Tensor:
    if model_type == "mogt":
        hidden = model.mogt(input_ids)
    else:
        hidden = model.transformer(input_ids)
    return F.linear(hidden[:, query_pos, :], model.lm_head.weight)


def evaluate(model, args, generator: KVRecallGenerator, device: torch.device):
    model.eval()
    results = []
    with torch.no_grad():
        for context_length in args.eval_contexts:
            for depth in args.eval_depths:
                total_loss = 0.0
                total_correct = 0
                total_count = 0
                started = time.perf_counter()
                for _ in range(args.eval_batches):
                    input_ids, targets, query_pos = generator.generate_batch(
                        args.batch_size,
                        context_length,
                        depth,
                        device,
                    )
                    with autocast_context(device, args.dtype):
                        logits = query_logits(model, args.model_type, input_ids, query_pos)
                        loss = F.cross_entropy(logits.float(), targets, reduction="sum")
                    predictions = logits.argmax(dim=-1)
                    total_loss += float(loss.item())
                    total_correct += int((predictions == targets).sum().item())
                    total_count += int(targets.numel())
                sync_if_cuda(device)
                elapsed = time.perf_counter() - started
                results.append(
                    {
                        "context_length": int(context_length),
                        "depth": float(depth),
                        "loss": total_loss / max(1, total_count),
                        "accuracy": total_correct / max(1, total_count),
                        "examples": total_count,
                        "elapsed_s": elapsed,
                    }
                )
    model.train()
    return results


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    generator = KVRecallGenerator(vocab_size=args.vocab_size, num_pairs=args.num_pairs)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    train_trace = []
    model.train()
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        depth = random.uniform(args.train_min_depth, args.train_max_depth)
        input_ids, targets, query_pos = generator.generate_batch(
            args.batch_size,
            args.train_context,
            depth,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            logits = query_logits(model, args.model_type, input_ids, query_pos)
            loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step == args.steps or step % args.log_every == 0:
            predictions = logits.detach().argmax(dim=-1)
            accuracy = float((predictions == targets).float().mean().item())
            train_trace.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": accuracy,
                    "depth": float(depth),
                    "grad_norm": float(grad_norm.item()),
                }
            )
            print(
                f"step={step} loss={loss.item():.4f} "
                f"acc={accuracy:.3f} depth={depth:.3f}",
                flush=True,
            )

    sync_if_cuda(device)
    train_elapsed = time.perf_counter() - started
    eval_results = evaluate(model, args, generator, device)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "model_type": args.model_type,
        "params": count_parameters(model),
        "config": vars(args),
        "train_trace": train_trace,
        "eval_results": eval_results,
        "train_elapsed_s": train_elapsed,
        "peak_memory_mb": peak_memory_mb(device),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Budget-matched synthetic key/value recall benchmark."
    )
    parser.add_argument("--model-type", choices=["mogt", "transformer"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--scan-impl", default="triton_hybrid")
    parser.add_argument("--connection-impl", default="cayley")
    parser.add_argument("--connection-damping", type=float, default=0.999)
    parser.add_argument("--scan-block-size", type=int, default=256)
    parser.add_argument("--transport-gate", action="store_true")
    parser.add_argument("--transport-gate-bias", type=float, default=2.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--train-context", type=int, default=1024)
    parser.add_argument("--eval-contexts", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--eval-depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--train-min-depth", type=float, default=0.05)
    parser.add_argument("--train-max-depth", type=float, default=0.95)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
