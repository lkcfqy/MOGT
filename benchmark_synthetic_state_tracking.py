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


class ModularStateGenerator:
    def __init__(self, vocab_size: int, num_states: int, max_delta: int):
        if num_states < 2:
            raise ValueError("num_states must be at least 2")
        if max_delta < 1:
            raise ValueError("max_delta must be positive")

        self.vocab_size = vocab_size
        self.num_states = num_states
        self.max_delta = max_delta
        self.state_base = RESERVED_TOKENS
        self.delta_base = self.state_base + num_states
        self.num_delta_tokens = 2 * max_delta + 1
        self.min_vocab_size = self.delta_base + self.num_delta_tokens
        if vocab_size < self.min_vocab_size:
            raise ValueError(
                f"vocab_size={vocab_size} too small; need at least {self.min_vocab_size}"
            )

    def generate_batch(self, batch_size: int, context_length: int, device: torch.device):
        if context_length < 3:
            raise ValueError("context_length must be at least 3")

        num_ops = context_length - 2
        initial = torch.randint(
            0,
            self.num_states,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        delta_indices = torch.randint(
            0,
            self.num_delta_tokens,
            (batch_size, num_ops),
            device=device,
            dtype=torch.long,
        )
        deltas = delta_indices - self.max_delta
        final_state = (initial + deltas.sum(dim=1)) % self.num_states

        input_ids = torch.empty(batch_size, context_length, device=device, dtype=torch.long)
        input_ids[:, 0] = self.state_base + initial
        input_ids[:, 1:-1] = self.delta_base + delta_indices
        input_ids[:, -1] = QUERY_MARK

        targets = self.state_base + final_state
        query_pos = context_length - 1
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


def evaluate(model, args, generator: ModularStateGenerator, device: torch.device):
    model.eval()
    results = []
    with torch.no_grad():
        for context_length in args.eval_contexts:
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            started = time.perf_counter()
            for _ in range(args.eval_batches):
                input_ids, targets, query_pos = generator.generate_batch(
                    args.batch_size,
                    context_length,
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
            results.append(
                {
                    "context_length": int(context_length),
                    "loss": total_loss / max(1, total_count),
                    "accuracy": total_correct / max(1, total_count),
                    "examples": total_count,
                    "elapsed_s": time.perf_counter() - started,
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

    generator = ModularStateGenerator(
        vocab_size=args.vocab_size,
        num_states=args.num_states,
        max_delta=args.max_delta,
    )
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    train_trace = []
    started = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        input_ids, targets, query_pos = generator.generate_batch(
            args.batch_size,
            args.train_context,
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
                    "grad_norm": float(grad_norm.item()),
                }
            )
            print(
                f"step={step} loss={loss.item():.4f} acc={accuracy:.3f}",
                flush=True,
            )

    sync_if_cuda(device)
    train_elapsed = time.perf_counter() - started
    eval_results = evaluate(model, args, generator, device)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "task": "modular_state_tracking",
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
        description="Synthetic modular state-tracking benchmark."
    )
    parser.add_argument("--model-type", choices=["mogt", "transformer"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=128)
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
    parser.add_argument("--train-context", type=int, default=128)
    parser.add_argument("--eval-contexts", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--num-states", type=int, default=16)
    parser.add_argument("--max-delta", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
