import argparse
import contextlib
import json
import os
import random
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from model_baseline_gru import GRUForCausalLM
from model_baseline_hf_mamba import HFMambaForCausalLM
from model_baseline_transformer import TransformerForCausalLM, choose_num_heads
from model_mogt import MOGTForCausalLM
from experiment_report import build_synthetic_standard_report


SET_MARK = 1
QUERY_MARK = 3
RESERVED_TOKENS = 16


def value_gate_input_dim(d_model: int, mode: str) -> int:
    if mode == "current":
        return d_model
    if mode == "current_prev":
        return 2 * d_model
    if mode == "current_prev_prefix":
        return 3 * d_model
    raise ValueError(f"unsupported value gate input mode: {mode}")


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


class LastValueGenerator:
    def __init__(self, vocab_size: int, num_values: int, min_updates: int, max_updates: int):
        if num_values < 2:
            raise ValueError("num_values must be at least 2")
        if min_updates < 1 or max_updates < min_updates:
            raise ValueError("invalid update count range")
        self.vocab_size = vocab_size
        self.num_values = num_values
        self.min_updates = min_updates
        self.max_updates = max_updates
        self.value_base = RESERVED_TOKENS
        self.filler_low = self.value_base + num_values
        if vocab_size <= self.filler_low:
            raise ValueError(f"vocab_size={vocab_size} too small; need > {self.filler_low}")

    def _sample_starts(self, context_length: int, num_updates: int):
        pair_width = 2
        query_pos = context_length - 1
        max_start = query_pos - pair_width
        if max_start + 1 < num_updates:
            raise ValueError("context_length is too short for requested updates")
        candidates = random.sample(range(max_start + 1), num_updates)
        return sorted(candidates)

    def generate_batch(
        self,
        batch_size: int,
        context_length: int,
        device: torch.device,
        *,
        dense_labels: bool = False,
    ):
        if context_length < 8:
            raise ValueError("context_length must be at least 8")

        input_ids = torch.randint(
            self.filler_low,
            self.vocab_size,
            (batch_size, context_length),
            device=device,
            dtype=torch.long,
        )
        input_ids[:, -1] = QUERY_MARK
        targets = torch.empty(batch_size, device=device, dtype=torch.long)
        dense_targets = None
        if dense_labels:
            dense_targets = torch.full(
                (batch_size, context_length),
                -100,
                device=device,
                dtype=torch.long,
            )
        query_pos = context_length - 1

        for batch_idx in range(batch_size):
            num_updates = random.randint(self.min_updates, self.max_updates)
            starts = self._sample_starts(context_length, num_updates)
            values = torch.randint(
                0,
                self.num_values,
                (num_updates,),
                device=device,
                dtype=torch.long,
            )
            for value_idx, start in enumerate(starts):
                input_ids[batch_idx, start] = SET_MARK
                input_ids[batch_idx, start + 1] = self.value_base + values[value_idx]
                if dense_labels:
                    end = starts[value_idx + 1] if value_idx + 1 < len(starts) else context_length
                    dense_targets[batch_idx, start + 1 : end] = self.value_base + values[value_idx]
            targets[batch_idx] = self.value_base + values[-1]

        if dense_labels:
            dense_targets[:, -1] = targets
            return input_ids, targets, query_pos, dense_targets
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
            if args.readout_init_std > 0:
                torch.nn.init.normal_(block.theta_read.weight, mean=0.0, std=args.readout_init_std)
            if args.transport_gate:
                transport_gate_dim = args.rank if args.transport_gate_width == "rank" else 1
                block.phi_transport_gate = torch.nn.Linear(
                    args.d_model,
                    transport_gate_dim,
                    bias=True,
                )
                torch.nn.init.zeros_(block.phi_transport_gate.weight)
                torch.nn.init.constant_(
                    block.phi_transport_gate.bias,
                    args.transport_gate_bias,
                )
                block.transport_gate_mode = args.transport_gate_mode
                block.transport_gate_scale = args.transport_gate_scale
            if args.value_gate:
                value_gate_dim = args.rank if args.value_gate_width == "rank" else 1
                block.phi_value_gate = torch.nn.Linear(
                    value_gate_input_dim(args.d_model, args.value_gate_input),
                    value_gate_dim,
                    bias=True,
                )
                block.value_gate_input_mode = args.value_gate_input
                torch.nn.init.zeros_(block.phi_value_gate.weight)
                torch.nn.init.constant_(
                    block.phi_value_gate.bias,
                    args.value_gate_bias,
                )
                block.couple_forget_to_value_gate = bool(args.couple_forget_to_value_gate)
        model.mogt.gradient_checkpointing = bool(args.gradient_checkpointing)
        if args.prefix_condition_position is not None:
            model.mogt.prefix_condition_position = args.prefix_condition_position
        return model.to(device)

    if args.model_type == "transformer":
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

    if args.model_type == "gru":
        # Some CUDA environments ship PyTorch without the cuDNN RNN symbols
        # needed by nn.GRU. The unfused path is slower but reliable enough for
        # synthetic baseline checks.
        torch.backends.cudnn.enabled = False
        model = GRUForCausalLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
        )
        return model.to(device)

    if args.model_type == "mamba":
        model = HFMambaForCausalLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
        )
        return model.to(device)

    raise ValueError(f"unknown model_type={args.model_type}")


def hidden_states(model, model_type: str, input_ids: torch.Tensor) -> torch.Tensor:
    if model_type == "mogt":
        return model.mogt(input_ids)
    if model_type == "transformer":
        return model.transformer(input_ids)
    if model_type == "gru":
        return model.backbone(input_ids)
    if model_type == "mamba":
        return model.backbone(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    raise ValueError(f"unknown model_type={model_type}")


def query_logits(model, model_type: str, input_ids: torch.Tensor, query_pos: int) -> torch.Tensor:
    hidden = hidden_states(model, model_type, input_ids)
    return F.linear(hidden[:, query_pos, :], model.lm_head.weight)


def dense_loss_and_logits(model, model_type: str, input_ids: torch.Tensor, dense_targets: torch.Tensor):
    hidden = hidden_states(model, model_type, input_ids)
    mask = dense_targets.ne(-100)
    selected_hidden = hidden[mask]
    selected_targets = dense_targets[mask]
    logits = F.linear(selected_hidden, model.lm_head.weight)
    loss = F.cross_entropy(logits.float(), selected_targets)
    return loss, logits, selected_targets


def evaluate(model, args, generator: LastValueGenerator, device: torch.device):
    model.eval()
    results = []
    eval_batch_size = int(args.eval_batch_size or args.batch_size)
    with torch.no_grad():
        for context_length in args.eval_contexts:
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            started = time.perf_counter()
            for _ in range(args.eval_batches):
                input_ids, targets, query_pos = generator.generate_batch(
                    eval_batch_size,
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


def gate_diagnostics(model, args, generator: LastValueGenerator, device: torch.device):
    if args.model_type != "mogt":
        return None
    if not any(hasattr(block, "phi_value_gate") for block in model.mogt.blocks):
        return None

    model.eval()
    batch_size = min(16, args.batch_size)
    with torch.no_grad():
        batch = generator.generate_batch(batch_size, args.train_context, device)
        input_ids = batch[0]
        masks = {
            "set": input_ids.eq(SET_MARK),
            "query": input_ids.eq(QUERY_MARK),
            "value": (input_ids >= generator.value_base)
            & (input_ids < generator.value_base + generator.num_values),
            "filler": input_ids >= generator.filler_low,
        }

        x = model.mogt.embedding(input_ids)
        prefix_condition = None
        prefix_position = getattr(model.mogt, "prefix_condition_position", None)
        if prefix_position is not None and -input_ids.size(1) <= int(prefix_position) < input_ids.size(1):
            prefix_condition = x[:, int(prefix_position), :]
        diagnostics = []
        for block_idx, block in enumerate(model.mogt.blocks):
            phi_value_gate = getattr(block, "phi_value_gate", None)
            if phi_value_gate is not None:
                x_norm = block.norm_mogt(x)
                gate_input = block._build_value_gate_input(x_norm, prefix_condition)
                gate = torch.sigmoid(phi_value_gate(gate_input).float())
                if gate.dim() == 3:
                    gate = gate.mean(dim=-1)
                stats = {}
                for name, mask in masks.items():
                    if int(mask.sum().item()) == 0:
                        stats[name] = None
                    else:
                        stats[name] = float(gate[mask].mean().item())
                diagnostics.append(
                    {
                        "block": block_idx,
                        "couple_forget_to_value_gate": bool(
                            getattr(block, "couple_forget_to_value_gate", False)
                        ),
                        "value_gate_width": "rank" if phi_value_gate.out_features == block.r else "scalar",
                        "mean_gate": stats,
                    }
                )
            x = block(x)
    model.train()
    return diagnostics


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    generator = LastValueGenerator(
        vocab_size=args.vocab_size,
        num_values=args.num_values,
        min_updates=args.min_updates,
        max_updates=args.max_updates,
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
        train_context = args.train_context
        if args.min_train_context and args.min_train_context < args.train_context:
            if args.curriculum_steps > 0:
                progress = min(1.0, step / float(args.curriculum_steps))
                max_context = int(
                    round(
                        args.min_train_context
                        + progress * (args.train_context - args.min_train_context)
                    )
                )
            else:
                max_context = args.train_context
            train_context = random.randint(args.min_train_context, max_context)

        batch = generator.generate_batch(
            args.batch_size,
            train_context,
            device,
            dense_labels=args.dense_loss,
        )
        if args.dense_loss:
            input_ids, targets, query_pos, dense_targets = batch
        else:
            input_ids, targets, query_pos = batch

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            if args.dense_loss:
                loss, dense_logits, dense_selected_targets = dense_loss_and_logits(
                    model,
                    args.model_type,
                    input_ids,
                    dense_targets,
                )
                logits = query_logits(model, args.model_type, input_ids, query_pos)
            else:
                logits = query_logits(model, args.model_type, input_ids, query_pos)
                loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step == args.steps or step % args.log_every == 0:
            predictions = logits.detach().argmax(dim=-1)
            accuracy = float((predictions == targets).float().mean().item())
            dense_accuracy = None
            if args.dense_loss:
                dense_predictions = dense_logits.detach().argmax(dim=-1)
                dense_accuracy = float(
                    (dense_predictions == dense_selected_targets).float().mean().item()
                )
            train_trace.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": accuracy,
                    "dense_accuracy": dense_accuracy,
                    "context_length": int(train_context),
                    "grad_norm": float(grad_norm.item()),
                }
            )
            dense_text = "" if dense_accuracy is None else f" dense_acc={dense_accuracy:.3f}"
            print(
                f"step={step} ctx={train_context} loss={loss.item():.4f} acc={accuracy:.3f}{dense_text}",
                flush=True,
            )

    sync_if_cuda(device)
    train_elapsed = time.perf_counter() - started
    eval_results = evaluate(model, args, generator, device)
    gate_stats = gate_diagnostics(model, args, generator, device)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "task": "last_value_tracking",
        "model_type": args.model_type,
        "params": count_parameters(model),
        "config": vars(args),
        "train_trace": train_trace,
        "eval_results": eval_results,
        "gate_diagnostics": gate_stats,
        "train_elapsed_s": train_elapsed,
        "peak_memory_mb": peak_memory_mb(device),
    }
    report["standard_report"] = build_synthetic_standard_report(
        task="last_value_tracking",
        args=args,
        parameter_count=report["params"],
        eval_results=eval_results,
        device=device,
        elapsed_seconds=train_elapsed,
        peak_memory_mb=report["peak_memory_mb"],
        notes="Synthetic last-value tracking benchmark.",
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic last-value state tracking benchmark."
    )
    parser.add_argument("--model-type", choices=["mogt", "transformer", "gru", "mamba"], required=True)
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
    parser.add_argument("--readout-init-std", type=float, default=0.0)
    parser.add_argument("--transport-gate", action="store_true")
    parser.add_argument(
        "--transport-gate-mode",
        choices=["multiply", "residual", "forget_relu"],
        default="multiply",
    )
    parser.add_argument("--transport-gate-width", choices=["scalar", "rank"], default="scalar")
    parser.add_argument("--transport-gate-scale", type=float, default=1.0)
    parser.add_argument("--transport-gate-bias", type=float, default=2.0)
    parser.add_argument("--value-gate", action="store_true")
    parser.add_argument("--value-gate-width", choices=["scalar", "rank"], default="scalar")
    parser.add_argument(
        "--value-gate-input",
        choices=["current", "current_prev", "current_prev_prefix"],
        default="current",
    )
    parser.add_argument("--value-gate-bias", type=float, default=0.0)
    parser.add_argument("--couple-forget-to-value-gate", action="store_true")
    parser.add_argument("--prefix-condition-position", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--train-context", type=int, default=128)
    parser.add_argument("--min-train-context", type=int, default=0)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--eval-contexts", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--min-updates", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--dense-loss", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
