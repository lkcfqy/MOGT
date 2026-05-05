import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup

from dataset import get_dataloaders
from experiment_report import build_lm_standard_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train a budget-matched scratch baseline on the MOGT data protocol.")
    parser.add_argument("--model-type", choices=["mamba_ssm"], default="mamba_ssm")
    parser.add_argument("--run-name", default="mamba_scratch_budget_v1")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-max-batches", type=int, default=10)
    parser.add_argument("--eval-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-chunk-size", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--report-output", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--latest-checkpoint-interval", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_gradient_checkpointing(requested: str, context_length: int) -> bool:
    if requested == "on":
        return True
    if requested == "off":
        return False
    return context_length >= 8192


def autocast_context(device, amp_dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def build_mamba_model(vocab_size: int, args, device):
    config = MambaConfig(
        d_model=args.d_model,
        n_layer=args.num_layers,
        vocab_size=vocab_size,
        d_intermediate=0,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=False,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    )
    model = MambaLMHeadModel(config)
    model.to(device)
    return model


def mamba_backbone_forward(model, input_ids, *, gradient_checkpointing: bool):
    backbone = model.backbone
    hidden_states = backbone.embedding(input_ids)
    residual = None

    for layer in backbone.layers:
        current_layer = layer
        if gradient_checkpointing and model.training:
            if residual is None:
                hidden_states, residual = checkpoint(
                    lambda hidden: current_layer(hidden, None, inference_params=None),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, residual = checkpoint(
                    lambda hidden, res: current_layer(hidden, res, inference_params=None),
                    hidden_states,
                    residual,
                    use_reentrant=False,
                )
        else:
            hidden_states, residual = layer(hidden_states, residual, inference_params=None)

    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
    return hidden_states


def chunked_lm_loss_from_hidden(model, hidden_states, labels, *, loss_chunk_size: int):
    vocab_size = model.lm_head.weight.size(0)
    total_loss = hidden_states.new_zeros(())
    total_tokens = 0

    for start in range(0, hidden_states.size(1), loss_chunk_size):
        end = min(start + loss_chunk_size, hidden_states.size(1))
        logits = model.lm_head(hidden_states[:, start:end, :])
        labels_chunk = labels[:, start:end]
        total_loss = total_loss + F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            labels_chunk.reshape(-1),
            reduction="sum",
        )
        total_tokens += labels_chunk.numel()

    return total_loss / total_tokens


def model_loss(model, input_ids, labels, *, loss_chunk_size: int, gradient_checkpointing: bool):
    hidden_states = mamba_backbone_forward(
        model,
        input_ids,
        gradient_checkpointing=gradient_checkpointing,
    )
    return chunked_lm_loss_from_hidden(
        model,
        hidden_states,
        labels,
        loss_chunk_size=loss_chunk_size,
    )


def evaluate_model(model, val_dl, device, amp_dtype, *, max_batches: int, loss_chunk_size: int):
    if max_batches <= 0:
        return None

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0

    try:
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_dl):
                if batch_idx >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                with autocast_context(device, amp_dtype):
                    loss = model_loss(
                        model,
                        x,
                        y,
                        loss_chunk_size=loss_chunk_size,
                        gradient_checkpointing=False,
                    )
                total_loss += float(loss.item())
                total_batches += 1
    finally:
        if was_training:
            model.train()

    if total_batches == 0:
        return None

    avg_loss = total_loss / total_batches
    return {
        "loss": avg_loss,
        "ppl": math.exp(avg_loss),
        "num_batches": total_batches,
    }


def save_checkpoint(path, model, optimizer, scheduler, *, epoch, batch_idx, global_step, train_loss, best_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": batch_idx,
            "global_accum_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_loss,
            "best_val_loss": None if best_metrics is None else best_metrics["loss"],
            "best_val_ppl": None if best_metrics is None else best_metrics["ppl"],
            "best_val_batches": None if best_metrics is None else best_metrics["num_batches"],
        },
        path,
    )


def load_resume_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint_data = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    global_step = int(checkpoint_data.get("global_accum_step", 0))
    if "scheduler_state_dict" in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
    else:
        scheduler.last_epoch = global_step
        if hasattr(scheduler, "lr_lambdas"):
            for group, base_lr, lr_lambda in zip(
                optimizer.param_groups,
                scheduler.base_lrs,
                scheduler.lr_lambdas,
            ):
                group["lr"] = base_lr * lr_lambda(global_step)
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        else:
            for _ in range(global_step):
                scheduler.step()

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    best_loss = checkpoint_data.get("best_val_loss")
    best_metrics = None
    if best_loss is not None:
        best_metrics = {
            "loss": float(best_loss),
            "ppl": float(checkpoint_data["best_val_ppl"]),
            "num_batches": int(checkpoint_data["best_val_batches"]),
        }

    return {
        "epoch": int(checkpoint_data.get("epoch", 0)),
        "next_batch_idx": int(checkpoint_data.get("step", -1)) + 1,
        "global_step": global_step,
        "train_loss": checkpoint_data.get("loss"),
        "best_metrics": best_metrics,
        "has_scheduler_state": "scheduler_state_dict" in checkpoint_data,
    }


def main():
    args = parse_args()
    if args.smoke:
        args.context_length = 8192 if args.context_length >= 8192 else args.context_length
        args.num_layers = min(args.num_layers, 2)
        args.d_model = min(args.d_model, 128)
        args.max_steps = min(args.max_steps, 2)
        args.grad_accum_steps = 1
        args.eval_interval = 1
        args.eval_max_batches = 1

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    loss_chunk_size = args.loss_chunk_size or (256 if args.context_length >= 32768 else 4096)
    gradient_checkpointing = resolve_gradient_checkpointing(args.gradient_checkpointing, args.context_length)

    checkpoint_dir = Path(args.checkpoint_dir or f"baseline_checkpoints/{args.run_name}_seed{args.seed}")
    report_output = Path(args.report_output or f"benchmark_runs/{args.run_name}_seed{args.seed}.json")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dl, val_dl, vocab_size = get_dataloaders(
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_mamba_model(vocab_size, args, device)
    param_count = sum(param.numel() for param in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(args.max_steps * 0.05)),
        num_training_steps=max(1, args.max_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    resume_state = None
    if args.resume_from:
        resume_state = load_resume_checkpoint(
            args.resume_from,
            model,
            optimizer,
            scheduler,
            device,
        )

    print("==================================================")
    print(f"Budget baseline: {args.model_type} | device={device} | run={args.run_name}")
    print(
        f"config: d_model={args.d_model}, layers={args.num_layers}, params={param_count/1e6:.2f}M, "
        f"ctx={args.context_length}, batch={args.batch_size}, accum={args.grad_accum_steps}"
    )
    print(
        f"train: max_steps={args.max_steps}, lr={args.lr}, loss_chunk={loss_chunk_size}, "
        f"checkpointing={'on' if gradient_checkpointing else 'off'}"
    )
    if resume_state is not None:
        print(
            f"resume: {args.resume_from} | global_step={resume_state['global_step']} | "
            f"next_batch={resume_state['next_batch_idx']} | "
            f"scheduler_state={'yes' if resume_state['has_scheduler_state'] else 'fast-forwarded'}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0 if resume_state is None else resume_state["global_step"]
    best_metrics = None if resume_state is None else resume_state["best_metrics"]
    validation_trace = []
    start_time = time.time()

    for epoch in range(args.max_epochs):
        if resume_state is not None and epoch < resume_state["epoch"]:
            continue
        torch.manual_seed(args.seed + epoch)
        for batch_idx, (x, y) in enumerate(train_dl):
            if (
                resume_state is not None
                and epoch == resume_state["epoch"]
                and batch_idx < resume_state["next_batch_idx"]
            ):
                continue

            x = x.to(device)
            y = y.to(device)

            with autocast_context(device, amp_dtype):
                loss = model_loss(
                    model,
                    x,
                    y,
                    loss_chunk_size=loss_chunk_size,
                    gradient_checkpointing=gradient_checkpointing,
                )
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                train_loss = float(loss.item())
                print(f"step={global_step} train_loss={train_loss:.4f}", flush=True)

                ran_eval = False
                if args.eval_max_batches > 0 and args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    metrics = evaluate_model(
                        model,
                        val_dl,
                        device,
                        amp_dtype,
                        max_batches=args.eval_max_batches,
                        loss_chunk_size=loss_chunk_size,
                    )
                    if metrics is not None:
                        ran_eval = True
                        validation_trace.append({"step": global_step, **metrics})
                        print(
                            f"val step={global_step} loss={metrics['loss']:.4f} "
                            f"ppl={metrics['ppl']:.2f} batches={metrics['num_batches']}",
                            flush=True,
                        )
                        if args.save_best and (best_metrics is None or metrics["loss"] < best_metrics["loss"]):
                            best_metrics = metrics
                            save_checkpoint(
                                checkpoint_dir / "best.pt",
                                model,
                                optimizer,
                                scheduler,
                                epoch=epoch,
                                batch_idx=batch_idx,
                                global_step=global_step,
                                train_loss=train_loss,
                                best_metrics=best_metrics,
                            )

                if args.latest_checkpoint_interval > 0 and global_step % args.latest_checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_dir / "latest.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        train_loss=train_loss,
                        best_metrics=best_metrics,
                    )

                if global_step >= args.max_steps:
                    if args.eval_at_end and args.eval_max_batches > 0 and not ran_eval:
                        metrics = evaluate_model(
                            model,
                            val_dl,
                            device,
                            amp_dtype,
                            max_batches=args.eval_max_batches,
                            loss_chunk_size=loss_chunk_size,
                        )
                        if metrics is not None:
                            validation_trace.append({"step": global_step, **metrics})
                            if args.save_best and (best_metrics is None or metrics["loss"] < best_metrics["loss"]):
                                best_metrics = metrics
                                save_checkpoint(
                                    checkpoint_dir / "best.pt",
                                    model,
                                    optimizer,
                                    scheduler,
                                    epoch=epoch,
                                    batch_idx=batch_idx,
                                    global_step=global_step,
                                    train_loss=train_loss,
                                    best_metrics=best_metrics,
                                )

                    save_checkpoint(
                        checkpoint_dir / "last.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        train_loss=train_loss,
                        best_metrics=best_metrics,
                    )

                    elapsed_s = time.time() - start_time
                    report = {
                        "config": {
                            "model_type": args.model_type,
                            "run_name": args.run_name,
                            "context_length": args.context_length,
                            "d_model": args.d_model,
                            "num_layers": args.num_layers,
                            "vocab_size": vocab_size,
                            "parameter_count": param_count,
                            "batch_size": args.batch_size,
                            "grad_accum_steps": args.grad_accum_steps,
                            "max_steps": args.max_steps,
                            "eval_interval": args.eval_interval,
                            "eval_max_batches_requested": args.eval_max_batches,
                            "seed": args.seed,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "loss_chunk_size": loss_chunk_size,
                            "gradient_checkpointing": gradient_checkpointing,
                            "resume_from": args.resume_from or None,
                            "amp_dtype": str(amp_dtype).replace("torch.", ""),
                        },
                        "result": {
                            "status": "ok",
                            "elapsed_s": elapsed_s,
                            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2)
                            if device.type == "cuda"
                            else None,
                            "train_loss_final": train_loss,
                            "best_val": best_metrics,
                            "checkpoint_dir": str(checkpoint_dir),
                            "best_checkpoint": str(checkpoint_dir / "best.pt") if best_metrics is not None else None,
                            "latest_checkpoint": str(checkpoint_dir / "latest.pt")
                            if (checkpoint_dir / "latest.pt").exists()
                            else None,
                            "last_checkpoint": str(checkpoint_dir / "last.pt"),
                        },
                        "validation_trace": validation_trace,
                        "notes": [
                            "Scratch Mamba-style baseline trained with the same GPT-2 tokenizer and WikiText-103 data pipeline as MOGT.",
                            "This is a training-budget matched baseline, not a pretrained quality anchor.",
                        ],
                    }
                    report["standard_report"] = build_lm_standard_report(
                        task="language_modeling",
                        config=report["config"],
                        result=report["result"],
                        device=device,
                        validation_trace=validation_trace,
                        notes="Scratch Mamba-style budget-matched language-modeling baseline.",
                    )
                    report_output.parent.mkdir(parents=True, exist_ok=True)
                    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(f"Wrote report to {report_output}")
                    return


if __name__ == "__main__":
    main()
