import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", os.environ.get("MOGT_ALLOC_CONF", "expandable_segments:True"))

import torch
from transformers import get_cosine_schedule_with_warmup

from dataset import get_dataloaders
from experiment_report import build_lm_standard_report
from model_baseline_transformer import TransformerForCausalLM, choose_num_heads


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a budget-matched scratch Transformer baseline on the MOGT data protocol."
    )
    parser.add_argument("--run-name", default="transformer_scratch_budget_v1")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=0, help="0 chooses a standard divisor automatically.")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
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
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
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


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def model_loss(model, input_ids, labels, *, loss_chunk_size: int):
    _, loss = model(
        input_ids,
        labels=labels,
        return_logits=False,
        loss_chunk_size=loss_chunk_size,
    )
    return loss


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
                    loss = model_loss(model, x, y, loss_chunk_size=loss_chunk_size)
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


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {path}")


def main():
    args = parse_args()
    if args.smoke:
        args.context_length = 8192 if args.context_length >= 8192 else args.context_length
        args.num_layers = min(args.num_layers, 2)
        args.d_model = min(args.d_model, 128)
        args.num_heads = args.num_heads or 4
        args.max_steps = min(args.max_steps, 2)
        args.grad_accum_steps = 1
        args.eval_interval = 1
        args.eval_max_batches = 1
        args.latest_checkpoint_interval = 0

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    loss_chunk_size = args.loss_chunk_size or (256 if args.context_length >= 32768 else 4096)
    gradient_checkpointing = resolve_gradient_checkpointing(args.gradient_checkpointing, args.context_length)
    num_heads = choose_num_heads(args.d_model, args.num_heads)

    checkpoint_dir = Path(args.checkpoint_dir or f"baseline_checkpoints/{args.run_name}_seed{args.seed}")
    report_output = Path(args.report_output or f"benchmark_runs/{args.run_name}_seed{args.seed}.json")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    train_dl, val_dl, vocab_size = get_dataloaders(
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TransformerForCausalLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=num_heads,
        rope_theta=args.rope_theta,
    )
    model.to(device)
    model.transformer.gradient_checkpointing = gradient_checkpointing

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
        resume_state = load_resume_checkpoint(args.resume_from, model, optimizer, scheduler, device)

    config = {
        "model_type": "scratch_transformer",
        "run_name": args.run_name,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": num_heads,
        "rope_theta": args.rope_theta,
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
        "save_best": args.save_best,
        "save_last": args.save_last,
        "resume_from": args.resume_from or None,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "attention": "torch.scaled_dot_product_attention",
    }

    print("==================================================")
    print(f"Budget baseline: scratch_transformer | device={device} | run={args.run_name}")
    print(
        f"config: d_model={args.d_model}, layers={args.num_layers}, heads={num_heads}, "
        f"params={param_count/1e6:.2f}M, ctx={args.context_length}, "
        f"batch={args.batch_size}, accum={args.grad_accum_steps}"
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
    train_loss = None
    start_time = time.time()
    current_phase = "setup"
    current_epoch = 0
    current_batch_idx = -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(args.max_epochs):
            current_epoch = epoch
            if resume_state is not None and epoch < resume_state["epoch"]:
                continue
            torch.manual_seed(args.seed + epoch)
            for batch_idx, (x, y) in enumerate(train_dl):
                current_phase = "train"
                current_batch_idx = batch_idx
                if (
                    resume_state is not None
                    and epoch == resume_state["epoch"]
                    and batch_idx < resume_state["next_batch_idx"]
                ):
                    continue

                x = x.to(device)
                y = y.to(device)

                with autocast_context(device, amp_dtype):
                    loss = model_loss(model, x, y, loss_chunk_size=loss_chunk_size)
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
                        current_phase = "eval"
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
                            if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
                                best_metrics = metrics
                                if args.save_best:
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
                        current_phase = "checkpoint"
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
                            current_phase = "final_eval"
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
                                if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
                                    best_metrics = metrics
                                    if args.save_best:
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

                        current_phase = "final_checkpoint"
                        last_checkpoint = None
                        if args.save_last:
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
                            last_checkpoint = str(checkpoint_dir / "last.pt")

                        elapsed_s = time.time() - start_time
                        report = {
                            "config": config,
                            "result": {
                                "status": "ok",
                                "elapsed_s": elapsed_s,
                                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2)
                                if device.type == "cuda"
                                else None,
                                "train_loss_final": train_loss,
                                "best_val": best_metrics,
                                "checkpoint_dir": str(checkpoint_dir),
                                "best_checkpoint": str(checkpoint_dir / "best.pt")
                                if args.save_best and best_metrics is not None
                                else None,
                                "latest_checkpoint": str(checkpoint_dir / "latest.pt")
                                if (checkpoint_dir / "latest.pt").exists()
                                else None,
                                "last_checkpoint": last_checkpoint,
                            },
                            "validation_trace": validation_trace,
                            "notes": [
                                "Scratch Transformer baseline trained with the same GPT-2 tokenizer and WikiText-103 data pipeline as MOGT.",
                                "This uses PyTorch scaled_dot_product_attention with a causal mask and RoPE positional encoding.",
                                "This is a training-budget matched baseline, not a pretrained quality anchor.",
                            ],
                        }
                        report["standard_report"] = build_lm_standard_report(
                            task="language_modeling",
                            config=config,
                            result=report["result"],
                            device=device,
                            validation_trace=validation_trace,
                            notes="Scratch Transformer budget-matched language-modeling baseline.",
                        )
                        write_report(report_output, report)
                        return

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed_s = time.time() - start_time
        report = {
            "config": config,
            "result": {
                "status": "oom",
                "elapsed_s": elapsed_s,
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None,
                "train_loss_final": train_loss,
                "best_val": best_metrics,
                "checkpoint_dir": str(checkpoint_dir),
                "global_step": global_step,
                "failed_phase": current_phase,
                "failed_epoch": current_epoch,
                "failed_batch_idx": current_batch_idx,
                "error": str(exc),
            },
            "validation_trace": validation_trace,
            "notes": [
                "OOM is retained as a useful systems datapoint for long-context Transformer scaling.",
            ],
        }
        report["standard_report"] = build_lm_standard_report(
            task="language_modeling",
            config=config,
            result=report["result"],
            device=device,
            validation_trace=validation_trace,
            notes="Scratch Transformer budget-matched language-modeling baseline OOM.",
        )
        write_report(report_output, report)
        return
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed_s = time.time() - start_time
        report = {
            "config": config,
            "result": {
                "status": "oom",
                "elapsed_s": elapsed_s,
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None,
                "train_loss_final": train_loss,
                "best_val": best_metrics,
                "checkpoint_dir": str(checkpoint_dir),
                "global_step": global_step,
                "failed_phase": current_phase,
                "failed_epoch": current_epoch,
                "failed_batch_idx": current_batch_idx,
                "error": str(exc),
            },
            "validation_trace": validation_trace,
            "notes": [
                "OOM is retained as a useful systems datapoint for long-context Transformer scaling.",
            ],
        }
        report["standard_report"] = build_lm_standard_report(
            task="language_modeling",
            config=config,
            result=report["result"],
            device=device,
            validation_trace=validation_trace,
            notes="Scratch Transformer budget-matched language-modeling baseline OOM.",
        )
        write_report(report_output, report)
        return


if __name__ == "__main__":
    main()
