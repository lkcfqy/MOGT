import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", os.environ.get("MOGT_ALLOC_CONF", "expandable_segments:True"))

import torch
from transformers import get_cosine_schedule_with_warmup

from dataset import get_dataloaders
from experiment_report import build_lm_standard_report
from model_baseline_transformer import choose_num_heads
from model_hybrid import HybridMOGTTransformerForCausalLM
from train_budget_transformer import (
    autocast_context,
    evaluate_model,
    is_oom_error,
    load_resume_checkpoint,
    model_loss,
    resolve_gradient_checkpointing,
    save_checkpoint,
    write_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a budget-matched MOGT/Transformer hybrid on the MOGT data protocol."
    )
    parser.add_argument("--run-name", default="hybrid_mogt_transformer_budget_v1")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=0, help="0 chooses a standard divisor automatically.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--hybrid-pattern",
        choices=[
            "alternating",
            "mogt_first_half",
            "mogt_second_half",
            "all_mogt",
            "all_transformer",
            "ratio_even",
        ],
        default="alternating",
    )
    parser.add_argument(
        "--mogt-layer-fraction",
        type=float,
        default=None,
        help="Optional exact MOGT layer fraction for ratio sweeps; overrides --hybrid-pattern.",
    )
    parser.add_argument(
        "--mogt-layer-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit zero-based MOGT layer indices; overrides --mogt-layer-fraction and --hybrid-pattern.",
    )
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--scan-impl", default="triton_hybrid")
    parser.add_argument("--connection-impl", choices=["matrix_exp", "cayley", "identity"], default="cayley")
    parser.add_argument("--connection-damping", type=float, default=0.999)
    parser.add_argument("--scan-block-size", type=int, default=256)
    parser.add_argument(
        "--mogt-residual-scale",
        type=float,
        default=1.0,
        help="Scale applied to the MOGT readout residual branch.",
    )
    parser.add_argument(
        "--mogt-residual-scale-start",
        type=float,
        default=None,
        help="Optional starting value for a linear MOGT readout residual-scale warmup schedule.",
    )
    parser.add_argument(
        "--mogt-residual-scale-warmup-steps",
        type=int,
        default=0,
        help="Optimizer steps used to linearly warm --mogt-residual-scale-start to --mogt-residual-scale.",
    )
    parser.add_argument(
        "--mogt-ffn-residual-scale",
        type=float,
        default=1.0,
        help="Scale applied to FFN residual branches inside MOGT blocks.",
    )
    parser.add_argument(
        "--mogt-residual-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a learned scalar gate on each MOGT readout residual branch.",
    )
    parser.add_argument(
        "--mogt-residual-gate-init",
        type=float,
        default=0.5,
        help="Initial sigmoid value for --mogt-residual-gate.",
    )
    parser.add_argument(
        "--zero-init-attention-out",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero-initialize Transformer attention output projections inside the hybrid model.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--mogt-lr-mult",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for parameters inside MOGT blocks.",
    )
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


def build_variant(args) -> str:
    pattern = args.hybrid_pattern
    if args.mogt_layer_indices is not None:
        pattern = "layers" + "-".join(str(index) for index in args.mogt_layer_indices)
    elif args.mogt_layer_fraction is not None:
        pattern = f"ratio{args.mogt_layer_fraction:g}".replace(".", "p")
    if args.zero_init_attention_out:
        pattern = f"{pattern}_zeroattn"
    if args.mogt_residual_scale != 1.0:
        scale_text = f"{args.mogt_residual_scale:g}".replace(".", "p")
        pattern = f"{pattern}_mogtscale{scale_text}"
    if residual_scale_schedule_enabled(args):
        start_text = f"{args.mogt_residual_scale_start:g}".replace(".", "p")
        target_text = f"{args.mogt_residual_scale:g}".replace(".", "p")
        pattern = f"{pattern}_mogtsched{start_text}to{target_text}s{args.mogt_residual_scale_warmup_steps}"
    if args.mogt_ffn_residual_scale != 1.0:
        scale_text = f"{args.mogt_ffn_residual_scale:g}".replace(".", "p")
        pattern = f"{pattern}_mogtffnscale{scale_text}"
    if args.mogt_residual_gate:
        gate_text = f"{args.mogt_residual_gate_init:g}".replace(".", "p")
        pattern = f"{pattern}_mogtgate{gate_text}"
    if args.mogt_lr_mult != 1.0:
        lr_mult_text = f"{args.mogt_lr_mult:g}".replace(".", "p")
        pattern = f"{pattern}_mogtlr{lr_mult_text}"
    return (
        f"hybrid_{pattern}_r{args.rank}_"
        f"{args.scan_impl}_{args.connection_impl}_damp{args.connection_damping:g}"
    )


def residual_scale_schedule_enabled(args) -> bool:
    return args.mogt_residual_scale_start is not None and args.mogt_residual_scale_warmup_steps > 0


def residual_scale_schedule_name(args) -> str:
    return "linear_warmup" if residual_scale_schedule_enabled(args) else "constant"


def residual_scale_for_step(args, global_step: int) -> float:
    if not residual_scale_schedule_enabled(args):
        return float(args.mogt_residual_scale)
    warmup_steps = max(1, int(args.mogt_residual_scale_warmup_steps))
    progress = min(max(float(global_step), 0.0) / warmup_steps, 1.0)
    start = float(args.mogt_residual_scale_start)
    target = float(args.mogt_residual_scale)
    return start + progress * (target - start)


def apply_residual_scale_schedule(model: HybridMOGTTransformerForCausalLM, args, global_step: int) -> float:
    scale = residual_scale_for_step(args, global_step)
    model.backbone.set_mogt_residual_scale(scale)
    return scale


def build_optimizer(model: HybridMOGTTransformerForCausalLM, args):
    if abs(float(args.mogt_lr_mult) - 1.0) <= 1e-12:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mogt_param_ids = {
        id(parameter)
        for block_type, block in zip(model.backbone.block_types, model.backbone.blocks)
        if block_type == "mogt"
        for parameter in block.parameters()
        if parameter.requires_grad
    }
    mogt_params = []
    base_params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in mogt_param_ids:
            mogt_params.append(parameter)
        else:
            base_params.append(parameter)
    groups = [{"params": base_params, "lr": args.lr}]
    if mogt_params:
        groups.append({"params": mogt_params, "lr": args.lr * args.mogt_lr_mult})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    if args.mogt_residual_scale_start is not None and args.mogt_residual_scale_warmup_steps <= 0:
        raise SystemExit("--mogt-residual-scale-start requires --mogt-residual-scale-warmup-steps > 0")
    if args.smoke:
        args.context_length = 8192 if args.context_length >= 8192 else args.context_length
        args.num_layers = min(args.num_layers, 2)
        args.d_model = min(args.d_model, 128)
        args.num_heads = args.num_heads or 4
        args.rank = min(args.rank, args.d_model)
        args.max_steps = min(args.max_steps, 2)
        args.grad_accum_steps = 1
        args.eval_interval = 1
        args.eval_max_batches = 1
        args.latest_checkpoint_interval = 0
        if args.scan_impl == "triton_hybrid" and not torch.cuda.is_available():
            args.scan_impl = "block_reference"

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type != "cuda" and args.scan_impl == "triton_hybrid":
        print("triton_hybrid requires CUDA; falling back to block_reference.")
        args.scan_impl = "block_reference"
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

    model = HybridMOGTTransformerForCausalLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=num_heads,
        r=args.rank,
        rope_theta=args.rope_theta,
        pattern=args.hybrid_pattern,
        mogt_layer_fraction=args.mogt_layer_fraction,
        explicit_mogt_layer_indices=args.mogt_layer_indices,
        zero_init_attention_out=args.zero_init_attention_out,
    )
    model.backbone.configure_mogt(
        scan_impl=args.scan_impl,
        connection_impl=args.connection_impl,
        connection_damping=args.connection_damping,
        scan_block_size=args.scan_block_size,
        mogt_residual_scale=residual_scale_for_step(args, 0),
        mogt_ffn_residual_scale=args.mogt_ffn_residual_scale,
        mogt_residual_gate=args.mogt_residual_gate,
        mogt_residual_gate_init=args.mogt_residual_gate_init,
    )
    model.to(device)
    model.backbone.gradient_checkpointing = gradient_checkpointing

    param_count = sum(param.numel() for param in model.parameters())
    optimizer = build_optimizer(model, args)
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
        "model_type": "hybrid_mogt_transformer",
        "variant": build_variant(args),
        "run_name": args.run_name,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": num_heads,
        "rank": args.rank,
        "hybrid_pattern": args.hybrid_pattern,
        "mogt_layer_fraction": args.mogt_layer_fraction,
        "mogt_layer_indices_requested": args.mogt_layer_indices,
        "mogt_layer_count": len(model.backbone.mogt_layer_indices),
        "mogt_layer_indices": sorted(model.backbone.mogt_layer_indices),
        "block_types": list(model.backbone.block_types),
        "rope_theta": args.rope_theta,
        "scan_impl": args.scan_impl,
        "connection_impl": args.connection_impl,
        "connection_damping": args.connection_damping,
        "scan_block_size": args.scan_block_size,
        "mogt_residual_scale": args.mogt_residual_scale,
        "mogt_residual_scale_start": args.mogt_residual_scale_start,
        "mogt_residual_scale_warmup_steps": args.mogt_residual_scale_warmup_steps,
        "mogt_residual_scale_schedule": residual_scale_schedule_name(args),
        "mogt_ffn_residual_scale": args.mogt_ffn_residual_scale,
        "mogt_residual_gate": args.mogt_residual_gate,
        "mogt_residual_gate_init": args.mogt_residual_gate_init,
        "zero_init_attention_out": args.zero_init_attention_out,
        "vocab_size": vocab_size,
        "parameter_count": param_count,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "eval_max_batches_requested": args.eval_max_batches,
        "seed": args.seed,
        "lr": args.lr,
        "mogt_lr_mult": args.mogt_lr_mult,
        "weight_decay": args.weight_decay,
        "loss_chunk_size": loss_chunk_size,
        "gradient_checkpointing": gradient_checkpointing,
        "save_best": args.save_best,
        "save_last": args.save_last,
        "resume_from": args.resume_from or None,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "attention": "torch.scaled_dot_product_attention_in_transformer_layers",
    }

    print("==================================================")
    print(f"Budget baseline: hybrid_mogt_transformer | device={device} | run={args.run_name}")
    print(
        f"config: d_model={args.d_model}, layers={args.num_layers}, heads={num_heads}, "
        f"rank={args.rank}, pattern={args.hybrid_pattern}, mogt_layers={config['mogt_layer_count']}, "
        f"params={param_count/1e6:.2f}M, ctx={args.context_length}, "
        f"batch={args.batch_size}, accum={args.grad_accum_steps}"
    )
    print(
        f"train: max_steps={args.max_steps}, lr={args.lr}, loss_chunk={loss_chunk_size}, "
        f"checkpointing={'on' if gradient_checkpointing else 'off'}, "
        f"mogt={args.scan_impl}/{args.connection_impl}, mogt_lr_mult={args.mogt_lr_mult:g}"
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
    current_mogt_residual_scale = apply_residual_scale_schedule(model, args, global_step)
    best_metrics = None if resume_state is None else resume_state["best_metrics"]
    validation_trace = []
    train_loss = None
    start_time = time.time()
    current_phase = "setup"
    current_epoch = 0
    current_batch_idx = -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if residual_scale_schedule_enabled(args):
        first_step_scale = residual_scale_for_step(args, global_step + 1)
        print(
            f"mogt residual scale schedule: {args.mogt_residual_scale_start:g} -> "
            f"{args.mogt_residual_scale:g} over {args.mogt_residual_scale_warmup_steps} steps; "
            f"initial={current_mogt_residual_scale:g}, first_step={first_step_scale:g}",
            flush=True,
        )

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
                current_mogt_residual_scale = apply_residual_scale_schedule(model, args, global_step + 1)

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
                                "mogt_residual_scale_final_runtime": current_mogt_residual_scale,
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
                                "Hybrid MOGT/Transformer trained with the same GPT-2 tokenizer and WikiText-103 data pipeline as scratch baselines.",
                                "Transformer layers provide dense content mixing; MOGT layers test long-context state transport inside a model-level backbone.",
                                "This is a training-budget matched baseline, not a pretrained quality anchor.",
                            ],
                        }
                        report["standard_report"] = build_lm_standard_report(
                            task="language_modeling",
                            config=config,
                            result=report["result"],
                            device=device,
                            validation_trace=validation_trace,
                            notes="Hybrid MOGT/Transformer budget-matched language-modeling run.",
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
                "mogt_residual_scale_final_runtime": current_mogt_residual_scale,
                "global_step": global_step,
                "failed_phase": current_phase,
                "failed_epoch": current_epoch,
                "failed_batch_idx": current_batch_idx,
                "error": str(exc),
            },
            "validation_trace": validation_trace,
            "notes": [
                "OOM is retained as a useful systems datapoint for hybrid long-context scaling.",
            ],
        }
        report["standard_report"] = build_lm_standard_report(
            task="language_modeling",
            config=config,
            result=report["result"],
            device=device,
            validation_trace=validation_trace,
            notes="Hybrid MOGT/Transformer budget-matched language-modeling run OOM.",
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
                "mogt_residual_scale_final_runtime": current_mogt_residual_scale,
                "global_step": global_step,
                "failed_phase": current_phase,
                "failed_epoch": current_epoch,
                "failed_batch_idx": current_batch_idx,
                "error": str(exc),
            },
            "validation_trace": validation_trace,
            "notes": [
                "OOM is retained as a useful systems datapoint for hybrid long-context scaling.",
            ],
        }
        report["standard_report"] = build_lm_standard_report(
            task="language_modeling",
            config=config,
            result=report["result"],
            device=device,
            validation_trace=validation_trace,
            notes="Hybrid MOGT/Transformer budget-matched language-modeling run OOM.",
        )
        write_report(report_output, report)
        return


if __name__ == "__main__":
    main()
