import os
import math

TRAIN_PRESETS = {
    "baseline_v1": {
        "env": {
            "MOGT_DISABLE_RESUME": "1",
            "MOGT_SCAN_IMPL": "triton_hybrid",
            "MOGT_CONNECTION_IMPL": "cayley",
            "MOGT_BATCH_SIZE": "1",
            "MOGT_GRAD_ACCUM_STEPS": "8",
            "MOGT_CONTEXT_LENGTH": "32768",
            "MOGT_D_MODEL": "768",
            "MOGT_NUM_LAYERS": "12",
            "MOGT_RANK": "16",
            "MOGT_MAX_EPOCHS": "1",
            "MOGT_MAX_GLOBAL_STEPS": "200",
            "MOGT_NUM_WORKERS": "2",
            "MOGT_EVAL_MAX_BATCHES": "10",
            "MOGT_EVAL_INTERVAL": "50",
            "MOGT_EVAL_AT_END": "1",
            "MOGT_SAVE_BEST": "1",
            "MOGT_SEED": "42",
        },
        "checkpoint_dir_template": "./mogt_checkpoints/baseline_v1_cayley_ctx32768_seed{seed}",
    },
    "baseline_v1_smoke": {
        "env": {
            "MOGT_DISABLE_RESUME": "1",
            "MOGT_SCAN_IMPL": "triton_hybrid",
            "MOGT_CONNECTION_IMPL": "cayley",
            "MOGT_BATCH_SIZE": "1",
            "MOGT_GRAD_ACCUM_STEPS": "8",
            "MOGT_CONTEXT_LENGTH": "32768",
            "MOGT_D_MODEL": "768",
            "MOGT_NUM_LAYERS": "12",
            "MOGT_RANK": "16",
            "MOGT_MAX_EPOCHS": "1",
            "MOGT_MAX_GLOBAL_STEPS": "5",
            "MOGT_NUM_WORKERS": "2",
            "MOGT_EVAL_MAX_BATCHES": "1",
            "MOGT_EVAL_AT_END": "1",
            "MOGT_SAVE_BEST": "1",
            "MOGT_SEED": "42",
        },
        "checkpoint_dir_template": "./mogt_checkpoints/baseline_v1_smoke_cayley_ctx32768_seed{seed}",
    },
}


def apply_train_preset():
    preset_name = os.environ.get("MOGT_RUN_PRESET", "").strip()
    if not preset_name:
        return ""
    if preset_name not in TRAIN_PRESETS:
        valid = ", ".join(sorted(TRAIN_PRESETS))
        raise ValueError(f"MOGT_RUN_PRESET must be one of: {valid}")

    preset = TRAIN_PRESETS[preset_name]
    for key, value in preset["env"].items():
        os.environ.setdefault(key, value)

    if "MOGT_CHECKPOINT_DIR" not in os.environ:
        seed = os.environ.get("MOGT_SEED", preset["env"].get("MOGT_SEED", "42"))
        os.environ["MOGT_CHECKPOINT_DIR"] = preset["checkpoint_dir_template"].format(seed=seed)

    return preset_name


ACTIVE_RUN_PRESET = apply_train_preset()

DEFAULT_ALLOC_CONF = os.environ.get("MOGT_ALLOC_CONF", "expandable_segments:True")
if DEFAULT_ALLOC_CONF:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", DEFAULT_ALLOC_CONF)

import torch
import glob
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from dataset import get_dataloaders
from model_mogt import MOGTForCausalLM
from optimizer_mogt import MOGTOptimizer

# ==================== 超参数配置区 ====================
# 【GCP L4 防显存溢出 (OOM) 安全参数】
# 由于使用了最严谨的加法循环积聚，模型需要记忆巨大的中间时序张量。
BATCH_SIZE = int(os.environ.get("MOGT_BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.environ.get("MOGT_GRAD_ACCUM_STEPS", "16"))
CONTEXT_LENGTH = int(os.environ.get("MOGT_CONTEXT_LENGTH", "2048"))
LEARNING_RATE = float(os.environ.get("MOGT_LR", "3e-4"))
MAX_EPOCHS = int(os.environ.get("MOGT_MAX_EPOCHS", "3"))
D_MODEL = int(os.environ.get("MOGT_D_MODEL", "768"))
NUM_LAYERS = int(os.environ.get("MOGT_NUM_LAYERS", "12"))
RANK = int(os.environ.get("MOGT_RANK", "16"))
NUM_WORKERS = int(os.environ.get("MOGT_NUM_WORKERS", "2"))
MAX_GLOBAL_STEPS = int(os.environ.get("MOGT_MAX_GLOBAL_STEPS", "0"))
SEED = int(os.environ.get("MOGT_SEED", "42"))
SCAN_IMPL = os.environ.get("MOGT_SCAN_IMPL", "sequential")
SCAN_BLOCK_SIZE = int(os.environ.get("MOGT_SCAN_BLOCK_SIZE", "256"))
SCAN_BLOCK_C = int(os.environ.get("MOGT_SCAN_BLOCK_C", "32"))
BLOCK_CARRY_SCAN = os.environ.get("MOGT_BLOCK_CARRY_SCAN", "auto")
CONNECTION_IMPL = os.environ.get("MOGT_CONNECTION_IMPL", "matrix_exp")
CONNECTION_DAMPING = float(os.environ.get("MOGT_CONNECTION_DAMPING", "0.999"))
DISABLE_RESUME = os.environ.get("MOGT_DISABLE_RESUME", "0") == "1"
DEFAULT_LOSS_CHUNK_SIZE = "256" if CONTEXT_LENGTH >= 32768 else "4096"
LOSS_CHUNK_SIZE = int(os.environ.get("MOGT_LOSS_CHUNK_SIZE", DEFAULT_LOSS_CHUNK_SIZE))
GRADIENT_CHECKPOINTING = os.environ.get(
    "MOGT_GRADIENT_CHECKPOINTING",
    "1" if CONTEXT_LENGTH >= 32768 else "0",
) == "1"
EVAL_MAX_BATCHES = int(os.environ.get("MOGT_EVAL_MAX_BATCHES", "0"))
EVAL_INTERVAL = int(os.environ.get("MOGT_EVAL_INTERVAL", "0"))
EVAL_AT_END = os.environ.get("MOGT_EVAL_AT_END", "1" if EVAL_MAX_BATCHES > 0 else "0") == "1"
SAVE_BEST = os.environ.get("MOGT_SAVE_BEST", "1" if EVAL_MAX_BATCHES > 0 else "0") == "1"

# 切换为 True 时，启动专属莫尔斯耗散拓扑保护，专门应对持续学习 (CL) 场景 (实验二)
USE_MOGT_OPTIMIZER = False

# 云端 / 本地防御重开存档路径
# 【GCP 注意】：如果你在 Colab 运行，请将此路径改为 '/content/drive/MyDrive/mogt_checkpoints'
CHECKPOINT_DIR = os.environ.get("MOGT_CHECKPOINT_DIR", "./mogt_checkpoints")

# 智能识别跑环境
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# ======================================================


def resolve_block_carry_scan(scan_impl: str, context_length: int, requested: str) -> str:
    if requested not in {"auto", "sequential", "doubling"}:
        raise ValueError("MOGT_BLOCK_CARRY_SCAN must be one of: auto, sequential, doubling")
    if requested != "auto":
        return requested
    if scan_impl == "triton_hybrid" and context_length >= 32768:
        return "doubling"
    return "sequential"

def get_latest_checkpoint():
    """ 轮询本地存档目录，提取最近时刻的 Checkpoint """
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        return None
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "mogt_ckpt_*.pt"))
    if not ckpts:
        return None
    # 按照文件创建时间排序，抓取最新的一条神级存档
    ckpts.sort(key=os.path.getctime)
    return ckpts[-1]

def save_checkpoint(model, optimizer, epoch, step, loss, *, filename=None, extra_state=None):
    """ 坚不可摧的持久化写盘 """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_name = filename or f"mogt_ckpt_e{epoch}_s{step}.pt"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    # 五件套密封
    payload = {
        'epoch': epoch,
        'step': step, # Batch 内的编号
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, ckpt_path)
    print(f"💾 [防断线存档] Checkpoint [{ckpt_path}] 已无损落盘！")


def evaluate_model(model, val_dl, amp_dtype, *, max_batches: int, loss_chunk_size: int):
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

                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=str(device) == 'cuda', dtype=amp_dtype):
                    _, loss = model(x, labels=y, return_logits=False, loss_chunk_size=loss_chunk_size)
                total_loss += float(loss.item())
                total_batches += 1
    finally:
        if was_training:
            model.train()

    if total_batches == 0:
        return None

    avg_loss = total_loss / total_batches
    ppl = math.exp(avg_loss)
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "num_batches": total_batches,
    }

def train():
    print(f"==================================================")
    print(f"🔥 启动 MOGT 基座级训练。运算后端侦测: {device}")
    print(f"==================================================")
    if ACTIVE_RUN_PRESET:
        print(f"🧬 Run preset: {ACTIVE_RUN_PRESET}")
    if device == "cuda" and os.environ.get("PYTORCH_ALLOC_CONF"):
        print(f"🧠 CUDA allocator 配置: {os.environ['PYTORCH_ALLOC_CONF']}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"🎯 全局随机种子已设定: {SEED}")

    # 1. 架设无边界数据管道
    train_dl, val_dl, vocab_size = get_dataloaders(
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # 2. 组装 130M 参数量的纯血 MOGT 模型
    model = MOGTForCausalLM(vocab_size=vocab_size, d_model=D_MODEL, num_layers=NUM_LAYERS, r=RANK)
    model.to(device)
    model.mogt.gradient_checkpointing = GRADIENT_CHECKPOINTING
    resolved_block_carry_scan = resolve_block_carry_scan(SCAN_IMPL, CONTEXT_LENGTH, BLOCK_CARRY_SCAN)
    for block in model.mogt.blocks:
        block.scan_impl = SCAN_IMPL
        block.scan_block_size = SCAN_BLOCK_SIZE
        block.scan_block_c = SCAN_BLOCK_C
        block.block_carry_scan = resolved_block_carry_scan
        block.connection_impl = CONNECTION_IMPL
        block.connection_damping = CONNECTION_DAMPING
    print(
        f"🧱 模型配置: d_model={D_MODEL}, layers={NUM_LAYERS}, rank={RANK}, "
        f"batch={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}, ctx={CONTEXT_LENGTH}"
    )
    print(
        f"🧭 Scan 配置: impl={SCAN_IMPL}, "
        f"block_size={SCAN_BLOCK_SIZE}, block_c={SCAN_BLOCK_C}, carry={resolved_block_carry_scan}"
    )
    print(f"🧲 Connection 配置: impl={CONNECTION_IMPL}, damping={CONNECTION_DAMPING:.6f}")
    print(f"🪓 Loss 配置: chunk_size={LOSS_CHUNK_SIZE}")
    print(f"🧷 Activation Checkpointing: {'on' if GRADIENT_CHECKPOINTING else 'off'}")
    if EVAL_MAX_BATCHES > 0:
        print(
            f"🧪 Eval 配置: max_batches={EVAL_MAX_BATCHES}, "
            f"interval={EVAL_INTERVAL or 'epoch_end'}, save_best={'on' if SAVE_BEST else 'off'}"
        )
    if MAX_GLOBAL_STEPS > 0:
        print(f"🏁 本次短程回归将在 Global Step {MAX_GLOBAL_STEPS} 后自动停止。")

    # 3. 优化器与显存救星 (自动混合精度缩辅)
    if USE_MOGT_OPTIMIZER:
        optimizer = MOGTOptimizer(model.parameters(), lr=LEARNING_RATE, gamma=0.1)
        print("🛡️ MOGTOptimizer 耗散保护机制已激活！进入抗遗忘警戒状态。")
    else:
        # LLM 的标准配置是加上一定量的 Weight Decay 以作正则化
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print("⚡ 使用原生 AdamW 引擎，专注单任务极限冲刺。")

    scaler = torch.amp.GradScaler('cuda', enabled=str(device)=='cuda')

    # 🏅 黄金搭档：加入预训练必备的余弦退火学习率调度器 (带 Warmup)
    from transformers import get_cosine_schedule_with_warmup
    total_steps = (len(train_dl) * MAX_EPOCHS) // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * 0.05) # 5% 的预热
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"📉 余弦调度器部署完成！总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    # 4. 极客战术：断点扫描与拦截重开 (Auto-Resume)
    start_epoch = 0
    start_step = 0 # Train DataLoader 内部的偏移
    global_accum_step = 0 # 真实权重更新次数
    best_val_loss = float("inf")

    latest_ckpt = None if DISABLE_RESUME else get_latest_checkpoint()

    if latest_ckpt:
        print(f"\n🔄 侦测到历史残存 Checkpoint: {latest_ckpt}")
        print(f"🚀 正在逆向剥离历史数据，即将原地满血重开...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))

        # [热补丁] 逆向推算真实的全局步数，修复控制台显示的“表面重开”假象
        global_accum_step = ((start_epoch * len(train_dl)) + start_step + 1) // GRAD_ACCUM_STEPS

        # [热补丁] 强行使学习率空转，跨越那些已经走过的旧岁月，无缝衔接刚才断掉的 LR 曲线！
        for _ in range(global_accum_step):
            scheduler.step()

        print(f"✅ 已成功将游标卡入 Epoch {start_epoch}, Inner Step {start_step}！(真实总进度已较准为: {global_accum_step})\n")
    else:
        print("\n🌱 未检测到先代数据，全参数冷启动...")
        if DISABLE_RESUME:
            print("🧼 已禁用自动续训，当前 run 将忽略旧 checkpoint。")

    # 5. 暴力开采主循环
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # 检测硬件是否支持 bfloat16 最佳高精，否则回退 float16
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    for epoch in range(start_epoch, MAX_EPOCHS):
        # 🛡️ 锁定洗牌随机种子，确保断点续连时 DataLoader 能洗出完全一致的牌序，完美跳过重复废料！
        torch.manual_seed(42 + epoch)
        print(f"\n========== Epoch {epoch+1}/{MAX_EPOCHS} 开卷 ==========")

        for batch_idx, (x, y) in enumerate(train_dl):

            # 若是断点恢复过来的，必须快进跳过前面已经练完的废料 (Skip Mechanism)
            if epoch == start_epoch and batch_idx <= start_step:
                continue

            x, y = x.to(device), y.to(device)

            # Context Manager: 将大张量运算卡入 fp16 节省几乎一半的显存与计算时间
            with torch.amp.autocast('cuda', enabled=str(device)=='cuda', dtype=amp_dtype):
                _, loss = model(x, labels=y, return_logits=False, loss_chunk_size=LOSS_CHUNK_SIZE)
                # 累加法则：使得模型在求导时误以为是全大 Batch 的平滑梯面
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            # 满足设定界限，发起猛烈真实的参数冲洗
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # 为了防止梯度炸穿造成 nan，引入梯度截断 (Clip)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step() # 🚀 将每步的动态学习率下发
                optimizer.zero_grad(set_to_none=True)

                global_accum_step += 1

                # 日志立刻下探控制台 (解除漫长等待，强制每步打印)
                if global_accum_step >= 1:
                    real_loss = loss.item() * GRAD_ACCUM_STEPS
                    print(f"♻️ Epoch {epoch+1} | Global Step {global_accum_step} | Loss 困惑度初窥: {real_loss:.4f}", flush=True)

                should_run_eval = (
                    EVAL_MAX_BATCHES > 0
                    and EVAL_INTERVAL > 0
                    and global_accum_step % EVAL_INTERVAL == 0
                )
                ran_eval_this_step = False
                if should_run_eval:
                    eval_metrics = evaluate_model(
                        model,
                        val_dl,
                        amp_dtype,
                        max_batches=EVAL_MAX_BATCHES,
                        loss_chunk_size=LOSS_CHUNK_SIZE,
                    )
                    if eval_metrics is not None:
                        print(
                            f"🧪 Val | Global Step {global_accum_step} | "
                            f"Loss={eval_metrics['loss']:.4f} | PPL={eval_metrics['ppl']:.2f} | "
                            f"Batches={eval_metrics['num_batches']}",
                            flush=True,
                        )
                        ran_eval_this_step = True
                        if SAVE_BEST and eval_metrics["loss"] < best_val_loss:
                            best_val_loss = eval_metrics["loss"]
                            save_checkpoint(
                                model,
                                optimizer,
                                epoch,
                                batch_idx,
                                real_loss,
                                filename="mogt_best.pt",
                                extra_state={
                                    "best_val_loss": best_val_loss,
                                    "best_val_ppl": eval_metrics["ppl"],
                                    "best_val_batches": eval_metrics["num_batches"],
                                    "global_accum_step": global_accum_step,
                                },
                            )

                # 高频存档逻辑 (Colab 免死金牌，比如每两百步存一次盘)
                saved_checkpoint_this_step = False
                if global_accum_step % 200 == 0:
                    real_loss = loss.item() * GRAD_ACCUM_STEPS
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        batch_idx,
                        real_loss,
                        extra_state={
                            "best_val_loss": best_val_loss,
                            "global_accum_step": global_accum_step,
                        },
                    )
                    saved_checkpoint_this_step = True

                if MAX_GLOBAL_STEPS > 0 and global_accum_step >= MAX_GLOBAL_STEPS:
                    real_loss = loss.item() * GRAD_ACCUM_STEPS
                    if EVAL_MAX_BATCHES > 0 and EVAL_AT_END and not ran_eval_this_step:
                        eval_metrics = evaluate_model(
                            model,
                            val_dl,
                            amp_dtype,
                            max_batches=EVAL_MAX_BATCHES,
                            loss_chunk_size=LOSS_CHUNK_SIZE,
                        )
                        if eval_metrics is not None:
                            print(
                                f"🧪 Val | Final Step {global_accum_step} | "
                                f"Loss={eval_metrics['loss']:.4f} | PPL={eval_metrics['ppl']:.2f} | "
                                f"Batches={eval_metrics['num_batches']}",
                                flush=True,
                            )
                            if SAVE_BEST and eval_metrics["loss"] < best_val_loss:
                                best_val_loss = eval_metrics["loss"]
                                save_checkpoint(
                                    model,
                                    optimizer,
                                    epoch,
                                    batch_idx,
                                    real_loss,
                                    filename="mogt_best.pt",
                                    extra_state={
                                        "best_val_loss": best_val_loss,
                                        "best_val_ppl": eval_metrics["ppl"],
                                        "best_val_batches": eval_metrics["num_batches"],
                                        "global_accum_step": global_accum_step,
                                    },
                                )

                    if not saved_checkpoint_this_step:
                        save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            batch_idx,
                            real_loss,
                            extra_state={
                                "best_val_loss": best_val_loss,
                                "global_accum_step": global_accum_step,
                            },
                        )
                    print(f"🛑 已达到设定的短程回归上限 Global Step={MAX_GLOBAL_STEPS}，正在收束退出。")
                    return

        if EVAL_MAX_BATCHES > 0 and EVAL_AT_END:
            eval_metrics = evaluate_model(
                model,
                val_dl,
                amp_dtype,
                max_batches=EVAL_MAX_BATCHES,
                loss_chunk_size=LOSS_CHUNK_SIZE,
            )
            if eval_metrics is not None:
                print(
                    f"🧪 Val | Epoch {epoch+1} End | "
                    f"Loss={eval_metrics['loss']:.4f} | PPL={eval_metrics['ppl']:.2f} | "
                    f"Batches={eval_metrics['num_batches']}",
                    flush=True,
                )
                if SAVE_BEST and eval_metrics["loss"] < best_val_loss:
                    best_val_loss = eval_metrics["loss"]
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        batch_idx,
                        loss.item() * GRAD_ACCUM_STEPS,
                        filename="mogt_best.pt",
                        extra_state={
                            "best_val_loss": best_val_loss,
                            "best_val_ppl": eval_metrics["ppl"],
                            "best_val_batches": eval_metrics["num_batches"],
                            "global_accum_step": global_accum_step,
                        },
                    )

        # 全 Epoch 剧终大盘存档
        save_checkpoint(
            model,
            optimizer,
            epoch,
            batch_idx,
            loss.item() * GRAD_ACCUM_STEPS,
            extra_state={
                "best_val_loss": best_val_loss,
                "global_accum_step": global_accum_step,
            },
        )

if __name__ == "__main__":
    train()
