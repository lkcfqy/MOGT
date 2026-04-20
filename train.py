import os
import torch
import glob
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from dataset import get_dataloaders
from model_mogt import MOGTForCausalLM
from optimizer_mogt import MOGTOptimizer

# ==================== 超参数配置区 ====================
# 【10GB 显存特装版参数】
BATCH_SIZE = 4                  # 单卡 10G/16G 可容纳的真实小块
GRAD_ACCUM_STEPS = 16           # 梯度累加，等效全局 Batch Size = 4*16 = 64
CONTEXT_LENGTH = 2048           # 模型最大吞吐长度 (O(N)加持下这里可以拉很高)
LEARNING_RATE = 3e-4
MAX_EPOCHS = 3

# 切换为 True 时，启动专属莫尔斯耗散拓扑保护，专门应对持续学习 (CL) 场景 (实验二)
USE_MOGT_OPTIMIZER = False 

# 云端 / 本地防御重开存档路径
# 【GCP 注意】：如果你在 Colab 运行，请将此路径改为 '/content/drive/MyDrive/mogt_checkpoints'
CHECKPOINT_DIR = "./mogt_checkpoints"

# 智能识别跑环境
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# ======================================================

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

def save_checkpoint(model, optimizer, epoch, step, loss):
    """ 坚不可摧的持久化写盘 """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"mogt_ckpt_e{epoch}_s{step}.pt")
    # 五件套密封
    torch.save({
        'epoch': epoch,
        'step': step, # Batch 内的编号
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, ckpt_path)
    print(f"💾 [防断线存档] Checkpoint [{ckpt_path}] 已无损落盘！")

def train():
    print(f"==================================================")
    print(f"🔥 启动 MOGT 基座级训练。运算后端侦测: {device}")
    print(f"==================================================")
    
    # 1. 架设无边界数据管道
    train_dl, val_dl, vocab_size = get_dataloaders(
        context_length=CONTEXT_LENGTH, 
        batch_size=BATCH_SIZE, 
        num_workers=2
    )
    
    # 2. 组装 130M 参数量的纯血 MOGT 模型
    model = MOGTForCausalLM(vocab_size=vocab_size, d_model=768, num_layers=12, r=16)
    model.to(device)
    
    # 3. 优化器与显存救星 (自动混合精度缩辅)
    if USE_MOGT_OPTIMIZER:
        optimizer = MOGTOptimizer(model.parameters(), lr=LEARNING_RATE, gamma=0.1)
        print("🛡️ MOGTOptimizer 耗散保护机制已激活！进入抗遗忘警戒状态。")
    else:
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print("⚡ 使用原生 AdamW 引擎，专注单任务极限冲刺。")
        
    scaler = GradScaler(enabled=str(device)=='cuda')
    
    # 4. 极客战术：断点扫描与拦截重开 (Auto-Resume)
    start_epoch = 0
    start_step = 0 # Train DataLoader 内部的偏移
    global_accum_step = 0 # 真实权重更新次数
    
    latest_ckpt = get_latest_checkpoint()
    
    if latest_ckpt:
        print(f"\\n🔄 侦测到历史残存 Checkpoint: {latest_ckpt}")
        print(f"🚀 正在逆向剥离历史数据，即将原地满血重开...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        print(f"✅ 已成功将游标卡入 Epoch {start_epoch}, Inner Step {start_step}！\\n")
    else:
        print("\\n🌱 未检测到先代数据，全参数冷启动...")
        
    # 5. 暴力开采主循环
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # 检测硬件是否支持 bfloat16 最佳高精，否则回退 float16
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        print(f"\\n========== Epoch {epoch+1}/{MAX_EPOCHS} 开卷 ==========")
        
        for batch_idx, (x, y) in enumerate(train_dl):
            
            # 若是断点恢复过来的，必须快进跳过前面已经练完的废料 (Skip Mechanism)
            if epoch == start_epoch and batch_idx <= start_step:
                continue
                
            x, y = x.to(device), y.to(device)
            
            # Context Manager: 将大张量运算卡入 fp16 节省几乎一半的显存与计算时间
            with autocast(enabled=str(device)=='cuda', dtype=amp_dtype):
                logits, loss = model(x, labels=y)
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
                optimizer.zero_grad(set_to_none=True)
                
                global_accum_step += 1
                
                # 日志下探控制台
                if global_accum_step % 5 == 0:
                    real_loss = loss.item() * GRAD_ACCUM_STEPS
                    print(f"♻️ Epoch {epoch+1} | Global Step {global_accum_step} | Loss 困惑度初窥: {real_loss:.4f}")
                
                # 高频存档逻辑 (Colab 免死金牌，比如每两百步存一次盘)
                if global_accum_step % 200 == 0:
                    real_loss = loss.item() * GRAD_ACCUM_STEPS
                    save_checkpoint(model, optimizer, epoch, batch_idx, real_loss)

        # 全 Epoch 剧终大盘存档
        save_checkpoint(model, optimizer, epoch, batch_idx, loss.item() * GRAD_ACCUM_STEPS)

if __name__ == "__main__":
    train()
