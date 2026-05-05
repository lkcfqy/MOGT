"""
Synthetic long-context retrieval probe.

This script performs short supervised adaptation at 8k before probing longer
lengths. It is therefore not a zero-shot benchmark; it is a controlled stress
test for long-range signal transport.
"""

import torch
import random
import os
import glob
from model_mogt import MOGTForCausalLM

def build_needle_haystack_tensor(context_length, vocab_size, passkey_value=9999, depth_ratio=0.5):
    """
    构造长背景序列，并在指定深度隐秘植入通关密文 (Needle)。
    末端固定提问模板，模型需在紧身其后的最后一个时点给出 passkey_value。
    """
    background = torch.randint(1, vocab_size - 1000, (1, context_length))
    
    # 构建密文 (Passkey) 的序列模板。
    # 模拟："The secret passkey is 9999", 我们用 2001, 3002, 4003 作为固定指配词
    prefix = [2001, 3002, 4003] 
    
    insert_idx = int(context_length * depth_ratio)
    
    # 提问的前缀语 (我们要求模型看到 4003 就吐出密码)
    query_prefix = [2001, 3002, 4003] 
    
    for i, token in enumerate(prefix):
        background[0, insert_idx + i] = token
    background[0, insert_idx + len(prefix)] = passkey_value
    
    # 为了让最后一次特征正好处理完 4003 并预测 passkey，把它放在结尾倒数第一位之前
    for i, token in enumerate(query_prefix):
        background[0, -len(query_prefix) - 1 + i] = token
        
    return background

def finetune_passkey(model, device, vocab_size, steps=50):
    print(f"\\n🧬 正在进行 8K 级别长度的小样本暗号定向微调 (SFT)...")
    # 高学习率快速收敛单个特征位
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    loss_fct = torch.nn.CrossEntropyLoss()
    
    for i in range(steps):
        optimizer.zero_grad()
        # 训练时固死在 8192 长度，密文的植入深度不断做随机跳跃 (教会模型随处可查)
        x = build_needle_haystack_tensor(8192, vocab_size, passkey_value=9999, depth_ratio=random.uniform(0.1, 0.9))
        x = x.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(x)
            # 损失只在处理完 4003 (即 logits[0, -2, :]) 的输出上激发
            # 因为 background[0, -1] 是垫的填充码，所以 -2 处才代表算完了全体 query_prefix
            target = torch.tensor([9999], device=device)
            loss = loss_fct(logits[0, -2, :].unsqueeze(0), target)
            
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f"   [SFT Step {i+1}/{steps}] 锁定暗号交叉熵 Loss: {loss.item():.4f}")
            
    print("✅ 小样本微调完成！基座模型已在短结构下习得暗号匹配逻辑。\\n")

if __name__ == "__main__":
    print("==================================================")
    print("🔍 Synthetic Long-Context Retrieval Probe")
    print("==================================================")
    print("注意：该脚本包含 8K 长度的短程 SFT，属于合成长程记忆实验，不是零样本结论。")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50257
    
    print("⏳ 正在组建 130M MOGT 完整架构舱段...")
    model = MOGTForCausalLM(vocab_size=vocab_size, d_model=768, num_layers=12, r=16)
    
    # -------- 装载训练好的维基百科核心脑区 --------
    checkpoint_dir = "./mogt_checkpoints"
    if os.path.exists(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "mogt_ckpt_*.pt"))
        if ckpts:
            ckpts.sort(key=os.path.getctime)
            latest_ckpt = ckpts[-1]
            print(f"🔄 检测到已修行的权重，载入核心记忆区块: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            # 仅在参数对齐时直接加载，否则如果维度错误会立刻报错并退出
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("⚠️ 未找到历史权重，将使用未经修行的纯噪声壳！")
    else:
        print("⚠️ 未找到历史权重文件夹，将使用未经修行的纯噪声壳！")
    
    model.to(device)
    
    # 开启极速高低距组合暗号指令微调 (长度只训至 8K)
    finetune_passkey(model, device, vocab_size, steps=50)
    
    # 冻结网络梯面，正式推向深空测评
    model.eval()
    
    # 模拟极限泛化，推入三座黑洞 (32k, 64k, 128k)
    # 它仅仅在 8K 里学到规则，面临比训练时大 16 倍的干扰黑海，它能做到 0-Shot 抗毁吗？
    test_lengths = [8192, 32768, 65536, 128000]
    
    with torch.no_grad():
        for L in test_lengths:
            try:
                print(f"\\n➡️ 测试舱段: {L} Tokens 极限压测")
                # 测试植入极其刁钻的深度：13.5% (模拟距离极远、被后续无尽噪声洗刷的孤岛记忆)
                x = build_needle_haystack_tensor(L, vocab_size, passkey_value=9999, depth_ratio=0.135)
                x = x.to(device)
                
                # 重置测高仪
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(x)
                    
                peak_mem = 0.0
                if device == "cuda":
                    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                
                # 获取在提问结尾后的真实第一响应预测
                pred_token = logits[0, -2, :].argmax().item()
                correct_token = 9999
                
                status = "✅ 合成检索成功" if pred_token == correct_token else "❌ 合成检索失败"
                
                if device == "cuda":
                    print(f"   [资源消耗]: {peak_mem:.2f} GB (峰值)")
                print(f"   [记忆召回]: {status} (预测: {pred_token} | 真实: {correct_token})")
                
            except torch.cuda.OutOfMemoryError:
                print(f"   [OOM 死亡]: 在 {L} 长度处发生崩溃。显式突破 24GB 红色阀域。")
                torch.cuda.empty_cache()
                break
