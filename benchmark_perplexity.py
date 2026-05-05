"""
Exploratory PPL probe.

This script is useful after a real MOGT checkpoint exists. Without a trained
checkpoint, the MOGT branch is skipped to avoid mixing random-init numbers into
language-model comparisons.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

from model_mogt import MOGTForCausalLM
from dataset import get_dataloaders

# To benchmark Perplexity (PPL) = exp( Avg CrossEntropyLoss )

def load_mogt(device, vocab_size, checkpoint_dir="./mogt_checkpoints"):
    model = MOGTForCausalLM(vocab_size=vocab_size, d_model=768, num_layers=12, r=16)
    if os.path.exists(checkpoint_dir):
        ckpts = glob.glob(os.path.join(checkpoint_dir, "mogt_ckpt_*.pt"))
        if ckpts:
            ckpts.sort(key=os.path.getctime)
            latest_ckpt = ckpts[-1]
            print(f"🔄 载入 MOGT 本地训练权重: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model.to(device), True
        else:
             print("⚠️ 未找到 MOGT 权重文件；为避免把随机初始化混进 PPL 对比，本轮将跳过 MOGT 分支。")
    else:
        print("⚠️ 未找到 MOGT 权重工作目录；为避免把随机初始化混进 PPL 对比，本轮将跳过 MOGT 分支。")
    return model.to(device), False

def calculate_ppl(model, val_dl, device, is_huggingface=False, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_dl):
            if i >= max_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if is_huggingface:
                    # HuggingFace 的 CausalLM 内部会将 labels 向右自动错位 1 位进行损失核对。
                    # 因为我们在外围 dataset.py 已经人工做过了 x 和 y 的错位映射，
                    # 此时如果再把 y 喂入 HF 的 labels 就会导致二重错位（Double-Shift）。
                    # 因此，面对纯净的 HF 模型，我们直接把尚未错位的连续实体 x 同步作为 labels 传给它，让它自己内部切分。
                    outputs = model(input_ids=x, labels=x)
                    loss = outputs.loss
                else:
                    # 自研 MOGT，没有暗箱包装，硬核原生的无遮挡输入输出
                    logits, loss = model(x, labels=y)
                    
            total_loss += loss.item()
            total_batches += 1
            
    avg_loss = total_loss / total_batches
    ppl = np.exp(avg_loss)
    return ppl

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print("📊 Exploratory PPL Probe: WikiText-103 Validation")
    print("==================================================")
    print("注意：该脚本当前用于探索性对比，不应替代严格校准过的主线评测。")
    
    # 限制批次数量以避免单卡评测验证集过长，25 个大 Context Batch 足以得到稳定抽样
    MAX_EVAL_BATCHES = 25 
    
    # 获取 WikiText-103 数据加载器 (使用预置的 GPT2Tokenizer)
    _, val_dl, vocab_size = get_dataloaders(context_length=1024, batch_size=4, num_workers=2)
    
    results = {}
    
    # ---------------- 1. 基线阵营：GPT-2 Small ----------------
    print("\n➡️ 降临: OpenAI GPT-2 Small (124M) ...")
    try:
        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        results['GPT-2 Small\n(124M)'] = calculate_ppl(gpt2, val_dl, device, is_huggingface=True, max_batches=MAX_EVAL_BATCHES)
        del gpt2
        torch.cuda.empty_cache()
    except Exception as e:
        print("GPT-2 测试脱机失败:", e)
        
    # ---------------- 2. 基线阵营：Mamba 原生模型 ----------------
    print("\n➡️ 降临: Pytorch Native Mamba (130M) ...")
    try:
        # 显式使用 bfloat16 防 OOM
        mamba = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.bfloat16).to(device)
        results['Mamba\n(130M)'] = calculate_ppl(mamba, val_dl, device, is_huggingface=True, max_batches=MAX_EVAL_BATCHES)
        del mamba
        torch.cuda.empty_cache()
    except Exception as e:
        print("Mamba 测试脱机失败:", e)
        
    # ---------------- 3. 主角阵营：MOGT 自研模型 ----------------
    print("\n➡️ 检查: 本机 MOGT checkpoint ...")
    try:
        mogt, has_checkpoint = load_mogt(device, vocab_size)
        if has_checkpoint:
            results['MOGT\n(130M)'] = calculate_ppl(mogt, val_dl, device, is_huggingface=False, max_batches=MAX_EVAL_BATCHES)
            del mogt
            torch.cuda.empty_cache()
    except Exception as e:
        print("MOGT 测试挂载失败:", e)
        

    print("\n✅ 所有维度的测绘完成！实机对战分数锁定:")
    for name, ppl in results.items():
        print(f" - {name.replace('\n', ' ')}: PPL = {ppl:.2f}")

    # ==================== 最终对决渲染图 ====================
    if not results:
        raise RuntimeError("没有可用的 PPL 结果；请检查模型下载或 checkpoint 状态。")

    plt.figure(figsize=(8, 6))
    
    names = list(results.keys())
    ppls = list(results.values())
    
    # 阵营色彩刻画
    colors = ['#ff7f0e', '#2ca02c', '#d62728'] 
    colors = colors[:len(ppls)]
    
    bars = plt.bar(names, ppls, color=colors, width=0.45, alpha=0.85, edgecolor='black', linewidth=1)
    
    for bar in bars:
        yval = bar.get_height()
        # 悬浮刻下精准困惑度
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(ppls)*0.02), f'{yval:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=13)
        
    plt.title('WikiText-103 Zero/Few-shot Validation Perplexity\n(Lower Score is Better)', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('PPL Score', fontsize=12)
    plt.ylim(0, max(ppls) * 1.15)
    
    # 背景网格辅佐辅助线
    plt.gca().set_axisbelow(True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    out_file = "perplexity_comparison.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"\n📊 跨阵营直方对比 PDF 已印发: {out_file}")

    import json
    with open('perplexity_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({
            "results": {k.replace('\n', ' '): v for k, v in results.items()}
        }, f, indent=4)
    print("💾 已永久留存定格算力数据至 perplexity_comparison.json")
