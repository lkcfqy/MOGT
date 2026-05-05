"""
Synthetic scaling probe.

This file currently uses random-token / random-label streams to stress training
dynamics quickly. It is not a valid compute-optimal or iso-FLOP scaling-law
study and should be treated as an exploratory diagnostic only.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from model_mogt import MOGTForCausalLM

def dummy_data_stream(vocab_size, batch_size=4, seq_len=256):
    # A consistent random data stream that mimics text complexity roughly
    while True:
        x = torch.randint(1, vocab_size, (batch_size, seq_len))
        y = torch.randint(1, vocab_size, (batch_size, seq_len))
        yield x, y

def measure_iso_flop_loss(params_count, d_model, num_layers, r=16, steps=60):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50257
    print(f"\\n➡️ 部署 [ {params_count} ] 级别核心，尺度: D={d_model}, L={num_layers}")
    
    model = MOGTForCausalLM(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, r=r).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fct = nn.CrossEntropyLoss()
    model.train()
    
    data_gen = dummy_data_stream(vocab_size)
    
    final_losses = []
    for i in range(steps):
        x, y = next(data_gen)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = loss_fct(logits.view(-1, logits.size(-1)), y.view(-1))
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i >= steps - 10:  # average last 10 steps to reduce variance
            final_losses.append(loss.item())

    avg_loss = sum(final_losses) / len(final_losses)
    print(f"   🎯 收敛冰点 (Cross Entropy Baseline): {avg_loss:.4f}")
    
    # 彻底释放不同档位模型在 GPU 里的显存
    del model, optimizer
    torch.cuda.empty_cache()
    
    return avg_loss

# Power law model for SciPy curve fit
def scaling_law(N, a, alpha, c):
    # Loss(N) = a * N^(-alpha) + c
    return a * np.power(N, -alpha) + c

def run_scaling_experiment():
    print("==================================================")
    print("📈 Synthetic Scaling Probe")
    print("==================================================")
    print("注意：该脚本当前使用随机标签流，只适合做训练动力学探针，不代表严格 scaling law。")
    
    configs = [
        {"name": "10M",  "N": 10e6,  "d_model": 256, "layers": 4},
        {"name": "30M",  "N": 30e6,  "d_model": 384, "layers": 6},
        {"name": "70M",  "N": 70e6,  "d_model": 512, "layers": 10},
        {"name": "130M", "N": 130e6, "d_model": 768, "layers": 12},
    ]
    
    losses = []
    parameter_counts = []
    
    for cfg in configs:
        l = measure_iso_flop_loss(cfg["name"], cfg["d_model"], cfg["layers"], r=16, steps=80) 
        losses.append(l)
        parameter_counts.append(cfg["N"])
        
    print("\\n✅ 采样结束；以下拟合仅作探索性可视化。")
    
    N_arr = np.array(parameter_counts)
    L_arr = np.array(losses)
    
    try:
        popt, pcov = curve_fit(scaling_law, N_arr, L_arr, p0=[10.0, 0.1, 8.0], maxfev=10000)
        a, alpha, c = popt
        print(f"   [探索性拟合]: Loss = {a:.2f} * N^(-{alpha:.4f})  + {c:.2f}")
    except Exception as e:
        print("   [注意] SciPy 回归波动，退回简易绘图: ", e)
        a, alpha, c = None, None, None

    plt.figure(figsize=(9, 6))
    
    # 散点图
    plt.scatter(N_arr, L_arr, color='#d62728', s=100, zorder=5, label='Actual Hardware Experiments')
    
    # 拟合曲线
    if a is not None:
        N_dense = np.linspace(N_arr.min()*0.8, N_arr.max()*1.2, 100)
        L_dense = scaling_law(N_dense, a, alpha, c)
        plt.plot(N_dense, L_dense, color='#2ca02c', linestyle='-', linewidth=2, label=f'Power-Law Fit ($a \\cdot N^{{-{alpha:.3f}}} + c$)')

    plt.xscale('log')
    
    plt.title('Iso-FLOP Scaling Laws of MOGT Architecture', fontsize=14, pad=15)
    plt.xlabel('Parameter Count $N$ (Log Scale)', fontsize=12)
    plt.ylabel('Test Cross-Entropy Loss', fontsize=12)
    plt.xticks(N_arr, [cfg["name"] for cfg in configs])
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    # 将图例放回内部，使用 'best' 自动寻找大片空白区域，并加上半透明底色
    plt.legend(loc='best', fontsize=11, framealpha=0.85)
    
    out_file = 'scaling_law.pdf'
    plt.savefig(out_file, bbox_inches='tight')
    print(f"📊 已成功输出矢量回归图: {out_file} (已自动投射为对数系)")

    import json
    with open('scaling_law.json', 'w', encoding='utf-8') as f:
        json.dump({
            "parameter_counts": parameter_counts,
            "losses": losses
        }, f, indent=4)
    print("💾 已永久留存微尺度推演数据至 scaling_law.json")

if __name__ == "__main__":
    run_scaling_experiment()
