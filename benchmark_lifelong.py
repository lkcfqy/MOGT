import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import copy
from model_mogt import MOGTForCausalLM
from optimizer_mogt import MOGTOptimizer

def create_task_data(vocab_size, pattern_id, batch_size=16, seq_len=128):
    """
    创建一个隔离的任务：
    Task A: 强制回答 pattern_id_A
    Task B: 强制回答 pattern_id_B
    """
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    # 模拟回答目标为特定的 pattern_id
    y = torch.full((batch_size, seq_len), pattern_id, dtype=torch.long)
    return x, y

def compute_fisher_diagonals(model, x, y, device):
    """
    经验 Fisher 信息对角近似提取。
    """
    model.train()
    loss_fct = nn.CrossEntropyLoss()
    
    # 清空旧梯度
    model.zero_grad()
    
    logits, _ = model(x.to(device))
    loss = loss_fct(logits.view(-1, logits.size(-1)), y.to(device).view(-1))
    
    # 计算一阶导
    loss.backward()
    
    fisher_diagonals = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            # 经验 Fisher 信息 = E[(∇L)^2]
            fisher_diagonals[p] = p.grad.detach() ** 2
            
    model.zero_grad()
    return fisher_diagonals

def run_lifelong_experiment():
    print("==================================================")
    print("🛡️ MOGT - 持续学习抗遗忘 (莫尔斯与耗散机制验证) 🛡️")
    print("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50257
    
    print("⏳ 正在初始化 MOGT 小型化实验舱段 (2层，快速跑批)...")
    base_model = MOGTForCausalLM(vocab_size=vocab_size, d_model=256, num_layers=2, r=16).to(device)
    
    # 构建环境
    task_A_id = 999
    task_B_id = 888
    
    x_A, y_A = create_task_data(vocab_size, task_A_id)
    x_B, y_B = create_task_data(vocab_size, task_B_id)
    x_A, y_A = x_A.to(device), y_A.to(device)
    x_B, y_B = x_B.to(device), y_B.to(device)
    
    loss_fct = nn.CrossEntropyLoss()
    
    # 阶段一：纯学习 Task A
    print("\\n➡️ [Stage 1] 让模型沉浸于 Task A (法国历史)，建立记忆底座...")
    optimizer_init = AdamW(base_model.parameters(), lr=1e-3)
    base_model.train()
    for _ in range(50):
        optimizer_init.zero_grad()
        logits, _ = base_model(x_A)
        loss = loss_fct(logits.view(-1, logits.size(-1)), y_A.view(-1))
        loss.backward()
        optimizer_init.step()
        
    # 测试一下 Task A 的原始极高准确率
    base_model.eval()
    with torch.no_grad():
        logits_A, _ = base_model(x_A)
        acc_A_initial = (logits_A.argmax(dim=-1) == y_A).float().mean().item()
    print(f"   🎯 Task A 初始掌握度 (准确率): {acc_A_initial * 100:.2f}%")
    
    # 分轨决斗开始！
    print("\\n➡️ [Stage 2] 双轨对撞测验：强行灌输互斥的 Task B (天体物理学)")
    
    # 分支 1：传统大模型更新 (AdamW) - 预判：灾难性遗忘
    model_adam = copy.deepcopy(base_model)
    opt_adam = AdamW(model_adam.parameters(), lr=1e-3)
    
    # 分支 2：MOGT 优化器更新 - 预判：拓扑长城防线抵抗覆盖
    model_mogt = copy.deepcopy(base_model)
    opt_mogt = MOGTOptimizer(model_mogt.parameters(), lr=1e-3, gamma=0.5, projection_threshold=1e-5)
    
    # 分支 3：EWC (Elastic Weight Consolidation) - 预判：强力抵抗，但刚性太强拖累收敛
    model_ewc = copy.deepcopy(base_model)
    opt_ewc = AdamW(model_ewc.parameters(), lr=1e-3)
    
    print("   ⚓️ 计算 Task A 的 Fisher 收敛盆面，并让 MOGTOptimizer 下锚！")
    fisher_diags = compute_fisher_diagonals(model_mogt, x_A, y_A, device)
    opt_mogt.state['fisher_diagonals'] = fisher_diags  # 这里是底层打入 Fisher 张量
    opt_mogt.anchor_task() # 正式启动保护
    
    print("   ⚓️ 抽取 EWC 的 Fisher 矩阵预备锚点数据！")
    ewc_fisher = compute_fisher_diagonals(model_ewc, x_A, y_A, device)
    ewc_ref_weights = {p: p.clone().detach() for p in model_ewc.parameters() if p.requires_grad}
    ewc_lambda = 5000.0 # EWC penalty 强度
    
    # 记录数组
    steps = list(range(1, 101))
    adam_retention_A = []
    mogt_retention_A = []
    ewc_retention_A = []
    
    for step in steps:
        # Adam 分支：纯无脑学 Task B
        model_adam.train()
        opt_adam.zero_grad()
        logits, _ = model_adam(x_B)
        loss = loss_fct(logits.view(-1, logits.size(-1)), y_B.view(-1))
        loss.backward()
        opt_adam.step()
        
        # MOGT 分支：在约束下学 Task B
        model_mogt.train()
        opt_mogt.zero_grad()
        logits, _ = model_mogt(x_B)
        loss = loss_fct(logits.view(-1, logits.size(-1)), y_B.view(-1))
        loss.backward()
        opt_mogt.step()
        
        # EWC 分支：传统惩罚学 Task B
        model_ewc.train()
        opt_ewc.zero_grad()
        logits, _ = model_ewc(x_B)
        loss_ce = loss_fct(logits.view(-1, logits.size(-1)), y_B.view(-1))
        
        ewc_penalty = 0.0
        for p in model_ewc.parameters():
            if p.requires_grad and p in ewc_fisher:
                ewc_penalty += (ewc_fisher[p] * (p - ewc_ref_weights[p]) ** 2).sum()
        
        loss_ewc = loss_ce + ewc_lambda * ewc_penalty
        loss_ewc.backward()
        opt_ewc.step()
        
        # 定期评测每个模型对旧任务 Task A 的残留准确率
        model_adam.eval()
        model_mogt.eval()
        model_ewc.eval()
        
        with torch.no_grad():
            preds_adam = model_adam(x_A)[0].argmax(dim=-1)
            acc_adam = (preds_adam == y_A[0]).float().mean().item()
            adam_retention_A.append(acc_adam * 100)
            
            preds_mogt = model_mogt(x_A)[0].argmax(dim=-1)
            acc_mogt = (preds_mogt == y_A[0]).float().mean().item()
            mogt_retention_A.append(acc_mogt * 100)
            
            preds_ewc = model_ewc(x_A)[0].argmax(dim=-1)
            acc_ewc = (preds_ewc == y_A[0]).float().mean().item()
            ewc_retention_A.append(acc_ewc * 100)
            
    print("✅ 对撞数据采集完成，正在渲染科研级对比图报...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, adam_retention_A, label='Vanilla AdamW (Catastrophic Forgetting)', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.plot(steps, ewc_retention_A, label='EWC Baseline (Rigid Penalty)', color='#2ca02c', linestyle='-.', linewidth=2)
    plt.plot(steps, mogt_retention_A, label='MOGT Optimizer (Topology Protected)', color='#1f77b4', linewidth=3)
    plt.axhline(acc_A_initial * 100, color='gray', linestyle=':', label='Theoretical Peak')
    
    plt.title('Continual Learning Retention: Task A Acc vs Task B Training Steps', fontsize=14, pad=15)
    plt.xlabel('Training Steps on Conflicting Task B', fontsize=12)
    plt.ylabel('Retention Accuracy of Task A (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    # 将图例放回内部，使用 'best' 自动寻找大片空白区域，并加上半透明底色防死角
    plt.legend(loc='best', fontsize=11, framealpha=0.85)
    
    out_file = 'lifelong_curve.pdf'
    plt.savefig(out_file, bbox_inches='tight')
    print(f"📊 已成功输出矢量图: {out_file} (可直接拖入 LaTeX)")

    import json
    with open('lifelong_curve.json', 'w', encoding='utf-8') as f:
        json.dump({
            "steps": steps,
            "adam_retention_A": adam_retention_A,
            "ewc_retention_A": ewc_retention_A,
            "mogt_retention_A": mogt_retention_A,
            "theoretical_peak": acc_A_initial * 100
        }, f, indent=4)
    print("💾 已永久留存拓扑演进数据至 lifelong_curve.json")

if __name__ == "__main__":
    run_lifelong_experiment()
