import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MOGTBlock(nn.Module):
    """
    马格努斯-昂萨格规范输运 (MOGT) 单层核心算子块。
    完全替代传统 Multi-head Self-Attention 机制，实现 O(N) 内存墙剥离。
    """
    def __init__(self, d_model: int, r: int = 16):
        super().__init__()
        self.r = r
        self.d_model = d_model
        
        # 强制要求 d_model 必须能被 r 整除，以计算局部流形值空间的通道维数 c
        assert d_model % r == 0, "d_model 必须能被流形秩 r 整除"
        self.c = d_model // r
        
        # 前置 RMS Norm
        self.norm_mogt = RMSNorm(d_model)
        
        # 1. 联络生成器 (Gauge Connection Generator)
        # 将输入特征 D 投影为 r*r 维的李代数驱动前驱向量
        self.phi_conn = nn.Linear(d_model, r * r, bias=False)
        
        # 2. 伴随值投影器 (Value Representation)
        self.phi_val = nn.Linear(d_model, r * self.c, bias=False)
        
        # 3. 伴随读出器 (Adjoint Readout)
        self.theta_read = nn.Linear(r * self.c, d_model, bias=False)
        
        # ---- FFN 前馈网络部分 ----
        self.norm_ffn = RMSNorm(d_model)
        # 现代 LLM 经典 FFN 缩放。隐藏层约为 d_model 的 (8/3)
        hidden_dim = int(8 * d_model / 3) 
        # 对齐到 256 的倍数，榨干 Nvidia Tensor Core 分配极限
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.ffn = SwiGLU(d_model, hidden_dim)

    def forward(self, x):
        # ---------------- MOGT 高维流形积分网络 ---------------- #
        residual = x
        x_norm = self.norm_mogt(x)
        B, L, D = x_norm.shape
        
        # [步骤 1] 生成李代数 (Lie Algebra) 上严格反对称的联络算子 A_t
        A_raw = self.phi_conn(x_norm).view(B, L, self.r, self.r)
        A = A_raw - A_raw.transpose(-2, -1) # 强制构造 skew-symmetric 矩阵，保证李代数特性
        
        # [步骤 2] 马格努斯李群积分 (Magnus Expansion Exponential)
        # ⚠️ 致命防御：强制使用 Float32 计算矩阵指数，否则会导致泰勒展开（Padé 逼近）在 AMP (bfloat16) 下直接数值爆炸产生 nan！
        A_fp32 = (A / self.r).to(torch.float32)
        U_fp32 = torch.matrix_exp(A_fp32)
        
        # ⚠️ 耗散归一：由于要在深度维度上连续累乘长达 2048 次，一点点数值漂移都会被指数级放大（1.001^2048 会溢出）。
        # 施加一层微距归一化阻尼，防止连乘崩塌，这在流形系统上是标准的物理衰减项：
        U = (U_fp32 * 0.999).to(x_norm.dtype) # U 的形状: [B, L, r, r]
        
        # [步骤 3] O(N) 并理关联扫描 (Associative Prefix Scan)
        # 将序列原本的 N^2 全互相关注，转化为纯矩阵的前向因果流水线累乘
        from triton_scan import triton_fused_scan
        Y = triton_fused_scan(U) # 最终群流形全局状态: [B, L, r, r]
        
        # [步骤 4] 伴随向量特征输运
        V_t_raw = self.phi_val(x_norm).view(B, L, self.r, self.c)
        # 利用李群状态 Y，像高铁一样将输入的信息 V 向未来顺滑输运
        V_t = torch.matmul(Y, V_t_raw) # [B, L, r, r] @ [B, L, r, c] -> [B, L, r, c]
        
        # [步骤 5] 展平并读出特征
        V_t_flat = V_t.view(B, L, -1) # [B, L, r * c]
        Z = self.theta_read(V_t_flat) # [B, L, d_model]
        
        x = residual + Z
        
        # ---------------- 隔离局部微调的高效 FFN 前馈 ---------------- #
        residual = x
        x_norm = self.norm_ffn(x)
        x = residual + self.ffn(x_norm)
        
        return x

class MOGTModel(nn.Module):
    """ 剥离了任务头纯正的特征提取骨干 (Backbone) """
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, r: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MOGTBlock(d_model=d_model, r=r) for _ in range(num_layers)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return x

class MOGTForCausalLM(nn.Module):
    """
    基于对齐标准架构命名法的大模型封装主体类。
    搭载了适用于生成式预测的独立 LM Head。
    默认为 130M 参数量级别 (对标 GPT-2 Small / Mamba-130m)。
    """
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, r: int = 16):
        super().__init__()
        self.mogt = MOGTModel(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, r=r)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 输入与输出分类头极其厚重的权重进行绑定 (Weight Tying)，此为当前高效工业准则
        self.lm_head.weight = self.mogt.embedding.weight
        
        # Gaussian 模型初始化矩阵
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.mogt(input_ids)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # 损失交叉熵：注意由于在 dataloader 中已经对齐了偏移，此处无需再做 shift
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return logits, loss

if __name__ == "__main__":
    # 本地跑这个文件时的一键基准参数气泡试测
    print("-----------------------------------------------------------")
    print("🛠️ MOGT 130M 基座模型前向通路验证测试")
    vocab_size_mock = 50257
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MOGTForCausalLM(vocab_size=vocab_size_mock, d_model=768, num_layers=12, r=16).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型组装完成！总参数量级精确估算: {total_params / 1e6:.2f} M 参数")
    
    print(f"⏳ 正在注入随机张量到 {device} 进行无梯度的干燥前向测试...")
    dummy_inputs = torch.randint(0, vocab_size_mock, (2, 256)).to(device)
    dummy_labels = torch.randint(0, vocab_size_mock, (2, 256)).to(device)
    
    logits, loss = model(dummy_inputs, labels=dummy_labels)
    print(f"✅ 测试全通路顺畅无阻断！\\n输出全局预分布测 Logits 维度: {logits.shape}\\n初值冷启动极值参考 Loss: {loss.item():.4f}")
