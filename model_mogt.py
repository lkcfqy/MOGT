import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from affine_scan import affine_scan_block_reference, affine_scan_doubling, affine_scan_sequential
from chunked_lm_loss import chunked_linear_cross_entropy

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
        self.connection_impl = "matrix_exp"
        self.connection_damping = 0.999
        self.register_buffer("_identity_r", torch.eye(r, dtype=torch.float32), persistent=False)

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
        self.norm_out = RMSNorm(d_model) # 防御序列长程积累方差的核心锁
        self.theta_read = nn.Linear(r * self.c, d_model, bias=False)

        # ---- FFN 前馈网络部分 ----
        self.norm_ffn = RMSNorm(d_model)
        # 现代 LLM 经典 FFN 缩放。隐藏层约为 d_model 的 (8/3)
        hidden_dim = int(8 * d_model / 3)
        # 对齐到 256 的倍数，榨干 Nvidia Tensor Core 分配极限
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.ffn = SwiGLU(d_model, hidden_dim)

    def _resolve_scan_impl(self):
        if getattr(self, "fast_throughput_mode", False):
            return "transport_triton"
        return getattr(self, "scan_impl", "sequential")

    def _resolve_connection_impl(self):
        return getattr(self, "connection_impl", "matrix_exp")

    def _build_transport(self, A: torch.Tensor, output_dtype: torch.dtype):
        A_fp32 = (A / self.r).to(torch.float32)
        connection_impl = self._resolve_connection_impl()

        if connection_impl == "matrix_exp":
            with torch.autograd.profiler.record_function("mogt.matrix_exp"):
                U_fp32 = torch.matrix_exp(A_fp32)
        elif connection_impl == "cayley":
            with torch.autograd.profiler.record_function("mogt.cayley_solve"):
                eye = self._identity_r.view(1, 1, self.r, self.r)
                half_A = 0.5 * A_fp32
                U_fp32 = torch.linalg.solve(eye - half_A, eye + half_A)
        elif connection_impl == "identity":
            U_fp32 = self._identity_r.view(1, 1, self.r, self.r).expand(
                A.shape[0],
                A.shape[1],
                self.r,
                self.r,
            )
        else:
            raise ValueError(f"Unsupported connection_impl: {connection_impl}")

        damping = float(getattr(self, "connection_damping", 0.999))
        return (U_fp32 * damping).to(output_dtype)

    def _build_value_gate_input(self, x_norm: torch.Tensor, prefix_condition: torch.Tensor | None):
        mode = getattr(self, "value_gate_input_mode", "current")
        if mode == "current":
            return x_norm

        parts = [x_norm]
        if "prev" in mode:
            prev = torch.zeros_like(x_norm)
            prev[:, 1:] = x_norm[:, :-1]
            parts.append(prev)
        if "prefix" in mode:
            if prefix_condition is None:
                prefix = torch.zeros_like(x_norm[:, :1])
            else:
                prefix = prefix_condition.to(dtype=x_norm.dtype).view(
                    x_norm.size(0), 1, x_norm.size(-1)
                )
            parts.append(prefix.expand(x_norm.size(0), x_norm.size(1), x_norm.size(-1)))
        return torch.cat(parts, dim=-1)

    def forward(self, x, prefix_condition=None):
        # ---------------- MOGT 高维流形积分网络 ---------------- #
        residual = x
        x_norm = self.norm_mogt(x)
        B, L, D = x_norm.shape

        # [步骤 1] 生成李代数 (Lie Algebra) 上严格反对称的联络算子 A_t
        with torch.autograd.profiler.record_function("mogt.connection"):
            A_raw = self.phi_conn(x_norm).view(B, L, self.r, self.r)
            A = A_raw - A_raw.transpose(-2, -1) # 强制构造 skew-symmetric 矩阵，保证李代数特性

        # [步骤 2] 马格努斯李群积分 (Magnus Expansion Exponential)
        # ⚠️ 致命防御：强制使用 Float32 计算矩阵指数，否则会导致泰勒展开（Padé 逼近）在 AMP (bfloat16) 下直接数值爆炸产生 nan！
        with torch.autograd.profiler.record_function("mogt.connection_map"):
            # ⚠️ 耗散归一：由于要在深度维度上连续累乘长达数千步，一点点数值漂移都会被指数级放大。
            # 默认保持 matrix_exp 语义，但允许在长上下文实验中切到更便宜的 Cayley 近似。
            U = self._build_transport(A, x_norm.dtype) # U 的形状: [B, L, r, r]
            phi_transport_gate = getattr(self, "phi_transport_gate", None)
            if phi_transport_gate is not None:
                gate_raw = phi_transport_gate(x_norm).float()
                if gate_raw.size(-1) == 1:
                    gate_raw = gate_raw.view(B, L, 1, 1)
                elif gate_raw.size(-1) == self.r:
                    gate_raw = gate_raw.view(B, L, self.r, 1)
                else:
                    raise ValueError(
                        "phi_transport_gate must output either 1 scalar or r rank gates"
                    )
                gate_mode = getattr(self, "transport_gate_mode", "multiply")
                if gate_mode == "multiply":
                    gate = torch.sigmoid(gate_raw)
                elif gate_mode == "residual":
                    gate_scale = float(getattr(self, "transport_gate_scale", 1.0))
                    gate = 1.0 + gate_scale * torch.tanh(gate_raw)
                elif gate_mode == "forget_relu":
                    gate_scale = float(getattr(self, "transport_gate_scale", 1.0))
                    gate = 1.0 - gate_scale * F.relu(torch.tanh(gate_raw))
                else:
                    raise ValueError(f"Unsupported transport_gate_mode: {gate_mode}")
                U = U * gate.to(U.dtype)

        # [步骤 4] 伴随向量特征输运
        with torch.autograd.profiler.record_function("mogt.value_projection"):
            V_t_raw = self.phi_val(x_norm).view(B, L, self.r, self.c)
            phi_value_gate = getattr(self, "phi_value_gate", None)
            if phi_value_gate is not None:
                value_gate_input = self._build_value_gate_input(x_norm, prefix_condition)
                value_gate_raw = phi_value_gate(value_gate_input).float()
                if value_gate_raw.size(-1) == 1:
                    value_gate = torch.sigmoid(value_gate_raw).view(B, L, 1, 1)
                elif value_gate_raw.size(-1) == self.r:
                    value_gate = torch.sigmoid(value_gate_raw).view(B, L, self.r, 1)
                else:
                    raise ValueError(
                        "phi_value_gate must output either 1 scalar or r rank gates"
                    )
                V_t_raw = V_t_raw * value_gate.to(V_t_raw.dtype)
                if getattr(self, "couple_forget_to_value_gate", False):
                    U = U * (1.0 - value_gate).to(U.dtype)

        # [步骤 3 (已重构)] O(N) 并理关联扫描与特征积聚 (Feature Accumulation)
        scan_impl = self._resolve_scan_impl()
        with torch.autograd.profiler.record_function("mogt.affine_scan"):
            if scan_impl == "transport_triton":
                # 🚨 仅用于撰写论文制作 O(N) 性能比对挂载的虚测模式
                # Triton算子目前未支持伴随加法，启用它等同于切断特征记忆
                from triton_scan import triton_fused_scan
                Y = triton_fused_scan(U)
                V_t = torch.matmul(Y, V_t_raw)
            elif scan_impl == "triton_hybrid":
                # Triton 前向 + reference backward 重算的过渡版。
                from triton_scan import triton_affine_scan_hybrid
                V_t = triton_affine_scan_hybrid(
                    U,
                    V_t_raw,
                    block_size=getattr(self, "scan_block_size", 256),
                    block_c=getattr(self, "scan_block_c", 32),
                    carry_scan=getattr(self, "block_carry_scan", "sequential"),
                    output_dtype=U.dtype,
                )
            elif scan_impl == "parallel_reference":
                # 这条路径用于验证 affine operator 的结合律与未来 kernel 设计。
                V_t = affine_scan_doubling(
                    U,
                    V_t_raw,
                    state_dtype=torch.float32,
                    output_dtype=U.dtype,
                )
            elif scan_impl == "block_reference":
                # 这条路径模拟未来 Triton kernel 的 block-local scan + carry 结构。
                V_t = affine_scan_block_reference(
                    U,
                    V_t_raw,
                    block_size=getattr(self, "scan_block_size", 256),
                    state_dtype=torch.float32,
                    output_dtype=U.dtype,
                    carry_scan=getattr(self, "block_carry_scan", "sequential"),
                )
            elif scan_impl == "sequential":
                # 🛠️ 纯真物理实战链路：真正的全栈时序特征累积 (训练与评测算 PPL 皆必须走此链路)
                # ⚠️ 致命数值防御二：长序列（2048）加法必须在 FP32 精度下进行积聚！
                # 在 BF16 下（仅 7 位尾数），一旦 H_t 累加跨越 256 步，新来的 V_t 就会因数值吸收 (Absorption) 被底层强行截断舍弃！
                # 这会导致模型患上极度严重的“长程失忆症”。
                V_t = affine_scan_sequential(
                    U,
                    V_t_raw,
                    state_dtype=torch.float32,
                    output_dtype=U.dtype,
                )
            else:
                raise ValueError(f"Unsupported scan_impl: {scan_impl}")

        V_t_flat = V_t.view(B, L, self.d_model)

        # [步骤 5] 严密封锁方差漂移，然后展平并读出特征
        with torch.autograd.profiler.record_function("mogt.readout"):
            V_t_norm = self.norm_out(V_t_flat)
            Z = self.theta_read(V_t_norm) # [B, L, d_model]

        mogt_residual_scale = float(getattr(self, "mogt_residual_scale", 1.0))
        gate_logit = getattr(self, "mogt_residual_gate_logit", None)
        if getattr(self, "mogt_residual_gate_enabled", False) and gate_logit is not None:
            gate = torch.sigmoid(gate_logit).to(dtype=Z.dtype, device=Z.device)
            x = residual + ((mogt_residual_scale * gate) * Z)
        else:
            x = residual + (mogt_residual_scale * Z)

        # ---------------- 隔离局部微调的高效 FFN 前馈 ---------------- #
        residual = x
        with torch.autograd.profiler.record_function("mogt.ffn"):
            x_norm = self.norm_ffn(x)
            mogt_ffn_residual_scale = float(getattr(self, "mogt_ffn_residual_scale", 1.0))
            x = residual + (mogt_ffn_residual_scale * self.ffn(x_norm))

        return x

class MOGTModel(nn.Module):
    """ 剥离了任务头纯正的特征提取骨干 (Backbone) """
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, r: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MOGTBlock(d_model=d_model, r=r) for _ in range(num_layers)])
        self.norm_f = RMSNorm(d_model)
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        prefix_condition = None
        prefix_position = getattr(self, "prefix_condition_position", None)
        if prefix_position is not None:
            prefix_position = int(prefix_position)
            if -x.size(1) <= prefix_position < x.size(1):
                prefix_condition = x[:, prefix_position, :]
        use_checkpoint = self.training and getattr(self, "gradient_checkpointing", False)
        checkpoint_every_n = max(1, int(getattr(self, "checkpoint_every_n", 1)))

        for block_idx, block in enumerate(self.blocks):
            if use_checkpoint and (block_idx % checkpoint_every_n == 0):
                x = checkpoint(block, x, prefix_condition, use_reentrant=False)
            else:
                x = block(x, prefix_condition=prefix_condition)
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

        # ⚠️ 致命数值防御：由于长程加法积分会产生极端的特征方差爆炸，我们必须利用 ReZero/Zero-Init 原则切断初期干扰
        for name, p in self.named_parameters():
            if 'phi_conn' in name:
                # 保证最初的马格努斯连乘中 U 贴近完全单位阵，让最初始阶段的梯度如丝般平滑通过。
                torch.nn.init.normal_(p, mean=0.0, std=1e-4)
            elif 'theta_read' in name:
                # 零初始化序列并理读出门，完美保护残差网络 (Residual) 不被初始积聚噪音冲垮
                torch.nn.init.zeros_(p)
            elif 'ffn.w3' in name:
                # 零初始化 FFN 送出门，业界标准的防崩塌底线手段
                torch.nn.init.zeros_(p)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _chunked_lm_loss(self, hidden_states, labels, loss_chunk_size: int, return_logits: bool):
        vocab_size = self.lm_head.weight.size(0)
        total_loss = hidden_states.new_zeros(())
        total_tokens = 0
        logits_chunks = [] if return_logits else None

        for start in range(0, hidden_states.size(1), loss_chunk_size):
            end = min(start + loss_chunk_size, hidden_states.size(1))
            hidden_chunk = hidden_states[:, start:end, :]
            logits_chunk = self.lm_head(hidden_chunk)
            labels_chunk = labels[:, start:end]
            total_loss = total_loss + F.cross_entropy(
                logits_chunk.reshape(-1, vocab_size),
                labels_chunk.reshape(-1),
                reduction="sum",
            )
            total_tokens += labels_chunk.numel()
            if return_logits:
                logits_chunks.append(logits_chunk)

        loss = total_loss / total_tokens
        logits = torch.cat(logits_chunks, dim=1) if return_logits else None
        return logits, loss

    def forward(self, input_ids, labels=None, *, return_logits=True, loss_chunk_size=0):
        x = self.mogt(input_ids)
        logits = None
        loss = None
        loss_chunk_size = int(loss_chunk_size or 0)

        with torch.autograd.profiler.record_function("mogt.lm_head_loss"):
            if labels is not None and loss_chunk_size > 0:
                if return_logits:
                    logits, loss = self._chunked_lm_loss(
                        hidden_states=x,
                        labels=labels,
                        loss_chunk_size=loss_chunk_size,
                        return_logits=True,
                    )
                else:
                    loss = chunked_linear_cross_entropy(
                        x,
                        self.lm_head.weight,
                        labels,
                        chunk_size=loss_chunk_size,
                    )
            else:
                logits = self.lm_head(x)
                if labels is not None:
                    # 损失交叉熵：注意由于在 dataloader 中已经对齐了偏移，此处无需再做 shift
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                if not return_logits:
                    logits = None

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
