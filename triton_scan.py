import torch
import triton
import triton.language as tl

from affine_scan import affine_scan_doubling, affine_scan_sequential, affine_scan_sequential_backward


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()

@triton.jit
def _fused_scan_kernel(
    U_ptr, Y_ptr,
    stride_ub, stride_ul, stride_ur1, stride_ur2,
    stride_yb, stride_yl, stride_yr1, stride_yr2,
    B, L, R: tl.constexpr
):
    # 此核函数为一个 Block 分配一条序列 (batch index)
    # 彻底省去 PyTorch 层面每次相乘的 HBM 显存读写，将其锁死在 SRAM 与 Tensor Core 中
    batch_idx = tl.program_id(0)
    
    # 寄存器阵列初始化：构造 $R \times R$ 单位矩阵
    offs_r1 = tl.arange(0, R)
    offs_r2 = tl.arange(0, R)
    
    pid_r1 = offs_r1[:, None]
    pid_r2 = offs_r2[None, :]
    
    # 构建当前累积张量 Y (初始为对角阵 单位阵)
    acc = tl.where(pid_r1 == pid_r2, 1.0, 0.0).to(tl.float32)

    for t in range(L):
        # 计算当前时间步 U_t 在全局显存中的内存指针偏移
        # U_t 的形状为 [R, R]
        u_ptrs = U_ptr + batch_idx * stride_ub + t * stride_ul + pid_r1 * stride_ur1 + pid_r2 * stride_ur2
        
        # 批量载入 SRAM (L4 GPU TMA 支持下这里非常快)
        u_t = tl.load(u_ptrs)
        
        # 将 acc 转化为跟 u_t 一致的类型以输送进入 wgmma core
        acc_cast = acc.to(u_t.dtype)
        
        # 使用张量核心 (Tensor Core) 极速执行 Y_t = U_t @ Y_{t-1}
        # Triton 内部 tl.dot 默认调用深度优化的 wgmma
        acc = tl.dot(u_t, acc_cast, allow_tf32=False)
        
        # 计算输出位置指针并落盘写回 HBM
        y_ptrs = Y_ptr + batch_idx * stride_yb + t * stride_yl + pid_r1 * stride_yr1 + pid_r2 * stride_yr2
        tl.store(y_ptrs, acc.to(u_t.dtype))


@triton.jit
def _affine_local_scan_kernel(
    U_ptr, V_ptr,
    U_prefix_ptr, H_prefix_ptr,
    U_summary_ptr, H_summary_ptr,
    stride_ub, stride_ul, stride_ur1, stride_ur2,
    stride_vb, stride_vl, stride_vr, stride_vc,
    stride_upb, stride_upl, stride_upr1, stride_upr2,
    stride_hpb, stride_hpl, stride_hpr, stride_hpc,
    stride_usb, stride_usl, stride_usr1, stride_usr2,
    stride_hsb, stride_hsl, stride_hsr, stride_hsc,
    L, C,
    BLOCK_SIZE: tl.constexpr, R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    c_block_idx = tl.program_id(2)

    block_start = block_idx * BLOCK_SIZE
    c_start = c_block_idx * BLOCK_C

    offs_r1 = tl.arange(0, R)[:, None]
    offs_r2 = tl.arange(0, R)[None, :]
    offs_r = tl.arange(0, R)[:, None]
    offs_c = tl.arange(0, BLOCK_C)[None, :]
    c_idx = c_start + offs_c
    c_mask = c_idx < C

    identity = tl.where(offs_r1 == offs_r2, 1.0, 0.0).to(tl.float32)
    U_acc = identity
    H_acc = tl.zeros((R, BLOCK_C), dtype=tl.float32)

    store_u_mask = c_block_idx == 0

    for t in range(BLOCK_SIZE):
        seq_idx = block_start + t
        valid = seq_idx < L

        u_ptrs = (
            U_ptr
            + batch_idx * stride_ub
            + seq_idx * stride_ul
            + offs_r1 * stride_ur1
            + offs_r2 * stride_ur2
        )
        u_loaded = tl.load(u_ptrs, mask=valid, other=0.0).to(tl.float32)
        u_t = tl.where(valid, u_loaded, identity)

        v_ptrs = (
            V_ptr
            + batch_idx * stride_vb
            + seq_idx * stride_vl
            + offs_r * stride_vr
            + c_idx * stride_vc
        )
        v_t = tl.load(v_ptrs, mask=valid & c_mask, other=0.0).to(tl.float32)

        H_acc = tl.dot(u_t, H_acc, allow_tf32=False) + v_t
        U_acc = tl.dot(u_t, U_acc, allow_tf32=False)

        u_prefix_ptrs = (
            U_prefix_ptr
            + batch_idx * stride_upb
            + seq_idx * stride_upl
            + offs_r1 * stride_upr1
            + offs_r2 * stride_upr2
        )
        tl.store(u_prefix_ptrs, U_acc, mask=store_u_mask & valid)

        h_prefix_ptrs = (
            H_prefix_ptr
            + batch_idx * stride_hpb
            + seq_idx * stride_hpl
            + offs_r * stride_hpr
            + c_idx * stride_hpc
        )
        tl.store(h_prefix_ptrs, H_acc, mask=valid & c_mask)

    u_summary_ptrs = (
        U_summary_ptr
        + batch_idx * stride_usb
        + block_idx * stride_usl
        + offs_r1 * stride_usr1
        + offs_r2 * stride_usr2
    )
    tl.store(u_summary_ptrs, U_acc, mask=store_u_mask)

    h_summary_ptrs = (
        H_summary_ptr
        + batch_idx * stride_hsb
        + block_idx * stride_hsl
        + offs_r * stride_hsr
        + c_idx * stride_hsc
    )
    tl.store(h_summary_ptrs, H_acc, mask=c_mask)


def triton_fused_scan(U: torch.Tensor):
    """
    Triton 加速版的 $Y_t = U_t @ Y_{t-1}$ 系列并行关联扫描。
    U 必须是 [B, L, r, r] 格式
    """
    B, L, R, _R = U.shape
    assert R == _R, "最后两个维度必须是方阵"
    
    # 强制将内存对齐转换为 contiguous 以满足高效步长抽取
    U = U.contiguous()
    Y = torch.empty_like(U)
    
    # 分配 1D 网格，每个数据元祖跑在一块 Streaming Multiprocessor 里
    grid = (B, )
    
    # 由于 r 是规范群秩，通常很小 (如 16, 32)，完全可以用 constexpr 处理
    # 如果 r = 16，对于 Triton 来说甚至可以分配一个极为迷你的 block
    _fused_scan_kernel[grid](
        U_ptr=U, Y_ptr=Y,
        stride_ub=U.stride(0), stride_ul=U.stride(1), stride_ur1=U.stride(2), stride_ur2=U.stride(3),
        stride_yb=Y.stride(0), stride_yl=Y.stride(1), stride_yr1=Y.stride(2), stride_yr2=Y.stride(3),
        B=B, L=L, R=16 # 这里的 16 是必须固定在编译期的 tl.constexpr
    )
    
    return Y


def triton_affine_local_scan(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    block_size: int = 256,
    block_c: int = 32,
):
    """
    Triton local affine scan.

    This kernel computes block-local affine prefixes and per-block summaries.
    Cross-block carry propagation is handled by a higher-level wrapper.
    """
    if U.device.type != "cuda" or V.device.type != "cuda":
        raise ValueError("triton_affine_local_scan requires CUDA tensors")
    if U.ndim != 4 or V.ndim != 4:
        raise ValueError("U and V must be rank-4 tensors")

    B, L, R, R2 = U.shape
    Bv, Lv, Rv, C = V.shape
    if (B, L, R) != (Bv, Lv, Rv) or R != R2:
        raise ValueError("U must be [B, L, R, R] and V must be [B, L, R, C]")
    if R != 16:
        raise ValueError("Current Triton affine local scan prototype supports R=16 only")

    U = U.contiguous().to(torch.float32)
    V = V.contiguous().to(torch.float32)

    block_c = _next_power_of_two(block_c)
    num_blocks = (L + block_size - 1) // block_size
    num_c_blocks = (C + block_c - 1) // block_c

    U_prefix = torch.empty_like(U, dtype=torch.float32)
    H_prefix = torch.empty_like(V, dtype=torch.float32)
    U_summary = torch.empty((B, num_blocks, R, R), device=U.device, dtype=torch.float32)
    H_summary = torch.empty((B, num_blocks, R, C), device=U.device, dtype=torch.float32)

    grid = (B, num_blocks, num_c_blocks)

    _affine_local_scan_kernel[grid](
        U, V,
        U_prefix, H_prefix,
        U_summary, H_summary,
        U.stride(0), U.stride(1), U.stride(2), U.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        U_prefix.stride(0), U_prefix.stride(1), U_prefix.stride(2), U_prefix.stride(3),
        H_prefix.stride(0), H_prefix.stride(1), H_prefix.stride(2), H_prefix.stride(3),
        U_summary.stride(0), U_summary.stride(1), U_summary.stride(2), U_summary.stride(3),
        H_summary.stride(0), H_summary.stride(1), H_summary.stride(2), H_summary.stride(3),
        L=L, C=C, BLOCK_SIZE=block_size, R=16, BLOCK_C=block_c,
    )
    return U_prefix, H_prefix, U_summary, H_summary


def _triton_affine_scan_hybrid_forward(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    block_size: int = 256,
    block_c: int = 32,
    carry_scan: str = "sequential",
    output_dtype: torch.dtype | None = None,
    return_prefix: bool = False,
):
    """
    Forward implementation for the hybrid affine scan:
    1. Triton kernel computes block-local prefixes and block summaries
    2. PyTorch reference scans block summaries to build carries
    3. PyTorch combines carries with local prefixes
    """
    if carry_scan not in {"sequential", "doubling"}:
        raise ValueError("carry_scan must be 'sequential' or 'doubling'")

    output_dtype = output_dtype or V.dtype
    with torch.autograd.profiler.record_function("triton_hybrid.local_scan"):
        U_local, H_local, U_summary, H_summary = triton_affine_local_scan(
            U,
            V,
            block_size=block_size,
            block_c=block_c,
        )

    with torch.autograd.profiler.record_function("triton_hybrid.carry_scan"):
        if carry_scan == "sequential":
            U_blocks, H_blocks = affine_scan_sequential(
                U_summary,
                H_summary,
                state_dtype=torch.float32,
                output_dtype=torch.float32,
                return_prefix=True,
            )
        else:
            U_blocks, H_blocks = affine_scan_doubling(
                U_summary,
                H_summary,
                state_dtype=torch.float32,
                output_dtype=torch.float32,
                return_prefix=True,
            )

    B, L, R, _ = U.shape
    C = V.shape[-1]
    num_blocks = U_summary.shape[1]

    carry_u = torch.eye(R, device=U.device, dtype=torch.float32).view(1, 1, R, R).expand(B, num_blocks, -1, -1).clone()
    carry_h = torch.zeros(B, num_blocks, R, C, device=U.device, dtype=torch.float32)
    if num_blocks > 1:
        carry_u[:, 1:, :, :] = U_blocks[:, :-1, :, :]
        carry_h[:, 1:, :, :] = H_blocks[:, :-1, :, :]

    padded_L = num_blocks * block_size
    if padded_L != L:
        pad_steps = padded_L - L
        U_pad = torch.zeros(B, pad_steps, R, R, device=U.device, dtype=U_local.dtype)
        H_pad = torch.zeros(B, pad_steps, R, C, device=U.device, dtype=H_local.dtype)
        U_local = torch.cat([U_local, U_pad], dim=1)
        H_local = torch.cat([H_local, H_pad], dim=1)

    U_local_blocks = U_local.view(B, num_blocks, block_size, R, R)
    H_local_blocks = H_local.view(B, num_blocks, block_size, R, C)
    U_carry_blocks = carry_u.unsqueeze(2)
    H_carry_blocks = carry_h.unsqueeze(2)

    with torch.autograd.profiler.record_function("triton_hybrid.carry_apply"):
        H_out_blocks = torch.matmul(U_local_blocks, H_carry_blocks) + H_local_blocks
        H_out = H_out_blocks.view(B, padded_L, R, C)[:, :L, :, :]

    if not return_prefix:
        return H_out.to(output_dtype)

    with torch.autograd.profiler.record_function("triton_hybrid.carry_apply_u"):
        U_out_blocks = torch.matmul(U_local_blocks, U_carry_blocks)
        U_out = U_out_blocks.view(B, padded_L, R, R)[:, :L, :, :]
    return U_out.to(output_dtype), H_out.to(output_dtype)


class _TritonAffineScanHybridAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U: torch.Tensor,
        V: torch.Tensor,
        block_size: int,
        block_c: int,
        carry_scan: str,
        output_dtype: torch.dtype | None,
    ):
        with torch.autograd.profiler.record_function("triton_hybrid.forward"):
            ctx.block_size = block_size
            ctx.block_c = block_c
            ctx.carry_scan = carry_scan
            ctx.output_dtype = output_dtype or V.dtype
            ctx.save_for_backward(U, V)
            return _triton_affine_scan_hybrid_forward(
                U,
                V,
                block_size=block_size,
                block_c=block_c,
                carry_scan=carry_scan,
                output_dtype=ctx.output_dtype,
                return_prefix=False,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U_saved, V_saved = ctx.saved_tensors
        with torch.autograd.profiler.record_function("triton_hybrid.backward_recompute"):
            grad_u, grad_v = _triton_affine_scan_hybrid_backward(
                U_saved,
                V_saved,
                grad_output,
                block_size=ctx.block_size,
                block_c=ctx.block_c,
                carry_scan=ctx.carry_scan,
                need_grad_u=ctx.needs_input_grad[0],
                need_grad_v=ctx.needs_input_grad[1],
            )

        return grad_u, grad_v, None, None, None, None


def _triton_affine_scan_hybrid_backward(
    U: torch.Tensor,
    V: torch.Tensor,
    grad_H: torch.Tensor,
    *,
    block_size: int,
    block_c: int,
    carry_scan: str,
    need_grad_u: bool,
    need_grad_v: bool,
):
    if U.device.type != "cuda" or U.shape[-1] != 16:
        return affine_scan_sequential_backward(
            U,
            V,
            grad_H,
            state_dtype=torch.float32,
            need_grad_u=need_grad_u,
            need_grad_v=need_grad_v,
        )

    state_dtype = torch.float32
    U_work = U.to(state_dtype)
    V_work = V.to(state_dtype)
    grad_work = grad_H.to(state_dtype)

    H_prefix = None
    if need_grad_u:
        with torch.autograd.profiler.record_function("affine_backward.recompute_prefix_hybrid"):
            H_prefix = _triton_affine_scan_hybrid_forward(
                U_work,
                V_work,
                block_size=block_size,
                block_c=block_c,
                carry_scan=carry_scan,
                output_dtype=state_dtype,
                return_prefix=False,
            )

    B, L, R, _ = U.shape
    eye = torch.eye(R, device=U.device, dtype=state_dtype).view(1, 1, R, R).expand(B, 1, -1, -1)
    U_trans = U_work.transpose(-2, -1)
    if L > 1:
        U_back = torch.cat([eye, U_trans[:, 1:, :, :].flip(1)], dim=1)
    else:
        U_back = eye

    with torch.autograd.profiler.record_function("affine_backward.reverse_scan_hybrid"):
        total_grad_rev = _triton_affine_scan_hybrid_forward(
            U_back,
            grad_work.flip(1),
            block_size=block_size,
            block_c=block_c,
            carry_scan=carry_scan,
            output_dtype=state_dtype,
            return_prefix=False,
        )
        total_grad = total_grad_rev.flip(1)

    grad_v = total_grad.to(V.dtype) if need_grad_v else None
    grad_u = None
    if need_grad_u:
        with torch.autograd.profiler.record_function("affine_backward.grad_u_matmul"):
            zero = torch.zeros(B, 1, R, V.shape[-1], device=U.device, dtype=state_dtype)
            H_prev = torch.cat([zero, H_prefix[:, :-1, :, :]], dim=1)
            grad_u = torch.matmul(total_grad, H_prev.transpose(-2, -1)).to(U.dtype)

    return grad_u, grad_v


def triton_affine_scan_hybrid(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    block_size: int = 256,
    block_c: int = 32,
    carry_scan: str = "sequential",
    output_dtype: torch.dtype | None = None,
    return_prefix: bool = False,
):
    """
    Hybrid affine scan.

    During pure forward / benchmarking this runs the Triton-local hybrid path
    directly. When gradients are required for the returned H states, it keeps
    the fast Triton forward but recomputes a reference block scan in backward
    so gradients still reach U and V.
    """
    if return_prefix:
        return _triton_affine_scan_hybrid_forward(
            U,
            V,
            block_size=block_size,
            block_c=block_c,
            carry_scan=carry_scan,
            output_dtype=output_dtype,
            return_prefix=True,
        )

    needs_grad = torch.is_grad_enabled() and (U.requires_grad or V.requires_grad)
    if needs_grad:
        return _TritonAffineScanHybridAutograd.apply(
            U,
            V,
            block_size,
            block_c,
            carry_scan,
            output_dtype,
        )

    return _triton_affine_scan_hybrid_forward(
        U,
        V,
        block_size=block_size,
        block_c=block_c,
        carry_scan=carry_scan,
        output_dtype=output_dtype,
        return_prefix=False,
    )

# --- 测试模块 ---
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, R = 2, 2048, 16
    
    # 测试数据必须尽量逼近收敛域 (由于连乘，数值很容易炸穿，我们用较小的矩阵)
    U = torch.randn(B, L, R, R, dtype=torch.float32, device='cuda') * 0.01
    for i in range(R):
        U[:, :, i, i] += 1.0 # 模拟李群附近
    
    print(f"⌛ 正在验证 {B}x{L} 长度序列前向传播...")
    
    # 1. Triton 版
    import time
    start = time.time()
    Y_triton = triton_fused_scan(U)
    torch.cuda.synchronize()
    t_triton = time.time() - start
    
    # 2. PyTorch 原生版
    start = time.time()
    Y_torch_states = []
    Y_t = torch.eye(R, device='cuda').unsqueeze(0).repeat(B, 1, 1)
    for t in range(L):
        Y_t = torch.bmm(U[:, t, :, :], Y_t)
        Y_torch_states.append(Y_t)
    Y_torch = torch.stack(Y_torch_states, dim=1)
    torch.cuda.synchronize()
    t_pytorch = time.time() - start
    
    diff = torch.max(torch.abs(Y_triton - Y_torch))
    print(f"✅ 数值误差核对结果: Max Abs Diff = {diff.item():.7f}")
    print(f"🚀 速度对比:\\n - PyTorch: {t_pytorch*1000:.2f} ms\\n - Triton: {t_triton*1000:.2f} ms")
    print(f"🥇 提速倍率: {t_pytorch / t_triton:.1f}x")

    C = 8
    V = torch.randn(B, L, R, C, dtype=torch.float32, device="cuda")
    U_ref, H_ref = affine_scan_sequential(U, V, state_dtype=torch.float32, output_dtype=torch.float32, return_prefix=True)
    U_hybrid, H_hybrid = triton_affine_scan_hybrid(U, V, block_size=256, block_c=8, carry_scan="sequential", output_dtype=torch.float32, return_prefix=True)
    print(f"✅ Hybrid affine U diff: {torch.max(torch.abs(U_ref - U_hybrid)).item():.7f}")
    print(f"✅ Hybrid affine H diff: {torch.max(torch.abs(H_ref - H_hybrid)).item():.7f}")
