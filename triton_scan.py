import torch
import triton
import triton.language as tl

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
