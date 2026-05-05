import torch


def affine_combine(U_next: torch.Tensor, V_next: torch.Tensor, U_prev: torch.Tensor, V_prev: torch.Tensor):
    """
    Combine two affine transport operators:
      (U_next, V_next) ⊗ (U_prev, V_prev) = (U_next @ U_prev, U_next @ V_prev + V_next)
    """
    U_out = torch.matmul(U_next, U_prev)
    V_out = torch.matmul(U_next, V_prev) + V_next
    return U_out, V_out


def affine_scan_sequential(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    state_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    return_prefix: bool = False,
):
    """
    Reference implementation of the causal affine recurrence:
      H_t = U_t @ H_{t-1} + V_t
    """
    if U.ndim != 4 or V.ndim != 4:
        raise ValueError("U and V must be rank-4 tensors")

    B, L, R, R2 = U.shape
    Bv, Lv, Rv, C = V.shape
    if (B, L, R) != (Bv, Lv, Rv) or R != R2:
        raise ValueError("U must be [B, L, R, R] and V must be [B, L, R, C]")

    output_dtype = output_dtype or V.dtype
    U_work = U.to(state_dtype)
    V_work = V.to(state_dtype)

    H_t = torch.zeros(B, R, C, device=U.device, dtype=state_dtype)
    H_states = []

    U_t = None
    U_states = None
    if return_prefix:
        eye = torch.eye(R, device=U.device, dtype=state_dtype)
        U_t = eye.unsqueeze(0).expand(B, -1, -1).clone()
        U_states = []

    for t in range(L):
        U_step = U_work[:, t, :, :]
        H_t = torch.bmm(U_step, H_t) + V_work[:, t, :, :]
        H_states.append(H_t.to(output_dtype))

        if return_prefix:
            U_t = torch.bmm(U_step, U_t)
            U_states.append(U_t.to(output_dtype))

    H_prefix = torch.stack(H_states, dim=1)
    if not return_prefix:
        return H_prefix

    U_prefix = torch.stack(U_states, dim=1)
    return U_prefix, H_prefix


def affine_scan_sequential_backward(
    U: torch.Tensor,
    V: torch.Tensor,
    grad_H: torch.Tensor,
    *,
    state_dtype: torch.dtype = torch.float32,
    need_grad_u: bool = True,
    need_grad_v: bool = True,
):
    """
    Custom backward for the causal affine recurrence:
      H_t = U_t @ H_{t-1} + V_t

    Given upstream gradients for every H_t, this recomputes the forward states
    in state_dtype and performs a reverse scan to produce gradients for U and V.
    """
    if U.ndim != 4 or V.ndim != 4 or grad_H.ndim != 4:
        raise ValueError("U, V and grad_H must be rank-4 tensors")

    B, L, R, R2 = U.shape
    Bv, Lv, Rv, C = V.shape
    Bg, Lg, Rg, Cg = grad_H.shape
    if (B, L, R) != (Bv, Lv, Rv) or R != R2:
        raise ValueError("U must be [B, L, R, R] and V must be [B, L, R, C]")
    if (B, L, R, C) != (Bg, Lg, Rg, Cg):
        raise ValueError("grad_H must match V shape [B, L, R, C]")

    U_work = U.to(state_dtype)
    V_work = V.to(state_dtype)
    grad_work = grad_H.to(state_dtype)

    H_prev_states = None
    if need_grad_u:
        H_prev_states = []

    with torch.autograd.profiler.record_function("affine_backward.recompute_forward"):
        H_prev = torch.zeros(B, R, C, device=U.device, dtype=state_dtype)
        for t in range(L):
            if need_grad_u:
                H_prev_states.append(H_prev)
            H_prev = torch.bmm(U_work[:, t, :, :], H_prev) + V_work[:, t, :, :]

    grad_u = torch.empty_like(U_work) if need_grad_u else None
    grad_v = torch.empty_like(V_work) if need_grad_v else None

    with torch.autograd.profiler.record_function("affine_backward.reverse_scan"):
        carry = torch.zeros(B, R, C, device=U.device, dtype=state_dtype)
        for t in range(L - 1, -1, -1):
            total_grad = grad_work[:, t, :, :] + carry
            if need_grad_v:
                grad_v[:, t, :, :] = total_grad
            if need_grad_u:
                grad_u[:, t, :, :] = torch.matmul(total_grad, H_prev_states[t].transpose(-2, -1))
            carry = torch.bmm(U_work[:, t, :, :].transpose(1, 2), total_grad)

    if grad_u is not None:
        grad_u = grad_u.to(U.dtype)
    if grad_v is not None:
        grad_v = grad_v.to(V.dtype)
    return grad_u, grad_v


def affine_scan_doubling(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    state_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    return_prefix: bool = False,
):
    """
    Parallel-reference inclusive scan using Hillis-Steele style doubling.
    This is not the final high-performance kernel, but it validates the
    associative affine operator without a Python loop over sequence length.
    """
    if U.ndim != 4 or V.ndim != 4:
        raise ValueError("U and V must be rank-4 tensors")

    B, L, R, R2 = U.shape
    Bv, Lv, Rv, C = V.shape
    if (B, L, R) != (Bv, Lv, Rv) or R != R2:
        raise ValueError("U must be [B, L, R, R] and V must be [B, L, R, C]")

    output_dtype = output_dtype or V.dtype
    U_prefix = U.to(state_dtype).clone()
    H_prefix = V.to(state_dtype).clone()

    offset = 1
    while offset < L:
        U_prev = U_prefix.clone()
        H_prev = H_prefix.clone()

        U_prefix[:, offset:, :, :] = torch.matmul(
            U_prev[:, offset:, :, :],
            U_prev[:, :-offset, :, :],
        )
        H_prefix[:, offset:, :, :] = (
            torch.matmul(U_prev[:, offset:, :, :], H_prev[:, :-offset, :, :])
            + H_prev[:, offset:, :, :]
        )
        offset *= 2

    H_out = H_prefix.to(output_dtype)
    if not return_prefix:
        return H_out

    U_out = U_prefix.to(output_dtype)
    return U_out, H_out


def affine_scan_block_reference(
    U: torch.Tensor,
    V: torch.Tensor,
    *,
    block_size: int = 256,
    state_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    carry_scan: str = "sequential",
    return_prefix: bool = False,
):
    """
    Block-structured reference for the future Triton kernel design.

    Each block computes a local prefix scan, then block summaries are scanned to
    produce carries that are applied back to local prefixes.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if carry_scan not in {"sequential", "doubling"}:
        raise ValueError("carry_scan must be 'sequential' or 'doubling'")

    if U.ndim != 4 or V.ndim != 4:
        raise ValueError("U and V must be rank-4 tensors")

    B, L, R, R2 = U.shape
    Bv, Lv, Rv, C = V.shape
    if (B, L, R) != (Bv, Lv, Rv) or R != R2:
        raise ValueError("U must be [B, L, R, R] and V must be [B, L, R, C]")

    output_dtype = output_dtype or V.dtype
    num_blocks = (L + block_size - 1) // block_size

    local_u_prefixes = []
    local_h_prefixes = []
    block_u_summaries = []
    block_h_summaries = []
    block_ranges = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, L)
        block_ranges.append((start, end))

        U_local, H_local = affine_scan_sequential(
            U[:, start:end, :, :],
            V[:, start:end, :, :],
            state_dtype=state_dtype,
            output_dtype=state_dtype,
            return_prefix=True,
        )
        local_u_prefixes.append(U_local)
        local_h_prefixes.append(H_local)
        block_u_summaries.append(U_local[:, -1, :, :])
        block_h_summaries.append(H_local[:, -1, :, :])

    block_u = torch.stack(block_u_summaries, dim=1)
    block_h = torch.stack(block_h_summaries, dim=1)

    if carry_scan == "sequential":
        block_u_inclusive, block_h_inclusive = affine_scan_sequential(
            block_u,
            block_h,
            state_dtype=state_dtype,
            output_dtype=state_dtype,
            return_prefix=True,
        )
    else:
        block_u_inclusive, block_h_inclusive = affine_scan_doubling(
            block_u,
            block_h,
            state_dtype=state_dtype,
            output_dtype=state_dtype,
            return_prefix=True,
        )

    eye = torch.eye(R, device=U.device, dtype=state_dtype).view(1, 1, R, R).expand(B, num_blocks, -1, -1).clone()
    zero = torch.zeros(B, num_blocks, R, C, device=U.device, dtype=state_dtype)
    carry_u = eye
    carry_h = zero
    if num_blocks > 1:
        carry_u[:, 1:, :, :] = block_u_inclusive[:, :-1, :, :]
        carry_h[:, 1:, :, :] = block_h_inclusive[:, :-1, :, :]

    global_u_blocks = []
    global_h_blocks = []
    for block_idx, (start, end) in enumerate(block_ranges):
        U_local = local_u_prefixes[block_idx]
        H_local = local_h_prefixes[block_idx]

        U_carry = carry_u[:, block_idx, :, :].unsqueeze(1)
        H_carry = carry_h[:, block_idx, :, :].unsqueeze(1)

        global_u = torch.matmul(U_local, U_carry)
        global_h = torch.matmul(U_local, H_carry) + H_local

        global_u_blocks.append(global_u)
        global_h_blocks.append(global_h)

    U_out = torch.cat(global_u_blocks, dim=1).to(output_dtype)
    H_out = torch.cat(global_h_blocks, dim=1).to(output_dtype)

    if not return_prefix:
        return H_out
    return U_out, H_out


if __name__ == "__main__":
    torch.manual_seed(42)
    U = torch.randn(2, 17, 4, 4) * 0.05
    U = U + torch.eye(4).view(1, 1, 4, 4)
    V = torch.randn(2, 17, 4, 3)

    U_seq, H_seq = affine_scan_sequential(U, V, state_dtype=torch.float64, output_dtype=torch.float64, return_prefix=True)
    U_par, H_par = affine_scan_doubling(U, V, state_dtype=torch.float64, output_dtype=torch.float64, return_prefix=True)
    U_blk, H_blk = affine_scan_block_reference(
        U,
        V,
        block_size=5,
        state_dtype=torch.float64,
        output_dtype=torch.float64,
        carry_scan="doubling",
        return_prefix=True,
    )

    print("max_prefix_U_diff", float((U_seq - U_par).abs().max()))
    print("max_prefix_H_diff", float((H_seq - H_par).abs().max()))
    print("max_block_U_diff", float((U_seq - U_blk).abs().max()))
    print("max_block_H_diff", float((H_seq - H_blk).abs().max()))
