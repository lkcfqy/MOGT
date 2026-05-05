import torch

from affine_scan import (
    affine_scan_block_reference,
    affine_scan_doubling,
    affine_scan_sequential,
    affine_scan_sequential_backward,
)


def build_test_case(batch_size=2, seq_len=17, rank=4, channels=3, device="cpu", dtype=torch.float64):
    torch.manual_seed(42)
    U = torch.randn(batch_size, seq_len, rank, rank, device=device, dtype=dtype) * 0.05
    U = U + torch.eye(rank, device=device, dtype=dtype).view(1, 1, rank, rank)
    V = torch.randn(batch_size, seq_len, rank, channels, device=device, dtype=dtype)
    return U, V


def run_affine_scan_sanity(device="cpu"):
    print(f"Affine Scan Sanity Check on {device}")
    U, V = build_test_case(device=device)

    U_seq, H_seq = affine_scan_sequential(
        U,
        V,
        state_dtype=torch.float64,
        output_dtype=torch.float64,
        return_prefix=True,
    )
    U_par, H_par = affine_scan_doubling(
        U,
        V,
        state_dtype=torch.float64,
        output_dtype=torch.float64,
        return_prefix=True,
    )
    U_blk, H_blk = affine_scan_block_reference(
        U,
        V,
        block_size=5,
        state_dtype=torch.float64,
        output_dtype=torch.float64,
        carry_scan="doubling",
        return_prefix=True,
    )

    max_u_diff = float((U_seq - U_par).abs().max())
    max_h_diff = float((H_seq - H_par).abs().max())
    max_u_block_diff = float((U_seq - U_blk).abs().max())
    max_h_block_diff = float((H_seq - H_blk).abs().max())

    print(f"max prefix-U diff: {max_u_diff:.3e}")
    print(f"max prefix-H diff: {max_h_diff:.3e}")
    print(f"max block-U diff: {max_u_block_diff:.3e}")
    print(f"max block-H diff: {max_h_block_diff:.3e}")

    tol = 1e-10
    if max_u_diff > tol or max_h_diff > tol or max_u_block_diff > tol or max_h_block_diff > tol:
        raise RuntimeError("Affine scan reference mismatch exceeded tolerance")

    print("✅ sequential / parallel / block reference 完全对齐。")


def run_triton_hybrid_sanity():
    if not torch.cuda.is_available():
        print("跳过 Triton hybrid sanity：当前无 CUDA。")
        return

    from triton_scan import triton_affine_scan_hybrid

    print("Triton Hybrid Sanity Check on cuda")
    U, V = build_test_case(batch_size=1, seq_len=64, rank=16, channels=8, device="cuda", dtype=torch.float32)
    U_ref, H_ref = affine_scan_sequential(
        U,
        V,
        state_dtype=torch.float32,
        output_dtype=torch.float32,
        return_prefix=True,
    )
    U_hybrid, H_hybrid = triton_affine_scan_hybrid(
        U,
        V,
        block_size=32,
        block_c=8,
        carry_scan="sequential",
        output_dtype=torch.float32,
        return_prefix=True,
    )

    max_u_diff = float((U_ref - U_hybrid).abs().max())
    max_h_diff = float((H_ref - H_hybrid).abs().max())
    print(f"max hybrid-U diff: {max_u_diff:.3e}")
    print(f"max hybrid-H diff: {max_h_diff:.3e}")

    tol = 5e-5
    if max_u_diff > tol or max_h_diff > tol:
        raise RuntimeError("Triton hybrid mismatch exceeded tolerance")

    print("✅ triton_hybrid 与 sequential reference 对齐。")


def run_affine_backward_sanity():
    print("Affine Backward Sanity Check on cpu")
    U, V = build_test_case(seq_len=9, rank=4, channels=3, device="cpu", dtype=torch.float64)
    grad_out = torch.randn_like(V)

    U_ref = U.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    H_ref = affine_scan_sequential(
        U_ref,
        V_ref,
        state_dtype=torch.float64,
        output_dtype=torch.float64,
    )
    grad_u_ref, grad_v_ref = torch.autograd.grad(H_ref, (U_ref, V_ref), grad_outputs=grad_out)

    grad_u, grad_v = affine_scan_sequential_backward(
        U,
        V,
        grad_out,
        state_dtype=torch.float64,
        need_grad_u=True,
        need_grad_v=True,
    )

    max_grad_u_diff = float((grad_u_ref - grad_u).abs().max())
    max_grad_v_diff = float((grad_v_ref - grad_v).abs().max())
    print(f"max grad-U diff: {max_grad_u_diff:.3e}")
    print(f"max grad-V diff: {max_grad_v_diff:.3e}")

    tol = 1e-10
    if max_grad_u_diff > tol or max_grad_v_diff > tol:
        raise RuntimeError("Affine backward mismatch exceeded tolerance")

    print("✅ affine backward 与 autograd reference 对齐。")


if __name__ == "__main__":
    run_affine_scan_sanity()
    run_affine_backward_sanity()
    run_triton_hybrid_sanity()
