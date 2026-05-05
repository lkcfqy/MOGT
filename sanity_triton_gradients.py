import torch

from model_mogt import MOGTBlock


KEYS = [
    "phi_conn.weight",
    "phi_val.weight",
    "theta_read.weight",
    "ffn.w1.weight",
    "ffn.w2.weight",
    "ffn.w3.weight",
    "norm_out.weight",
]


def collect_grad_report(scan_impl: str, *, device="cuda"):
    block = MOGTBlock(64, 16).to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    block.train()
    block.scan_impl = scan_impl
    if scan_impl == "triton_hybrid":
        block.scan_block_size = 16
        block.scan_block_c = 4

    x = torch.randn(
        1,
        32,
        64,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    try:
        y = block(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        params = dict(block.named_parameters())
        return {
            "status": "ok",
            "grads": {
                key: None if params[key].grad is None else float(params[key].grad.float().abs().mean())
                for key in KEYS
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Triton Gradient Sanity on {device}")

    sequential = collect_grad_report("sequential", device=device)
    print("sequential:", sequential)

    hybrid = collect_grad_report("triton_hybrid", device=device)
    print("triton_hybrid:", hybrid)

    if sequential["status"] != "ok":
        raise RuntimeError("sequential path failed unexpectedly")

    if hybrid["status"] != "ok":
        raise RuntimeError(f"triton_hybrid failed unexpectedly: {hybrid['error']}")

    missing = [key for key, value in hybrid["grads"].items() if value is None]
    if missing:
        raise RuntimeError(f"triton_hybrid unexpectedly missing gradients: {missing}")

    print("✅ triton_hybrid 现在会把梯度传回 phi_conn / phi_val。")


if __name__ == "__main__":
    main()
