import copy

import torch

from model_mogt import MOGTForCausalLM


PARAM_KEYS = [
    "mogt.blocks.0.phi_conn.weight",
    "mogt.blocks.0.phi_val.weight",
    "mogt.blocks.0.theta_read.weight",
    "mogt.blocks.0.ffn.w1.weight",
    "mogt.embedding.weight",
]


def build_models(device: str, dtype: torch.dtype):
    torch.manual_seed(1234)
    base = MOGTForCausalLM(vocab_size=257, d_model=64, num_layers=1, r=16).to(device).to(dtype)
    sequential = copy.deepcopy(base)
    hybrid = copy.deepcopy(base)

    sequential.mogt.blocks[0].scan_impl = "sequential"
    hybrid.mogt.blocks[0].scan_impl = "triton_hybrid"
    hybrid.mogt.blocks[0].scan_block_size = 16
    hybrid.mogt.blocks[0].scan_block_c = 4
    hybrid.mogt.blocks[0].block_carry_scan = "doubling"
    return sequential, hybrid


def run_regression():
    if not torch.cuda.is_available():
        print("跳过 triton training regression：当前无 CUDA。")
        return

    device = "cuda"
    dtype = torch.bfloat16
    sequential, hybrid = build_models(device, dtype)

    input_ids = torch.randint(0, 257, (1, 32), device=device)
    labels = torch.randint(0, 257, (1, 32), device=device)

    sequential.train()
    hybrid.train()
    opt_seq = torch.optim.AdamW(sequential.parameters(), lr=1e-3)
    opt_hyb = torch.optim.AdamW(hybrid.parameters(), lr=1e-3)
    opt_seq.zero_grad(set_to_none=True)
    opt_hyb.zero_grad(set_to_none=True)

    logits_seq, loss_seq = sequential(input_ids, labels)
    logits_hyb, loss_hyb = hybrid(input_ids, labels)
    loss_seq.backward()
    loss_hyb.backward()

    logits_max_diff = float((logits_seq.float() - logits_hyb.float()).abs().max().detach())
    loss_diff = float((loss_seq.float() - loss_hyb.float()).abs().detach())
    print(f"logits_max_diff: {logits_max_diff:.3e}")
    print(f"loss_diff: {loss_diff:.3e}")

    max_grad_diff = 0.0
    params_seq = dict(sequential.named_parameters())
    params_hyb = dict(hybrid.named_parameters())
    for key in PARAM_KEYS:
        grad_seq = params_seq[key].grad.float()
        grad_hyb = params_hyb[key].grad.float()
        grad_max_diff = float((grad_seq - grad_hyb).abs().max())
        max_grad_diff = max(max_grad_diff, grad_max_diff)
        print(f"{key} grad_max_diff: {grad_max_diff:.3e}")

    opt_seq.step()
    opt_hyb.step()

    max_param_diff = 0.0
    for key in PARAM_KEYS:
        param_max_diff = float((params_seq[key].float() - params_hyb[key].float()).abs().max().detach())
        max_param_diff = max(max_param_diff, param_max_diff)
        print(f"{key} param_max_diff: {param_max_diff:.3e}")

    if logits_max_diff > 5e-4 or loss_diff > 5e-4 or max_grad_diff > 5e-4 or max_param_diff > 5e-4:
        raise RuntimeError(
            "triton_hybrid training regression exceeded tolerance: "
            f"logits={logits_max_diff:.3e}, loss={loss_diff:.3e}, "
            f"grad={max_grad_diff:.3e}, param={max_param_diff:.3e}"
        )

    print("✅ triton_hybrid 与 sequential 在小型训练回归上保持对齐。")


if __name__ == "__main__":
    run_regression()
