import argparse
import json
import torch
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime, timezone
from pathlib import Path

from model_mogt import MOGTBlock
from affine_scan import affine_scan_block_reference, affine_scan_doubling, affine_scan_sequential

try:
    from triton_scan import triton_affine_scan_hybrid, triton_fused_scan
    HAS_TRITON_SCAN = True
except Exception as exc:
    print(f"⚠️ Triton scan 不可用，将跳过 transport-only 基准。错误: {exc}")
    HAS_TRITON_SCAN = False

try:
    from flash_attn import flash_attn_qkvpacked_func
    HAS_FLASH = True
except ImportError:
    print("⚠️ 未检测到 flash_attn，但我们将继续执行对比模拟。请确保在此步骤拥有真实环境以获得严格打分。")
    HAS_FLASH = False

def maybe_synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def prepare_affine_inputs(B, L, D, r=16, device="cuda", dtype=torch.bfloat16):
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    block = MOGTBlock(d_model=D, r=r).to(device).to(dtype)

    x_norm = block.norm_mogt(x)
    A_raw = block.phi_conn(x_norm).view(B, L, r, r)
    A = A_raw - A_raw.transpose(-2, -1)
    U = (torch.matrix_exp((A / r).float()) * 0.999).to(dtype)
    V = block.phi_val(x_norm).view(B, L, r, block.c)
    return U, V


def benchmark_callable(fn, *, device="cuda", warmup=5, iters=20):
    for _ in range(warmup):
        fn()

    maybe_synchronize(device)
    start = time.time()
    for _ in range(iters):
        fn()
    maybe_synchronize(device)
    return (time.time() - start) / iters * 1000


def benchmark_affine_scan(B, L, D, *, r=16, device="cuda", warmup=5, iters=20):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    U, V = prepare_affine_inputs(B, L, D, r=r, device=device, dtype=dtype)
    results = {}
    carry_scan = "doubling" if L >= 32768 else "sequential"

    results["affine_sequential_ms"] = benchmark_callable(
        lambda: affine_scan_sequential(U, V, state_dtype=torch.float32, output_dtype=U.dtype),
        device=device,
        warmup=warmup,
        iters=iters,
    )
    results["affine_parallel_ref_ms"] = benchmark_callable(
        lambda: affine_scan_doubling(U, V, state_dtype=torch.float32, output_dtype=U.dtype),
        device=device,
        warmup=warmup,
        iters=iters,
    )
    results["affine_block_ref_ms"] = benchmark_callable(
        lambda: affine_scan_block_reference(
            U,
            V,
            block_size=256,
            state_dtype=torch.float32,
            output_dtype=U.dtype,
            carry_scan=carry_scan,
        ),
        device=device,
        warmup=warmup,
        iters=iters,
    )
    if HAS_TRITON_SCAN and device.startswith("cuda"):
        results["affine_triton_hybrid_ms"] = benchmark_callable(
            lambda: triton_affine_scan_hybrid(
                U,
                V,
                block_size=256,
                block_c=32,
                carry_scan=carry_scan,
                output_dtype=U.dtype,
            ),
            device=device,
            warmup=warmup,
            iters=iters,
        )

    if HAS_TRITON_SCAN and device.startswith("cuda"):
        results["transport_only_triton_ms"] = benchmark_callable(
            lambda: torch.matmul(triton_fused_scan(U), V),
            device=device,
            warmup=warmup,
            iters=iters,
        )

    return results

def benchmark_flash_attention(B, L, D, num_heads=12, device='cuda', warmup=5, iters=20):
    head_dim = D // num_heads
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    use_flash = HAS_FLASH and device.startswith("cuda")
    # FlashAttention expects format: qkv -> [B, L, 3, num_heads, head_dim]
    qkv = torch.randn(B, L, 3, num_heads, head_dim, device=device, dtype=dtype)
    
    # Warmup
    if use_flash:
        for _ in range(warmup):
            flash_attn_qkvpacked_func(qkv, causal=True)
    else:
        # Fallback to PyTorch SDPA (which integrates FlashAttention inside on Ampere if available)
        q = qkv[:, :, 0].transpose(1, 2) # [B, heads, L, head_dim]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        for _ in range(warmup):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
    maybe_synchronize(device)
    start = time.time()
    for _ in range(iters):
        if use_flash:
            _ = flash_attn_qkvpacked_func(qkv, causal=True)
        else:
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    maybe_synchronize(device)
    t_avg = (time.time() - start) / iters
    return t_avg * 1000 # ms


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the real affine transport operator used by MOGT.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", default="benchmark_runs/throughput_core_operator.json")
    parser.add_argument("--output-pdf", default="benchmark_runs/throughput_core_operator.pdf")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    return parser.parse_args()

if __name__ == "__main__":
    print("==================================================================")
    print("🏁 MOGT Affine Transport vs Attention 核心算子吞吐量测试")
    print("==================================================================")

    args = parse_args()
    B = args.batch_size
    D = args.d_model
    lengths = args.lengths

    affine_results = {
        "affine_sequential_ms": [],
        "affine_parallel_ref_ms": [],
        "affine_block_ref_ms": [],
        "affine_triton_hybrid_ms": [],
        "transport_only_triton_ms": [],
        "attention_ms": [],
    }
    attn_oom_idx = -1

    for L in lengths:
        try:
            result = benchmark_affine_scan(
                B,
                L,
                D,
                r=args.rank,
                device=args.device,
                warmup=args.warmup,
                iters=args.iters,
            )

            for key, value in result.items():
                affine_results[key].append((L, value))

            seq_ms = result["affine_sequential_ms"]
            ref_ms = result["affine_parallel_ref_ms"]
            blk_ms = result["affine_block_ref_ms"]
            print(f"L={L}: affine_seq={seq_ms:.2f}ms, affine_parallel_ref={ref_ms:.2f}ms, affine_block_ref={blk_ms:.2f}ms")
            if "affine_triton_hybrid_ms" in result:
                print(f"      affine_triton_hybrid={result['affine_triton_hybrid_ms']:.2f}ms")
            if "transport_only_triton_ms" in result:
                print(f"      transport_only_triton={result['transport_only_triton_ms']:.2f}ms")
        except torch.cuda.OutOfMemoryError:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                print(f"L={L}: affine operator OOM")
            continue

        if attn_oom_idx == -1:
            try:
                t_attn = benchmark_flash_attention(
                    B,
                    L,
                    D,
                    device=args.device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                affine_results["attention_ms"].append((L, t_attn))
                print(f"      attention={t_attn:.2f}ms")
            except torch.cuda.OutOfMemoryError:
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                attn_oom_idx = lengths.index(L)
                print(f"L={L}: FlashAttention OOM")
        else:
            print(f"L={L}: FlashAttention OOM")

    print("\n✅ 实况测绘结束！正在启用 Matplotlib 降维打击绘图...")
    
    plt.figure(figsize=(9, 6))

    plotted = [
        ("affine_sequential_ms", "#1f77b4", "o", "-", "Affine Scan (Sequential Ref)"),
        ("affine_parallel_ref_ms", "#2ca02c", "^", "-", "Affine Scan (Parallel Ref)"),
        ("affine_block_ref_ms", "#8c564b", "v", "-", "Affine Scan (Block Ref)"),
        ("affine_triton_hybrid_ms", "#17becf", "P", "-", "Affine Scan (Triton Hybrid)"),
        ("transport_only_triton_ms", "#9467bd", "D", "--", "Transport-only Triton (Legacy Proxy)"),
        ("attention_ms", "#ff7f0e", "s", "--", "Attention Core (Flash/SDPA)"),
    ]
    for key, color, marker, linestyle, label in plotted:
        if affine_results[key]:
            xs, ys = zip(*affine_results[key])
            plt.plot(xs, ys, marker=marker, linewidth=2.5, markersize=7, color=color, linestyle=linestyle, label=label)

    if attn_oom_idx != -1 and attn_oom_idx < len(lengths):
        oom_x = lengths[attn_oom_idx]
        plt.axvline(x=oom_x, color='red', linestyle=':', linewidth=2.0, alpha=0.8, label=f'Attention OOM at L={oom_x}')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Affine Transport Core vs Attention Core (Log-Log Scale)', fontsize=14, pad=15)
    plt.xlabel('Sequence Length $L$ (Tokens)', fontsize=12)
    plt.ylabel('Average Forward Time (ms)', fontsize=12)
    plt.xticks(lengths, [str(L) for L in lengths], rotation=45)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=11)
    
    output_pdf = Path(args.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"📊 已成功输出矢量对照对数图: {output_pdf}")

    gpu_name = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.device(args.device))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "core_operator_throughput",
        "schema_version": "mogt-throughput-v1",
        "lengths": lengths,
        "batch_size": B,
        "d_model": D,
        "rank": args.rank,
        "device": args.device,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "has_triton_scan": HAS_TRITON_SCAN,
        "has_flash_attn": HAS_FLASH,
        "warmup": args.warmup,
        "iters": args.iters,
        "results_ms": affine_results,
        "attn_oom_idx": attn_oom_idx,
        "notes": (
            "Core-operator timing. Affine scan excludes connection/value projection "
            "and LM head; attention timing measures FlashAttention/SDPA core only."
        ),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    print(f"💾 已永久留存原始观测数据至 {output_json}")
