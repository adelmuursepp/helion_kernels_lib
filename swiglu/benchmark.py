import torch
import triton

# Kernel imports
from swiglu_pytorch import swiglu_pytorch
from swiglu_pytorch_compile import swiglu_pytorch_compile_separate, swiglu_pytorch_compile_stacked
from swiglu_cutedsl import swiglu_cutedsl
from swiglu_cutedsl_pipelined import swiglu_cutedsl_pipelined
from swiglu_helion_inference import swiglu_helion

from common import MATMUL_CONFIGS


def validate(fn, tokens, d_model, hidden_dim, dtype):
    torch.manual_seed(0)
    x  = torch.randn(tokens,     d_model,     device="cuda", dtype=dtype)
    w1 = torch.randn(hidden_dim, d_model,     device="cuda", dtype=dtype)
    w2 = torch.randn(hidden_dim, d_model,     device="cuda", dtype=dtype)

    # Function to benchmark
    out = fn(x, w1, w2)

    # Reference: computation in FP32
    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    gate = xf @ w1f.T
    up = xf @ w2f.T
    # torch.sigmoid operates on fp32
    expected = (gate * torch.sigmoid(gate) * up).float()

    out_f      = out.float()
    diff       = (out_f - expected).abs()
    max_abs    = diff.max().item()
    max_rel    = (diff / (expected.abs().clamp(min=1.0))).max().item()
    mean_rel   = (diff / (expected.abs().clamp(min=1.0))).mean().item()
    # clamp denominator at 1.0 so near-zero reference values don't blow up
    # relative error. rtol=0.1 accommodates FP accumulation order differences
    # between kernels 
    return max_abs, max_rel, mean_rel


def benchmark(fn, tokens, d_model, hidden_dim, dtype):
    x  = torch.randn(tokens,     d_model, device="cuda", dtype=dtype)
    w1 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)
    w2 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)

    ms = triton.testing.do_bench(lambda: fn(x, w1, w2))

    # reads: x [tokens, d_model], w1 [hidden_dim, d_model], w2 [hidden_dim, d_model]
    # write: out [tokens, hidden_dim]
    bytes_accessed = (tokens * d_model + 2 * hidden_dim * d_model + tokens * hidden_dim) * x.element_size()
    gb_per_s = (bytes_accessed / 1e9) / (ms / 1e3)

    return ms, gb_per_s

if __name__ == "__main__":
    import csv
    import os
    from datetime import datetime

    def swiglu_compile_stacked(x, w1, w2):
        W = torch.cat([w1, w2], dim=0)
        return swiglu_pytorch_compile_stacked(x, W)

    kernels = [
        ("pytorch_eager",            swiglu_pytorch),
        ("pytorch_compile_separate", swiglu_pytorch_compile_separate),
        ("pytorch_compile_stacked",  swiglu_compile_stacked),
        ("cutedsl",                  swiglu_cutedsl),
        ("cutedsl_pipelined",        swiglu_cutedsl_pipelined),
        ("helion",                   swiglu_helion),
    ]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "tokens", "d_model", "hidden_dim", "dtype", "ms", "gb_per_s", "max_abs", "max_rel", "mean_rel"])

        print(f"{'kernel':>25} {'tokens':>8} {'d_model':>8} {'hidden':>8} {'dtype':>10} {'ms':>10} {'GB/s':>10} {'max_abs':>10} {'max_rel':>9} {'mean_rel':>10}")
        print("-" * 115)

        for kernel_name, fn in kernels:
            for tokens, d_model, hidden_dim, dtype in MATMUL_CONFIGS:
                dtype_str = str(dtype).split(".")[-1]
                try:
                    max_abs, max_rel, mean_rel = validate(fn, tokens, d_model, hidden_dim, dtype)
                    ms, gb_per_s = benchmark(fn, tokens, d_model, hidden_dim, dtype)
                except FileNotFoundError as e:
                    print(f"{kernel_name:>25} {tokens:>8} {d_model:>8} {hidden_dim:>8} {dtype_str:>10} -- skipped: {e}")
                    continue
                writer.writerow([kernel_name, tokens, d_model, hidden_dim, dtype_str, f"{ms:.4f}", f"{gb_per_s:.1f}", f"{max_abs:.6f}", f"{max_rel:.4f}", f"{mean_rel:.6f}"])
                print(f"{kernel_name:>25} {tokens:>8} {d_model:>8} {hidden_dim:>8} {dtype_str:>10} {ms:>10.4f} {gb_per_s:>10.1f} {max_abs:>10.6f} {max_rel:>9.4f} {mean_rel:>10.6f}")

    print(f"\nResults saved to {filename}")

