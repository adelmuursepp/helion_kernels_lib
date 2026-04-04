import torch
import torch.nn.functional as F
import cutlass
from swiglu_pytorch import swiglu_pytorch
from swiglu_pytorch_compile import swiglu_pytorch_compile_separate, swiglu_pytorch_compile_stacked
from swiglu_cutedsl import swiglu_cutedsl
from swiglu_cutedsl_pipelined import swiglu_cutedsl_pipelined
from swiglu_helion_inference import swiglu_helion
from common import MATMUL_CONFIGS

WARMUP = 10
REPEAT = 100

def time_kernel(fn, *args):
    for _ in range(WARMUP):
        fn(*args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(REPEAT):
        fn(*args)
    end.record()

    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / REPEAT
    return ms


def validate(fn, tokens, d_model, hidden_dim, dtype):
    torch.manual_seed(0)
    x  = torch.randn(tokens,     d_model,     device="cuda", dtype=dtype)
    w1 = torch.randn(hidden_dim, d_model,     device="cuda", dtype=dtype)
    w2 = torch.randn(hidden_dim, d_model,     device="cuda", dtype=dtype)

    out = fn(x, w1, w2)

    # Reference: pytorch eager in BF16, same precision as all kernels.
    # Pure FP32 reference (x.float() @ w.T.float()) diverges from BF16 kernels
    # at large hidden dims because intermediate BF16 rounding compounds through
    # silu*up, causing large absolute errors even when the kernel is correct.
    expected = swiglu_pytorch(x, w1, w2)

    out_f      = out.float()
    expected_f = expected.float()
    diff       = (out_f - expected_f).abs()
    max_abs    = diff.max().item()
    max_rel    = (diff / (expected_f.abs().clamp(min=1.0))).max().item()
    mean_rel   = (diff / (expected_f.abs().clamp(min=1.0))).mean().item()
    # clamp denominator at 1.0 so near-zero reference values don't blow up
    # relative error. rtol=0.1 accommodates FP accumulation order differences
    # between kernels (cutedsl max_rel ~6.9%, helion mean_rel ~0.27%).
    return max_abs, max_rel, mean_rel


def benchmark(fn, tokens, d_model, hidden_dim, dtype):
    x  = torch.randn(tokens,     d_model, device="cuda", dtype=dtype)
    w1 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)
    w2 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)

    ms = time_kernel(fn, x, w1, w2)

    # reads: x [tokens, d_model], w1 [hidden_dim, d_model], w2 [hidden_dim, d_model]
    # write: out [tokens, hidden_dim]
    bytes_accessed = (tokens * d_model + 2 * hidden_dim * d_model + tokens * hidden_dim) * x.element_size()
    gb_per_s = (bytes_accessed / 1e9) / (ms / 1e3)

    return ms, gb_per_s

if __name__ == "__main__":
    import csv
    import os
    from datetime import datetime

    cutlass.cuda.initialize_cuda_context()

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
                max_abs, max_rel, mean_rel = validate(fn, tokens, d_model, hidden_dim, dtype)
                ms, gb_per_s = benchmark(fn, tokens, d_model, hidden_dim, dtype)
                dtype_str = str(dtype).split(".")[-1]
                writer.writerow([kernel_name, tokens, d_model, hidden_dim, dtype_str, f"{ms:.4f}", f"{gb_per_s:.1f}", f"{max_abs:.6f}", f"{max_rel:.4f}", f"{mean_rel:.6f}"])
                print(f"{kernel_name:>25} {tokens:>8} {d_model:>8} {hidden_dim:>8} {dtype_str:>10} {ms:>10.4f} {gb_per_s:>10.1f} {max_abs:>10.6f} {max_rel:>9.4f} {mean_rel:>10.6f}")

    print(f"\nResults saved to {filename}")

