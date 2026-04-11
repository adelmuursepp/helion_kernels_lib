import torch
import triton

from attention_pytorch import attention_pytorch
from attention_pytorch_compile import attention_pytorch_compile
from attention_helion_inference import attention_helion

from common import ATTENTION_CONFIGS

# Disable memory efficient attention so pytorch uses FA2 consistently,
# matching the algorithm used by the Helion kernel for a fair comparison
torch.backends.cuda.enable_mem_efficient_sdp(False)


def validate(fn, batch, num_heads, seq_len, head_dim, dtype):
    torch.manual_seed(0)
    q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    out = fn(q, k, v)

    # Reference: full FP32 computation as ground truth
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / (head_dim ** 0.5)
    attn = torch.softmax(qf @ kf.transpose(-2, -1) * scale, dim=-1)
    expected = (attn @ vf).float()

    out_f = out.float()
    diff = (out_f - expected).abs()
    max_abs  = diff.max().item()
    max_rel  = (diff / expected.abs().clamp(min=1.0)).max().item()
    mean_rel = (diff / expected.abs().clamp(min=1.0)).mean().item()
    return max_abs, max_rel, mean_rel


def benchmark(fn, batch, num_heads, seq_len, head_dim, dtype):
    q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    # do_bench runs warmup iterations then timed iterations, flushing L2 cache
    # between each timed run. return_mode="all" gives per-iteration timings.
    # https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
    times = triton.testing.do_bench(lambda: fn(q, k, v), return_mode="all")
    ms_mean = sum(times) / len(times)
    ms_std  = (sum((t - ms_mean) ** 2 for t in times) / len(times)) ** 0.5

    # reads: q, k, v each [batch, num_heads, seq_len, head_dim]
    # write: out [batch, num_heads, seq_len, head_dim]
    elem = batch * num_heads * seq_len * head_dim
    bytes_accessed = 4 * elem * torch.empty(1, dtype=dtype).element_size()
    gb_per_s = (bytes_accessed / 1e9) / (ms_mean / 1e3)

    return ms_mean, ms_std, gb_per_s


if __name__ == "__main__":
    import csv
    import os
    from datetime import datetime

    kernels = [
        ("pytorch_eager",   attention_pytorch),
        ("pytorch_compile", attention_pytorch_compile),
        ("helion_flash",    attention_helion),
    ]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "batch", "num_heads", "seq_len", "head_dim", "dtype", "ms_mean", "ms_std", "gb_per_s", "max_abs", "max_rel", "mean_rel"])

        print(f"{'kernel':>20} {'batch':>6} {'heads':>6} {'seq_len':>8} {'head_dim':>9} {'dtype':>10} {'ms_mean':>10} {'ms_std':>8} {'GB/s':>10} {'max_abs':>10} {'max_rel':>9} {'mean_rel':>10}")
        print("-" * 125)

        for kernel_name, fn in kernels:
            for batch, num_heads, seq_len, head_dim, dtype in ATTENTION_CONFIGS:
                dtype_str = str(dtype).split(".")[-1]
                try:
                    max_abs, max_rel, mean_rel = validate(fn, batch, num_heads, seq_len, head_dim, dtype)
                    ms_mean, ms_std, gb_per_s = benchmark(fn, batch, num_heads, seq_len, head_dim, dtype)
                except FileNotFoundError as e:
                    print(f"{kernel_name:>20} {batch:>6} {num_heads:>6} {seq_len:>8} {head_dim:>9} {dtype_str:>10} -- skipped: {e}")
                    continue
                writer.writerow([kernel_name, batch, num_heads, seq_len, head_dim, dtype_str, f"{ms_mean:.4f}", f"{ms_std:.4f}", f"{gb_per_s:.1f}", f"{max_abs:.6f}", f"{max_rel:.4f}", f"{mean_rel:.6f}"])
                print(f"{kernel_name:>20} {batch:>6} {num_heads:>6} {seq_len:>8} {head_dim:>9} {dtype_str:>10} {ms_mean:>10.4f} {ms_std:>8.4f} {gb_per_s:>10.1f} {max_abs:>10.6f} {max_rel:>9.4f} {mean_rel:>10.6f}")

    print(f"\nResults saved to {filename}")
