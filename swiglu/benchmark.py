import torch
import torch.nn.functional as F
import cutlass
from swiglu_pytorch import swiglu_pytorch
from swiglu_pytorch_compile import swiglu_pytorch_compile
from swiglu_cutedsl import swiglu_cutedsl

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


def benchmark(fn, tokens, hidden_dim, dtype):
    x1 = torch.randn(tokens, hidden_dim, device="cuda", dtype=dtype)
    x2 = torch.randn(tokens, hidden_dim, device="cuda", dtype=dtype)

    ms = time_kernel(fn, x1, x2)

    # 2 reads (x1, x2) + 1 write (output), each of size tokens * hidden_dim
    bytes_accessed = 3 * tokens * hidden_dim * x1.element_size()
    gb_per_s = (bytes_accessed / 1e9) / (ms / 1e3)

    return ms, gb_per_s

if __name__ == "__main__":
    import csv
    from datetime import datetime

    cutlass.cuda.initialize_cuda_context()

    kernels = [
        ("pytorch_eager",   swiglu_pytorch),
        ("pytorch_compile", swiglu_pytorch_compile),
        ("cutedsl",         swiglu_cutedsl),
    ]

    configs = [
        (512,   4096,  torch.bfloat16),
        (2048,  4096,  torch.float32),
        (2048,  4096,  torch.bfloat16),
        (8192,  4096,  torch.bfloat16),
        (8192,  8192,  torch.bfloat16),
        (32768, 8192,  torch.bfloat16),
        (65536, 8192,  torch.bfloat16),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "tokens", "hidden_dim", "dtype", "ms", "gb_per_s"])

        print(f"{'kernel':>10} {'tokens':>8} {'hidden':>8} {'dtype':>10} {'ms':>10} {'GB/s':>10}")
        print("-" * 62)

        for kernel_name, fn in kernels:
            for tokens, hidden_dim, dtype in configs:
                ms, gb_per_s = benchmark(fn, tokens, hidden_dim, dtype)
                dtype_str = str(dtype).split(".")[-1]
                writer.writerow([kernel_name, tokens, hidden_dim, dtype_str, f"{ms:.4f}", f"{gb_per_s:.1f}"])
                print(f"{kernel_name:>10} {tokens:>8} {hidden_dim:>8} {dtype_str:>10} {ms:>10.4f} {gb_per_s:>10.1f}")

    print(f"\nResults saved to {filename}")

