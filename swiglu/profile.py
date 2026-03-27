import torch
import torch.nn.functional as F
from swiglu_pytorch import swiglu_pytorch
from swiglu_pytorch_compile import swiglu_pytorch_compile

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


def benchmark(tokens, hidden_dim, dtype):
    x1 = torch.randn(tokens, hidden_dim, device="cuda", dtype=dtype)
    x2 = torch.randn(tokens, hidden_dim, device="cuda", dtype=dtype)

    ms = time_kernel(swiglu, x1, x2)

    # 2 reads (x1, x2) + 1 write (output), each of size tokens * hidden_dim
    bytes_accessed = 3 * tokens * hidden_dim * x1.element_size()
    gb_per_s = (bytes_accessed / 1e9) / (ms / 1e3)

    return ms, gb_per_s

if __name__ == "__main__":
    import csv
    from datetime import datetime

    configs = [
        (2048,  4096,  torch.float32),
        (2048,  4096,  torch.bfloat16),
        (8192,  4096,  torch.bfloat16),
        (8192,  8192,  torch.bfloat16),
        (32768, 8192,  torch.bfloat16),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "tokens", "hidden_dim", "dtype", "ms", "gb_per_s"])

        print(f"{'tokens':>8} {'hidden':>8} {'dtype':>10} {'ms':>10} {'GB/s':>10}")
        print("-" * 52)

        for tokens, hidden_dim, dtype in configs:
            ms, gb_per_s = benchmark(tokens, hidden_dim, dtype)
            dtype_str = str(dtype).split(".")[-1]
            writer.writerow(["naive", tokens, hidden_dim, dtype_str, f"{ms:.4f}", f"{gb_per_s:.1f}"])
            print(f"{tokens:>8} {hidden_dim:>8} {dtype_str:>10} {ms:>10.4f} {gb_per_s:>10.1f}")

    print(f"\nResults saved to {filename}")

