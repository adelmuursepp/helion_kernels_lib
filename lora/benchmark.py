import torch
import torch.nn.functional as F
from lora_pytorch import lora_pytorch
from lora_pytorch_compile import lora_pytorch_compile
from lora_helion_inference import lora_helion
from common import LORA_CONFIGS

WARMUP = 10
REPEAT = 100


def time_kernel(fn, *args):
    for _ in range(WARMUP):
        fn(*args)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(REPEAT):
        fn(*args)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEAT  # ms


def validate(fn, tokens, in_dim, out_dim, rank, dtype, atol=0.1):
    torch.manual_seed(0)
    x   = torch.randn(tokens,  in_dim,  device="cuda", dtype=dtype)
    W   = torch.randn(out_dim, in_dim,  device="cuda", dtype=dtype)
    A   = torch.randn(out_dim, rank,    device="cuda", dtype=dtype)
    B   = torch.randn(rank,    in_dim,  device="cuda", dtype=dtype)

    out = fn(x, W, A, B)

    # Reference: compute in float32 for numerical stability
    expected = (
        x.float() @ W.float().T
        + x.float() @ B.float().T @ A.float().T
    ).to(dtype)

    diff    = (out.float() - expected.float()).abs()
    max_err = diff.max().item()
    n_wrong = (diff >= atol).sum().item()
    passed  = n_wrong == 0
    return passed, max_err, n_wrong


def benchmark(fn, tokens, in_dim, out_dim, rank, dtype):
    x   = torch.randn(tokens,  in_dim,  device="cuda", dtype=dtype)
    W   = torch.randn(out_dim, in_dim,  device="cuda", dtype=dtype)
    A   = torch.randn(out_dim, rank,    device="cuda", dtype=dtype)
    B   = torch.randn(rank,    in_dim,  device="cuda", dtype=dtype)

    ms = time_kernel(fn, x, W, A, B)

    # Reads:  x [tokens, in_dim], W [out_dim, in_dim], A [out_dim, rank], B [rank, in_dim]
    # Write:  out [tokens, out_dim]
    elem = x.element_size()
    bytes_accessed = (
        tokens * in_dim
        + out_dim * in_dim
        + out_dim * rank
        + rank * in_dim
        + tokens * out_dim
    ) * elem
    gb_per_s = (bytes_accessed / 1e9) / (ms / 1e3)

    return ms, gb_per_s


if __name__ == "__main__":
    import csv
    import os
    from datetime import datetime

    kernels = [
        ("pytorch_eager",   lora_pytorch),
        ("pytorch_compile", lora_pytorch_compile),
        ("helion",          lora_helion),
    ]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "tokens", "in_dim", "out_dim", "rank", "dtype", "ms", "gb_per_s", "valid"])

        hdr = f"{'kernel':>20} {'tokens':>8} {'in_dim':>8} {'out_dim':>8} {'rank':>6} {'dtype':>10} {'ms':>10} {'GB/s':>10} {'valid':>8}"
        print(hdr)
        print("-" * len(hdr))

        for kernel_name, fn in kernels:
            for tokens, in_dim, out_dim, rank, dtype in LORA_CONFIGS:
                passed, max_err, n_wrong = validate(fn, tokens, in_dim, out_dim, rank, dtype)
                ms, gb_per_s = benchmark(fn, tokens, in_dim, out_dim, rank, dtype)
                dtype_str  = str(dtype).split(".")[-1]
                valid_str  = "OK" if passed else f"FAIL(max={max_err:.3f},n={n_wrong})"
                writer.writerow([kernel_name, tokens, in_dim, out_dim, rank, dtype_str,
                                 f"{ms:.4f}", f"{gb_per_s:.1f}", valid_str])
                print(f"{kernel_name:>20} {tokens:>8} {in_dim:>8} {out_dim:>8} {rank:>6} {dtype_str:>10} {ms:>10.4f} {gb_per_s:>10.1f} {valid_str:>8}")

    print(f"\nResults saved to {filename}")
