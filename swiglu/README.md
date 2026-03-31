# SwiGLU Kernel Profiling

Compares three implementations of the SwiGLU activation (`SiLU(x1) * x2`) on H100 GPUs.

## Kernels

| File | Name in results | Description |
|------|----------------|-------------|
| `swiglu_pytorch.py` | `pytorch_eager` | Naive PyTorch — unfused, 5 memory passes |
| `swiglu_pytorch_compile.py` | `pytorch_compile` | `torch.compile` — auto-fused by Inductor |
| `autotune_helion.py` | `helion` | Helion DSL — fused, autotuned tile config |
| `swiglu_cutedsl.py` | `cutedsl` | CuteDSL — handwritten, PTX SiLU via `ex2.approx` |

## Workload Configs

| tokens | hidden_dim | dtype |
|--------|-----------|-------|
| 512 | 4096 | bfloat16 |
| 2048 | 4096 | float32 |
| 2048 | 4096 | bfloat16 |
| 8192 | 4096 | bfloat16 |
| 8192 | 8192 | bfloat16 |
| 32768 | 8192 | bfloat16 |
| 65536 | 8192 | bfloat16 |

## Usage

```bash
# Run autotuning for Helion (once, takes ~1-4h)
sbatch job-autotune.sh

# Run benchmarks
sbatch job-profile.sh

```



## TODO

- [ ] Verify numerical accuracy of Helion and CuteDSL outputs against PyTorch baseline
