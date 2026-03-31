# SwiGLU Kernel Profiling

Compares implementations of the SwiGLU activation (`SiLU(x1) * x2`) on H100 GPUs.

## Kernels

| File | Name in results | Description |
|------|----------------|-------------|
| `swiglu_pytorch.py` | `pytorch_eager` | Naive PyTorch unfused, 5 memory passes |
| `swiglu_pytorch_compile.py` | `pytorch_compile` | `torch.compile` auto-fused by Inductor |
| `autotune_helion.py` | `helion` | Helion DSL. Fused, autotuned tile config |
| `swiglu_cutedsl.py` | `cutedsl` | CuteDSL simple |

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




## TODO

- [ ] Check accuracy of Helion and CuteDSL outputs against PyTorch baseline
