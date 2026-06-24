# Results Summary — V2.1

## Push-T (3 seeds, T4 fp32, budget=50)

| Model | Seed 3072 | Seed 3073 | Seed 3074 | Mean ± Std |
|---|---|---|---|---|
| **Hybrid Mamba-2+Attention** | 92% | 98% | 94% | **94.7% ± 3.1%** |
| LeWM official | 86% | 90% | 82% | **86.0% ± 4.0%** |
| Gap | +6% | +8% | +12% | **+8.7%** |

## TwoRoom (3 seeds, T4 fp32, budget=50)

| Model | Seed 3072 | Seed 3073 | Seed 3074 | Mean ± Std |
|---|---|---|---|---|
| **Hybrid Mamba-2+Attention** | 84% | 76% | 96% | **85.3% ± 10.1%** |
| LeWM official | 78% | 72% | 92% | **80.7% ± 10.3%** |
| Gap | +6% | +4% | +4% | **+4.7% (tied in CI)** |

## V1 Hybrid CfC+Attention (TwoRoom, abandoned)

| Model | Success | Note |
|---|---|---|
| Hybrid CfC+Attention T=16 | 78% | SIGReg + CfC hidden state interaction |
| Hybrid CfC+Attention T=4 | ~72% | Training instable |

## CEM Time (T4 fp32, post-compile)

| Model | Push-T | TwoRoom |
|---|---|---|
| Hybrid Mamba-2+Attention | ~85s/ep | ~180s/ep |
| LeWM official | ~20s/ep | ~30s/ep |
| Ratio | ~4.25× | ~6× |

First ep compile: Hybrid ~1160s (Push-T), LeWM ~98s.
Mamba-2 Triton kernel overhead trên T4. Cần A100/5090 cho speed fair.

## Horizon test (Push-T, H=5/10/20)

| Model | H=5 | H=10 | H=20 |
|---|---|---|---|
| Hybrid Mamba-2+Attention | **94.7%** | 42% | 2% |
| LeWM official | **86.0%** | 40% | 4% |

Error compound common ở latent world models.
