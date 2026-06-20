# Hybrid Mamba-2+Attention World Model

Hybrid block-level JEPA predictor cho robot manipulation. Thay MLP (LeWM AR) bằng Mamba-2 discrete state — vừa giữ temporal advantage, vừa không khuếch đại noise.

**Push-T: 94.7% ± 3.1% beat LeWM official 86.0% ± 4.0% (+8.7%) — cùng T4 fp32, 3 seeds.**

## Kết quả

| Task | Model | 3 seeds | Mean ± std |
|---|---|---|---|
| Push-T | **Hybrid Mamba-2** | 92, 98, 94 | **94.7% ± 3.1%** |
| Push-T | LeWM official | 82, 86, 90 | 86.0% ± 4.0% |
| TwoRoom | **Hybrid Mamba-2** | 84, 76, 96 | **85.3% ± 10.1%** |
| TwoRoom | LeWM official | 72, 78, 92 | 80.7% ± 10.3% |

## Thư mục

| Thư mục | Nội dung |
|---|---|
| `le-wm-v2.1/` | Hybrid Mamba-2+Attention — code chính |
| `le-wm-vo/` | V0: real robot bionic hand + CfC vs AR comparison |
| `le-wm-v1/` | V1: Hybrid CfC+Attention (abandoned, tham khảo) |

## Checkpoints & Data

HuggingFace: [hhian/checkpoints](https://huggingface.co/hhian/checkpoints)

## Tham khảo

LeWorldModel (Maes et al. 2026) — baseline AR predictor.
CfC (Hasani et al. 2022) — ODE stateful baseline.
Mamba-2 (Dao & Gu 2024) — SSM backbone.
Ma & Najarian (2025) — exponential memory decay analysis.
