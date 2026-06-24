# V2.1 — Hybrid Mamba-2+Attention (Main Result)

## Status: **Done** ✅
Push-T: 94.7% ± 3.1% (3 seeds, T4) — beat LeWM AR 86.0% ± 4.0% (+8.7%)
TwoRoom: 85.3% ± 10.1% — tied with LeWM 80.7% ± 10.3%

## Architecture
6×{Self-Attn(AdaLN) → Mamba-2}, T=4, heads=16, d_state=256, expand=4
Attention:Mamba-2 ≈ 1.43:1 (787K:550K per block)
Predictor 9.36M, total 16.6M

## Config
| Param | Value |
|---|---|
| Epochs | 10 |
| Batch | 128 |
| lr | 5e-5 |
| bf16 | Yes |
| GPU | RTX 5090 (train), T4 (eval) |

## Papers
| Paper | Link | Contribution |
|---|---|---|
| Mamba-2 (Dao & Gu 2024) | link-paper/19 | SSD core layer |
| LeWM (Maes 2026) | link-paper/00 | JEPA template |
| Hymba (ICLR 2025) | link-paper/21 | Head-wise hybrid ref |
| NVIDIA Mamba-2-Hybrid | link-paper/23 | Large hybrid ref |
| Drama (ICLR 2025) | link-paper/22 | Mamba WM ref |
| TransMamba (AAAI 2026) | link-paper/?? | Dynamic hybrid switching |

## Docs
- `docs/general/results-summary.md` — full tables
- `docs/general/eval-protocol.md` — eval protocol
- `docs/general/setup-full.md` — install guide
- `docs/general/module-specs.md` — predictor spec

## Key Files
- `le-wm-v2.1/module.py` — Mamba2Predictor
- `le-wm-v2.1/README.md` — install + eval guide
