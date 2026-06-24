# V3 — Multi-agent Social World Models

## 3 Phases

| Phase | Camera | Agent | Robot | Cơ chế chính |
|---|---|---|---|---|
| **V3.0** | 1 overhead | 1 shared brain | 2 robot bodies | 1 predictor, 2 action heads |
| **V3.1** | Overhead + 2 ego | 2 agent (1/robot) | 2 robots | Cross-attn + CLIP + MDN-GMM |
| **V3.2** | 2 ego (no overhead) | 2 agent | 2 robots | Partial obs + cross-attn + CLIP |

## Docs

| File | Nội dung |
|---|---|
| `docs/v3/v3.0.md` | 1 agent, 2 robots (overhead cam only) |
| `docs/v3/v3.1.md` | 2 agents, overhead + ego cams, cross-attn, CLIP |
| `docs/v3/v3.2.md` | 2 agents, ego only (partial obs), cross-attn |
| `docs/v3/social-predictor.md` | MDN-GMM, NLL loss, KL regularization |

## Key Papers

| Paper | For |
|---|---|
| COMBO (ICLR 2025) | Multi-agent world model framework |
| S3AP (2025) | Social reasoning via structured rep |
| CLIP (Radford 2021) | Joint embedding space alignment |
| MDN (Bishop 1994) | Multi-modal future prediction (GMM) |
| VAE (Kingma 2014) | KL regularization theory |
| V-JEPA (Meta 2024) | Video JEPA for multi-view |

## Timeline
- V3 = after V2.5 robot demo (ISEF trường tháng 9)
- V3.1 = after ISEF tỉnh (tháng 11)
- V3.2 = ISEF quốc gia (tháng 1-3)
