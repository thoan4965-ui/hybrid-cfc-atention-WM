# Tài liệu tham khảo — LeWM Project

## Baseline (World Model)
| Paper | Link | Ghi chú |
|-------|------|---------|
| LeWorldModel (Maes et al. 2026) | https://arxiv.org/abs/2603.19312 | JEPA world model gốc, code: lucas-maes/le-wm |
| DINO-WM (Zhou et al. 2024) | https://arxiv.org/abs/2411.04983 | World model với DINOv2 encoder |
| PLDM (2024) | https://arxiv.org/abs/2311.00978 | End-to-end JEPA với VICReg 7-term loss |
| JEPA (LeCun 2022) | https://openreview.net/forum?id=BZ5a1r-kVsf | Joint Embedding Predictive Architecture |

## Mamba — State Space Models
| Paper | Link | Ghi chú |
|-------|------|---------|
| Mamba-1 (Gu & Dao 2023) | https://arxiv.org/abs/2312.00752 | Selective SSM, O(n), code: state-spaces/mamba |
| Mamba-2 (Dao & Gu 2024, ICML) | https://arxiv.org/abs/2405.21060 | SSD layer, 2-8× faster training |
| Mamba-3 (Lahoti et al. 2026, ICLR Oral) | https://arxiv.org/abs/2603.15569 | Exp-trapezoidal, complex-valued, MIMO |

## Hybrid Attention-Mamba
| Paper | Link | Ghi chú |
|-------|------|---------|
| Jamba (Lieber et al. 2024) | https://arxiv.org/abs/2403.19887 | First hybrid Transformer-Mamba-MoE, 1:7 ratio |
| NVIDIA Mamba-2-Hybrid (Waleffe et al. 2024) | https://arxiv.org/abs/2406.07887 | 8B hybrid beat Transformer all 12 benchmarks |
| Hymba (Dong et al. 2025, ICLR Spotlight) | https://arxiv.org/abs/2411.13676 | Head-wise parallel hybrid, 1:1 ratio |
| TransMamba (Li et al. 2026, AAAI) | https://arxiv.org/abs/2503.24067 | Sequence-level dynamic switching |

## World Model + Mamba
| Paper | Link | Ghi chú |
|-------|------|---------|
| Drama (Wang et al. 2025, ICLR) | https://arxiv.org/abs/2410.08893 | Mamba-2 world model cho Atari, 7M params |

## Mamba — Lý thuyết
| Paper | Link | Ghi chú |
|-------|------|---------|
| Understanding Input Selectivity (2025) | https://arxiv.org/abs/2506.11891 | S6 memory decay + wavelet approximation |
| Primacy & Recency in Mamba (2025) | https://arxiv.org/abs/2506.15156 | Memory effects, Δ controls forgetting |
| Hidden Attention of Mamba (2024) | https://arxiv.org/abs/2403.01590 | Mamba as implicit attention |
| Computational Limits of SSMs (2025, CPAL) | https://proceedings.mlr.press/v280/chen25b.html | Mamba ∈ TC⁰, same as Transformer |

## Vision Encoder
| Paper | Link | Ghi chú |
|-------|------|---------|
| Vision Mamba - Vim (Zhu et al. 2024, ICML) | https://arxiv.org/abs/2401.09417 | Pure SSM vision backbone |
| MambaVision (Hatamizadeh et al. 2025, CVPR) | https://arxiv.org/abs/2407.08083 | Hybrid Mamba-Transformer vision |
| TinyViT (Wu et al. 2022, ECCV) | https://arxiv.org/abs/2207.10666 | LeWM encoder gốc |
| ViT (Dosovitskiy et al. 2021, ICLR) | https://arxiv.org/abs/2010.11929 | Vision Transformer |

## Temporal Model — CfC
| Paper | Link | Ghi chú |
|-------|------|---------|
| CfC (Hasani et al. 2022, Nature MI) | https://www.nature.com/articles/s42256-022-00556-7 | Closed-form continuous-time, baseline V1 |
| LTC (Hasani et al. 2021) | https://arxiv.org/abs/2106.13898 | Liquid Time-Constant networks |

## Nén & Deploy
| Paper | Link | Ghi chú |
|-------|------|---------|
| Quamba (ICLR 2025) | https://arxiv.org/abs/2405.13654 | Mamba quantization, W8A8 |
| SLiM (Google DeepMind, ICML 2025) | — | Quantize + sparse + low-rank |
| CompACT (2025) | https://arxiv.org/abs/2503.03062 | Nén visual obs → 8 tokens, 40× planning speedup |

## Công cụ
| Tool | Link | Ghi chú |
|------|------|---------|
| stable-pretraining | https://github.com/galilai-group/stable-pretraining | Training framework (PyTorch + Lightning) |
| stable-worldmodel | https://github.com/galilai-group/stable-worldmodel | Environment + planning + eval |
| state-spaces/mamba | https://github.com/state-spaces/mamba | Mamba-1/2/3 CUDA kernels |
| ncps (CfC) | https://github.com/mlech26l/ncps | CfC / ODE-RNN Torch implementation |

## ISEF & Sáng tạo trẻ
| Link | Ghi chú |
|------|---------|
| https://www.societyforscience.org/isef/ | ISEF official |
| https://www.societyforscience.org/isef/categories-and-subcategories/all-categories/ | ISEF 22 categories |
| https://en.embarkchina.org/blog/1082.html | ISEF 2024 SOFT 1st analysis |
| https://pith.science/paper/2403.19887 | Third-party Jamba review |
