# Architecture Overview — LeWM Project

## Evolution

```
V0 (done)        V1 (abandoned)        V2.1 (active)          V3 (future)
─────────────────────────────────────────────────────────────────────────
Real robot       TwoRoom sim           TwoRoom sim             Overhead cam
Bionic hand      Hybrid                Mamba-2+Attention       Multi-agent
8-DOF            CfC+Attention         6×{Self-Attn→Mamba-2}   2 robots
Data tự xây      78% (SIGReg issue)    Predictor 9.36M         Joint latent
```

## V2.1 Architecture (active)

```
Input: T=4 frames (224×224×3) + actions
  │
  ▼
[TinyViT Encoder] ──→ [Projector 192→2048→192 + BN]
  │
  ▼
[Noise Filter: Denoiser MLP residual]
  │
  ▼
[Predictor: 6×{Self-Attention(AdaLN) → Mamba-2}]
  │  heads=16, d_state=256, expand=4
  │  787K Attention + 550K Mamba-2 per block = 1.43:1 ratio
  │  Total predictor: 9.36M params
  │
  ▼
[Pred Projector MLP 192→2048→192 + BN]
  │
  ▼
Loss: MSE(pred, target) + λ·SIGReg(latent)
```

## V3 Social Architecture (planned)

```
Overhead cam ──→ Joint Encoder ──→ Joint Latent
                                  ↕
                    Social Predictor (Mamba-2+Attention)
                                  ↕
                    CEM planner across agents

2 robots sharing 1 latent space
```

## Key design decisions

| Decision | Why |
|---|---|
| JEPA (not generative) | Avoid pixel-level reconstruction. Yann LeCun 2022. |
| Mamba-2 over CfC | CfC SIGReg interaction bug in V1. Mamba-2 2-8× faster than Mamba-1. |
| Attention:Mamba 1.43:1 | Balanced hybrid — Attention handles recall, Mamba handles long-range. |
| T=4 | LeWM standard. Balance context vs compute. |
| SIGReg λ=0.09 | LeWM default. Prevents latent collapse. |
