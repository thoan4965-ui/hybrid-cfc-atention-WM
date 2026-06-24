# Module Specs — V2.1

## Mamba2Predictor (`le-wm-v2.1/module.py`)

```
6×{Self-Attention(AdaLN) → Mamba-2}
  Heads: 16, d_head: 64, d_model: 192
  Mamba-2: d_state=256, expand=4, d_conv=4
  AdaLN: modulation từ timestep embedding
  Attention params: ~787K per block
  Mamba-2 params: ~550K per block
  Total predictor: 9.36M params
```

### Input
- `x`: (B, T, d_model) — latent embeddings
- `t`: (B,) — timestep cho AdaLN
- `actions`: (B, T-1, act_dim) — action sequence

### Forward
```
for each block:
    x = SelfAttention(AdaLN(x, t)) + x
    x = Mamba2(x) + x
```

## Embedder (`le-wm-v2.1/module.py`)

```
TinyViT encoder → Projector(192→2048→192 + BN)
Noise Filter: Denoiser MLP(192→2048→192, residual)
```

### Input
- `images`: (B, T, 3, H, W) — RGB frames
- `target_idx`: int — index of target frame to embed

### Output
- `embeddings`: (B, d_model) — latent representation

## MLP (Action Encoder)

```
Linear(act_dim → 192) + SiLU
```

## Loss

```
L = MSE(pred_latent, target_latent) + λ * SIGReg(latent)
```
- λ = 0.09 (mặc định LeWM)
- SIGReg = cosine similarity regularization
