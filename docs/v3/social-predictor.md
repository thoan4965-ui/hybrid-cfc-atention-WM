# Social Predictor — MDN-GMM + KL + NLL

## Problem
Standard JEPA predictor outputs single future latent (MSE loss). For social multi-agent:
- Future is inherently **multi-modal**: robot 2 might go left OR right
- Single Gaussian can't capture multiple possible futures
- Planning needs to consider multiple scenarios

## Solution: Mixture Density Network (MDN)

Predict K Gaussian components instead of a single latent:

```
Input: z_t (joint latent at time t)
    │
    ▼
SocialPredictor: Mamba-2+Attention + MDN head
    │
    ▼
Output: K × (μ_k ∈ ℝ^D, σ_k² ∈ ℝ^D, π_k ∈ ℝ)
    │  D = latent_dim (192)
    │  K = number of mixture components (5-10)
    │  π_k = mixing weights, Σ π_k = 1 (softmax)
    ▼
Predicted distribution: p(z_{t+1} | z_t) = Σ π_k · N(z | μ_k, σ_k²)
```

## Loss Function

### NLL (Negative Log-Likelihood)
```
NLL = -log( Σ_k π_k · N(z_target | μ_k, σ_k²) )
     = -log( Σ_k π_k · (2πσ_k²)^{-D/2} · exp(-||z_target - μ_k||² / 2σ_k²) )
```

Properties:
- Naturally handles multi-modal targets: assigns high prob to any plausible future
- Variance σ² learned: model can express uncertainty (high σ² = uncertain)
- Log-sum-exp is numerically stable with log-sum-exp trick

### KL Regularization
```
KL(pred || prior) = KL(Σ π_k · N(μ_k, σ_k²) || N(0, I))
```

Closed-form for GMM vs single Gaussian:
```
KL(GMM || N(0,I)) ≈ Σ π_k · KL(N(μ_k, σ_k²) || N(0,I))
```
Where:
```
KL(N(μ, σ²) || N(0,1)) = 0.5 · (-log σ² - 1 + σ² + μ²)
```

Total loss:
```
L = NLL + λ_KL · KL(pred || prior) + λ_SIG · SIGReg(z)
```

### Comparison to Standard Loss

| Loss | Single future | Multi-modal | Uncertainty | Collapse prevention |
|---|---|---|---|---|
| MSE | ✅ | ❌ | ❌ | SIGReg |
| NLL (GMM) | ✅ | ✅ | ✅ (via σ²) | SIGReg + KL |
| KL | ✅ | ✅ | ✅ | ✅ (KL itself) |

## V3 Application

### V3.0 — 1 agent
- Single predictor output for joint latent
- MSE loss sufficient (single future per CEM plan)

### V3.1 — 2 agents + cross-attn
- Each robot has its own MDN head
- NLL captures multi-modal other-agent behavior
- KL keeps latent distributions regularized
- Cross-attn provides other-agent context for each MDN

### V3.2 — Partial obs
- MDN critical: from ego view, other robot's future is highly uncertain
- High σ² on uncertain predictions
- KL prevents collapse when observations are ambiguous

## Hyperparameters

| Param | Value | Notes |
|---|---|---|
| K (components) | 5 | Start, increase if multi-modal behavior observed |
| λ_KL | 0.01 | Weak regularization, tune if collapse |
| σ²_min | 0.01 | Clamp σ² to prevent NaN from division by zero |
| π temp | 1.0 | Softmax temperature for mixing weights |
