# V1 — Hybrid CfC+Attention TwoRoom (Abandoned)

## Status: **Abandoned**
V1 replaced CfC with Mamba-2 because SIGReg noise × ODE hidden state caused performance collapse at long horizons (78% goal=25 → 6% goal=100).

## Architecture
Predictor: 6×{Self-Attn → CfC ODE}
Loss: MSE + λ·SIGReg (λ=0.09, num_proj=1024)
Eval: TwoRoom (goal=25, budget 50) → 78%

## Why It Failed
- CfC has stateful ODE hidden state (temporal advantage: 34× lower drift than AR)
- But SIGReg noise accumulates through ODE → collapses at long horizons
- Switch to discrete state (Mamba-2) solved this

## Papers
| Paper | Link | Contribution |
|---|---|---|
| CfC (Nature MI 2022) | link-paper/01 | ODE-RNN backbone |
| Hybrid Transformer+LNN | link-paper/09 | Precedent for hybrid |
| Liquid-S4 | link-paper/04 | CfC+S4 ref |
| Drone Racing CfC | link-paper/02 | CfC OOD gen example |
