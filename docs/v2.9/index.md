# V2.9 — GA + Gradient + Hebbian + Dopamine

## Overview

Neuroevolution action-only with 4 parallel learning mechanisms orchestrated by evolved dopamine weights. No reward function. No gradient. Survival + curiosity fitness.

## Files

| File | Lines | Role |
|---|---|---|
| ae.py | 24 | Autoencoder 10→16→10, NaN-safe |
| cppn.py | 52 | CPPN → policy + prediction + dopamine weights |
| env_ant.py | 56 | NoRewardAnt, 3 rings, 6 foods, energy=20 |
| genome.py | 93 | JIT mutate (scan), vmap crossover, 2nd genome for dopamine |
| hebbian.py | 13 | Hebbian update with scale (w_hebb), returns all 4 keys |
| main.py | 215 | Run loop: GA + gradient + hebbian + dopamine + HF checkpoint |
| render_video.py | 55 | Video render with food overlay |

## Architecture

### 3-tier learning

1. **500 steps (1 gen)**
   - `policy_forward` → `[action, pred_next_obs]`
   - Hebbian update (scale = w_hebb)
   - Gradient backprop (scale = w_grad) on `||next_obs - pred_next||²`
   - Both update shared w_ih — balanced by dopamine

2. **End of gen**
   - AE: encode action(8)+fitness+energy → 16-dim tag
   - Dopamine bonus = AE loss → curiosity fitness
   - `f_total = steps_alive + 50 × dopamine`

3. **Across generations**
   - Tournament selection
   - Crossover: genome(100) + tag(16) + dopa(3)
   - JIT mutate (lax.scan) + vmap crossover
   - Elitism (top 2)
   - Lamarckian: tag diff → genome bias

### Dopamine (2nd genome)

- 3 floats per agent, evolved independently of main genome
- `softmax(dopa × 3)` → w_grad, w_hebb, w_ga
- w_grad: gradient update strength [0,1]
- w_hebb: hebbian update strength [0,1]
- w_ga: GA mutation rate (replaces Mechanism X)

## Key parameters

| Param | Value |
|---|---|
| energy_init | 20 |
| energy_cost | 0.4 |
| torque_cost | 0.05 |
| food_energy | 50 |
| rings | 3 (bk=5,10,15) |
| foods total | 6 (2 per ring) |
| MAX_GENES | 100 |
| AE dim | 10→16→10 |
| TAG_DIM | 16 |
| lr_gradient | 0.001 |
| lr_ae | 0.001 |
| AE steps | 30 |
| Multi-eval | 3 seeds |

## Checkpoint

- Auto save every 500 gen
- HF repo: `hhian/checkpoints`
- Resume: downloads latest checkpoint, continues
- Files: `checkpoints/cp_{gen}.npz`

## Bug history (all fixed)

1. float(innov) in JIT scan → innov.astype(float32)
2. int(innov) in JIT → convert at caller
3. hebbian_update missing w_pred, w_dopa in return
4. AE encode/decode using raw data instead of normalized
5. render_video env.step receiving tuple
6. expression.py dead (mechanism_x removed)
7. Checkpoint missing dopas + backward compat
