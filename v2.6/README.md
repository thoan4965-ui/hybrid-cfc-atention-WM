# V2.6 — Structured Genome + Coordinate CPPN + No-Reward Evolution

Neuroevolution action only trên Brax Ant (MJX GPU).

## Stack
- **Genome:** structured (neuron + connection genes, innovation numbers)
- **CPPN:** coordinate-based, no fixed weight matrix (JIT)
- **GA:** tournament selection, gene-level crossover, subst/ins/del mutation, elitism
- **Fitness:** steps_alive (no reward, survival-only)
- **Energy:** init=100, cost=0.2/step + 0.05×torque², 8 food patches (finite + respawn)
- **Acceleration:** JAX JIT + vmap (128 agents × 500 steps in ~0.8s)

## Results (200 gen × 128 pop, T4)
- Best fitness: 500 (agent survived full episode)
- Mean fitness ~430
- ~11 phút

## Files
- `v2_6/genome.py` — genome init, mutate (subst/ins/del)
- `v2_6/cppn.py` — coordinate CPPN query + policy forward
- `v2_6/env_ant.py` — NoRewardAnt (reward=0, energy, food)
- `v2_6/main.py` — eval_batch, select_crossover, render, run
- `results/v26_result.npz` — best genome + fitness curve
