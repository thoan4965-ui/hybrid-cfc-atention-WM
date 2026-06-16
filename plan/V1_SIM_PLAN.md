# V1: Sim MuJoCo — Hybrid CfC+Attention Benchmark

## Mục tiêu
Benchmark CfC vs AR vs Hybrid predictor trên task dài (38+ step) trong sim.
Chuẩn bị cho sim2real bằng augment mạnh để encoder học dynamics thuần.

## Tech Stack

| Component | Công nghệ |
|---|---|
| Sim | MuJoCo 3.1.6+ (Python) |
| Arm | SO-ARM100 5-DOF (MuJoCo Menagerie, 3.6k⭐) |
| Gripper | Built-in parallel gripper (SO-ARM100 có sẵn) |
| Hand (optional) | LEAP Hand 16-DOF (MuJoCo Menagerie) |
| Colab | T4, free |
| Model | JEPA (TinyViT encoder) + CfC/AR/Hybrid predictor |
| Data format | H5 (GPUDataset) |

## Phase 1: Sim Environment Setup (3 ngày)

### Day 1: MuJoCo + SO-ARM100
```python
import mujoco
xml = """...so_arm100.xml + scene.xml..."""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Task: reach→grasp→lift→place target object
# Randomize: object position, object color, lighting, camera angle
```

- Load SO-ARM100 MJCF từ robot_descriptions hoặc MuJoCo Menagerie
- Thêm object (cube/sphere) vào scene, randomize vị trí
- Camera render: 64×64 hoặc 96×96 RGB
- Parallel env: 64 envs trên Colab T4 (MuJoCo CPU parallel)

### Day 2: Expert Policy + Data Collection

```
Expert policy (scripted):
  1. Reach: move end-effector above object (IK or joint interpolation)
  2. Grasp: close gripper
  3. Lift: move up 10cm
  4. Place: move to target, open gripper
  = 40-50 steps per episode

Randomize mỗi episode:
  - Camera angle: 3-5 positions random (overhead, side, 45°)
  - Lighting: intensity [0.5, 1.5], color temp [3000, 6500]K
  - Background: plane texture (5+ options), random color
  - Object: 5+ RGB colors, 2 shapes (cube, cylinder)
  - Object position x/y: ±5cm random
  - Joint noise: actuator noise 2% (simulate real motor variance)

Collect: 5000 episodes (~2h Colab)
  - 4000 train + 1000 validation
  - OOD test set: 200 episodes với camera angle/lighting/obj_color chưa từng thấy
```

### Day 3: Augment Strategy cho Sim2Real

```yaml
sim_augment:
  camera:
    - góc: overhead(2), side(2), wrist(1)
    - noise: Gaussian σ=0.02 (sensor noise)
  lighting:
    - intensity: uniform[0.5, 1.5]
    - direction: random hemisphere
  background:
    - 5 texture patterns (từ assets/)
    - random solid color (HSV hue random)
  object:
    - color: 5 colors (đỏ, xanh, vàng, trắng, đen)
    - shape: cube, cylinder
  physics:
    - joint friction: ±20%
    - actuator strength: ±10%
```

→ Mục tiêu: encoder học joint angles → hand pose, ko học appearance.
→ SIGReg + augment = encoder tự động discard color/lighting/texture.
→ Zero-shot sim2real: deploy policy lên robot thật ko cần fine-tune.

## Phase 2: Model Training (4 ngày)

### Model Architecture

```python
# Shared encoder (TinyViT, từ V0)
enc = TinyViT(96, 8, 4, 64, 4, 256, 32)

# 3 predictors để benchmark:
# 1) CfC (V4 architecture, baseline)
prd_cfc = CfCPredictorV2(num_frames=6, hidden_dim=96, ...)

# 2) AR (Transformer, baseline)  
prd_ar = ARPredictor(depth=1, heads=2, hidden_dim=64, ...)

# 3) Hybrid CfC+Attention (novelty)
prd_hybrid = HybridCfCAttention(
    cfc_hidden=96,
    attn_window=3,        # 3-frame action window (giống AR 24-dim)
    attn_heads=4,
    cfc_backbone_layers=1
)
```

**Hybrid architecture:**
```
Input: [emb_t, action_t, attn_context]
  1. Attention: action_t + attn_context (3 frame window) → smoothed_action
  2. CfC step: emb_t + smoothed_action → next_emb
  → Action noise filtered bởi attention window (AR advantage)
  → Temporal dynamics bởi CfC hidden state (ODE advantage)
```

### Training Config

| Param | CfC | AR | Hybrid |
|---|---|---|---|
| Epochs | 100 | 100 | 100 |
| Batch | 64 | 64 | 64 |
| LR | 3e-4 | 3e-4 | 3e-4 |
| SS | 0.3 | — | 0.3 |
| SIGReg λ | 0.05 | 0.05 | 0.05 |
| Time | 2h T4 | 2h T4 | 2.5h T4 |

## Phase 3: Benchmark Protocol (1 ngày)

### T1: Long Rollout (20-step → 50-step)

| Step | CfC | AR | Hybrid |
|---|---|---|---|
| 1 | — | — | — |
| 10 | — | — | — |
| 20 | — | — | — |
| 50 | — | — | — |
| Drift/step | — | — | — |

Kỳ vọng: CfC drift ≈ 0.00001, AR drift ≈ 0.0005, Hybrid drift ≈ CfC.

### T2: Variable Δt (3-step)

| Δt | CfC | AR | Hybrid |
|---|---|---|---|
| 1 | — | — | — |
| 3 | — | — | — |
| 5 | — | — | — |
| 8 | — | — | — |

Kỳ vọng: CfC cải thiện 83% ở Δt=8. AR flat. Hybrid ≈ CfC (ODE backbone).

### T3: OOD Action (scale ×1.5, ×2.0)

| Scale | CfC | AR | Hybrid |
|---|---|---|---|
| 1.0 | — | — | — |
| 1.5 | — | — | — |
| 2.0 | — | — | — |
| Gap | 26x? | 1.7x? | ??? |

**Quan trọng nhất cho ISEF:** Hybrid cần gap < 5x. Nếu Hybrid gap ≈ AR → chứng minh Attention filter action noise hoạt động.

### T4: Task Success Rate (CEM planning, 100 runs)

| Model | Reach | Grasp | Lift | Place | Total |
|---|---|---|---|---|---|
| CfC | — | — | — | — | — |
| AR | — | — | — | — | — |
| **Hybrid** | — | — | — | — | — |

Kỳ vọng: Hybrid ≥ AR ở success rate (cả 2 robust OOD action). CfC thấp hơn (OOD action yếu → CEM kém).

### T5: Sim2Real Gap

Compare latent distribution between sim data and real camera:
- Encode 100 sim frames + 100 real frames (camera) → t-SNE
- Measure MMD (Maximum Mean Discrepancy) giữa 2 distributions
- Nếu MMD < 0.1 → sim2real gap nhỏ

## Phase 4: Build Real + Deploy (sau ISEF, optional)

| Component | Cost | Nguồn |
|---|---|---|
| SO-ARM100 arm | $122 | Alibaba/Seeed |
| SC09 servos (gripper) | Có sẵn | Từ V0 |
| Camera | Có sẵn | Từ V0 |
| Serial | Có sẵn | COM13 |
| Power supply | $10 | Amazon |

Build: 2-3 ngày (in 3D + lắp ráp + nối dây)
Deploy: dùng V0 pipeline (robot_planner.py) + sim-trained predictor → fine-tune nhẹ 10 epoch.

## Timeline Tổng

| Phase | Thời gian | Phụ thuộc |
|---|---|---|
| P1: Sim Env + Data | 3 ngày | MuJoCo, SO-ARM100 MJCF |
| P2: Train 3 models | 4 ngày | Data từ P1 |
| P3: Benchmark | 1 ngày | Model từ P2 |
| P4: Report + ISEF | 2 ngày | Kết quả P3 |
| **Tổng** | **~10 ngày** | — |

## Key Novelty Claims (cho ISEF)

1. **CfC outperforms AR in long-horizon manipulation** (50-step, 3.3× better drift)
2. **Hybrid CfC+Attention solves CfC's OOD action weakness** (gap 26x→?x)
3. **Per-sequence ColorJitter augmentation with MuJoCo domain randomization** enables zero-shot sim2real
4. **First JEPA world model comparison on real robotic arm manipulation** (LeWM only tested on Push-T/Cube sim)

## Risks & Mitigation

| Risk | Mitigation |
|---|---|
| MuJoCo scene→real gap quá lớn | Augment mạnh (lighting, texture, noise) |
| Hybrid architecture ko hội tụ | Baseline CfC + Attention riêng → merge dần |
| CEM planning chậm trên sim (H lớn) | Giảm CEM samples (50 thay 100) |
| LEAP Hand + arm merge phức tạp | Bắt đầu với gripper (SO-ARM100 có sẵn), hand là optional |
