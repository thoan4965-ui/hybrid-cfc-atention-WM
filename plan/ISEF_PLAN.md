# KẾ HOẠCH ISEF: THAY CfC VÀO THAY CHO TRANSFORMER AR
**Ngày lập:** 2026-06-07  
**Kỹ sư trưởng:** AI Architect  
**Mục tiêu:** Chứng minh CfC tốt hơn Transformer AR trên 5 dimensions

---

## TỔNG QUAN

### Trạng thái hiện tại
- **Model đang dùng:** ResNet10tEncoder + CfCPredictor (cfc_resnet10t.yaml)
- **Model gốc LeWM:** ViT + ARPredictor (Transformer AR) (lewm.yaml)
- **Dataset:** bionic_hand_dataset_v2.h5 (89 episodes, 8900 frames, 224x224)
- **Config:** embed_dim=192, img_size=224, history_size=3, num_preds=3

### Goal ISEF
- Thay CfC vào thay cho Transformer AR trong kiến trúc LeWM
- So sánh 5 dimensions: **Speed, Stability, Accuracy, Smoothness, Physics**
- Chứng minh CfC tốt hơn Transformer AR
- Chạy real-time trên CPU local (i7-1165G7)

---

## PHASE 1: CHUẨN BỊ DATA (1 ngày)

### Task 1.1: Resize dataset 224→96
**Mục đích:** Giảm compute, giữ composition, đủ thấy grasp

**Tham số:**
- Input: `bionic_hand_dataset_v2.h5` (224x224)
- Output: `bionic_hand_dataset_v3_96.h5` (96x96)
- Interpolation: `cv2.INTER_AREA` (tốt nhất cho downscaling)

**Bài test:**
```python
# Test 1: Kiểm tra số episodes và frames giữ nguyên
assert new_dataset.num_episodes == 89
assert new_dataset.total_frames == 8900

# Test 2: Kiểm tra kích thước ảnh
assert new_dataset.frames.shape[1:] == (96, 96, 3)

# Test 3: Kiểm tra metadata giữ nguyên
assert new_dataset.episode_lengths == old_dataset.episode_lengths
assert new_dataset.actions.shape == old_dataset.actions.shape
```

**Script:** `scripts/resize_data_v3.py`

---

### Task 1.2: Augmentation Pipeline (Chống vẹt màu + hình học)

**Vấn đề:** LeWM paper thừa nhận vẫn bị vẹt màu sắc và hình học dù có cải thiện. Cần augmentation mạnh hơn.

**Chiến lược data:**
```
Dataset hiện có (89 episodes):
  25 empty (rỗng)
  13 yellow (chai vàng)  ← hue shift tạo thêm biến thể
  13 red (chai đỏ)       ← GIỮ LẠI, serve as OOD color test
  13 decoupled empty
  25 old episodes
```

**Tại sao giữ chai đỏ:**
- Hue shift trên chai vàng chỉ tạo dải: vàng → cam → vàng nhạt → xanh lá nhạt
- **Không bao giờ tạo ra đỏ** (quá xa trên vòng HSV, >120°)
- Chai đỏ = màu ngoài dải hue shift → test case tự nhiên cho color invariance
- Hue shift + red bottle = **bổ sung nhau**, không thay thế

**Augmentation pipeline:**
```python
from torchvision.transforms import v2

augmentation = v2.Compose([
    # Chống vẹt màu
    v2.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1  # ±18° trong HSV
    ),
    
    # Chống vẹt vị trí + scale
    v2.RandomResizedCrop(
        size=96,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),
    
    # Chống vẹt góc nghiêng
    v2.RandomRotation(degrees=5),
])
```

**KHÔNG dùng:**
- `RandomHorizontalFlip` — tay robot bất đối xứng, flip làm sai logic grasp
- `ElasticTransform` — biến dạng hình học tay → sai vật lý

**Bài test augmentation không phá structure:**
```python
img_orig = dataset[0]['pixels']
img_aug = augmentation(img_orig)

# Test 1: Shape giữ nguyên
assert img_aug.shape == (3, 96, 96)

# Test 2: Correlation > 0.6 (không phá quá nhiều)
corr = F.cosine_similarity(img_orig.flatten(), img_aug.flatten(), dim=0)
assert corr > 0.6

# Test 3: Augmentation đa dạng (10 lần không giống nhau)
augs = [augmentation(img_orig) for _ in range(10)]
pairwise_sim = [F.cosine_similarity(augs[i].flatten(), augs[j].flatten(), dim=0) 
                for i in range(10) for j in range(i+1, 10)]
assert max(pairwise_sim) < 0.98  # không phải copy
```

**Tích hợp vào train.py:**
```python
# Trong train.py, sau get_img_preprocessor
transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

# Thêm augmentation nếu đang train (không dùng khi validate)
if cfg.get('augmentation', False):
    from torchvision.transforms import v2
    aug_transform = v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.RandomRotation(degrees=5),
    ])
    transforms.append(aug_transform)
```

---

## PHASE 2: XÂY DỰNG KIẾN TRÚC MỚI (2 ngày)

### Task 2.1: Tạo TinyViT Encoder
**Mục đích:** Train từ scratch, không pretrained, phù hợp task nhỏ

**Tham số:**
```python
class TinyViT:
    num_layers = 4
    hidden_dim = 64
    num_heads = 4
    patch_size = 8
    mlp_dim = 256
    output_dim = 32  # latent dim
    img_size = 96
```

**Params tính toán:**
- Patch embed: (96/8)² = 144 tokens
- Each layer: Attention (64×64×4) + MLP (64→256→64) ≈ 60K
- Total: ~245K params

**Bài test:**
```python
# Test 1: Forward pass shape
encoder = TinyViT()
x = torch.randn(2, 3, 96, 96)
out = encoder(x)
assert out.shape == (2, 32)  # (B, latent_dim)

# Test 2: Số params
total_params = sum(p.numel() for p in encoder.parameters())
assert 240_000 < total_params < 260_000
```

**File:** `le-wm/module.py` (thêm class TinyViT)

---

### Task 2.2: Update CfCPredictor cho kiến trúc mới
**Mục đích:** Phù hợp latent_dim=32, giảm overfitting

**Tham số:**
```python
CfCPredictor:
    input_dim = 32  # latent dim
    hidden_dim = 64  # CfC units
    output_dim = 32
    action_dim = 32
    backbone_layers = 1  # giảm từ 2
    backbone_units = 64  # giảm từ 256
    wiring = FullyConnected  # default khi input_size là int
```

**Params tính toán:**
- CfC cell: ~42K params
- Total: ~42K params

**Bài test:**
```python
# Test 1: Forward pass shape
predictor = CfCPredictor(
    input_dim=32, hidden_dim=64, output_dim=32, action_dim=32
)
x = torch.randn(2, 3, 32)  # (B, T, D)
c = torch.randn(2, 3, 32)  # (B, T, A_emb)
out = predictor(x, c)
assert out.shape == (2, 3, 32)

# Test 2: Số params
total_params = sum(p.numel() for p in predictor.parameters())
assert 40_000 < total_params < 45_000
```

**File:** `le-wm/module.py` (update CfCPredictor)

---

### Task 2.3: Tạo ARPredictor baseline (Transformer AR)
**Mục đích:** Baseline để so sánh với CfC

**Tham số:**
```python
ARPredictor:
    input_dim = 32
    hidden_dim = 64
    output_dim = 32
    depth = 2  # giảm từ 6
    heads = 4  # giảm từ 16
    mlp_dim = 128  # giảm từ 2048
    dim_head = 16  # giảm từ 64
```

**Params tính toán:**
- Attention: 2 layers × (64×64×4) ≈ 33K
- MLP: 2 layers × (64→128→64) ≈ 16K
- Total: ~50K params (gần bằng CfC)

**Bài test:**
```python
# Test 1: Forward pass shape
predictor = ARPredictor(
    input_dim=32, hidden_dim=64, output_dim=32,
    depth=2, heads=4, mlp_dim=128, dim_head=16
)
x = torch.randn(2, 3, 32)
c = torch.randn(2, 3, 32)
out = predictor(x, c)
assert out.shape == (2, 3, 32)

# Test 2: Số params
total_params = sum(p.numel() for p in predictor.parameters())
assert 45_000 < total_params < 55_000
```

**File:** `le-wm/module.py` (giữ nguyên ARPredictor, chỉ update params)

---

### Task 2.4: Update Action Encoder
**Mục đích:** Phù hợp 8 DOF, latent_dim=32

**Tham số:**
```python
Embedder:
    input_dim = 8  # 8 servos
    smoothed_dim = 8
    emb_dim = 32  # latent dim
    mlp_scale = 2  # giảm từ 4
```

**Bài test:**
```python
encoder = Embedder(input_dim=8, emb_dim=32, mlp_scale=2)
x = torch.randn(2, 3, 8)  # (B, T, action_dim)
out = encoder(x)
assert out.shape == (2, 3, 32)
```

---

## PHASE 3: TẠO CONFIG FILES (0.5 ngày)

### Task 3.1: Tạo config cho TinyViT + CfC
**File:** `le-wm/config/train/model/vit_cfc.yaml`

```yaml
_target_: jepa.JEPA

encoder:
  _target_: module.TinyViT
  img_size: 96
  patch_size: 8
  num_layers: 4
  hidden_dim: 64
  num_heads: 4
  mlp_dim: 256
  output_dim: 32

predictor:
  _target_: module.CfCPredictor
  num_frames: ${history_size}
  input_dim: 32
  hidden_dim: 64
  output_dim: 32
  action_dim: 32
  backbone_layers: 1
  backbone_units: 64

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: 32
  mlp_scale: 2

# Không cần projector/pred_proj vì encoder output thẳng 32-d
projector: null
pred_proj: null
```

---

### Task 3.2: Tạo config cho TinyViT + Transformer AR (baseline)
**File:** `le-wm/config/train/model/vit_ar.yaml`

```yaml
_target_: jepa.JEPA

encoder:
  _target_: module.TinyViT
  img_size: 96
  patch_size: 8
  num_layers: 4
  hidden_dim: 64
  num_heads: 4
  mlp_dim: 256
  output_dim: 32

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: 32
  hidden_dim: 64
  output_dim: 32
  depth: 2
  heads: 4
  mlp_dim: 128
  dim_head: 16

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: 32
  mlp_scale: 2

projector: null
pred_proj: null
```

---

### Task 3.3: Update root config
**File:** `le-wm/config/train/lewm.yaml`

```yaml
img_size: 96  # thay đổi từ 224
embed_dim: 32  # thay đổi từ 192
history_size: 3
num_preds: 3

loss:
  sigreg:
    weight: 0.05  # grid search {0.001, 0.01, 0.05, 0.1}
    kwargs:
      knots: 9  # giảm từ 17
      num_proj: 256  # giảm từ 1024
```

---

## PHASE 4: HUẤN LUYỆN (2 ngày trên Colab)

### Task 4.1: Train TinyViT + CfC
**Lệnh:**
```bash
python train.py model=vit_cfc data=bionic_hand_v3_96 \
    trainer.max_epochs=100 \
    loss.sigreg.weight=0.05
```

**Bài test sau mỗi 10 epochs:**
```python
# Test 1: Loss giảm
assert train_loss < 1.0
assert val_loss < 1.2

# Test 2: Không overfit (train-val gap < 0.3)
assert val_loss - train_loss < 0.3
```

---

### Task 4.2: Train TinyViT + Transformer AR (baseline)
**Lệnh:**
```bash
python train.py model=vit_ar data=bionic_hand_v3_96 \
    trainer.max_epochs=100 \
    loss.sigreg.weight=0.05
```

**Bài test:** Giống Task 4.1

---

### Task 4.3: Grid search λ SIGReg
**Tham số:** λ ∈ {0.001, 0.01, 0.05, 0.1}

**Bài test chọn λ tốt nhất:**
```python
# Chọn λ có val_loss thấp nhất + separation cao nhất
best_lambda = min(results, key=lambda x: x['val_loss'] - 0.1*x['separation'])
```

---

## PHASE 5: SO SÁNH 5 DIMENSIONS (1 ngày)

### Task 5.1: Speed (Tốc độ inference)
**Script:** `scripts/benchmark_speed.py`

**Đo lường:**
```python
# CPU i7-1165G7, batch_size=1
import time

def benchmark(model, H=5, n_runs=1000):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = model.predict(x, c)  # H steps
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)

cfc_time = benchmark(cfc_model)
ar_time = benchmark(ar_model)

# Bài test
assert cfc_time[0] < ar_time[0] / 3  # CfC nhanh hơn 3x
assert cfc_time[0] < 0.005  # < 5ms/step
```

**Kết quả kỳ vọng:**
- CfC: ~2ms/step
- Transformer AR: ~8ms/step
- Speedup: 4x

---

### Task 5.2: Stability (Độ ổn định theo horizon)
**Script:** `scripts/benchmark_stability.py`

**Đo lường:**
```python
horizons = [1, 3, 5, 10]
for H in horizons:
    cfc_mse = compute_mse(cfc_model, H)
    ar_mse = compute_mse(ar_model, H)
    
    # Bài test: CfC ổn định hơn ở H lớn
    if H >= 5:
        assert cfc_mse < ar_mse * 0.8  # CfC thấp hơn 20%
```

**Kết quả kỳ vọng:**
- H=1: CfC ≈ AR
- H=5: CfC < AR (20-30%)
- H=10: CfC < AR (40-50%)

---

### Task 5.3: Accuracy (Độ chính xác)
**Script:** `scripts/benchmark_accuracy.py`

**Đo lường:**
```python
# Metrics: MSE, Cosine Sim, Separation
for H in [1, 5]:
    cfc_metrics = evaluate(cfc_model, H)
    ar_metrics = evaluate(ar_model, H)
    
    # Bài test
    assert cfc_metrics['separation'] > 0.01  # dương rõ rệt
    if H == 5:
        assert cfc_metrics['separation'] > ar_metrics['separation']
```

**Kết quả kỳ vọng:**
- H=1: Separation ≈ 0.02 (cả 2)
- H=5: CfC separation 0.015, AR separation 0.008

---

### Task 5.4: Smoothness (Độ mượt trajectory)
**Script:** `scripts/benchmark_smoothness.py`

**Đo lường:**
```python
# Latent velocity variance
def compute_smoothness(model, n_rollouts=100):
    velocities = []
    for _ in range(n_rollouts):
        trajectory = model.rollout(x0, actions, H=10)
        v = torch.diff(trajectory, dim=1)  # velocity
        velocities.append(v.norm(dim=-1).var())
    return np.mean(velocities)

cfc_var = compute_smoothness(cfc_model)
ar_var = compute_smoothness(ar_model)

# Bài test
assert cfc_var < ar_var / 2  # CfC mượt hơn 2x
```

**Kết quả kỳ vọng:**
- CfC velocity variance: 0.002
- AR velocity variance: 0.005
- CfC mượt hơn 2.5x

---

### Task 5.5: Physics understanding (Hiểu vật lý)
**Script:** `scripts/benchmark_physics.py`

**Đo lường:**
```python
# Qualitative: plot latent trajectory
# Quantitative: smooth cost (ticks/step)

def compute_smooth_cost(model, n_episodes=50):
    costs = []
    for ep in range(n_episodes):
        actions = model.plan(goal_state, H=5)
        smooth_cost = compute_action_smoothness(actions)
        costs.append(smooth_cost)
    return np.mean(costs)

cfc_cost = compute_smooth_cost(cfc_model)
ar_cost = compute_smooth_cost(ar_model)

# Bài test
assert cfc_cost < 10.0  # ticks/step
assert cfc_cost < ar_cost
```

**Kết quả kỳ vọng:**
- CfC smooth cost: 6 ticks/step
- AR smooth cost: 12 ticks/step

---

## PHASE 6: TÍCH HỢP VÀO ROBOT (1 ngày)

### Task 6.1: Update run_mpc.py
**Thay đổi:**
- Load TinyViT encoder thay vì ResNet10t
- Update latent_dim từ 192→32
- Update action normalization

**Bài test:**
```python
# Test 1: Load model thành công
model = load_model('weights_cfc_best.pt')
assert model.encoder.output_dim == 32

# Test 2: Inference real-time
latency = measure_latency(model, n_runs=100)
assert latency < 0.050  # < 50ms total (encoder + predictor + CEM)
```

---

### Task 6.2: Chạy test suite T-01 đến T-06
**Lệnh:**
```bash
python scripts/task_01_sanity.py
python scripts/task_02_one_step_pred.py
python scripts/task_03_action_conditioning.py
python scripts/task_04_goal_state_check.py
python scripts/task_05_controller_unit_test.py
python scripts/task_06_decision_test.py
```

**Bài test:**
```python
# T-02: Cosine Sim > 0.80
assert t02_result['cosine_sim'] > 0.80

# T-03: Delta Cosine > 0.05
assert t03_result['delta_cosine'] > 0.05

# T-04: Separation > 0.01
assert t04_result['separation'] > 0.01

# T-06: AI_ADDS_VALUE + Diff Dist > 10 ticks
assert t06_result['verdict'] == 'AI_ADDS_VALUE'
assert t06_result['diff_dist'] > 10.0
```

---

### Task 6.3: Chạy trên robot thật
**Lệnh:**
```bash
python scripts/run_mpc.py --real
```

**Bài test:**
```python
# Test 1: Grasp thành công chai vàng
success_yellow = test_grasp('yellow_bottle')
assert success_yellow

# Test 2: Grasp thành công chai đỏ (generalization)
success_red = test_grasp('red_bottle')
assert success_red

# Test 3: Không grasp khi không có chai
no_grasp_empty = test_no_grasp()
assert no_grasp_empty
```

---

## TIMELINE TỔNG

| Phase | Tasks | Thời gian | Output |
|-------|-------|-----------|--------|
| 1 | Resize data | 1 ngày | bionic_hand_dataset_v3_96.h5 |
| 2 | Build architecture | 2 ngày | TinyViT, CfC, AR configs |
| 3 | Create configs | 0.5 ngày | vit_cfc.yaml, vit_ar.yaml |
| 4 | Train models | 2 ngày (Colab) | weights_cfc_best.pt, weights_ar_best.pt |
| 5 | Benchmark 5D | 1 ngày | speed/stability/accuracy/smoothness/physics report |
| 6 | Integrate robot | 1 ngày | Real robot demo |
| **Total** | | **7.5 ngày** | **ISEF-ready** |

---

## RỦI RO VÀ GIẢI PHÁP

### Rủi ro 1: CfC không train được
**Giải pháp:** 
- Giảm learning rate: 5e-5 → 1e-5
- Tăng epochs: 100 → 150
- Thử λ SIGReg khác

### Rủi ro 2: Separation quá thấp
**Giải pháp:**
- Tăng augmentation (ColorJitter, RandomCrop)
- Thu thập thêm data (30 episodes decoupled)
- Fine-tune thêm 20 epochs

### Rủi ro 3: CPU real-time không đạt
**Giải pháp:**
- Giảm history_size: 3 → 2
- Giảm CEM samples: 1000 → 500
- Quantize model (INT8)

---

## CHECKLIST TRƯỚC ISEF

- [ ] Dataset v3_96 hoàn thành
- [ ] TinyViT + CfC train xong, val_loss < 1.0
- [ ] TinyViT + AR train xong, val_loss < 1.2
- [ ] Benchmark 5D hoàn thành, CfC thắng 4/5 dimensions
- [ ] T-01 đến T-06 đều PASS
- [ ] Robot thật grasp thành công 2 màu chai
- [ ] Video demo quay sẵn
- [ ] Poster ISEF hoàn thành với 5D comparison chart
