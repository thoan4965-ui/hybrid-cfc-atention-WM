# TASK BREAKDOWN: CfC vs Transformer AR - ISEF 2026
**Kỹ sư trưởng:** AI Architect  
**Người thực hiện:** AI Coder  
**Ngày:** 2026-06-07  
**Mục tiêu:** Thay CfC vào thay cho Transformer AR, chứng minh CfC tốt hơn trên 5 dimensions

---

## QUY TẮC LÀM VIỆC

1. Làm đúng thứ tự task. Không nhảy cóc.
2. Mỗi task có TEST GATE - phải pass mới chuyển task tiếp.
3. Sau mỗi task, báo cáo: "DONE Task X.Y - [kết quả test]".
4. Nếu test gate FAIL, dừng lại báo cáo lỗi, không tự sửa.
5. Tất cả path dùng absolute path `d:/ai_training/...`.
6. Python dùng `d:\ai_training\.venv\Scripts\python.exe`.

---

## PHASE 1: RESIZE DATASET 224→96

### Task 1.1: Viết script resize data 🔵 `[DỄ - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/scripts/resize_data_v3.py`

**Việc cần làm:**
1. Đọc `d:/ai_training/bionic_hand_dataset_v2.h5`
2. Resize tất cả frames từ 224×224 xuống 96×96 dùng `cv2.resize(..., (96, 96), interpolation=cv2.INTER_AREA)`
3. Giữ nguyên metadata: `ep_len`, `ep_offset`, `action`, `proprio`
4. Ghi ra `d:/ai_training/bionic_hand_dataset_v3_96.h5`
5. In ra: số episodes, tổng frames, kích thước file (MB)

**Cấu trúc H5 output:**
```
pixels      (8900, 96, 96, 3)  uint8
action      (8900, 8)          float32
proprio     (8900, 8)          float32
ep_len      (89,)              int32
ep_offset   (89,)              int64
```

**TEST GATE:**
```bash
d:\ai_training\.venv\Scripts\python.exe -c "
import h5py, numpy as np
f = h5py.File('d:/ai_training/bionic_hand_dataset_v3_96.h5', 'r')
assert f['pixels'].shape == (8900, 96, 96, 3), f'Wrong pixel shape: {f[\"pixels\"].shape}'
assert f['action'].shape == (8900, 8)
assert len(f['ep_len']) == 89
assert f['ep_offset'][-1] + f['ep_len'][-1] == 8900
print('PASS: All assertions passed')
f.close()
"
```

---

### Task 1.2: Tạo data config cho dataset 96×96 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/config/train/data/bionic_hand_v2.yaml`

**Việc cần làm:**
1. Copy file cũ, sửa `dataset_name` trỏ đến `d:/ai_training/bionic_hand_dataset_v3_96.h5`

**TEST GATE:**
```python
# Đọc config và xác nhận dataset_name đúng
```

---

## PHASE 2: XÂY DỰNG KIẾN TRÚC MỚI

### Task 2.1: Viết class TinyViT trong module.py 🔴 `[KHÓ - Mô hình lớn]`
**File cần sửa:** `d:/ai_training/le-wm/module.py`

**Việc cần làm:** Thêm class `TinyViT` vào cuối file module.py.

**Đặc tả kỹ thuật:**
```python
class TinyViT(nn.Module):
    """
    Custom tiny ViT cho task grasp 8 DOF.
    Train từ scratch, không pretrained.
    
    Args:
        img_size: int = 96
        patch_size: int = 8
        num_layers: int = 4
        hidden_dim: int = 64
        num_heads: int = 4
        mlp_dim: int = 256
        output_dim: int = 32  # latent dim
        dropout: float = 0.0
    
    Output shape: (B, output_dim)  # (B, 32)
    """
```

**Luồng forward:**
1. **Patch Embed:** `nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)` — output (B, 64, 12, 12)
2. **Flatten + Transpose:** thành (B, 144, 64) — 144 tokens
3. **CLS Token:** Thêm 1 learnable CLS token → (B, 145, 64)
4. **Position Embedding:** Learnable pos_embed (1, 145, 64) cộng vào
5. **Transformer Blocks:** 4 lớp, mỗi lớp có:
   - LayerNorm → MultiHeadAttention (4 heads, dim_head=16) → residual
   - LayerNorm → MLP (64 → 256 → 64, GELU) → residual
6. **Final LayerNorm** trên CLS token
7. **Output Projection:** `nn.Linear(64, output_dim)` → (B, 32)

**Không dùng:**
- Dropout (dataset nhỏ, overfit có lợi)
- AdaLN modulation (không cần conditioning)
- Mask token
- Pretrained weights

**Số params mục tiêu:** 240K - 260K

**TEST GATE:**
```python
from module import TinyViT
import torch

model = TinyViT(img_size=96, patch_size=8, num_layers=4, 
                hidden_dim=64, num_heads=4, mlp_dim=256, output_dim=32)
x = torch.randn(2, 3, 96, 96)
out = model(x)

# Test 1: Output shape
assert out.shape == (2, 32), f"Expected (2, 32), got {out.shape}"

# Test 2: Parameter count
n_params = sum(p.numel() for p in model.parameters())
assert 240_000 <= n_params <= 260_000, f"Expected 240K-260K params, got {n_params}"

# Test 3: Trainable
assert all(p.requires_grad for p in model.parameters())

# Test 4: Forward stability (no NaN)
assert not torch.isnan(out).any()

print(f"PASS: Output shape {out.shape}, Params {n_params:,}")
```

---

### Task 2.2: Viết class CfCPredictor mới (CFC_V2) 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/module.py`

**Việc cần làm:** Thêm class `CfCPredictorV2` hoặc sửa `CfCPredictor` hiện tại để support param mới.

**Đặc tả kỹ thuật:**
```python
class CfCPredictorV2(nn.Module):
    """
    CfC predictor cho latent dim nhỏ (32d).
    
    Args:
        input_dim: int = 32      # latent dim (D)
        hidden_dim: int = 64     # CfC units
        output_dim: int = 32     # output latent dim
        action_dim: int = 32     # action embedding dim
        backbone_layers: int = 1 # giảm từ 2
        backbone_units: int = 64 # giảm từ 256
        use_mixed: bool = False  # chỉ CfC, không LSTM+CfC
    """
```

**Lưu ý critical:**
- Wiring mặc định khi `input_size` là int: `FullyConnected` (ncps tự tạo)
- `batch_first=True` để khớp (B, T, D)
- **KHÔNG hardcode `backbone_layers=2, backbone_units=256` như cũ**
- Input = concat(x, c) → (B, T, 32+32=64)
- Output = (B, T, 32)

**Số params mục tiêu:** 40K - 45K

**TEST GATE:**
```python
from module import CfCPredictorV2
import torch

model = CfCPredictorV2(
    input_dim=32, hidden_dim=64, output_dim=32, action_dim=32,
    backbone_layers=1, backbone_units=64
)
x = torch.randn(2, 3, 32)  # (B, T, D)
c = torch.randn(2, 3, 32)  # (B, T, A_emb)
out = model(x, c)

# Test 1: Output shape
assert out.shape == (2, 3, 32), f"Expected (2, 3, 32), got {out.shape}"

# Test 2: Parameter count
n_params = sum(p.numel() for p in model.parameters())
assert 40_000 <= n_params <= 45_000, f"Expected 40K-45K params, got {n_params}"

# Test 3: No NaN
assert not torch.isnan(out).any()

print(f"PASS: Output shape {out.shape}, Params {n_params:,}")
```

---

### Task 2.3: Cập nhật ARPredictor cho kích thước nhỏ 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/module.py`

**Việc cần làm:** Sửa tham số mặc định của ARPredictor hoặc thêm phiên bản mới.

**Đặc tả kỹ thuật - ARPredictor cho 32-d:**
```python
# Khi gọi:
ARPredictor(
    num_frames=3,
    input_dim=32,
    hidden_dim=64,
    output_dim=32,
    depth=2,        # giảm từ 6
    heads=4,        # giảm từ 16
    mlp_dim=128,    # giảm từ 2048
    dim_head=16,    # giảm từ 64
    dropout=0.0,
)
```

**Số params mục tiêu:** 45K - 55K

**TEST GATE:**
```python
from module import ARPredictor
import torch

model = ARPredictor(
    num_frames=3, input_dim=32, hidden_dim=64, output_dim=32,
    depth=2, heads=4, mlp_dim=128, dim_head=16
)
x = torch.randn(2, 3, 32)
c = torch.randn(2, 3, 32)
out = model(x, c)

# Test 1: Output shape
assert out.shape == (2, 3, 32)

# Test 2: Parameter count
n_params = sum(p.numel() for p in model.parameters())
assert 45_000 <= n_params <= 55_000, f"Expected 45K-55K params, got {n_params}"

print(f"PASS: Output shape {out.shape}, Params {n_params:,}")
```

---

### Task 2.4: Cập nhật Embedder (Action Encoder) cho 32-d 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/module.py`

**Việc cần làm:** Đảm bảo `Embedder` hỗ trợ tham số mới, không cần sửa code vì đã có tham số.

**Đặc tả khi gọi:**
```python
Embedder(
    input_dim=???,    # set động bởi train.py
    smoothed_dim=8,   # bằng input_dim nếu input_dim <= 24
    emb_dim=32,       # latent dim
    mlp_scale=2,      # giảm từ 4
)
```

**TEST GATE:**
```python
from module import Embedder
import torch

model = Embedder(input_dim=24, smoothed_dim=8, emb_dim=32, mlp_scale=2)
x = torch.randn(2, 3, 24)  # (B, T, 24) với frameskip=3
out = model(x)

assert out.shape == (2, 3, 32)
print(f"PASS: Output shape {out.shape}")
```

---

### Task 2.5: Cập nhật SIGReg cho 32-d latent 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/module.py`

**Việc cần làm:** Đảm bảo SIGReg được gọi với tham số mới (qua config).

**Đặc tả khi gọi:**
```python
SIGReg(knots=9, num_proj=256)  # scale theo latent=32
```

**TEST GATE:**
```python
from module import SIGReg
import torch

sigreg = SIGReg(knots=9, num_proj=256)
x = torch.randn(10, 4, 32)  # (T, B, D)
loss = sigreg(x)

assert loss > 0
print(f"PASS: SIGReg loss = {loss.item():.6f}")
```

---

## PHASE 3: TẠO CONFIG FILES

### Task 3.1: Tạo model config TinyViT + CfC 🔵 `[DỄ - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/le-wm/config/train/model/vit_tiny_cfc.yaml`

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
  _target_: module.CfCPredictorV2
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
  smoothed_dim: 8
  emb_dim: 32
  mlp_scale: 2

projector: null
pred_proj: null
```

**TEST GATE:**
```python
# Kiểm tra Hydra instantiate được model
import hydra
cfg = ... # load vit_tiny_cfc.yaml
model = hydra.utils.instantiate(cfg)
assert model.encoder.output_dim == 32
assert model.predictor.output_dim == 32
print("PASS: Model instantiated successfully")
```

---

### Task 3.2: Tạo model config TinyViT + Transformer AR (baseline) 🔵 `[DỄ - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/le-wm/config/train/model/vit_tiny_ar.yaml`

**Nội dung:** Copy từ Task 3.1, thay `predictor` thành ARPredictor.

```yaml
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
  dropout: 0.0
  emb_dropout: 0.0
```

**TEST GATE:** Giống Task 3.1.

---

### Task 3.3: Cập nhật root config lewm.yaml 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/config/train/lewm.yaml`

**Thay đổi cụ thể:**
```yaml
img_size: 96        # WAS: 224
embed_dim: 32       # WAS: 192

loss:
  sigreg:
    weight: 0.05    # WAS: 0.09, sẽ grid search
    kwargs:
      knots: 9      # WAS: 17
      num_proj: 256 # WAS: 1024

# Giữ nguyên:
# history_size: 3
# num_preds: 3
# trainer.max_epochs: 100
```

**TEST GATE:**
```python
from omegaconf import OmegaConf
cfg = OmegaConf.load('d:/ai_training/le-wm/config/train/lewm.yaml')
assert cfg.img_size == 96
assert cfg.embed_dim == 32
assert cfg.loss.sigreg.kwargs.knots == 9
print("PASS: Config updated")
```

---

## PHASE 4: CẬP NHẬT TRAIN.PY

### Task 4.1: Thêm augmentation pipeline 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/train.py`

**Vị trí chèn:** Sau dòng 176 (sau `transforms = [...]`).

**Code cần thêm:**
```python
# Augmentation pipeline (chỉ khi train)
if cfg.get('augmentation', True):
    from torchvision.transforms import v2
    aug_transform = v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.RandomRotation(degrees=5),
    ])
    # Wrap augmentation cho stable-pretraining pipeline
    transforms.append(
        dt.transforms.WrapTorchTransform(
            aug_transform, source='pixels', target='pixels'
        )
    )
```

**Lưu ý:** Augmentation chỉ áp dụng cho training set (DataLoader shuffle=True). Validation set không dùng augmentation.

**TEST GATE:**
```python
# Kiểm tra transform hoạt động
img = torch.rand(3, 96, 96)
aug = v2.Compose([v2.ColorJitter(hue=0.1), v2.RandomResizedCrop(96), v2.RandomRotation(5)])
out = aug(img)
assert out.shape == (3, 96, 96)
print("PASS: Augmentation works")
```

---

### Task 4.2: Xóa code remap_checkpoint_keys cho ViT 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/train.py`

**Việc cần làm:** Đánh dấu function `remap_checkpoint_keys` là deprecated, thêm comment: "# DEPRECATED: chỉ dùng cho ViT cũ, TinyViT không cần remap"

**Không xóa hẳn** — giữ lại để backward compatibility với checkpoint cũ.

---

### Task 4.3: Xử lý projector/pred_proj null 🔵 `[DỄ - Mô hình nhỏ]`
**File cần sửa:** `d:/ai_training/le-wm/jepa.py`

**Việc cần làm:** Kiểm tra dòng 26-27. `projector` và `pred_proj` đã có fallback `nn.Identity()` khi None. Code hiện tại đúng, **không cần sửa**.

**Xác nhận:** Xem dòng 26: `self.projector = projector or nn.Identity()` — OK.

**TEST GATE:**
```python
from jepa import JEPA
import torch.nn as nn
model = JEPA(None, None, None, projector=None, pred_proj=None)
assert isinstance(model.projector, nn.Identity)
assert isinstance(model.pred_proj, nn.Identity)
print("PASS: Identity fallback works")
```

---

## PHASE 5: CẬP NHẬT BUILD SCRIPTS (LOAD MODEL)

### Task 5.1: Tạo shared function load_model_for_inference 🔴 `[KHÓ - Mô hình lớn]`
**File cần tạo:** `d:/ai_training/le-wm/model_loader.py`

**Mục đích:** Tất cả scripts (run_mpc, task_02-06, evaluate_all_epochs, generate_goal_state) đều có code load_model trùng lặp. Tạo 1 hàm shared để tránh duplicate.

**Đặc tả:**
```python
def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """
    Tự động detect architecture từ checkpoint keys.
    
    Returns:
        model: JEPA instance (eval mode)
        metadata: dict with keys:
            - 'input_dim': action encoder input dim (8 or 24)
            - 'latent_dim': latent space dimension (32 or 192)
            - 'arch': 'tinyvit', 'resnet10t', or 'vit'
            - 'img_size': expected input image size (96 or 224)
    """
```

**Logic detect architecture:**
```python
keys = clean_state_dict.keys()

if any('.backbone.' in k for k in keys):
    arch = 'resnet10t'
    latent_dim = 192  # ResNet10t hiện tại
elif any('patch_embed' in k and 'encoder.' in k for k in keys):
    # TinyViT có conv patch_embed
    arch = 'tinyvit'
    # Detect latent_dim từ output_proj.weight
    latent_dim = clean_state_dict['encoder.output_proj.weight'].shape[0]
else:
    arch = 'vit'
    latent_dim = 192
```

**TEST GATE:**
```python
from model_loader import load_model_from_checkpoint

model, meta = load_model_from_checkpoint(
    'd:/ai_training/models/weights_resnet_moi.pt'
)
assert meta['arch'] == 'resnet10t'
assert meta['latent_dim'] == 192
print(f"PASS: Loaded {meta['arch']} with latent_dim={meta['latent_dim']}")
```

---

### Task 5.2: Cập nhật evaluate_all_epochs.py 🔴 `[KHÓ - Mô hình lớn]`
**File cần sửa:** `d:/ai_training/scripts/evaluate_all_epochs.py`

**Việc cần làm:**
1. Import `load_model_from_checkpoint` từ `model_loader`
2. Thay thế hàm `load_model()` (dòng 60-105) bằng wrapper gọi `load_model_from_checkpoint`
3. Cập nhật các dòng hardcode 192 → dùng `meta['latent_dim']`
4. Các dòng cần sửa:
   - Dòng 91-93: `CfCPredictor(num_frames=3, input_dim=192, ...)` → dùng `latent_dim`
   - Dòng 92: `Embedder(..., emb_dim=192)` → dùng `latent_dim`
   - Dòng 93: `MLP(projector_in_dim, 2048, 192, ...)` → nếu TinyViT, projector=None
   - Dòng 94: `MLP(192, 2048, 192, ...)` → nếu TinyViT, pred_proj=None
   - Dòng 108-111: `DecisionCEMController` hardcode `3, 192` → dùng HS và latent_dim

**LƯU Ý CRITICAL:** evaluate_all_epochs.py dùng `SIGReg(knots=17, num_proj=1024)` ở dòng 376. Khi latent_dim=32, cần dùng `knots=9, num_proj=256`. Sửa thành auto-detect từ latent_dim:
```python
if latent_dim == 32:
    sigreg_module = SIGReg(knots=9, num_proj=256)
else:
    sigreg_module = SIGReg(knots=17, num_proj=1024)
```

**TEST GATE:**
```bash
d:\ai_training\.venv\Scripts\python.exe -c "
from scripts.evaluate_all_epochs import load_model_from_checkpoint
# Test với model cũ (backward compat)
# Test với model mới khi có checkpoint
print('PASS: evaluate_all_epochs imports OK')
"
```

---

## PHASE 6: CẬP NHẬT TẤT CẢ TEST SCRIPTS

Các scripts này có code load_model trùng lặp. Cần cập nhật tất cả.

### Task 6.1: task_02_one_step_pred.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File:** `d:/ai_training/scripts/task_02_one_step_pred.py`

**Việc cần làm:**
1. Thay `load_model()` bằng gọi `load_model_from_checkpoint`
2. Các hằng số cần sửa từ 192 → `meta['latent_dim']`:
   - Dòng 93: `MLP(projector_in_dim, 2048, 192, ...)`
   - Dòng 93: `pred_proj=MLP(192, 2048, 192, ...)`
3. Dòng 89-91: `CfCPredictor`, `Embedder` → dùng latent_dim
4. ImageNet normalization giữ nguyên (dùng cho ảnh đã resize 96 hoặc 224)

**TEST GATE:** Chạy thử với checkpoint cũ (backward compat).

---

### Task 6.2: task_03_action_conditioning.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File:** `d:/ai_training/scripts/task_03_action_conditioning.py`

**Việc cần làm:** Giống Task 6.1.

---

### Task 6.3: task_04_goal_state_check.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File:** `d:/ai_training/scripts/task_04_goal_state_check.py`

**Việc cần làm:** Giống Task 6.1.

---

### Task 6.4: task_06_decision_test.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File:** `d:/ai_training/scripts/task_06_decision_test.py`

**Việc cần làm:**
1. Giống Task 6.1 cho phần load_model
2. `DecisionCEMController` (dòng 128): sửa `expanded_z_init = ... expand(..., 3, 192)` → `3, latent_dim`
3. Dòng 384: `z_real_seq = torch.cat(z_list, dim=0) # (T, 192)` → comment đúng dim

---

### Task 6.5: task_01_sanity.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File:** `d:/ai_training/scripts/task_01_sanity.py`

**Việc cần làm:** Đọc file, xem có hardcode 192 không. Nếu có → sửa.

---

### Task 6.6: generate_goal_state.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File:** `d:/ai_training/scripts/generate_goal_state.py`

**Việc cần làm:**
1. Giống Task 6.1
2. Dòng 184: `z_target_seq = info["emb"] # (1, 3, 192)` → comment đúng dim
3. Dòng 187: `z_target = z_target_seq[:, -1].clone() # (1, 192)` → comment đúng dim
4. Image resize: nếu img_size=96, cần resize ảnh trước khi encode. Code hiện tại đọc pixels từ H5 đã là 96×96 → OK.

---

## PHASE 7: CẬP NHẬT run_mpc.py

### Task 7.1: Tích hợp model loader mới vào run_mpc.py 🔴 `[KHÓ - Mô hình lớn]`
**File:** `d:/ai_training/scripts/run_mpc.py`

**Việc cần làm:**
1. Import `load_model_from_checkpoint` từ `model_loader`
2. Thay thế toàn bộ phần load model (dòng 250-317) bằng:
```python
from model_loader import load_model_from_checkpoint
model, meta = load_model_from_checkpoint(model_path)
latent_dim = meta['latent_dim']
input_dim = meta['input_dim']
arch = meta['arch']
img_size = meta['img_size']
```

3. Cập nhật tất cả hằng số 192 → `latent_dim`:
   - Dòng 95: `expanded_z_init = ... expand(..., 3, 192)`
   - Dòng 330: `z_target = torch.load(goal_path) # (1, latent_dim)`
   - Dòng 388-398: khởi tạo `z_history` và dummy frame

4. **Dummy frame** (dòng 388): dùng `img_size` thay vì 224
```python
dummy_frame = torch.zeros(1, 3, img_size, img_size)
```

5. **Camera resize** (dòng 436): dùng `img_size` thay vì 224
```python
frame_resized = cv2.resize(frame_cropped, (img_size, img_size))
```

6. Cập nhật `LatentCEMController` (dòng 92-96):
```python
expanded_z_init = z_init_history.unsqueeze(1).expand(current_bs, self.num_samples, 3, latent_dim)
```

7. Gán `model.arch = arch` và `model.latent_dim = latent_dim` để controller dùng.

**TEST GATE:**
```bash
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\run_mpc.py --real
# Kiểm tra: không crash, similarity hiển thị đúng
# Kiểm tra: loop time < 66ms
```

---

## PHASE 8: HUẤN LUYỆN

### Task 8.1: Train TinyViT + CfC 🔵 `[DỄ - Mô hình nhỏ]`
**Lệnh chạy trên Colab:**
```bash
python train.py \
    model=vit_tiny_cfc \
    data=bionic_hand_v3_96 \
    img_size=96 \
    embed_dim=32 \
    loss.sigreg.weight=0.05 \
    loss.sigreg.kwargs.knots=9 \
    loss.sigreg.kwargs.num_proj=256 \
    trainer.max_epochs=100 \
    output_model_name=vit_cfc \
    subdir=vit_cfc_exp
```

**TEST GATE (sau train):**
```python
# Loss < 1.0
# val_loss không diverge
# Có file weights_epoch_100.pt
```

---

### Task 8.2: Train TinyViT + Transformer AR (baseline) 🔵 `[DỄ - Mô hình nhỏ]`
**Lệnh:**
```bash
python train.py \
    model=vit_tiny_ar \
    data=bionic_hand_v3_96 \
    img_size=96 \
    embed_dim=32 \
    loss.sigreg.weight=0.05 \
    trainer.max_epochs=100 \
    output_model_name=vit_ar \
    subdir=vit_ar_exp
```

---

### Task 8.3: Grid search λ SIGReg cho CfC 🔵 `[DỄ - Mô hình nhỏ]`
**Lệnh (chạy tuần tự):**
```bash
for lambda in 0.001 0.01 0.05 0.1; do
    python train.py \
        model=vit_tiny_cfc \
        loss.sigreg.weight=$lambda \
        trainer.max_epochs=50 \
        output_model_name=vit_cfc_lambda_$lambda \
        subdir=vit_cfc_lambda_$lambda
done
```

**Chọn best λ:** λ có val_loss thấp nhất + separation cao nhất ở epoch 50.

---

## PHASE 9: BENCHMARK 5 DIMENSIONS

### Task 9.1: Viết script benchmark_speed.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/scripts/benchmark_speed.py`

**Việc cần làm:**
1. Load CfC model và AR model
2. Đo CPU time cho 1000 lần predict (H=5, batch=1)
3. Báo cáo: mean ± std (ms)
4. So sánh speedup = AR_time / CfC_time

**TEST GATE:**
```python
# CfC < 5ms/step
# AR < 15ms/step
# speedup >= 2x
```

---

### Task 9.2: Viết script benchmark_stability.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/scripts/benchmark_stability.py`

**Việc cần làm:**
1. Load cả 2 model
2. Với mỗi horizon H ∈ {1, 3, 5, 10}:
   - Rollout từ cùng start state
   - Tính MSE so với ground truth
3. Plot: MSE vs Horizon cho cả 2 model

**TEST GATE:**
```python
# H=5: CfC MSE < AR MSE * 0.8
# H=10: CfC MSE < AR MSE * 0.6
```

---

### Task 9.3: Viết script benchmark_accuracy.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/scripts/benchmark_accuracy.py`

**Việc cần làm:**
1. Load cả 2 model
2. Tính: MSE, Cosine Sim, Separation @ H=1 và H=5
3. So sánh

**TEST GATE:**
```python
# Separation > 0.01 cho cả 2
# H=5: CfC separation > AR separation
```

---

### Task 9.4: Viết script benchmark_smoothness.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/scripts/benchmark_smoothness.py`

**Việc cần làm:**
1. Load cả 2 model
2. Rollout 100 trajectories với H=10
3. Tính velocity variance = Var(||z_{t+1} - z_t||)
4. So sánh

**TEST GATE:**
```python
# CfC velocity variance < AR velocity variance / 2
```

---

### Task 9.5: Viết script benchmark_physics.py 🟡 `[TRUNG BÌNH - Mô hình nhỏ]`
**File cần tạo:** `d:/ai_training/scripts/benchmark_physics.py`

**Việc cần làm:**
1. Load cả 2 model
2. Chạy CEM planner trên 50 episodes
3. Tính smooth cost (ticks/bước hành động)
4. So sánh

**TEST GATE:**
```python
# CfC smooth_cost < 10 ticks/step
# CfC smooth_cost < AR smooth_cost
```

---

## PHASE 10: KIỂM THỬ TỔNG HỢP

### Task 10.1: Chạy full test suite cho CfC model 🔵 `[DỄ - Mô hình nhỏ]`
**Lệnh:**
```bash
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\task_01_sanity.py
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\task_02_one_step_pred.py
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\task_03_action_conditioning.py
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\task_04_goal_state_check.py
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\task_05_controller_unit_test.py
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\task_06_decision_test.py
```

**TEST GATE TỔNG:**
| Test | Ngưỡng PASS |
|------|------------|
| T-02 Cosine Sim | > 0.80 |
| T-03 Delta Cosine | > 0.05 |
| T-04 Separation | > 0.01 |
| T-05 Controller | PASS |
| T-06 Diff Dist | > 10 ticks + AI_ADDS_VALUE |

---

### Task 10.2: Chạy real robot test 🔵 `[DỄ - Mô hình nhỏ]`
**Lệnh:**
```bash
d:\ai_training\.venv\Scripts\python.exe d:\ai_training\scripts\run_mpc.py --real
```

**TEST GATE:**
1. Grasp thành công chai vàng
2. Grasp thành công chai đỏ (color generalization)
3. Không grasp khi không có chai
4. Loop time P95 < 66ms (đạt 15Hz)

---

## TỔNG KẾT

### Dependency graph
```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
                                        │
                                        ▼
Phase 5 ──► Phase 6 ──► Phase 7 ──► Phase 8
                                        │
                                        ▼
                              Phase 9 ──► Phase 10
```

### Những hố cần tránh (PITFALLS)

1. **PITFALL 1: Hardcode 192 ở mọi nơi**
   - Ảnh hưởng: TẤT CẢ scripts
   - Giải pháp: Dùng `meta['latent_dim']` từ `load_model_from_checkpoint`

2. **PITFALL 2: ViT remap_checkpoint_keys**
   - Khi load TinyViT checkpoint, `remap_checkpoint_keys` sẽ fail vì không có key `encoder.layer.X.attention...`
   - Giải pháp: Detect `arch='tinyvit'` và bỏ qua remap (dùng `if arch == 'vit': remap`)

3. **PITFALL 3: CfCPredictor mới vs cũ**
   - Checkpoint cũ có key `predictor.cfc.rnn.cell...`
   - Checkpoint mới (backbone_layers=1, backbone_units=64) có structure khác
   - Giải pháp: Không cần backward compat giữa predictor cũ và mới (train từ scratch)

4. **PITFALL 4: projector/pred_proj null**
   - Checkpoint cũ có key `projector.net.0.weight`, checkpoint mới (TinyViT) không có
   - Giải pháp: `strict=False` khi load, kiểm tra missing keys

5. **PITFALL 5: Image resize trong camera stream**
   - Camera capture ảnh gốc (640×480), crop theo camera_config, rồi resize
   - Code cũ resize về 224, code mới phải resize về 96
   - Cần đọc `img_size` từ `meta` thay vì hardcode

6. **PITFALL 6: goal_state.pt tương thích**
   - goal_state.pt cũ encode với img_size=224, latent_dim=192
   - Model mới cần img_size=96, latent_dim=32
   - Giải pháp: Chạy lại `generate_goal_state.py` với model mới

7. **PITFALL 7: SIGReg parameters**
   - `knots=17` và `num_proj=1024` là cho 192-d latent
   - Với 32-d latent phải dùng `knots=9` và `num_proj=256`
   - Dùng sai gây SIGReg loss quá nhỏ hoặc quá lớn

8. **PITFALL 8: Action normalization Z-score**
   - Các scripts dùng 2 bộ Z-score: 1 cho ViT, 1 cho ResNet
   - Cần thêm bộ thứ 3 cho TinyViT (dùng Z-score từ dataset V3)
   - Hoặc đơn giản: luôn đọc mean/std từ dataset H5 thay vì hardcode
