# 📓 NHẬT KÝ LÀM VIỆC CHUNG (PROJECT LOGBOOK)
> **Dự án:** LeWM — Bionic Hand World Model
> **Mục tiêu:** World model cho robotic hand grasp (8 DOF) — so sánh CfC (ODE-RNN) vs AR (Transformer) predictor

---

## ⚙️ 1. THÔNG SỐ TOÀN CỤC & MÔI TRƯỜNG

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| **Hardware** | i7-1165G7, 8GB RAM, Intel Iris Xe | CPU inference, Colab T4 training |
| **OS** | Windows 11 | |
| **Python** | 3.14.0 | |
| **Dataset** | `bionic_hand_dataset_v3_96.h5` | 96x96, zstd compressed |
| **Dataset size** | ~5000 frames | 8-DOF action, 3 RGB channels |
| **History size** | 3 frames | |
| **Num preds** | 3 frames | |
| **Frameskip** | 3 | 15fps raw → 5fps effective |
| **Embed dim** | 32 | |
| **Encoder** | TinyViT (4 layers, 64 hidden) |  |
| **Action encoder** | Embedder (8→32) | |
| **Loss** | Pred MSE + SIGReg (λ=0.05) | |
| **Config variants** | | |
| AR predictor | hidden_dim=64, depth=1, heads=2, mlp_dim=96, dim_head=16 | 52,448 params |
| CfC V1 predictor | hidden_dim=64, backbone_layers=1, backbone_units=64 | 27K params |
| CfC V2 predictor | hidden_dim=96, backbone_layers=1, backbone_units=96 | 55,808 params |
| CfC V3 predictor | hidden_dim=96, backbone_layers=1, backbone_units=96 | 55,808 params |
| **CfC 128/2/128** | hidden_dim=128, backbone_layers=2, backbone_units=128 | 111,392 params — **gấp đôi AR, sai** |

---

## 🛠️ 2. TRẠNG THÁI VẬT LÝ & CƠ HỌC (CALIBRATION)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| **Cổng Serial** | COM13 | |
| **Baudrate** | 1Mbps | |
| **Servos** | [1,2,4,5,6,7,8,9] | 3 fingers × 2 servos |

---

## ⚠️ BÀI HỌC KINH NGHIỆM (TỪ RESEARCH LOGBOOK)

### CẦN NÉ (7 Rules từ CfC Post-Mortem)
1. **Không bao giờ** train/eval CfC với batch-style predict (nhiều frames 1 lúc)
2. **Không dùng** `timespans=None` nếu muốn ODE advantage
3. **Không so sánh** CfC với AR trên task short fixed-step — CfC lợi thế ở variable Δt và long rollout
4. **Không lấy** AR evaluation code rồi sửa surface — CfC cần evaluation loop hoàn toàn khác
5. **Check `num_frames=1`** trong config — CfC không phải Transformer
6. **Không hardcode params** trong model_loader — đọc từ checkpoint header
7. **Không import model_class rồi tự instantiate** — dùng hydra instantiate từ config gốc

### Key Insights
- CfC là RNN/ODE: nhận 1 frame + hidden state, predict 1 frame. Ko copy AR's batch predict.
- Phase 1 build hidden state từ history, Phase 2 predict future — hidden state CARRY xuyên suốt
- Action index trong dataset: `act_emb[t]` = action tại frame t (dẫn từ frame t → frame t+1)
- Bug phổ biến: dùng `act_emb[:, ctx_len + t]` sai, đúng phải là `act_emb[:, feed_idx]`
- CfC ODE interpolation: `x(t) = σ(-f·t)g + (1-σ(-f·t))h` — cho phép predict ở ARBITRARY t

---

## 🔬 3. LỊCH SỬ PHÁN QUYẾT KIỂM THỬ THỰC TẾ (TEST VERDICTS)

| Test | Model | pred_loss | Ghi chú |
|---|---|---|---|
| Fair eval | AR epoch 10 | 0.00131 | Baseline — best so far |
| Fair eval | CfC V2 epoch 30 | 0.00304 | 2.33x worse than AR |
| Zero-action | AR | 560x degrade | Mạnh action conditioning |
| Zero-action | CfC V2 | 70x degrade | Yếu hơn |
| Speed | CfC V2 | 0.69ms | 3x faster than AR |
| Speed | AR | 2.03ms | |

---

## 📝 4. NHẬT KÝ THAY ĐỔI CHI TIẾT (CHANGELOG)

### [2026-06-07] - Config param fix: V2+V3 128/2/128 → 96/1/96 + nhiều bug
* **Người thực hiện:** AI Engineer & User
* **Nội dung thay đổi:**
  1. **Phát hiện config sai:** `vit_tiny_cfc_v2.yaml` và `vit_tiny_cfc_v3.yaml` ghi `hidden_dim=128, backbone_layers=2, backbone_units=128` → 111K params (gấp đôi AR 52K). Checkpoint thực tế là 96/1/96 = 55.8K.
  2. **Action index bug (4 file):** Phase 2 dùng `act_emb[:, ctx_len+t]` (action 3,4,5) thay vì `act_emb[:, feed_idx]` (action 2,3,4). CfC nhận (f2, a3) thay vì (f2, a2) — action pairing sai.
  3. **Action encoder input_dim sai:** Notebook ghi `frameskip * action_dim = 24`, nhưng data thực tế trả action (8,) single-frame (GPUDataset, không có SeqDataset). Checkpoint xác nhận Conv1d(8,8). Sửa về `input_dim=8`.
  4. **Bài học:** Không assumption config file đúng — verify bằng checkpoint weights. Không nhân frameskip vào action_dim cho CfC.
  5. Sửa: `vit_tiny_cfc_v2.yaml`, `vit_tiny_cfc_v3.yaml`, `train_colab_cfc_v3.ipynb`, `evaluate_fair.py`, `model_loader.py`, `evaluate_all_epochs.py`

### [2026-06-07] - Fix action index bug CfC V3 + caveman config
* **Người thực hiện:** AI Engineer
* **Nội dung thay đổi:**
  1. Phát hiện bug action index sai trong tất cả CfC V3 code: Phase 2 dùng `act_emb[:, ctx_len + t]` (action 3,4,5) thay vì `act_emb[:, feed_idx]` (action 2,3,4) → CfC nhận (f2, a3) thay vì (f2, a2)
  2. Sửa 4 file: `train_colab_cfc_v3.ipynb`, `evaluate_fair.py`, `model_loader.py`, `evaluate_all_epochs.py`
  3. Cài caveman plugin cho opencode (fixed installer bug thiếu `help`+`stats` commands)
  4. AGENTS.md rewrite: Tiếng Việt ưu tiên #1, caveman luôn bật, quy tắc 1-7 rút gọn
  5. Hợp nhất split paths (XDG vs AppData) — đồng bộ config cả 2 nơi

### [2026-06-07] - Cài MCP memory server + logbook skill
* **Người thực hiện:** AI Engineer & User
* **Nội dung thay đổi:**
  1. Tạo `opencode.json` với MCP memory server
  2. Tạo skill `.opencode/skills/logbook-manager/SKILL.md` — quản lý logbook tự động
  3. Tạo `plan/project_logbook.md` — logbook chuẩn hóa cho dự án

### [2026-06-07] - CfC ODE research + logbook cleanup
* **Người thực hiện:** AI Engineer
* **Nội dung thay đổi:**
  1. Xác nhận CfC ODE interpolation capability (Nature MI 2022 Eq.4)
  2. Ghi phát hiện vào `research_logbook.md`
  3. Sửa lỗi: logbook bị duplicate section (CfC post-mortem)

### [2026-06-07] - CfC V2 evaluation + post-mortem
* **Người thực hiện:** AI Engineer & User
* **Nội dung thay đổi:**
  1. Fix `model_loader.py`: hardcoded params (hidden_dim=64→96, action_dim=24→8)
  2. Fix `evaluate_fair.py`: CfC action key, speed test, evaluate all epochs
  3. Run fair eval: CfC V2 best epoch 30 (pred_loss=0.00304) vs AR epoch 10 (0.00131)
  4. Post-mortem: CfC dùng sai architecture (batch-style predict, timespans=None)
  5. 7 "cần né" rules documented

### [2026-06-07] - CfC V2 training (Colab)
* **Người thực hiện:** User
* **Nội dung thay đổi:**
  1. Train CfC V2 100 epochs trên Colab T4 (hidden_dim=96, backbone=1×96)
  2. Download checkpoints về `MODELS/cfc_models/`
  3. Config: CfC predictor 3→3 (batch-style, suboptimal)

### [2026-06-07] - CfC V1 training + evaluation
* **Người thực hiện:** User
* **Nội dung thay đổi:**
  1. Train CfC 100 epochs (hidden_dim=64, backbone=1×64)
  2. Evaluate với `evaluate_all_epochs.py`
  3. Phát hiện CfC V1 thua AR (vái)

### [2026-06-07] - AR training + baseline
* **Người thực hiện:** User
* **Nội dung thay đổi:**
  1. Train AR predictor 100 epochs trên Colab
  2. Epoch 10 best: pred_loss=0.00131
  3. Baseline cho tất cả so sánh sau này
