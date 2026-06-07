# CfC V3 Plan — Dùng CfC ODE Đúng Architecture

## Vấn Đề CfC V2

| Vấn đề | CfC V2 (sai) | CfC V3 (đúng) |
|---------|-------------|----------------|
| Input per step | 3 frames batch | 1 frame + hidden state |
| Predict | 3 frames 1 lần | 1 frame, loop 3 lần |
| Hidden state | Không có (CfC tự unroll) | Carry giữa history → predict |
| timespans | None (vanilla RNN) | Giá trị thực Δt |
| Evaluation | Copy từ AR (batch predict) | Loop sequential riêng |

## Training Loop Đúng

```python
# ── TRAINING CfC V3 ──
# Dataset trả về (T_total, C, H, W) với T_total = history_size + num_preds = 6
# emb: (B, T_total, D), act_emb: (B, T_total, A)

# Phase 1: Build hidden state từ history frames
h = None
for t in range(history_size):  # t = 0, 1, 2 (3 frames history)
    inp = torch.cat([emb[:, t:t+1], act_emb[:, t:t+1]], dim=-1)  # (B, 1, D+A)
    _, h = predictor.cfc(inp, h)

# Phase 2: Predict future frames, carry hidden state
pred_loss = 0
for t in range(num_preds):  # t = 0, 1, 2 (predict 3 frames)
    inp = torch.cat([emb[:, history_size + t:t+history_size + 1],
                     act_emb[:, history_size + t:t+history_size + 1]], dim=-1)
    out, h = predictor.cfc(inp, h)  # teacher forcing hoặc autoregressive
    pred_loss += (out.squeeze(1) - target[:, t]).pow(2).mean()
```

## Config Thay Đổi

```yaml
predictor:
  _target_: module.CfCPredictorV2
  num_frames: 1                # RNN: 1 frame/bước
  input_dim: 32
  hidden_dim: 96               # giữ nguyên như V2 (55K params)
  output_dim: 32
  action_dim: 32
  backbone_layers: 1
  backbone_units: 96
  # Không thay đổi — CfCPredictorV2 vẫn dùng được
  # Chỉ thay đổi cách gọi forward
```

## Evaluation Đúng

```python
# CfC V3 evaluation
def predict_cfc_v3(model_dict, pixel_emb, action_emb, history_size=3, num_preds=3):
    predictor = model_dict["predictor"]
    
    history_emb = pixel_emb[:, :history_size]    # (1, 3, D)
    history_act = action_emb[:, :history_size]   # (1, 3, D)
    future_act = action_emb[:, history_size:]    # (1, 3, D)
    target_emb = pixel_emb[:, history_size:]     # (1, 3, D)
    
    # Phase 1: Build hidden state
    h = None
    for t in range(history_size):
        inp = torch.cat([history_emb[:, t:t+1], history_act[:, t:t+1]], dim=-1)
        _, h = predictor.cfc(inp, h)
    
    # Phase 2: Predict future (teacher forcing với action thật)
    predictions = []
    for t in range(num_preds):
        inp = torch.cat([predictions[-1] if predictions else history_emb[:, -1:],
                         future_act[:, t:t+1]], dim=-1)
        out, h = predictor.cfc(inp, h)
        predictions.append(out)
    
    pred_target = torch.cat(predictions, dim=1)  # (1, 3, D)
    return pred_target, target_emb
```

## Ưu Điểm CfC V3 So Với V2

1. **Hidden state carry** — context tích lũy qua các bước, không giới hạn context window
2. **ODE đúng** — có thể dùng timespans thực tế khi test camera jitter
3. **Fair với AR** — cả 2 đều predict từng bước, AR attention window vs CfC hidden state
4. **Variable Δt sẵn sàng** — chỉ cần set timespans ≠ None là CfC hiểu thời gian

## Camera Jitter Test (Sau CfC V3)

CfC lợi thế nhất ở đây:
- AR: fixed positional encoding → thứ tự frame bị xáo trộn → hỏng
- CfC: `timespans` là giá trị thời gian thực → frame bị drop, Δt thay đổi → vẫn ổn

## Research: CfC ODE Hoạt Động Thế Nào

### timespans = None → CfC = Vanilla RNN
Khi `timespans=None` (như CfC V1/V2), CfC dùng time steps mặc định (uniform spacing).
Gating mechanism `σ(-f(x)·t)` vẫn chạy nhưng với t không phản ánh thời gian thực.
→ **Mất ODE advantage**, hoạt động như GRU/LSTM thường.

### timespans = actual_Δt → CfC = ODE Đúng
CfC có explicit time dependence trong formulation:
```
x(t) = σ(-f(x,I;θ)·t) ⊙ g(x,I;θ) + [1-σ(-f(x,I;θ)·t)] ⊙ h(x,I;θ)
```
- `t` là thời gian thực (hoặc Δt từ frame trước)
- `σ(-f(x,I)·t)` là gating phụ thuộc thời gian
- Khi t=0: output = g(x,I) (input dominates)
- Khi t→∞: output → h(x,I) (steady state)
- **t càng lớn → càng dựa vào hidden state, càng ít dựa vào input**

### Cách Set timespans Đúng
```python
# Camera 10Hz → mỗi frame cách 0.1s
timespans = torch.full((B, T, 1), 0.1, device=device)  # uniform

# Camera jitter: Δt thực tế giữa các frame
timespans = torch.tensor([Δt0, Δt1, Δt2, ...])  # mỗi Δt khác nhau

# CfC forward với timespans
out, h = self.cfc(inputs, timespans=timespans)
```

### Khi Nào CfC Thực Sự Thắng AR (Từ Paper)

| Scenario | AR (Transformer) | CfC (ODE) | Winner |
|----------|:-:|:-:|:-:|
| Short fixed-step (5 frames) | 0.0013 | 0.0030 | AR |
| Variable Δt (camera jitter) | Hỏng (pos encoding fixed) | Robust (timespans real) | **CfC** |
| Long sequence (>100 steps) | O(n²) memory explode | O(n) sequential | **CfC** |
| Real-time CPU inference | 2.03ms | 0.69ms | **CfC** |
| Irregular sampling | Hỏng | SOTA 98% accuracy | **CfC** (Nature MI 2022) |
| Missing data | Hỏng | Robust | **CfC** |
| Walker2D dynamics | Base | **+18%** outperform | **CfC** (Nature MI 2022) |

### SOTA Cho Long Sequence (2024-2025)
- **Mamba / S4 / LinOSS** — SSM O(n) training (parallel scan), O(1) inference
- **FACTS** (2025) — Graph-structured state-space cho world model
- **NextLat** (2025) — Transformer + latent dynamics RNN hybrid
- CfC giành cho **medium sequence (<200 steps) + irregular time**

### Quyết Định Cho Bionic Hand
- **Hiện tại**: AR epoch 10 thắng (0.0013 vs 0.0030) — task quá ngắn (5 frames fixed 10Hz)
- **CfC V3**: Retrain với hidden state carry + teacher forcing → có thể cải thiện nhưng không kỳ vọng thắng AR trên short task
- **Camera jitter test**: CfC V3 mới thực sự phát huy — cần test sau khi V3 chạy
- **Long-term**: Nếu cần predict >50 frames, chuyển hẳn sang CfC hoặc SSM
