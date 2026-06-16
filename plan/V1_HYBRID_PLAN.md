# V1: Hybrid CfC+Attention — Benchmark

## Mục tiêu
Benchmark Hybrid CfC+Attention predictor vs LeWM (AR) pretrained checkpoint trên Push-T + TwoRoom.
Chứng minh: (1) Hybrid beat AR ở manipulation (Push-T), (2) CfC temporal fix AR drift ở long-horizon navigation (TwoRoom).

## Chiến lược

| Quyết định | Lý do |
|---|---|
| **Ko build 6-DOF arm sim cho V1** | SO-ARM100 tốn thời gian env engineering. Novelty ở predictor, ko phải environment. Để dành cho Social phase. |
| **Push-T + TwoRoom** | Data + checkpoint có sẵn từ LeWM. Push-T = manipulation, TwoRoom = navigation. Bao quát cả short/long horizon. |
| **Hybrid only vs AR** | CfC đã có T1-T4 ablation riêng. Hybrid gộp CfC temporal + AR action robustness. |
| **Social sau V1** | Xem `project_logbook.md` — Social Type 1 (1 cam overhead) trước, Type 2 (3 cam) nâng cấp. |

## Phase 1: Download data + checkpoint (1 ngày)

Từ HF Hub collection `quentinll/lewm`:
- `quentinll/lewm-pusht` — dataset + model checkpoint
- `quentinll/lewm-tworooms` — dataset + model checkpoint

Dùng `swm.data.load_dataset()` (Lance format) và `swm.policy.AutoCostModel()` để load pretrained AR checkpoint.

## Phase 2: Implement Hybrid predictor (2 ngày)

### Kiến trúc

```python
class HybridCfCPredictor(nn.Module):
    """
    Attention action buffer (3-frame window) + CfC temporal backbone.
    Hybrid = CfC temporal (long rollout 3.3x + variable Δt 1.8x)
           + AR action robustness (CEM OOD noise)
    """
    def __init__(self, cfc_hidden=96, attn_heads=4, attn_window=3):
        self.cfc = CfCPredictorV2(num_frames=6, hidden_dim=cfc_hidden, ...)
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=attn_heads, batch_first=True)
    
    def forward(self, emb, act_emb):
        # act_emb: (B, T, D) — 3-frame action context
        # Attention: action → smoothed_action (phân tán OOD noise như AR's 24-dim stack)
        smoothed, _ = self.attn(act_emb[:, -1:], act_emb, act_emb)
        # CfC step: emb + smoothed_action → next_latent (ODE ổn định)
        return self.cfc(emb, smoothed)
```

### Cần thêm vào LeWM codebase

| File | Thay đổi |
|---|---|
| `module.py` | Thêm `HybridCfCPredictor` class (~60 dòng) |
| `config/train/model/vit_tiny_hybrid.yaml` | Config mới |
| `jepa.py` | Forward pass handle Hybrid (dùng CfC step) |

## Phase 3: Train Hybrid (1 ngày Colab T4)

| Task | Epochs | Batch | LR | SS | SIGReg | Time |
|---|---|---|---|---|---|---|
| Push-T | 100 | 64 | 3e-4 | 0.3 | 0.05 | ~3h |
| TwoRoom | 100 | 128 | 3e-4 | 0.3 | 0.05 | ~2h |

## Phase 4: Benchmark (1 ngày)

### Push-T (manipulation, short-horizon)

| Metric | LeWM (AR) paper | Hybrid (kỳ vọng) |
|---|---|---|
| Success rate | 96% | ≥ 96% |
| Planning speed | 0.98s | ≈ (cùng CEM H=5) |

### TwoRoom (navigation, long-horizon)

| Metric | LeWM (AR) paper | Hybrid (kỳ vọng) |
|---|---|---|
| Success rate | 87% | **> 90%** |
| Lý do | CEM H=5 ko thấy cửa | CfC hidden state nhớ direction |

## Timeline

| Phase | Thời gian |
|---|---|
| P1: Download data + checkpoint | 1 ngày |
| P2: Implement Hybrid predictor | 2 ngày |
| P3: Train Hybrid (Push-T + TwoRoom) | 1 ngày |
| P4: Benchmark + report | 1 ngày |
| **Tổng** | **~5 ngày** |

## Social (sau V1)

Xem roadmap đầy đủ ở `project_logbook.md`:

```
V1 (tháng 6-8)  →  Social T1 (tháng 9-10)  →  Social T2 (tháng 11-12)
                          |                        |
                    1 cam overhead            3 cam (overhead + 2 ego)
                    joint latent              cross-attn multi-view
                    CLIP goal                 CLIP goal
                    2×SO-ARM100               FK + ReadPos support
```
