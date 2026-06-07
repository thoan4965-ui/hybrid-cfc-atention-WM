# Research Logbook — Bionic Hand LeWM Project

## Date: 2026-06-07
## Status: Training Phase — CfC completed (epoch 100, val/loss ~0.22), AR training in progress

---

## TOP 2 RESEARCH IDEAS (Current Focus)

### Idea 1: CfC + Transformer Hybrid for World Model Prediction
**Gap identified:** All existing CfC+Transformer hybrids are for classification/forecasting. **None exist for sequential prediction / embodied control / world models.**

**Key papers found (2025):**
- "Hybrid liquid NN + transformer encoder reservoir" — respiratory sound classification
- "Hybrid Transformer with LNN for energy forecasting" — building energy
- "CViTLNN: ViT + Liquid NN" — COVID-19 detection
- "Hybrid ConvNeXt-LNN" — satellite image classification

**Why it's unique:** First to apply CfC+Transformer hybrid to **JEPA-style prediction** where:
- Transformer captures spatial attention across image patches
- CfC captures continuous-time dynamics between frames
- Combined → better temporal coherence than pure Transformer, better spatial reasoning than pure CfC

**Feasibility:** 6/10 difficulty, 2-4 weeks prototype
**Resource needed:** Current setup (TinyViT + CfC + AR) already provides baselines

---

### Idea 2: LeWM + Social Model (Theory of Mind for Embodied AI)
**Gap identified:** Review paper "Modeling the Mental World for Embodied AI" (2025-2026) just defined the field. **Nobody has combined Liquid Neural Networks (CfC) with social world models.** All existing work uses standard Transformers or GNNs.

**Key papers found:**
- "Modeling the Mental World for Embodied AI: A Comprehensive Review" (2026) — defines MWM vs PWM
- "Latent Perspective-Taking via Schrödinger Bridge" (2026) — belief inference
- "AToM: Theory-of-Mind Human Motion Prediction" (2025) — ToM in HRI
- "Combo: Compositional world models for multi-agent cooperation" (2025 ICLR)
- "Embodied Multi-Agent Coordination by Aligning World Models Through Dialogue" (2026)

**Why GNN is used (and why we can beat it):**
GNN advantages: relational inductive bias, permutation invariance, message passing
GNN disadvantages: hard to train, no temporal dynamics, graph construction overhead

**Our angle:** CfC's ODE dynamics naturally model continuous social interaction without explicit graph construction. For 2-3 agents in continuous interaction, CfC cross-agent attention + liquid dynamics may outperform GNN.

---

## PRACTICAL SUB-DIRECTIONS (Doable with current resources)

### A. Self-Play with 2 Policy Copies
- Same CfC policy cloned → Agent A and Agent B share workspace
- Must coordinate: avoid collision, share resources, grasp different objects
- Surprising behavior expected: turn-taking, spatial partitioning

### B. Emergent Coordination
- Agent A grasps object X, Agent B grasps object Y
- No explicit communication channel
- CfC predictor must implicitly model "what will the other agent do?"

### C. Counterfactual Prediction (Proto-ToM)
- Query: "If I don't act, what will the other agent do?"
- Requires policy to run mental simulation of other agent's predictor
- First step toward Theory of Mind in liquid world models

---

## FRONTIER RESEARCH (Post-ISEF — Long term)

### Population-Based Training
- Multiple agents with different parameters
- Evolutionary pressure → emergent communication/language
- Requires: auto-curriculum engine, population size 10-100

### Open-Ended Environment
- No fixed task — agents self-create goals
- Requires: intrinsic motivation (curiosity, empowerment), environment auto-generation
- Evaluation challenge: meaningful vs chaos?

### Known Challenges
1. **Training instabilities:** Multi-agent non-stationary → gradient explosion, policy collapse
2. **Sim-to-real:** Emergent sim behavior may not transfer to real robot
3. **Time:** 10-100x more training than single-agent

---

## WHY THESE IDEAS MATTER

1. **CfC + Transformer** fills a niche: sequential embodied prediction with liquid dynamics
2. **Social LeWM** bridges physical AI and social intelligence — robots that understand humans
3. Both leverage existing codebase (LeWM + CfC/AR) → not from scratch
4. Both are small enough for single-researcher with Colab + CPU laptop

---

## IMPLEMENTATION ROADMAP (Post-ISEF Priority)

### Phase 1: CfC-Transformer Hybrid (3-5 tuần) — CHỌN LÀM TRƯỚC
**Goal:** Thay FFN trong Transformer block bằng CfC, xử lý temporal theo từng vị trí patch

**Architecture HybridBlock (chi tiết):**
```
Input: (B, T, N, D) — B=batch, T=6 frames, N=144 patches (12×12), D=dim

1. Spatial Attention (giữ nguyên):
   reshape → (B×T, N, D)
   self-attention per frame → mỗi patch "thấy" các patch khác trong cùng frame
   output: (B×T, N, D)

2. Thay FFN bằng Temporal CfC:
   reshape → (B×N, T, D)
   Với mỗi vị trí patch i (từ 1..N):
     CfC_ode processes [patch_i_frame_0, patch_i_frame_1, ..., patch_i_frame_T] 
     → học dynamics thời gian của CÙNG 1 vị trí không gian
   output: (B×N, T, D)

3. Reshape back → (B, T, N, D)
4. Residual + LayerNorm
```
→ Attention: **bộ não nhìn toàn cục** (spatial context)
→ CfC: **ít sai số tích lũy** (continuous temporal dynamics)
→ Cả 2 fusion trong 1 block = hybrid thật

**Tại sao khác Paper 2025:**
- Paper 2025: [Tran] → [LNN] nối tiếp (pipeline hybrid)
- Mình: Attention → + → CfC trong cùng block (architectural hybrid)

**Độ khó:** 6/10
- Code: Dễ (~3 ngày) — chỉ cần reshape tensor đúng
- Design: Trung bình — phải đảm bảo CfC nhận sequence đủ dài (T≥6)

**Kiểm chứng:** So pred_loss với baseline (TinyViT thuần + CfC predictor)
- Nếu giảm → contribution #1: "First architectural CfC-Transformer hybrid for embodied prediction"
- Nếu không → ablation study vẫn có giá trị

---

### Phase 2: Nhét Hybrid vào Social (3-4 tuần, sau Phase 1)
**Goal:** Áp dụng hybrid encoder cho 2-agent coordination

**Tại sao nhét được:**
- Hybrid encoder hiểu temporal tốt hơn → dự đoán quỹ đạo đối phương chính xác hơn
- Chỉ cần thay encoder cũ = encoder hybrid, giữ nguyên CfC predictor

**Setup PyBullet (tối giản, laptop-friendly):**
- 2 robot 4 DOF (x, y, z, gripper), workspace 1×1m
- 2 vật: bóng đỏ + lập phương xanh
- Observation: top-down 96×96 RGB
- Task: mỗi robot cầm 1 vật, tránh xung đột

**Training:**
- Self-play: 1 policy điều khiển cả 2 robot
- Data: (pixels, joint_action) — action gộp 8 dim
- Hybrid encoder → embedding (B, T, D)
- CfC predictor → dự đoán embedding tương lai
- Loss: pred_loss + sigreg

**Độ khó:** 7/10
- Data: Cần tự thu thập (PyBullet simulation)
- Environment: Thiết kế 2 robot + shared workspace
- Evaluation: Collision rate, success rate, emergent coordination?

**Lưu ý quan trọng:**
- Không claim "Theory of Mind" → claim "Implicit Multi-Agent Coordination" hoặc "Predictive Avoidance"
- CfC 252K params đủ cho predictive coordination, không đủ cho recursive belief
- Trung thực với reviewers = điểm cộng

---

### Phase 3: Benchmark + Viết (2-3 tuần)
**So sánh 3 model:**
1. Baseline: TinyViT thuần + CfC predictor
2. Hybrid: TinyViT-CfC hybrid + CfC predictor
3. AR: TinyViT + AR predictor (đang train)

**Metrics:**
- Single-agent: pred_loss, inference speed, stability
- Multi-agent: collision rate, coordination success

**Output:** Paper NeurIPS/CoRL workshop hoặc báo cáo ISEF nâng cấp

### Phase 2: Multi-Agent Social (3-4 tuần, song song hoặc sau Phase 1)
**Goal:** Áp dụng hybrid encoder cho 2-agent coordination

**Setup PyBullet:**
- 2 robot 4 DOF (x, y, z, gripper) trong workspace 1×1m
- 2 vật: bóng đỏ + lập phương xanh
- Observation: top-down 96×96 RGB
- Task: mỗi robot cầm 1 vật, tránh xung đột

**Training:**
- Self-play: 1 policy điều khiển cả 2 robot
- Data: (pixels, joint_action) — action gộp 8 dim
- Hybrid encoder → embedding (B, T, D)
- CfC predictor → dự đoán embedding tương lai
- Loss: pred_loss + sigreg + optional social loss

**Evaluation:**
- Collision rate vs baseline
- Success rate (cả 2 robot cầm đúng vật)
- Emergent behavior: có tự phân chia vật không?

### Phase 3: So sánh + Viết (2-3 tuần)
- Benchmark: CfC hybrid vs AR vs baseline (3 model)
- Single-agent: pred_loss, speed, stability
- Multi-agent: collision rate, coordination metric
- Viết paper/report

---

## DESIGN DECISIONS

### Vì sao không GNN?
GNN cần:
- Xây graph rõ ràng (agent nào nối agent nào)
- Message passing (O(N²))
- Không có temporal dynamics tự nhiên

CfC Social:
- Không graph (agent học từ quan sát)
- O(N) với N agent
- Temporal dynamics tích hợp (ODE)
- Chạy real-time CPU

### Vì sao không claim "Theory of Mind"?
- CfC 252K params không đủ capacity cho recursive belief
- Task đơn giản = predictive coordination, không phải ToM thật
- Reviewers sẽ bắt lỗi overclaim
- Đặt tên đúng: "Implicit Multi-Agent Coordination" hoặc "Predictive Avoidance"

---

## KỲ VỌNG THỰC TẾ

### Nếu thành công:
- Hybrid giảm pred_loss 10-20% so với baseline
- 2 robot tự tránh xung đột trong 80%+ episodes
- Paper ra được NeurIPS/CoRL workshop (nhỏ nhưng novel)

### Nếu thất bại:
- Hybrid không cải thiện → vẫn có ablation negative = contribution
- Multi-agent không emerge → vẫn có dataset + benchmark = contribution
- Không mất gì, codebase vẫn phát triển

---

---

## INVESTIGATION RESULTS (2026-06-07)

### Investigation v1 (torch.randn) — KHÔNG ĐÁNG TIN CẬY
- Dùng random noise thay vì data thật → kết quả vô nghĩa

### Investigation v2 (Real H5 data) — 3 VẤN ĐỀ UNFAIR

| Vấn đề | CfC | AR | Giải thích |
|--------|-----|----|-------------|
| **Params** | 27K | 52K | CfC chỉ bằng 1/2 AR |
| **Evaluation input** | 6 frames (có target!) | 3 frames (history only) | CfC được cho ĐÁP ÁN rồi predict lại |
| **Frame step** | Predict 3 frames cùng lúc | Predict từng frame autoregressive | Khác biệt kiến trúc, không phải unfair |

### Kết quả AR (công bằng):
- Normal loss: 0.001
- Zero-action: **560x** tệ hơn → AR phụ thuộc rất mạnh vào actions (PASS)
- Random: **679x** tệ hơn → AR học được dynamics thật (PASS)
- Shuffle: 0.95x → frameskip=3 (1.8s) đủ khác nhau, nhưng shuffle test cần đánh giá lại

### Kết quả CfC (KHÔNG CÔNG BẰNG — cần retrain):
- Normal loss: 0.44 (vs AR 0.001 — yếu hơn 400 lần)
- Zero-action: **0.85x** (tốt HƠN không có action?!) → không dùng action
- Random: **1.17x** (thua cả random)
- Shuffle: **1.00x** → không học temporal dynamics
- **NGUYÊN NHÂN:** CfC được feed 6 frames trong evaluation (có target), training chỉ nhận 3 frames

### VẤN ĐỀ THỨ 4 (CRITICAL): CfC ĐANG DÙNG SAI CÁCH

**CfC là RNN/ODE — phải nhận 1 frame, predict 1 frame, carry hidden state forward.**

Hiện tại đang cho CfC nhận 3 frames cùng lúc như Transformer → sai hoàn toàn architecture.

**Cách đúng CfC hoạt động:**
```
h0 = init
out1, h1 = cfc(x0, a0, h0)   → predict frame 1
out2, h2 = cfc(x1, a1, h1)   → predict frame 2 (h1 carry context)
out3, h3 = cfc(x2, a2, h2)   → predict frame 3 (h2 carry context)
```

**So sánh công bằng theo ARCHITECTURE:**

| | CfC (ODE/RNN) | AR (Transformer) |
|---|---|---|
| Input mỗi bước | 1 frame + hidden state | 3 frames (context window) |
| Predict mỗi bước | 1 frame | 1 frame |
| Context mechanism | Hidden state (accumulated) | Attention trên context window |
| Steps cần thiết | 3 forward pass (sequential) | 3 forward pass (autoregressive) |

**CfC theo cách ĐÚNG sẽ:**
- Không cần 3 history frames → chỉ cần 1 frame gần nhất + hidden state
- Hidden state carry toàn bộ history → không cần re-process old frames
- Đây chính là lợi thế CfC:** ODE hidden state = compressed history**

### KẾ HOẠCH RETRAIN CfC CÔNG BẰNG

**Config mới — dùng CfC ĐÚNG CÁCH:**
```yaml
predictor:
  _target_: module.CfCPredictorV2
  num_frames: 1              # ← RNN: nhận 1 frame mỗi bước
  input_dim: 32
  hidden_dim: 128            # ← tăng từ 64 lên 128
  output_dim: 32
  action_dim: 32
  backbone_layers: 2         # ← tăng từ 1 lên 2
  backbone_units: 128        # ← tăng từ 64 lên 128
  # CfC carry hidden state giữa các bước → không cần context window
```

**Training đúng cho CfC:**
```python
# Sequential prediction với hidden state
h = None
for t in range(T):
    out, h = predictor(emb[:, t], act_emb[:, t], h)
    pred_loss += (out - target[:, t]).pow(2).mean()
```

**Evaluation công bằng (SAU retrain):**
- CfC: 1 frame input + hidden state, predict sequential, carry hidden state
- AR: 3 frame context window, predict autoregressive
- Cả hai predict 3 future steps
- Metric: MSE, speed, robustness (camera jitter)

**Camera jitter test (FUTURE):**
- Drop random frames (0%, 10%, 20%, 50%) từ validation data
- CfC: dùng timespans vraiment (có Δt biến đổi)
- AR: positional encoding fixed → broken khi drop frame
- Hypothesis: CfC robust hơn với camera jitter

### CẦN LÀM TỐI NAY:

**Retrain CfC ĐÚNG CÁCH:**

**CfCPredictorV3 — dùng RNN đúng:**
```python
# Phase 1: Build hidden state từ history
h = None
for t in range(history_size):  # 1 hoặc 3 frames
    inp = torch.cat([pixel_emb[:, t], act_emb[:, t]], dim=-1).unsqueeze(1)
    _, h = self.cfc(inp, h)

# Phase 2: Predict future (giống AR, từng bước)
predictions = []
for t in range(num_preds):
    inp = torch.cat([last_emb, act_emb[:, history_size + t]], dim=-1).unsqueeze(1)
    out, h = self.cfc(inp, h)
    predictions.append(out)
    last_emb = out  # hoặc teacher forcing
```

**Config retrain:**
- `backbone_layers=2, backbone_units=128` (~55K params, matching AR 52K)
- `history_size=3` (giống AR, vì cần đủ context ban đầu)
- CfC carry hidden state giữa các bước (đúng RNN nature)
- Predict 1 bước/lần, giống AR
- Same encoder (TinyViT), same data, same epochs

**So sánh công bằng SAU retrain:**
- CfC: 3 frames → build hidden → predict 1 step + carry hidden → repeat
- AR: 3 frames → attention → predict 1 step → shift window → repeat  
- Cả hai: 3 forward passes, predict 1 bước/lần
- Metric: MSE, speed, action-conditioning, camera jitter robustness

---

## POST-MORTEM: CfC V2 Evaluation (2026-06-07)

### MỤC TIÊU
Retrain CfC với params ngang AR (55K ≈ 52K), evaluation fair: 3 context → 3 future.

### KẾT QUẢ
- CfC V2 epoch 30 best: pred_loss = 0.00304
- AR epoch 10: pred_loss = 0.00131
- AR vẫn thắng 2.33x

### NHỮNG GÌ ĐÃ SỬA SO VỚI V1
✓ Params: 27K → 55.8K (ngang AR 52K)
✓ Dtype: uint8 → float32 CHW
✓ Transform: v2.ToImage skip → permute load time
✓ Action encoder: input_dim=24 (frameskip×action_dim) → 8
✓ SeqDataset: tạo sequence (T, C, H, W) cho encoder.encode()

### NHỮNG GÌ VẪN SAI (CRITICAL)

#### 1. CfC vẫn bị dùng sai Architecture
**Vấn đề:** CfC là RNN/ODE — nhận **1 frame + hidden state**, predict 1 frame.
Nhưng training code cho predictor nhận **3 frames cùng lúc**:
```python
pred = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])  # SAI!
```
CfC (PyTorch `CfC` class) khi nhận `(B, T, D)` tự động unroll nội bộ, không carry hidden state giữa history và prediction phase.

**Hậu quả:** 
- CfC không được dùng hidden state đúng cách
- Không có sequential processing — giống MLP hơn RNN
- Mất lợi thế ODE (không có timespans, không có hidden state carry)

**Fix cần thiết (CfC V3 / CfC đúng cách):**
```python
# Phase 1: Build hidden state
h = None
for t in range(history_size):
    inp = torch.cat([emb[:, t], act_emb[:, t]], dim=-1)
    _, h = self.cfc(inp.unsqueeze(1), h)  # (B, 1, D)

# Phase 2: Predict autoregressive
preds = []
for t in range(num_preds):
    inp = torch.cat([last_emb, future_act[:, t]], dim=-1)
    out, h = self.cfc(inp.unsqueeze(1), h)
    preds.append(out)
```

#### 2. `timespans=None` — Không dùng ODE advantage
CfC's ODE advantage là variable Δt (robust khi camera jitter/dropped frames).
Nhưng training luôn set `timespans=None` → CfC hoạt động như vanilla RNN.

**Fix:** Set `timespans = torch.linspace(0, 1, T)` cho uniform spacing,
hoặc giá trị thực tế khi test camera jitter.

#### 3. `num_frames=6` không đúng với RNN nature
`num_frames` trong config chỉ là số frame tối đa predictor có thể nhận,
nhưng CfC thực chất chỉ cần `num_frames=1` (RNN step).

#### 4. Evaluation vẫn cho CfC 3 frames 1 lúc
```python
pred_emb = predictor(history_emb, history_act)  # (1, 3, D) SAI!
```
Đúng phải là loop 3 lần, mỗi lần 1 frame + hidden state.

### NGUYÊN NHÂN GỐC RỄ
- **Không tách biệt được CfC (RNN) vs AR (Transformer) evaluation logic**
- Copy evaluation code từ AR sang CfC mà không đổi architecture pattern
- `forward_fn` trong training design cho AR, không cho CfC sequential
- `model_loader.py` dùng chung cho cả 2, nhưng `predict_cfc` trong evaluate_fair.py
  vẫn gọi predictor batch-style thay vì loop + hidden state

### CẦN NÉ
1. **Không bao giờ** train/eval CfC với batch-style predict (nhiều frames 1 lúc)
2. **Không dùng** `timespans=None` nếu muốn ODE advantage
3. **Không so sánh** CfC với AR trên task short fixed-step — CfC lợi thế ở variable Δt và long rollout
4. **Không lấy** AR evaluation code rồi sửa surface — CfC cần evaluation loop hoàn toàn khác
5. **Check `num_frames=1`** trong config — CfC không phải Transformer
6. **Không hardcode weight/bias/config trong model_loader** — mỗi checkpoint có architecture params khác nhau.
   - `model_loader.py` hardcode: `hidden_dim=64, backbone_units=64, action_encoder.input_dim=24` 
   - CfC V2 train với: `hidden_dim=96, backbone_units=96, action_encoder.input_dim=8`
   - Nếu load sai → RuntimeError vì `size mismatch`.
   - **Fix:** model_loader phải đọc params từ checkpoint header thay vì hardcode, hoặc support nhiều variant rõ ràng (cfc_v1, cfc_v2, cfc_v3).
7. **Không import model_class rồi tự instantiate với params guess** — luôn dùng hydra instantiate từ config gốc hoặc đọc từ checkpoint metadata.
```

## KẾ HOẠCH CfC V3 — DÙNG CfC ĐÚNG NHẤT

Xem `D:\ai_training\plan\cfc_v3_plan.md`
