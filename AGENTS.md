# LeWM Project — AGENT CONTEXT

## ⚡ MASTER PROTOCOL — 3 pillars of truth
**Mọi kết luận, quyết định, số liệu trong dự án này phải đáp ứng đủ 3 điều kiện:**

1. **📐 Lý thuyết (Theory)** — Hiểu rõ nền tảng toán học/khoa học, không đoán mò
2. **📄 Paper/Số liệu (Paper & Data)** — Trích dẫn từ paper gốc hoặc thực nghiệm, không tự bịa
3. **🧪 Test thực tế (Empirical)** — Chạy thử, có số liệu cụ thể, không suy luận suông

**Thiếu 1 trong 3 → kết luận chưa chắc chắn → ghi rõ "chưa biết, cần X để kết luận"**

## Project Scale
- **V0:** 1 bionic hand 8-DOF real, grasp confirmed (REAL ROBOT — duy nhất)
- **V1:** Hybrid CfC+Attention TwoRoom — **78%** (T=16, heads=8 — sai lầm, đáng lẽ T=4)
- **V1.1:** Denoiser + λ sweep + heads=16 + **T=4** (Colab T4) — fix sai lầm V1
- **V2:** Mamba predictor — **SIMULATION** (future)
- **Social T1:** 1 overhead cam, joint latent, 2×SO-ARM100 (MuJoCo) — **SIMULATION**
- **Social T2:** 3 cam (overhead + 2 ego), cross-attn — **SIMULATION**
- **Hardware:** i7-1165G7 CPU, 8GB RAM, Iris Xe; free/cheap compute (Colab, Kaggle)

## Roadmap

```
V0 [done]     1 bionic hand 8-DOF real (grasp confirmed) — Sáng tạo trẻ 30/6
V1 [15/6-30/6] V1.1 Denoiser λ sweep T=4 + báo cáo Sáng tạo trẻ
V2 [7-8/2026]  Mamba+Attention predictor, fair comp vs CfC+Attn vs AR (T=4)
              TwoRoom → Push-T → Cube → Reacher, tất cả T=4
Social T1 [9-10/2026] 1 overhead cam, joint latent 
              2×SO-ARM100, MuJoCo, CLIP goal → ISEF
Social T2 [11-12/2026] 3 cam (overhead + 2 ego), cross-attn 
              → quốc gia + ISEF
```

## Strategy: Fair architecture comparison (fixed T=4)
**KO tìm optimal T cho từng architecture. Chốt T=4 cho tất cả.**
- LeWM paper dùng T=4 (history_size=3, num_preds=1) cho cả 4 benchmark
- Mọi architecture (AR, CfC+Attn, Mamba+Attn) dùng **cùng T=4, batch=128**
- Ai beat LeWM ở T=4 → architectural improvement thật, ko phải tuning artifact
- Lý do: encoder (TinyViT) là bottleneck compute 99% → T=4 vs T=16 ko ảnh hưởng speed
- CfC ko kịp error accumulation ở T=4 → đánh giá CfC đúng bản chất
- Fair comparison giữa các architecture mới ra novelty thật

## ⚖️ Nguyên tắc thiết kế: Tỉ lệ hybrid luôn 1:1
- **Mọi architecture hybrid (CfC+Attn, Mamba+Attn) phải giữ tỉ lệ CfC:Mamba ≈ Attention ≈ 1:1**
- Lý do: Attention chịu trách nhiệm spatial feature extraction, temporal model (CfC/Mamba) chịu trách nhiệm động học
- 1:1 = cân bằng giữa 2 nhiệm vụ, ko thằng nào áp đảo thằng nào
- Config hiện tại: backbone_units=384 → CfC=764K, Attention=787K → ratio ≈ 0.97:1 ≈ 1:1 ✅
- Nếu V2 Mamba thay CfC: giữ tỉ lệ params tương đương CfC cũ

## ⚡ CHỐT CỨNG — V2.1 dùng Mamba-2 (ko phải Mamba-3)
- Mamba-2 có wheel cho mọi CUDA 12-13 → `pip install mamba-ssm` 30 giây, ko build
- Với T=4, Mamba-2 ≈ Mamba-3 về chất lượng
- d_state=64 đủ cho tất cả task kể cả Social T=16-32
- Mamba-3 code giữ trong module.py để tham khảo, upgrade sau nếu cần

## V1 Plan chốt
- **Arch:** 6×{Self-Attn(AdaLN) → CfC(ODE)}, ~15.5M params
  - Attention: 28% predictor params (2.36M), Softmax (ko Linear), heads=8, dim_head=64
  - CfC(ODE): 51% predictor params (4.20M), backbone_units=384, cfc_hidden=256
  - AdaLN: 12%, còn lại là pos_embed + proj
- **T=4** (history_size=3, num_preds=1, frameskip=5) — giống LeWM paper
  - T=4 × frameskip=5 = 20 sim steps
  - Encoder (TinyViT) là bottleneck compute → T ko ảnh hưởng training speed
- **Data:** HF Hub (quentinll/lewm-*) → download trực tiếp
- **Train:** Colab T4 + Kaggle GPU (free). Vast RTX 5080 nếu có budget ($0.175/h)
- **Precision:** 16-mixed (FP16) trên T4; bf16 nếu có supported GPU (RTX 5080)
- **Checkpoint:** mỗi epoch upload .pt + .ckpt lên HF hhian/checkpoints
- **Resume:** auto tìm .ckpt trong spt.Manter cache hoặc run_dir/resume.ckpt
- **So sánh:** Fair comp: cùng T=4, batch=128. Kiến trúc nào tốt hơn mới là novelty.
- **SIGReg giữ nguyên** — ko đụng loss function
- **Hybrid với T=4 cho tất cả** — LeWM paper T=4, fair comparison mới ra novelty thật
- **OOD action: ko phá CEM** — cost ranking vẫn giữ dù scale lệch. CEM cần thứ tự (ranking), ko cần giá trị tuyệt đối chính xác.
- **Social phase:** giữ nguyên Softmax Attention — T=100 trên L40S 48GB vẫn OK với gradient checkpointing (không cần Linear Attention).

## Priorities
1. ✅ HybridCfCPredictor implemented
2. ✅ Config đồng bộ (seed, bf16, num_workers, lr)
3. ✅ Resume flow + HF upload (đã fix 3 critical bugs)
4. 🔄 V1.1 λ sweep — chờ budget 21/6 RTX 5080
5. 🔄 Báo cáo Sáng tạo trẻ (30/6) — V1.1 λ sweep + V0 real robot
6. 📅 V2 Mamba+Attention — fair comp vs CfC+Attn vs AR (T=4)
7. 📅 Push-T, Cube, Reacher — tất cả T=4

## Social Plan (preliminary — sau V1)
- **Giữ Softmax Attention** — L40S 48GB + gradient checkpointing xử lý T=100
- **Cần Linear Attention** khi nào? Khi T > 1000 (ruled out)
- **Social T1:** 1 cam overhead, joint latent, 2×SO-ARM100, CLIP goal
- **Social T2:** 3 cam (overhead + 2 ego), cross-attn

## Temporal flow (T=4)
- Mỗi block: Attention → CfC step-by-step qua T frames
- CfC hidden state carry temporal trong block (h₁→h₂→h₃)
- Temporal flow xuyên 6 blocks qua residual connection
- Training: teacher forcing (frame thật), hidden state reset mỗi batch
- Rollout: CFc ở T=4 ko kịp error accumulation → ổn định

## Mục tiêu: Fair architecture comparison (T=4)
- **Ai beat LeWM paper số ở T=4 → architectural improvement thật**
- Ko waste effort tìm optimal T cho từng architecture
- Nếu Mamba+Attention beat CfC+Attention ở T=4 → V2 win
- Nếu ko → CfC+Attention vẫn ngon, Mamba ko cần

## File Structure
- `le-wm-paper/`: LeWM paper gốc (lucas-maes/le-wm) — tham khảo, ko sửa
- `le-wm-v1/`: V1 clone sạch từ paper + HybridCfCPredictor
- `le-wm/`: V0 fork (modified cho bionic hand)
- `code-new/`: V0 bionic hand code
- `data/`: calib, config V0
- `MODELS/`: checkpoints cũ + V1
- `plan/`: logbook + plans

## CẦN NÉ (cập nhật)
- #59: Fair comp: cùng T=4, batch=128. Ai beat LeWM ở T=4 → improvement thật. Ko tìm optimal T.
- #60: Colab fix bug → chạy clean. T4 free cho V1.1 λ sweep. Vast/Kaggle khi có budget.
- **Ko waste effort tìm optimal T cho từng architecture.**
- **Mọi architecture dùng T=4, batch=128.**
- **Ko thay đổi T trong cùng 1 experiment — train T=16 eval T=4 = mismatch.**

## ⚠️ RULES NGHIÊM NGẶT (từ bug đã fix)
1. **Resume**: Dùng `glob *.ckpt`, không hardcode filename — Lightning đặt tên khác `epoch_{idx}.ckpt`
2. **HF upload**: Dùng `{subdir}/{run_name}/ep_{epoch}` — không hardcode, không env var
3. **Hidden state**: `_carry_mode=True` cho rollout, `False` cho train/val — phân biệt rõ
4. **Eval seed**: Tất cả task dùng `seed=3072` — đồng bộ với train seed
5. **Precision**: Luôn ghi precision vào config — bf16 trên GPU hỗ trợ, fp16 trên T4
6. **Checkpoint**: Mỗi epoch upload .pt + .ckpt lên HF — dùng path đúng từ đầu
7. **Dead params**: Kiểm tra config flow — `cfc_units` chạy 4 class ko dùng đến CfC
8. **Denoiser dim**: Phải configurable — không hardcode 192
9. **GitHub token**: Token hết hạn → push fail. Dùng token mới, hoặc SSH.
10. **HF token**: Token hết hạn → upload fail. Check hạn trước khi launch.

## Budget & Tài nguyên
- Tuần: 150-250k VND (~$6-10)
- Colab T4: nhiều acc (SIM Shopee 500đ/cái), ~5h/session
- Kaggle GPU: nhiều acc, ~12h/session
- Kaggle TPU v5e-8: có thể dùng nếu port code
- Vast RTX 5080: $0.175/h nếu có budget
- **Hiện tại: ko budget, dùng Colab/Kaggle free.**
- **Sáng tạo trẻ 30/6:** V0 + V1.1 λ sweep, Colab free.
- **ISEF tháng 9:** cần budget (~$72-120) cho V2 benchmark.

## Work Protocol
- **Output sau mỗi tool call** — tao báo real-time: đang làm gì, kết quả, bước tiếp theo
- **Ko dùng sub-agent code** — tao code + report trực tiếp
- Hermes Agent (qua MCP) cho research, web search (khi setup xong)
- Mày nói "dừng" → tao stop ngay, ko thinking
- Mày là manager — tao executor + reporter
