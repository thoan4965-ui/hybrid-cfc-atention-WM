# LeWM Project — AGENT CONTEXT (CURRENT: V2.1 Mamba-2+Attention)

## ⚡ MASTER PROTOCOL — 3 pillars of truth + 3-step decision

**3 pillars of truth:** Mọi kết luận phải đủ Lý thuyết (Theory) + Paper/Data + Thực nghiệm (Empirical). Thiếu 1 → ghi "chưa biết, cần X".

**3-step decision:**
1. Test trước — đủ data ko? Mơ hồ ko? Thiếu → dừng, báo thiếu.
2. Chọn đường tối ưu bền vững — robust, maintain được, ko technical debt.
3. Mơ hồ = thiếu data → nói thẳng "cần X", ko suy diễn, ko bịa.

**Lý thuyết là quan trọng nhất** — đọc paper gốc, hiểu bản chất toán học trước khi code/kết luận. Nếu ko hiểu → hỏi, đừng giả vờ.

**Nguyên tắc so sánh:** Phải xét đầy đủ mọi phương diện (params, T, budget, task, seed). Thiếu 1 yếu tố = kết luận chưa chắc.

**🔥 Không suy luận số từ đồ thị/ảnh:** Figure/PNG ko đọc được text. Nếu cần số → hỏi user mô tả hoặc chỉ dùng bảng số. Nếu ko có bảng → ghi "ước lượng, ko chính xác". Sai lầm: Figure 18 LeWM — tao tự bịa epoch 0 = 0.04, thực tế ~0.2.

## Project
- **V0 [done]:** Bionic hand 8-DOF real, grasp confirmed (data tự xây)
- **V1 [abandoned]:** Hybrid CfC+Attention TwoRoom — 78%
- **V2.1 [active]:** Mamba-2+Attention predictor (TwoRoom)
- **V3 [future]:** Overhead cam, 1 agent, 2 robot
- **V3.1 [future]:** Overhead + 2 ego, 2 agent, 2 robot
- **V3.2 [future]:** 2 ego, 2 agent, 2 robot

## V2.1 Config
- **6×{Self-Attn(AdaLN) → Mamba-2}**, T=4, heads=16, d_state=256, expand=4
- Attention:Mamba-2 ≈ 1.43:1 (787K:550K per block)
- Predictor 9.36M, total 16.6M
- Eval: budget=50, goal_offset=25 (mọi task)
- Seed: 3072 (train + eval)
- Epochs: 10, batch=128, lr=5e-5, bf16
- Mamba-2 (wheel sẵn), Mamba-3 giữ tham khảo
