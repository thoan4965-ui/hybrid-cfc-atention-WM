# Phụ lục A: Nhật ký phát triển dự án

*Ghi chép từ ngày 7/6 đến 15/6/2026. Viết theo dạng nhật ký cá nhân, ghi lại quyết định, thất bại, và bài học.*

---

## Tuần 1 (7/6 - 10/6): Chạy baseline

**7/6** — Hôm nay chạy thử AR baseline trên TwoRoom. LeWM paper có code, clone về, cài dependencies mất nửa ngày. Chạy được, loss giảm. Ok. Cũng bắt đầu đọc paper CfC của Hasani et al., thấy ODE-RNN temporal memory hấp dẫn, muốn thử thay AR bằng CfC xem sao.

**8/6** — Implement CfC predictor xong. Chạy thử, loss ko giảm. Debug cả buổi — hóa ra action index bị sai: `prop_indices` vs `action_indices`. Fix xong CfC chạy được, nhưng rollout eval thua AR 61×. Fuck. CfC hidden state có vấn đề. Research cả ngày — phát hiện CfC cần scheduled sampling để ổn định rollout. Cũng đọc kỹ LeWM paper, thấy benchmark numbers: TwoRoom 87%, Push-T 96%, Cube 88%, Reacher 49%.

**9/6** — Scheduled sampling + teacher loss fix = CfC rollout ngang AR. VICTORY. Nhưng vẫn ko beat được AR. Lao vào đọc lý thuyết CfC ODE — hiểu tại sao CfC yếu hơn ở T ngắn: error accumulation cần T lớn mới thấy ưu điểm.

Cũng chạy robot thật hôm nay. V0 bionic hand 8-DOF grasp thử chai nước — thành công ngay lần đầu. Pipeline: camera → CEM → servo execute → position error detect. Cảm giác sướng hơn chạy sim nhiều.

**10/6** — 11 bugs trong 1 ngày. Sửa mệt nghỉ. Nhưng grasp confirmed, encoder gap đã check, data pipeline ổn. V0 done.

## Tuần 2 (11/6 - 14/6): Kiến trúc Hybrid

**12/6** — Quyết định: làm Hybrid CfC+Attention, 6 blocks, heads=8, T=16. Thay vì chọn CfC hay AR, sao ko kết hợp cả 2? CfC temporal + Attention spatial. Config: 15.5M params, TwoRoom trước.

**13/6** — Implement HybridCfCPredictor xong. Log nhiều bug: checkpoint upload lỗi path, resume ko hoạt động, seed handling sai. Fix dần. Config align với LeWM paper: seed 3072, bf16, num_workers. 

Bắt đầu train trên Vast L40S. $1/h, đắt nhưng nhanh. Dashboard real-time, thấy loss giảm từ từ. 3h sau: TwoRoom **78%**. Hmm, thấp hơn AR 87%.

**14/6** — Phân tích: CfC hidden state bị nhiễu từ SIGReg regularization. SIGReg λ=0.09 quá cao cho task đơn giản, CfC khuếch đại nhiễu vì ODE hidden state carry. 

Giải pháp 1: Thêm Denoiser MLP giữa SIGReg và CfC. 
Giải pháp 2: Sweep λ (0.09, 0.05, 0.01) — có thể λ quá cao.
Giải pháp 3: BF16 thay FP16 — SIGReg dùng cos/sin, FP16 5-bit exponent dễ overflow.

Đặt tên lại cho rõ: V1.1 = Denoiser + λ sweep (chạy Colab T4 free). V2 = Mamba predictor (tương lai).

## Tuần 3 (15/6): Bug fix marathon

**15/6** — REVIEW TOÀN BỘ CODE. Phát hiện 3 critical bugs:

1. **Resume ko hoạt động** — code tìm file `.ckpt` không tồn tại. Lightning đặt tên file khác. Hậu quả: mọi lần restart là train lại từ đầu. Mất data. Fix: ModelCheckpoint + glob tìm file mới nhất.

2. **Hidden state carry sai** — CfC carry hidden state qua các validation batch khác nhau (= episode khác nhau). Validation loss sai vì CfC warm-start từ episode trước. Fix: thêm `_carry_mode` flag.

3. **HF upload path sai** — hardcode `hybrid_v2` + env var default 0.09. Upload λ=0.01 lên thư mục λ=0.09. Fix: dùng subdir + run_name từ config.

Sửa xong push. Cũng phát hiện: cfc_units là dead code (config truyền vào CfC nhưng CfC ko dùng), denoiser dim hardcode 192. Fix hết.

**Lesson:** Ko tìm optimal T cho từng architecture. Chốt T=4 cho tất cả, fair comparison mới ra novelty thật. LeWM paper dùng T=4, mình cũng T=4. CfC vs AR vs Mamba — cùng config, ai hơn là improvement thật.

## Tổng kết

- **V0**: Robot thật grasp OK ✅
- **V1**: Hybrid CfC+Attention TwoRoom 78% ✅
- **V1.1**: Denoiser + λ sweep (Colab T4 free) 🔄
- **V2**: Mamba predictor 📅
- **Social**: Multi-robot simulation 📅

*Toàn bộ logbook gốc (2700+ dòng) tại GitHub repository.*
