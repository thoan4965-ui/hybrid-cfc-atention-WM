# LeWM Project — AGENT CONTEXT (CURRENT: V2.1 Mamba-2+Attention)

## ⚡ MASTER PROTOCOL — 3 pillars of truth + 3-step decision

### 3 pillars of truth (cho mọi kết luận)
Mọi kết luận phải đủ: **Lý thuyết (Theory) + Paper/Data + Thực nghiệm (Empirical)**. Thiếu 1 → ghi "chưa biết, cần X".

### 3-step decision (cho mọi hành động)
1. **Test trước** — kiểm tra data có đủ ko? Có mơ hồ ko? Nếu thiếu data → dừng, báo "thiếu data", ko đoán.
2. **Chọn đường tối ưu bền vững** — ko chỉ "chạy được là được". Chọn giải pháp robust, maintain được lâu dài, ko technical debt.
3. **Mơ hồ = thiếu data** — ko thể kết luận chắc chắn = thiếu thông tin. Nói thẳng "chưa đủ data để kết luận, cần X". Ko suy diễn, ko bịa.

### ⚠️ Lý thuyết là quan trọng nhất
**Đọc kỹ paper gốc, hiểu bản chất toán học trước khi code hay kết luận.**
- Sai lầm lớn nhất đến từ thiếu hiểu biết lý thuyết: CfC ≠ RNN (CfC là closed-form), Mamba O(T) ≠ Attention O(T²), SIGReg ≠ MSE
- Nếu không đọc được số liệu / bảng biểu / công thức → **hỏi user**, đừng im lặng giả vờ hiểu
- Mỗi lần gặp architecture mới: đọc paper → tóm tắt toán học cốt lõi → verify với source code → mới đưa ra quyết định
- Lịch sử lỗi: "CfC O(T·d²)" (tự bịa), "Mamba > CfC về speed" (ko có paper so sánh), "T=16 > T=4" (ko đọc LeWM paper kỹ) — tất cả đều do đọc lướt lý thuyết

### Áp dụng
- Trước mỗi lệnh cài đặt: check CUDA+Torch, check wheel có sẵn ko, dataset có public ko
- Trước mỗi train: verify config, data path, checkpoint resume
- Trước mỗi kết luận: đủ 3 pillars chưa? Nếu chưa → báo thiếu

## Project Scale
- **V0 [done]:** 1 bionic hand 8-DOF real, grasp confirmed
- **V1 [abandoned]:** Hybrid CfC+Attention TwoRoom — 78% (T=16 sai lầm, đáng lẽ T=4)
- **V2.1 [active]:** Mamba-2+Attention predictor, TwoRoom (đang train trên Vast)
- **Social T1 [future]:** 1 overhead cam, joint latent, 2×SO-ARM100 (MuJoCo)
- **Social T2 [future]:** 3 cam (overhead + 2 ego), cross-attn

## V2.1 Architecture
- **6×{Self-Attn(AdaLN) → Mamba-2}**, T=4, heads=6, d_state=64
- Attention:Mamba-2 ≈ 1.3:1 params (288K:221K)
- Mamba-2 (có wheel sẵn, ko build), Mamba-3 giữ trong module.py để tham khảo

## Strategy: Fair architecture comparison (fixed T=4)
- LeWM paper dùng **T=4** cho cả 4 benchmark
- Mọi architecture dùng **cùng T=4, batch=128, seed=3072**
- **Eval: mọi task budget=50, goal_offset=25** (LeWM paper config)
- Ai beat LeWM ở T=4 → architectural improvement thật

## Priorities
1. ✅ V2.1 Mamba-2 code + config + all-in-one script
2. 🔄 V2.1 train TwoRoom trên Vast (RTX 5080, $0.175/h)
3. 📅 V2.1 eval TwoRoom + Push-T + Cube + Reacher
4. 📅 Báo cáo Sáng tạo trẻ 30/6 (V0 + V2.1)
5. 📅 ISEF tháng 9 (cần budget ~$72-120)

## Rules từ bug (nghiêm ngặt — ko vi phạm)
1. **Resume:** `glob *.ckpt`, ko hardcode filename. Lightning đặt tên `.ckpt` khác `output_model_name`.
2. **HF upload:** `{subdir}/{run_name}/ep_{epoch}` — ko hardcode, ko env var.
3. **Check CUDA+Torch trước cài wheel:** `python -c "import torch; print(f'{torch.__version__}, CUDA: {torch.version.cuda}')"`. Sai version → 404 wheel → mất $.
4. **Ko build source nếu có wheel:** causal-conv1d, mamba-ssm đều có wheel 274-533 MB. Check release page trước, build chỉ khi ko có wheel.
5. **Dataset .tar.zst → download + giải nén thủ công** (download_data.py). `stable-worldmodel` chưa hỗ trợ auto-detect `.h5` nếu thiếu `hdf5plugin`.
6. **Eval seed:** seed=3072 đồng bộ train. Mọi task, mọi architecture.
7. **Precision:** bf16 trên GPU hỗ trợ. Ko dùng fp16 trừ khi GPU ko hỗ trợ bf16.
8. **CEM context history bug:** phải truyền `context_emb`, ko dùng 1 frame. CfC cần 3-frame history.
9. **Ko tự bịa threshold/cơ chế** — phải có test thật. "cost < 0.05 là tốt" là bịa.
10. **🔥 Check thư mục trước sửa/push — KHÔNG SỬA BỪA.** `ls`/`Get-ChildItem` xem cấu trúc file. Hiểu quan hệ: config nào → file nào → import nào → phụ thuộc module nào. Nếu ko rõ → hỏi user trước.
11. **🔥 Kiểm tra format support trước dùng function.** `swm.data.load_dataset` chỉ hỗ trợ `lance, folder, lerobot, video`. HDF5 cần `hdf5plugin` + `HDF5Dataset` trực tiếp. Ko tin docs mù quáng — check source code.
12. **🔥 Đủ 3 pillars (Theory + Paper + Empirical) cho mọi kết luận.** Thiếu 1 → ghi "chưa biết". Ko suy diễn, ko bịa.
13. **🔥 Eval config đúng LeWM paper: mọi task budget=50, goal_offset=25.** TwoRoom từng để sai 150/100 vì ko đọc paper kỹ. Budget 50 × frameskip 5 = 250 env steps — đủ cho TwoRoom. Chỉ tăng nếu benchmark chứng minh thiếu.
14. **🔥 Wrapper try/except cho mọi I/O (network, file, HF, disk).** `api.upload_file` crash training epoch 0 vì token hết hạn. Training ko được dừng vì lỗi phụ. In warning, log error, continue.
15. **🔥 Kiểm tra token/dịch vụ trước khi dùng.** HF token hết hạn → 401. Kiểm tra `api.whoami()` trước session, ko đợi lúc runtime mới biết.

## Budget
- Vast RTX 5080: $0.175/h (đang dùng)
- Colab T4: free (nhiều acc)
- Kaggle GPU: free (nhiều acc)

## Work Protocol
- Output mỗi tool call
- Ko sub-agent code
- Manager → worker
- Mày nói "dừng" → tao stop ngay
