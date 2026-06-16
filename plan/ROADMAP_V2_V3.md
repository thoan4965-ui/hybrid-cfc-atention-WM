# Roadmap V2 → V3 — LeWM Project

## Timeline

```
V2.1    [7-8/2026]  Mamba-2+Attention predictor + LeWM 4 benchmarks  ⚡ CHỐT CỨNG
V2.5    [9/2026]    Vim encoder + Mamba-2 (pure SSM) + ONNX nén CPU
Social T1 [10-11/2026] Overhead cam, 2×SO-ARM100, hybrid predictor
V3.1    [12/2026]*  Ko overhead cam, Vim encoder, T=16-32 (dự phòng)
```

---

## V2.1: Mamba-2+Attention Hybrid (benchmark) — ⚡ CHỐT CỨNG

**Mục tiêu:** Fair comp predictor architecture, giữ encoder ViT-tiny. Novelty: hybrid SSM-Attention predictor cho world model.

**Quyết định kiến trúc:** Mamba-2 thay vì Mamba-3. Lý do:
- Mamba-2 có wheel pre-built cho mọi CUDA (12-13) — `pip install mamba-ssm` 30 giây, ko cần build
- Mamba-3 build từ source trên Vast gặp nhiều lỗi (CUDA toolkit mismatch, undefined symbol)
- Với T=4, Mamba-2 và Mamba-3 cho chất lượng tương đương (khác biệt chỉ xuất hiện ở T>1000)
- d_state=64 đủ cho tất cả task hiện tại và Social T1 (T=4)
- Tiết kiệm thời gian + budget — tradeoff xứng đáng
- **Nếu cần Mamba-3 cho Social V3.1:** rent template CUDA 12.4 riêng, build 1 lần. Code Mamba-3 đã có sẵn trong module.py

### Thành phần
- **Encoder:** ViT-tiny (~5M, patch=14, hidden=192) — giống LeWM paper
- **Predictor:** 6×{Self-Attn(AdaLN) → Mamba-2} hybrid (tỉ lệ 1:1 params)
- **Train:** num_workers=6, persistent_workers=True, prefetch_factor=3
- **T=4**, batch=128, lr=5e-5, loss: MSE + SIGReg (λ=0.09)
- **Precision:** RTX 5080=bf16-mixed (Colab ko support mamba, debug thẳng trên Vast)

### Config chốt
```yaml
predictor:
  _target_: module.Mamba2Predictor
  num_frames: ${history_size}     # history_size=3 → T=4
  input_dim: ${embed_dim}         # 192
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 6
  dim_head: 64
  d_state: 64                     # Mamba-2 — đủ cho T=4 đến Social T=16-32
  dropout: 0.1
  emb_dropout: 0.0
```
Tỉ lệ Attention : Mamba-2 ≈ 288K : 221K ≈ 1.3:1 ✅

### Workflow
```
Rent Vast (CUDA 12.x + PyTorch 2.6-2.12)
→ pip install mamba-ssm (wheel 30s, ko build)
→ pip install stable-pretraining stable-worldmodel
→ train + eval
```

### Vast All-in-One Runbook (V2.1 Mamba-2)
Copy-paste từ đầu đến cuối:

```bash
# === 0. Clone repo ===
git clone https://github.com/thoan4965-ui/hybrid-cfc-atention-WM.git le-wm-v2.1
cd le-wm-v2.1
git checkout v2.1

# === 1. Install dependencies ===
pip install stable-pretraining stable-worldmodel huggingface_hub hydra-core einops

# === 2. Install Mamba-2 wheel (ko build, 30s) ===
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps
pip install causal-conv1d einops

# === 3. Verify import ===
python -c "from mamba_ssm import Mamba2; print('Mamba-2 OK')"

# === 4. Train TwoRoom ===
python train.py model=mamba2_hybrid data=tworoom trainer.max_epochs=100 optimizer.lr=5e-5 loader.batch_size=128 seed=3072

# === 5. Eval TwoRoom (tự động chọn ckpt mới nhất) ===
python eval.py task=tworoom policy=$(python -c "
import os
from pathlib import Path
import stable_worldmodel as swm
d = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
c = sorted(d.rglob('*_weights.ckpt'), key=os.path.getmtime)
print(str(c[-1].parent) if c else str(d))")

# === 6. (Optional) Nếu cần train thêm task khác ===
# python train.py model=mamba2_hybrid data=pusht trainer.max_epochs=100 optimizer.lr=5e-5 loader.batch_size=128 seed=3072
# python train.py model=mamba2_hybrid data=cube trainer.max_epochs=100 optimizer.lr=5e-5 loader.batch_size=128 seed=3072
# python train.py model=mamba2_hybrid data=reacher trainer.max_epochs=100 optimizer.lr=5e-5 loader.batch_size=128 seed=3072
```

**CFG** (thay đổi nếu template khác torch/cuda/python version):
- PyTorch 2.10 + CUDA 12.x → dùng wheel `cu12torch2.10` (dòng trên)
- PyTorch 2.10 + CUDA 13.x → dùng `cu13torch2.10`
- Cần check: `python -c "import torch; print(torch.__version__)"`
- Mặc định Vast PyTorch template = 2.10 + CUDA 12.8 → wheel trên chạy OK

### Benchmark (4 tasks, LeWM paper config)
| Task | Budget | Goal offset | Max steps |
|------|--------|-------------|-----------|
| TwoRoom | **150** | 100 | 300 |
| Push-T | 50 | 25 | 100 |
| Cube | 50 | 25 | 100 |
| Reacher | 50 | 25 | 100 |

### So sánh (cùng T=4, batch=128)
1. **AR (LeWM paper)** — baseline chính (87% TwoRoom)
2. **Mamba-2+Attention (V2.1)** — architecture đề xuất

### CEM planning config
- 300 samples, 10 iterations, 30 elites, điều chỉnh elits hàm ý
- Horizon=5 latent steps (25 env steps với frameskip=5)
- MPC: execute all then replan

### Compute
- Debug + train: RTX 5080 1-2 ngày (~$1-2)
  - Vast template: CUDA 12.x bất kỳ — `pip install mamba-ssm` chạy wheel sẵn
  - Nếu template CUDA 13.0 → dùng `mamba_ssm` wheel cu13torch2.10 hoặc build từ source (10 phút)
- Nếu beat AR 87% TwoRoom → novelty confirmed

### Output
- Bảng kết quả 4 tasks × architectures
- Phân tích: Mamba-2 vs AR ở T=4
- Code: `le-wm-v2.1/` (branch v2.1)
- ISEF submission material

---

## V2.5: Full SSM Pipeline + CPU Deploy

**Mục tiêu:** Kiến trúc 100% SSM (Vim encoder + Mamba-2 predictor), ONNX INT8, đo quality degradation trên CPU. Novelty: first full SSM world model + edge deploy.

### Thành phần
- **Encoder:** Vim-T (Vision Mamba, ~7M, O(n) patches)
- **Predictor:** 6×{Mamba-2} pure SSM (ko Attention)
- **Train:** end-to-end từ pixels, SIGReg + MSE

### Tại sao Vim thay vì MambaVision?
| | Vim (pure SSM) | MambaVision (hybrid) |
|---|---|---|
| SSM pipeline | ✅ 100% — giữ novelty | ❌ Có Attention → mất novelty |
| CPU speed | ✅ Nhanh (scan đơn giản) | ❌ Chậm (attention CPU) |
| Novelty | **Full SSM world model** | Hybrid — giống V2 |

### Nén + Deploy pipeline
```
Train FP32 → ONNX export (pure-PyTorch Mamba fallback)
→ INT8 quantization (Quamba2-style)
→ Benchmark CPU i7-1165G7
```

### CEM bottleneck analysis
CEM planning = 300 samples × 10 iters × T steps → 3000×T forward passes.
- V2 (ViT+T=4): ~1s GPU
- V2.5 (Vim+T=4): INT8 CPU ước ~5-15s (cần test)
- CompACT tokenizer: giảm encoder bottleneck thêm

### Metric targets
| Metric | V2 (L40S GPU) | V2.5 (i7 CPU mục tiêu) |
|--------|---------------|----------------------|
| Model size | ~28MB FP16 | **~7-14MB INT8** |
| CEM planning | ~1s | **~5-15s** (chấp nhận được) |
| Quality TwoRoom | ~87% (beat paper) | Giảm **1-3%** |
| RAM usage | N/A GPU | **~500MB-1GB** |

### Novelty claim
"First full state-space-model world model (no attention). Export ONNX INT8 → CEM planning trên CPU laptop i7 trong ~Xs, quality degradation chỉ ~Y%. Mở đường cho world model trên thiết bị không GPU."

### Compute
- Train: Kaggle TPU v5e-8 (free) hoặc Kaggle GPU (free)
- ONNX benchmark: local CPU

---

## Social T1: Overhead cam + 2×SO-ARM100

**Mục tiêu:** Social learning với 1 camera overhead, 2 robot. Novelty: first social JEPA world model.

### Encoder decision
| Option | Lý do chọn |
|--------|-------------|
| **ViT-tiny** | Có overhead cam → global view → T=4 đủ. Đơn giản, tận dụng code V2 |
| CNN ResNet-18 | Backup nếu ViT ko đủ. LeWM đã test SIGReg với ResNet ✅ |
| Vim/MambaVision | Ko cần — overhead đã cho đủ context |

**Chốt:** Thử ViT-tiny trước. Nếu social task khó → CNN (1 dòng config, an toàn).

### Kiến trúc
- **Encoder:** ViT-tiny (overhead view) → joint latent cho 2 robot
- **Predictor:** Mamba-2+Attention hybrid (giữ Attention cho retrieval multi-agent)
- **CLIP goal** encoding
- **T=4** (overhead cho global view → ko cần T cao)

### Thành phần
- 1 overhead RGB camera
- 2×SO-ARM100 robot trong MuJoCo
- Joint latent: encode state cả 2 robot + interaction
- Cross-Attention nếu cần

### Benchmark
- Task: chuyển đồ, phối hợp, tránh nhau
- So sánh: single agent vs multi-agent

### Notes
- Giữ Attention vì social tasks cần retrieval chính xác (ai làm gì, đồ ở đâu)
- Ko ONNX, ko nén ở phase này
- Nếu thời gian + budget: thử CNN encoder + T=8-16

---

## V3.1: Ko overhead cam, Vim encoder, T cao (dự phòng)

**Mục tiêu:** Long-horizon social tasks. Mỗi robot chỉ có ego view → cần T=16-32. **Chỉ làm nếu còn thời gian.**

### Tại sao cần T cao?
- Ko có overhead → mỗi robot ko thấy robot kia
- Phải suy luận từ temporal context (T cao hỗ trợ)
- Mamba O(T) cho phép T=16-32 mà ko OOM

### Thành phần
- **Encoder:** Vim (O(n) patches, CPU-friendly)
- **Predictor:** Mamba-2+Attention hybrid, T=16-32
- **CEM planning:** ~10-20s (chấp nhận được cho offline social)

### Tradeoff
| | Social T1 (overhead) | V3.1 (ego only) |
|---|---|---|
| Cam | 1 overhead | Ego-only |
| T | 4 | 16-32 |
| CEM time | ~1s | ~10-20s |
| Complexity | Thấp | Cao |
| Risk | Thấp | Cao |

### Risk
- Cần budget GPU lớn hơn
- Quality chưa chắc giữ được ở T cao
- **Dễ ko kịp ISEF → chuyển qua dự phòng**

---

## Novelty Stack (đầy đủ)

### Lớp 1 — Hybrid temporal predictor (V2)
- LeWM dùng AR (Transformer). Chúng tôi thay bằng Hybrid Mamba-3+Attention cho JEPA world model.
- Paper LLM đã làm hybrid (Jamba, NVIDIA Hybrid nhưng chưa có cho robot world model).
- **Claim:** *First hybrid SSM-Attention predictor for JEPA world model in robot manipulation.*

### Lớp 2 — SIGReg + stateful model (V1 → V2)
- SIGReg noise isotropic → weighted sum vẫn noisy với stateless FFN (AR).
- Với CfC (stateful ODE) hoặc Mamba (stateful SSM): noise tích phân.
- **Claim:** *First analysis of SIGReg interaction with stateful world model predictors.*

### Lớp 3 — Full SSM pipeline (V2.5)
- Vim (spatial SSM) + Mamba-2 (temporal SSM) = 100% SSM, ko Attention anywhere.
- **Claim:** *First full state-space-model world model for robot manipulation.*

### Chú thích: Mamba-2 thay vì Mamba-3
- **Lý do:** Mamba-2 có wheel cho CUDA 13.0 (Vast template mặc định), ko cần build từ source.
- **Tác động novelty:** Ko đổi — novelty là "thay MLP (stateless) bằng stateful SSM trong JEPA predictor", ko phải "dùng Mamba-3" cụ thể.
- **Chất lượng T=4:** Mamba-2 ≈ Mamba-3 — khác biệt lý thuyết (complex-valued state, exponential-trapezoidal) chỉ xuất hiện ở T>1000.
- **Social:** d_state=64 đủ cho T=16-32.
- **Lưu ý khi deploy:** Ko cần `headdim` param (Mamba-2 ko có khái niệm này).*

### Lớp 4 — Edge CPU deploy (V2.5)
- ONNX INT8 → CEM planning trên CPU laptop latency cụ thể + quality degradation numbers.
- **Claim:** *First CPU-deployable world model with measured latency/quality tradeoff.*

### Lớp 5 — Social learning (Social T1)
- Overhead cam + 2 robot + joint latent world model.
- **Claim:** *First social JEPA world model for multi-robot coordination.*

### Rủi ro novelty
- Hybrid predictor: LLM đã làm → claim phải hẹp: "cho robot world model"
- Full SSM: Vim có trong vision rồi → novelty là world model dynamics, ko phải classification
- ONNX: kỹ thuật có sẵn → novelty là số cụ thể (latency/quality tradeoff)

---

## Compression & Deploy Research

### Mamba quantization
- **Quamba** (ICLR 2025): W8A8 Mamba 2.8B, 1.72× speedup, 0.9% drop
- **Quamba2** (2025): W4A8, 3× generation, 4× memory, 1.6% drop
- Tested on Orin Nano (edge GPU)

### Mamba + ONNX
- CUDA selective scan kernel KO export được
- Giải pháp: pure-PyTorch scan fallback (IBM granite-4.0-h đã làm)
- PyTorch team đang hỗ trợ ONNX scan op (5/2026)

### Knowledge Distillation
- Teacher: V2 (Mamba-3+Attn best quality)
- Student: V2.5 (Vim + Mamba-2)
- KD in latent space → student học latent dynamics từ teacher
- Giảm 50% params với <2% drop

### Pipeline
```
V2 baseline FP16 ~28MB
  → INT8 (Quamba)      ~14MB     1.5-2× speed, -1% drop
  → KD (15M→7M)        ~7MB      2× speed, -1% drop
  → KD + INT8          ~3.5MB    4× speed, -2% drop
  → SLiM sparse+int4   ~2MB      6× speed, -3% drop
```

---

## Budget

| Phase | GPU | Cost | Ghi chú |
|-------|-----|------|---------|
| V2 debug | Colab T4 free | $0 | Debug rule #60 |
| V2 train + eval | RTX 5080 1-2 ngày | $5-10 | Nếu beat AR → confirmed |
| V2.5 train | Kaggle TPU/GPU free | $0 | Kaggle nhiều acc |
| V2.5 ONNX | Local CPU | $0 | i7 laptop |
| Social T1 | Colab/Kaggle | $0-10 | 2 robot MuJoCo |

---

## CEM Planning Analysis

### Bottleneck
CEM = 300 samples × 10 iters × T steps × predictor forward pass.
- **Encoder:** Chạy 1 lần cho start+goal → ko phải bottleneck
- **Predictor:** Mới là bottleneck — 3000×T forward passes

### T vs Time scaling
| | Attention O(T²) | Mamba O(T) |
|---|---|---|
| T=4 | 16 ops | 4 ops |
| T=16 | 256 ops | 16 ops |
| T=32 | 1024 ops | 32 ops |
| CEM T=32 | ~300K ops | ~96K ops |

### CPU time ước tính (i7-1165G7, INT8)

Số chính xác cần empirical, ước lượng từ FLOPs:

### Component latency ước lượng (V2.5, INT8 trên i7 4C/8T)

| Component | Ops | ~ms |
|-----------|-----|-----|
| Vim-T encoder 224x224 (1 frame) | ~1.5G | **30-50ms** |
| Mamba-2 forward (T=4, dim=256) | ~50M | **3-8ms** |
| Mamba-2 forward (T=16, dim=512) | ~400M | **20-40ms** |

**CEM planning 300 samples × 10 iters × T=4:**
  - encode 2 frames: 2 × 40ms = 80ms
  - 3000 Mamba passes × 5ms = 15s
  - Tổng: ~15s

**Tối ưu:**
  - CompACT tokenizer: skip encoder mỗi step
  - ONNX Runtime C++ (thay Python): 2-5× nhanh hơn
  - Giảm CEM samples (300→100): 3× nhanh hơn

---

## Critical Rules (từ bug V1)
1. **Resume:** glob *.ckpt, ko hardcode filename
2. **HF upload:** {subdir}/{run_name}/ep_{epoch}
3. **Hidden state:** Mamba-2 tự quản lý (ko cần `_carry_mode`)
4. **Eval seed:** Tất cả seed=3072 (đồng bộ với train seed)
5. **Precision:** bf16-mixed trên RTX 5080
6. **T=4:** Giữ nguyên cho fair comp. Social T1 mới tăng T
7. **Tỉ lệ 1:1:** Mamba-2 ≈ Attention — heads=6, d_state=64 → ~1.3:1 ✅
8. **Debug rule #60:** Debug trực tiếp trên Vast (Colab ko hỗ trợ mamba-ssm wheel cho cu128)
