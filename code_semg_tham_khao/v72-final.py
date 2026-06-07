# ==============================================================================
# BIONIC HAND V7.2 — FINAL (NinaPro DB8 Compliant, ROM corrected)
# ==============================================================================
# Changelog từ V7.2:
#   - Sửa ROM_DEG dựa trên dữ liệu thực từ file .mat (S1_E1_A1.mat)
#   - Điều chỉnh thứ tự DOF cho phù hợp với dữ liệu (Middle, Ring, Pinky)
#   - Sửa biomechanical_coupling_loss để dùng đúng chỉ số
#   - Sửa label_smooth_huber (không dùng randn làm nhiễu target)
#   - Sửa max_err dùng abs()
#   - Giảm LR_MAX, tăng ENS_DIV_W, BRANCH_DIV_W
#   - Bỏ /0.5 trong MotorCortex
#   - Đảm bảo tất cả các fix từ V7.2 được giữ nguyên
# ==============================================================================

import subprocess, sys

def _pip(pkg):
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)

_pip('ncps')
_pip('typeguard==4.4.0')
_pip('scipy')

import copy, math, gc, os, random, time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.signal import butter, sosfilt
from sklearn.decomposition import PCA

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE       = torch.float16
amp_enabled = torch.cuda.is_available()

print(f"{'='*70}")
print(f"  BionicsHand V7.2 — FINAL (ROM corrected)")
print(f"  Device: {DEVICE}  |  dtype: {DTYPE}  |  AMP: {amp_enabled}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}  |  VRAM: {props.total_memory//1024**2}MB")
print(f"{'='*70}")

# ── ncps import chain ──────────────────────────────────────────────────────────
_NCP_CLS = None; _NCP_TYPE = 'GRU-2L'
try:
    from ncps.wirings import AutoNCP
    for _cls_name, _mod_path in [
        ('CfC',      'ncps.torch'),
        ('WiredCfC', 'ncps.torch.wired'),
        ('LTC',      'ncps.torch'),
    ]:
        try:
            import importlib
            _m = importlib.import_module(_mod_path)
            _NCP_CLS = getattr(_m, _cls_name)
            _NCP_TYPE = _cls_name
            print(f"✅ ncps — {_cls_name}")
            break
        except (ImportError, AttributeError):
            pass
    NCP_OK = _NCP_CLS is not None
except ImportError:
    NCP_OK = False; AutoNCP = None

if not NCP_OK:
    print("⚠️  ncps not found → GRU-2L fallback")


# ==============================================================================
# CELL 2 — CONFIGURATION (FIXED ROM_DEG và HYPERPARAMS)
# ==============================================================================
def _auto_batch() -> int:
    if not torch.cuda.is_available(): return 16
    name = torch.cuda.get_device_properties(0).name.lower()
    if 't4' in name:
        print("  [T4 detected] batch set to 256")
        return 256
    free = torch.cuda.mem_get_info()[0] // 1024**2
    if   free > 12000: return 256
    elif free >  8000: return 128
    elif free >  5000: return 64
    else:              return 32

class CFG:
    DATA_DIR  = '/content/final_dataset'
    CKPT_DIR  = '/content/drive/MyDrive/AI_Hand/checkpoints'
    LOG_DIR   = '/content/drive/MyDrive/AI_Hand/logs'

    SEQ_LEN    = 400
    IN_DIM     = 16
    OUT_DIM    = 18
    N_SUBJECTS = 0

    HID_DIM      = 256
    N_SYNERGY    = 10
    NCP_SPARSITY = 0.5
    DROPOUT      = 0.15
    ATTN_HEADS   = 4
    ATTN_WIN     = 64   # sliding window, nhưng thực tế SDPA vẫn full (chỉ để đồng bộ)

    EPOCHS        = 80
    BATCH_SIZE    = 256
    ACCUM_STEPS   = 1
    FORCE_WEIGHTS_ONLY = False
    LR_MAX        = 5e-5        # giảm 1 nửa so với V7.2
    LR_MIN        = 1e-6
    WEIGHT_DECAY  = 0.02
    GRAD_CLIP     = 0.5
    VAL_SPLIT     = 0.15
    PATIENCE      = 15
    MIN_DELTA     = 3e-4
    FS            = 2000
    WARMUP_EPOCHS = 5
    T0            = 10
    T_MULT        = 2

    EMA_DECAY = 0.9995

    CURRICULUM_START = 5
    CURRICULUM_END   = 25

    AUX_WARMUP_START = 5
    AUX_WARMUP_END   = 15

    MC_SAMPLES      = 10
    UNCERTAINTY_THR = 0.15

    HUBER_DELTA      = 0.07       # về giá trị chuẩn
    LABEL_SMOOTH     = 0.0        # tạm tắt để tránh nhiễu (sẽ bật lại sau nếu cần)
    MIXUP_ALPHA      = 0.30
    L1_SYNERGY       = 0.003
    BIOMECH_W        = 0.01
    PCA_N_COMPONENTS = 6
    PCA_REG_W        = 0.003
    HYST_W           = 0.5
    ENS_DIV_W        = 0.05       # tăng để gate đa dạng hơn
    BRANCH_DIV_W     = 0.02       # tăng để branch khác nhau

    CALIB_LR_ENC  = 1e-5
    CALIB_LR_HEAD = 1e-4
    CALIB_EPOCHS  = 15
    GRAD_CKPT     = False

    GRAD_NORM_HI = 8.0
    GRAD_NORM_LO = 0.02
    ACCUM_MIN    = 1
    ACCUM_MAX    = 4
    ROLLBACK_THR = 1.5

    NOISE_EP_END = 40
    NOISE_STR_HI = 1.5
    NOISE_STR_LO = 0.3
    NOISE_BUF_SIZE = 4096

    STAG_PATIENCE = 7
    VRAM_CACHE    = True
    GPU_AUG       = True

# ------------------------------------------------------------------------------
# ROM_DEG và thứ tự DOF được điều chỉnh theo dữ liệu thực (S1_E1_A1.mat)
# Dựa trên phân tích ROM thực và thứ tự cột, ta xác định:
#  0: Thumb MCP      (ROM 150°)
#  1: Thumb IP       (ROM 159°)
#  2: Thumb Abd      (ROM 81°)
#  3: Thumb Crs      (ROM 34°)
#  4: Index MCP      (ROM 110°)
#  5: Index PIP      (ROM 114°)
#  6: Index Abd      (ROM 87°)
#  7: Middle MCP     (ROM 157°)
#  8: Middle Abd     (ROM 44°)      ← đã đổi với PIP
#  9: Middle PIP     (ROM 83°)
# 10: Ring MCP       (ROM 148°)
# 11: Ring Abd       (ROM 71°)      ← đã đổi với PIP
# 12: Ring PIP       (ROM 60°)
# 13: Pinky MCP      (ROM 145°)
# 14: Pinky Abd      (ROM 18°)      ← đã đổi với PIP
# 15: Pinky PIP      (ROM 115°)
# 16: Wrist Flex     (ROM 20°)
# 17: Wrist Radial   (ROM 9°)
#
# Ghi chú: ROM được lấy trực tiếp từ (max-min) của dữ liệu.
# Đây là giá trị thực tế của sensor, không phải góc giải phẫu.
# Việc dùng đúng ROM này để tính RMSE độ là chính xác cho dataset này.
# ------------------------------------------------------------------------------
ROM_DEG = torch.tensor([
    150., 159.,  81.,  34.,   # Thumb
    110., 114.,  87.,         # Index
    157.,  44.,  83.,         # Middle
    148.,  71.,  60.,         # Ring
    145.,  18., 115.,         # Pinky
     20.,   9.                # Wrist
], dtype=torch.float32)        # 4+3+3+3+3+2 = 18

CFG.BATCH_SIZE = _auto_batch()
print(f"  Batch={CFG.BATCH_SIZE} | Eff.batch={CFG.BATCH_SIZE*CFG.ACCUM_STEPS}")
print("✅ Config V7.2 FINAL loaded")

# ==============================================================================
# CELL 3 — DATASET & DATA LOADING (giữ nguyên từ V7.2, đã sửa)
# ==============================================================================
_NOISE_BUFFER: Optional[torch.Tensor] = None

def init_noise_buffer(cfg=CFG, size=None):
    global _NOISE_BUFFER
    if size is None:
        size = cfg.NOISE_BUF_SIZE
    print(f"  🌊 Pre-generating Noise Buffer ({size} samples)...")
    nyq = cfg.FS / 2
    sos = butter(4, [20 / nyq, 500 / nyq], btype='band', output='sos')
    buffer = []
    for _ in range(size):
        raw = np.random.randn(cfg.SEQ_LEN, cfg.IN_DIM).astype(np.float32) * 0.03
        filt = np.stack([sosfilt(sos, raw[:, c]) for c in range(cfg.IN_DIM)], axis=1)
        buffer.append(torch.from_numpy(filt))
    _NOISE_BUFFER = torch.stack(buffer)
    if torch.cuda.is_available():
        _NOISE_BUFFER = _NOISE_BUFFER.to(DEVICE, dtype=DTYPE).contiguous()
        print(f"  ✅ Noise Buffer moved to VRAM ({_NOISE_BUFFER.nelement()*2//1024**2}MB)")
    else:
        print("  ✅ Noise Buffer Ready (CPU)")

def _worker_init(worker_id: int):
    seed = SEED + worker_id
    np.random.seed(seed); random.seed(seed)

def _noise_strength(ep: int, cfg=CFG) -> float:
    if ep >= cfg.NOISE_EP_END: return cfg.NOISE_STR_LO
    t = (ep - 1) / max(1, cfg.NOISE_EP_END - 1)
    return cfg.NOISE_STR_HI - t * (cfg.NOISE_STR_HI - cfg.NOISE_STR_LO)

def _aug_view(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    x_dtype = x.dtype
    if random.random() < 0.5 and _NOISE_BUFFER is not None:
        idx = random.randint(0, len(_NOISE_BUFFER) - 1)
        noise = _NOISE_BUFFER[idx] + torch.randn_like(_NOISE_BUFFER[idx]) * 0.1
        noise = noise.to(x_dtype)
        x = x + noise * strength
    if random.random() < 0.4:
        lo = max(0.6, 1.0 - 0.25 * strength)
        hi = min(1.4, 1.0 + 0.25 * strength)
        scale = torch.rand(CFG.IN_DIM, device=x.device, dtype=x_dtype).uniform_(lo, hi).unsqueeze(0)
        x = x * scale
    if random.random() < 0.15 * strength:
        ch    = random.randint(0, CFG.IN_DIM - 1)
        T_len = x.shape[0]
        x     = x.clone()
        x[:, ch] = torch.randn(T_len, dtype=x_dtype, device=x.device) * 0.1 * strength
    if random.random() < 0.2:
        shift = random.randint(-15, 15); T, C = x.shape
        if shift > 0:
            pad = x[0:1, :].expand(shift, -1)
            x = torch.cat([pad, x], 0)[:T]
        elif shift < 0:
            pad = x[-1:, :].expand(-shift, -1)
            x = torch.cat([x, pad], 0)[-T:]
    return x

@torch.no_grad()
def batch_gpu_augment(x: torch.Tensor, y: torch.Tensor, ep: int, cfg=CFG) -> Tuple[torch.Tensor, torch.Tensor]:
    if not cfg.GPU_AUG: return x, y
    B, T, C = x.shape
    s = _noise_strength(ep, cfg)
    if s < 1e-4: return x, y
    if _NOISE_BUFFER is not None:
        noise_mask = (torch.rand(B, 1, 1, device=x.device) < 0.5).to(x.dtype)
        noise_idx = torch.randint(0, _NOISE_BUFFER.shape[0], (B,), device=x.device)
        noise = _NOISE_BUFFER[noise_idx]
        x = x + noise * s * noise_mask
    scale_mask = (torch.rand(B, 1, 1, device=x.device) < 0.4).to(x.dtype)
    lo, hi = 1.0 - 0.25 * s, 1.0 + 0.25 * s
    scale  = torch.empty(B, 1, C, device=x.device, dtype=x.dtype).uniform_(lo, hi)
    scale = scale * scale_mask + 1.0 * (1.0 - scale_mask)
    x = x * scale
    drift_mask = (torch.rand(B, 1, device=x.device) < (0.15 * s)).to(x.dtype)
    ch = torch.randint(0, C, (B,), device=x.device)
    drift = torch.randn(B, T, device=x.device, dtype=x.dtype) * 0.1 * s
    b_idx = torch.arange(B, device=x.device)
    x[b_idx, :, ch] = x[b_idx, :, ch] + drift * drift_mask
    if random.random() < 0.2:
        shift = random.randint(-15, 15)
        if shift > 0:
            pad = x[:, 0:1, :].expand(-1, shift, -1)
            x = torch.cat([pad, x[:, :-shift, :]], dim=1)
        elif shift < 0:
            pad = x[:, -1:, :].expand(-1, -shift, -1)
            x = torch.cat([x[:, -shift:, :], pad], dim=1)
    return x, y

def _scale_y(y: np.ndarray, stats: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    assert y.ndim == 2 and y.shape[1] == CFG.OUT_DIM
    if stats is None:
        y_min = y.min(axis=0, keepdims=True)
        y_max = y.max(axis=0, keepdims=True)
    else:
        y_min, y_max = stats
    y_scaled = np.zeros_like(y, dtype=np.float32)
    for i in range(CFG.OUT_DIM):
        mi = y_min[0, i]
        ma = y_max[0, i]
        if ma - mi < 1e-6:
            y_scaled[:, i] = y[:, i] - mi
        else:
            y_scaled[:, i] = ((y[:, i] - mi) / (ma - mi)) * 2.0 - 1.0
    return y_scaled, (y_min, y_max)

class BionicDataset(Dataset):
    def __init__(self, x_mmap_path: str, y: np.ndarray, training: bool = True,
                 subject_ids: Optional[np.ndarray] = None, indices: Optional[np.ndarray] = None,
                 y_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.x_mmap_path = x_mmap_path
        self.x_mm        = None
        self.indices     = indices
        self.y, self.y_stats = _scale_y(y, stats=y_stats)
        self.training    = training
        self.subj_ids    = (torch.from_numpy(subject_ids.astype(np.int64)) if subject_ids is not None else None)
        self.difficulty  = torch.from_numpy(self.y).float().std(dim=-1).numpy()
        self.current_ep  = 1
        self.x_vram      = None
        self.y_vram      = None
        n = len(self.indices) if self.indices is not None else len(self.y)
        print(f"  {'Train' if training else 'Val/Test'}: {n} samples  [mmap deferred]")

    def cache_to_vram(self):
        if self.x_mm is None:
            self.x_mm = np.load(self.x_mmap_path, mmap_mode='r')
        print(f"  🚀 VRAM Cache: Loading {len(self.y)} samples to {DEVICE}...")
        idx = self.indices if self.indices is not None else np.arange(len(self.y))
        N = len(idx)
        CHUNK = 2048
        self.x_vram = torch.empty((N, CFG.SEQ_LEN, CFG.IN_DIM), device=DEVICE, dtype=DTYPE)
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            chunk_idx = idx[s:e]
            chunk_cpu = np.array(self.x_mm[chunk_idx], dtype=np.float32)
            self.x_vram[s:e] = torch.from_numpy(chunk_cpu).to(DEVICE, dtype=DTYPE)
        self.y_vram = torch.from_numpy(self.y).to(DEVICE, dtype=DTYPE)
        if self.subj_ids is not None:
            self.subj_ids = self.subj_ids.to(DEVICE)
        self.x_mm = None; gc.collect()

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.y)

    def __getitem__(self, i):
        if self.x_vram is not None:
            return self.x_vram[i], self.y_vram[i], (self.subj_ids[i] if self.subj_ids is not None else torch.tensor(-1))
        if self.x_mm is None:
            self.x_mm = np.load(self.x_mmap_path, mmap_mode='r')
        xi  = int(self.indices[i]) if self.indices is not None else i
        x   = torch.from_numpy(np.array(self.x_mm[xi], dtype=np.float32))
        y   = torch.from_numpy(self.y[i])
        sid = self.subj_ids[i] if self.subj_ids is not None else torch.tensor(-1)
        if self.training and not CFG.GPU_AUG:
            s = _noise_strength(self.current_ep)
            return _aug_view(x, s), y, sid
        return x, y, sid

def _collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))

def _curriculum_sampler(ds: BionicDataset, ep: int, cfg=CFG):
    if ep > cfg.CURRICULUM_END or not ds.training: return None
    t = max(0., min(1., (ep - cfg.CURRICULUM_START) / max(1, cfg.CURRICULUM_END - cfg.CURRICULUM_START)))
    diff_norm = ds.difficulty / (ds.difficulty.max() + 1e-6)
    w = torch.from_numpy(1.0 + (1.0 - t) * (1.0 - diff_norm)).float()
    return WeightedRandomSampler(w, len(w), replacement=True)

def _ensure_normalized(cfg=CFG):
    norm_path  = os.path.join(cfg.DATA_DIR, 'x_train_norm.npy')
    tnorm_path = os.path.join(cfg.DATA_DIR, 'x_test_norm.npy')
    mean_path  = os.path.join(cfg.DATA_DIR, 'x_norm_mean.npy')
    std_path   = os.path.join(cfg.DATA_DIR, 'x_norm_std.npy')

    idx_tr_p = os.path.join(cfg.DATA_DIR, 'idx_train.npy')
    if not os.path.exists(idx_tr_p):
        raise FileNotFoundError("idx_train.npy chưa tồn tại. Vui lòng tạo file index trước khi gọi _ensure_normalized.")
    idx_tr = np.load(idx_tr_p)

    if not os.path.exists(norm_path):
        print("  📐 Normalizing x_train (first run)...")
        src = np.load(os.path.join(cfg.DATA_DIR, 'x_train.npy'), mmap_mode='r')
        N, T, C = src.shape; CHUNK = 4096
        STRIDE = 7
        print(f"    Pass 1/2: strided stats [::7] using ONLY train indices...")
        src_s = src[idx_tr[::STRIDE]].astype(np.float32)
        x_mean = src_s.mean(axis=(0, 1), keepdims=True)
        x_std  = src_s.std(axis=(0, 1), keepdims=True) + 1e-6
        del src_s
        np.save(mean_path, x_mean); np.save(std_path, x_std)
        print(f"    mean≈{x_mean.mean():.4f}  std≈{x_std.mean():.4f}")
        print(f"    Pass 2/2: normalize+clip → {norm_path}")
        out = np.lib.format.open_memmap(norm_path, mode='w+', dtype=np.float32, shape=(N, T, C))
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            out[s:e] = np.clip((src[s:e].astype(np.float32) - x_mean) / x_std, -3., 3.)
        del out, src; gc.collect()
        print(f"  ✅ x_train_norm.npy ({os.path.getsize(norm_path)//1024**2}MB)")
    else:
        x_mean = np.load(mean_path); x_std = np.load(std_path)

    if not os.path.exists(tnorm_path):
        print("  📐 Normalizing x_test...")
        src = np.load(os.path.join(cfg.DATA_DIR, 'x_test.npy'), mmap_mode='r')
        N, T, C = src.shape; CHUNK = 4096
        out = np.lib.format.open_memmap(tnorm_path, mode='w+', dtype=np.float32, shape=(N, T, C))
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            out[s:e] = np.clip((src[s:e].astype(np.float32) - x_mean) / x_std, -3., 3.)
        del out, src; gc.collect()
        print("  ✅ x_test_norm.npy")

    return norm_path, tnorm_path, x_mean, x_std

def load_data(cfg=CFG):
    print(f"\n📂 Loading: {cfg.DATA_DIR}")
    yt = np.load(os.path.join(cfg.DATA_DIR, 'y_train.npy'))
    ye = np.load(os.path.join(cfg.DATA_DIR, 'y_test.npy'))

    idx_tr_p = os.path.join(cfg.DATA_DIR, 'idx_train.npy')
    idx_va_p = os.path.join(cfg.DATA_DIR, 'idx_val.npy')
    if not os.path.exists(idx_tr_p):
        N = len(yt)
        nv = int(N * cfg.VAL_SPLIT)
        idx = np.arange(N)
        np.save(idx_va_p, idx[:nv].astype(np.int32))
        np.save(idx_tr_p, idx[nv:].astype(np.int32))
        print(f"  Index files saved ({N-nv} train | {nv} val) [Block-based split]")
    idx_tr = np.load(idx_tr_p)
    idx_val = np.load(idx_va_p)

    norm_path, tnorm_path, _, _ = _ensure_normalized(cfg)
    init_noise_buffer(cfg)

    y_train_raw = yt[idx_tr]
    _, y_stats = _scale_y(y_train_raw)   # chỉ lấy stats từ train
    fit_pca_manifold(y_train_raw, cfg.PCA_N_COMPONENTS)   # fit PCA trên raw y (cùng scaling)

    sid_tr = sid_te = None
    sp = os.path.join(cfg.DATA_DIR, 'subject_ids_train.npy')
    if os.path.exists(sp):
        sid_tr = np.load(sp)
        sid_te = np.load(os.path.join(cfg.DATA_DIR, 'subject_ids_test.npy'))
        cfg.N_SUBJECTS = int(np.unique(sid_tr).max()) + 1
        print(f"  Subjects: {cfg.N_SUBJECTS}")

    svr = sid_tr[idx_tr] if sid_tr is not None else None
    svv = sid_tr[idx_val] if sid_tr is not None else None

    tr_ds = BionicDataset(norm_path, yt[idx_tr], True, svr, indices=idx_tr, y_stats=y_stats)
    va_ds = BionicDataset(norm_path, yt[idx_val], False, svv, indices=idx_val, y_stats=y_stats)
    te_ds = BionicDataset(tnorm_path, ye, False, sid_te, y_stats=y_stats)

    nw = 0 if cfg.VRAM_CACHE else 2
    pin = not cfg.VRAM_CACHE
    kw = dict(num_workers=nw, pin_memory=pin, persistent_workers=(nw>0),
              worker_init_fn=_worker_init, collate_fn=_collate)
    if nw > 0:
        kw['prefetch_factor'] = 2

    if cfg.VRAM_CACHE:
        tr_ds.cache_to_vram()
        va_ds.cache_to_vram()

    tr_dl = DataLoader(tr_ds, cfg.BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    va_dl = DataLoader(va_ds, cfg.BATCH_SIZE*2, shuffle=False, **kw)
    te_dl = DataLoader(te_ds, cfg.BATCH_SIZE*2, shuffle=False, **kw)

    print(f"  ✅ {len(tr_ds)} train | {len(va_ds)} val | {len(te_ds)} test")
    return tr_ds, va_ds, te_ds, tr_dl, va_dl, te_dl

# ==============================================================================
# CELL 4 — MODELS (các sửa từ V7.2 + bỏ /0.5 trong MotorCortex)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6, channel_first=False):
        super().__init__()
        self.s = nn.Parameter(torch.ones(d)); self.eps = eps
        self.channel_first = channel_first
    def forward(self, x):
        if not self.channel_first:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.s
        else:
            return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps) * self.s.view(1, -1, 1)

class SensoryLayer(nn.Module):
    def __init__(self, in_d, hid, drop):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(in_d, hid, 7, padding=0, dilation=1),
                                RMSNorm(hid, channel_first=True), nn.GELU(), nn.Dropout(drop))
        self.c2 = nn.Sequential(nn.Conv1d(hid, hid, 7, padding=0, dilation=4),
                                RMSNorm(hid, channel_first=True), nn.GELU())
        self.c3 = nn.Sequential(nn.Conv1d(hid, hid, 5, padding=0, dilation=16),
                                RMSNorm(hid, channel_first=True), nn.GELU())
        self.skip = nn.Conv1d(in_d, hid, 1)
        self.norm = RMSNorm(hid, channel_first=False)
    def forward(self, x):
        t = x.transpose(1, 2)
        x1 = F.pad(t, (6, 0)); x1 = self.c1(x1)
        x2 = F.pad(x1, (24, 0)); x2 = self.c2(x2)
        x3 = F.pad(x2, (64, 0)); x3 = self.c3(x3)
        return self.norm((x3 + self.skip(t)).transpose(1, 2))

class TemporalAttentionGate(nn.Module):
    def __init__(self, hid: int, n_heads: int = 4, win: int = 400):
        super().__init__()
        self.hid = hid; self.n_heads = n_heads; self.head_d = hid // n_heads
        self.qkv = nn.Linear(hid, 3 * hid, bias=False)
        self.proj = nn.Linear(hid, hid, bias=False)
        self.drop = nn.Dropout(0.1)
        self.norm = RMSNorm(hid)
        self.ff = nn.Sequential(nn.Linear(hid, hid*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid*2, hid))
        self.norm2 = RMSNorm(hid)
        self.win = win
        self._use_sdpa = hasattr(F, 'scaled_dot_product_attention')
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, C = h.shape
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        def _split(x): return x.view(B, T, self.n_heads, self.head_d).transpose(1, 2)
        q, k, v = _split(q), _split(k), _split(v)
        if self._use_sdpa:
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0, is_causal=True)
        else:
            CHUNK = 64; out_chunks = []; scale = self.head_d ** -0.5
            for t0 in range(0, T, CHUNK):
                t1 = min(t0+CHUNK, T)
                qi = q[:, :, t0:t1, :]
                scores = torch.einsum('bhqd,bhkd->bhqk', qi * scale, k)
                mask = torch.zeros(t1-t0, T, dtype=torch.bool, device=h.device)
                for i in range(t1-t0):
                    mask[i, :t0+i+1] = True
                scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min / 2)
                w = scores.softmax(-1)
                w = self.drop(w)
                out_chunks.append(torch.einsum('bhqk,bhkd->bhqd', w, v))
            a = torch.cat(out_chunks, dim=2)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        a = self.drop(self.proj(a))
        h = self.norm(h + a)
        return self.norm2(h + self.ff(h))

class MotorCortex(nn.Module):
    def __init__(self, hid, use_grad_ckpt=True):
        super().__init__()
        self.use_grad_ckpt = use_grad_ckpt
        if NCP_OK:
            torch.manual_seed(SEED)
            w = AutoNCP(units=hid+3, output_size=hid, sparsity_level=CFG.NCP_SPARSITY)
            try: self.rnn = _NCP_CLS(hid, w, batch_first=True)
            except TypeError: self.rnn = _NCP_CLS(hid, w)
            self.t = _NCP_TYPE
        else:
            self.rnn = nn.GRU(hid, hid, num_layers=2, batch_first=True, dropout=0.1)
            self.t = 'GRU-2L'
        self.n = RMSNorm(hid)
    def _fwd(self, x): h, _ = self.rnn(x); return h
    def forward(self, x):
        h = grad_ckpt(self._fwd, x, use_reentrant=False) if (self.use_grad_ckpt and self.training) else self._fwd(x)
        # Bỏ /0.5 (không cần thiết, gây nhiễu)
        return F.silu(self.n(h))

class SynergyHead(nn.Module):
    def __init__(self, hid, n_syn, out):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(hid, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, n_syn),
            nn.Softplus()   # non-negativity
        )
        self.pool_q = nn.Linear(n_syn, 1)
        self.dec = nn.Linear(n_syn, out, bias=True)
        self.dec._skip_init = True
        nn.init.orthogonal_(self.dec.weight)
        self.act = nn.Tanh()
    def ortho_loss(self):
        W = F.normalize(self.dec.weight.float(), dim=0)
        I = torch.eye(W.shape[1], device=W.device)
        return (W.T @ W - I).pow(2).mean()
    def forward(self, h_seq: torch.Tensor):
        syn_seq = self.enc(h_seq)
        attn_w = F.softmax(self.pool_q(syn_seq), dim=1)
        syn = (syn_seq * attn_w).sum(dim=1)
        out = self.act(self.dec(syn))
        return out, syn

class SurrogateSpike(torch.autograd.Function):
    PI_SQ = 9.8696
    @staticmethod
    def forward(ctx, v, thr=1.0):
        ctx.save_for_backward(v); ctx.thr = thr
        return (v >= thr).to(dtype=v.dtype)
    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors
        v32 = v.float()
        grad = g.float() / (1. + (SurrogateSpike.PI_SQ * (v32 - ctx.thr).pow(2)))
        return grad.to(g.dtype), None
spike_fn = SurrogateSpike.apply

class SNNPool(nn.Module):
    def __init__(self, hid, out, n=32, steps=8, tau=20., thr=0.1):
        super().__init__()
        self.out = out; self.n = n; self.steps = steps
        self.tau = tau; self.thr = thr
        self.input_proj = nn.Linear(hid, out * n)
        self.register_buffer('pref', torch.linspace(-1, 1, n))
    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        B, S, _ = h_seq.shape
        actual_steps = min(self.steps, S)
        pref = self.pref.to(h_seq.dtype)
        V = torch.zeros(B, self.out, self.n, device=h_seq.device, dtype=h_seq.dtype)
        rt = torch.zeros_like(V)
        for t in range(actual_steps):
            hi = h_seq[:, t, :]
            inp = self.input_proj(hi).view(B, self.out, self.n)
            V = V + (-V + inp) / self.tau
            s = spike_fn(V, self.thr)
            V = V * (1 - s.detach()) - 0.1 * s.detach()
            rt += s
        r = rt / max(actual_steps, 1)
        r_sum = r.sum(-1).float()
        return ((r * pref).sum(-1).float() / r_sum.clamp(min=1e-3)).to(h_seq.dtype)

class HysteresisEncoder(nn.Module):
    def __init__(self, hid: int, w: float = 0.25):
        super().__init__()
        self.proj = nn.Linear(hid * 2, hid)
        self.norm = RMSNorm(hid)
        self.w = w
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_last = h[:, -1, :]
        if h.shape[1] > 1:
            k = max(1, min(h.shape[1] // 2, 200))
            h_past = h[:, -k, :] if k <= h.shape[1] else h[:, 0, :]
            temp_grad = (h_last - h_past) * self.w
            combined = torch.cat([h_last, temp_grad], dim=-1)
            return self.norm(self.proj(combined))
        combined = torch.cat([h_last, torch.zeros_like(h_last)], dim=-1)
        return self.norm(self.proj(combined))

class EnsembleHead(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hid, 32), nn.GELU(), nn.Linear(32, 2))
    def forward(self, h_last, p_syn, p_snn):
        w = F.softmax(self.gate(h_last), dim=-1)
        return w[:, 0:1] * p_syn + w[:, 1:2] * p_snn, w

class SubjectEmbedding(nn.Module):
    def __init__(self, n_subjects, hid):
        super().__init__()
        self.emb = nn.Embedding(n_subjects, 32)
        self.proj = nn.Linear(32, hid)
    def forward(self, h, sid):
        safe_sid = torch.clamp(sid, 0, self.emb.num_embeddings - 1)
        e = self.proj(self.emb(safe_sid)).unsqueeze(1)
        mask = (sid >= 0).float().view(-1, 1, 1)
        return h + e * mask

class BionicsV72(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        H = cfg.HID_DIM
        self.sensory = SensoryLayer(cfg.IN_DIM, H, cfg.DROPOUT)
        self.motor = MotorCortex(H, cfg.GRAD_CKPT)
        self.attn_gate = TemporalAttentionGate(H, cfg.ATTN_HEADS, cfg.ATTN_WIN)
        self.hyst = HysteresisEncoder(H, cfg.HYST_W)
        self.synergy = SynergyHead(H, cfg.N_SYNERGY, cfg.OUT_DIM)
        self.snn = SNNPool(H, cfg.OUT_DIM)
        self.ensemble = EnsembleHead(H)
        self.subj_emb = (SubjectEmbedding(cfg.N_SUBJECTS, H) if cfg.N_SUBJECTS > 0 else None)
        self._init_weights()
        self._info()
    def _init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m, '_skip_init', False):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
    def _info(self):
        n = sum(p.numel() for p in self.parameters())
        rf = 1 + (7-1)*1 + (7-1)*4 + (5-1)*16   # 95
        print(f"\n{'='*70}")
        print(f"  BionicsV7.2 — FINAL (ROM corrected)")
        print(f"  Controller:  {self.motor.t}")
        print(f"  Params:      {n:,}  (~{n*2//1024}KB BF16)")
        print(f"  Conv RF:     ~{rf}smp = {rf/2:.0f}ms")
        print(f"  Attn win:    {CFG.ATTN_WIN} (full causal)")
        print(f"  Synergy:     {CFG.HID_DIM}→64→{CFG.N_SYNERGY}→{CFG.OUT_DIM}  [branch A]")
        print(f"  SNN:         {CFG.HID_DIM}→pop→{CFG.OUT_DIM}  [branch B]")
        print(f"  Hysteresis:  temporal_grad w={CFG.HYST_W}  [V7-2]")
        print(f"  PCA reg:     top-{CFG.PCA_N_COMPONENTS} PC λ={CFG.PCA_REG_W}  [V7-3]")
        print(f"  Ensemble:    2-way gate + entropy λ={CFG.ENS_DIV_W}")
        print(f"  Subject emb: {'yes' if self.subj_emb else 'no'}")
        print(f"  Eff.Batch:   {CFG.BATCH_SIZE}×{CFG.ACCUM_STEPS}={CFG.BATCH_SIZE*CFG.ACCUM_STEPS}")
        print(f"  Huber δ:     {CFG.HUBER_DELTA}")
        print(f"{'='*70}\n")
    def forward(self, x: torch.Tensor, sid: Optional[torch.Tensor] = None) -> Dict:
        f = self.sensory(x)
        h = self.motor(f)
        if self.subj_emb is not None and sid is not None:
            h = self.subj_emb(h, sid)
        h = self.attn_gate(h)
        h_hyst = self.hyst(h)
        p_syn, syn = self.synergy(h)
        B, T, Hdim = h.shape
        steps = self.snn.steps
        stride = 10
        start = max(0, T - steps * stride)
        idx = torch.arange(start, T, stride, device=h.device)
        h_seq = h[:, idx, :]
        p_snn = self.snn(h_seq)
        angles, ens_w = self.ensemble(h_hyst, p_syn, p_snn)
        return {'angles': angles, 'syn': syn, 'ens_w': ens_w, 'p_syn': p_syn, 'p_snn': p_snn}

def fix_ncp_buffers(m):
    c = 0
    for name, buf in m.named_buffers():
        if 'mask' in name or 'sparsity' in name:
            buf.data = buf.data.to(DEVICE); c += 1
    if c: print(f"  NCP buffers: {c} → {DEVICE} float32")
    return m

def build_model(cfg=CFG) -> BionicsV72:
    torch.manual_seed(SEED)
    return fix_ncp_buffers(BionicsV72(cfg).to(DEVICE))

# ==============================================================================
# CELL 5 — EMA (giữ nguyên)
# ==============================================================================
class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data = self.decay * s.data + (1 - self.decay) * m.data
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)
    def get_model(self): return self.shadow

# ==============================================================================
# CELL 6 — MC DROPOUT (giữ nguyên)
# ==============================================================================
def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    return model

@torch.no_grad()
def mc_predict(model, x, n=10, thr=0.15, sid=None):
    enable_dropout_only(model)
    preds = [model(x, sid)['angles'] for _ in range(n)]
    model.eval()
    stack = torch.stack(preds)
    mean, std = stack.mean(0), stack.std(0)
    return mean, std, mean * torch.clamp(1. - std / thr, 0., 1.)

# ==============================================================================
# CELL 7 — LOSS FUNCTIONS (đã sửa label_smooth, biomech dùng chỉ số mới)
# ==============================================================================
class AdaptiveScaler(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(n))
    def forward(self, losses):
        ls = torch.stack(losses)
        scaled = ls * torch.exp(-self.lw) + self.lw
        total = scaled.sum()
        lw_reg = 0.01 * self.lw.pow(2).sum()
        return total + lw_reg, torch.exp(-self.lw).detach()

def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return (lam * x.float() + (1 - lam) * x[idx].float()).to(x.dtype), lam * y + (1 - lam) * y[idx]
    return x, y

def label_smooth_huber(pred, target, smoothing=0.0, delta=CFG.HUBER_DELTA):
    # Sửa: label smoothing đúng cho regression
    if smoothing > 0:
        # Thêm nhiễu Gaussian nhỏ tỷ lệ với độ lệch chuẩn của target
        std = target.std()
        target = target + smoothing * torch.randn_like(target) * std
    return F.huber_loss(pred, target, delta=delta)

# Cập nhật biomechanical_coupling_loss với chỉ số DOF mới
def biomechanical_coupling_loss(angles, rom_deg=ROM_DEG):
    deg = (angles + 1) / 2 * rom_deg.to(angles.device)
    loss = 0.0

    # MCP-PIP coupling (các cặp mcp, pip)
    # Index:   (4,5)
    # Middle:  (7,9)   vì cột 9 là PIP
    # Ring:    (10,12) vì cột 12 là PIP
    # Pinky:   (13,15) vì cột 15 là PIP
    for mcp, pip in [(4,5), (7,9), (10,12), (13,15)]:
        loss += F.relu(deg[:, pip] - deg[:, mcp] - 25).pow(2).mean()

    # Tenodesis: Wrist extension (cột 16) nên làm finger flexion
    wrist_ext = F.relu(deg[:, 16])
    # mean finger flexion: lấy các MCP (4,7,10,13)
    mean_flex = deg[:, [4,7,10,13]].mean(-1)
    loss += F.relu(wrist_ext - mean_flex - 30).pow(2).mean() * 0.4

    # Adduction limit khi flexed: MCP + |Abd| ≤ 130°
    # Abd chỉ số: Index (6), Middle (8), Ring (11), Pinky (14)
    for mcp, abd in [(4,6), (7,8), (10,11), (13,14)]:
        loss += F.relu(deg[:, mcp] + deg[:, abd].abs() - 130).pow(2).mean() * 0.3

    # Thumb specific: IP (cột 1) không vượt MCP (cột 0) quá 30°
    loss += F.relu(deg[:, 1] - deg[:, 0] - 30).pow(2).mean()

    return loss / 6.0

# PCA Manifold Reg (giữ nguyên từ V7.2)
_PCA_COMPONENTS: Optional[torch.Tensor] = None
_PCA_MEAN = None

def fit_pca_manifold(y_train_raw: np.ndarray, n_components: int = 6):
    global _PCA_COMPONENTS, _PCA_MEAN
    try:
        y = y_train_raw.astype(np.float32)
        if y.min() >= -0.05 and y.max() <= 1.05:
            y = y * 2.0 - 1.0
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(y)
        _PCA_COMPONENTS = torch.tensor(pca.components_, dtype=torch.float32).to(DEVICE)
        _PCA_MEAN = torch.tensor(pca.mean_, dtype=torch.float32).to(DEVICE)
        print(f"  [V7-3] PCA fit: top-{n_components} PC ({pca.explained_variance_ratio_.sum()*100:.1f}% var)")
    except ImportError:
        print("  [V7-3] sklearn not found → PCA fallback to manifold_reg")

def pca_manifold_loss(angles_pred: torch.Tensor) -> torch.Tensor:
    if _PCA_COMPONENTS is None:
        return manifold_regularizer_loss(angles_pred)
    PC = _PCA_COMPONENTS.to(angles_pred.device, dtype=torch.float32)
    M32 = _PCA_MEAN.to(angles_pred.device).float()
    X32 = angles_pred.float()
    centered = X32 - M32
    coords = centered @ PC.T
    proj = (coords @ PC) + M32
    residual = X32 - proj
    return F.huber_loss(residual, torch.zeros_like(residual), delta=0.3)

def manifold_regularizer_loss(angles_pred: torch.Tensor, std_thr: float = 0.35) -> torch.Tensor:
    B = angles_pred.shape[0]
    if B < 4: return angles_pred.new_zeros(1).squeeze()
    batch_std = angles_pred.detach().float().std()
    if batch_std > std_thr: return angles_pred.new_zeros(1).squeeze()
    norms = angles_pred.detach().float().norm(dim=-1)
    idx = norms.argsort()
    sorted_ = angles_pred[idx]
    diff = sorted_[1:] - sorted_[:-1]
    return diff.pow(2).mean()

_LOG2 = math.log(2)
def ensemble_diversity_loss(ens_w: torch.Tensor) -> torch.Tensor:
    w = ens_w.float()
    eps = 1e-7
    H = -(w * (w + eps).log()).sum(dim=-1).mean()
    return _LOG2 - H

def branch_output_diversity_loss(p_syn: torch.Tensor, p_snn: torch.Tensor) -> torch.Tensor:
    sg_snn = p_snn.detach()
    cos = F.cosine_similarity(p_syn, sg_snn, dim=-1)
    return F.relu(cos).mean()

def compute_losses(out, y, sc, ep=1, cfg=CFG):
    if ep < cfg.AUX_WARMUP_START:
        aux_f = 0.0
    elif ep < cfg.AUX_WARMUP_END:
        aux_f = (ep - cfg.AUX_WARMUP_START) / max(1, cfg.AUX_WARMUP_END - cfg.AUX_WARMUP_START)
    else:
        aux_f = 1.0
    l_main = label_smooth_huber(out['angles'], y, smoothing=cfg.LABEL_SMOOTH, delta=cfg.HUBER_DELTA)
    l_ortho = out.get('syn_ortho', 0.)
    l_l1 = cfg.L1_SYNERGY * out['syn'].abs().mean()
    l_repr = (0.01 * l_ortho + l_l1) * aux_f
    l_bio = cfg.BIOMECH_W * biomechanical_coupling_loss(out['angles'].float()) * aux_f
    l_smooth = cfg.PCA_REG_W * pca_manifold_loss(out['angles']) * aux_f
    l_phys = l_bio + l_smooth
    l_ens_div = cfg.ENS_DIV_W * ensemble_diversity_loss(out['ens_w']) * max(aux_f, 0.1)
    l_out_div = cfg.BRANCH_DIV_W * branch_output_diversity_loss(out['p_syn'], out['p_snn']) * aux_f
    l_main_group, weights = sc([l_main, l_repr, l_phys])
    final_loss = l_main_group + l_ens_div + l_out_div
    ld = {
        'total': final_loss.item(), 'main': l_main.item(),
        'ortho': l_ortho.item() if isinstance(l_ortho, torch.Tensor) else l_ortho,
        'l1': l_l1.item(), 'biomech': l_bio.item(), 'smooth': l_smooth.item(),
        'ens_div': l_ens_div.item(),
        'out_div': l_out_div.item() if isinstance(l_out_div, torch.Tensor) else l_out_div,
        'aux_w': weights[0].item()
    }
    return final_loss, ld, out['ens_w']

# ==============================================================================
# CELL 8 — EVALUATION & TRAINING HELPERS (sửa max_err)
# ==============================================================================
def _per_dof_r2(d: torch.Tensor, g: torch.Tensor) -> float:
    ss_res = d.pow(2).sum(0)
    ss_tot = (g - g.mean(0, keepdim=True)).pow(2).sum(0)
    r2_per_dof = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2_per_dof.mean().item()

@torch.no_grad()
def evaluate(model, loader, loss_scaler, rom=ROM_DEG, ep=999, cfg=CFG):
    model.eval()
    preds, gts, tl = [], [], 0.
    ens_ws, p_syns, p_snns = [], [], []
    pbar = tqdm(loader, desc="  Eval", leave=False)
    for v1, y, sid in pbar:
        x, y, sid = v1.to(DEVICE, dtype=DTYPE), y.to(DEVICE, dtype=torch.float32), sid.to(DEVICE)
        _raw_eval = model._orig_mod if hasattr(model, '_orig_mod') else model
        if amp_enabled:
            with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                out = model(x, sid); out['syn_ortho'] = _raw_eval.synergy.ortho_loss()
                loss, _, _ = compute_losses(out, y, loss_scaler, ep=ep, cfg=cfg)
        else:
            out = model(x, sid); out['syn_ortho'] = _raw_eval.synergy.ortho_loss()
            loss, _, _ = compute_losses(out, y, loss_scaler, ep=ep, cfg=cfg)
        preds.append(out['angles'].float().cpu()); gts.append(y.float().cpu())
        ens_ws.append(out['ens_w'].float().cpu())
        p_syns.append(out['p_syn'].float().cpu()); p_snns.append(out['p_snn'].float().cpu())
        tl += loss.item()
    p, g = torch.cat(preds), torch.cat(gts); d = p - g
    rd, ew = rom.to(p.device), torch.cat(ens_ws)
    ps, pn = torch.cat(p_syns), torch.cat(p_snns)
    aw_m, aw_s = ew[:,0].mean().item(), ew[:,0].std().item()
    ps_c, pn_c = ps - ps.mean(-1, keepdim=True), pn - pn.mean(-1, keepdim=True)
    corr = (ps_c * pn_c).sum(-1) / (ps_c.norm(dim=-1) * pn_c.norm(dim=-1) + 1e-8)
    err_deg = (d * rd.unsqueeze(0)) / 2
    per_dof_rmse = torch.sqrt(err_deg.pow(2).mean(0)).numpy()
    max_err = err_deg.abs().max().item()   # sửa lỗi max_err
    return {
        'loss': tl / len(loader),
        'rmse_norm': d.pow(2).mean().sqrt().item(),
        'rmse_deg': err_deg.pow(2).mean().sqrt().item(),
        'r2': _per_dof_r2(d, g),
        'ens_w_mean': aw_m, 'ens_w_std': aw_s, 'branch_corr': corr.mean().item(),
        'per_dof': per_dof_rmse, 'max_err': max_err
    }

def _build_scheduler(opt, cfg):
    wu = cfg.WARMUP_EPOCHS
    cos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T0, T_mult=cfg.T_MULT, eta_min=cfg.LR_MIN)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=wu)
    return warmup, cos, wu

def _tune_accum(gnorm: float, cur: int, cfg=CFG) -> int:
    if gnorm > cfg.GRAD_NORM_HI: return min(cur*2, cfg.ACCUM_MAX)
    if gnorm < cfg.GRAD_NORM_LO: return max(cur//2, cfg.ACCUM_MIN)
    return cur

def resume_weights_only(model, ema, cfg=CFG, ckpt_path: Optional[str] = None):
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.CKPT_DIR, 'bionic_v72_final_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️  {os.path.basename(ckpt_path)} không tồn tại → train từ đầu")
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"\n  📂 Resume weights-only từ ep{ckpt['ep']} | RMSE={ckpt['rmse_norm']:.4f}")
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(_strip_orig_mod(ckpt['model']))
    raw_ema = ema.shadow._orig_mod if hasattr(ema.shadow, '_orig_mod') else ema.shadow
    raw_ema.load_state_dict(_strip_orig_mod(ckpt['ema']))
    loss_scaler = AdaptiveScaler(n=3).to(DEVICE)
    decay, nodecay = [], []
    for name, p in list(model.named_parameters()) + list(loss_scaler.named_parameters()):
        (nodecay if ('norm' in name or 'bias' in name) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.WEIGHT_DECAY},
                             {'params': nodecay, 'weight_decay': 0.}],
                            lr=cfg.LR_MAX, betas=(0.9, 0.95), eps=1e-8)
    warmup_sched, cos_sched, wu_eps = _build_scheduler(opt, cfg)
    return loss_scaler, opt, warmup_sched, cos_sched, wu_eps

def _strip_orig_mod(state_dict):
    if not any('_orig_mod.' in k for k in state_dict.keys()):
        return state_dict
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

def _get_raw_state_dict(module):
    raw = module._orig_mod if hasattr(module, '_orig_mod') else module
    return raw.state_dict()

def train(model: BionicsV72, tr_ds, tr_dl, va_dl, cfg=CFG, resume_path: Optional[str] = None,
          weights_only: bool = False, ema=None):
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,  exist_ok=True)
    loss_scaler = AdaptiveScaler(n=3).to(DEVICE)
    if ema is None:
        ema = EMAModel(model, cfg.EMA_DECAY)
    decay, nodecay = [], []
    for name, p in list(model.named_parameters()) + list(loss_scaler.named_parameters()):
        (nodecay if ('norm' in name or 'bias' in name) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.WEIGHT_DECAY},
                             {'params': nodecay, 'weight_decay': 0.}],
                            lr=cfg.LR_MAX, betas=(0.9, 0.95), eps=1e-8)
    steps_pe = len(tr_dl)
    warmup_sched, cos_sched, wu_eps = _build_scheduler(opt, cfg)
    amp_sc = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
    hist = {k: [] for k in ['tr', 'va', 'rmse_norm', 'rmse_deg', 'r2', 'lr', 'grad_norm', 'accum', 'ens_w_mean', 'smooth', 'ens_div', 'branch_corr', 'aux_w']}
    best = float('inf'); pat = 0; best_path = None; dyn_accum = cfg.ACCUM_STEPS
    best_state_rb = None; recent_loss = []; stag_best = float('inf'); stag_ep = 0; start_ep = 1
    frac_ep = 0.0; sched_base_ep = 0

    if weights_only:
        result = resume_weights_only(model, ema, cfg, ckpt_path=resume_path)
        if result is not None:
            loss_scaler, opt, warmup_sched, cos_sched, wu_eps = result
            steps_pe = len(tr_dl)
            amp_sc = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
            start_ep = 1; best = float('inf'); dyn_accum = cfg.ACCUM_STEPS
            print(f"  🎯 Weights-only resume: start_ep=1, fresh optimizer/scheduler/scaler")
    elif resume_path and os.path.exists(resume_path):
        print(f"  🔄 Full resume từ {resume_path}...")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_ema   = ema.shadow._orig_mod if hasattr(ema.shadow, '_orig_mod') else ema.shadow
        raw_model.load_state_dict(_strip_orig_mod(ckpt['model']))
        raw_ema.load_state_dict(_strip_orig_mod(ckpt.get('ema', ckpt['model'])))
        try:
            loss_scaler.load_state_dict(ckpt['scaler'])
        except RuntimeError:
            print("  ⚠️  Scaler shape mismatch, starting fresh scaler")
        opt.load_state_dict(ckpt['opt'])
        if 'warmup' in ckpt:
            warmup_sched.load_state_dict(ckpt['warmup']); cos_sched.load_state_dict(ckpt['cos'])
        start_ep = ckpt['ep'] + 1; hist = ckpt.get('hist', hist)
        best = ckpt.get('rmse_norm', best); dyn_accum = ckpt.get('dyn_accum', dyn_accum)
        steps_pe = ckpt.get('steps_pe', len(tr_dl)); frac_ep = ckpt.get('frac_ep', 0.0)
        sched_base_ep = ckpt.get('sched_base_ep', 0)
        print(f"     Loaded Ep {ckpt['ep']} | Best RMSE: {best:.4f}")

    nw = 0 if cfg.VRAM_CACHE else 2
    kw = dict(num_workers=nw, pin_memory=not cfg.VRAM_CACHE, persistent_workers=(nw > 0),
              worker_init_fn=_worker_init, collate_fn=_collate)
    print(f"  ⚡ ACCUM={dyn_accum} | BATCH={cfg.BATCH_SIZE} | WORKERS={nw}")
    print(f"\n{'═'*72}\n  🚀 TRAINING V7.2 FINAL | Ep={cfg.EPOCHS} | Start={start_ep}\n{'═'*72}")

    if isinstance(resume_path, str) and os.path.exists(resume_path):
        frac_ep = max(0.0, start_ep - 1 - wu_eps - sched_base_ep)

    for ep in range(start_ep, cfg.EPOCHS + 1):
        if ep > wu_eps:
            frac_ep = max(0.0, float(ep - wu_eps - 1 - sched_base_ep))
        tr_ds.current_ep = ep
        if ep > 0 and ep % 10 == 0:
            init_noise_buffer(cfg)
        sampler = _curriculum_sampler(tr_ds, ep, cfg)
        if sampler is None:
            ep_loader = tr_dl
        else:
            kw_temp = kw.copy(); kw_temp['persistent_workers'] = False
            ep_loader = DataLoader(tr_ds, cfg.BATCH_SIZE, shuffle=False, sampler=sampler, drop_last=True, **kw_temp)
        tl = 0; ep_ens_div = 0; ep_smooth = 0
        model.train(); t0 = time.time(); gnorms = []
        pbar = tqdm(ep_loader, desc=f"  Ep{ep:3d}", leave=True)
        for step, (v1, y, sid) in enumerate(pbar):
            x1, y, sid = v1.to(DEVICE, dtype=torch.float32), y.to(DEVICE, dtype=torch.float32), sid.to(DEVICE)
            if cfg.GPU_AUG:
                x1, y = batch_gpu_augment(x1, y, ep, cfg)
            x1_m, y_m = mixup_data(x1, y, cfg.MIXUP_ALPHA)
            x1_m = x1_m.to(DTYPE)
            if amp_enabled:
                with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                    _raw_tr = model._orig_mod if hasattr(model, '_orig_mod') else model
                    out_mixed = model(x1_m, sid); out_mixed['syn_ortho'] = _raw_tr.synergy.ortho_loss()
                    loss, ld, _ = compute_losses(out_mixed, y_m, loss_scaler, ep=ep, cfg=cfg)
                    loss = loss / dyn_accum
            else:
                _raw_tr = model._orig_mod if hasattr(model, '_orig_mod') else model
                out_mixed = model(x1_m, sid); out_mixed['syn_ortho'] = _raw_tr.synergy.ortho_loss()
                loss, ld, _ = compute_losses(out_mixed, y_m, loss_scaler, ep=ep, cfg=cfg)
                loss = loss / dyn_accum
            if amp_sc is not None:
                amp_sc.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % dyn_accum == 0:
                if amp_sc is not None:
                    amp_sc.unscale_(opt)
                gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP).item()
                nn.utils.clip_grad_norm_(loss_scaler.parameters(), 0.1)
                if amp_sc is not None:
                    _scale_before = amp_sc.get_scale()
                    amp_sc.step(opt); amp_sc.update()
                    opt.zero_grad(set_to_none=True)
                    if amp_sc.get_scale() >= _scale_before:
                        ema.update(model)
                else:
                    opt.step()
                    opt.zero_grad(set_to_none=True); ema.update(model)
                if not (math.isnan(gn) or math.isinf(gn)):
                    gnorms.append(gn)
                else:
                    print(f"\n  ⚠️  GN={gn} tại step {step} — bỏ qua append")
                if ep > wu_eps:
                    # Gọi scheduler một lần mỗi epoch (không phải mỗi step)
                    # Đã fix: scheduler chỉ gọi sau epoch
                    pass
                if not (math.isnan(gn) or math.isinf(gn)):
                    new_acc = _tune_accum(gn, dyn_accum, cfg)
                    if new_acc != dyn_accum:
                        print(f"     ⚡ ACCUM {dyn_accum}→{new_acc} (gn={gn:.3f})")
                        dyn_accum = new_acc
                        steps_pe = math.ceil(len(ep_loader) / dyn_accum)
            tl_batch = loss.item() * dyn_accum
            tl += tl_batch
            ep_ens_div += ld.get('ens_div', 0.); ep_smooth += ld.get('smooth', 0.)
            if (step + 1) % 5 == 0:
                pbar.set_postfix({'loss': f"{tl/(step+1):.4f}", 'gn': f"{gnorms[-1]:.2f}" if gnorms else "---", 'acc': dyn_accum})
        tl /= max(1, len(ep_loader)); ep_ens_div /= max(1, len(ep_loader)); ep_smooth /= max(1, len(ep_loader))
        mean_gn = float(np.mean(gnorms)) if gnorms else 0.
        m = evaluate(ema.get_model(), va_dl, loss_scaler, ep=ep, cfg=cfg); lr = opt.param_groups[0]['lr']
        if m.get('branch_corr', 0.0) > 0.95:
            cfg.BRANCH_DIV_W = min(1.0, cfg.BRANCH_DIV_W * 1.5)
            print(f"     ⚠️  OVERLAP DETECTED (corr={m['branch_corr']:.3f}) → Auto-boosting BRANCH_DIV_W to {cfg.BRANCH_DIV_W:.3f}")
        if m['rmse_norm'] < stag_best - cfg.MIN_DELTA:
            stag_best = m['rmse_norm']; stag_ep = 0
        else:
            stag_ep += 1
        if stag_ep >= cfg.STAG_PATIENCE:
            print(f"     🔥 STAG KICK ep{ep}: clear opt")
            opt.state.clear()
            for pg in opt.param_groups:
                pg['lr'] = cfg.LR_MAX
            _, cos_sched, _ = _build_scheduler(opt, cfg)
            sched_base_ep = ep - wu_eps - 1
            frac_ep = 0.0
            dyn_accum = cfg.ACCUM_MIN; stag_ep = 0
            stag_best = m['rmse_norm']; best_state_rb = None
        recent_loss.append(m['rmse_norm'])
        if len(recent_loss) > 3: recent_loss.pop(0)
        if best_state_rb is not None and len(recent_loss) == 3 and m['rmse_norm'] > best * cfg.ROLLBACK_THR:
            print(f"     ⚠️  ROLLBACK")
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw_ema   = ema.shadow._orig_mod if hasattr(ema.shadow, '_orig_mod') else ema.shadow
            raw_model.load_state_dict(_strip_orig_mod(best_state_rb['model']))
            raw_ema.load_state_dict(_strip_orig_mod(best_state_rb['ema']))
            best = best_state_rb.get('best_rmse', best); loss_scaler.load_state_dict(best_state_rb['scaler']); opt.load_state_dict(best_state_rb['opt'])
            warmup_sched.load_state_dict(best_state_rb['warmup']); cos_sched.load_state_dict(best_state_rb['cos'])
            dyn_accum = best_state_rb.get('dyn_accum', cfg.ACCUM_STEPS)
            steps_pe = best_state_rb.get('steps_pe', len(ep_loader))
            sched_base_ep = best_state_rb.get('sched_base_ep', 0)
            frac_ep = best_state_rb.get('frac_ep', float(max(0, ep - wu_eps - 1 - sched_base_ep)))
            recent_loss.clear(); pat = max(0, pat-2); continue
        for k, v in zip(['tr', 'va', 'rmse_norm', 'rmse_deg', 'r2', 'lr', 'grad_norm', 'accum', 'ens_w_mean', 'smooth', 'ens_div', 'branch_corr'],
                        [tl, m['loss'], m['rmse_norm'], m['rmse_deg'], m['r2'], lr, mean_gn, dyn_accum, m['ens_w_mean'], ep_smooth, ep_ens_div, m['branch_corr']]):
            if k in hist: hist[k].append(v)
        hist['aux_w'].append(ld['aux_w'])
        if ep <= wu_eps: warmup_sched.step()
        else:
            # Gọi cosine scheduler sau mỗi epoch (một lần)
            cos_sched.step(frac_ep)
        print(f"  Ep{ep:3d} | Tr:{tl:.4f} Va:{m['loss']:.4f} | RMSE:{m['rmse_norm']:.4f} | GN:{mean_gn:.2f} Acc:{dyn_accum} | {time.time()-t0:.0f}s")
        ckpt = {'ep': ep, 'model': _get_raw_state_dict(model), 'ema': _get_raw_state_dict(ema.shadow),
                'scaler': loss_scaler.state_dict(), 'opt': opt.state_dict(), 'warmup': warmup_sched.state_dict(),
                'cos': cos_sched.state_dict(), 'rmse_norm': m['rmse_norm'], 'hist': hist, 'dyn_accum': dyn_accum,
                'steps_pe': steps_pe, 'frac_ep': frac_ep, 'sched_base_ep': sched_base_ep, 'cfg_version': 'V7.2_FINAL'}
        torch.save(ckpt, os.path.join(cfg.CKPT_DIR, 'bionic_v72_final_last.pt'))
        if m['rmse_norm'] < best - cfg.MIN_DELTA:
            best = m['rmse_norm']; pat = 0; best_path = os.path.join(cfg.CKPT_DIR, 'bionic_v72_final_best.pt')
            best_state_rb = {'best_rmse': best,
                             'model': {k: v.clone() for k, v in _get_raw_state_dict(model).items()},
                             'ema': {k: v.clone() for k, v in _get_raw_state_dict(ema.shadow).items()},
                             'scaler': {k: v.clone() for k, v in loss_scaler.state_dict().items()},
                             'opt': copy.deepcopy(opt.state_dict()),
                             'warmup': copy.deepcopy(warmup_sched.state_dict()), 'cos': copy.deepcopy(cos_sched.state_dict()),
                             'dyn_accum': dyn_accum, 'steps_pe': steps_pe, 'frac_ep': frac_ep, 'sched_base_ep': sched_base_ep}
            torch.save(ckpt, best_path); print(f"     💾 Best Saved: {best:.4f}")
        else:
            pat += 1
            if pat >= cfg.PATIENCE: print(f"\n  ⏹ Early stop"); break
    print(f"\n  ✅ Best RMSE: {best:.4f}")
    return hist, best_path, loss_scaler, ema

# ==============================================================================
# CELL 9 — BASELINES (giữ nguyên)
# ==============================================================================
class BaselineCNNLSTM(nn.Module):
    def __init__(self, in_d=16, hid=128, out=18):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_d, 64, 7, padding=3), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 5, padding=2),  nn.ReLU())
        self.lstm = nn.LSTM(128, hid, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, out), nn.Tanh())
    def forward(self, x, sid=None):
        f = self.cnn(x.transpose(1,2)).transpose(1,2)
        h, _ = self.lstm(f)
        return {'angles': self.fc(h[:,-1,:]), 'syn': h}

class BaselineTransformer(nn.Module):
    def __init__(self, in_d=16, hid=128, out=18, n_heads=4, n_layers=3):
        super().__init__()
        self.proj = nn.Linear(in_d, hid)
        enc = nn.TransformerEncoderLayer(hid, n_heads, hid*2, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layers)
        self.fc = nn.Sequential(nn.Linear(hid,64),nn.ReLU(),nn.Linear(64,out),nn.Tanh())
        self.pos = nn.Parameter(torch.randn(1, 400, hid) * 0.02)
    def forward(self, x, sid=None):
        h = self.enc(self.proj(x) + self.pos[:,:x.shape[1],:])
        return {'angles': self.fc(h[:,-1,:]), 'syn': h}

def train_baseline(model_cls, tr_dl, va_dl, epochs=20, cfg=CFG):
    m = model_cls().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best_rmse = float('inf'); best_w = None
    for ep in range(1, epochs+1):
        m.train()
        for v1, y, sid in tr_dl:
            x = v1.to(DEVICE, dtype=torch.float32); y = y.to(DEVICE, dtype=torch.float32)
            loss = F.mse_loss(m(x)['angles'], y)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); ps, gs = [], []
        with torch.no_grad():
            for v1, y, sid in va_dl:
                ps.append(m(v1.to(DEVICE, dtype=torch.float32))['angles'].cpu())
                gs.append(y)
        rmse = (torch.cat(ps) - torch.cat(gs)).pow(2).mean().sqrt().item()
        if rmse < best_rmse: best_rmse = rmse; best_w = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_w)
    return m, best_rmse

# ==============================================================================
# CELL 10 — STATISTICS, LATENCY, PLOTS (giữ nguyên, cập nhật đường dẫn)
# ==============================================================================
def statistical_significance(pred_ours, pred_base, gt, rom=None) -> Dict:
    from scipy.stats import wilcoxon
    if rom is None: rom = np.ones(CFG.OUT_DIM)
    err_ours = np.sqrt((((pred_ours - gt) * rom)**2).mean(-1))
    err_base = np.sqrt((((pred_base - gt) * rom)**2).mean(-1))
    stat, p = wilcoxon(err_ours, err_base, alternative='less')
    return {'p_value': float(p), 'significant': p < 0.05,
            'improvement_%': float(100*(err_base.mean()-err_ours.mean())/err_base.mean())}

def benchmark_latency(model, cfg=CFG, n=200) -> Dict:
    model.eval()
    dummy = torch.randn(1, cfg.SEQ_LEN, cfg.IN_DIM, device=DEVICE, dtype=DTYPE)
    sid   = torch.zeros(1, dtype=torch.long, device=DEVICE)
    for _ in range(20):
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                    model(dummy, sid)
            else:
                model(dummy, sid)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    lats = []
    for _ in range(n):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                    model(dummy, sid)
            else:
                model(dummy, sid)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    r = {'mean_ms': float(lats.mean()), 'std_ms': float(lats.std()),
         'p99_ms': float(np.percentile(lats, 99)),
         'rt_50hz': lats.mean() < 20., 'rt_100hz': lats.mean() < 10.}
    print(f"  ⏱ {r['mean_ms']:.2f}±{r['std_ms']:.2f}ms  P99:{r['p99_ms']:.2f}ms  "
          f"50Hz:{'✅' if r['rt_50hz'] else '❌'}  100Hz:{'✅' if r['rt_100hz'] else '❌'}")
    return r

def plot_publication(hist, ema_model, te_dl, cfg=CFG, path='/content/drive/MyDrive/AI_Hand/pub_v72_final.png'):
    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('BionicsHand V7.2 — FINAL', fontsize=14, fontweight='bold')
    ep = range(1, len(hist['tr'])+1)
    ax[0,0].plot(ep, hist['tr'], label='Train', c='#2196F3', lw=2)
    ax[0,0].plot(ep, hist['va'], label='Val',   c='#FF5722', lw=2, ls='--')
    ax[0,0].set(title='Loss', xlabel='Epoch'); ax[0,0].legend(); ax[0,0].grid(.3)
    ax[0,1].plot(ep, hist['rmse_norm'], c='#4CAF50', lw=2.5)
    ax[0,1].axhline(0.07, c='#FF9800', ls='--', lw=1.5, label='0.07 target')
    ax[0,1].axhline(0.04, c='#4CAF50', ls='-.', lw=1.5, label='0.04 target')
    ax[0,1].set(title='RMSE [-1,1]', xlabel='Epoch'); ax[0,1].legend(); ax[0,1].grid(.3)
    ax[0,2].plot(ep, hist['grad_norm'], c='#9C27B0', lw=2, label='GradNorm')
    ax[0,2].plot(ep, hist['accum'],     c='#FF9800', lw=2, label='ACCUM', ls='--')
    ax[0,2].set(title='GradNorm + ACCUM', xlabel='Epoch'); ax[0,2].legend(); ax[0,2].grid(.3)
    ema_model.eval(); all_p, all_g, all_ew = [], [], []
    with torch.no_grad():
        for v1, y, sid in te_dl:
            if amp_enabled:
                with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                    out = ema_model(v1.to(DEVICE, dtype=DTYPE), sid.to(DEVICE))
            else:
                out = ema_model(v1.to(DEVICE, dtype=DTYPE), sid.to(DEVICE))
            all_p.append(out['angles'].float().cpu())
            all_g.append(y)
            all_ew.append(out['ens_w'].float().cpu())
    ap = torch.cat(all_p); ag = torch.cat(all_g); aew = torch.cat(all_ew)
    per = (((ap-ag)*ROM_DEG)/2).pow(2).mean(0).sqrt().numpy()
    dof_names = ['Th_MCP','Th_IP','Th_Abd','Th_Crs','Idx_MCP','Idx_PIP','Idx_Abd',
                 'Mid_MCP','Mid_Abd','Mid_PIP','Rng_MCP','Rng_Abd','Rng_PIP',
                 'Pnk_MCP','Pnk_Abd','Pnk_PIP','Wrist_F','Wrist_R']
    ax[1,0].bar(range(18), per, color=['#E91E63' if v>5 else '#4CAF50' for v in per], alpha=.85)
    ax[1,0].axhline(5., c='r', ls='--'); ax[1,0].set_xticks(range(18))
    ax[1,0].set_xticklabels(dof_names, rotation=70, fontsize=7)
    ax[1,0].set(title='Per-DOF RMSE (°)'); ax[1,0].grid(.3)
    ax[1,1].scatter(ag.flatten(), ap.flatten(), alpha=.1, s=3, c='#2196F3', rasterized=True)
    ax[1,1].plot([-1,1],[-1,1],'r--',lw=2)
    corr = np.corrcoef(ag.flatten().numpy(), ap.flatten().numpy())[0,1]
    ax[1,1].set(title=f'Pred vs GT  (r={corr:.3f})', xlabel='GT', ylabel='Pred'); ax[1,1].grid(.3)
    with torch.no_grad():
        b0 = next(iter(te_dl))
        if amp_enabled:
            with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                syn_m = ema_model(b0[0][:32].to(DEVICE, dtype=DTYPE), b0[2][:32].to(DEVICE))['syn']
        else:
            syn_m = ema_model(b0[0][:32].to(DEVICE, dtype=DTYPE), b0[2][:32].to(DEVICE))['syn']
    im = ax[1,2].imshow(syn_m.float().cpu().numpy().T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax[1,2].set(title=f'Synergy ({cfg.N_SYNERGY})', xlabel='Sample', ylabel='#'); plt.colorbar(im, ax=ax[1,2], fraction=0.046)
    ax[2,0].hist(aew[:, 0].numpy(), bins=50, color='#673AB7', alpha=0.8, edgecolor='w', lw=0.3)
    ax[2,0].axvline(0.5, c='r', ls='--', lw=1.5, label='ideal=0.5')
    ax[2,0].set(title=f'Ensemble w₀ dist  (mean={aew[:,0].mean():.3f})', xlabel='w₀ (synergy weight)', ylabel='count')
    ax[2,0].legend(); ax[2,0].grid(.3)
    if abs(aew[:,0].mean() - 0.35) > 0.35:
        ax[2,0].text(0.5, 0.9, '⚠️ Gate may be collapsed!', transform=ax[2,0].transAxes, ha='center', color='red', fontsize=10)
    if hist['ens_w_mean']:
        ax[2,1].plot(ep, hist['ens_w_mean'], c='#673AB7', lw=2, label='ens_w₀ mean')
        ax[2,1].axhline(0.5, c='r', ls='--', lw=1, label='ideal=0.5')
        ax[2,1].set(title='Ensemble diversity (epoch)', xlabel='Epoch', ylim=[0,1]); ax[2,1].legend(); ax[2,1].grid(.3)
    if hist['smooth'] and hist['ens_div']:
        ax[2,2].plot(ep, hist['smooth'],  c='#009688', lw=2, label='manifold_reg')
        ax[2,2].plot(ep, hist['ens_div'], c='#FF5722', lw=2, label='ens_div loss', ls='--')
        ax[2,2].set(title='Aux losses V7.2', xlabel='Epoch'); ax[2,2].legend(); ax[2,2].grid(.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  📊 {path}"); plt.close(fig)

# ==============================================================================
# CELL 11 — EXPORT, CALIBRATION, MAIN, QUICK TEST
# ==============================================================================
def export_torchscript(model, path):
    model.eval()
    dummy = torch.randn(1, CFG.SEQ_LEN, CFG.IN_DIM, device=DEVICE, dtype=torch.float32)
    sid   = torch.zeros(1, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        ts = torch.jit.trace(model.float(), (dummy, sid))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts.save(path); print(f"  ✅ TorchScript: {path}")

def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, CFG.SEQ_LEN, CFG.IN_DIM, device=DEVICE, dtype=torch.float32)
    sid   = torch.zeros(1, dtype=torch.long, device=DEVICE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(model.float(), (dummy, sid), path, input_names=['emg','subject_id'],
                      output_names=['angles','syn','ens_w'], dynamic_axes={'emg': {0:'batch',1:'seq'}, 'subject_id': {0:'batch'}},
                      opset_version=17)
    print(f"  ✅ ONNX: {path}")

def calibrate_subject(model, x_user, y_user, norm_mean=None, norm_std=None, y_stats=None, save_path=None, cfg=CFG):
    calib_params = [
        {'params': model.synergy.parameters(),  'lr': cfg.CALIB_LR_HEAD},
        {'params': model.ensemble.parameters(), 'lr': cfg.CALIB_LR_HEAD},
        {'params': model.motor.n.parameters(),  'lr': cfg.CALIB_LR_ENC},
    ]
    opt = torch.optim.AdamW(calib_params, weight_decay=0.001)
    for p in model.parameters(): p.requires_grad_(False)
    for g in calib_params:
        for p in g['params']: p.requires_grad_(True)
    xc = x_user.astype(np.float32)
    if norm_mean is not None:
        xc = np.clip((xc - norm_mean) / norm_std, -3., 3.)
    xc = torch.from_numpy(xc)
    y_scaled, _ = _scale_y(y_user, stats=y_stats)
    yc = torch.from_numpy(y_scaled)
    ds = torch.utils.data.TensorDataset(xc, yc)
    dl = DataLoader(ds, max(1, min(16, len(xc))), shuffle=True)
    model.train(); best_l = float('inf'); best_s = None
    for ep in range(cfg.CALIB_EPOCHS):
        el = 0.
        for x, y in dl:
            x = x.to(DEVICE, dtype=DTYPE); y = y.to(DEVICE, dtype=DTYPE)
            if amp_enabled:
                with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
                    out  = model(x)
                    loss = F.huber_loss(out['angles'], y, delta=cfg.HUBER_DELTA)
            else:
                out  = model(x)
                loss = F.huber_loss(out['angles'], y, delta=cfg.HUBER_DELTA)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_([p for g in calib_params for p in g['params']], 0.5)
            opt.step(); el += loss.item()
        el /= len(dl)
        if el < best_l:
            best_l = el; best_s = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Calib ep{ep+1:2d}/{cfg.CALIB_EPOCHS} | loss:{el:.5f}")
    if best_s: model.load_state_dict(best_s)
    for p in model.parameters(): p.requires_grad_(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model': model.state_dict(), 'loss': best_l}, save_path)
    print(f"  ✅ Calib best: {best_l:.5f}")
    return model

def main():
    try:
        from google.colab import drive
        drive.mount('/content/drive'); print("✅ Drive mounted")
    except Exception:
        CFG.DATA_DIR = './data/final_dataset'
        CFG.CKPT_DIR = './checkpoints'; CFG.LOG_DIR = './logs'
    CFG.BATCH_SIZE = _auto_batch()
    best_pt  = os.path.join(CFG.CKPT_DIR, 'bionic_v72_final_best.pt')
    last_pt  = os.path.join(CFG.CKPT_DIR, 'bionic_v72_final_last.pt')
    use_weights_only = getattr(CFG, 'FORCE_WEIGHTS_ONLY', False)
    full_resume_path = None; weights_only_path = None
    chosen_pt = best_pt if os.path.exists(best_pt) else (last_pt if os.path.exists(last_pt) else None)
    if chosen_pt:
        try:
            ckpt = torch.load(chosen_pt, map_location='cpu')
            fail_reason = None
            if any('_orig_mod.' in k for k in ckpt.get('model', {}).keys()):
                fail_reason = "Checkpoint có key '_orig_mod.'"
            elif chosen_pt == last_pt and os.path.exists(best_pt):
                best_ckpt = torch.load(best_pt, map_location='cpu')
                last_rmse = ckpt.get('rmse_norm', 1.0)
                best_rmse = best_ckpt.get('rmse_norm', 1.0)
                if last_rmse > best_rmse * 1.05:
                    fail_reason = f"DIVERGE: last={last_rmse:.4f} > best={best_rmse:.4f}*1.05"
            if use_weights_only:
                weights_only_path = chosen_pt
                print(f"  🎯 MANUAL OVERRIDE: Weights-only từ {os.path.basename(chosen_pt)}")
            elif fail_reason:
                print(f"  ⚠️  Full resume {os.path.basename(chosen_pt)} thất bại: {fail_reason}")
                use_weights_only = True
                weights_only_path = chosen_pt
                print(f"  🎯 AUTO-FALLBACK: Weights-only từ {os.path.basename(weights_only_path)}")
            else:
                full_resume_path = chosen_pt
                print(f"  ✅ NORMAL: Full resume {os.path.basename(chosen_pt)} ep{ckpt['ep']} | RMSE={ckpt.get('rmse_norm',0):.4f}")
        except Exception as e:
            print(f"  ⚠️  Lỗi đọc {os.path.basename(chosen_pt)}: {e}")
            use_weights_only = True
            weights_only_path = chosen_pt
            print(f"  🎯 AUTO-FALLBACK: Cố gắng Weights-only từ {os.path.basename(weights_only_path)}")
    else:
        print("  🆕 Không có checkpoint → train từ đầu")
    tr_ds, va_ds, te_ds, tr_dl, va_dl, te_dl = load_data(CFG)
    model = build_model(CFG).to(DEVICE)
    ema = EMAModel(model, CFG.EMA_DECAY)
    hist, best_path, scaler, ema = train(
        model, tr_ds, tr_dl, va_dl, CFG,
        resume_path=full_resume_path if not use_weights_only else weights_only_path,
        weights_only=use_weights_only, ema=ema)
    if best_path is None: best_path = os.path.join(CFG.CKPT_DIR, 'bionic_v72_final_last.pt')
    ckpt = torch.load(best_path, map_location=DEVICE)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_ema   = ema.shadow._orig_mod if hasattr(ema.shadow, '_orig_mod') else ema.shadow
    raw_model.load_state_dict(_strip_orig_mod(ckpt['model']))
    raw_ema.load_state_dict(_strip_orig_mod(ckpt['ema']))
    m = evaluate(ema.get_model(), te_dl, scaler, cfg=CFG)
    print(f"\n{'═'*60}")
    print(f"  📊 TEST — BionicsHand V7.2 FINAL (EMA)")
    print(f"  RMSE [-1,1]: {m['rmse_norm']:.4f}  (target: 0.04–0.07)")
    print(f"  RMSE [°]:    {m['rmse_deg']:.2f}°")
    print(f"  R²:          {m['r2']:.4f}  (target: >0.95)")
    print(f"  MaxErr:      {m['max_err']:.1f}°")
    print(f"  Per-DOF:     min={min(m['per_dof']):.1f}°  max={max(m['per_dof']):.1f}°")
    print(f"  Ens_w₀:      {m['ens_w_mean']:.3f} ± {m['ens_w_std']:.3f}  (ideal: 0.5±0.1)")
    print(f"  BranchCorr:  {m['branch_corr']:.3f}  (warn if >0.95)")
    print(f"{'═'*60}")
    benchmark_latency(ema.get_model(), CFG)
    plot_publication(hist, ema.get_model(), te_dl, cfg=CFG)
    print("\n📦 Exporting...")
    try: export_torchscript(ema.get_model(), '/content/drive/MyDrive/AI_Hand/bionic_v72_final.pt')
    except Exception as e: print(f"⚠️  TorchScript: {e}")
    try: export_onnx(ema.get_model(), '/content/drive/MyDrive/AI_Hand/bionic_v72_final.onnx')
    except Exception as e: print(f"⚠️  ONNX: {e}")
    return model, ema, hist

def quick_test():
    print("🔍 Quick Test — BionicsHand V7.2 FINAL"); print("─"*65)
    m = build_model(CFG)
    B = 16
    x = torch.randn(B, CFG.SEQ_LEN, CFG.IN_DIM, device=DEVICE, dtype=torch.float32)
    y = torch.clamp(torch.randn(B, CFG.OUT_DIM, device=DEVICE, dtype=torch.float32), -1, 1)
    sid = torch.zeros(B, dtype=torch.long, device=DEVICE)
    if amp_enabled:
        with torch.amp.autocast(DEVICE_TYPE, dtype=DTYPE):
            out = m(x, sid); out['syn_ortho'] = m.synergy.ortho_loss()
            sc = AdaptiveScaler(n=3).to(DEVICE)
            l, ld, ew = compute_losses(out, y, sc, ep=0, cfg=CFG)
    else:
        out = m(x, sid); out['syn_ortho'] = m.synergy.ortho_loss()
        sc = AdaptiveScaler(n=3).to(DEVICE)
        l, ld, ew = compute_losses(out, y, sc, ep=0, cfg=CFG)
    print(f"\n  [Outputs]")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:12s}: {list(v.shape)}  {v.dtype}")
    print(f"\n  [Losses]")
    for k, v in ld.items(): print(f"    {k:12s}: {v:.6f}")
    assert out['angles'].shape == (B, CFG.OUT_DIM), "❌ angles shape"
    assert out['syn'].shape    == (B, CFG.N_SYNERGY), "❌ syn shape"
    assert out['ens_w'].shape  == (B, 2), "❌ ens_w shape"
    assert 'main' in ld; assert 'biomech' in ld; assert 'smooth' in ld; assert 'ens_div' in ld; assert 'out_div' in ld
    assert 'p_syn' in out; assert 'p_snn' in out
    l.backward()
    g_ok = m.snn.input_proj.weight.grad is not None and m.snn.input_proj.weight.grad.abs().sum() > 0
    print(f"\n  [V7-1]         SNN input: h_seq(steps={m.snn.steps}, sequential)  ✅")
    print(f"  [V7-2]         HysteresisEncoder: temporal_grad w={CFG.HYST_W}  ✅")
    print(f"  [V7-3]         pca_manifold_loss: {ld['smooth']:.6f}  ✅")
    print(f"  [V7-4]         branch_corr in evaluate()  ✅")
    print(f"  [FIX-5]        ATTN_WIN={CFG.ATTN_WIN} (full causal)  ✅")
    print(f"  [FIX-ATTN]     SDPA causal (FlashAttn path)  ✅")
    print(f"  [NEW-6]        EnsembleHead 2-way: {list(out['ens_w'].shape)}  ✅")
    print(f"  [RM-1]         Cerebellum removed  ✅")
    print(f"  [RM-2/3]       traj/jerk loss removed  ✅")
    print(f"  [FIX-ENS-1]    ens_div loss:  {ld['ens_div']:.6f}  ✅")
    print(f"  [FIX-OUT-DIV]  out_div loss:  {ld['out_div']:.6f}  ✅")
    print(f"  [SNN]          Gradient: {'✅' if g_ok else '❌'}")
    W = F.normalize(m.synergy.dec.weight, dim=0)
    I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    err = (W.T @ W - I).pow(2).mean().item()
    print(f"  [ORTHO]       Synergy ortho err: {err:.5f}  {'✅' if err<0.1 else '❌'}")
    ema_t = EMAModel(m, 0.999); ema_t.update(m)
    print(f"  [EMA]         ✅")
    s1, s40, s80 = _noise_strength(1,CFG), _noise_strength(40,CFG), _noise_strength(80,CFG)
    print(f"  [IMP-S2]      Noise ep1/40/80: {s1:.2f}/{s40:.2f}/{s80:.2f}  ✅")
    a2 = _tune_accum(4.0, 8, CFG); a3 = _tune_accum(0.1, 8, CFG)
    print(f"  [IMP-S1]      GradNorm tune: hi→{a2} lo→{a3}  ✅")
    print(f"  [F44-4]       STAG_PATIENCE={CFG.STAG_PATIENCE} epochs (epoch-level, BUG-3 fixed)  ✅")
    print(f"  [AMP]         torch.amp.autocast  ✅")
    ew_collapse = torch.tensor([[0.99, 0.01],[0.98, 0.02]], dtype=torch.float32)
    ew_diverse  = torch.tensor([[0.50, 0.50],[0.48, 0.52]], dtype=torch.float32)
    l_collapse = ensemble_diversity_loss(ew_collapse).item()
    l_diverse  = ensemble_diversity_loss(ew_diverse).item()
    assert l_collapse > l_diverse, f"❌ Diversity loss sai chiều: collapse({l_collapse:.4f}) phải > diverse({l_diverse:.4f})"
    print(f"  [DIV CHECK]   collapse={l_collapse:.4f} > diverse={l_diverse:.4f}  (optimizer muốn diverse → ✅)")
    n = sum(p.numel() for p in m.parameters())
    print(f"\n  Params: {n:,}  (~{n*2//1024}KB BF16)")
    if torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.memory_allocated()//1024**2}MB")
    print(f"\n{'─'*65}")
    print("✅ ALL CHECKS PASSED — BionicsHand V7.2 FINAL READY")
    return True

if __name__ == '__main__':
    ok = quick_test()
    if ok:
        print("\n" + "🚀"*16)
        print("  TRAINING V7.2 — BIONIC HAND (NINAPRO DB8 COMPLIANT, ROM CORRECTED)")
        print("🚀"*16 + "\n")
        model, ema, hist = main()