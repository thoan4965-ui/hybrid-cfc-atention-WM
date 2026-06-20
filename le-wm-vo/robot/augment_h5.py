"""
Augment H5 dataset: 50% gốc + 50% augmented (background replace + ColorJitter).
Per-sequence augment: 6 consecutive frames share the same color params.
Usage:
  .venv\Scripts\python.exe augment_h5.py path/to/bionic_hand_dataset_v3_96.h5 --bg bg.png --output bionic_hand_v4_aug.h5
"""
import argparse
import h5py
import numpy as np
import torch
import random
from pathlib import Path
from torchvision.transforms import functional as F, functional as F
import random

IMG_SIZE = 96
HUE_MAX = 0.3
BRIGHT_MAX = 0.5
SAT_MAX = 0.3
SEQ = 6  # history 3 + predict 3


def load_bg(path):
    """Load background image (ko tay). Returns (96,96,3) float32 [0,1]."""
    import cv2
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return img


def bg_replace(frame, bg):
    diff = np.abs(frame.astype(float) - bg.astype(float))
    mask = diff.mean(axis=-1) > 0.05
    aug = frame.copy()
    aug[~mask] = bg[~mask]
    return aug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", type=str, help="Path to original H5")
    parser.add_argument("--bg", type=str, required=True, help="Background image (no hand)")
    parser.add_argument("--output", type=str, default=None, help="Output H5 path")
    args = parser.parse_args()
    
    bg = load_bg(args.bg)
    inp = Path(args.input_h5)
    out = Path(args.output or inp.stem + "_aug.h5")
    
    from torchvision.transforms import functional as F
    import random
    
    print(f"Loading H5: {inp}")
    with h5py.File(inp, 'r') as f:
        N = f['pixels'].shape[0]
        pix = f['pixels'][:]  # (N, 96, 96, 3)
        act = f['action'][:]  # (N, 8)
    
    print(f"  Total frames: {N}")
    N2 = N * 2
    half = N
    
    pix_out = np.empty((N2, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    act_out = np.empty((N2, 8), dtype=np.float32)
    
    # Copy original: first half
    print("Copying original frames...")
    pix_out[:half] = pix  # uint8 gốc
    act_out[:half] = act.astype(np.float32)
    
    # Augment: second half
    print("Augmenting second half (bg replace + per-sequence ColorJitter)...")
    for i in range(0, N, SEQ):
        if (i+1) % 1000 == 0:
            print(f"  {min(i+SEQ, N)}/{N}")
        
        # Generate ONE jitter params for this 6-frame sequence
        bright = 1.0 + BRIGHT_MAX * (2 * random.random() - 1)
        hue = HUE_MAX * (2 * random.random() - 1)
        sat = 1.0 + SAT_MAX * (2 * random.random() - 1)
        
        end = min(i + SEQ, N)
        for j in range(i, end):
            frame = pix[j].astype(np.float32) / 255.0
            frame = bg_replace(frame, bg)
            
            # Apply same color params to every frame in this sequence
            t = torch.from_numpy(frame).permute(2, 0, 1)
            t = F.adjust_brightness(t, bright)
            t = F.adjust_hue(t, hue)
            t = F.adjust_saturation(t, sat)
            
            frame = t.permute(1, 2, 0).numpy()
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            
            idx = half + j
            if idx < N2:
                pix_out[idx] = frame
                act_out[idx] = act[j].astype(np.float32)
    
    # Save
    print(f"Saving {N2} frames to {out} (uint8, gzip)...")
    with h5py.File(out, 'w') as f:
        f.create_dataset('pixels', data=pix_out, compression='gzip', compression_opts=6)
        f.create_dataset('action', data=act_out)
    
    # Clean up float32→uint8 cho consistent với old format
    print(f"  pixels shape: {pix_out.shape}, dtype: {pix_out.dtype}")
    print(f"  action shape: {act_out.shape}")
    print(f"✓ Done: {out}")
    print(f"  Original: {N} frames")
    print(f"  Augmented: {N} frames")
    print(f"  Total: {N2} frames (use this for fine-tune)")


if __name__ == "__main__":
    main()
