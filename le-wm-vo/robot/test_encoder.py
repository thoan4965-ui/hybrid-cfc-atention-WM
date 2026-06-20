"""
Encoder robustness test.
Capture 3 scenes (same grasp pose), encode, report pairwise MSE.
No robot execution, no thresholds.

Usage:
  python test_encoder.py --checkpoint path/to/v4.ckpt --model cfc
  python test_encoder.py --checkpoint path/to/v4.ckpt --model cfc --cam 0

Output:
  Pairwise MSE matrix + summary
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "le-wm"))
from module import TinyViT
from camera_goal import capture, save_image

IMG_SIZE = 96
LATENT_DIM = 32


def load_encoder(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    cleaned = {}
    for k, v in sd.items():
        nk = k[6:] if k.startswith("model.") else k
        cleaned[nk] = v

    enc = TinyViT(IMG_SIZE, patch_size=8, num_layers=4, hidden_dim=64,
                  num_heads=4, mlp_dim=256, output_dim=LATENT_DIM).to(device)
    enc.load_state_dict({k.replace("encoder.", ""): v for k, v in cleaned.items()
                         if k.startswith("encoder.")}, strict=True)
    enc.eval()
    return enc


def encode(enc, img, device="cpu"):
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        latent = enc(t).squeeze(0).cpu().numpy()
    return latent


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Encoder robustness test: 3 scenes, 1 grasp pose")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt")
    parser.add_argument("--model", type=str, default="cfc", choices=["cfc", "ar"])
    parser.add_argument("--cam", type=int, default=None, help="Default camera ID")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    enc = load_encoder(args.checkpoint, device)
    print(f"Encoder loaded: {sum(p.numel() for p in enc.parameters()):,} params")

    print(f"\n{'='*60}")
    print(f"ENCODER ROBUSTNESS TEST")
    print(f"1 grasp pose × 3 scenes")
    print(f"Step 1: Place hand in GRASP pose, then press ENTER for each scene")
    print(f"{'='*60}")

    scenes = [
        ("A_light_normal",  "Normal lighting, no miến"),
        ("B_light_bright",  "Phone light / lamp, no miến"),
        ("C_light_dark",    "Dim / shaded, no miến"),
        ("D_mien_normal",   "Lót miến, normal lighting"),
        ("E_mien_bright",   "Lót miến, bright light"),
    ]

    latents = {}
    for key, desc in scenes:
        cam_id = args.cam
        input(f"\n{key}: {desc}. Press ENTER to capture (cam {cam_id or 'default'})...")
        img = capture(camera_id=cam_id)
        latent = encode(enc, img, device)
        latents[key] = latent
        save_image(img, Path(f"test_enc_{key}.png"))
        print(f"  latent norm = {np.linalg.norm(latent):.4f}")

    print(f"\n{'='*60}")
    print(f"PAIRWISE MSE REPORT")
    print(f"{'='*60}")

    keys = list(latents.keys())
    mse_matrix = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            mse_matrix[i, j] = mse(latents[ki], latents[kj])

    header = f"{'':>18}" + "".join(f"{k:>18}" for k in keys)
    print(f"\n{header}")
    print("-" * (18 + 18 * len(keys)))
    for i, ki in enumerate(keys):
        row = f"{ki:>18}"
        for j in range(len(keys)):
            row += f"{mse_matrix[i, j]:>18.6f}"
        print(row)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    base = keys[0]
    for key in keys:
        if key == base: continue
        val = mse_matrix[keys.index(base), keys.index(key)]
        ratio = val / mse_matrix[keys.index(base), keys.index(base)] if mse_matrix[keys.index(base), keys.index(base)] > 0 else 1
        print(f"  {key} vs {base}: MSE = {val:.6f} (ratio = {ratio:.2f}x)")

    print(f"\nResults saved as test_enc_*.png")
    print(f"No conclusion — report numbers only.")


if __name__ == "__main__":
    main()
