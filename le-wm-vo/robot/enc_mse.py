"""
Encode 2 PNG images -> report MSE between their latents.
Usage:
  python enc_mse.py PNG1 PNG2 --checkpoint ../MODELS/cfc_models/vit_cfc_v4_epoch_30.ckpt
"""
import argparse, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "le-wm"))
from module import TinyViT
from camera_goal import read_image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("png1", type=str)
    p.add_argument("png2", type=str)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    enc = TinyViT(96, 8, 4, 64, 4, 256, 32).to(args.device)
    enc.load_state_dict({k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    enc.eval()

    for i, path in enumerate([args.png1, args.png2]):
        img = read_image(path)
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(args.device)
        with torch.no_grad():
            z = enc(t).squeeze(0).cpu().numpy()
        print(f"{i+1}. {path}: norm={np.linalg.norm(z):.4f}")

    z1 = read_and_encode(args.png1, enc, args.device)
    z2 = read_and_encode(args.png2, enc, args.device)
    mse = float(np.mean((z1 - z2)**2))
    print(f"\nMSE = {mse:.6f}")

def read_and_encode(path, enc, device):
    img = read_image(path)
    t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return enc(t).squeeze(0).cpu().numpy()

if __name__ == "__main__":
    main()
