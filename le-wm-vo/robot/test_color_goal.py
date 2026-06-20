"""
Test màu chai + capture goal trong 1 session (ko restart camera).
Usage:
  .venv\Scripts\python.exe test_color_goal.py --checkpoint MODELS/cfc_models/vit_cfc_v4_finetune_v3/best_flat.ckpt
"""
import argparse, json, sys, time, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "le-wm"))
from module import TinyViT

SERVO_IDS = [1, 2, 4, 5, 6, 7, 8, 9]
BASE = Path(__file__).parent

def load_encoder(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    cleaned = {}
    for k, v in sd.items():
        nk = k[6:] if k.startswith("model.") else k
        cleaned[nk] = v
    enc = TinyViT(96, 8, 4, 64, 4, 256, 32).to(device)
    enc.load_state_dict({k.replace("encoder.", ""): v for k, v in cleaned.items()
                         if k.startswith("encoder.")}, strict=True)
    enc.eval()
    return enc

def encode(enc, img, device="cpu"):
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return enc(t).squeeze(0).cpu().numpy()

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def move_grasp(pkt, goal="grasp"):
    """Move all servos to calib position."""
    mode = "grasp" if goal == "grasp" else "neutral"
    calib_path = BASE.parent / "data/calib" / f"calib_{mode}.json"
    with open(calib_path) as f:
        data = json.load(f)
    pos = data[f"{mode}_pos"]
    from serial_servo import move_all
    move_all(pkt, pos)
    time.sleep(1.0)

def capture(camera_id=1):
    """Capture frame from camera (1 session, ko restart)."""
    import cv2, os as _os
    cam = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if _os.name == 'nt' else cv2.CAP_ANY)
    if not cam.isOpened():
        cam = cv2.VideoCapture(camera_id)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE, -6)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    for _ in range(5):
        cam.read()
    ret, frame = cam.read()
    cam.release()
    if not ret:
        raise RuntimeError("Camera capture failed")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Crop + resize (camera_config.json)
    CFG = Path(__file__).parent.parent / "data/config/camera_config.json"
    if CFG.exists():
        with open(CFG) as f:
            c = json.load(f)
        cx, cy, sz = c.get("crop_x", 264), c.get("crop_y", 8), c.get("crop_size", 364)
        frame = frame[cy:cy+sz, cx:cx+sz]
    frame = cv2.resize(frame, (96, 96))
    return frame.astype(np.float32) / 255.0

def save_image(img, path):
    import cv2 as _cv2
    img_uint8 = (img * 255).astype(np.uint8)
    img_bgr = _cv2.cvtColor(img_uint8, _cv2.COLOR_RGB2BGR)
    _cv2.imwrite(str(path), img_bgr)

def connect_serial():
    from serial_servo import connect
    pH, pkt = connect("COM13")
    return pH, pkt

def main():
    parser = argparse.ArgumentParser(description="Test màu chai + capture goal (1 session)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cam", type=int, default=1)
    args = parser.parse_args()

    print("Loading encoder...")
    enc = load_encoder(args.checkpoint, args.device)
    
    print("Connecting serial COM13...")
    pH, pkt = connect_serial()
    
    # Open camera 1 lần duy nhất + 20s ổn định
    import cv2, os as _os, time
    cam = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if _os.name == 'nt' else cv2.CAP_ANY)
    if not cam.isOpened():
        cam = cv2.VideoCapture(args.cam)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE, -6)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    print("  Camera stabilizing 30s...")
    time.sleep(30)
    for _ in range(5):
        cam.read()
    
    def capture_once():
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        CFG = Path(__file__).parent.parent / "data/config/camera_config.json"
        if CFG.exists():
            with open(CFG) as f:
                c = json.load(f)
            cx, cy, sz = c.get("crop_x", 264), c.get("crop_y", 8), c.get("crop_size", 364)
            frame = frame[cy:cy+sz, cx:cx+sz]
        frame = cv2.resize(frame, (96, 96))
        return frame.astype(np.float32) / 255.0

    print(f"\n{'='*60}")
    print("TEST MÀU CHAI + CAPTURE GOAL (1 camera session)")
    print(f"{'='*60}\n")
    
    # Step 1: Neutral → ENTER → Grasp + chai A → capture
    input("Step 1: Tay NEUTRAL. Đặt chai A. ENTER để GRASP + chụp...")
    move_grasp(pkt, "grasp")
    time.sleep(1.5)
    img_a = capture_once()
    lat_a = encode(enc, img_a, args.device)
    save_image(img_a, BASE / "b3_chai_a.png")
    print(f"  Chai A: norm={np.linalg.norm(lat_a):.4f}")
    
    # Step 2: Về neutral đổi chai
    input("Step 2: Tay về NEUTRAL. Đổi chai B. ENTER...")
    move_grasp(pkt, "neutral")
    time.sleep(1.0)
    
    # Step 3: Grasp + chai B → capture
    input("Step 3: ENTER để GRASP chai B + chụp...")
    move_grasp(pkt, "grasp")
    time.sleep(1.5)
    img_b = capture_once()
    lat_b = encode(enc, img_b, args.device)
    save_image(img_b, BASE / "b3_chai_b.png")
    print(f"  Chai B: norm={np.linalg.norm(lat_b):.4f}")
    
    mse_val = mse(lat_a, lat_b)
    print(f"\nMSE(chai_A, chai_B) = {mse_val:.6f}")
    if mse_val < 0.5:
        print("✅ Encoder ổn")
    else:
        print("⚠️ MSE cao")
    
    # Step 4: Goal
    goal_bottle = input("\nStep 4: Chai nào làm goal? (A/B, ENTER=giữ nguyên): ").strip().upper()
    if goal_bottle not in ('A', 'B'):
        goal_bottle = 'B'
    print(f"  Giữ tay GRASP với chai {goal_bottle}. ENTER để capture goal...")
    if goal_bottle == 'A':
        img_goal = img_a
        lat_goal = lat_a
    else:
        img_goal = capture_once()
        lat_goal = encode(enc, img_goal, args.device)
    np.save(BASE.parent / "data/goals/goal_v3.npy", lat_goal)
    save_image(img_goal, BASE.parent / "data/goals/goal_v3.png")
    print(f"  Goal saved: data/goals/goal_v3.npy (norm={np.linalg.norm(lat_goal):.4f})")
    
    # Step 5: Về neutral
    input("Step 5: ENTER về NEUTRAL + kết thúc...")
    move_grasp(pkt, "neutral")
    
    cam.release()
    from serial_servo import disconnect
    disconnect(pH)
    
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ TEST MÀU")
    print(f"  MSE(A, B) = {mse_val:.6f}")
    print(f"  Goal: data/goals/goal_v3.npy")
    print(f"\n  Chạy CEM:")
    print(f"  python robot_planner.py --model cfc --checkpoint {args.checkpoint} --goal data/goals/goal_v3 --serial COM13")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
