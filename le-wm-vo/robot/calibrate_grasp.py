"""Calibrate grasp thresholds — run ONCE with bottle."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_servo import connect, disconnect, read_load, move_all

BASE = os.path.dirname(os.path.abspath(__file__))
neutral = json.load(open(os.path.join(BASE, "..", "data/calib", "calib_neutral.json")))["neutral_pos"]
grasp = json.load(open(os.path.join(BASE, "..", "data/calib", "calib_grasp.json")))["grasp_pos"]

GRASP_SERVOS = [2, 4, 8]  # Thumb-khép, Index-gập, Middle-khép

print("Connecting COM13...")
pH, pkt = connect("COM13")

print("\n=== Moving to NEUTRAL ===")
move_all(pkt, neutral)
time.sleep(1.5)

print("\n=== ĐẶT CHAI VÀO + BẤM ENTER ĐỂ NẮM ===")
input()

print("Moving to GRASP...")
move_all(pkt, grasp)
time.sleep(3.0)

print("\n=== GRASP LOADS (after 3s settle) ===")
thresholds = {}
for sid in GRASP_SERVOS:
    ld = read_load(pkt, sid) or 0
    thresholds[str(sid)] = ld
    print(f"  Servo {sid}: load = {ld}")

safe = {str(sid): int(thresholds[str(sid)] * 0.8) for sid in GRASP_SERVOS}
cfg = {
    "grasp_servos": GRASP_SERVOS,
    "thresholds": safe,
    "wait_seconds": 2.0,
    "min_blocked": 2,
    "note": f"Auto-calibrated S2(cái-khép), S4(trỏ-gập), S8(giữa-khép). Raw: {thresholds}. Thresholds = raw × 0.8."
}
with open(os.path.join(BASE, "load_threshold.json"), "w") as f:
    json.dump(cfg, f, indent=2)

print(f"\n  Raw loads: {thresholds}")
print(f"  Thresholds (×0.8): {safe}")
print(f"  ✓ Saved to load_threshold.json")

disconnect(pH)
