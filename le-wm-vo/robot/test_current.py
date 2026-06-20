"""Measure servo load: idle → moving → grasp (using JSON calib)."""
import sys, os, time, json
sys.path.insert(0, r"d:\ai_training\code-new")
from serial_servo import connect, disconnect, read_all_loads, move_all, PORT

BASE = os.path.dirname(os.path.abspath(__file__))

def load_json(path):
    with open(path) as f:
        return json.load(f)

neutral = load_json(os.path.join(BASE, "..", "data/calib", "calib_neutral.json"))["neutral_pos"]
grasp = load_json(os.path.join(BASE, "..", "data/calib", "calib_grasp.json"))["grasp_pos"]

print(f"Connecting {PORT}...")
pH, pkt = connect(PORT)

print("\n=== STATE 1: Idle (neutral, no load) ===")
move_all(pkt, neutral)
time.sleep(1.5)
loads1 = read_all_loads(pkt)
for sid, l in loads1.items():
    print(f"  Servo {sid}: load={l}")
max_idle = max(loads1.values())

print("\n=== STATE 2: Moving (neutral→grasp) ===")
move_all(pkt, grasp)
time.sleep(0.3)
loads2 = read_all_loads(pkt)
for sid, l in loads2.items():
    print(f"  Servo {sid}: load={l}")
max_moving = max(loads2.values())

print("\n=== STATE 3: GRASPING BOTTLE ===")
print("  PUT BOTTLE IN HAND, then press Enter...")
input()
move_all(pkt, grasp)
time.sleep(1.0)
loads3 = read_all_loads(pkt)
for sid, l in loads3.items():
    print(f"  Servo {sid}: load={l}")
max_grasp = max(loads3.values())

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"  Max Idle:   {max_idle}")
print(f"  Max Moving: {max_moving}")
print(f"  Max Grasp:  {max_grasp}")
threshold = max(max_moving * 2, 500)
print(f"\n  Suggested threshold: {threshold} (2× max_moving, min 500)")
print(f"  Grasp margin: {max_grasp - threshold}")

cfg = {"threshold": threshold, "max_idle": max_idle, "max_moving": max_moving, "max_grasp": max_grasp}
with open(os.path.join(BASE, "load_threshold.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print(f"\n  ✓ Saved to load_threshold.json")

disconnect(pH)
