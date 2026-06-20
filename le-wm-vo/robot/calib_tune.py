"""Calibrate servo positions.
Usage:
  calib_tune.py neutral          → release all, adjust by hand, lock all
  calib_tune.py neutral --set    → edit by typing numbers
  calib_tune.py grasp            → release all, adjust by hand
  calib_tune.py grasp --set      → edit by typing numbers
  calib_tune.py neutral --show   → show current values only
"""
import sys, os, time, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_servo import connect, disconnect, release_torque, enable_torque, read_position, move_servo

SERVO_IDS = [1, 2, 4, 5, 6, 7, 8, 9]
BASE = os.path.dirname(os.path.abspath(__file__))

def show_positions(calib, calib_key):
    print(f"\n  Current {calib_key}:")
    data = calib[calib_key]
    print(f"    Thumb:  S1={data['1']:>4}  S2={data['2']:>4}")
    print(f"    Index:  S4={data['4']:>4}  S5={data['5']:>4}  S6={data['6']:>4}")
    print(f"    Middle: S7={data['7']:>4}  S8={data['8']:>4}  S9={data['9']:>4}")

def mode_hand(calib, calib_key, mode):
    """Release all, adjust by hand, lock all."""
    pH, pkt = connect("COM13")
    
    print("\n  → Releasing ALL servos...")
    for sid in SERVO_IDS:
        release_torque(pkt, sid)
    print("  ✓ Free — adjust by hand")
    
    input("\n  → Press ENTER to LOCK & SAVE...")
    
    new_positions = {}
    for sid in SERVO_IDS:
        pos = read_position(pkt, sid)
        enable_torque(pkt, sid)
        new_positions[str(sid)] = int(pos) if pos is not None else calib[calib_key].get(str(sid), 500)
    
    calib[calib_key] = new_positions
    calib["note"] = calib.get("note", "") + f" | hand-tuned {time.strftime('%m-%d')}"
    
    calib_path = os.path.join(BASE, "..", "data/calib", f"calib_{mode}.json")
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    
    show_positions(calib, calib_key)
    print(f"\n  ✓ Saved to {calib_path}")
    disconnect(pH)

def mode_set(calib, calib_key, mode):
    """Edit by typing numbers, then move servo to verify."""
    print(f"\n  === EDIT {calib_key.upper()} ===")
    
    # Read actual current positions from robot (not from old JSON)
    pH, pkt = connect("COM13")
    for sid in SERVO_IDS:
        pos = read_position(pkt, sid)
        if pos is not None and pos > 0:
            calib[calib_key][str(sid)] = int(pos)
    
    show_positions(calib, calib_key)
    print("  (values read from actual servo positions)")
    
    while True:
        print(f"\n  Enter: <servo_id>=<value> (e.g. 2=650) to set")
        print("         'go' to test on robot, 'save' to save, 'q' to quit")
        cmd = input("  > ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'save':
            calib["note"] = calib.get("note", "") + f" | numeric-tuned {time.strftime('%m-%d')}"
            calib_path = os.path.join(BASE, "..", "data/calib", f"calib_{mode}.json")
            with open(calib_path, "w") as f:
                json.dump(calib, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to {calib_path}")
            break
        elif cmd == 'go':
            print(f"  Moving servos to current {calib_key}...")
            for sid in SERVO_IDS:
                val = int(calib[calib_key][str(sid)])
                move_servo(pkt, sid, val)
            time.sleep(0.5)
            print(f"  ✓ Done — check robot")
        elif '=' in cmd:
            try:
                sid_str, val_str = cmd.split('=')
                sid = int(sid_str.strip())
                val = int(val_str.strip())
                if sid not in SERVO_IDS:
                    print(f"  ⚠️ Invalid servo: {sid}. Valid: {SERVO_IDS}")
                    continue
                if val < 0 or val > 1023:
                    print(f"  ⚠️ Value {val} out of range [0, 1023]")
                    continue
                calib[calib_key][str(sid)] = val
                move_servo(pkt, sid, val)  # move immediately for feedback
                print(f"  ✓ S{sid} = {val}  (moved)")
                time.sleep(0.3)
                actual = read_position(pkt, sid)
                print(f"    Actual position: {actual}")
                show_positions(calib, calib_key)
            except ValueError:
                print(f"  ⚠️ Format: 2=650")
        else:
            print(f"  ⚠️ Unknown. Use: 2=650 / go / save / q")
    
    disconnect(pH)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("neutral", "grasp"):
        print("Usage: calib_tune.py neutral|grasp [--set|--show]")
        sys.exit(1)
    
    mode = sys.argv[1]
    calib_path = os.path.join(BASE, "..", "data/calib", f"calib_{mode}.json")
    
    if not os.path.exists(calib_path):
        calib = {f"{mode}_pos": {str(sid): 500 for sid in SERVO_IDS}}
    else:
        calib = json.load(open(calib_path))
    
    calib_key = f"{mode}_pos"
    
    if len(sys.argv) > 2 and sys.argv[2] == "--set":
        mode_set(calib, calib_key, mode)
    elif len(sys.argv) > 2 and sys.argv[2] == "--show":
        show_positions(calib, calib_key)
    else:
        mode_hand(calib, calib_key, mode)
