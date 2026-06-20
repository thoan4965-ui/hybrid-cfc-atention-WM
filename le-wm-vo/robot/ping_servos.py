"""
Ping 8 servo SC09 — xác nhận phần cứng sống.
COM13, baudrate 1000000
"""

import sys
sys.path.append(r"d:\ai_training\tools\STServo_Python\STServo_Python\stservo-env")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS

PORT     = "COM13"
BAUDRATE = 1000000

# 8 servo IDs — bỏ ID3 (xoay ngón cái đã tháo)
SERVO_MAP = {
    1: "Cái   — TIP",
    2: "Cái   — Dạng",
    4: "Trỏ   — TIP",
    5: "Trỏ   — Dạng",
    6: "Trỏ   — Khép",
    7: "Giữa  — TIP",
    8: "Giữa  — Khép",
    9: "Giữa  — Dạng",
}

portHandler   = PortHandler(PORT)
packetHandler = sms_sts(portHandler)

if not portHandler.openPort():
    print(f"[ERR] Không mở được {PORT}")
    sys.exit(1)

if not portHandler.setBaudRate(BAUDRATE):
    print(f"[ERR] Không set baudrate {BAUDRATE}")
    portHandler.closePort()
    sys.exit(1)

print(f"\nPing servo trên {PORT} @ {BAUDRATE} baud\n{'─'*45}")

ok_count = 0
for sid, label in SERVO_MAP.items():
    model, result, error = packetHandler.ping(sid)
    if result == COMM_SUCCESS and error == 0:
        print(f"  [ID {sid:02d}] ✓  {label}  (model={model})")
        ok_count += 1
    else:
        err_str = packetHandler.getTxRxResult(result)
        print(f"  [ID {sid:02d}] ✗  {label}  ({err_str})")

print(f"{'─'*45}")
print(f"Kết quả: {ok_count}/{len(SERVO_MAP)} servo respond\n")

portHandler.closePort()
