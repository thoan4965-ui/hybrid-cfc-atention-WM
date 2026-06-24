# V0 — Bionic Hand 8-DOF (Real Robot)

## Overview
First prototype: 8-DOF bionic hand from SC09 servos, 3D-printed frame, USB camera. 
Purpose: compare AR vs CfC rollout drift on real hardware.

## Key Results
- CfC drift 0.000014/step = 34× lower than AR
- Scheduled Sampling (SS) improves CfC rollout 29×
- AR needs batch >128 to stabilize

## Hardware
- 8× SC09 servos (digital, 360° continuous)
- Webcam 480p (camera observation)
- RP2350 controller
- 3D-printed frame (DexHand-style)

## Papers Used
| Paper | Contribution |
|---|---|
| LeWM (Maes 2026) | JEPA architecture template |
| CfC (Hasani 2022) | ODE-RNN predictor |
| DexHand (Rob Knight) | Mechanical design ref |
| Scheduled Sampling (Bengio 2015) | Improved CfC rollout |

## Docs
- `docs/datasheets/sc09-servo-datasheet.md` — SC09 specs
- `docs/datasheets/scservo-sdk-api.md` — SCServo API
- `docs/datasheets/serial-protocol.md` — COM13 protocol
- `le-wm-vo/robot/README.md` — hardware setup guide
