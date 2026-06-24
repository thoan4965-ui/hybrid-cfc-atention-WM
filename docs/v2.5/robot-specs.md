# V2.5 — Robot Hardware Specs

## 4-DOF Arm Configuration

| Joint | Servo | Range | Link length |
|---|---|---|---|
| Base rotation | SG90 | 0-180° | — |
| Shoulder | MG90S (metal gear) | 0-180° | 10cm |
| Elbow | SG90 | 0-180° | 10cm |
| Wrist/gripper | SG90 | 0-180° | 5cm |

Total reach: ~25cm (shoulder+elbow+wrist)
Payload: ~50g (SG90 torque = 0.18 kg·cm at 4.8V)
Weight: ~200g total (frame + servos + controller)

## Servo Specs

| Spec | SG90 | MG90S |
|---|---|---|
| Torque (4.8V) | 0.18 kg·cm | 0.20 kg·cm |
| Speed (4.8V) | 0.12s/60° | 0.11s/60° |
| Operating voltage | 4.8-6.0V | 4.8-6.0V |
| Gear | Plastic | Metal |
| Weight | 9g | 13.4g |
| Dimensions | 23×12.2×29mm | 23×12.2×29mm |

## Sim → Real

| Aspect | Sim (MuJoCo PincherX-100) | Real (4-DOF build) | Gap |
|---|---|---|---|
| Joint angles | Continuous | PWM 0-180° | Linear calibration → map |
| Torque | Max torque | 0.18 kg·cm | Scale down payload |
| Control freq | 1 KHz | 50 Hz (PWM) | Downsample sim ctrl |
| Observation | Proprioception + camera | Same (USB camera) | ColorJitter + noise aug |

## Calibration

Each SG90 needs per-joint calibration:
```
pwm_angle = a * real_angle + b
```
Collect 3-5 points per servo, linear regression, store (a,b) per joint.
Apply in model: use calibrated angle for action embedding.

## 3D Print

- Frame: PLA, 0.2mm layer, 20% infill
- STL source: PincherX-100 design (modified for SG90 mounts)
- FABRI_CREATOR / ElectroT3D for printing (check local library)
- Total print time: ~12-15 hours
