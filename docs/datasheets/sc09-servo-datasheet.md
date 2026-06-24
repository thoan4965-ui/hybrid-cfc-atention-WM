# SC09 Servo Datasheet (Waveshare)

## Specs

| Param | Value |
|---|---|
| Model | SC09 (SCS CL protocol) |
| Rotation | 300°, 0~1023 |
| Baudrate | 38400~1Mbps |
| No-load current | 150mA@6V |
| Locked-rotor current | **1.0A** |
| Feedback | Position, Speed, Load, Voltage, Temp, Moving |
| Datasheet | https://www.waveshare.com/wiki/SC09_Servo |
| SDK | `scservo_sdk` via `scscl` Python wrapper |

## Load behavior (empirical)

| State | Load value (addr 60-61) |
|---|---|
| Idle | 0 |
| Moving | 500-2024 (peak at start) |
| Blocked (grasp) | stays HIGH |

## Calibration (V0)

| Servo | Function | Neutral | Grasp | Range |
|---|---|---|---|---|
| 1 | Thumb — flex | 1013 | 550 | [550, 1013] |
| 2 | Thumb — adduct | 51 | 650 | [51, 650] |
| 4 | Index — flex | 220 | 400 | [220, 400] |
| 5 | Index — adduct | 327 | 180 | [180, 327] |
| 6 | Index — adduct | 896 | 1048 | [896, 1048] |
| 7 | Middle — flex | 801 | 622 | [622, 801] |
| 8 | Middle — adduct | 731 | 803 | [731, 803] |
| 9 | Middle — adduct | 735 | 624 | [624, 735] |

## Antagonistic structure (DO NOT MODIFY)

```
Thumb:  Servo 1 (flex) + Servo 2 (adduct)           → P1 + P2 ≈ const
Index:  Servo 4 (flex) + Servo 5+6 (adduct)          → P4 + P5 + P6 ≈ const
Middle: Servo 7 (flex) + Servo 8+9 (adduct)          → P7 + P8 + P9 ≈ const
```
