# Serial Protocol — V0 Bionic Hand

## Connection

| Component | Port | Baudrate |
|---|---|---|
| PC → USB-UART Adapter | COM13 | 1Mbps |
| USB-UART → RP2350 (Pi Pico 2) | UART | 1Mbps |
| RP2350 → Servo bus | UART (SCS CL) | 1Mbps |

## Hardware chain

```
PC (COM13) ←→ USB-UART Adapter ←→ RP2350 (Pico 2) ←→ SC09 Servos (IDs 1,2,4,5,6,7,8,9)
```

## Power

- 3×18650 (11.1V) → Step-down (6V, 3A+) → Servo bus power via USB-UART adapter
- RP2350 powered via USB from PC

## Protocol

- SCS CL protocol (half-duplex UART)
- Packet format: `0xFF 0xFF ID LENGTH INSTRUCTION PARAM... CHECKSUM`
- Position range: 0-1023 (300° rotation)
- Feedback: position, load, voltage, temp, moving flag
