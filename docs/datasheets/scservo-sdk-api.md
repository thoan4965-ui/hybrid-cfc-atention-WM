# SCServo SDK — Python API

## Install

```bash
pip install scservo-sdk
# Hoặc dùng wrapper: pip install scscl
```

## Core API (scservo_sdk)

```python
from scservo_sdk import *

# Initialize port
port = PortHandler('/dev/ttyUSB0')  # hoặc 'COM13' trên Windows
port.open_port()
port.set_baudrate(1000000)  # 1Mbps

# Initialize packet handler
packet = PacketHandler(1.0)  # 1.0 = protocol version

# Write position
SCS_GOAL_POSITION = 42  # address cho goal position
packet.write2ByteTxRx(port, servo_id, SCS_GOAL_POSITION, target_pos)

# Read position
SCS_PRESENT_POSITION = 56
pos, result, error = packet.read2ByteTxRx(port, servo_id, SCS_PRESENT_POSITION)

# Read load
SCS_PRESENT_LOAD = 60
load, result, error = packet.read2ByteTxRx(port, servo_id, SCS_PRESENT_LOAD)
```

## Wrapper (scscl)

```python
# TODO: add if used
```
