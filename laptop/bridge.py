import serial
import time

# Cau hinh cong COM (User can thay doi cho dung)
PORT_GLOVE = 'COM3'
PORT_HAND = 'COM4'

try:
    ser_glove = serial.Serial(PORT_GLOVE, 115200, timeout=0.1)
    ser_hand = serial.Serial(PORT_HAND, 115200, timeout=0.1)
    print("Bridge connected.")
except Exception as e:
    print(f"Error: {e}")
    exit()

while True:
    if ser_glove.in_waiting:
        line = ser_glove.readline().decode().strip()
        if line.startswith("VAL:"):
            val = int(line.split(':')[1])
            # Thuat toan chuyen doi tu ADC sang goc o day
            angle = int(val / 65535 * 180)
            
            # Gui sang tay
            ser_hand.write(f"S:0:{angle}\n".encode())
    
    time.sleep(0.001)
