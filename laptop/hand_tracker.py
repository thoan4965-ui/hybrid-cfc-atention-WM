import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial
import time
import os

# Download model if not exists
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    import urllib.request
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# Cau hinh Serial
PORT = 'COM12' 
try:
    ser = serial.Serial(PORT, 115200, timeout=0.1)
    print(f"Connected to {PORT}")
except:
    print(f"Cannot connect to {PORT}")
    ser = None

# MediaPipe Tasks Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ha do phan giai
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    last_state = -1 
    frame_count = 0
    print("Hand Tracking Started. Press 'q' to quit.")
    
    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        
        frame_count += 1
        if frame_count % 2 != 0: # Bo qua 1 nua so khung hinh de giam tai
            cv2.imshow("Hand Tracker", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        # Dung timestamp mili giay
        timestamp = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp)

        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                fingers = []
                # Thumb: Check distance or simple x comparison (depends on left/right)
                # Toi dung logic: neu x dau ngon xa x khop ngon thi la mo
                if abs(hand_lms[4].x - hand_lms[0].x) > abs(hand_lms[3].x - hand_lms[0].x):
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 4 ngon con lai: tip.y < pip.y (y nhỏ là cao hơn)
                for tip_id in [8, 12, 16, 20]:
                    if hand_lms[tip_id].y < hand_lms[tip_id-2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers = sum(fingers)
                print(f"Seeing hand: {total_fingers} fingers up")
                
                current_state = 1 if total_fingers >= 4 else 0
                
                if current_state != last_state:
                    if ser:
                        ser.write(str(current_state).encode())
                    print(f"Fingers: {total_fingers} -> Send: {current_state}")
                    last_state = current_state

        # Doc phan hoi tu Mach (Echo)
        if ser and ser.in_waiting:
            feedback = ser.read(ser.in_waiting).decode().strip()
            print(f"MCU Response: {feedback}")

        cv2.imshow("Hand Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser: ser.close()
