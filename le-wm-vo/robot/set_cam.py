"""
Script căn chỉnh góc Camera thời gian thực (set_cam.py).
Tính năng:
1. Cho phép di chuyển ô vuông crop bằng phím mũi tên (Lên, Xuống, Trái, Phải).
2. Cho phép phóng to/thu nhỏ ô vuông crop bằng phím 'i' (phóng to) và 'o' (thu nhỏ).
3. Nhấn 's' để lưu tọa độ cố định vào camera_config.json.
4. Tự động load lại cấu hình cũ nếu có để đảm bảo vị trí không đổi.
"""
import time 
import cv2
import sys
import json
import os
import argparse

CONFIG_PATH = r"d:\ai_training\data\config\camera_config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Cấu hình mặc định (khớp camera_config.json hiện tại)
    return {
        "camera_idx": 1,
        "crop_x": 264,
        "crop_y": 8,
        "crop_size": 364,
        "img_size": 96,
        "warmup_frames": 5,
        "line_spacing_ratio": 0.86
    }

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Đã lưu cấu hình camera cố định tại: {CONFIG_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Cong cu can chinh camera va thiet lap o vuong crop")
    parser.add_argument("--cam", type=int, default=None, help="Index cua camera")
    args = parser.parse_args()

    config = load_config()
    
    # Nếu người dùng truyền tham số --cam thì ưu tiên ghi đè vào config
    if args.cam is not None:
        config["camera_idx"] = args.cam

    camera_idx = config["camera_idx"]
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] Khong mo duoc camera index {camera_idx} bang DSHOW. Thu mo mac dinh...")
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"[ERR] Khong the ket noi voi camera index {camera_idx}!")
            sys.exit(1)

    # Camera stabilization: 20s để exposure/WB ổn định
    print("  Camera stabilizing 30s...")
    time.sleep(30)
    for _ in range(5):
        cap.read()

    # Đọc thử 1 frame để lấy kích thước thật của camera
    ret, frame = cap.read()
    if not ret:
        print("[ERR] Khong doc duoc frame tu camera!")
        cap.release()
        sys.exit(1)
    
    img_h, img_w, _ = frame.shape

    # Đảm bảo tọa độ trong config không vượt quá kích thước ảnh
    crop_size = min(config.get("crop_size", 224), min(img_w, img_h))
    crop_x = max(0, min(config.get("crop_x", int((img_w - crop_size)/2)), img_w - crop_size))
    crop_y = max(0, min(config.get("crop_y", int((img_h - crop_size)/2)), img_h - crop_size))
    
    # Tải tỷ lệ khoảng cách đường song song (mặc định 60% của crop box)
    line_spacing_ratio = config.get("line_spacing_ratio", 0.6)

    print("\n" + "="*60)
    print(" HƯỚNG DẪN ĐIỀU CHỈNH CĂN CHỈNH CAMERA & ROBOT")
    print(f" Camera đang mở: Index {camera_idx} (Kích thước gốc: {img_w}x{img_h})")
    print("-" * 60)
    print("  [Mũi tên hoặc W/A/S/D]              : Di chuyển ô vuông crop")
    print("  [Phím i]                            : THU NHỎ vùng nhìn (Tăng crop_size)")
    print("  [Phím o]                            : PHÓNG TO vùng nhìn (Giảm crop_size)")
    print("  [Phím [ ]                           : THU HẸP khoảng cách 2 đường song song")
    print("  [Phím ] ]                           : PHÌNH RỘNG khoảng cách 2 đường song song")
    print("  [Phím Space hoặc Enter]             : LƯU cấu hình cố định & THOÁT")
    print("  [Phím q hoặc ESC]                   : Thoát không lưu")
    print("="*60 + "\n")

    cv2.namedWindow("Live Preview - Thiet Lap Crop Zone", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Mat ket noi voi camera!")
            break

        # Sao chép frame để vẽ hướng dẫn mà không ghi đè lên ảnh gốc
        display_frame = frame.copy()

        # Vẽ ô vuông crop màu xanh lá (Green)
        cv2.rectangle(display_frame, (crop_x, crop_y), (crop_x + crop_size, crop_y + crop_size), (0, 255, 0), 2)
        
        # Tính toán tọa độ tâm và các đường căn chỉnh song song
        center_x = crop_x + crop_size // 2
        center_y = crop_y + crop_size // 2
        half_spacing = int((crop_size * line_spacing_ratio) / 2)

        # 1. Vẽ HAI ĐƯỜNG DỌC SONG SONG (Cyan) - Căn biên trái/phải robot
        v_left_x = center_x - half_spacing
        v_right_x = center_x + half_spacing
        cv2.line(display_frame, (v_left_x, crop_y), (v_left_x, crop_y + crop_size), (255, 255, 0), 2)
        cv2.line(display_frame, (v_right_x, crop_y), (v_right_x, crop_y + crop_size), (255, 255, 0), 2)

        # 2. Vẽ HAI ĐƯỜNG NGANG SONG SONG (Magenta) - Căn biên trên/dưới robot
        h_top_y = center_y - half_spacing
        h_bottom_y = center_y + half_spacing
        cv2.line(display_frame, (crop_x, h_top_y), (crop_x + crop_size, h_top_y), (255, 0, 255), 2)
        cv2.line(display_frame, (crop_x, h_bottom_y), (crop_x + crop_size, h_bottom_y), (255, 0, 255), 2)

        # 3. Vẽ Tâm ngắm (+) màu đỏ ở chính giữa
        cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
        cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)

        # 4. Vẽ hai đường chéo X mỏng để căn góc nghiêng
        cv2.line(display_frame, (crop_x, crop_y), (crop_x + crop_size, crop_y + crop_size), (0, 0, 150), 1)
        cv2.line(display_frame, (crop_x + crop_size, crop_y), (crop_x, crop_y + crop_size), (0, 0, 150), 1)
        
        # Vẽ văn bản thông tin lên màn hình preview
        cv2.putText(display_frame, f"CROP ZONE: {crop_size}x{crop_size} (X:{crop_x}, Y:{crop_y})", 
                    (crop_x + 5, crop_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Robot Bounds: {half_spacing * 2}px ({line_spacing_ratio * 100:.0f}%)", 
                    (crop_x + 5, crop_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        cv2.putText(display_frame, "Keys: [ / ] to scale Robot Bounds | i / o to Zoom Crop | Space/Enter to Save", 
                    (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Live Preview - Thiet Lap Crop Zone", display_frame)

        # Đọc phím nhấn
        key = cv2.waitKey(30)
        
        if key == ord('q') or key == 27: # q hoac ESC
            break
        elif key == 32 or key == 13 or key == ord('\r') or key == ord('\n'): # Space or Enter to save
            config["crop_x"] = crop_x
            config["crop_y"] = crop_y
            config["crop_size"] = crop_size
            config["camera_idx"] = camera_idx
            config["line_spacing_ratio"] = round(line_spacing_ratio, 3)
            # Preserve img_size and warmup_frames
            config.setdefault("img_size", 96)
            config.setdefault("warmup_frames", 5)
            save_config(config)
            break
        
        # Zoom Crop Box
        elif key == ord('i'): # Phóng to ô vuông (Nhìn rộng hơn)
            if crop_size + 10 <= min(img_w, img_h) and crop_x + crop_size + 10 <= img_w and crop_y + crop_size + 10 <= img_h:
                crop_size += 10
        elif key == ord('o'): # Thu nhỏ ô vuông (Tập trung hơn)
            if crop_size - 10 >= 100:
                crop_size -= 10
                
        # Căn chỉnh độ rộng đường song song (brackets)
        elif key == ord('['): # Thu hẹp tỷ lệ đường song song
            if line_spacing_ratio - 0.02 >= 0.1:
                line_spacing_ratio -= 0.02
        elif key == ord(']'): # Phình rộng tỷ lệ đường song song
            if line_spacing_ratio + 0.02 <= 0.9:
                line_spacing_ratio += 0.02
                
        # Di chuyển ô vuông (Phím mũi tên Windows hoặc WASD)
        elif key == 2490368 or key == ord('w'): # Up
            if crop_y - 8 >= 0:
                crop_y -= 8
        elif key == 2621440 or key == ord('s'): # Down
            if crop_y + crop_size + 8 <= img_h:
                crop_y += 8
        elif key == 2424832 or key == ord('a'): # Left
            if crop_x - 8 >= 0:
                crop_x -= 8
        elif key == 2555904 or key == ord('d'): # Right
            if crop_x + crop_size + 8 <= img_w:
                crop_x += 8
        
        # Bổ sung xử lý phím mũi tên tiêu chuẩn cho một số nền tảng khác
        elif key == 0 or key == 0xE0: # Special keys
            key2 = cv2.waitKey(1)
            if key2 == 72: # Up
                if crop_y - 8 >= 0: crop_y -= 8
            elif key2 == 80: # Down
                if crop_y + crop_size + 8 <= img_h: crop_y += 8
            elif key2 == 75: # Left
                if crop_x - 8 >= 0: crop_x -= 8
            elif key2 == 77: # Right
                if crop_x + crop_size + 8 <= img_w: crop_x += 8

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Đã tắt live preview.")

if __name__ == "__main__":
    main()
