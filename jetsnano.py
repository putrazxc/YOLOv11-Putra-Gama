import cv2
import time
import numpy as np
from ultralytics import YOLO

# Inisialisasi model dan video
model = YOLO("modelbest.pt")
cap = cv2.VideoCapture("rtsp://admin:Uptti212!@10.155.155.162:554/Streaming/Channels/102")

# Konstanta
REAL_DISTANCE = 2.5  # jarak real dalam meter

# Pengaturan frame
target_width = 640  # Dikurangi untuk performa Jetson Nano
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_height = int(target_width * original_height / original_width)

# Tracking variables
prev_frame_time = 0
track_memory = {}
object_id = 0
total_motor = 0
total_mobil = 0

# Mask setup
mask = None
mask_setup_done = False

def create_mask_from_image(image_path):
    """Buat mask dari file gambar"""
    global mask
    try:
        mask_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            mask_img = cv2.resize(mask_img, (target_width, target_height))
            _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            return True
        return False
    except:
        return False

def create_polygon_mask():
    """Buat mask polygon default"""
    global mask
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    
    points = np.array([
        [50, target_height],
        [target_width-50, target_height],
        [target_width-100, 100],
        [100, 100]
    ], np.int32)
    
    cv2.fillPoly(mask, [points], 255)
    return mask

def get_mask_bounds():
    """Dapatkan batas mask untuk penentuan garis"""
    if mask is None:
        return 100, target_height
    
    top_y = None
    for y in range(mask.shape[0]):
        if np.any(mask[y, :] > 0):
            top_y = y
            break
    
    bottom_y = None
    for y in range(mask.shape[0] - 1, -1, -1):
        if np.any(mask[y, :] > 0):
            bottom_y = y
            break
    
    return top_y if top_y is not None else 100, bottom_y if bottom_y is not None else target_height

def setup_mask():
    """Setup mask dari file atau buat default"""
    global mask_setup_done
    if create_mask_from_image("mask.png") or create_mask_from_image("mask.jpg"):
        print("Mask berhasil dimuat dari file")
    else:
        create_polygon_mask()
        print("Mask polygon default dibuat")
    
    mask_setup_done = True

def is_in_mask(x, y):
    """Cek apakah titik berada dalam mask"""
    if mask is None:
        return True
    
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] > 0
    return False

def apply_mask_to_detections(boxes):
    """Filter deteksi berdasarkan mask"""
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if is_in_mask(cx, cy):
            filtered_boxes.append(box)
    
    return filtered_boxes

def draw_lines_and_mask(frame, line1_y, line2_y):
    """Gambar garis deteksi dan mask overlay"""
    # Gambar mask overlay
    if mask is not None:
        mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_overlay[:, :, 1] = 0  # Hilangkan channel hijau
        mask_overlay[:, :, 2] = 0  # Hilangkan channel biru
        frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)
    
    # Gambar garis deteksi
    cv2.line(frame, (0, line1_y), (target_width, line1_y), (0, 255, 0), 2)
    cv2.line(frame, (0, line2_y), (target_width, line2_y), (0, 0, 255), 2)
    
    # Label garis
    cv2.putText(frame, "Start Line", (10, line1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "End Line", (10, line2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def process_detections(frame, filtered_boxes, results, line1_y, line2_y, current_time):
    """Proses deteksi dan tracking"""
    global object_id, total_motor, total_mobil, track_memory
    
    current_boxes = []
    
    for box in filtered_boxes:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        class_name = results.names[int(cls)]
        
        # Warna berdasarkan kelas
        color = (0, 255, 255) if class_name == "motor" else (255, 0, 0)
        
        # Gambar bounding box dan centroid
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Tracking
        matched_id = None
        for tid, data in track_memory.items():
            prev_cx, prev_cy = data["centroid"]
            if abs(prev_cx - cx) < 50 and abs(prev_cy - cy) < 50:
                matched_id = tid
                break
        
        if matched_id is None:
            matched_id = object_id
            object_id += 1
        
        current_boxes.append(matched_id)
        track = track_memory.get(matched_id)
        
        if not track:
            track_memory[matched_id] = {
                "centroid": (cx, cy),
                "t1": None,
                "t2": None,
                "sent": False,
                "class": class_name,
                "speed": None,
                "last_seen": current_time
            }
            track = track_memory[matched_id]
        else:
            track["centroid"] = (cx, cy)
            track["last_seen"] = current_time
        
        # Deteksi crossing garis
        if line1_y - 10 < cy < line1_y + 10 and track["t1"] is None and is_in_mask(cx, cy):
            track["t1"] = time.time()
            print(f"Object {matched_id} ({class_name}) crossed start line")
        
        if line2_y - 10 < cy < line2_y + 10 and track["t2"] is None and is_in_mask(cx, cy):
            track["t2"] = time.time()
            print(f"Object {matched_id} ({class_name}) crossed end line")
        
        # Hitung kecepatan
        if track["t1"] and track["t2"] and not track["sent"]:
            time_diff = track["t2"] - track["t1"]
            if time_diff > 0:
                speed = round((REAL_DISTANCE / time_diff) * 3.6, 2)
                track["speed"] = speed
                track["sent"] = True
                
                # Update counter
                class_lower = track["class"].lower()
                if class_lower in ["mobil", "car"]:
                    total_mobil += 1
                elif class_lower in ["motor", "motorbike"]:
                    total_motor += 1
                
                print(f"KECEPATAN TERDETEKSI: {class_name} - {speed} km/h")
                print(f"Total Motor: {total_motor}, Total Mobil: {total_mobil}")
        
        # Tampilkan kecepatan di frame
        if track["speed"] is not None:
            cv2.putText(frame, f"{track['speed']} km/h", (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Bersihkan tracking lama
    for tid in list(track_memory):
        if tid not in current_boxes:
            if current_time - track_memory[tid]["last_seen"] > 3:
                del track_memory[tid]
    
    return frame

def print_statistics():
    """Print statistik detail"""
    print(f"\n=== STATISTIK ===")
    print(f"Total Motor: {total_motor}")
    print(f"Total Mobil: {total_mobil}")
    print(f"Objects being tracked: {len(track_memory)}")
    print("================\n")

def reset_counters():
    """Reset counter dan tracking"""
    global total_motor, total_mobil, track_memory
    total_motor = 0
    total_mobil = 0
    track_memory.clear()
    print("Counter dan tracking direset")

def main():
    """Fungsi utama untuk menjalankan deteksi kecepatan"""
    global prev_frame_time, mask_setup_done
    
    print("Memulai deteksi kecepatan...")
    print(f"Resolusi frame: {target_width}x{target_height}")
    print("=== KONTROL ===")
    print("- 'q': Keluar")
    print("- 'r': Reset counter")
    print("- 'p': Print statistik")
    print("===============")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video selesai, mengulang...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Setup mask jika belum
            if not mask_setup_done:
                setup_mask()
            
            # Dapatkan batas mask dan tentukan garis
            mask_top, mask_bottom = get_mask_bounds()
            mask_height = mask_bottom - mask_top
            
            line1_y = mask_top + 30
            line_distance = int(mask_height * 0.20)
            line2_y = line1_y + line_distance
            
            if line2_y > mask_bottom - 10:
                line2_y = mask_bottom - 10
                line1_y = line2_y - line_distance
            
            # Hitung FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
            prev_frame_time = current_time
            
            # YOLO inference
            results = model(frame)[0]
            
            # Filter deteksi dengan mask
            filtered_boxes = apply_mask_to_detections(results.boxes.data)
            
            # Gambar garis dan mask
            frame = draw_lines_and_mask(frame, line1_y, line2_y)
            
            # Proses deteksi dan tracking
            frame = process_detections(frame, filtered_boxes, results, 
                                     line1_y, line2_y, current_time)
            
            # Tampilkan info
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Motor: {total_motor}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Mobil: {total_mobil}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Tampilkan frame
            cv2.imshow("Speed Detection", frame)
            
            # Kontrol keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                reset_counters()
            elif key == ord('p'):
                print_statistics()
    
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup selesai")

# Jalankan program
if __name__ == "__main__":
    main()