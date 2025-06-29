from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np
import threading
import requests
from ultralytics import YOLO
import mysql.connector
from datetime import datetime, date

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Database connection
db = mysql.connector.connect(host="localhost", user="root", password="asdedorio123", database="haha", port=3306)
cursor = db.cursor()

# Initialize components
model = YOLO("best.pt")
cap = cv2.VideoCapture("Video.mp4")
REAL_DISTANCE = 2 
ESP8266_IP = "http://172.20.10.3/" 

# Video settings
target_width = 640
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_height = int(target_width * original_height / original_width)

# Global variables
prev_frame_time = 0
track_memory = {}
object_id = 0
total_motor = 0
total_mobil = 0
mask = None
mask_setup_done = False

class ClassificationSmoother:
    def __init__(self):
        self.class_history = {}
        self.history_size = 3
    
    def smooth_classification(self, track_id, detected_class, confidence):
        if track_id not in self.class_history:
            self.class_history[track_id] = []
        
        self.class_history[track_id].append((detected_class, confidence))
        if len(self.class_history[track_id]) > self.history_size:
            self.class_history[track_id].pop(0)
        
        class_scores = {}
        for cls, conf in self.class_history[track_id]:
            class_scores.setdefault(cls, []).append(conf)
        
        best_class, best_score = detected_class, 0
        for cls, scores in class_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score, best_class = avg_score, cls
        
        return best_class
    
    def cleanup_track(self, track_id):
        self.class_history.pop(track_id, None)

classification_smoother = ClassificationSmoother()

def setup_mask():
    global mask, mask_setup_done
    try:
        for ext in ["png", "jpg"]:
            mask_img = cv2.imread(f"mask.{ext}", cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask_img = cv2.resize(mask_img, (target_width, target_height))
                _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                print(f"Mask loaded from mask.{ext}")
                mask_setup_done = True
                return
    except:
        pass
    
    # Create default polygon mask
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    points = np.array([[100, target_height], [target_width-100, target_height], 
                      [target_width-200, 200], [200, 200]], np.int32)
    cv2.fillPoly(mask, [points], 255)
    print("Default polygon mask created")
    mask_setup_done = True

def get_mask_bounds():
    if mask is None:
        return 200, target_height
    
    top_y = next((y for y in range(mask.shape[0]) if np.any(mask[y, :] > 0)), 200)
    bottom_y = next((y for y in range(mask.shape[0] - 1, -1, -1) if np.any(mask[y, :] > 0)), target_height)
    return top_y, bottom_y

def is_in_mask(x, y):
    return mask is None or (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0)

def apply_mask_to_detections(boxes, fps):
    conf_threshold = 0.3 if fps < 10 else (0.4 if fps < 20 else 0.5)
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_in_mask(cx, cy):
                filtered_boxes.append(box)
    
    return filtered_boxes

def send_to_esp8266(speed):
    try:
        response = requests.get(f"{ESP8266_IP}:80/update", params={"kecepatan": int(speed)}, timeout=1)
        print(f"ESP8266 {'success' if response.status_code == 200 else 'failed'}: {speed} km/h")
    except Exception as e:
        print(f"ESP8266 error: {e}")

def generate_frames():
    global prev_frame_time, object_id, total_motor, total_mobil, mask_setup_done
    offset = 7
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (target_width, target_height))
        
        if not mask_setup_done:
            setup_mask()
        
        mask_top, mask_bottom = get_mask_bounds()
        mask_height = mask_bottom - mask_top
        line1_y = mask_top + 10
        line_distance = int(mask_height * 0.40)
        line2_y = min(line1_y + line_distance, mask_bottom - 10)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_time

        results = model(frame)[0]
        filtered_boxes = apply_mask_to_detections(results.boxes.data, fps)

        if mask is not None:
            mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_overlay[:, :, 0] = 0
            mask_overlay[:, :, 1] = 0
            mask_overlay[:, :, 2] = 0
            frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)

        current_boxes = []
        base_tolerance = 50
        fps_factor = max(1, 30 / max(fps, 1))
        dynamic_tolerance = int(base_tolerance * fps_factor)
        line_tolerance = int(9 * fps_factor)

        for box in filtered_boxes:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = results.names[int(cls)]

            # Find matching track
            matched_id = None
            min_distance = float('inf')
            for tid, data in track_memory.items():
                prev_cx, prev_cy = data["centroid"]
                distance = ((prev_cx - cx) ** 2 + (prev_cy - cy) ** 2) ** 0.5
                if distance < dynamic_tolerance and distance < min_distance:
                    matched_id, min_distance = tid, distance

            if matched_id is None:
                matched_id = object_id
                object_id += 1

            current_boxes.append(matched_id)
            
            # Update or create track
            if matched_id not in track_memory:
                track_memory[matched_id] = {
                    "centroid": (cx, cy), "t1": None, "t2": None, "sent": False,
                    "class": class_name, "speed": None, "last_seen": current_time, "confidence": conf
                }
            else:
                track = track_memory[matched_id]
                track.update({
                    "centroid": (cx, cy), "last_seen": current_time,
                    "class": classification_smoother.smooth_classification(matched_id, class_name, conf)
                })

            track = track_memory[matched_id]

            # Line crossing detection
            if (line1_y - line_tolerance < cy < line1_y + line_tolerance and 
                track["t1"] is None and is_in_mask(cx, cy)):
                track["t1"] = time.time()
                print(f"Object {matched_id} ({track['class']}) crossed line 1")

            if (line2_y - line_tolerance < cy < line2_y + line_tolerance and 
                track["t2"] is None and is_in_mask(cx, cy)):
                track["t2"] = time.time()
                print(f"Object {matched_id} ({track['class']}) crossed line 2")

            # Speed calculation and database insertion
            if track["t1"] and track["t2"] and not track["sent"]:
                time_diff = track["t2"] - track["t1"]
                if time_diff > 0:
                    speed = round((REAL_DISTANCE / time_diff) * 3.6 + offset, 2)
                    track["speed"] = speed
                    track["sent"] = True

                    class_lower = track["class"].lower()
                    if class_lower in ["mobil", "car"]:
                        total_mobil += 1
                    elif class_lower in ["motor", "motorbike"]:
                        total_motor += 1

                    socketio.emit("data_info", {
                        "kecepatan": speed, "fps": int(fps),
                        "jumlah_mobil": total_mobil, "jumlah_motor": total_motor
                    })

                    threading.Thread(target=send_to_esp8266, args=(speed,)).start()

                    # Database operations
                    try:
                        now = datetime.now()
                        cursor.execute("INSERT INTO kendaraan (jenis, kecepatan, waktu) VALUES (%s, %s, %s)",
                                     (track["class"], speed, now))
                        db.commit()

                        today = date.today()
                        if class_lower in ["mobil", "car"]:
                            update_query = "INSERT INTO rekap_harian (tanggal, jumlah_mobil, jumlah_motor) VALUES (%s, 1, 0) ON DUPLICATE KEY UPDATE jumlah_mobil = jumlah_mobil + 1"
                        elif class_lower in ["motor", "motorbike"]:
                            update_query = "INSERT INTO rekap_harian (tanggal, jumlah_mobil, jumlah_motor) VALUES (%s, 0, 1) ON DUPLICATE KEY UPDATE jumlah_motor = jumlah_motor + 1"
                        else:
                            update_query = None

                        if update_query:
                            cursor.execute(update_query, (today,))
                            db.commit()
                            
                        print(f"Speed recorded: {speed} km/h for {track['class']}")
                    except Exception as e:
                        print(f"Database error: {e}")

            # Visualization
            color = (0, 255, 255) if track["class"].lower() in ["motor", "motorbike"] else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            if track["speed"] is not None:
                cv2.putText(frame, f"{track['speed']} km/h", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Cleanup old tracks
        cleanup_time = 5 if fps < 10 else 3
        for tid in list(track_memory):
            if tid not in current_boxes and current_time - track_memory[tid]["last_seen"] > cleanup_time:
                classification_smoother.cleanup_track(tid)
                del track_memory[tid]

        _, buffer = cv2.imencode(".jpg", frame)
        frame_encoded = base64.b64encode(buffer).decode("utf-8")
        socketio.emit("video_frame", frame_encoded)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ind.html")
def ind():
    try:
        cursor.execute("SELECT * FROM kendaraan ORDER BY waktu DESC LIMIT 10")
        kendaraan_data = cursor.fetchall()
        cursor.execute("SELECT * FROM rekap_harian ORDER BY tanggal DESC LIMIT 10")
        rekap_data = cursor.fetchall()
        return render_template("ind.html", kendaraan=kendaraan_data, rekap=rekap_data)
    except Exception as e:
        print(f"Database error: {e}")
        return render_template("ind.html", kendaraan=[], rekap=[])

@app.route("/stat.html")
def stat():
    try:
        cursor.execute("SELECT * FROM kendaraan ORDER BY waktu DESC LIMIT 10")
        kendaraan_data = cursor.fetchall()
        cursor.execute("SELECT * FROM rekap_harian ORDER BY tanggal DESC LIMIT 10")
        rekap_data = cursor.fetchall()
        return render_template("stat.html", kendaraan=kendaraan_data, rekap=rekap_data)
    except Exception as e:
        print(f"Database error: {e}")
        return render_template("stat.html", kendaraan=[], rekap=[])

@app.route("/info.html")
def contact():
    return render_template("info.html")

@socketio.on("connect")
def connect():
    socketio.start_background_task(generate_frames)

if __name__ == "__main__":
    try:
        socketio.run(app, host="0.0.0.0", port=8000)
    finally:
        cursor.close()
        db.close()
