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
import socket

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="asdedorio123",  
    database="haha",
    port=3306
)
cursor = db.cursor()

model = YOLO("11best.pt")
cap = cv2.VideoCapture("Video.mp4")

REAL_DISTANCE = 2 
ESP8266_IP = "http://172.20.10.3/" 
ESP8266_PORT = 80 

target_width = 640
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_height = int(target_width * original_height / original_width)

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
        self.history_size = 3  # Gunakan 3 frame
    
    def smooth_classification(self, track_id, detected_class, confidence):
        if track_id not in self.class_history:
            self.class_history[track_id] = []
        

        self.class_history[track_id].append((detected_class, confidence))
        if len(self.class_history[track_id]) > self.history_size:
            self.class_history[track_id].pop(0)
        
 
        class_scores = {}
        for cls, conf in self.class_history[track_id]:
            if cls not in class_scores:
                class_scores[cls] = []
            class_scores[cls].append(conf)
        

        best_class = detected_class
        best_score = 0
        for cls, scores in class_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_class = cls
        
        return best_class
    
    def cleanup_track(self, track_id):
        if track_id in self.class_history:
            del self.class_history[track_id]

classification_smoother = ClassificationSmoother()

def create_mask_from_image(image_path):
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

def create_polygon_mask(frame_shape):
    global mask
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    points = np.array([
        [100, target_height],
        [target_width-100, target_height],
        [target_width-200, 200],
        [200, 200]
    ], np.int32)
    
    cv2.fillPoly(mask, [points], 255)
    return mask

def get_mask_bounds():
    if mask is None:
        return 200, target_height
    
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
    
    return top_y if top_y is not None else 200, bottom_y if bottom_y is not None else target_height

def setup_mask():
    global mask, mask_setup_done
    
    if create_mask_from_image("mask.png") or create_mask_from_image("mask.jpg"):
        print("Mask berhasil dimuat dari file")
    else:
        dummy_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        create_polygon_mask(dummy_frame.shape)
        print("Mask polygon default dibuat")
    
    mask_setup_done = True

def is_in_mask(x, y):
    if mask is None:
        return True
    
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] > 0
    return False


def apply_mask_to_detections(boxes, results, fps):
    filtered_boxes = []
    

    if fps < 10:
        conf_threshold = 0.3
    elif fps < 20:
        conf_threshold = 0.4
    else:
        conf_threshold = 0.5
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        

        if conf < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if is_in_mask(cx, cy):
            filtered_boxes.append(box)
    
    return filtered_boxes

def send_to_esp8266(speed):
    try:
        url = f"{ESP8266_IP}:{ESP8266_PORT}/update"
        params = {"kecepatan": int(speed)}
        response = requests.get(url, params=params, timeout=1)
        if response.status_code == 200:
            print(f"Berhasil kirim ke ESP8266: {speed} km/h")
        else:
            print(f"Gagal kirim ke ESP8266, status: {response.status_code}")
    except Exception as e:
        print(f"Error saat mengirim ke ESP8266: {e}")

def find_esp8266_ip():
    base_ip = "172.20.10." 
    port = 80
    
    for i in range(1, 255):
        ip = base_ip + str(i)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                try:
                    test_url = f"http://{ip}:{port}/update?kecepatan=0"
                    response = requests.get(test_url, timeout=1)
                    if response.status_code == 200:
                        print(f"ESP8266 ditemukan di: {ip}")
                        return f"http://{ip}"
                except:
                    continue
        except:
            continue
    
    return None

def generate_frames():
    global prev_frame_time, object_id, total_motor, total_mobil, mask_setup_done
    offset=7
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
        
        line1_y = mask_top + 10 #setingan30awal
        
        line_distance = int(mask_height * 0.40)
        line2_y = line1_y + line_distance
        
        if line2_y > mask_bottom - 10:
            line2_y = mask_bottom - 10
            line1_y = line2_y - line_distance
        
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_time

        results = model(frame)[0]
        
        filtered_boxes = apply_mask_to_detections(results.boxes.data, results, fps)

        if mask is not None:
            mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_overlay[:, :, 0] = 0
            mask_overlay[:, :, 1] = 0
            mask_overlay[:, :, 2] = 0
            frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)
       
        # bagian untuk: garis deteksi
        #cv2.line(frame, (0, line1_y), (target_width, line1_y), (0, 255, 0), 2)
        #cv2.line(frame, (0, line2_y), (target_width, line2_y), (255, 0, 0), 2)

        current_boxes = []

        for box in filtered_boxes:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = results.names[int(cls)]

            base_tolerance = 50
            fps_factor = max(1, 30 / max(fps, 1))  # Normalisasi ke 30 FPS
            dynamic_tolerance = int(base_tolerance * fps_factor)
            
            matched_id = None
            min_distance = float('inf')
            
            for tid, data in track_memory.items():
                prev_cx, prev_cy = data["centroid"]
                distance = ((prev_cx - cx) ** 2 + (prev_cy - cy) ** 2) ** 0.5
                
                if distance < dynamic_tolerance and distance < min_distance:
                    matched_id = tid
                    min_distance = distance

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
                    "last_seen": current_time,
                    "confidence": conf
                }
                track = track_memory[matched_id]
            else:
                track["centroid"] = (cx, cy)
                track["last_seen"] = current_time
                
        
                track["class"] = classification_smoother.smooth_classification(
                    matched_id, class_name, conf
                )

            base_line_tolerance = 9
            line_tolerance = int(base_line_tolerance * fps_factor)
            
            if line1_y - line_tolerance < cy < line1_y + line_tolerance and track["t1"] is None and is_in_mask(cx, cy):
                track["t1"] = time.time()
                print(f"Object {matched_id} ({track['class']}) crossed line 1")

            if line2_y - line_tolerance < cy < line2_y + line_tolerance and track["t2"] is None and is_in_mask(cx, cy):
                track["t2"] = time.time()
                print(f"Object {matched_id} ({track['class']}) crossed line 2")

            if track["t1"] and track["t2"] and not track["sent"]:
                time_diff = track["t2"] - track["t1"]
                if time_diff > 0:
                    speed = round((REAL_DISTANCE / time_diff) * 3.6 + offset, 2)
                    track["speed"] = speed

                    class_lower = track["class"].lower()
                    if class_lower in ["mobil", "car"]:
                        total_mobil += 1
                    elif class_lower in ["motor", "motorbike"]:
                        total_motor += 1
                    else:
                        print(f"Kelas tidak dikenal: {track['class']}")

                    socketio.emit("data_info", {
                        "kecepatan": speed,
                        "fps": int(fps),
                        "jumlah_mobil": total_mobil,
                        "jumlah_motor": total_motor
                    })

                    threading.Thread(target=send_to_esp8266, args=(speed,)).start()

                    # Database send dengan error handling
                    try:
                        now = datetime.now()
                        insert_kendaraan = """
                            INSERT INTO kendaraan (jenis, kecepatan, waktu)
                            VALUES (%s, %s, %s)
                        """
                        cursor.execute(insert_kendaraan, (track["class"], speed, now))
                        db.commit()

                        today = date.today()
                        if class_lower in ["mobil", "car"]:
                            update_rekap = """
                                INSERT INTO rekap_harian (tanggal, jumlah_mobil, jumlah_motor)
                                VALUES (%s, 1, 0)
                                ON DUPLICATE KEY UPDATE jumlah_mobil = jumlah_mobil + 1
                            """
                        elif class_lower in ["motor", "motorbike"]:
                            update_rekap = """
                                INSERT INTO rekap_harian (tanggal, jumlah_mobil, jumlah_motor)
                                VALUES (%s, 0, 1)
                                ON DUPLICATE KEY UPDATE jumlah_motor = jumlah_motor + 1
                            """
                        else:
                            update_rekap = None

                        if update_rekap:
                            cursor.execute(update_rekap, (today,))
                            db.commit()
                            
                        print(f"Speed recorded: {speed} km/h for {track['class']}")

                    except Exception as e:
                        print(f"Database error: {e}")

                    track["sent"] = True

            color = (0, 255, 255) if track["class"].lower() in ["motor", "motorbike"] else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # bagian: class confidence
            # label = f"{track['class']} ({conf:.2f})"
            # cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # jika butuh :  object ID
            # cv2.putText(frame, f"ID:{matched_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if track["speed"] is not None:
                cv2.putText(frame, f"{track['speed']} km/h", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # PERBAIKAN: Cleanup tracks dengan waktu yang lebih lama untuk FPS rendah
        cleanup_time = 5 if fps < 10 else 3
        for tid in list(track_memory):
            if tid not in current_boxes:
                if current_time - track_memory[tid]["last_seen"] > cleanup_time:
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
