import cv2
import numpy as np
from ultralytics import YOLO
from extract_plate import get_plate
import os
import time

# -------------------------
# ADDED: BYTE TRACK IMPORT
# -------------------------
from bytetrack import ByteTracker
tracker = ByteTracker()
saved_ids = set()   # avoid duplicate evidence

# -------------------------
# Params (change if needed)
# -------------------------
YOLO_MODEL = "yolov8m.pt"
INPUT_PATH = "tr.mp4"        # video or image path
MODE = "video"               # "image" or "video"
OUT_VIDEO_DIR = "output_video"
OUT_VIDEO_FILE = os.path.join(OUT_VIDEO_DIR, "red_light_output.mp4")

# Evidence base
BASE_DIR = "evidence_red_light"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUT_VIDEO_DIR, exist_ok=True)

# Load model
model = YOLO(YOLO_MODEL)
coco = model.model.names
VehicleLabels = ["car", "bus", "truck", "motorcycle", "bicycle"]

# -------------------------
# Utility: save evidence
# -------------------------
def save_evidence(frame_img, plate_img, plate_text, violation_type, frame_id=None):
    plate_text = plate_text.replace(" ", "").upper() if plate_text else "UNKNOWN"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{plate_text}_{timestamp}"
    save_dir = os.path.join(BASE_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    frame_path = os.path.join(save_dir, "frame.jpg")
    cv2.imwrite(frame_path, frame_img)

    if plate_img is not None:
        plate_path = os.path.join(save_dir, "plate.jpg")
        cv2.imwrite(plate_path, plate_img)

    info_path = os.path.join(save_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"Violation Type: {violation_type}\n")
        f.write(f"Plate: {plate_text}\n")
        f.write(f"Frame ID: {frame_id}\n")
        f.write(f"Timestamp: {timestamp}\n")

    print(f"📁 Evidence saved → {folder_name}")

# -------------------------
# Red detection for image
# -------------------------
def detect_red_light_image(frame, tl_box):
    x1, y1, x2, y2 = tl_box
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 80]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 100, 80]); upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

    red_pixels = cv2.countNonZero(mask)
    return red_pixels > mask.size * 0.008

# -------------------------
# Red detection for video
# -------------------------
def detect_red_light_video(frame, tl_box):
    x1, y1, x2, y2 = tl_box
    h = y2 - y1
    ry1 = y1 + 4
    ry2 = y1 + max(6, int(h * 0.25))
    rx1 = x1 + 6
    rx2 = x2 - 6

    H, W = frame.shape[:2]
    rx1, rx2 = max(0, rx1), min(W, rx2)
    ry1, ry2 = max(0, ry1), min(H, ry2)

    if ry2 <= ry1 or rx2 <= rx1:
        return False

    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    return mean_v > 120

# -------------------------
# draw label helper
# -------------------------
def draw_text(frame, text, pos, color=(0,0,255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# -------------------------
# main frame processing
# -------------------------
def process_frame(frame, frame_id=0, is_image=False):
    original = frame.copy()
    results = model.predict(frame, conf=0.65, verbose=False)

    tl_box = None
    detections = []     # for tracker

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = coco[int(cls)]

            if label == "traffic light":
                tl_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                draw_text(frame, "Traffic Light", (x1, max(12,y1-8)), (0,255,0))

            if label in VehicleLabels:
                detections.append([x1, y1, x2-x1, y2-y1, float(conf)])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,200,0), 2)

    if tl_box is None:
        return frame

    # -------------------------
    # STOP LINE (moved further down)
    # -------------------------
    tl_x1, tl_y1, tl_x2, tl_y2 = tl_box
    stopline_y = tl_y2 + 120    # <<< ONLY CHANGE YOU ASKED FOR
    stopline_y = min(frame.shape[0]-1, stopline_y)
    cv2.line(frame, (0, stopline_y), (frame.shape[1], stopline_y), (0,255,255), 2)

    # red light check
    red_on = detect_red_light_image(frame, tl_box) if is_image else detect_red_light_video(frame, tl_box)

    if red_on:
        draw_text(frame, "RED LIGHT", (tl_x1, tl_y1 - 30), (0,0,255))

    # -------------------------
    # RUN BYTE TRACK
    # -------------------------
    tracks = tracker.update_tracks(detections)

    H, W = frame.shape[:2]
    x_min_band = int(W * 0.05)
    x_max_band = int(W * 0.95)

    for t in tracks:
        if not t.is_confirmed():
            continue

        tid = t.track_id
        x, y, w, h = t.to_tlwh()
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

        cx = (x1 + x2) // 2
        if not (x_min_band <= cx <= x_max_band):
            continue

        crossed = (y1 < stopline_y - 8) and (y2 > stopline_y + 6)

        if crossed and red_on:

            # avoid duplicates
            if tid in saved_ids:
                continue

            saved_ids.add(tid)

            draw_text(frame, "RED LIGHT JUMPING", (50,50), (0,0,255))
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

            plate = None
            try:
                plate = get_plate(original)
            except:
                pass

            if plate:
                save_evidence(original, plate["crop"], plate["clean_text"], "red_light_jumping", frame_id)
            else:
                save_evidence(original, None, "UNKNOWN", "red_light_jumping", frame_id)

    return frame

# -------------------------
# run image mode
# -------------------------
def run_image(path):
    frame = cv2.imread(path)
    out = process_frame(frame, 0, True)

    cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RESULT", frame.shape[1], frame.shape[0])
    cv2.imshow("RESULT", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------
# run video mode
# -------------------------
def run_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUT_VIDEO_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RESULT", w, h)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        processed = process_frame(frame, frame_id, False)

        out.write(processed)
        cv2.imshow("RESULT", processed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# -------------------------
# select mode
# -------------------------
if MODE == "image":
    run_image(INPUT_PATH)
else:
    run_video(INPUT_PATH)
