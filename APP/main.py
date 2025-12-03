import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import ByteTracker
from ANPR.anpr import ANPR

try:
    from extract_plate import get_plate
except:
    get_plate = None

# ----------------------------
# CONFIG
# ----------------------------
YOLO_MODEL = "yolov8m.pt"         # traffic + vehicles
HELMET_MODEL = "helmet.pt"
ANPR_WEIGHTS = "runs/detect/plate_detector10/weights/best.pt"

INPUT_PATH = "clip.mp4"
MODE = "video"

OUT_VIDEO_DIR = "output_video"
OUT_VIDEO_FILE = os.path.join(OUT_VIDEO_DIR, "combined_output.mp4")

EVIDENCE_RED_DIR = "evidence_red_light"
EVIDENCE_HELMET_DIR = "evidence_no_helmet"
CHALLAN_DIR = "challans"

os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
os.makedirs(EVIDENCE_RED_DIR, exist_ok=True)
os.makedirs(EVIDENCE_HELMET_DIR, exist_ok=True)
os.makedirs(CHALLAN_DIR, exist_ok=True)

TRAFFIC_CONF = 0.65
HELMET_CONF = 0.45
VehicleLabels = ["car", "bus", "truck", "motorcycle", "bicycle"]

# YOUR helmet labels
HELMET_NEGATIVE_LABELS = {"no helmet", "nohelmet", "without helmet"}

# ----------------------------
# LOAD MODELS
# ----------------------------
print("Loading traffic model...")
traffic_model = YOLO(YOLO_MODEL)

print("Loading helmet model...")
helmet_model = YOLO(HELMET_MODEL)

print("Loading ANPR...")
anpr = ANPR(ANPR_WEIGHTS)

tracker = ByteTracker()
saved_ids_red = set()
saved_ids_helmet = set()

# ----------------------------
# ---- AUTO CHALLAN PDF ----
# ----------------------------
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def generate_challan(plate_text, violation_type, evidence_folder):
    if not plate_text:
        plate_text = "UNKNOWN"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    challan_id = f"{plate_text}_{timestamp}"
    pdf_path = os.path.join(CHALLAN_DIR, challan_id + ".pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "TRAFFIC VIOLATION CHALLAN")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Challan ID: {challan_id}")
    c.drawString(50, height - 130, f"Vehicle Number: {plate_text}")
    c.drawString(50, height - 160, f"Violation Type: {violation_type}")
    c.drawString(50, height - 190, f"Date & Time: {time.strftime('%d-%m-%Y %H:%M:%S')}")

    fine = 1000 if violation_type == "red_light_jumping" else 500
    c.drawString(50, height - 220, f"Penalty Amount: ₹{fine}")

    frame_img_path = os.path.join(evidence_folder, "frame.jpg")
    plate_img_path = os.path.join(evidence_folder, "plate.jpg")

    y = height - 350
    if os.path.exists(frame_img_path):
        c.drawString(50, y + 120, "Frame Evidence:")
        c.drawImage(frame_img_path, 50, y - 50, width=300, height=200)
    if os.path.exists(plate_img_path):
        c.drawString(370, y + 120, "Plate:")
        c.drawImage(plate_img_path, 370, y - 50, width=200, height=100)

    c.save()
    print("📄 Challan generated →", pdf_path)

    # auto-open
    try:
        os.startfile(pdf_path)
        print("📂 Challan opened.")
    except Exception as e:
        print("⚠ Could not open PDF:", e)

# ----------------------------
# SAVE EVIDENCE + CHALLAN
# ----------------------------
def save_evidence(base_dir, frame_img, plate_img, plate_text, violation_type, frame_id=None):

    plate_text = (plate_text or "UNKNOWN").replace(" ", "").upper()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    folder_name = f"{plate_text}_{timestamp}"
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, "frame.jpg"), frame_img)
    if plate_img is not None:
        cv2.imwrite(os.path.join(save_dir, "plate.jpg"), plate_img)

    with open(os.path.join(save_dir, "info.txt"), "w") as f:
        f.write(f"Violation: {violation_type}\n")
        f.write(f"Plate: {plate_text}\n")
        f.write(f"Frame ID: {frame_id}\n")
        f.write(f"Timestamp: {timestamp}\n")

    print("Saved evidence →", save_dir)

    generate_challan(plate_text, violation_type, save_dir)

# ----------------------------
# RED-LIGHT DETECTORS
# ----------------------------
def detect_red_light_image(frame, tl_box):
    x1, y1, x2, y2 = tl_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,100,80]); upper1 = np.array([10,255,255])
    lower2 = np.array([170,100,80]); upper2 = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    return cv2.countNonZero(mask) > mask.size * 0.008

def detect_red_light_video(frame, tl_box):
    x1, y1, x2, y2 = tl_box
    h = y2 - y1
    roi = frame[y1+4:y1+int(h*0.25), x1+6:x2-6]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:,:,2])) > 120

# ----------------------------
# ANPR
# ----------------------------
def run_anpr(frame):
    try:
        results = anpr.process_frame(frame)
        if not results:
            return None
        results = sorted(results, key=lambda x: x.get("conf",0), reverse=True)
        return results[0]
    except:
        return None

# ----------------------------
# MAIN PROCESS
# ----------------------------
def process_frame(frame, frame_id=0, is_image=False):
    original = frame.copy()
    results = traffic_model.predict(frame, conf=TRAFFIC_CONF, verbose=False)[0]

    tl_box = None
    vehicle_dets = []

    for b in results.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        label = traffic_model.model.names[int(b.cls[0])]

        if label == "traffic light":
            tl_box = (x1,y1,x2,y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        if label in VehicleLabels:
            vehicle_dets.append([x1,y1,x2-x1,y2-y1,float(b.conf[0])])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)

    # HELMET DETECTION
    h_res = helmet_model.predict(frame, conf=HELMET_CONF, verbose=False)[0]
    nohelmet_boxes = []

    for b in h_res.boxes:
        raw = helmet_model.model.names[int(b.cls[0])]
        if raw is None:
            continue
        lab = str(raw).lower().replace(" ", "")
        if lab in {"nohelmet", "nohelmet", "withouthelmet"}:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            nohelmet_boxes.append((x1,y1,x2,y2))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"NO HELMET",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    tracks = tracker.update_tracks(vehicle_dets)

    # ---------------- RED LIGHT LOGIC ----------------
    if tl_box:
        x1,y1,x2,y2 = tl_box
        stopline_y = y2 + 120
        cv2.line(frame,(0,stopline_y),(frame.shape[1],stopline_y),(0,255,255),2)

        red_on = detect_red_light_image(frame, tl_box) if is_image else detect_red_light_video(frame, tl_box)
        if red_on:
            cv2.putText(frame,"RED LIGHT",(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        W = frame.shape[1]
        xmin_band = int(W*0.05)
        xmax_band = int(W*0.95)

        for t in tracks:
            tid = t.track_id
            x,y,w,h = t.to_tlwh()
            X1,Y1,X2,Y2 = int(x),int(y),int(x+w),int(y+h)

            cx = (X1+X2)//2
            if not (xmin_band <= cx <= xmax_band):
                continue

            crossed = (Y1 < stopline_y - 8) and (Y2 > stopline_y + 6)
            if crossed and red_on:
                if tid in saved_ids_red:
                    continue
                saved_ids_red.add(tid)

                cv2.rectangle(frame,(X1,Y1),(X2,Y2),(0,0,255),3)
                cv2.putText(frame,"RED LIGHT JUMPING",(50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

                plate_crop=None
                plate_text="UNKNOWN"

                if get_plate:
                    try:
                        p = get_plate(original)
                        if p:
                            plate_crop=p["crop"]
                            plate_text=p.get("clean_text","UNKNOWN")
                    except:
                        pass

                if plate_crop is None:
                    plate = run_anpr(original)
                    if plate:
                        plate_crop = plate.get("crop")
                        plate_text = plate.get("clean_text","UNKNOWN")

                save_evidence(EVIDENCE_RED_DIR, original, plate_crop, plate_text,
                              "red_light_jumping", frame_id)

    # ---------------- NO HELMET LOGIC ----------------
    for (hx1,hy1,hx2,hy2) in nohelmet_boxes:
        area = (hx2-hx1)*(hy2-hy1)
        if area < 1500:
            continue

        plate = run_anpr(original)
        plate_text = plate.get("clean_text") if plate else None

        if plate_text:
            sig = plate_text
        else:
            sig = f"NOHELM_{((hx1+hx2)//2)//10}{((hy1+hy2)//2)//10}"

        if sig in saved_ids_helmet:
            continue
        saved_ids_helmet.add(sig)

        crop = plate.get("crop") if plate else None

        save_evidence(EVIDENCE_HELMET_DIR, original, crop,
                      plate_text or "UNKNOWN", "no_helmet", frame_id)

    return frame


# ----------------------------
# RUNNERS
# ----------------------------
def run_image(path):
    frame = cv2.imread(path)
    out = process_frame(frame,0,True)
    cv2.imshow("RESULT",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(path):
    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w=int(cap.get(3)); h=int(cap.get(4))
    out=cv2.VideoWriter(OUT_VIDEO_FILE,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    frame_id=0
    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_id+=1
        processed=process_frame(frame,frame_id,False)
        out.write(processed)
        cv2.imshow("RESULT",processed)
        if cv2.waitKey(1)&0xFF==27: break
    cap.release(); out.release(); cv2.destroyAllWindows()


# ----------------------------
# MAIN
# ----------------------------
ext = os.path.splitext(INPUT_PATH)[1].lower()
if MODE=="image" or ext in [".jpg",".jpeg",".png"]:
    run_image(INPUT_PATH)
else:
    run_video(INPUT_PATH)
