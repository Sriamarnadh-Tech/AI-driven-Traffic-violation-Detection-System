from ultralytics import YOLO
import cv2
import os

# -----------------------------
# IMPORT ANPR MODULE
# -----------------------------
from ANPR.anpr import ANPR

# load plate detector
anpr = ANPR("runs/detect/plate_detector10/weights/best.pt")

def get_plate(frame):
    results = anpr.process_frame(frame)
    if not results:
        return None
    return sorted(results, key=lambda x: x["conf"], reverse=True)[0]


# -----------------------------
# LOAD HELMET MODEL
# -----------------------------
model = YOLO("helmet.pt")

# -----------------------------
# INPUT
# -----------------------------
input_path = "clip.mp4"

ext = os.path.splitext(input_path)[1].lower()
is_image = ext in [".jpg", ".jpeg", ".png"]
is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]

# -----------------------------
# CREATE EVIDENCE FOLDER
# -----------------------------
os.makedirs("evidence", exist_ok=True)


# ===================================================
# PROCESS IMAGE
# ===================================================
def process_image():

    print("🖼️ Processing IMAGE...")

    frame = cv2.imread(input_path)
    original = frame.copy()

    results = model.predict(frame, conf=0.5, verbose=False)

    violation_detected = False

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):

            label = model.names[int(cls)].lower()

            if label not in ["no helmet", "nohelmet", "without helmet"]:
                continue

            violation_detected = True

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "NO HELMET", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # -----------------------------
    # IF VIOLATION → ANPR
    # -----------------------------
    if violation_detected:
        print("🚨 NO HELMET DETECTED — Running ANPR...")

        plate = get_plate(original)
        if plate:
            crop = plate["crop"]
            text = plate.get("clean_text", "")

            print("📌 Plate Extracted:", text)

            cv2.imwrite("evidence/plate_crop.jpg", crop)
            cv2.imwrite("evidence/violation_image.jpg", frame)

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ===================================================
# PROCESS VIDEO
# ===================================================
def process_video():

    print("🎥 Processing VIDEO...\n")
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "output_video/no_helmet_output.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✔️ Video completed.")
            break

        frame_id += 1
        original = frame.copy()

        results = model.predict(frame, conf=0.5, verbose=False)

        violation_detected = False

        # -----------------------------
        # DETECTION LOOP
        # -----------------------------
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):

                label = model.names[int(cls)].lower()

                if label not in ["no helmet", "nohelmet", "without helmet"]:
                    continue

                violation_detected = True

                x1, y1, x2, y2 = map(int, box)

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"NO HELMET",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3,
                )

        # -----------------------------
        # SAVE EVIDENCE FRAME
        # -----------------------------
        if violation_detected:

            print(f"🚨 Frame {frame_id}: NO HELMET DETECTED")

            # Save evidence frame
            cv2.imwrite(f"evidence/frame_{frame_id}.jpg", frame)

            # Run ANPR
            plate = get_plate(original)

            if plate:
                crop = plate["crop"]
                text = plate.get("clean_text", "")

                print("📌 Plate:", text)

                cv2.imwrite(f"evidence/frame_{frame_id}_plate.jpg", crop)

        # -----------------------------
        # WRITE + SHOW
        # -----------------------------
        out.write(frame)

        cv2.imshow("Video Result", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ===================================================
# RUN
# ===================================================
if is_image:
    process_image()

elif is_video:
    process_video()

else:
    print("❌ Unsupported file type")
