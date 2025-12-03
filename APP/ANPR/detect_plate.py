from ultralytics import YOLO
import cv2

class PlateDetector:
    def __init__(self, model_path="runs/detect/plate_detector10/weights/best.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        plates = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "conf": float(box.conf[0])
            })

        return plates
