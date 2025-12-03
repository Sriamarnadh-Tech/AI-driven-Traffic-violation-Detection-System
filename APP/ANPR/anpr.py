from .detect_plate import PlateDetector
from .ocr_reader import OCRReader
from .text_cleaner import clean_text as clean_fn

class ANPR:
    def __init__(self, model_path="runs/detect/plate_detector10/weights/best.pt"):
        self.detector = PlateDetector(model_path)
        self.ocr = OCRReader()

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        results = []

        for det in detections:
            crop = det["crop"]

            raw = self.ocr.read_text(crop)
            clean = clean_fn(raw)

            results.append({
                "bbox": det["bbox"],
                "raw_text": raw,
                "clean_text": clean,
                "crop": crop,
                "conf": det["conf"]
            })

        return results
