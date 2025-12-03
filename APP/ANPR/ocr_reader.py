import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRReader:
    def __init__(self):
        pass

    def read_text(self, crop_img):
        if crop_img is None or crop_img.size == 0:
            return ""

        # Resize slightly for clarity
        h, w = crop_img.shape[:2]
        scale = 2 if h < 60 else 1.5
        img = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Adaptive threshold (works better for Indian plates)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )

        # Debug preview (optional)
        # cv2.imshow("OCR crop", thresh); cv2.waitKey(1)

        # OCR
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        text = text.strip()
        print("RAW OCR:", text)  # <-- important debug

        return text
