from ANPR.anpr import ANPR
import cv2

anpr = ANPR("runs/detect/plate_detector10/weights/best.pt")   

def get_plate(frame):
    results = anpr.process_frame(frame)

    if not results:
        return None

    
    results = sorted(results, key=lambda x: x["conf"], reverse=True)
    best = results[0]

    return best
