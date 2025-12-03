import cv2
import os

print("File exists (test8.jpg):", os.path.exists("test8.jpg"))
print("Helmet model exists:", os.path.exists("helmet.pt"))
print("Vehicle model exists:", os.path.exists("yolov8m.pt"))

try:
    from extract_plate import get_plate
    print("ANPR import: OK")
except Exception as e:
    print("ANPR import ERROR:", e)
