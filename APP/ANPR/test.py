import cv2
from anpr import ANPR

anpr = ANPR("runs/detect/plate_detector10/weights/best.pt")

img = cv2.imread("test8.jpg")
results = anpr.process_frame(img)

print("\n=== ANPR RESULTS ===")

for r in results:
    x1, y1, x2, y2 = r["bbox"]
    text = r["clean_text"]

    print("RAW:", r["raw_text"])
    print("CLEAN:", r["clean_text"])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("ANPR RESULT", img)
cv2.waitKey(0)
