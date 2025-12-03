import cv2

cap = cv2.VideoCapture("Video1.mp4")
success, frame = cap.read()
if success:
    cv2.imwrite("test1.jpg", frame)
    print("Frame extracted as test1.jpg")
else:
    print("Failed to extract frame")
cap.release()
