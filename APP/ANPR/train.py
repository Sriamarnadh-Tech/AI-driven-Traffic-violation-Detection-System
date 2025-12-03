from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="D:\Traffic\ANPR\datasets\data.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    name="plate_detector"
)
