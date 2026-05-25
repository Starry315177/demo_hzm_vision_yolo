from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="D:/Code/RM_vision/dataset/data.yaml",
    epochs=200,
    imgsz=640,
    device="cpu"
)