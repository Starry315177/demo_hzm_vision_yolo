from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx")
print("✅ ONNX模型导出完成！文件保存在 runs/detect/train/weights/best.onnx")