from ultralytics import YOLO

model = YOLO('weights/best.pt')
success = model.export(format='onnx', imgsz=640, simplify=True)
print(f"Exportado: {success}")