from ultralytics import YOLO

# Carga modelo base
model = YOLO('yolo11n.pt')  # Nano para ligero

# Entrena (ajusta paths)
results = model.train(
    data='data/data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    device=0,  # GPU
    project='runs/train',
    name='humo_yolo11'
)

print("Entrenamiento completado. Modelo en runs/train/weights/best.pt")