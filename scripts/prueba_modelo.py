from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np  # Para convertir BGR a RGB

# Carga el modelo
model = YOLO(os.path.join('.', 'weights', 'best.pt'))

# Umbrales y clases (ajusta según tu entrenamiento)
conf_threshold = 0.5
classes = [0, 1]  # 0: humo, 1: vapor o imagen impresa
alerta_threshold = 3

print("Iniciando cámara de detección de humo/vapor. Apunta a la escena. Presiona Ctrl+C en consola o cierra ventana para salir.")

# Inicializa cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

frame_count = 0
detecciones_consecutivas = 0

# Configura matplotlib para ventana no bloqueante
plt.ion()  # Modo interactivo
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.title('Detección de Humo/Vapor')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Procesando...")

        # Inferencia
        results = model(frame, conf=conf_threshold, verbose=False)

        alerta = False
        num_detecciones = 0

        # Procesa detecciones
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls in classes and conf > conf_threshold:
                        num_detecciones += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = (255, 0, 0) if alerta else (0, 255, 0)  # Rojo/Verde (BGR)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{model.names[cls]}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Lógica de alerta
        if num_detecciones > 0:
            detecciones_consecutivas += 1
            if detecciones_consecutivas >= alerta_threshold:
                alerta = True
                print(f"¡ALERTA! Detección en frame {frame_count}.")
                # Opcional: cv2.imwrite(f'alerta_{frame_count}.jpg', frame)
        else:
            detecciones_consecutivas = 0

        # Status en frame
        status = f"Frame: {frame_count} | Detecc: {num_detecciones} | Alerta: {'SI' if alerta else 'NO'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Muestra con matplotlib (convierte BGR a RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.clear()
        ax.imshow(frame_rgb)
        ax.set_title('Detección de Humo/Vapor')
        ax.axis('off')
        plt.draw()
        plt.pause(0.03)  # ~30 FPS; ajusta para más lento/rápido

except KeyboardInterrupt:
    print("Interrumpido por usuario.")

finally:
    # Limpieza
    cap.release()
    plt.close(fig)
    print("Cámara cerrada. Total frames:", frame_count)