from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import requests

app = Flask(__name__)

# ✅ DESCARGA AUTOMÁTICA DEL MODELO DESDE EL REPO (si no existe)
MODEL_PATH = 'weights/best.pt'  # Usa el que ya tienes en weights/
if not os.path.exists(MODEL_PATH):
    print("Clonando modelo desde GitHub...")
    MODEL_URL = 'https://github.com/leonardo231520/detector-de-humo/raw/main/weights/best.pt'
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Modelo descargado exitosamente.")
    else:
        print(f"Error descargando: {r.status_code}")
        raise Exception("No se pudo descargar el modelo")

# Carga el modelo
model = YOLO(MODEL_PATH)
print("Modelo YOLO cargado correctamente.")

@app.route('/test_image', methods=['POST'])
def test_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    
    file = request.files['image']
    img_path = 'temp_image.jpg'
    file.save(img_path)
    
    # Inferencia con YOLO
    results = model(img_path, conf=0.5, verbose=False)
    
    detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Asume clase 0 = 'humo' (ajusta si es diferente en data.yaml)
                if cls == 0:
                    detections.append({
                        'class_id': cls,
                        'class_name': model.names[cls].upper(),  # Ej: 'HUMO'
                        'confidence': conf
                    })
    
    # Limpia
    os.remove(img_path)
    
    return jsonify({
        'status': 'success',
        'detections': detections,
        'confidence': max([d['confidence'] for d in detections] or [0])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)