from utils import clear_dir, remove_bg, get_points_cloud, convert_to_silhouette
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import requests
import os
from typing import Dict, Union, Any
import trimesh

app = Flask(__name__)
CORS(app)
ip_esp32 = ""
ip_cam = ""
counter_frames = 0


FRAMES_FOLDER = Path('frames')
FRAMES_FOLDER.mkdir(exist_ok=True)

PREPROCESS_FOLDER = Path(f'{FRAMES_FOLDER}/pre-process')
PREPROCESS_FOLDER.mkdir(exist_ok=True)

POSTPROCESS_FOLDER = Path(f'{FRAMES_FOLDER}/post-process')
POSTPROCESS_FOLDER.mkdir(exist_ok=True)

SCAN_FOLDER = Path('scan')
SCAN_FOLDER.mkdir(exist_ok=True)

@app.route('/scan/stream', methods=['GET'])
def get_ip() -> Dict[str, str]:
    global ip_scanner
    if ip_cam != "": 
        return jsonify({
            'ip': ip_cam
        }), 200
    else:
        return jsonify({
            'ip': "Not Ip"
        }), 400
    
@app.route('/scan/stream', methods=['POST'])
def connect_ip_camera() -> Dict[str, str]:
    """
    Endpoint para vincular una ip de un esp32 cam
    Returns:
        Dict[str, str]: Estado de la operación
    """
    global ip_cam
    try:
        data = request.get_json()
        if not data or "ip_cam" not in data:
            return jsonify({
                'status': 'error',
                'message': 'IP no proporcionada'
            }), 400
        ip_cam = data["ip_cam"]
        print(f"Nuevo ESP32 vinculado: {ip_cam}")
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/scan/pair', methods=['POST'])
def connect_ip_esp32() -> Dict[str, str]:
    """
    Endpoint para vincular una ip de un esp32
    Returns:
        Dict[str, str]: Estado de la operación
    """
    global ip_esp32
    try:
        data = request.get_json()
        if not data or "ip_esp32" not in data:
            return jsonify({
                'status': 'error',
                'message': 'IP no proporcionada'
            }), 400
        ip_esp32 = data["ip_esp32"]
        print(f"Nuevo ESP32 vinculado: {ip_esp32}")
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/scan/load', methods=['POST'])
def upload_frame():
    """
    Endpoint para recibir frames enviados desde el frontend.
    """
    global counter_frames
    try:
        # Verificar si la clave 'image' está presente en los archivos enviados
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No se encontró archivo en la solicitud con la clave "image"'
            }), 400

        file = request.files['image']

        # Validar nombre del archivo
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nombre de archivo vacío'
            }), 400

        # Validar tipo de archivo (opcional)
        if not file.content_type.startswith("image/"):
            return jsonify({
                'status': 'error',
                'message': 'El archivo no es una imagen válida'
            }), 400

        # Guardar el archivo
        filename = f"frame-{counter_frames + 1}.jpg"
        file_path = os.path.join(PREPROCESS_FOLDER, filename)
        file.save(file_path)
        counter_frames += 1

        return jsonify({
            'status': 'success',
            'message': f"Frame recibido y guardado como {filename}"
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error al procesar el frame: {str(e)}"
        }), 500

@app.route('/scan/process', methods=['GET'])
def start_scan() -> Dict:
    """
    Endpoint para activar el esp32 y generar el modelo
    Returns:
        Any: Archivo del modelo 3D o mensaje de error
    """
    try:
        # Procesar imágenes
        print("--------- PROCESANDO IMAGENES --------")
        remove_bg(PREPROCESS_FOLDER, POSTPROCESS_FOLDER)

        # Generar modelo 3D
        model_dir = SCAN_FOLDER / "3d_models"
        if get_points_cloud(POSTPROCESS_FOLDER, SCAN_FOLDER):
            model_path = model_dir / "scanned_object.glb"
            if model_path.exists():
                response = send_file(
                    str(model_path),
                    mimetype='model/gltf-binary',
                    as_attachment=True,
                    download_name='model.glb'
                )
                
                # Agregar headers CORS necesarios
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                
                return response
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Archivo no encontrado: {model_path}'
                }), 404
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error al generar el modelo 3D'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/scan/start', methods=['GET'])
def start_motor() -> Dict[str, str]:
    """
    Endpoint para iniciar el giro del motor en el ESP32
    Returns:
        Dict[str, str]: Estado de la operación
    """
    # Limpiar directorios de frames anteriores
    clear_dir(PREPROCESS_FOLDER)
    clear_dir(POSTPROCESS_FOLDER)
    global ip_esp32, counter_frames
    counter_frames = 0
    try:
        if ip_esp32 == "":
            return jsonify({
                'status': 'error',
                'message': 'ESP32 no conectado'
            }), 400
        print("-------------------- GIRANDO PLATAFORMA ------------------------")
        # Enviar comando al ESP32 para iniciar el motor
        response = requests.get(f'http://{ip_esp32}/scan/start')
        if response.status_code == 200:
            return jsonify({'status': 'success'})
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error al comunicarse con el ESP32'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)