# api/transcribe.py
# Este archivo debe ir dentro de una carpeta llamada 'api' en tu proyecto Vercel.

import os
from flask import Flask, request, jsonify
# Importa faster_whisper para la versión de código abierto
from faster_whisper import WhisperModel
import io

# Inicializa la aplicación Flask
app = Flask(__name__)

# --- Configuración del modelo Whisper de código abierto ---
# Se recomienda establecer la variable de entorno HF_HOME para que el modelo se cachee en /tmp en Vercel.
# Esto acelera las llamadas subsiguientes una vez que el modelo se ha descargado.
os.environ["HF_HOME"] = "/tmp"

# Elige el tamaño del modelo: "tiny", "base", "small", "medium", "large".
# "base" es un buen punto de partida para español. Para mejor precisión, "small" o "medium".
# "device='cpu'" significa que usará la CPU. Puedes usar "cuda" si Vercel ofrece GPU.
# "compute_type='int8'" es una optimización para CPU que mejora la velocidad.
model_size = "base"

# Cargar el modelo una vez al inicio de la aplicación para mayor eficiencia.
try:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(
        f"Modelo Whisper de código abierto '{model_size}' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo Whisper de código abierto: {e}")
    model = None  # Indica que el modelo no pudo ser cargado


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Endpoint para transcribir archivos de audio usando el modelo Whisper de código abierto.
    Espera un archivo de audio en la solicitud POST (multipart/form-data) bajo la clave 'audio_file'.
    """
    # Verifica si el modelo se cargó correctamente al inicio
    if model is None:
        return jsonify({"error": "El modelo Whisper de código abierto no está disponible."}), 500

    if 'audio_file' not in request.files:
        return jsonify({"error": "No se encontró el archivo de audio ('audio_file')."}), 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({"error": "Nombre de archivo de audio vacío."}), 400

    audio_data = audio_file.read()

    audio_stream = io.BytesIO(audio_data)
    audio_stream.name = audio_file.filename

    try:
        # Transcribe el audio utilizando el modelo Whisper de código abierto.
        # Es crucial especificar el idioma si se conoce para una mejor precisión.
        segments, info = model.transcribe(
            audio_stream, language="es", beam_size=5)

        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text

        return jsonify({"transcription": transcribed_text.strip()}), 200

    except Exception as e:
        print(f"Error durante la transcripción: {e}")
        return jsonify({"error": f"Error interno del servidor durante la transcripción: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
