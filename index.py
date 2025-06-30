# api/index.py
# Este archivo contiene el código de la función sin servidor de Vercel.

import os
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import io

# Inicializa la aplicación Flask
app = Flask(__name__)

# --- Configuración del modelo Whisper de código abierto ---
# Se recomienda establecer la variable de entorno HF_HOME para que el modelo se cachee en /tmp en Vercel.
# Esto acelera las llamadas subsiguientes una vez que el modelo se ha descargado.
os.environ["HF_HOME"] = "/tmp"
# Puedes cambiar a "small" o "medium" para más precisión (y mayor tamaño/RAM)
model_size = "base"

# Cargar el modelo una vez al inicio para que las siguientes llamadas sean rápidas.
# Esto es crucial en entornos serverless para reutilizar la instancia caliente.
try:
    # Asegúrate de que la versión de faster-whisper en requirements.txt coincida con una versión existente.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(
        f"Modelo Whisper de código abierto '{model_size}' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo Whisper de código abierto: {e}")
    model = None  # Indica que el modelo no pudo ser cargado

    # --- Ruta para la raíz de /api (para verificar que el servicio está activo) ---
    # Vercel mapeará api/index.py a la URL /api.
    # Por lo tanto, esta ruta responderá a https://your-project.vercel.app/api
    @app.route('/api', methods=['GET'])
    def api_root():
        return jsonify({"message": "Servicio de transcripción activo. Usa /api/transcribe para transcribir."}), 200

    # --- Endpoint principal para la transcripción ---
    # Esta ruta responderá a https://your-project.vercel.app/api/transcribe
    @app.route('/api/transcribe', methods=['POST'])
    def transcribe_audio():
        """
        Endpoint para transcribir archivos de audio usando el modelo Whisper de código abierto.
        Espera un archivo de audio en la solicitud POST (multipart/form-data) bajo la clave 'audio_file'.
        """
        # Verifica si el modelo se cargó correctamente al inicio
        if model is None:
            return jsonify({"error": "El modelo Whisper de código abierto no está disponible."}), 500

        # 1. Verifica si se envió un archivo en la solicitud
        if 'audio_file' not in request.files:
            return jsonify({"error": "No se encontró el archivo de audio ('audio_file')."}), 400

        audio_file = request.files['audio_file']

        # 2. Verifica si el archivo está vacío
        if audio_file.filename == '':
            return jsonify({"error": "Nombre de archivo de audio vacío."}), 400

        # 3. Lee el contenido binario del archivo de audio
        audio_data = audio_file.read()

        # 4. Convierte los datos binarios en un flujo de BytesIO
        audio_stream = io.BytesIO(audio_data)
        audio_stream.name = audio_file.filename  # Asigna el nombre de archivo original

        try:
            # 5. Transcribe el audio utilizando el modelo Whisper cargado.
            segments, info = model.transcribe(
                audio_stream, language="es", beam_size=5)

            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text

            # 6. Devuelve la transcripción como una respuesta JSON
            return jsonify({"transcription": transcribed_text.strip()}), 200

        except Exception as e:
            # 7. Manejo de cualquier error durante la transcripción
            print(f"Error durante la transcripción: {e}")
            return jsonify({"error": f"Error interno del servidor durante la transcripción: {str(e)}"}), 500

    if __name__ == '__main__':
        # Para pruebas locales, Vercel no usa esto directamente
        app.run(debug=True)
