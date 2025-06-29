# api/transcribe.py
# Este archivo contiene el código de la función sin servidor de Vercel.

import os
from flask import Flask, request, jsonify
import openai
import io

# Inicializa la aplicación Flask
app = Flask(__name__)

# Configura tu clave API de OpenAI.
# Es crucial que NO la pongas directamente aquí.
# Vercel te permite configurar variables de entorno (como OPENAI_API_KEY)
# de forma segura. La función la leerá de las variables de entorno.
# Asegúrate de configurar OPENAI_API_KEY en tu proyecto de Vercel.
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Endpoint para transcribir archivos de audio usando la API de OpenAI Whisper.
    Espera un archivo de audio en la solicitud POST (multipart/form-data).
    """
    # Verifica si se envió un archivo en la solicitud
    if 'audio_file' not in request.files:
        return jsonify({"error": "No se encontró el archivo de audio."}), 400

    audio_file = request.files['audio_file']
    
    # Verifica si el archivo está vacío
    if audio_file.filename == '':
        return jsonify({"error": "Nombre de archivo de audio vacío."}), 400

    # Lee el contenido del archivo de audio
    audio_data = audio_file.read()
    
    # Crea un objeto BytesIO en memoria para pasarlo a la API de OpenAI
    # Esto simula un archivo para la API de OpenAI sin guardarlo en disco
    audio_stream = io.BytesIO(audio_data)
    audio_stream.name = audio_file.filename # Asigna el nombre de archivo original

    try:
        # Llama a la API de OpenAI Whisper para la transcripción
        # Asegúrate de que tu cuenta de OpenAI tenga créditos activos.
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_stream
        )
        
        # Extrae el texto transcrito de la respuesta
        transcribed_text = transcription.text

        # Devuelve la transcripción como una respuesta JSON
        return jsonify({"transcription": transcribed_text}), 200

    except openai.APIError as e:
        # Manejo de errores específicos de la API de OpenAI
        print(f"Error de la API de OpenAI: {e}")
        return jsonify({"error": f"Error de la API de OpenAI: {e.user_message}"}), e.status_code
    except Exception as e:
        # Manejo de cualquier otro error inesperado
        print(f"Error inesperado durante la transcripción: {e}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    # Para pruebas locales, Vercel no usa esto directamente
    app.run(debug=True)

