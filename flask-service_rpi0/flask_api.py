#!/usr/bin/env python3
from flask import Flask, request, jsonify
import os
import base64
import json
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import time
from io import BytesIO

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage as fb_storage

# OpenAI v1 client
from openai import OpenAI

# Google Cloud TTS
from google.cloud import texttospeech

# Load environment vars from .env
load_dotenv()

app = Flask(__name__)

# ─── OpenAI Client ─────────────────────────────────────────────────────────────
# Instantiate the new client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Google Cloud TTS Client ──────────────────────────────────────────────────
# Make sure GOOGLE_APPLICATION_CREDENTIALS is set to your service account JSON
tts_client = texttospeech.TextToSpeechClient()

# ─── Firebase Storage ──────────────────────────────────────────────────────────
# Ensure this points at your downloaded JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/home/Mihai/gpt_api/aplicatielicenta-bf604-firebase-adminsdk-fbsvc-cf92a81613.json"
)

cred = credentials.Certificate(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)
firebase_admin.initialize_app(cred, {
    "storageBucket": "aplicatielicenta-bf604.firebasestorage.app"
})
bucket = fb_storage.bucket()

# Audio playback using system tools (mpg123, aplay, or omxplayer)
def play_audio_file(file_path):
    """
    Play audio file using available system audio players
    Priority: mpg123 > aplay > omxplayer > paplay
    """
    try:
        # Try mpg123 first (best for MP3)
        result = subprocess.run(['which', 'mpg123'], capture_output=True)
        if result.returncode == 0:
            subprocess.run(['mpg123', '--quiet', file_path], check=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Try omxplayer (Raspberry Pi optimized)
        result = subprocess.run(['which', 'omxplayer'], capture_output=True)
        if result.returncode == 0:
            subprocess.run(['omxplayer', '-o', 'local', '--no-keys', file_path], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Try paplay (PulseAudio)
        result = subprocess.run(['which', 'paplay'], capture_output=True)
        if result.returncode == 0:
            # Convert MP3 to WAV first for paplay
            wav_path = file_path.replace('.mp3', '.wav')
            subprocess.run(['ffmpeg', '-i', file_path, '-y', wav_path], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['paplay', wav_path], check=True)
            os.unlink(wav_path)  # Clean up WAV file
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Try aplay as last resort (may not work with MP3)
        result = subprocess.run(['which', 'aplay'], capture_output=True)
        if result.returncode == 0:
            # Convert MP3 to WAV first for aplay
            wav_path = file_path.replace('.mp3', '.wav')
            subprocess.run(['ffmpeg', '-i', file_path, '-y', wav_path], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['aplay', wav_path], check=True)
            os.unlink(wav_path)  # Clean up WAV file
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("No suitable audio player found. Please install mpg123, omxplayer, or ffmpeg+aplay")
    return False

# ─── Language Configuration ───────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'voice': 'en-US-Standard-C', 'language_code': 'en-US'},
    'ro': {'name': 'Romanian', 'voice': 'ro-RO-Standard-A', 'language_code': 'ro-RO'},
    'es': {'name': 'Spanish', 'voice': 'es-ES-Standard-A', 'language_code': 'es-ES'},
    'fr': {'name': 'French', 'voice': 'fr-FR-Standard-A', 'language_code': 'fr-FR'},
    'de': {'name': 'German', 'voice': 'de-DE-Standard-A', 'language_code': 'de-DE'},
    'it': {'name': 'Italian', 'voice': 'it-IT-Standard-A', 'language_code': 'it-IT'},
    'pt': {'name': 'Portuguese', 'voice': 'pt-BR-Standard-A', 'language_code': 'pt-BR'},
    'ru': {'name': 'Russian', 'voice': 'ru-RU-Standard-A', 'language_code': 'ru-RU'},
    'zh': {'name': 'Chinese', 'voice': 'zh-CN-Standard-A', 'language_code': 'zh-CN'},
    'ja': {'name': 'Japanese', 'voice': 'ja-JP-Standard-A', 'language_code': 'ja-JP'},
    'ko': {'name': 'Korean', 'voice': 'ko-KR-Standard-A', 'language_code': 'ko-KR'},
    'ar': {'name': 'Arabic', 'voice': 'ar-XA-Standard-A', 'language_code': 'ar-XA'},
    'hi': {'name': 'Hindi', 'voice': 'hi-IN-Standard-A', 'language_code': 'hi-IN'},
}

# ─── TTS Helper with Google Cloud TTS ─────────────────────────────────────────
def speak_with_google_tts(text: str, language_code: str = 'en-US', voice_name: str = 'en-US-Standard-C'):
    """
    Convert text to speech using Google Cloud TTS and play it
    """
    try:
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
            volume_gain_db=0.0
        )

        # Perform the text-to-speech request
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.audio_content)
            temp_file_path = temp_file.name

        # Play the audio using system audio player
        audio_played = play_audio_file(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return audio_played

    except Exception as e:
        print(f"Google TTS error: {e}")
        # Fallback to espeak
        fallback_speak(text)
        return False

def fallback_speak(text: str, speed: int = 175):
    """Fallback TTS using espeak"""
    try:
        subprocess.run(["espeak", "-s", str(speed), text], check=True)
    except Exception as e:
        print(f"Espeak error: {e}")

def get_language_config(lang_code: str):
    """Get language configuration or default to English"""
    return SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES['en'])

# ─── /caption Endpoint ─────────────────────────────────────────────────────────
@app.route('/caption', methods=['POST'])
def caption_image():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400

    # Extract parameters from request
    b64_image = data['image']
    custom_prompt = data.get('prompt', "What do you see in this image?")
    language = data.get('language', 'en')  # Default to English
    use_google_tts = data.get('use_google_tts', True)  # Option to use Google TTS
    
    # Get language configuration
    lang_config = get_language_config(language)
    
    # Build the multimodal message
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": custom_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
        ]
    }]

    try:
        # Use the v1 API to get image caption
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=300,
            messages=messages
        )
        caption = resp.choices[0].message.content

        # Speak the caption using Google Cloud TTS or fallback
        tts_success = False
        if use_google_tts:
            tts_success = speak_with_google_tts(
                caption, 
                lang_config['language_code'], 
                lang_config['voice']
            )
        
        if not tts_success:
            # Fallback to espeak
            fallback_speak(caption)

        # Save a timestamped JSON record in Storage
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "caption": caption,
            "prompt": custom_prompt,
            "language": language,
            "language_name": lang_config['name'],
            "tts_method": "google_cloud" if tts_success else "espeak"
        }
        
        blob_name = f"captions/{record['timestamp']}.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(record, indent=2),
            content_type="application/json"
        )

        return jsonify({
            "response": caption,
            "language": language,
            "language_name": lang_config['name'],
            "prompt_used": custom_prompt,
            "tts_method": "google_cloud" if tts_success else "espeak",
            "storage_path": blob_name,
            "download_url": blob.public_url
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── /languages Endpoint ──────────────────────────────────────────────────────
@app.route('/languages', methods=['GET'])
def get_supported_languages():
    """Return list of supported languages"""
    return jsonify({
        "supported_languages": SUPPORTED_LANGUAGES
    }), 200

# ─── /test_tts Endpoint ────────────────────────────────────────────────────────
@app.route('/test_tts', methods=['POST'])
def test_tts():
    """Test TTS with custom text and language"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing text data"}), 400
    
    text = data['text']
    language = data.get('language', 'en')
    use_google_tts = data.get('use_google_tts', True)
    
    lang_config = get_language_config(language)
    
    try:
        tts_success = False
        if use_google_tts:
            tts_success = speak_with_google_tts(
                text, 
                lang_config['language_code'], 
                lang_config['voice']
            )
        
        if not tts_success:
            fallback_speak(text)
        
        return jsonify({
            "message": "TTS test completed",
            "text": text,
            "language": language,
            "language_name": lang_config['name'],
            "tts_method": "google_cloud" if tts_success else "espeak"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── /latest Endpoint (unchanged) ───────────────────────────────────────────────
@app.route('/latest', methods=['GET'])
def latest_data():
    try:
        import mariadb
        conn = mariadb.connect(
            user="db_user",
            password="your_password",
            host="localhost",
            port=3306,
            database="sensor_db"
        )
        cur = conn.cursor()
        cur.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify({"message": "No sensor data found"}), 404

        return jsonify({
            "id": row[0],
            "timestamp": str(row[1]),
            "mode": row[2],
            "voltage_hw": row[3],
            "soc_hw": row[4],
            "voltage_sw": row[5],
            "soc_sw": row[6],
            "gps_lat": row[7],
            "gps_lon": row[8],
            "relay1_state": row[9],
            "relay2_state": row[10],
            "username": row[11],
            "password": row[12]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Run App ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
