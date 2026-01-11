from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List
import speech_recognition as sr
import tempfile
import os

app = FastAPI()

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = sr.Recognizer()

# In-memory log storage
logs: List[dict] = []
MAX_LOGS = 100

def add_log(message: str, level: str = "info"):
    """Add a log message to the log store"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "level": level
    }
    logs.append(log_entry)
    # Keep only the most recent logs
    if len(logs) > MAX_LOGS:
        logs.pop(0)
    print(f"[{timestamp}] {message}")  # Also print to console

@app.get("/logs")
async def get_logs():
    """Get recent logs"""
    return {"logs": logs[-50:]}  # Return last 50 logs

# Add initial log
add_log("🚀 Backend server started", "info")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    tmp_path = None
    try:
        add_log("📥 Received audio upload", "info")
        content = await audio.read()
        add_log(f"File size: {len(content)} bytes", "info")

        # Detect file type
        content_type = audio.content_type or ""
        filename = audio.filename or ""
        add_log(f"Filename: {filename}, Content-Type: {content_type}", "info")

        if "webm" in content_type.lower() or filename.endswith(".webm"):
            suffix = ".webm"
        elif "ogg" in content_type.lower() or filename.endswith(".ogg"):
            suffix = ".ogg"
        elif "mp3" in content_type.lower() or filename.endswith(".mp3"):
            suffix = ".mp3"
        else:
            suffix = ".wav"
        add_log(f"Detected suffix: {suffix}", "info")

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        add_log(f"Temporary file saved", "info")

        # Convert to WAV if needed
        if suffix != ".wav":
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(tmp_path)
                print(f"Original audio: {audio_segment.channels} channels, {audio_segment.frame_rate} Hz, {len(audio_segment)/1000:.2f}s")
                
                # Force mono + 16kHz
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                wav_path = tmp_path.replace(suffix, ".wav")
                audio_segment.export(wav_path, format="wav")
                add_log(f"✅ Converted to WAV: {audio_segment.channels}ch, {audio_segment.frame_rate}Hz, {len(audio_segment)/1000:.2f}s", "success")
                
                os.remove(tmp_path)
                tmp_path = wav_path
            except ImportError:
                add_log("❌ pydub not installed", "error")
                return {"text": "Error: pydub required. Install: pip install pydub"}
            except Exception as e:
                error_msg = str(e)
                # Filter out verbose ffmpeg configuration output
                if "--enable" in error_msg or len(error_msg) > 200:
                    add_log(f"❌ Error converting audio", "error")
                    return {"text": "Could not understand audio"}
                add_log(f"❌ Error converting audio: {e}", "error")
                return {"text": "Could not understand audio"}

        # Transcribe
        add_log("🔄 Starting transcription...", "info")
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            add_log(f"Audio recorded: {len(audio_data.frame_data)} bytes", "info")

        try:
            text = recognizer.recognize_google(audio_data)
            add_log(f"✅ Transcription successful: {text}", "success")
        except sr.UnknownValueError:
            text = "Could not understand audio"
            add_log("⚠️ Could not understand audio", "warning")
        except sr.RequestError as e:
            text = f"Speech recognition error: {e}"
            add_log(f"❌ Speech recognition request error: {e}", "error")

        return {"text": text}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            add_log("🗑️ Temporary file deleted", "info")
