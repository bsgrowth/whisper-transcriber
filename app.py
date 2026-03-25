import os
import tempfile
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.post("/transcribe")
def transcribe():
    data = request.get_json(force=True, silent=True)
    if not data or "audioUrl" not in data:
        return jsonify({"error": "Missing required field: audioUrl"}), 400

    audio_url = data["audioUrl"]

    try:
        response = requests.get(audio_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download audio: {str(e)}"}), 400

    content_type = response.headers.get("Content-Type", "audio/mpeg")
    ext = _ext_from_content_type(content_type)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        os.unlink(tmp_path)

    return jsonify({"text": transcript.text})


def _ext_from_content_type(content_type: str) -> str:
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/mp4": ".mp4",
        "audio/x-m4a": ".m4a",
        "audio/flac": ".flac",
    }
    base = content_type.split(";")[0].strip().lower()
    return mapping.get(base, ".mp3")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
