# Flask Audio Processing Service

This is a Flask-based project for handling audio recordings. It uses WebSocket to return results in real-time.

## Prerequisites

1. Install Python dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg (required for audio processing):
```bash
sudo apt update
sudo apt install ffmpeg
```

## Configuration

Before starting the server, you **must** update the audio directory path.

Open `webserver.py` and change line 85:

```python
AUDIO_DIR = "/path/to/your/local/audio/files"
```

Replace `"/path/to/your/local/audio/files"` with your actual local path.

## Running the Service

Use `gunicorn` with `nohup` to run the service. Since WebSocket is used, only **single-threaded** mode is supported.

```bash
nohup gunicorn --workers=1 --threads=1 -b 0.0.0.0:5000 webserver:app > log.out 2>&1 &
```

This will start the service and write output to `log.out`.

## Notes

- Ensure port 5000 is open and accessible(or use nginx).
- Make sure to use a modern browser that supports WebSocket.

