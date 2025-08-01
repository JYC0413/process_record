import re
import os
import io
import time
import json
import torch
import tempfile
from datetime import datetime, timezone
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit
import httpx  # 替换requests

from pydub.utils import mediainfo
from pydub import AudioSegment
import noisereduce as nr
import numpy as np

from dotenv import load_dotenv

from speechbrain.inference import EncoderClassifier

load_dotenv()

gaia_api_key = os.getenv("GAIA_API_KEY")

language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="pretrained_models/lang-id-commonlanguage_ecapa"
)

# import webrtcvad

# 创建语言标签映射
LANGUAGE_MAPPING = {
    'Chinese_Taiwan': 'zh',
    'Chinese_China': 'zh',
    'Chinese_Hongkong': 'zh',
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Italian': 'it',
    'Dutch': 'nl',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Hindi': 'hi',
    'Indonesian': 'id',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'Malay': 'ms',
    'Persian': 'fa',
    'Hebrew': 'he',
    'Swedish': 'sv',
    'Norwegian': 'no',
    'Danish': 'da',
    'Finnish': 'fi',
    'Greek': 'el',
    'Hungarian': 'hu',
    'Czech': 'cs',
    'Slovak': 'sk',
    'Bulgarian': 'bg',
    'Croatian': 'hr',
    'Serbian': 'sr',
    'Slovenian': 'sl',
    'Estonian': 'et',
    'Latvian': 'lv',
    'Lithuanian': 'lt',
    'Romanian': 'ro',
    'Ukrainian': 'uk',
    'Catalan': 'ca',
    'Basque': 'eu',
    'Galician': 'gl',
    'Welsh': 'cy',
    'Irish': 'ga',
    'Icelandic': 'is'
}

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")  # 改为threading，兼容requests/httpx

AUDIO_DIR = r"../9090/record"


def parse_timestamp_from_filename(filename):
    try:
        match = re.search(r'recording_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})', filename)
        if match:
            time_str = match.group(1)

            dt = datetime.fromisoformat(time_str)

            timestamp_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)

            return timestamp_utc
        else:
            print("No valid timestamp found in filename.")
    except Exception as e:
        print(f"Error parsing timestamp from filename: {filename}， error: {e}")
        return None


def get_files_in_range(start_time, end_time):
    files = []
    for root, _, fnames in os.walk(AUDIO_DIR):
        for fname in fnames:
            if fname.endswith(".wav") and fname.startswith("recording_"):
                ts = parse_timestamp_from_filename(fname)
                if ts and start_time <= ts <= end_time:
                    files.append((ts, os.path.relpath(os.path.join(root, fname), AUDIO_DIR)))
    files.sort()
    return [fname for ts, fname in files]


def merge_wav_files_grouped(file_list, max_duration=60):
    """
    合并音频文件，每组总时长不超过 max_duration（秒），返回分组后的 AudioSegment 列表
    """
    groups = []
    current_group = None
    current_duration = 0.0

    for fname in file_list:
        audio = AudioSegment.from_wav(os.path.join(AUDIO_DIR, fname))
        # 音量标准化到 -13 LUFS
        target_lufs = -13.0
        change_in_dBFS = target_lufs - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        # 降噪处理
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
        audio = audio._spawn(reduced_noise.astype(audio.array_type).tobytes())

        audio_duration = len(audio) / 1000.0  # 秒
        if current_group is None:
            current_group = audio
            current_duration = audio_duration
        elif current_duration + audio_duration > max_duration:
            groups.append(current_group)
            current_group = audio
            current_duration = audio_duration
        else:
            current_group += audio
            current_duration += audio_duration

    if current_group is not None:
        groups.append(current_group)
    return groups


def merge_wav_files(file_list):
    merged = None
    for fname in file_list:
        audio = AudioSegment.from_wav(os.path.join(AUDIO_DIR, fname))
        # 先音量标准化到 -13 LUFS
        target_lufs = -13.0
        change_in_dBFS = target_lufs - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        # 降噪处理
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
        audio = audio._spawn(reduced_noise.astype(audio.array_type).tobytes())

        print(f"Processing file: {fname}, duration: {len(audio) / 1000:.2f} seconds")

        if merged is None:
            merged = audio
        else:
            merged += audio
    return merged


def clear_file_name(filename):
    # 文件名里的T替换为_，并清理非法字符
    filename = filename.replace('T', '_')
    print("Original filename:", filename)
    return re.sub(r'[\\/:*?"<>|]', '_', filename)


@app.route("/", methods=["GET"])
def index():
    return render_template("selectRecord.html")


@app.route("/download", methods=["POST"])
def download():
    start = request.form.get("start_time")
    end = request.form.get("end_time")
    if not start or not end:
        return "Invalid time range", 400
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    files = get_files_in_range(start_dt, end_dt)
    if not files:
        return "No files found in range", 404
    merged_audio = merge_wav_files(files)
    buf = io.BytesIO()
    merged_audio.export(buf, format="wav")
    buf.seek(0)
    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"{clear_file_name(start) + '-' + clear_file_name(end)}.wav"
    )


@app.route("/workflow", methods=["POST"])
def workflow():
    form = request.form.to_dict()
    start = form.get("start_time")
    end = form.get("end_time")
    process_type = form.get("process_type", "summary")
    need_transcribe = form.get("need_transcribe", "on") == "on"
    step = form.get("step", "transcribe")
    transcript_from_front = form.get("transcript", None)
    custom_api_url = form.get("custom_api_url") or form.get("custom_api_url2")
    if not start or not end:
        return jsonify({"message": "Invalid time range"}), 400
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    files = get_files_in_range(start_dt, end_dt)
    if not files:
        return jsonify({"message": "No files found in range"}), 404

    import uuid
    task_id = str(uuid.uuid4())

    def workflow_task(task_id, form, files):
        # Step 1: Transcribe subtitles
        if step == "transcribe":
            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'start', 'message': 'Start processing audio'})
            if process_type == "custom_api" and not need_transcribe:
                # No transcription needed, go to next step directly
                socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'done', 'message': 'No transcription needed, go to next step directly', 'result': None})
                return
            merged_audio_group = merge_wav_files_grouped(files)
            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'merge', 'message': f'Audio grouping completed, {len(merged_audio_group)} groups in total'})
            transcripts = ""
            total_content = ""
            standard_code = ""
            with httpx.Client() as client:
                for i, audio in enumerate(merged_audio_group):
                    socketio.emit('workflow_progress', {'task_id': task_id, 'step': f'transcribe_{i+1}', 'message': f'Transcribing group {i+1}'})
                    buf = io.BytesIO()
                    audio.export(buf, format="wav")
                    buf.seek(0)
                    if i == 0:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                            tmpfile.write(buf.read())
                            tmpfile_path = tmpfile.name
                            signal = language_id.load_audio(tmpfile_path)
                            prediction = language_id.classify_batch(signal)
                            predicted_scores = prediction[0]
                            confidence_scores = prediction[1]
                            predicted_language = language_id.hparams.label_encoder.decode_torch(
                                torch.argmax(predicted_scores, dim=1)
                            )
                            standard_code = LANGUAGE_MAPPING.get(predicted_language[0], predicted_language[0].lower())
                            if torch.max(confidence_scores).item() < 0.5:
                                standard_code = "en"
                            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'lang_detect', 'message': f'Detected language: {standard_code}, confidence: {torch.max(confidence_scores).item():.4f}'})
                    if standard_code:
                        data = {
                            'max_len': "1024",
                            'language': standard_code
                        }
                    else:
                        data = {}
                    buf.seek(0)
                    file_bytes = buf.getvalue()
                    files_httpx = {"file": ("audio.wav", file_bytes, "audio/wav")}
                    response = client.post(
                        "http://35.238.174.232:9080/v1/audio/transcriptions",
                        files=files_httpx,
                        data=data,
                        timeout=120.0
                    )
                    transcript = response.json()
                    if transcript:
                        previous_duration = sum(len(merged_audio_group[j]) for j in range(i))
                        if "text" in transcript:
                            text_content = transcript["text"]
                            adjusted_lines = []
                            for line in text_content.split('\n'):
                                match = re.search(r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)', line)
                                if match:
                                    start_time = match.group(1)
                                    end_time = match.group(2)
                                    content = match.group(3)
                                    start_ms = (int(start_time.split(':')[0]) * 3600 +
                                                int(start_time.split(':')[1]) * 60 +
                                                float(start_time.split(':')[2])) * 1000
                                    end_ms = (int(end_time.split(':')[0]) * 3600 +
                                              int(end_time.split(':')[1]) * 60 +
                                              float(end_time.split(':')[2])) * 1000
                                    adjusted_start_ms = start_ms + previous_duration
                                    adjusted_end_ms = end_ms + previous_duration
                                    adjusted_start = f"{int(adjusted_start_ms / 3600000):02d}:{int((adjusted_start_ms % 3600000) / 60000):02d}:{int((adjusted_start_ms % 60000) / 1000):02d}.{int(adjusted_start_ms % 1000):03d}"
                                    adjusted_end = f"{int(adjusted_end_ms / 3600000):02d}:{int((adjusted_end_ms % 3600000) / 60000):02d}:{int((adjusted_end_ms % 60000) / 1000):02d}.{int(adjusted_end_ms % 1000):03d}"
                                    adjusted_line = f"[{adjusted_start} --> {adjusted_end}] {content}"
                                    total_content += content
                                    adjusted_lines.append(adjusted_line)
                                else:
                                    adjusted_lines.append(line)
                            transcript["text"] = '\n'.join(adjusted_lines)
                        if not transcripts:
                            transcripts = transcript["text"]
                        else:
                            transcripts += "\n" + transcript["text"]
                        socketio.emit('workflow_progress', {'task_id': task_id, 'step': f'transcribe_{i+1}_done', 'message': f'Group {i+1} transcription completed'})
                    else:
                        socketio.emit('workflow_progress', {'task_id': task_id, 'step': f'transcribe_{i+1}_fail', 'message': f'Group {i+1} transcription failed'})
            # Return transcription result
            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'done', 'message': 'Transcription completed', 'result': transcripts})
            return

        # Step 2: Processing (summary or custom API)
        elif step == "process":
            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'start', 'message': 'Start processing'})
            if process_type == "summary":
                socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'summary', 'message': 'Generating summary'})
                custom_prompt = form.get('custom_prompt')
                transcript = transcript_from_front or ""
                system_message = custom_prompt or "You are a professional meeting assistant. Please summarize the following meeting transcript. Your summary should be clear, concise, and well-structured. Follow this format:\n\n1. Title: A short and relevant title for the meeting\n\n2. Summary: A high-level summary of the discussion\n\n3. Key Discussion Points:\n\n* Bullet points covering the main topics discussed\n\n* Include any important decisions made\n\n* Note any action items and who is responsible\n\n4. Next Steps:\n\n* Clearly list the next steps or follow-up items\n\n* Include deadlines (if mentioned) and responsible persons"
                payload = json.dumps({
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": transcript
                        }
                    ]
                })
                headers = {
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + gaia_api_key
                }
                url = f"https://0xb2962131564bc854ece7b0f7c8c9a8345847abfb.gaia.domains/v1/chat/completions"
                max_retries = 3
                retry_delay = 3
                success = False
                error_message = ""
                with httpx.Client() as client:
                    for attempt in range(max_retries):
                        try:
                            response = client.post(url, headers=headers, data=payload, timeout=120.0)
                            response.raise_for_status()
                            response_data = response.json()
                            translation_data = response_data['choices'][0]['message']['content']
                            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'done', 'message': 'Summary completed', 'result': translation_data})
                            success = True
                            break
                        except Exception as e:
                            error_message = str(e)
                            time.sleep(retry_delay)
                if not success:
                    socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'done', 'message': f'Summary failed: {error_message}', 'result': error_message})
            # custom_api
            elif process_type == "custom_api":
                socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'custom_api', 'message': 'Calling custom API'})
                api_url = custom_api_url or os.getenv("CUSTOM_API_URL", "http://your-custom-api/handle")
                try:
                    if need_transcribe:
                        payload = {
                            "text": transcript_from_front or ""
                        }
                        with httpx.Client() as client:
                            resp = client.post(api_url, json=payload, timeout=120.0)
                            resp.raise_for_status()
                            result = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    else:
                        buf = io.BytesIO()
                        merged_audio = merge_wav_files(files)
                        merged_audio.export(buf, format="wav")
                        buf.seek(0)
                        files_httpx = {"file": ("audio.wav", buf.getvalue(), "audio/wav")}
                        with httpx.Client() as client:
                            resp = client.post(api_url, files=files_httpx, timeout=120.0)
                            resp.raise_for_status()
                            result = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'done', 'message': 'API processing completed', 'result': result})
                except Exception as e:
                    socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'done', 'message': f'API call failed: {e}', 'result': str(e)})

    socketio.start_background_task(workflow_task, task_id, form, files)
    return jsonify({'task_id': task_id})


if __name__ == "__main__":
    socketio.run(app, debug=True)
