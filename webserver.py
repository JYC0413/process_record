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

# 加载工作流配置文件
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "workflows_config.json")

def get_workflows_config():
    """动态获取最新的工作流配置"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load workflows config: {e}")
        return {
            "whisper_api": {
                "url": "https://whisper.gaia.domains/v1/audio/transcriptions",
                "api_key": "gaia-OTBiYjlmZDEtNTc3OS00MjI5LWI0NDgtZDIxNTNmYjEwZDRj-IYuaA5AxGFTywJWq"
            },
            "workflows": []
        }

def get_whisper_api_config():
    """获取当前的 Whisper API 配置"""
    config = get_workflows_config()
    return (
        config.get("whisper_api", {}).get("url", "https://whisper.gaia.domains/v1/audio/transcriptions"),
        config.get("whisper_api", {}).get("api_key", "gaia-OTBiYjlmZDEtNTc3OS00MjI5LWI0NDgtZDIxNTNmYjEwZDRj-IYuaA5AxGFTywJWq")
    )

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

AUDIO_DIR = r"../../michael/echokit/8090/record"


def parse_timestamp_from_filename(filename):
    try:
        match = re.search(r'recording_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})', filename)
        if match:
            time_str = match.group(1)

            dt = datetime.fromisoformat(time_str.replace("_", ":"))

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
    return render_template("selectRecord.html", workflows=get_workflows_config().get("workflows", []))

@app.route("/config", methods=["GET"])
def config_page():
    """配置页面路由"""
    return render_template("config.html")

@app.route("/get_workflows", methods=["GET"])
def get_workflows():
    return jsonify({"workflows": get_workflows_config().get("workflows", [])})


@app.route("/get_config", methods=["GET"])
def get_config():
    """获取当前配置信息的API端点"""
    config = get_workflows_config()
    return jsonify({
        "whisper_api": config.get("whisper_api", {}),
        "workflows_count": len(config.get("workflows", []))
    })


@app.route("/get_full_config", methods=["GET"])
def get_full_config():
    """获取当前配置信息的API端点"""
    config = get_workflows_config()
    return config


@app.route("/update_config", methods=["POST"])
def update_config():
    """更新配置的API端点"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        # 获取当前配置
        current_config = get_workflows_config()

        # 更新Whisper API配置
        if "whisper_api" in data:
            current_config["whisper_api"] = data["whisper_api"]

        # 保存更新后的配置
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "success", "message": "Configuration updated successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to update config: {str(e)}"}), 500


@app.route("/update_full_config", methods=["POST"])
def update_full_config():
    """更新配置的API端点"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        # 保存更新后的配置
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "success", "message": "Configuration updated successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to update config: {str(e)}"}), 500


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


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """处理音频转文字的接口，使用WebSocket通知进度"""
    start = request.form.get("start_time")
    end = request.form.get("end_time")

    if not start or not end:
        return jsonify({"message": "Invalid time range"}), 400
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    files = get_files_in_range(start_dt, end_dt)
    if not files:
        return jsonify({"message": "No files found in range"}), 404

    import uuid
    task_id = str(uuid.uuid4())

    def transcribe_task(task_id, files):
        socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'start', 'message': 'Start processing audio'})

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

                # 动态获取 Whisper API 配置
                whisper_url, whisper_api_key = get_whisper_api_config()
                headers = {}
                if whisper_api_key:
                    headers['Authorization'] = f'Bearer {whisper_api_key}'

                response = client.post(
                    whisper_url,
                    files=files_httpx,
                    data=data,
                    headers=headers,
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
                                adjusted_end = f"{int(adjusted_end_ms / 3600000):02d}:{int((adjusted_end_ms % 3600000) / 60000):02d}:{int((adjusted_start_ms % 60000) / 1000):02d}.{int(adjusted_start_ms % 1000):03d}"

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

    socketio.start_background_task(transcribe_task, task_id, files)
    return jsonify({'task_id': task_id})


@app.route("/process_workflow", methods=["POST"])
def process_workflow():
    """处理工作流的接口，同步返回结果"""
    start = request.form.get("start_time")
    end = request.form.get("end_time")
    selected_workflow = request.form.get("selected_workflow")
    transcript = request.form.get("transcript", "")
    custom_api_url = request.form.get("custom_api_url") or request.form.get("text_api_url") or request.form.get("audio_api_url")
    custom_api_key = request.form.get("text_api_key") or request.form.get("audio_api_key")
    custom_prompt = request.form.get("custom_prompt", "")
    input_type = request.form.get("input_type", "text")  # 默认为文本处理

    # 获取选定的工作流配置
    workflow_config = None
    if selected_workflow:
        for workflow in get_workflows_config().get("workflows", []):
            if workflow.get("name") == selected_workflow:
                workflow_config = workflow
                break

    if not workflow_config:
        return jsonify({"message": "Invalid workflow selected"}), 400

    if not start or not end:
        return jsonify({"message": "Invalid time range"}), 400

    print(f"Processing workflow: {selected_workflow}, input_type: {input_type}, config: {workflow_config}")

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    files = get_files_in_range(start_dt, end_dt)
    if not files and input_type == "audio":
        return jsonify({"message": "No files found in range"}), 404

    # 提取工作流配置
    api_url = workflow_config.get("api_endpoint") or custom_api_url

    # 决定使用哪个API Key
    # 如果是自定义API工作流并且用户提供了API Key，则使用用户的
    # 否则使用工作流配置中的API Key，或者默认的gaia_api_key
    if selected_workflow and "Custom API" in selected_workflow and custom_api_key:
        api_key = custom_api_key
    else:
        api_key = workflow_config.get("api_key") or gaia_api_key

    workflow_input_type = workflow_config.get("input_type", "text")
    workflow_custom_prompt = workflow_config.get("custom_prompt", "")

    # 使用前端传来的自定义提示词，如果没有则使用工作流配置中的提示词
    final_custom_prompt = custom_prompt or workflow_custom_prompt

    try:
        # 文本处理工作流
        if workflow_input_type == "text":
            # 使用 Gaia AI 处理
            if api_url:

                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": final_custom_prompt
                        },
                        {
                            "role": "user",
                            "content": transcript
                        }
                    ]
                }
                headers = {
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + api_key
                }

                with httpx.Client() as client:
                    response = client.post(api_url, headers=headers, json=payload, timeout=120.0)
                    response.raise_for_status()
                    response_data = response.json()
                    print(f"Response from API: {response_data}")
                    result = response_data['choices'][0]['message']['content']
                    return jsonify({"result": result})

            # 使用自定义 API 处理文本
            elif api_url:
                payload = {"text": transcript}
                headers = {}

                # 如果有API Key，添加到请求头中
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'

                with httpx.Client() as client:
                    resp = client.post(api_url, json=payload, headers=headers, timeout=120.0)
                    resp.raise_for_status()
                    if resp.headers.get("content-type", "").startswith("application/json"):
                        return resp.json()
                    else:
                        return jsonify({"result": resp.text})
            else:
                return jsonify({"message": "No API endpoint configured"}), 400

        # 音频处理工作流
        elif workflow_input_type == "audio":
            if not api_url:
                return jsonify({"message": "No API endpoint configured"}), 400

            buf = io.BytesIO()
            merged_audio = merge_wav_files(files)
            merged_audio.export(buf, format="wav")
            buf.seek(0)
            files_httpx = {"file": ("audio.wav", buf.getvalue(), "audio/wav")}

            headers = {}

            # 如果有API Key，添加到请求头中
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            with httpx.Client() as client:
                resp = client.post(api_url, files=files_httpx, headers=headers, timeout=120.0)
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                else:
                    return jsonify({"result": resp.text})
        else:
            return jsonify({"message": f"Unsupported input type: {workflow_input_type}"}), 400

    except Exception as e:
        print(f"Error processing workflow: {str(e)}")
        return jsonify({"message": f"Error processing workflow: {str(e)}"}), 500


# 保留原接口用于向后兼容，但功能简化
@app.route("/workflow", methods=["POST"])
def workflow():
    step = request.form.get("step", "transcribe")

    if step == "transcribe":
        return transcribe()
    else:
        return process_workflow()


if __name__ == "__main__":
    socketio.run(app, debug=True)
