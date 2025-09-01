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
# 添加 pyannote.audio 导入
from pyannote.audio import Pipeline

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
                "api_key": "",
                "model": "",
                "prompt": ""
            },
            "workflows": []
        }


def get_whisper_api_config():
    """获取当前的 Whisper API 配置"""
    config = get_workflows_config()
    return (
        config.get("whisper_api", {}).get("url", "https://whisper.gaia.domains/v1/audio/transcriptions"),
        config.get("whisper_api", {}).get("api_key", ""),
        config.get("whisper_api", {}).get("model", ""),
        config.get("whisper_api", {}).get("prompt", "")
    )


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
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")  # 改为threading，兼容requests/httpx

AUDIO_DIR = os.getenv("AUDIO_DIR")

def parse_timestamp_from_filename(filename):
    try:
        # match = re.search(r'recording_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+\+\d{2}_\d{2})', filename)
        match = re.search(os.getenv("AUDIO_MATCH"), filename)
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


def get_folders_with_audio():
    """遍历AUDIO_DIR下的所有文件夹，返回包含录音文件的文件夹及其时间范围"""
    result = []

    # 获取AUDIO_DIR下的所有直接子文件夹
    for item in os.listdir(AUDIO_DIR):
        folder_path = os.path.join(AUDIO_DIR, item)

        # 如果不是文件夹，跳过
        if not os.path.isdir(folder_path):
            continue

        # 查找文件夹中的录音文件
        wav_files = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.endswith(".wav") and fname.startswith("recording_"):
                    ts = parse_timestamp_from_filename(fname)
                    if ts:
                        wav_files.append((ts, os.path.join(root, fname)))

        # 如果文件夹中有录音文件
        if wav_files:
            # 按时间戳排序
            wav_files.sort()

            # 获取最早和最晚的时间戳
            earliest_time = wav_files[0][0]
            latest_time = wav_files[-1][0]

            # 计算持续时间（分钟）
            duration_minutes = (latest_time - earliest_time).total_seconds() / 60

            # 添加到结果列表
            result.append({
                "folder_name": item,
                "folder_path": os.path.relpath(folder_path, AUDIO_DIR),
                "start_time": earliest_time.isoformat(),
                "end_time": latest_time.isoformat(),
                "duration_minutes": round(duration_minutes, 1),
                "file_count": len(wav_files)
            })

    # 按开始时间排序
    result.sort(key=lambda x: x["start_time"])

    return result


def get_files_in_range(start_time, end_time, folder_path=None):
    files = []

    # 如果指定了文件夹路径，只在该路径下查找
    if folder_path:
        search_path = os.path.join(AUDIO_DIR, folder_path)
        # print(search_path)
        for root, _, fnames in os.walk(search_path):
            for fname in fnames:
                if fname.endswith(".wav") and fname.startswith("recording_"):
                    ts = parse_timestamp_from_filename(fname)
                    if ts and start_time <= ts <= end_time:
                        files.append((ts, os.path.relpath(os.path.join(root, fname), AUDIO_DIR)))
    else:
        # 原有逻辑，在整个AUDIO_DIR下查找
        for root, _, fnames in os.walk(AUDIO_DIR):
            for fname in fnames:
                if fname.endswith(".wav") and fname.startswith("recording_"):
                    ts = parse_timestamp_from_filename(fname)
                    if ts and start_time <= ts <= end_time:
                        files.append((ts, os.path.relpath(os.path.join(root, fname), AUDIO_DIR)))

    files.sort()
    return [fname for ts, fname in files]


def merge_wav_files_grouped(file_list, max_duration=30):
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
    # print("Original filename:", filename)
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


@app.route("/get_folders", methods=["GET"])
def get_folders():
    """返回包含录音文件的文件夹���表及其时间范围"""
    folders = get_folders_with_audio()
    return jsonify({"folders": folders})


@app.route("/download", methods=["POST"])
def download():
    start = request.form.get("start_time")
    end = request.form.get("end_time")
    folder_path = request.form.get("folder_path", "")  # 获取可选的文件夹路径参数

    if not start or not end:
        return jsonify({"message": "Invalid time range"}), 400
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    # 使用可选的folder_path参数调用get_files_in_range
    files = get_files_in_range(start_dt, end_dt, folder_path)

    if not files:
        return jsonify({"message": "No files found in range"}), 404
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


@app.route("/get_whisper_models", methods=["POST"])
def get_whisper_models():
    """获取可用的Whisper模型列表"""
    try:
        data = request.get_json()
        # print(data)
        whisper_url = data.get("url")
        whisper_api_key = data.get("api_key", "")
        # 从API URL中提取基础URL
        base_url_parts = whisper_url.split("/v1/")
        if len(base_url_parts) > 1:
            base_url = base_url_parts[0]
            models_url = f"{base_url}/v1/models"
        else:
            # 如果URL结构不是预期的，尝试猜测
            models_url = whisper_url.rsplit("/", 1)[0].replace("audio/transcriptions", "models")

        print(f"Fetching Whisper models from: {models_url}")
        headers = {}
        if whisper_api_key:
            headers['Authorization'] = f'Bearer {whisper_api_key}'

        with httpx.Client() as client:
            response = client.get(models_url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                models_data = response.json()
                print(f"Received models data: {models_data}")
                # 过滤出whisper模型
                whisper_models = [model['id'] for model in models_data.get('data', [])]
                return jsonify({"models": whisper_models})
            else:
                return jsonify(
                    {"models": [], "error": f"Failed to get models: {response.status_code}"}), response.status_code
    except Exception as e:
        return jsonify({"models": [], "error": f"Error fetching models: {str(e)}"}), 500


@app.route("/get_api_models", methods=["POST"])
def get_api_models():
    """获取API提供的模型列表"""
    try:
        data = request.get_json()
        api_url = data.get("url")
        api_key = data.get("api_key", "")

        # 从API URL中提取基础URL
        base_url_parts = api_url.split("/v1/")
        if len(base_url_parts) > 1:
            base_url = base_url_parts[0]
            models_url = f"{base_url}/v1/models"
        else:
            # 如果URL结构不是预期的，尝试猜测
            models_url = api_url.rsplit("/", 1)[0] + "/models"

        print(f"Fetching API models from: {models_url}")
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        with httpx.Client() as client:
            response = client.get(models_url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                models_data = response.json()
                print(f"Received models data: {models_data}")
                # 尝试提取模型列表
                models = []

                # 常见的API响应结构处理
                if 'data' in models_data and isinstance(models_data['data'], list):
                    # OpenAI类似格式
                    models = [model['id'] for model in models_data['data']]
                elif 'models' in models_data and isinstance(models_data['models'], list):
                    # 另一种常见格式
                    models = [model['id'] if isinstance(model, dict) and 'id' in model else model
                              for model in models_data['models']]
                elif isinstance(models_data, list):
                    # 直接返回模型列表的情况
                    models = [model['id'] if isinstance(model, dict) and 'id' in model else model
                              for model in models_data]

                return jsonify({"models": models})
            else:
                return jsonify(
                    {"models": [], "error": f"Failed to get models: {response.status_code}"}), response.status_code
    except Exception as e:
        return jsonify({"models": [], "error": f"Error fetching models: {str(e)}"}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """处理音频转文字的接口，使用WebSocket通知进度"""
    start = request.form.get("start_time")
    end = request.form.get("end_time")
    selected_language = request.form.get("language", "auto")
    folder_path = request.form.get("folder_path", "")  # 获取可选的文件夹路径参数

    if not start or not end:
        return jsonify({"message": "Invalid time range"}), 400
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    # 使用可选的folder_path参数调用get_files_in_range
    files = get_files_in_range(start_dt, end_dt, folder_path)

    if not files:
        return jsonify({"message": "No files found in range"}), 404

    import uuid
    task_id = str(uuid.uuid4())

    def transcribe_task(task_id, files, selected_language):
        socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'start', 'message': 'Starting audio processing'})

        merged_audio_group = merge_wav_files_grouped(files)
        socketio.emit('workflow_progress',
                      {'task_id': task_id, 'step': 'merge', 'message': f'Audio grouping completed, with a total of {len(merged_audio_group)} groups'})

        # 创建一个完整的合并音频用于说话人分区
        full_audio = None
        for audio in merged_audio_group:
            if full_audio is None:
                full_audio = audio
            else:
                full_audio += audio

        # 创建临时文件保存完整的音频用于说话人分区
        full_audio_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            full_audio.export(tmpfile, format="wav")
            full_audio_path = tmpfile.name

        socketio.emit('workflow_progress',
                      {'task_id': task_id, 'step': 'diarization_prepare', 'message': 'Preparing for speaker diarization'})

        transcripts = ""
        total_content = ""
        standard_code = ""

        # 如果用户选择了特定语言（不是自动检测），直接使用该语言
        if selected_language != "auto":
            standard_code = selected_language
            socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'lang_select',
                                                'message': f'Using the user-selected language: {standard_code}'})

        # 首先完成所有音频段的转录，不进行说话人分区
        with httpx.Client() as client:
            for i, audio in enumerate(merged_audio_group):
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': f'transcribe_{i + 1}', 'message': f'Transcribing group {i + 1}'})
                buf = io.BytesIO()
                audio.export(buf, format="wav")
                buf.seek(0)

                # 创建临时文件用于语言检测
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmpfile.write(buf.read())
                    tmpfile_path = tmpfile.name
                    buf.seek(0)

                    # 只在自动检测模式下检测第一组音频的语言
                    if i == 0 and selected_language == "auto":
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
                        socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'lang_detect',
                                                            'message': f'Detected language: {standard_code}, confidence: {torch.max(confidence_scores).item():.4f}'})

                    # 使用确定的语言代码进行转录
                    if standard_code:
                        data = {
                            'max_len': "1024",
                            'language': standard_code
                        }
                    else:
                        data = {}

                    # 动态获取 Whisper API 配置
                    whisper_url, whisper_api_key, whisper_model, whisper_prompt = get_whisper_api_config()
                    headers = {}
                    if whisper_api_key:
                        headers['Authorization'] = f'Bearer {whisper_api_key}'

                    # 如果设置了模型，添加到请求数据中
                    if whisper_model:
                        data['model'] = whisper_model

                    # 如果设置了prompt，添加到请求数据中
                    if whisper_prompt:
                        data['prompt'] = whisper_prompt

                    # 添加重试机制，最多尝试3次
                    max_retries = 3
                    retry_delay = 2  # 初始延迟2秒
                    success = False
                    transcript = None

                    for attempt in range(1, max_retries + 1):
                        try:
                            buf.seek(0)
                            file_bytes = buf.getvalue()
                            files_httpx = {"file": ("audio.wav", file_bytes, "audio/wav")}

                            response = client.post(
                                whisper_url,
                                files=files_httpx,
                                data=data,
                                headers=headers,
                                timeout=120.0
                            )

                            if response.status_code == 200:
                                transcript = response.json()
                                success = True
                                break
                            else:
                                socketio.emit('workflow_progress', {
                                    'task_id': task_id,
                                    'step': f'transcribe_{i + 1}_retry',
                                    'message': f'Attempt {attempt}/{max_retries} failed with status code {response.status_code}. Retrying...'
                                })
                        except Exception as e:
                            socketio.emit('workflow_progress', {
                                'task_id': task_id,
                                'step': f'transcribe_{i + 1}_retry',
                                'message': f'Attempt {attempt}/{max_retries} failed: {str(e)}. Retrying...'
                            })

                        # 如果不是最后一次尝试，则等待后重试
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避

                    # 处理转录结果或失败情况
                    previous_duration = sum(len(merged_audio_group[j]) for j in range(i))

                    if success and transcript:
                        # 处理成功获取的转录结果
                        if "text" in transcript:
                            text_content = transcript["text"]
                            adjusted_lines = []
                            for line in text_content.split('\n'):
                                match = re.search(
                                    r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)', line)
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
                        socketio.emit('workflow_progress', {'task_id': task_id, 'step': f'transcribe_{i + 1}_done',
                                                            'message': f'Group {i + 1} transcription completed'})

                    else:
                        # 处理转录失败的情况 - 添加空的时间段标记
                        audio_duration_ms = len(audio)
                        start_ms = previous_duration
                        end_ms = start_ms + audio_duration_ms

                        adjusted_start = f"{int(start_ms / 3600000):02d}:{int((start_ms % 3600000) / 60000):02d}:{int((start_ms % 60000) / 1000):02d}.{int(start_ms % 1000):03d}"
                        adjusted_end = f"{int(end_ms / 3600000):02d}:{int((end_ms % 3600000) / 60000):02d}:{int((end_ms % 60000) / 1000):02d}.{int(end_ms % 1000):03d}"

                        empty_transcript = f"[{adjusted_start} --> {adjusted_end}] "

                        if not transcripts:
                            transcripts = empty_transcript
                        else:
                            transcripts += "\n" + empty_transcript

                        socketio.emit('workflow_progress', {
                            'task_id': task_id,
                            'step': f'transcribe_{i + 1}_skip',
                            'message': f'Transcription for group {i + 1} failed. After {max_retries} attempts, it has been abandoned. An empty segment will be added, and processing will continue'
                        })

                # 删除临时文件
                try:
                    os.unlink(tmpfile_path)
                except:
                    pass

        # 所有转录完成后，对整个音频进行一次性说话人分区
        socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'diarization', 'message': 'Performing speaker diarization'})

        try:
            # 处理说话人分区
            diarization = process_diarization(full_audio_path)
            if diarization:
                # 将说话人信息添加到完整的转录文本中
                transcripts = assign_speakers_to_transcript(transcripts, diarization)
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'diarization_done', 'message': 'Speaker diarization completed'})
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'done', 'message': 'Transcription completed',
                               'result': transcripts})
            else:
                socketio.emit('workflow_progress', {'task_id': task_id, 'step': 'diarization_error',
                                                    'message': 'Speaker diarization failed. Processing will continue without speaker information'})
        except Exception as e:
            socketio.emit('workflow_progress',
                          {'task_id': task_id, 'step': 'diarization_error', 'message': f'Speaker diarization error: {str(e)}'})
        finally:
            # 删除临时文件
            try:
                if full_audio_path:
                    os.unlink(full_audio_path)
            except Exception as e:
                print(f"清理临时文件出错: {str(e)}")

    socketio.start_background_task(transcribe_task, task_id, files, selected_language)
    return jsonify({'task_id': task_id})


@app.route("/process_workflow", methods=["POST"])
def process_workflow():
    """处理工作流的接口，同步返回结果"""
    start = request.form.get("start_time")
    end = request.form.get("end_time")
    selected_workflow = request.form.get("selected_workflow")
    transcript = request.form.get("transcript", "")
    custom_api_url = request.form.get("text_api_url") or request.form.get("audio_api_url")
    custom_api_key = request.form.get("text_api_key") or request.form.get("audio_api_key")
    custom_prompt = request.form.get("custom_prompt", "")
    input_type = request.form.get("input_type", "text")  # 默认为文本处理
    folder_path = request.form.get("folder_path", "")  # 获取可选的文件夹路径参数

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

    # 使用可选的folder_path参数调用get_files_in_range
    files = get_files_in_range(start_dt, end_dt, folder_path)

    if not files and input_type == "audio":
        return jsonify({"message": "No files found in range"}), 404

    # 提取工作流配置
    api_url = workflow_config.get("api_endpoint") or custom_api_url
    api_model = workflow_config.get("api_model", "")

    # 决定使用哪个API Key
    # 如果是自定义API工作流并且用户提供了API Key，则使用用户的
    # 否则使用工作流配置中的API Key，或者默认的gaia_api_key
    if selected_workflow and "Custom API" in selected_workflow and custom_api_key:
        api_key = custom_api_key
    else:
        api_key = workflow_config.get("api_key")

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

                # 如果有指定模型，添加到请求中
                if api_model:
                    payload["model"] = api_model

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
                if api_model:
                    payload["model"] = api_model

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
            data = {}

            # 如果有指定模型，添加到请求中
            if api_model:
                data["model"] = api_model

            # 如果有API Key，添加到请求头中
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            with httpx.Client() as client:
                resp = client.post(api_url, files=files_httpx, data=data, headers=headers, timeout=120.0)
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


@app.route("/chat", methods=["POST"])
def chat():
    """处理基于转录结果的后续对话"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "没有收到数据"}), 400

        # 获取用户消息和历史记录
        user_message = data.get("message", "")
        message_history = data.get("history", [])
        workflow_result = data.get("workflow_result", "")
        transcript = data.get("transcript", "")
        selected_workflow = data.get("selected_workflow", "")

        # 获取选定的工作流配置
        workflow_config = None
        if selected_workflow:
            for workflow in get_workflows_config().get("workflows", []):
                if workflow.get("name") == selected_workflow:
                    workflow_config = workflow
                    break

        # 使用默认的第一个文本工作流作为聊天API
        if not workflow_config:
            for workflow in get_workflows_config().get("workflows", []):
                if workflow.get("input_type") == "text":
                    workflow_config = workflow
                    break

        if not workflow_config:
            return jsonify({"error": "No suitable API configuration found"}), 400

        # 提取API配置
        api_url = workflow_config.get("api_endpoint", "")
        api_model = workflow_config.get("api_model", "")
        api_key = workflow_config.get("api_key")

        if not api_url:
            return jsonify({"error": "API endpoint not configured"}), 400

        # 构建系统消息，提供上下文
        system_message = "You are an AI assistant. The following is a transcript and analysis of a meeting; please answer the user's questions based on this information."

        # 如果有转录和处理结果，添加到系统消息中
        context = ""
        if transcript:
            context += f"\n\nMeeting transcript:\n{transcript}"
        if workflow_result:
            context += f"\n\nAnalysis results:\n{workflow_result}"

        if context:
            system_message += context

        # 构建消息历史
        messages = [{"role": "system", "content": system_message}]

        # 只取最新的几轮对话，避免超出token限制
        recent_messages = message_history[-10:] if len(message_history) > 10 else message_history
        messages.extend(recent_messages)

        # 准备请求负载
        payload = {
            "messages": messages
        }

        # 如果有指定模型，添加到请求中
        if api_model:
            payload["model"] = api_model

        # 准备请求头
        headers = {
            'Content-Type': 'application/json'
        }

        # 添加API密钥（如果有）
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        # 发送请求
        with httpx.Client() as client:
            response = client.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=60.0
            )

            # 检查响应
            if response.status_code != 200:
                print(f"API返回错误: {response.status_code}, {response.text}")
                return jsonify({"error": f"API returned an error: {response.status_code}"}), 500

            try:
                result = response.json()
                # 处理不同格式的API响应
                if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                    # OpenAI格式响应
                    ai_response = result["choices"][0]["message"]["content"]
                elif "response" in result:
                    # 自定义格式1
                    ai_response = result["response"]
                elif "content" in result:
                    # 自定义格式2
                    ai_response = result["content"]
                elif "text" in result:
                    # 自定义格式3
                    ai_response = result["text"]
                else:
                    # 尝试将整个响应作为文本返回
                    ai_response = str(result)

                return jsonify({"response": ai_response})
            except Exception as e:
                print(f"处理API响应时出错: {str(e)}")
                return jsonify({"error": f"Error occurred while processing the API response: {str(e)}"}), 500

    except Exception as e:
        print(f"聊天处理出错: {str(e)}")
        return jsonify({"error": f"Error occurred while processing the request: {str(e)}"}), 500


# 保留原接口用于向后兼容，但功能简化
@app.route("/workflow", methods=["POST"])
def workflow():
    step = request.form.get("step", "transcribe")

    if step == "transcribe":
        return transcribe()
    else:
        return process_workflow()


# 在现有路由之后添加
@app.route('/download_segment', methods=['POST'])
def download_segment():
    """下载特定时间段的音频片段"""
    try:
        start_time = request.form.get('start_time')
        end_time = request.form.get('end_time')
        folder_path = request.form.get('folder_path', '')
        segment_start = float(request.form.get('segment_start', 0))
        segment_end = float(request.form.get('segment_end', 0))

        if not start_time or not end_time or segment_start >= segment_end:
            return jsonify({'message': 'Invalid time parameters'}), 400

        # 将ISO时间格式转换为datetime对象
        start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_datetime = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

        # 获取符合时间范围的音频文件列表
        audio_files = get_files_in_range(start_datetime, end_datetime, folder_path)

        if not audio_files:
            return jsonify({'message': 'No audio files found in the specified time range'}), 404

        # 合并音频文件
        merged_audio = merge_wav_files(audio_files)

        # 提取指定时间段的片段
        total_duration = len(merged_audio) / 1000  # 转换为秒
        segment_start = min(segment_start, total_duration)
        segment_end = min(segment_end, total_duration)

        # 转换为毫秒
        segment_start_ms = int(segment_start * 1000)
        segment_end_ms = int(segment_end * 1000)

        # 提取片段
        audio_segment = merged_audio[segment_start_ms:segment_end_ms]

        # 导出为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_segment.export(temp_file.name, format='wav')
        temp_file.close()

        # 发送文件
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f"segment_{segment_start:.1f}-{segment_end:.1f}.wav",
            mimetype='audio/wav'
        )

    except Exception as e:
        app.logger.error(f"Error in download_segment: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500


# 初始化说话人分区模型
def initialize_speaker_diarization():
    """初始化pyannote说话人分区模型"""
    try:
        # 从环境变量或配置文件获取token
        hf_token = os.getenv("HF_TOKEN")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token)

        # 检测是否有GPU可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        return pipeline
    except Exception as e:
        print(f"初始化说话人分区模型失败: {str(e)}")
        return None


# 全局变量，延迟加载
speaker_diarization_pipeline = None


def get_speaker_diarization():
    """获取说话人分区模型，如果未初始化则进行初始化"""
    global speaker_diarization_pipeline
    if speaker_diarization_pipeline is None:
        speaker_diarization_pipeline = initialize_speaker_diarization()
    return speaker_diarization_pipeline


def process_diarization(audio_path):
    """处理音频文件，返回说话人分区结果"""
    pipeline = get_speaker_diarization()
    if not pipeline:
        return None

    try:
        diarization = pipeline(audio_path)
        return diarization
    except Exception as e:
        print(f"说话人分区处理失败: {str(e)}")
        return None


def assign_speakers_to_transcript(transcript, diarization):
    """将说话人信息分配给转录文本，并根据说话人变化和文本内容智能分段"""
    if not diarization:
        return transcript

    # 创建说话人ID映射，将原始ID映射到新的编号系统
    original_speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        original_speakers.add(speaker)

    # 创建新的编号系统，从01开始
    speaker_mapping = {}
    for num, speaker in enumerate(sorted(original_speakers)):
        speaker_mapping[speaker] = f"SPEAKER_{num + 1:02d}"
        print(f"Mapping speaker '{speaker}' to '{speaker_mapping[speaker]}'")

    # 解析转录文本中的时间戳
    transcript_lines = []
    for line in transcript.split('\n'):
        match = re.search(r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)', line)
        if not match:
            transcript_lines.append(line)
            continue

        start_time = match.group(1)
        end_time = match.group(2)
        content = match.group(3)

        # 将时间戳格式转换为秒
        start_seconds = (int(start_time.split(':')[0]) * 3600 +
                         int(start_time.split(':')[1]) * 60 +
                         float(start_time.split(':')[2]))
        end_seconds = (int(end_time.split(':')[0]) * 3600 +
                       int(end_time.split(':')[1]) * 60 +
                       float(end_time.split(':')[2]))

        # 收集该时间段内的所有说话人片段
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # print(f"Turn: {turn.start:.2f}-{turn.end:.2f}, Speaker: {speaker}")
            # 检查说话人分区段与转录段是否有重叠
            if max(turn.start, start_seconds) < min(turn.end, end_seconds):
                # 计算重叠部分的起止时间
                overlap_start = max(turn.start, start_seconds)
                overlap_end = min(turn.end, end_seconds)
                # 使用映射后的说话人ID
                mapped_speaker = speaker_mapping.get(speaker, "SPEAKER_00")
                # print(f"mapped_speaker: {mapped_speaker}, overlap: {overlap_start:.2f}-{overlap_end:.2f}")
                speaker_segments.append((overlap_start, overlap_end, mapped_speaker))

        # 对说话人片段按时间排序
        speaker_segments.sort()

        # 如果没有找到说话人，使用SPEAKER_00
        if not speaker_segments:
            new_line = f"[{start_time} --> {end_time}] [SPEAKER_00] {content}"
            transcript_lines.append(new_line)
            continue

        # 尝试根据说话人变化和文本内容分割该段
        segments = split_transcript_by_speakers(content, start_seconds, end_seconds, speaker_segments)

        # 将分割后的片段添加到转录结果中
        for seg_start, seg_end, seg_speaker, seg_content in segments:
            # 将秒转换回时间戳格式
            seg_start_time = f"{int(seg_start / 3600):02d}:{int((seg_start % 3600) / 60):02d}:{seg_start % 60:06.3f}"
            seg_end_time = f"{int(seg_end / 3600):02d}:{int((seg_end % 3600) / 60):02d}:{seg_end % 60:06.3f}"

            new_line = f"[{seg_start_time} --> {seg_end_time}] [{seg_speaker}] {seg_content}"
            transcript_lines.append(new_line)

    return '\n'.join(transcript_lines)


def split_transcript_by_speakers(text, start_time, end_time, speaker_segments):
    """
    根据说话人变化和文本内容分割转录文本

    Args:
        text: 转录文本内容
        start_time: 转录段开始时间（秒）
        end_time: 转录段结束时间（秒）
        speaker_segments: 说话人时间段列表，每项为 (start, end, speaker)

    Returns:
        分割后的片段列表，每项为 (start, end, speaker, content)
    """
    # 如果文本很短或没有标点，不进行分割
    if len(text) < 10 or not re.search(r'[.!?。！？]', text):
        # 找出时间最长的说话人
        main_speaker = find_main_speaker_from_segments(speaker_segments, start_time, end_time)
        return [(start_time, end_time, main_speaker, text)]

    # 查找可能的分割点（句号、问号、感叹号等）
    split_points = []
    for match in re.finditer(r'[.!?。！？]', text):
        # 标点符号位置
        pos = match.start()
        # 估计标点在整个时间段中的相对位置
        relative_pos = pos / len(text)
        # 计算估计的时间点
        estimated_time = start_time + relative_pos * (end_time - start_time)
        split_points.append((pos, estimated_time))

    # 如果没有找到分割点，使用主要说话人
    if not split_points:
        main_speaker = find_main_speaker_from_segments(speaker_segments, last_time, end_time)
        return [(start_time, end_time, main_speaker, text)]

    # 找到最适合的分割点
    segments = []
    last_pos = 0
    last_time = start_time

    for split_pos, split_time in split_points:
        # 查找最接近这个时间点的说话人变化
        best_speaker = None
        min_diff = float('inf')

        for seg_start, seg_end, speaker in speaker_segments:
            # 说话人片段覆盖这个分割点之前的部分
            if seg_start <= split_time and last_time >= seg_start:
                diff = abs(seg_end - split_time)
                if diff < min_diff:
                    min_diff = diff
                    best_speaker = speaker

        # 如果没有找到适合的说话人，使用时间最长的说话人
        if not best_speaker:
            best_speaker = find_main_speaker_from_segments(speaker_segments, last_time, split_time)

        # 分割点加1是为了包含标点符号
        segment_text = text[last_pos:split_pos + 1].strip()
        if segment_text:  # 确保文本不为空
            segments.append((last_time, split_time, best_speaker, segment_text))

        last_pos = split_pos + 1
        last_time = split_time

    # 处理最后一个片段
    if last_pos < len(text):
        last_segment_text = text[last_pos:].strip()
        if last_segment_text:
            best_speaker = find_main_speaker_from_segments(speaker_segments, last_time, end_time)
            segments.append((last_time, end_time, best_speaker, last_segment_text))

    return segments


def find_main_speaker_from_segments(speaker_segments, start_time, end_time):
    """从说话人片段中找出指定时间段内的主要说话人"""
    speakers = {}

    for seg_start, seg_end, speaker in speaker_segments:
        # 检查说话人片段与目标时间段是否有重叠
        if max(seg_start, start_time) < min(seg_end, end_time):
            # 计算重叠时间
            overlap = min(seg_end, end_time) - max(seg_start, start_time)
            speakers[speaker] = speakers.get(speaker, 0) + overlap

    # 如果没有找到说话人，返回默认值
    if not speakers:
        return "SPEAKER_00"

    # 返回占比最高的说话人
    return max(speakers, key=speakers.get)


if __name__ == "__main__":
    socketio.run(app, debug=True)

