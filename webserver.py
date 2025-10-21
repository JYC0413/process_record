import re
import os
import io
import time
import json
import torch
import tempfile
from google import genai
from datetime import datetime, timezone
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit
import httpx  # 替换requests

from pydub.utils import mediainfo
from pydub import AudioSegment
import noisereduce as nr
import numpy as np

from dotenv import load_dotenv

from typing import NamedTuple


class TransObj(NamedTuple):
    start: float
    end: float
    value: str


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


def transcribe_task(task_id, files, selected_language, speakers_num, folder_path):
    # 检查是否可以直接使用现有的diarization.txt文件
    if folder_path and selected_language == "auto" and speakers_num == 0:
        print("Checking for existing diarization.txt file...")
        # 构建diarization.txt的完整路径
        diarization_file_path = os.path.join(AUDIO_DIR, folder_path, "diarization.txt")
        print(f"Looking for diarization file at: {diarization_file_path}")
        # 检查文件是否存在
        if os.path.isfile(diarization_file_path):
            try:
                # 读取文件内容
                with open(diarization_file_path, 'r', encoding='utf-8') as f:
                    transcripts = f.read()
                print("Found existing diarization.txt file, using it directly.")
                print(transcripts)
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'done', 'message': 'Transcription completed',
                               'result': transcripts})
                return
            except Exception as e:
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'error',
                               'message': f'Error reading diarization.txt file: {str(e)}'})
                # 如果读取失败，继续正常流程

    client = genai.Client()

    def hhmmss_to_seconds(t: str) -> float:
        # 支持 "HH:MM:SS" 或 "HH:MM:SS.xxx"
        try:
            parts = t.split(':')
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        except:
            return 0.0

    socketio.emit('workflow_progress',
                  {'task_id': task_id, 'step': 'start', 'message': 'Starting audio processing'})

    merged_audio_group = merge_wav_files_grouped(files, 900)  # 每组不超过900秒（15分钟）
    socketio.emit('workflow_progress',
                  {'task_id': task_id, 'step': 'merge',
                   'message': f'Audio grouping completed, with a total of {len(merged_audio_group)} groups'})

    # 全局结果和参考音频引用
    all_transcriptions = []
    global_speaker_map = []
    speaker_ref_files = {}  # speaker -> uploaded file object (genai)
    transcription_schema = {
        "type": "object",
        "properties": {
            "transcriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                        "speaker": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["start_time", "end_time", "speaker", "text"]
                }
            },
            "speaker_map": {
                "type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "sound_characteristics": {"type": "string"}
                    },
                    "required": ["speaker", "sound_characteristics"]
                }
            },
        },
        "required": ["transcriptions","speaker_map"]
    }

    try:
        for i, audio in enumerate(merged_audio_group):
            socketio.emit('workflow_progress',
                          {'task_id': task_id, 'step': f'transcribe_{i + 1}',
                           'message': f'Transcribing group {i + 1} / {len(merged_audio_group)}'})

            # 导出当前组音频为临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                audio.export(tmpfile, format="wav")
                tmpfile_path = tmpfile.name

            try:
                combined_file = client.files.upload(file=tmpfile_path)
            except Exception as e:
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'error',
                               'message': f'Failed to upload combined audio: {str(e)}'})
                return

            # 构造 prompt
            if i == 0:
                prompt = ('请将我上传的音频文件转录为文本，并精确标注每句话的起始时间、结束时间以及说话人。**请不要翻译，保持语音中的原语言输出转录文本。**'
                          '输出格式必须是 JSON，包含 "transcriptions"（数组） 和 "speaker_map"（说话人映射和特征）。'
                          '时间格式为 "HH:MM:SS.xxx"，speaker 请用 "Speaker 1" 等标识。')
                contents = [prompt, combined_file]
            else:
                # 构造包含参考说话人文件的 prompt
                ref_lines = []
                ref_objs = []
                for sp, uploaded in speaker_ref_files.items():
                    ref_lines.append(f"- **{sp}：** 使用文件 {uploaded.name} 的声音特征。")
                    ref_objs.append(uploaded)
                refs_text = "\n".join(ref_lines) if ref_lines else ""
                prompt = (f'**这是长音频的第 {i+1} 个片段。请严格保持说话人编号与提供的参考音频一致。**\n'
                          f'1. 要转录的音频文件： {combined_file.name}\n'
                          f'2. 说话人参考：\n{refs_text}\n'
                          '3. 如果出现与参考都不同的新说话人，请按顺序标记为 Speaker X（例如 Speaker 5）。\n'
                          '请将音频转录为文本，并精确标注每句话的起始时间、结束时间以及说话人。**请不要翻译，保持语音中的原语言输出转录文本。** 输出格式必须是 JSON：包含 "transcriptions" 数组以及 "speaker_map"。')
                contents = [prompt, combined_file] + ref_objs

            socketio.emit('workflow_progress',
                          {'task_id': task_id, 'step': f'upload_{i + 1}',
                           'message': f'Calling Gemini for group {i + 1}'})

            # 调用 Gemini
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": transcription_schema,
                    }
                )
                transcription_json_string = response.text
                data = json.loads(transcription_json_string)
            except Exception as e:
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'error',
                               'message': f'Error from Gemini on group {i+1}: {str(e)}'})
                return

            # 合并 transcriptions
            group_transcriptions = data.get('transcriptions', [])
            group_speaker_map = data.get('speaker_map', [])

            # 累计结果
            all_transcriptions.extend(group_transcriptions)

            # 如果是第一次，保存全局的 speaker_map
            if i == 0 and group_speaker_map:
                global_speaker_map = group_speaker_map

            # 为每个 speaker 生成参考音频（如果还没有）
            # 使用当前组的 transcriptions（时间相对于本组起始），从当前 audio 中裁切
            try:
                # 先收集每个 speaker 的时间片段列表，便于后续合并拼接作为参考样本
                speaker_entries = {}
                for entry in group_transcriptions:
                    sp = entry.get('speaker')
                    if not sp:
                        continue
                    st = hhmmss_to_seconds(entry.get('start_time', '00:00:00'))
                    ed = hhmmss_to_seconds(entry.get('end_time', '00:00:00'))
                    if ed <= st:
                        continue
                    speaker_entries.setdefault(sp, []).append((st, ed))

                MIN_REF_SEC = 10
                MAX_REF_SEC = 30

                for sp, intervals in speaker_entries.items():
                    if sp in speaker_ref_files:
                        continue

                    intervals.sort()
                    collected_seg = AudioSegment.silent(duration=0)
                    total_len = 0.0

                    # 依次拼接同一说话人的多个片段，直到达到最小长度或耗尽片段
                    for st, ed in intervals:
                        start_ms = int(max(0, st * 1000))
                        end_ms = int(min(len(audio), ed * 1000))
                        seg = audio[start_ms:end_ms]
                        seg_dur = len(seg) / 1000.0
                        if seg_dur <= 0:
                            continue

                        # 如果单段过长且还没收集任何样本，则截断为 MAX_REF_SEC
                        if seg_dur > MAX_REF_SEC and total_len == 0:
                            seg = seg[: int(MAX_REF_SEC * 1000)]
                            seg_dur = len(seg) / 1000.0

                        collected_seg += seg
                        total_len += seg_dur

                        if total_len >= MIN_REF_SEC:
                            break

                    # 如果仍然没有有效样本，跳过该 speaker（不会中断整个流程）
                    if total_len <= 0:
                        socketio.emit('workflow_progress',
                                      {'task_id': task_id, 'step': f'ref_skip_{sp}',
                                       'message': f'No valid reference audio found for {sp}, skipping.'})
                        continue

                    # 导出临时参考文件并上传
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_tmp:
                        ref_path = ref_tmp.name
                    collected_seg.export(ref_path, format="wav")
                    try:
                        uploaded_ref = client.files.upload(file=ref_path)
                        speaker_ref_files[sp] = uploaded_ref
                        socketio.emit('workflow_progress',
                                      {'task_id': task_id, 'step': f'ref_upload_{sp}',
                                       'message': f'Uploaded reference for {sp} as {uploaded_ref.name}'})
                    except Exception as e:
                        socketio.emit('workflow_progress',
                                      {'task_id': task_id, 'step': 'error',
                                       'message': f'Failed to upload reference for {sp}: {str(e)}'})
                        # 不中断整个流程，继续下一个
            except Exception as e:
                socketio.emit('workflow_progress',
                              {'task_id': task_id, 'step': 'error',
                               'message': f'Error creating reference audios: {str(e)}'})

            # 更新全局 speaker_map，如果后续段落提供了更完整的 speaker_map ，合并之
            if group_speaker_map:
                # 用 speaker 字段做去重合并
                known = {s['speaker'] for s in global_speaker_map}
                for s in group_speaker_map:
                    if s['speaker'] not in known:
                        global_speaker_map.append(s)
            socketio.emit('workflow_progress',
                          {'task_id': task_id, 'step': f'completed_{i + 1}',
                           'message': f'Completed group {i + 1}', 'group_count': i + 1})
            # 清理当前组临时合并文件（可选）
            try:
                os.remove(tmpfile_path)
            except:
                pass

        # 全部分组处理完毕，构造最终 JSON 字符串
        final_result = {
            "transcriptions": all_transcriptions,
            "speaker_map": global_speaker_map
        }
        transcripts = json.dumps(final_result, ensure_ascii=False, indent=2)
        socketio.emit('workflow_progress',
                      {'task_id': task_id, 'step': 'done', 'message': 'Transcription completed',
                       'result': transcripts})
    except Exception as e:
        socketio.emit('workflow_progress',
                      {'task_id': task_id, 'step': 'error',
                       'message': f'Unexpected error in transcribe_task: {str(e)}'})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """处理音频转文字的接口，使用WebSocket通知进度"""
    start = request.form.get("start_time")
    end = request.form.get("end_time")
    task_id = request.form.get("task_id")
    selected_language = request.form.get("language", "auto")
    folder_path = request.form.get("folder_path", "")  # 获取可选的文件夹路径参数
    speakers_num = int(request.form.get("speakers_num", 0))  # 获取说话人数量并转换为整数  # 获取说话人数量

    if not start or not end:
        return jsonify({"message": "Invalid time range"}), 400
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    # 使用可选的folder_path参数调用get_files_in_range
    files = get_files_in_range(start_dt, end_dt, folder_path)

    if not files:
        return jsonify({"message": "No files found in range"}), 404

    socketio.start_background_task(transcribe_task, task_id, files, selected_language, speakers_num, folder_path)
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


if __name__ == "__main__":
    socketio.run(app, port=8000, debug=True)
