import io
import os
import time
import torch
import httpx
import threading
from pydub import AudioSegment
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from webserver import get_whisper_api_config, assign_speakers_to_transcript

load_dotenv()

WATCH_DIR = os.getenv("AUDIO_DIR")  # 根目录（递归监听）
TARGET_PREFIX = "recording_"
TARGET_EXT = ".wav"
IDLE_TIMEOUT = 300  # 5分钟

# 每个子目录一个定时器
idle_timers = {}  # {folder_abs_path: threading.Timer}
idle_lock = threading.Lock()


def is_target_wav(path: str) -> bool:
    name = os.path.basename(path)
    return name.startswith(TARGET_PREFIX) and name.lower().endswith(TARGET_EXT)


def folder_key(path: str) -> str:
    """把文件路径映射到它的父目录（绝对路径，做key）"""
    return os.path.abspath(os.path.dirname(path))

def time_to_ms(time_str):
    """将时间字符串转换为毫秒"""
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

def ms_to_time(ms):
    """将毫秒转换为时间字符串"""
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def run_pyannote_pipeline(folder_abs: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 遍历该目录下所有 recording_*.wav 文件，按时间排序后合并
    wav_files = [f for f in os.listdir(folder_abs) if is_target_wav(f)]
    wav_files.sort()  # 按文件名排序，假设文件名中包含时间戳
    combined = AudioSegment.empty()

    # 处理字幕文件
    combined_subtitles = []
    accumulated_duration = 0  # 累积的音频长度，单位为毫秒

    for wf in wav_files:
        full_path = os.path.join(folder_abs, wf)
        audio = AudioSegment.from_wav(full_path)
        combined += audio

        # 处理对应的字幕文件
        txt_path = os.path.splitext(full_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                subtitles = f.readlines()

            # 调整时间戳并添加到合并列表
            for subtitle in subtitles:
                # 匹配格式如 [00:00:00.000 --> 00:00:02.920] Hello, human.
                if subtitle.startswith("[") and "-->" in subtitle and "]" in subtitle:
                    # 提取时间戳部分和内容部分
                    timestamp_part = subtitle.split("]")[0] + "]"
                    content_part = subtitle.split("]", 1)[1] if len(subtitle.split("]")) > 1 else ""

                    # 提取开始和结束时间
                    time_parts = timestamp_part.strip("[]").split("-->")
                    start_time_str = time_parts[0].strip()
                    end_time_str = time_parts[1].strip()

                    # 转换为毫秒
                    start_ms = time_to_ms(start_time_str)
                    end_ms = time_to_ms(end_time_str)

                    # 加上累积的时间
                    new_start_ms = start_ms + accumulated_duration
                    new_end_ms = end_ms + accumulated_duration

                    # 转回时间字符串
                    new_start_str = ms_to_time(new_start_ms)
                    new_end_str = ms_to_time(new_end_ms)

                    # 构建新的字幕行
                    new_subtitle = f"[{new_start_str} --> {new_end_str}]{content_part}"
                    combined_subtitles.append(new_subtitle)
                else:
                    combined_subtitles.append(subtitle)
            combined_subtitles.append("\n") # 每个文件的字幕后加个空行

        # 更新累积的音频长度
        accumulated_duration += len(audio)

    if not wav_files:
        print(f"⚠️  目录下无目标文件，跳过: {folder_abs}")
        return

    combined_path = os.path.join(folder_abs, "combined.wav")
    combined.export(combined_path, format="wav")
    print(f"✅ 合并完成: {combined_path}")

    # 保存合并后的字幕
    if combined_subtitles:
        combined_subtitle_path = os.path.join(folder_abs, "combined.txt")
        with open(combined_subtitle_path, "w", encoding="utf-8") as f:
            f.writelines(combined_subtitles)
        print(f"✅ 合并字幕完成: {combined_subtitle_path}")

        # apply pretrained pipeline
        diarization = pipeline(combined_path)

        # 输出结果到文本文件
        result_txt = os.path.join(folder_abs, "diarization.txt")
        with open(result_txt, "w", encoding="utf-8") as f:
            result = assign_speakers_to_transcript("".join(combined_subtitles), diarization)
            f.write(result)
        print(f"✅ 说话人分离结果已保存: {result_txt}")

def on_folder_idle(folder_abs: str):
    """某个子文件夹5分钟没有新的 recording_*.wav 创建时触发"""
    with idle_lock:
        # 清理已经触发的timer引用
        idle_timers.pop(folder_abs, None)
    print(f"✅ [IDLE] 目录已闲置: {folder_abs}（5分钟内无新 recording_*.wav）")
    run_pyannote_pipeline(folder_abs)


def reset_idle_timer_for_folder(folder_abs: str):
    """只重置该子目录的5分钟定时器"""
    folder_abs = os.path.abspath(folder_abs)
    with idle_lock:
        old = idle_timers.get(folder_abs)
        if old:
            old.cancel()
        t = threading.Timer(IDLE_TIMEOUT, on_folder_idle, args=(folder_abs,))
        t.daemon = True
        idle_timers[folder_abs] = t
        t.start()
    # 仅用于观察
    print(f"⏱️  重置定时器: {folder_abs} -> 5分钟")


def is_file_stable(file_path: str, probe_interval=2, tries=3) -> bool:
    """简单判断文件写入是否完成：多次探测大小是否不变"""
    last_size = -1
    for _ in range(tries):
        if not os.path.exists(file_path):
            return False
        size = os.path.getsize(file_path)
        if size == last_size and size > 0:
            return True
        last_size = size
        time.sleep(probe_interval)
    return False


def whisper_transcribe(file_path):
    whisper_url, whisper_api_key, whisper_model, whisper_prompt = get_whisper_api_config()
    headers = {}
    data = {}
    client = httpx.Client()
    if whisper_api_key:
        headers['Authorization'] = f'Bearer {whisper_api_key}'

    # 如果设置了模型，添加到请求数据中
    if whisper_model:
        data['model'] = whisper_model

    # 如果设置了prompt，添加到请求数据中
    if whisper_prompt:
        data['prompt'] = whisper_prompt

    data['language'] = "auto"

    # 添加重试机制，最多尝试3次
    max_retries = 3
    retry_delay = 2  # 初始延迟2秒
    success = False
    transcript = None

    audio = AudioSegment.from_wav(file_path)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)

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
        except Exception as e:
            print(f"⚠️  第{attempt}次尝试转录时出错: {e}")

        # 如果不是最后一次尝试，则等待后重试
        if attempt < max_retries:
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数退避

    if success and transcript:
        # 处理成功获取的转录结果
        if "text" in transcript:
            text_content = transcript["text"]
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            print(f"✅ 转录成功，保存到: {txt_path}")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)


def handle_new_file(file_path: str):
    """检测到新目标文件后的处理：重置所在目录的定时器；等待稳定后做单文件处理"""
    folder_abs = folder_key(file_path)

    # 需求2：定时器逻辑——只要有“新创建的符合要求的文件”，就重置该目录定时器
    reset_idle_timer_for_folder(folder_abs)

    # 需求1：如果你还需要对每个新文件做 whisper，可在稳定后执行
    if is_file_stable(file_path):
        print(f"✅ 文件已稳定，开始单文件处理: {file_path}")
        whisper_transcribe(file_path)  # 你自己的逻辑
    else:
        print(f"⚠️  文件未稳定，跳过单文件处理: {file_path}")


class PerFolderIdleHandler(FileSystemEventHandler):
    def _maybe_handle(self, path: str):
        if is_target_wav(path):
            print(f"🆕 检测到目标文件: {path}")
            # 有些录音程序先写到临时名，再“rename”为最终名；
            # 因此我们在 on_created 和 on_moved 的目标路径都处理
            threading.Thread(target=handle_new_file, args=(path,), daemon=True).start()

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_handle(event.src_path)

    def on_moved(self, event):
        # 处理“临时文件 -> recording_*.wav”的重命名
        if not event.is_directory:
            self._maybe_handle(event.dest_path)


def start_watch():
    handler = PerFolderIdleHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.start()
    print(f"开始递归监控: {WATCH_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()
        # 优雅清理定时器
        with idle_lock:
            for t in idle_timers.values():
                t.cancel()
            idle_timers.clear()


if __name__ == "__main__":
    start_watch()
