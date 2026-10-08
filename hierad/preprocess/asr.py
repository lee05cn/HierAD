"""
ASR 模块 - 视频 → 音频提取 → 转录 → 字幕+时间戳

完整流程:
  1. extract_audio_from_video() - ffmpeg 提取音频
  2. transcribe_with_asr_service() - Whisper HTTP 服务（默认，端口 8001）
  3. transcribe_with_whisper() - 本地 openai-whisper（兜底）
  4. transcribe_video() - 一站式：视频 → JSON

加载已有结果:
  ASRResult.from_whisper_json() - 从 Whisper JSON 加载
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Callable

import requests

from hierad.config import (
    ASR_DIARIZE as _CONFIG_ASR_DIARIZE,
    ASR_URL as _CONFIG_ASR_URL,
    get_asr_provider,
)

AsrBackend = Literal["auto", "service", "local", "dashscope"]


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    ffmpeg: str = "ffmpeg",
) -> Optional[str]:
    """
    从视频提取音频（16kHz 单声道，Whisper 推荐格式）

    Args:
        video_path: 输入视频路径
        output_path: 输出音频路径，默认临时文件
        sample_rate: 采样率，Whisper 推荐 16000
        ffmpeg: ffmpeg 可执行路径

    Returns:
        音频文件路径，失败返回 None
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        import os
        os.close(fd)

    cmd = [
        ffmpeg, "-y", "-v", "quiet",
        "-i", str(video_path),
        "-vn", "-ar", str(sample_rate), "-ac", "1",
        "-f", "wav", str(output_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not Path(output_path).exists() or Path(output_path).stat().st_size < 44:
            return None
        return output_path
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + (e.stdout or "")
        if "does not contain any stream" in err or "Output file #0 does not contain" in err:
            raise RuntimeError("视频无音轨，无法提取音频进行 ASR") from e
        return None
    except FileNotFoundError:
        return None


def segments_from_asr_service_payload(data: Any) -> List[Dict[str, Any]]:
    """解析 Whisper ASR HTTP 服务返回的 data 字段"""
    if not isinstance(data, list):
        raise ValueError("ASR 服务返回的 data 必须为数组")

    segments = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start = float(item.get("start", 0))
        end = float(item.get("end", start))
        seg: Dict[str, Any] = {"start": start, "end": end, "text": text}
        speaker = item.get("speaker")
        if speaker is not None and str(speaker).strip():
            seg["speaker"] = str(speaker)
        segments.append(seg)
    return segments


def transcribe_with_asr_service(
    audio_path: str,
    output_json_path: Optional[str] = None,
    asr_url: Optional[str] = None,
    timeout: int = 3600,
    diarize: bool = False,
) -> "ASRResult":
    """
    调用 Whisper ASR HTTP 服务（faster-whisper）

    服务接口: POST {asr_url}
      body: {"audio_path": "/abs/path/to.wav", "diarize": false}

    diarize=false（默认）: 仅转录，不做说话人分离（HierAD gap 切分推荐）
    diarize=true: 额外 pyannote 说话人分离（慢）
    """
    audio_path = str(Path(audio_path).resolve())
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"音频不存在: {audio_path}")

    url = (asr_url or _CONFIG_ASR_URL or "").strip()
    if not url:
        raise ValueError("ASR 服务 URL 未配置，请设置 HIERAD_ASR_URL")

    r = requests.post(
        url,
        json={"audio_path": audio_path, "diarize": diarize},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("code", 1) != 0:
        raise RuntimeError(f"ASR 服务错误: {payload.get('msg', payload)}")

    segments = segments_from_asr_service_payload(payload.get("data", []))
    asr_result = ASRResult(segments)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)

    return asr_result


def transcribe_with_dashscope_asr(
    audio_path: str,
    output_json_path: Optional[str] = None,
) -> "ASRResult":
    from hierad.providers.asr import transcribe_with_dashscope

    segments = transcribe_with_dashscope(audio_path)
    asr_result = ASRResult(segments)
    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    return asr_result


def transcribe_with_whisper(
    audio_path: str,
    output_json_path: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = None,
) -> "ASRResult":
    """
    使用 Whisper 对音频进行转录

    Args:
        audio_path: 音频文件路径（wav/mp3）
        output_json_path: 输出 JSON 路径，不指定则只返回 ASRResult
        model_size: Whisper 模型大小 tiny/base/small/medium/large
        language: 语言代码，如 "en"/"zh"，None 为自动检测

    Returns:
        ASRResult
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "需要安装 openai-whisper: pip install openai-whisper"
        ) from None

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=False,
        verbose=False,
    )

    segments = []
    for s in result.get("segments", []):
        start = float(s.get("start", 0))
        end = float(s.get("end", start))
        text = (s.get("text") or "").strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})

    asr_result = ASRResult(segments)

    if output_json_path:
        output_data = {"segments": segments}
        if result.get("language"):
            output_data["language"] = result["language"]
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    return asr_result


def transcribe_video(
    video_path: str,
    output_json_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = None,
    asr_backend: AsrBackend = "auto",
    asr_url: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    diarize: Optional[bool] = None,
) -> "ASRResult":
    """
    一站式：视频 → 字幕+时间戳 JSON

    流程: 提取音频 → ASR 服务 / 本地 Whisper → 输出 JSON

    Args:
        video_path: 输入视频路径
        output_json_path: 输出 JSON 路径，默认 work_dir/{video_stem}_asr.json
        work_dir: 工作目录，默认视频同目录
        model_size: 本地 Whisper 模型（asr_backend=local 或 auto 兜底时）
        language: 本地 Whisper 语言
        asr_backend: auto（先 HTTP 服务再本地）| service | local | dashscope
        asr_url: 覆盖 ASR 服务 URL
        diarize: 是否说话人分离，默认 False（HierAD 不需要）

    Returns:
        ASRResult
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    work_dir = Path(work_dir or video_path.parent)
    work_dir.mkdir(parents=True, exist_ok=True)

    if output_json_path is None:
        output_json_path = str(work_dir / f"{video_path.stem}_asr.json")

    use_diarize = _CONFIG_ASR_DIARIZE if diarize is None else diarize
    if asr_backend == "auto" and get_asr_provider() == "dashscope":
        asr_backend = "dashscope"

    if progress_callback:
        progress_callback("提取音频")
    audio_path = extract_audio_from_video(str(video_path))
    if not audio_path:
        raise RuntimeError("音频提取失败（视频可能没有音轨）")

    try:
        if progress_callback:
            progress_callback("Whisper 转录")
        if asr_backend == "local":
            result = transcribe_with_whisper(
                audio_path,
                output_json_path=output_json_path,
                model_size=model_size,
                language=language,
            )
        elif asr_backend == "dashscope":
            result = transcribe_with_dashscope_asr(
                audio_path,
                output_json_path=output_json_path,
            )
        elif asr_backend == "service":
            result = transcribe_with_asr_service(
                audio_path,
                output_json_path=output_json_path,
                asr_url=asr_url,
                diarize=use_diarize,
            )
        else:
            try:
                result = transcribe_with_asr_service(
                    audio_path,
                    output_json_path=output_json_path,
                    asr_url=asr_url,
                    diarize=use_diarize,
                )
            except (requests.RequestException, RuntimeError, ValueError):
                result = transcribe_with_whisper(
                    audio_path,
                    output_json_path=output_json_path,
                    model_size=model_size,
                    language=language,
                )
    finally:
        if audio_path and audio_path.startswith(tempfile.gettempdir()):
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass

    if progress_callback:
        progress_callback("ASR 完成")
    return result


class ASRResult:
    """
    ASR 结果封装

    支持 Whisper JSON 格式: {"segments": [{"start", "end", "text"}, ...]}
    或带 speaker 的格式: [{"start", "end", "speaker", "text"}, ...]
    """

    def __init__(self, segments: List[Dict[str, Any]]):
        self.segments = sorted(segments, key=lambda x: x["start"])

    @classmethod
    def from_whisper_json(cls, path: str) -> "ASRResult":
        """从 Whisper 输出 JSON 加载"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ASR 文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Whisper 格式: {"segments": [{"start", "end", "text"}, ...]}
        if "segments" in data:
            raw = data["segments"]
        elif isinstance(data, list):
            raw = data
        else:
            raise ValueError(f"不支持的 ASR JSON 格式: {list(data.keys()) if isinstance(data, dict) else 'list'}")

        segments = []
        for s in raw:
            start = float(s.get("start", 0))
            end = float(s.get("end", start))
            text = s.get("text", "").strip()
            if not text:
                continue
            segments.append({
                "start": start,
                "end": end,
                "speaker": s.get("speaker", ""),
                "text": text,
            })

        return cls(segments)

    def __len__(self) -> int:
        return len(self.segments)

    def dump_to_json(self, path: str) -> None:
        """保存为 Whisper 兼容的 JSON 格式"""
        data = {"segments": self.segments}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
