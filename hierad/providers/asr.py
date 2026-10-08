"""阿里云百炼 ASR：本地 wav → {start,end,text}。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from hierad.config import ASR_MODEL, get_dashscope_api_key


def _require_dashscope():
    try:
        import dashscope  # noqa: F401
        from dashscope.audio.asr import Recognition, RecognitionCallback
    except ImportError as e:
        raise ImportError(
            'DashScope ASR 需要: pip install "hierad[cloud]" 或 pip install dashscope'
        ) from e
    return dashscope, Recognition, RecognitionCallback


def _sentence_to_seg(sentence: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(sentence, dict):
        return None
    text = (sentence.get("text") or "").strip()
    if not text:
        return None
    start_ms = sentence.get("begin_time", sentence.get("beginTime", 0)) or 0
    end_ms = sentence.get("end_time", sentence.get("endTime", start_ms)) or start_ms
    return {
        "start": float(start_ms) / 1000.0,
        "end": float(end_ms) / 1000.0,
        "text": text,
    }


def transcribe_with_dashscope(
    audio_path: str,
    *,
    model: Optional[str] = None,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """用百炼实时识别读本地 wav，返回 HierAD 段列表。

    注意：dashscope 的 Recognition.call() 是同步模式，结果在返回值里，
    不会走 on_event 回调；因此这里直接读 call() 的返回结果。
    """
    dashscope, Recognition, _ = _require_dashscope()
    key = get_dashscope_api_key()
    if not key:
        raise ValueError("DashScope ASR 需要环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = key

    path = Path(audio_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"音频不存在: {path}")

    rec = Recognition(
        model=model or ASR_MODEL,
        format="wav",
        sample_rate=sample_rate,
        callback=None,
    )
    result = rec.call(str(path))
    raw = result.get_sentence()

    sentences: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        sentences = [s for s in raw if isinstance(s, dict)]
    elif isinstance(raw, dict):
        sentences = [raw]

    segments: List[Dict[str, Any]] = []
    for sentence in sentences:
        # 只保留已结束的句子，避免中间态结果
        if sentence.get("sentence_end") is False:
            continue
        seg = _sentence_to_seg(sentence)
        if seg:
            segments.append(seg)

    # 去重：实时接口可能重复推送同一句
    unique: List[Dict[str, Any]] = []
    seen = set()
    for seg in segments:
        key_t = (round(seg["start"], 3), round(seg["end"], 3), seg["text"])
        if key_t in seen:
            continue
        seen.add(key_t)
        unique.append(seg)
    unique.sort(key=lambda x: x["start"])
    return unique
