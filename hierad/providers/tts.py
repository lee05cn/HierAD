"""阿里云百炼 CosyVoice TTS：文本 → 本地 wav。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

from hierad.config import TTS_CLOUD_VOICE, TTS_MODEL, get_dashscope_api_key


def _require_dashscope_tts():
    try:
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer
    except ImportError as e:
        raise ImportError(
            'DashScope TTS 需要: pip install "hierad[cloud]" 或 pip install dashscope'
        ) from e
    return dashscope, SpeechSynthesizer


def synthesize_with_dashscope(
    texts: Sequence[str],
    save_dir: Path,
    *,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
) -> list:
    dashscope, SpeechSynthesizer = _require_dashscope_tts()
    key = get_dashscope_api_key()
    if not key:
        raise ValueError("DashScope TTS 需要环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = key

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = model or TTS_MODEL
    voice_name = voice or TTS_CLOUD_VOICE
    # speech_rate 只在显式给定时才传：None 走 SDK 默认 1.0，避免悄悄改掉既有行为
    rate = float(speed) if speed is not None else None
    paths = []
    for i, text in enumerate(texts):
        text = (text or "").strip()
        if not text:
            raise ValueError(f"第 {i} 条 AD 文本为空")
        kwargs = {}
        if rate is not None:
            kwargs["speech_rate"] = rate
        synthesizer = SpeechSynthesizer(model=model_name, voice=voice_name, **kwargs)
        audio = synthesizer.call(text)
        if not audio:
            raise RuntimeError(f"DashScope TTS 未返回音频: index={i}")
        out = save_dir / f"{i:04d}_{uuid4().hex[:8]}.wav"
        out.write_bytes(audio if isinstance(audio, (bytes, bytearray)) else bytes(audio))
        paths.append(str(out))
    return paths


def synthesize_preview(
    text: str,
    voice: str,
    *,
    model: Optional[str] = None,
    speed: Optional[float] = None,
) -> bytes:
    """合成一句短样例，直接返回音频字节（网页「试听」用，不落盘）。

    注意：dashscope 默认返回 mp3 字节，这里原样透传，调用方按 audio/mpeg 处理。
    """
    dashscope, SpeechSynthesizer = _require_dashscope_tts()
    key = get_dashscope_api_key()
    if not key:
        raise ValueError("DashScope TTS 需要环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = key

    text = (text or "").strip()
    if not text:
        raise ValueError("试听文本为空")
    model_name = model or TTS_MODEL
    voice_name = voice or TTS_CLOUD_VOICE
    kwargs = {}
    if speed is not None:
        kwargs["speech_rate"] = float(speed)
    synthesizer = SpeechSynthesizer(model=model_name, voice=voice_name, **kwargs)
    audio = synthesizer.call(text)
    if not audio:
        raise RuntimeError("DashScope TTS 未返回音频")
    return audio if isinstance(audio, (bytes, bytearray)) else bytes(audio)
