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
    paths = []
    for i, text in enumerate(texts):
        text = (text or "").strip()
        if not text:
            raise ValueError(f"第 {i} 条 AD 文本为空")
        synthesizer = SpeechSynthesizer(model=model_name, voice=voice_name)
        audio = synthesizer.call(text)
        if not audio:
            raise RuntimeError(f"DashScope TTS 未返回音频: index={i}")
        out = save_dir / f"{i:04d}_{uuid4().hex[:8]}.wav"
        out.write_bytes(audio if isinstance(audio, (bytes, bytearray)) else bytes(audio))
        paths.append(str(out))
    return paths
