"""
CosyVoice TTS 客户端：按 AD 时间戳生成旁白音轨
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from hierad.config import TTS_CLOUD_VOICE, TTS_SPK_ID, TTS_SPEED, TTS_URL, get_tts_provider


_TC_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})[.,](\d{1,3})$"
)


def timecode_to_sec(tc: str) -> float:
    tc = str(tc).strip().replace(" ", "")
    m = _TC_RE.match(tc)
    if not m:
        raise ValueError(f"无法解析时间码: {tc!r}")
    h = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = m.group(4).ljust(3, "0")[:3]
    return h * 3600 + mm * 60 + ss + int(frac) / 1000.0


def synthesize_ad_clips(
    texts: Sequence[str],
    save_dir: Union[str, Path],
    *,
    tts_url: Optional[str] = None,
    voice_id: Optional[str] = None,
    prompt_text: Optional[str] = None,
    spk_id: Optional[str] = None,
    speed: Optional[float] = None,
    timeout: float = 180.0,
) -> List[str]:
    """
    逐条调用 TTS，返回 wav 路径列表。
    provider=local 时走本机 CosyVoice /inference_single；
    provider=dashscope 时走百炼 CosyVoice 云端音色。
    """
    if get_tts_provider() == "dashscope":
        from hierad.providers.tts import synthesize_with_dashscope

        return synthesize_with_dashscope(
            list(texts),
            Path(save_dir),
            voice=TTS_CLOUD_VOICE,
        )

    base = (tts_url or TTS_URL).rstrip("/")
    endpoint = f"{base}/inference_single"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_spk = spk_id if spk_id is not None else TTS_SPK_ID
    resolved_speed = float(TTS_SPEED if speed is None else speed)

    paths: List[str] = []
    for i, text in enumerate(texts):
        text = (text or "").strip()
        if not text:
            raise ValueError(f"第 {i} 条 AD 文本为空")
        payload: Dict[str, Any] = {
            "text": text,
            "save_dir": str(save_dir.resolve()),
            "speed": resolved_speed,
        }
        if resolved_spk:
            payload["spk_id"] = resolved_spk
        if voice_id:
            payload["voice_id"] = voice_id
            if prompt_text:
                payload["prompt_text"] = prompt_text
        resp = requests.post(
            endpoint,
            json=payload,
            proxies={"http": None, "https": None},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"TTS HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if data.get("code", 1) != 0:
            raise RuntimeError(f"TTS 失败: {data.get('msg', data)}")
        wav = data.get("data")
        if not wav or not Path(wav).exists():
            raise RuntimeError(f"TTS 未返回有效音频: {data}")
        paths.append(str(wav))
    return paths


def _atempo_chain(factor: float) -> str:
    """ffmpeg atempo 单段仅支持 0.5–2.0，拆成多级。"""
    parts: List[str] = []
    remaining = factor
    while remaining > 2.0 + 1e-6:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - 1e-6:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def speed_change_audio(
    wav,
    factor: float,
    *,
    ffmpeg: str = "ffmpeg",
):
    """
    保持音调的变速（ffmpeg atempo）。factor>1 更快。
    wav: pydub.AudioSegment
    """
    if factor <= 1.01:
        return wav
    factor = min(max(factor, 0.5), 3.0)
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError("需要 pydub") from e

    with tempfile.TemporaryDirectory(prefix="hierad_tts_") as tmp:
        inp = Path(tmp) / "in.wav"
        out = Path(tmp) / "out.wav"
        wav.export(str(inp), format="wav")
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(inp),
            "-filter:a", _atempo_chain(factor),
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError("变速需要 ffmpeg") from e
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            raise RuntimeError(f"ffmpeg atempo 失败: {err}") from e
        return AudioSegment.from_file(str(out))


def fit_clip_to_gap(
    wav,
    gap_ms: int,
    *,
    max_speedup: float = 1.55,
    fade_ms: int = 80,
):
    """
    将 TTS 片段适配到 gap：先加速（保音调），仍超长才裁切淡出。
    """
    if gap_ms <= 0:
        return wav[:0]
    if len(wav) <= gap_ms:
        out = wav
    else:
        needed = len(wav) / float(gap_ms)
        if needed <= max_speedup:
            out = speed_change_audio(wav, needed)
            if len(out) > gap_ms:
                out = out[:gap_ms]
        else:
            out = speed_change_audio(wav, max_speedup)
            if len(out) > gap_ms:
                out = out[:gap_ms]
    if fade_ms > 0 and len(out) > fade_ms * 2:
        out = out.fade_in(fade_ms).fade_out(fade_ms)
    return out


def build_ad_track(
    entries: Sequence[Dict[str, Any]],
    wav_paths: Sequence[str],
    output_wav: Union[str, Path],
    *,
    track_len_sec: float,
    frame_rate: int = 22050,
    fade_sec: float = 0.08,
    fit_to_gap: bool = True,
    max_speedup: float = 1.55,
) -> Path:
    """
    将各 AD wav 叠加到与视频等长的静音轨上（按 start_time 对齐）。
    若 TTS 超过间隙且 fit_to_gap=True，先加速再必要时裁切。
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError("AD TTS 混音需要 pydub，请先 pip install pydub") from e

    if len(entries) != len(wav_paths):
        raise ValueError(f"条目数 {len(entries)} 与 wav 数 {len(wav_paths)} 不一致")

    track_ms = max(int(track_len_sec * 1000), 1)
    bg = AudioSegment.silent(duration=track_ms, frame_rate=frame_rate)
    fade_ms = int(fade_sec * 1000)

    for entry, wav_path in zip(entries, wav_paths):
        start = entry.get("start_time") or entry.get("start")
        end = entry.get("end_time") or entry.get("end")
        if start is None or end is None:
            raise ValueError(f"缺少时间码: {entry!r}")
        start_sec = timecode_to_sec(str(start))
        end_sec = timecode_to_sec(str(end))
        pos_ms = int(start_sec * 1000)
        if pos_ms >= track_ms:
            continue

        wav = AudioSegment.from_file(str(wav_path))
        if wav.frame_rate != frame_rate:
            wav = wav.set_frame_rate(frame_rate)
        if wav.channels != 1:
            wav = wav.set_channels(1)

        gap_ms = max(int((end_sec - start_sec) * 1000), 1)
        if fit_to_gap:
            wav = fit_clip_to_gap(wav, gap_ms, max_speedup=max_speedup, fade_ms=fade_ms)
        elif len(wav) > fade_ms * 2:
            wav = wav.fade_in(fade_ms).fade_out(fade_ms)

        # 不超出视频结尾
        if pos_ms + len(wav) > track_ms:
            wav = wav[: max(track_ms - pos_ms, 0)]
        if len(wav) <= 0:
            continue
        bg = bg.overlay(wav, position=pos_ms)

    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    bg.export(str(out), format="wav")
    return out


def synthesize_and_build_track(
    entries: Sequence[Dict[str, Any]],
    work_dir: Union[str, Path],
    *,
    track_len_sec: float,
    tts_url: Optional[str] = None,
    voice_id: Optional[str] = None,
    prompt_text: Optional[str] = None,
    spk_id: Optional[str] = None,
    speed: Optional[float] = None,
    max_speedup: float = 1.55,
) -> Tuple[Path, List[str]]:
    """
    对非空 ad_script 做 TTS，并生成整轨 ad_audio.wav。
    返回 (track_path, clip_wav_paths)。
    """
    work_dir = Path(work_dir)
    tts_dir = work_dir / "tts"
    usable = [
        e for e in entries
        if (e.get("ad_script") or e.get("text") or "").strip()
    ]
    if not usable:
        raise ValueError("没有可用于 TTS 的 ad_script")

    texts = [
        " ".join(
            line.strip()
            for line in (e.get("ad_script") or e.get("text") or "").splitlines()
            if line.strip()
        )
        for e in usable
    ]
    clips = synthesize_ad_clips(
        texts,
        tts_dir,
        tts_url=tts_url,
        voice_id=voice_id,
        prompt_text=prompt_text,
        spk_id=spk_id,
        speed=speed,
    )
    track = build_ad_track(
        usable,
        clips,
        work_dir / "ad_audio.wav",
        track_len_sec=track_len_sec,
        max_speedup=max_speedup,
    )
    return track, clips
