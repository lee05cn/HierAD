"""
AD 间隙切分 - 从 ASR 对白段反推无声间隙

Whisper 给出的是「说话时间」；AD 应插入「不说话的时间」。
实践中 Whisper 常把稀疏旁白/蒙太奇拉成超长 speech，因此默认：
  1) 按文本密度收缩异常长 ASR 段
  2) 用 ffmpeg silencedetect 从 speech 中挖掉真实静音
  3) 过长 gap 切成多段 AD 窗口
"""

from __future__ import annotations

import re
import subprocess
from typing import List, Tuple, Dict, Any, Optional

from .asr import ASRResult


def get_video_duration(video_path: str, ffprobe: str = "ffprobe") -> float:
    """用 ffprobe 获取视频时长（秒）"""
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(f"无法获取视频时长: {video_path}") from e


def merge_speech_intervals(
    intervals: List[Tuple[float, float]],
    merge_tolerance: float = 0.1,
) -> List[Tuple[float, float]]:
    """合并重叠或紧邻的对白时间区间"""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_tolerance:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(
    base: List[Tuple[float, float]],
    holes: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """从 base 区间中挖掉 holes。"""
    out: List[Tuple[float, float]] = []
    holes = merge_speech_intervals(holes, merge_tolerance=0.0)
    for s, e in base:
        cur = [(s, e)]
        for hs, he in holes:
            nxt: List[Tuple[float, float]] = []
            for a, b in cur:
                if he <= a or hs >= b:
                    nxt.append((a, b))
                    continue
                if hs > a:
                    nxt.append((a, hs))
                if he < b:
                    nxt.append((he, b))
            cur = nxt
        out.extend((a, b) for a, b in cur if b - a > 1e-3)
    return out


def repair_sparse_asr_segments(
    segments: List[Dict[str, Any]],
    *,
    min_chars_per_sec: float = 2.0,
    min_seg_sec: float = 10.0,
    max_keep_sec: float = 8.0,
    chars_per_sec_est: float = 12.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    收缩「时长很长但文本很少」的 ASR 段。

    Returns:
        (repaired_speech, blocked_intervals)
        blocked = 被释放的尾段；默认禁止当 AD 间隙（除非静音检测确认）。
    """
    repaired: List[Tuple[float, float]] = []
    blocked: List[Tuple[float, float]] = []
    for seg in segments:
        st = float(seg.get("start", 0))
        en = float(seg.get("end", st))
        if en <= st:
            continue
        text = (seg.get("text") or "").strip()
        dur = en - st
        cps = len(text) / max(dur, 1e-6)
        if dur >= min_seg_sec and cps < min_chars_per_sec:
            est = max(len(text) / max(chars_per_sec_est, 1.0), 0.5)
            keep = min(max(est, 1.0), max_keep_sec, dur)
            repaired.append((st, st + keep))
            if st + keep < en - 0.05:
                blocked.append((st + keep, en))
        else:
            repaired.append((st, en))
    return merge_speech_intervals(repaired), merge_speech_intervals(blocked, merge_tolerance=0.05)


def filter_gaps_against_blocked(
    gaps: List[Tuple[float, float]],
    blocked: List[Tuple[float, float]],
    allowed_in_blocked: Optional[List[Tuple[float, float]]] = None,
    *,
    min_gap_sec: float = 3.0,
    overlap_ratio: float = 0.5,
    max_blocked_sec: float = 20.0,
) -> List[Tuple[float, float]]:
    """
    丢弃主要落在 blocked（稀疏 ASR 释放区）内的 gap；
    若与 allowed_in_blocked（静音检测）交集足够，则保留交集部分。
    但若 gap 所在的 blocked 区间本身超长（> max_blocked_sec），说明该段
    几乎无对白、是纯视觉叙事（如战场闪回），直接允许 AD，不依赖静音确认。
    """
    if not blocked:
        return [(s, e) for s, e in gaps if e - s >= min_gap_sec]

    allowed = merge_speech_intervals(allowed_in_blocked or [], merge_tolerance=0.05)
    out: List[Tuple[float, float]] = []
    for s, e in gaps:
        dur = e - s
        if dur < min_gap_sec:
            continue
        # 计算与 blocked 的重叠，并找出主要落在哪个 blocked 区间
        overlap = 0.0
        main_blocked_dur = 0.0
        for bs, be in blocked:
            ov = max(0.0, min(e, be) - max(s, bs))
            overlap += ov
            if ov / dur >= overlap_ratio * 0.5:
                main_blocked_dur = max(main_blocked_dur, be - bs)
        if overlap / dur < overlap_ratio:
            out.append((s, e))
            continue
        # 主要在 blocked 内：若 blocked 区间超长，直接允许（纯视觉叙事段）
        if main_blocked_dur > max_blocked_sec:
            out.append((s, e))
            continue
        # 否则只保留与静音的交集
        pieces = []
        for as_, ae in allowed:
            lo, hi = max(s, as_), min(e, ae)
            if hi - lo >= min_gap_sec:
                pieces.append((lo, hi))
        out.extend(pieces)
    return out


def detect_silence_intervals(
    video_path: str,
    *,
    noise_db: float = -25.0,
    min_silence_sec: float = 2.0,
    ffmpeg: str = "ffmpeg",
) -> List[Tuple[float, float]]:
    """ffmpeg silencedetect，返回 [(start, end), ...]。"""
    af = f"silencedetect=noise={noise_db}dB:d={min_silence_sec}"
    cmd = [
        ffmpeg, "-hide_banner", "-nostats",
        "-i", video_path,
        "-af", af,
        "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，无法做静音 refinement") from e

    starts: List[float] = []
    ends: List[float] = []
    for line in (r.stderr or "").splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m:
            ends.append(float(m.group(1)))

    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n) if ends[i] > starts[i]]


def gaps_from_speech_segments(
    speech: List[Tuple[float, float]],
    video_duration: float,
    min_gap_sec: float = 3.0,
    margin_sec: float = 0.25,
) -> List[Tuple[float, float]]:
    """从对白区间计算 AD 可用无声间隙。"""
    if video_duration <= 0:
        return []

    speech = merge_speech_intervals(speech)
    if not speech:
        if video_duration >= min_gap_sec:
            return [(0.0, video_duration)]
        return []

    gaps: List[Tuple[float, float]] = []

    first_gap_end = speech[0][0] - margin_sec
    if first_gap_end >= min_gap_sec:
        gaps.append((0.0, first_gap_end))

    for i in range(len(speech) - 1):
        gap_start = speech[i][1] + margin_sec
        gap_end = speech[i + 1][0] - margin_sec
        if gap_end - gap_start >= min_gap_sec:
            gaps.append((gap_start, gap_end))

    last_gap_start = speech[-1][1] + margin_sec
    if video_duration - last_gap_start >= min_gap_sec:
        gaps.append((last_gap_start, video_duration))

    return gaps


def split_long_gaps(
    gaps: List[Tuple[float, float]],
    max_chunk_sec: float,
    min_gap_sec: float = 3.0,
) -> List[Tuple[float, float]]:
    """将过长 gap 切成不超过 max_chunk_sec 的窗口。"""
    if max_chunk_sec is None or max_chunk_sec <= 0:
        return gaps
    out: List[Tuple[float, float]] = []
    for s, e in gaps:
        dur = e - s
        if dur <= max_chunk_sec:
            if dur >= min_gap_sec:
                out.append((s, e))
            continue
        t = s
        while t + min_gap_sec <= e:
            chunk_end = min(t + max_chunk_sec, e)
            if chunk_end - t >= min_gap_sec:
                out.append((t, chunk_end))
            t = chunk_end
    return out


def _format_dialogue_line(seg: Dict[str, Any]) -> str:
    text = (seg.get("text") or "").strip()
    if not text:
        return ""
    speaker = (seg.get("speaker") or "").strip()
    if speaker:
        return f"{speaker}: {text}"
    return text


def dialogue_context_for_gap(
    gap_start: float,
    gap_end: float,
    speech_segments: List[Dict[str, Any]],
    max_before: int = 3,
    max_after: int = 3,
) -> Tuple[str, str]:
    """获取 gap 前后的对白上下文（供 Stage2；Stage3 不再使用）。"""
    before_lines: List[str] = []
    after_lines: List[str] = []

    for seg in speech_segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        line = _format_dialogue_line(seg)
        if not line:
            continue
        if end <= gap_start + 0.05:
            before_lines.append(line)
        elif start >= gap_end - 0.05:
            after_lines.append(line)

    before = " | ".join(before_lines[-max_before:]) if before_lines else "(None)"
    after = " | ".join(after_lines[:max_after]) if after_lines else "(None)"
    return before, after


def load_ad_gap_segments(
    whisper_path: str,
    video_path: str,
    min_gap_sec: float = 3.0,
    margin_sec: float = 0.25,
    *,
    repair_sparse: bool = True,
    min_chars_per_sec: float = 2.0,
    silence_refine: bool = True,
    silence_noise_db: float = -25.0,
    silence_min_sec: float = 2.0,
    max_gap_chunk_sec: Optional[float] = 12.0,
) -> Tuple[List[Tuple[float, float]], ASRResult, float]:
    """
    从 Whisper JSON + 视频时长计算 AD 间隙切分点。

    Returns:
        (gaps, asr_result, video_duration)
    """
    asr_result = ASRResult.from_whisper_json(whisper_path)
    duration = get_video_duration(video_path)

    blocked: List[Tuple[float, float]] = []
    if repair_sparse:
        speech, blocked = repair_sparse_asr_segments(
            asr_result.segments,
            min_chars_per_sec=min_chars_per_sec,
        )
    else:
        speech = [(float(s["start"]), float(s["end"])) for s in asr_result.segments]
        speech = merge_speech_intervals(speech)

    silences: List[Tuple[float, float]] = []
    if silence_refine:
        try:
            silences = detect_silence_intervals(
                video_path,
                noise_db=silence_noise_db,
                min_silence_sec=min(silence_min_sec, min_gap_sec),
            )
            speech = subtract_intervals(speech, silences)
        except RuntimeError:
            pass

    gaps = gaps_from_speech_segments(
        speech,
        duration,
        min_gap_sec=min_gap_sec,
        margin_sec=margin_sec,
    )
    # 稀疏 ASR 释放区：无静音确认则不当 AD
    gaps = filter_gaps_against_blocked(
        gaps,
        blocked,
        allowed_in_blocked=silences,
        min_gap_sec=min_gap_sec,
    )
    if max_gap_chunk_sec is not None:
        gaps = split_long_gaps(gaps, max_gap_chunk_sec, min_gap_sec=min_gap_sec)
    return gaps, asr_result, duration
