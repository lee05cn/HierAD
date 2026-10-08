"""ASR 与时间区间对齐工具"""

from typing import List, Dict, Any, Tuple


def _format_line(seg: Dict[str, Any]) -> str:
    text = (seg.get("text") or "").strip()
    if not text:
        return ""
    speaker = (seg.get("speaker") or "").strip()
    return f"{speaker}: {text}" if speaker else text


def dialogue_in_interval(
    start_sec: float,
    end_sec: float,
    speech_segments: List[Dict[str, Any]],
    max_lines: int = 20,
) -> str:
    """提取时间区间内的对白（与区间有重叠即计入）"""
    lines: List[str] = []
    for seg in speech_segments:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", s))
        if e <= start_sec or s >= end_sec:
            continue
        line = _format_line(seg)
        if line:
            lines.append(line)
    if not lines:
        return "(None)"
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"... (+{len(lines) - max_lines + 1} more lines)"]
    return " | ".join(lines)


def find_clip_index_for_time(
    time_sec: float,
    clip_ranges: List[Tuple[float, float]],
) -> int:
    """按时间戳查找所属 clip 索引"""
    for i, (start, end) in enumerate(clip_ranges):
        if start <= time_sec <= end:
            return i
    best_i, best_dist = 0, float("inf")
    for i, (start, end) in enumerate(clip_ranges):
        mid = (start + end) / 2
        dist = abs(time_sec - mid)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def clips_for_interval(
    start_sec: float,
    end_sec: float,
    clip_ranges: List[Tuple[float, float]],
) -> List[int]:
    """返回与区间重叠的所有 clip 索引"""
    mid = (start_sec + end_sec) / 2
    indices = set()
    for i, (cs, ce) in enumerate(clip_ranges):
        if ce > start_sec and cs < end_sec:
            indices.add(i)
    if not indices:
        indices.add(find_clip_index_for_time(mid, clip_ranges))
    return sorted(indices)
