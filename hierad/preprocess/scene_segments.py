"""
PySceneDetect 场景切分 — 预处理阶段用于 Stage1 全片 clip

合并过短场景、拆分过长场景，并提取每 clip 的 ASR 重叠文本。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from .asr import ASRResult
from .asr_overlap import dialogue_in_interval
from .cutter import cut_by_segments, _sec_to_timecode
from .gap_segments import get_video_duration
from hierad.progress import ProgressReporter, progress_scope


def merge_short_scenes(
    ranges: List[Tuple[float, float]],
    min_scene_sec: float = 5.0,
) -> List[Tuple[float, float]]:
    """合并时长不足 min_scene_sec 的相邻场景"""
    if not ranges:
        return []
    merged: List[Tuple[float, float]] = [ranges[0]]
    for start, end in ranges[1:]:
        prev_s, prev_e = merged[-1]
        if (end - start) < min_scene_sec or (prev_e - prev_s) < min_scene_sec:
            merged[-1] = (prev_s, end)
        else:
            merged.append((start, end))
    # 收尾：若最后一段仍太短，并入前一段
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_scene_sec:
        s, _ = merged.pop()
        ps, pe = merged[-1]
        merged[-1] = (ps, pe if pe >= s else _)
    return merged


def split_long_scenes(
    ranges: List[Tuple[float, float]],
    max_scene_sec: float = 90.0,
) -> List[Tuple[float, float]]:
    """拆分超过 max_scene_sec 的场景"""
    out: List[Tuple[float, float]] = []
    for start, end in ranges:
        dur = end - start
        if dur <= max_scene_sec:
            out.append((start, end))
            continue
        n = int(dur // max_scene_sec) + (1 if dur % max_scene_sec > 0 else 0)
        step = dur / n
        for i in range(n):
            s = start + i * step
            e = start + (i + 1) * step if i < n - 1 else end
            out.append((s, e))
    return out


def detect_scene_ranges(
    video_path: str,
    threshold: float = 27.0,
    min_scene_sec: float = 5.0,
    max_scene_sec: float = 90.0,
) -> List[Tuple[float, float]]:
    """PySceneDetect → 合并短场景 → 拆分长场景"""
    from hierad.describe.scene_detect import detect_video_scenes

    raw = detect_video_scenes(video_path, threshold=threshold)
    if not raw:
        dur = get_video_duration(video_path)
        return [(0.0, dur)] if dur > 0 else []
    merged = merge_short_scenes(raw, min_scene_sec=min_scene_sec)
    return split_long_scenes(merged, max_scene_sec=max_scene_sec)


def build_scene_clip_manifest(
    video_path: str,
    whisper_path: str,
    output_dir: str,
    threshold: float = 27.0,
    min_scene_sec: float = 5.0,
    max_scene_sec: float = 90.0,
    max_clips: Optional[int] = None,
    reporter: Optional[ProgressReporter] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    检测场景、切 clip、生成 manifest。

    Returns:
        manifest dict with clip_paths, timecodes, dialogue_overlap, etc.
    """
    with progress_scope(reporter, "场景检测", message="PySceneDetect"):
        ranges = detect_scene_ranges(
            video_path,
            threshold=threshold,
            min_scene_sec=min_scene_sec,
            max_scene_sec=max_scene_sec,
        )
    if max_clips is not None:
        ranges = ranges[:max_clips]
    if not ranges:
        raise ValueError("未检测到任何场景片段")

    asr = ASRResult.from_whisper_json(whisper_path)
    cut_result = cut_by_segments(
        video_path,
        ranges,
        output_dir,
        reporter=reporter,
        show_progress=show_progress and reporter is None,
    )

    clips: List[Dict[str, Any]] = []
    for i, ((start, end), (tc_start, tc_end, path)) in enumerate(zip(ranges, cut_result)):
        clips.append({
            "clip_idx": i,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "timecode": f"{tc_start} --> {tc_end}",
            "video_path": path,
            "dialogue_overlap": dialogue_in_interval(start, end, asr.segments),
            "duration_sec": round(end - start, 3),
        })

    duration = get_video_duration(video_path)
    return {
        "video_path": video_path,
        "whisper_path": whisper_path,
        "video_duration_sec": duration,
        "scene_detect_threshold": threshold,
        "min_scene_sec": min_scene_sec,
        "max_scene_sec": max_scene_sec,
        "clip_count": len(clips),
        "clips": clips,
    }


def save_scene_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_scene_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
