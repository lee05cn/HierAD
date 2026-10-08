"""
基于 PySceneDetect 的视频场景切分，并将 Whisper 片段映射到场景。

使用 ContentDetector 检测镜头/场景切换，再按时间重叠将 DenseDescription 归入各场景。
"""

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from .models import DenseDescription, SceneInfo


def timecode_to_seconds(timecode: str) -> float:
    """将 HH:MM:SS.mmm 或 HH:MM:SS 转为秒"""
    tc = (timecode or "").strip().replace(",", ".")
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t = datetime.strptime(tc, fmt)
            return (
                t.hour * 3600
                + t.minute * 60
                + t.second
                + t.microsecond / 1_000_000
            )
        except ValueError:
            continue
    return 0.0


def detect_video_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> List[Tuple[float, float]]:
    """
    用 PySceneDetect ContentDetector 检测场景边界。

    Returns:
        [(start_sec, end_sec), ...] 按时间顺序
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(str(path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    duration = video.duration.get_seconds() if video.duration else 0.0
    if not scene_list:
        return [(0.0, duration)] if duration > 0 else [(0.0, 0.0)]

    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]


def assign_segment_to_scene(mid_sec: float, scene_ranges: List[Tuple[float, float]]) -> int:
    """按片段中点时间归属场景"""
    if not scene_ranges:
        return 0
    for i, (start, end) in enumerate(scene_ranges):
        is_last = i == len(scene_ranges) - 1
        if is_last:
            if start <= mid_sec <= end:
                return i
        elif start <= mid_sec < end:
            return i
    # 兜底：归入最近的场景
    best_i, best_dist = 0, float("inf")
    for i, (start, end) in enumerate(scene_ranges):
        center = (start + end) / 2
        dist = abs(mid_sec - center)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def aggregate_descriptions_by_scenes(
    descriptions: List[DenseDescription],
    scene_ranges: List[Tuple[float, float]],
) -> List[SceneInfo]:
    """将 DenseDescription 列表按 PySceneDetect 场景范围聚合为 SceneInfo"""
    if not descriptions:
        return []

    if not scene_ranges:
        scene_ranges = [(0.0, 0.0)]

    scene_ids: List[int] = []
    for desc in descriptions:
        start = timecode_to_seconds(desc.timecode_start)
        end = timecode_to_seconds(desc.timecode_end)
        mid = (start + end) / 2 if end > start else start
        scene_ids.append(assign_segment_to_scene(mid, scene_ranges))

    scenes: List[SceneInfo] = []
    i = 0
    while i < len(descriptions):
        sid = scene_ids[i]
        j = i + 1
        while j < len(descriptions) and scene_ids[j] == sid:
            j += 1
        window = descriptions[i:j]
        scenes.append(
            SceneInfo(
                scene_id=len(scenes),
                start_segment=window[0].segment_idx,
                end_segment=window[-1].segment_idx,
                start_time=window[0].timecode_start,
                end_time=window[-1].timecode_end,
            )
        )
        i = j
    return scenes


def aggregate_scenes_pyscenedetect(
    descriptions: List[DenseDescription],
    video_path: str,
    threshold: float = 27.0,
) -> List[SceneInfo]:
    """PySceneDetect 检测 + 片段映射的一站式接口"""
    ranges = detect_video_scenes(video_path, threshold=threshold)
    return aggregate_descriptions_by_scenes(descriptions, ranges)
