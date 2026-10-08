"""PySceneDetect 场景聚合测试"""

import pytest

from hierad.describe.models import DenseDescription
from hierad.describe.scene_detect import (
    timecode_to_seconds,
    assign_segment_to_scene,
    aggregate_descriptions_by_scenes,
)
from hierad.describe import Stage2Understanding


def test_timecode_to_seconds():
    assert timecode_to_seconds("00:01:05.500") == pytest.approx(65.5)
    assert timecode_to_seconds("00:00:00.000") == 0.0


def test_assign_segment_to_scene():
    ranges = [(0.0, 10.0), (10.0, 25.0), (25.0, 40.0)]
    assert assign_segment_to_scene(5.0, ranges) == 0
    assert assign_segment_to_scene(15.0, ranges) == 1
    assert assign_segment_to_scene(30.0, ranges) == 2


def test_aggregate_descriptions_by_scenes():
    descs = [
        DenseDescription(0, "00:00:00.000", "00:00:05.000", 0.0, 5.0, "A", "X", "a", ""),
        DenseDescription(1, "00:00:05.000", "00:00:10.000", 5.0, 10.0, "A", "X", "b", ""),
        DenseDescription(2, "00:00:12.000", "00:00:18.000", 12.0, 18.0, "B", "Y", "c", ""),
        DenseDescription(3, "00:00:20.000", "00:00:28.000", 20.0, 28.0, "B", "Y", "d", ""),
    ]
    ranges = [(0.0, 10.0), (10.0, 30.0)]
    scenes = aggregate_descriptions_by_scenes(descs, ranges)
    assert len(scenes) == 2
    assert scenes[0].start_segment == 0
    assert scenes[0].end_segment == 1
    assert scenes[1].start_segment == 2
    assert scenes[1].end_segment == 3


def test_stage2_aggregate_fixed_window_fallback():
    """无 video_path 时使用固定窗口兜底"""
    descs = [
        DenseDescription(i, "00:00:00", "00:00:05", 0.0, 5.0, "Kitchen", "JOHN", "action", "")
        for i in range(10)
    ]
    s2 = Stage2Understanding(use_scenedetect=True, use_llm=False, scene_window=3)
    scenes = s2.aggregate_scenes(descs, video_path=None)
    assert len(scenes) >= 1
    assert scenes[0].start_segment == 0


def test_stage2_pyscenedetect_on_short_clip():
    """短片段无切点时整段作为单场景"""
    pytest.importorskip("scenedetect")
    clip = "/path/to/HierAD-old/data/1005_Signs/segments/0001.mp4"
    try:
        from pathlib import Path
        if not Path(clip).exists():
            pytest.skip("测试视频不存在")
    except Exception:
        pytest.skip("测试视频不可用")

    descs = [
        DenseDescription(0, "00:03:00.000", "00:03:02.000", "Bedroom", "A", "act", ""),
        DenseDescription(1, "00:03:31.000", "00:03:33.000", "Bedroom", "A", "act2", ""),
    ]
    s2 = Stage2Understanding(use_scenedetect=True, use_llm=False)
    scenes = s2.aggregate_scenes(descs, video_path=clip)
    assert len(scenes) >= 1
    assert scenes[0].start_segment == 0
