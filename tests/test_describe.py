"""Describe 模块测试"""

import pytest

from hierad.describe.models import DenseDescription, SceneInfo
from hierad.describe.utils import (
    parse_stage1_output,
    parse_scene_summary_output,
    parse_llm_json,
    compute_cosine_adjacent_similarities,
    normalize_character_with_mapping,
)
from hierad.describe import Stage2Understanding


def test_parse_stage1_output():
    """测试 Stage1 输出解析"""
    text = """SETTING: Kitchen
CHARACTERS: JOHN, MARY
ACTION: John stirs the pot."""
    r = parse_stage1_output(text)
    assert r["setting"] == "Kitchen"
    assert "JOHN" in r["characters"] or "John" in r["characters"]
    assert "stirs" in r["action"]


def test_parse_stage1_markdown_bold_labels():
    text = """**Setting:** Desert
**Characters:** TONY STARK
**Action:** Tony lies injured on the ground."""
    r = parse_stage1_output(text)
    assert r["setting"] == "Desert"
    assert "TONY" in r["characters"].upper()
    assert "injured" in r["action"].lower()


def test_parse_scene_summary_output():
    """测试场景摘要解析"""
    text = """EVENT: Graham woke up.
PROGRESSION: lies in bed → sits up → walks to door
CHARACTERS: GRAHAM"""
    r = parse_scene_summary_output(text)
    assert "Graham" in r["event"]
    assert len(r["progression_list"]) >= 2
    assert "GRAHAM" in r["characters"]


def test_parse_llm_json():
    """测试 LLM JSON 解析"""
    text = 'Some text {"key": "value"} more'
    r = parse_llm_json(text)
    assert r is not None
    assert r["key"] == "value"


def test_compute_cosine_similarities():
    """测试余弦相似度"""
    import numpy as np

    emb = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    sims = compute_cosine_adjacent_similarities(emb)
    assert len(sims) == 2
    assert sims[0] > 0.99  # 相同向量
    assert abs(sims[1]) < 0.01  # 正交


def test_normalize_character_with_mapping():
    """测试角色名标准化"""
    mapping = {"Johnny": "TOMMY WISEAU", "JOHNNY": "TOMMY WISEAU"}
    assert normalize_character_with_mapping("Johnny", mapping) == "TOMMY WISEAU"
    assert normalize_character_with_mapping("unknown", mapping) == "UNKNOWN"


def test_stage2_aggregate_fixed_window():
    """测试 Stage2 固定窗口场景聚合（无 PySceneDetect 视频路径）"""
    descs = [
        DenseDescription(i, "00:00:00", "00:00:05", 0.0, 5.0, "Kitchen", "JOHN", "action", "")
        for i in range(10)
    ]
    s2 = Stage2Understanding(use_scenedetect=False, use_llm=False, scene_window=3)
    scenes = s2.aggregate_scenes(descs)
    assert len(scenes) >= 1
    assert scenes[0].start_segment == 0
    assert scenes[0].end_segment >= 0
