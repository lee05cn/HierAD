"""Pipeline 模块测试"""

import json
import tempfile
from pathlib import Path

import pytest


def test_load_canonical_mapping_from_json():
    """测试从 JSON 文件加载角色映射"""
    from hierad.pipeline import load_canonical_mapping

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"mapping": {"Johnny": "TOMMY WISEAU", "Lisa": "JULIETTE"}}, f)
        path = f.name
    try:
        result = load_canonical_mapping(Path(path))
        assert result["Johnny"] == "TOMMY WISEAU"
        assert result["Lisa"] == "JULIETTE"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_canonical_mapping_from_dir():
    """测试从 actor_db 目录加载"""
    from hierad.pipeline import load_canonical_mapping

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "config.json").write_text(
            json.dumps({"mapping": {"A": "B"}}, ensure_ascii=False)
        )
        result = load_canonical_mapping(p)
        assert result["A"] == "B"


def test_load_canonical_mapping_nonexistent():
    """测试不存在的路径"""
    from hierad.pipeline import load_canonical_mapping

    assert load_canonical_mapping(Path("/nonexistent/path")) == {}
    assert load_canonical_mapping(None) == {}
