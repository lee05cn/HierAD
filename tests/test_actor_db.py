"""Actor DB 模块测试"""

import json
import tempfile
from pathlib import Path

import pytest

from hierad.actor_db import ActorDatabase


def test_actor_database_load():
    """测试 ActorDatabase 加载"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "config.json").write_text(
            json.dumps({
                "movie_name": "Test",
                "mapping": {"Johnny": "TOMMY WISEAU", "Lisa": "JULIETTE"}
            }, ensure_ascii=False)
        )
        db = ActorDatabase(p)
        db.load()
        assert db.normalize("Johnny") == "TOMMY WISEAU"
        assert db.normalize("johnny") == "TOMMY WISEAU"  # 大小写不敏感
        assert db.mapping["Lisa"] == "JULIETTE"


def test_actor_database_empty():
    """测试空数据库"""
    with tempfile.TemporaryDirectory() as d:
        db = ActorDatabase(Path(d))
        db.load()
        assert db.normalize("X") == "X"
        assert db.mapping == {}
