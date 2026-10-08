"""
从 VD-agent 音色库解析 voice_id（如「康辉」）
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

from hierad.config import PROJECT_ROOT, VOICE_DB_PATH, VOICE_NAME


def default_voice_db() -> Path:
    if VOICE_DB_PATH:
        return Path(VOICE_DB_PATH)
    env = os.getenv("VD_AGENT_DB")
    if env:
        return Path(env)
    return PROJECT_ROOT.parent / "VD-agent-master" / "web_app" / "web_app.db"


def resolve_voice_profile(
    name_or_id: Optional[str] = None,
    *,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    按名称（模糊）或 UUID 查找音色。默认 VOICE_NAME（康辉）。
    返回 {id, name, prompt_text, audio_path} 或 None。
    """
    key = (name_or_id or VOICE_NAME or "").strip()
    if not key:
        return None
    db = Path(db_path) if db_path else default_voice_db()
    if not db.exists():
        raise FileNotFoundError(f"音色库数据库不存在: {db}")

    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        # 精确 id
        row = con.execute(
            "SELECT id, name, prompt_text, audio_path, status FROM voice_profiles WHERE id = ?",
            (key,),
        ).fetchone()
        if row is None:
            # 精确名
            row = con.execute(
                "SELECT id, name, prompt_text, audio_path, status FROM voice_profiles "
                "WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (key,),
            ).fetchone()
        if row is None:
            # 模糊：优先 completed
            row = con.execute(
                "SELECT id, name, prompt_text, audio_path, status FROM voice_profiles "
                "WHERE name LIKE ? ORDER BY "
                "CASE WHEN status = 'completed' THEN 0 ELSE 1 END, created_at DESC LIMIT 1",
                (f"%{key}%",),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"] or "",
            "prompt_text": row["prompt_text"] or "",
            "audio_path": row["audio_path"] or "",
            "status": row["status"] or "",
        }
    finally:
        con.close()


def resolve_voice_id_and_prompt(
    name_or_id: Optional[str] = None,
    *,
    db_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """返回 (voice_id, prompt_text, display_name)。"""
    profile = resolve_voice_profile(name_or_id, db_path=db_path)
    if not profile:
        return None, None, None
    return profile["id"], profile.get("prompt_text") or None, profile.get("name")
