"""HierAD Web 的 SQLite 持久化层。

设计取舍：
- 只用标准库 sqlite3，不引入 ORM；连接按操作创建，请求线程与工作线程各用各的。
- WAL + busy_timeout，允许「一个写、多个读」；写频率最高的是任务进度，1 秒一次。
- 本模块只管存取，不碰 FastAPI，也不碰任务调度，便于单测。
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from hierad.config import WEB_DB_PATH

DB_PATH = WEB_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    iterations    INTEGER NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL,
    last_login_at REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    ip         TEXT,
    user_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    filename     TEXT NOT NULL,
    video_path   TEXT NOT NULL,
    work_dir     TEXT NOT NULL,
    duration_sec REAL NOT NULL DEFAULT 0,
    status       TEXT NOT NULL,
    stage        TEXT NOT NULL DEFAULT '',
    current      INTEGER,
    total        INTEGER,
    message      TEXT NOT NULL DEFAULT '',
    error        TEXT NOT NULL DEFAULT '',
    asr_segments INTEGER NOT NULL DEFAULT 0,
    ad_count     INTEGER NOT NULL DEFAULT 0,
    ad_video     TEXT,
    ad_srt       TEXT,
    tts_voice    TEXT,
    tts_speed    REAL,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS tts_voices (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    voice_id   TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    sort       INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

JOB_COLUMNS = (
    "job_id",
    "user_id",
    "filename",
    "video_path",
    "work_dir",
    "duration_sec",
    "status",
    "stage",
    "current",
    "total",
    "message",
    "error",
    "asr_segments",
    "ad_count",
    "ad_video",
    "ad_srt",
    "tts_voice",
    "tts_speed",
    "created_at",
    "started_at",
    "finished_at",
    "updated_at",
)


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 5000")
    return con


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    con = connect()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(db_path: Optional[str] = None) -> None:
    """建库建表；可重复调用。"""
    global DB_PATH
    if db_path:
        DB_PATH = db_path
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = connect()
    try:
        con.execute("PRAGMA journal_mode = WAL")
        con.executescript(SCHEMA)
        # 迁移：老库的 jobs 表没有 tts_voice / tts_speed（CREATE TABLE IF NOT EXISTS 不改已有表）
        job_cols = {row["name"] for row in con.execute("PRAGMA table_info(jobs)").fetchall()}
        if "tts_voice" not in job_cols:
            con.execute("ALTER TABLE jobs ADD COLUMN tts_voice TEXT")
        if "tts_speed" not in job_cols:
            con.execute("ALTER TABLE jobs ADD COLUMN tts_speed REAL")
        con.execute("PRAGMA user_version = 1")
        con.commit()
    finally:
        con.close()


# ==================== 用户 / 会话 ====================


def _user_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def count_users() -> int:
    with connection() as con:
        row = con.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])


def create_user(
    username: str, password_hash: str, salt: str, iterations: int
) -> Dict[str, Any]:
    """创建用户。库中第一个用户自动成为管理员。

    用 BEGIN IMMEDIATE 把「数人数 + 插入」放在一个写事务里，避免并发注册时
    两个请求同时认为自己才是第一个。
    """
    con = connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        is_admin = 1 if int(row["n"]) == 0 else 0
        now = time.time()
        cur = con.execute(
            "INSERT INTO users (username, password_hash, salt, iterations, is_admin, "
            "is_active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (username, password_hash, salt, iterations, is_admin, now),
        )
        user_id = int(cur.lastrowid)
        con.commit()
        return {
            "id": user_id,
            "username": username,
            "is_admin": bool(is_admin),
            "is_active": True,
            "created_at": now,
            "last_login_at": None,
        }
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with connection() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with connection() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_dict(row) if row else None


def set_last_login(user_id: int, when: Optional[float] = None) -> None:
    with connection() as con:
        con.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (when or time.time(), user_id),
        )


def create_session(
    token_hash: str,
    user_id: int,
    expires_at: float,
    ip: str = "",
    user_agent: str = "",
) -> None:
    with connection() as con:
        con.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, ip, user_agent) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (token_hash, user_id, time.time(), expires_at, ip, user_agent),
        )


def get_session_user(token_hash: str) -> Optional[Dict[str, Any]]:
    with connection() as con:
        row = con.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ? AND u.is_active = 1",
            (token_hash, time.time()),
        ).fetchone()
        return _user_dict(row) if row else None


def delete_session(token_hash: str) -> None:
    with connection() as con:
        con.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def purge_expired_sessions() -> int:
    with connection() as con:
        cur = con.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
        return int(cur.rowcount or 0)


# ==================== 任务 ====================


def upsert_job(job: Dict[str, Any]) -> None:
    """按 job_id 插入或更新。调用方负责把 JOB_COLUMNS 填全。"""
    data = {key: job.get(key) for key in JOB_COLUMNS}
    data["updated_at"] = job.get("updated_at") or time.time()
    placeholders = ", ".join("?" for _ in JOB_COLUMNS)
    updates = ", ".join(f"{key} = excluded.{key}" for key in JOB_COLUMNS if key != "job_id")
    sql = (
        f"INSERT INTO jobs ({', '.join(JOB_COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(job_id) DO UPDATE SET {updates}"
    )
    with connection() as con:
        con.execute(sql, tuple(data[key] for key in JOB_COLUMNS))


def get_job_row(job_id: str) -> Optional[Dict[str, Any]]:
    with connection() as con:
        row = con.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(
    owner_id: Optional[int], limit: int = 20, offset: int = 0
) -> List[Dict[str, Any]]:
    """owner_id=None 表示不过滤（管理员视角）。"""
    limit = min(max(int(limit), 1), 100)
    offset = max(int(offset), 0)
    with connection() as con:
        if owner_id is None:
            rows = con.execute(
                "SELECT j.*, u.username AS owner FROM jobs j "
                "LEFT JOIN users u ON u.id = j.user_id "
                "ORDER BY j.created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT j.*, u.username AS owner FROM jobs j "
                "LEFT JOIN users u ON u.id = j.user_id "
                "WHERE j.user_id = ? ORDER BY j.created_at DESC LIMIT ? OFFSET ?",
                (owner_id, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]


def queued_job_ids() -> List[str]:
    with connection() as con:
        rows = con.execute(
            "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC"
        ).fetchall()
        return [str(row["job_id"]) for row in rows]


def mark_running_interrupted(message: str) -> int:
    now = time.time()
    with connection() as con:
        cur = con.execute(
            "UPDATE jobs SET status = 'error', error = ?, message = '', "
            "finished_at = ?, updated_at = ? WHERE status = 'running'",
            (message, now, now),
        )
        return int(cur.rowcount or 0)


def delete_job(job_id: str) -> None:
    with connection() as con:
        con.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))


def delete_jobs(job_ids: Iterable[str]) -> None:
    ids = [str(x) for x in job_ids]
    if not ids:
        return
    with connection() as con:
        con.executemany("DELETE FROM jobs WHERE job_id = ?", [(x,) for x in ids])


# ==================== TTS 音色库 ====================

# 百炼 CosyVoice v2 预置音色（已实测可用；其余音色可在网页管理里自行添加）
TTS_VOICE_SEED = (
    ("龙小淳", "longxiaochun_v2", "温柔女声 · 默认"),
    ("龙婉", "longwan_v2", "知性女声"),
)


def seed_tts_voices() -> None:
    """首次启动时写入预置音色；已有数据则跳过。可重复调用。"""
    with connection() as con:
        row = con.execute("SELECT COUNT(*) AS n FROM tts_voices").fetchone()
        if int(row["n"]) > 0:
            return
        now = time.time()
        for i, (name, voice_id, note) in enumerate(TTS_VOICE_SEED):
            con.execute(
                "INSERT INTO tts_voices (name, voice_id, note, sort, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, voice_id, note, i, now),
            )


def list_tts_voices() -> List[Dict[str, Any]]:
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM tts_voices ORDER BY sort ASC, id ASC"
        ).fetchall()
        return [dict(row) for row in rows]


def upsert_tts_voice(
    name: str,
    voice_id: str,
    note: str = "",
    sort: int = 0,
    voice_row_id: Optional[int] = None,
) -> Dict[str, Any]:
    """新增或更新一条音色。voice_row_id 给定时更新；否则按 name 去重（同名改为更新）。"""
    name = (name or "").strip()
    voice_id = (voice_id or "").strip()
    if not name or not voice_id:
        raise ValueError("音色显示名与百炼音色名都不能为空。")
    note = (note or "").strip()
    sort = int(sort)
    now = time.time()
    with connection() as con:
        if voice_row_id is not None:
            cur = con.execute(
                "UPDATE tts_voices SET name = ?, voice_id = ?, note = ?, sort = ? "
                "WHERE id = ?",
                (name, voice_id, note, sort, voice_row_id),
            )
            if cur.rowcount == 0:
                raise ValueError("要更新的音色不存在。")
            row = con.execute(
                "SELECT * FROM tts_voices WHERE id = ?", (voice_row_id,)
            ).fetchone()
            return dict(row)
        existing = con.execute(
            "SELECT id FROM tts_voices WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if existing:
            rid = int(existing["id"])
            con.execute(
                "UPDATE tts_voices SET voice_id = ?, note = ?, sort = ? WHERE id = ?",
                (voice_id, note, sort, rid),
            )
            row = con.execute("SELECT * FROM tts_voices WHERE id = ?", (rid,)).fetchone()
            return dict(row)
        cur = con.execute(
            "INSERT INTO tts_voices (name, voice_id, note, sort, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, voice_id, note, sort, now),
        )
        row = con.execute(
            "SELECT * FROM tts_voices WHERE id = ?", (int(cur.lastrowid),)
        ).fetchone()
        return dict(row)


def delete_tts_voice(voice_row_id: int) -> None:
    with connection() as con:
        con.execute("DELETE FROM tts_voices WHERE id = ?", (voice_row_id,))


# ==================== 旧任务导入 ====================


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _find_input(work_dir: Path) -> Optional[Path]:
    for path in sorted(work_dir.glob("input.*")):
        if path.is_file():
            return path
    return None


def import_legacy_jobs(jobs_dir: Union[str, Path]) -> int:
    """把鉴权之前 `webapp/jobs/<id>/progress.json` 的任务导入 DB。

    旧任务没有 owner，`user_id=NULL`，只有管理员能在历史里看到。
    用 INSERT OR IGNORE，重复调用不会覆盖新任务。
    """
    root = Path(jobs_dir)
    if not root.exists():
        return 0

    imported = 0
    for work_dir in sorted(root.iterdir()):
        if not work_dir.is_dir() or work_dir.name.startswith("_"):
            continue
        progress_path = work_dir / "progress.json"
        if not progress_path.exists():
            continue
        data = _read_json(progress_path)
        if not data:
            continue

        job_id = str(data.get("job_id") or work_dir.name)
        input_path = _find_input(work_dir)
        video_path = str(data.get("video_path") or (input_path or work_dir / "input.mp4"))
        cfg = _read_json(work_dir / "config.json")
        status = str(data.get("status") or "error")
        message = ""
        error = str(data.get("error") or cfg.get("ad_error") or "")
        started_at = None
        finished_at = None
        mtime = work_dir.stat().st_mtime
        if status in ("queued", "running"):
            status = "error"
            error = "服务重启前未完成，请重新提交。"
        else:
            finished_at = mtime

        ad_video = cfg.get("ad_video_path")
        ad_srt = cfg.get("ad_srt_path")
        if ad_video and not Path(ad_video).exists():
            ad_video = None
        if ad_srt and not Path(ad_srt).exists():
            ad_srt = None

        now = time.time()
        job = {
            "job_id": job_id,
            "user_id": None,
            "filename": str(data.get("filename") or Path(video_path).name),
            "video_path": video_path,
            "work_dir": str(work_dir),
            "duration_sec": float(data.get("duration_sec") or 0),
            "status": status,
            "stage": str(data.get("stage") or ""),
            "current": data.get("current"),
            "total": data.get("total"),
            "message": message,
            "error": error,
            "asr_segments": int(data.get("asr_segments") or 0),
            "ad_count": int(data.get("ad_count") or 0),
            "ad_video": ad_video,
            "ad_srt": ad_srt,
            "created_at": mtime,
            "started_at": started_at,
            "finished_at": finished_at,
            "updated_at": now,
        }

        with connection() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO jobs ({}) VALUES ({})".format(
                    ", ".join(JOB_COLUMNS),
                    ", ".join("?" for _ in JOB_COLUMNS),
                ),
                tuple(job[key] for key in JOB_COLUMNS),
            )
            imported += 1 if cur.rowcount else 0

    return imported
