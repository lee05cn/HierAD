"""webapp/db.py 的单元测试：建库、用户、会话、任务、旧任务导入。

全部使用临时数据库，不碰真实 webapp/hierad_web.db。
"""

import json
import time

import pytest

from webapp import db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "test_web.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db()
    return path


def _job_row(job_id, user_id, status="done", **overrides):
    now = time.time()
    row = {
        "job_id": job_id,
        "user_id": user_id,
        "filename": f"{job_id}.mp4",
        "video_path": f"/tmp/{job_id}.mp4",
        "work_dir": f"/tmp/{job_id}",
        "duration_sec": 65.0,
        "status": status,
        "stage": "完成",
        "current": 5,
        "total": 5,
        "message": "",
        "error": "",
        "asr_segments": 12,
        "ad_count": 6,
        "ad_video": f"/tmp/{job_id}/ad.mp4",
        "ad_srt": f"/tmp/{job_id}/ad.srt",
        "created_at": now,
        "started_at": now,
        "finished_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


def test_init_db_creates_tables(temp_db):
    with db.connection() as con:
        names = {
            row["name"]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {"users", "sessions", "jobs", "meta"} <= names


def test_first_user_is_admin(temp_db):
    first = db.create_user("alice", "hash-a", "salt-a", 1000)
    second = db.create_user("bob", "hash-b", "salt-b", 1000)

    assert first["is_admin"] is True
    assert second["is_admin"] is False
    assert db.count_users() == 2


def test_username_is_case_insensitive(temp_db):
    db.create_user("Alice", "h", "s", 1000)
    assert db.get_user_by_username("alice")["username"] == "Alice"
    assert db.get_user_by_username("ALICE") is not None


def test_duplicate_username_rejected(temp_db):
    db.create_user("alice", "h", "s", 1000)
    with pytest.raises(Exception):
        db.create_user("alice", "h2", "s2", 1000)


def test_session_lifecycle(temp_db):
    user = db.create_user("alice", "h", "s", 1000)

    db.create_session("token-live", user["id"], time.time() + 60)
    assert db.get_session_user("token-live")["id"] == user["id"]

    db.create_session("token-dead", user["id"], time.time() - 1)
    assert db.get_session_user("token-dead") is None

    db.create_session("token-gone", user["id"], time.time() + 60)
    db.delete_session("token-gone")
    assert db.get_session_user("token-gone") is None

    assert db.purge_expired_sessions() == 1
    assert db.get_session_user("token-live") is not None


def test_inactive_user_session_rejected(temp_db):
    user = db.create_user("alice", "h", "s", 1000)
    db.create_session("token", user["id"], time.time() + 60)
    with db.connection() as con:
        con.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user["id"],))
    assert db.get_session_user("token") is None


def test_job_upsert_and_owner_filter(temp_db):
    owner = db.create_user("alice", "h", "s", 1000)
    other = db.create_user("bob", "h", "s", 1000)

    db.upsert_job(_job_row("j1", owner["id"]))
    db.upsert_job(_job_row("j2", owner["id"], status="running"))

    row = db.get_job_row("j1")
    assert row["status"] == "done"
    assert row["user_id"] == owner["id"]

    # upsert 覆盖同一条记录
    db.upsert_job(_job_row("j2", owner["id"], status="error", error="boom"))
    assert db.get_job_row("j2")["error"] == "boom"

    own = db.list_jobs(owner["id"])
    assert [r["job_id"] for r in own] == ["j2", "j1"]  # 按 created_at 倒序

    assert all(r["job_id"] != "j1" for r in db.list_jobs(other["id"]))
    assert {r["job_id"] for r in db.list_jobs(None)} == {"j1", "j2"}
    assert db.list_jobs(None)[0]["owner"] == "alice"


def test_queued_ids_and_interrupt(temp_db):
    owner = db.create_user("alice", "h", "s", 1000)
    db.upsert_job(_job_row("q1", owner["id"], status="queued"))
    db.upsert_job(_job_row("q2", owner["id"], status="queued"))
    db.upsert_job(_job_row("r1", owner["id"], status="running"))

    assert sorted(db.queued_job_ids()) == ["q1", "q2"]
    assert db.mark_running_interrupted("中断") == 1
    assert db.get_job_row("r1")["status"] == "error"
    assert db.get_job_row("q1")["status"] == "queued"


def test_delete_jobs(temp_db):
    owner = db.create_user("alice", "h", "s", 1000)
    db.upsert_job(_job_row("j1", owner["id"]))
    db.upsert_job(_job_row("j2", owner["id"]))

    db.delete_job("j1")
    assert db.get_job_row("j1") is None
    db.delete_jobs(["j2"])
    assert db.list_jobs(None) == []


def test_import_legacy_jobs(temp_db, tmp_path):
    jobs_dir = tmp_path / "jobs"
    done_dir = jobs_dir / "donejob"
    done_dir.mkdir(parents=True)
    (done_dir / "progress.json").write_text(
        json.dumps(
            {
                "job_id": "donejob",
                "filename": "old.mp4",
                "status": "done",
                "asr_segments": 3,
                "ad_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (done_dir / "config.json").write_text(
        json.dumps({"ad_video_path": "", "ad_srt_path": ""}), encoding="utf-8"
    )

    running_dir = jobs_dir / "runningjob"
    running_dir.mkdir()
    (running_dir / "progress.json").write_text(
        json.dumps({"job_id": "runningjob", "status": "running"}), encoding="utf-8"
    )

    assert db.import_legacy_jobs(jobs_dir) == 2
    done = db.get_job_row("donejob")
    assert done["user_id"] is None
    assert done["status"] == "done"
    broken = db.get_job_row("runningjob")
    assert broken["status"] == "error"
    assert "重启" in broken["error"]

    # 重复导入不会覆盖已有行
    assert db.import_legacy_jobs(jobs_dir) == 0
