"""webapp 鉴权与任务归属的接口测试。

用临时数据库 + 临时任务目录，不触发真实流水线；任务行直接写进 DB。
"""

import time

import pytest
from fastapi.testclient import TestClient

from webapp import auth, db, jobs, server


def _job_row(job_id, user_id, status="done"):
    now = time.time()
    return {
        "job_id": job_id,
        "user_id": user_id,
        "filename": f"{job_id}.mp4",
        "video_path": f"/tmp/{job_id}.mp4",
        "work_dir": f"/tmp/{job_id}",
        "duration_sec": 60.0,
        "status": status,
        "stage": "完成",
        "current": 1,
        "total": 1,
        "message": "",
        "error": "",
        "asr_segments": 5,
        "ad_count": 2,
        "ad_video": None,
        "ad_srt": None,
        "created_at": now,
        "started_at": now,
        "finished_at": now,
        "updated_at": now,
    }


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    """每个用例独立 DB 与任务目录；工厂可创建多个互不共享 Cookie 的客户端。"""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "web.db"))
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 1000)
    monkeypatch.setattr(server, "WEB_ALLOW_REGISTRATION", True)
    monkeypatch.setattr(server, "WEB_REGISTRATION_CODE", "")
    auth._rate_events.clear()

    opened = []

    def factory():
        client = TestClient(server.app)
        client.__enter__()
        opened.append(client)
        return client

    yield factory

    for client in opened:
        client.__exit__(None, None, None)
    auth._rate_events.clear()


def register(client, username, password="password123"):
    return client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )


def login(client, username, password="password123"):
    return client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )


def test_register_logs_in_and_me(make_client):
    client = make_client()
    r = register(client, "alice")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["username"] == "alice"
    assert body["is_admin"] is True  # 第一个注册用户

    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["username"] == "alice"

    cookie = r.headers.get("set-cookie", "").lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


def test_unauthenticated_access(make_client):
    client = make_client()

    assert client.get("/api/me").status_code == 401
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs/whatever").status_code == 401

    page = client.get("/app", follow_redirects=False)
    assert page.status_code == 302
    assert page.headers["location"].startswith("/login")


def test_login_logout(make_client):
    client = make_client()
    register(client, "alice")

    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401

    assert login(client, "alice").status_code == 200
    assert client.get("/api/me").status_code == 200

    # 退出后会话立即失效
    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401


def test_login_errors_are_generic(make_client):
    client = make_client()
    register(client, "alice")
    client.post("/api/auth/logout")

    wrong_pw = login(client, "alice", "not-the-password")
    unknown = login(client, "nobody", "whatever123")
    assert wrong_pw.status_code == 401
    assert unknown.status_code == 401
    assert wrong_pw.json()["detail"] == unknown.json()["detail"] == "用户名或密码错误。"


def test_duplicate_registration(make_client):
    client = make_client()
    register(client, "alice")
    assert register(client, "alice").status_code == 409


def test_registration_code(make_client, monkeypatch):
    monkeypatch.setattr(server, "WEB_REGISTRATION_CODE", "letmein")
    client = make_client()

    denied = client.post(
        "/api/auth/register", json={"username": "alice", "password": "password123"}
    )
    assert denied.status_code == 403

    ok = client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "password123", "code": "letmein"},
    )
    assert ok.status_code == 201


def test_job_ownership_and_admin_scope(make_client):
    admin = make_client()
    bob = make_client()
    carol = make_client()

    admin_id = register(admin, "admin").json()["id"]
    bob_id = register(bob, "bob").json()["id"]
    register(carol, "carol")

    db.upsert_job(_job_row("job-bob", bob_id))

    # 自己的任务：可见
    assert bob.get("/api/jobs/job-bob").status_code == 200

    # 别人的任务：404（不泄露存在性）
    assert carol.get("/api/jobs/job-bob").status_code == 404
    assert carol.get("/api/jobs/job-bob/video").status_code == 404

    # 管理员：能看，且历史列表支持 scope=all
    assert admin.get("/api/jobs/job-bob").status_code == 200

    own = admin.get("/api/jobs").json()["jobs"]
    assert all(j["job_id"] != "job-bob" for j in own)

    everything = admin.get("/api/jobs?scope=all").json()["jobs"]
    assert [j["job_id"] for j in everything] == ["job-bob"]
    assert everything[0]["owner"] == "bob"

    # 普通用户忽略 scope=all
    assert carol.get("/api/jobs?scope=all").json()["jobs"] == []

    # 管理员 id 仅用于确认顺序：第一个注册者是 admin
    assert admin_id != bob_id


def test_app_page_after_login(make_client):
    client = make_client()
    register(client, "alice")
    assert client.get("/app").status_code == 200
