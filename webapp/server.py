"""HierAD Web 服务（局域网团队版：SQLite 持久化 + 账号鉴权）。

启动： python -m webapp.server
打开： http://<主机>:8077

注意：uvicorn 必须单进程（不要 --workers > 1）。任务队列与登录限流都在进程内存，
多 worker 会各自为政。默认绑定 127.0.0.1；供局域网访问请在 config.yaml 的
`web.host` 设为 `0.0.0.0`（或用 HIERAD_WEB_HOST 环境变量）。
"""

from __future__ import annotations

import hmac
import json
import socket
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from hierad.config import (
    WEB_ALLOW_REGISTRATION,
    WEB_HOST,
    WEB_PORT,
    WEB_REGISTRATION_CODE,
    get_dashscope_api_key,
    get_tts_provider,
    set_dashscope_api_key,
)
from hierad.preprocess import get_video_duration

from . import auth, db, jobs

STATIC_DIR = Path(__file__).parent / "static"
INCOMING_DIR = jobs.JOBS_DIR / "_incoming"

HOST = WEB_HOST
PORT = WEB_PORT

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_DURATION_SEC = 330.0  # 目标片段 3–5 分钟；留 30 秒余量，免得正好 5:00 的被卡掉
MIN_DURATION_SEC = 8.0    # 下限故意放宽，方便拿短片段冒烟测试（正常输入是 3–5 分钟）
CHUNK = 1024 * 1024

# 不带登录也能访问的路径：宣传页、登录页、静态资源、认证接口
PUBLIC_PAGES = {"/", "/features", "/landing", "/login"}
PUBLIC_API = {"/api/auth/login", "/api/auth/register", "/api/auth/config"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    db.seed_tts_voices()
    db.purge_expired_sessions()
    try:
        imported = db.import_legacy_jobs(jobs.JOBS_DIR)
        if imported:
            print(f"[webapp] 已导入 {imported} 个鉴权之前的旧任务（仅管理员可见）")
    except Exception as e:  # 旧任务导入失败不该拦住服务启动
        print(f"[webapp] 旧任务导入失败：{type(e).__name__}: {e}")
    jobs.recover()
    yield


app = FastAPI(
    title="HierAD",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# 样式与脚本（static/app.css, static/app.js），所有页面共用
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================== 中间件：会话解析 / 访问控制 / CSRF ====================


def _is_public(path: str) -> bool:
    if path in PUBLIC_PAGES or path in PUBLIC_API:
        return True
    return path.startswith("/static/")


def _same_origin(request: Request) -> bool:
    """浏览器带了 Origin 就必须与 Host 同源，否则视为跨站写请求。"""
    origin = request.headers.get("origin")
    if not origin:
        return True  # curl / 同源导航；SameSite=Lax 仍是兜底
    host = request.headers.get("host", "").lower()
    try:
        netloc = urlsplit(origin).netloc.lower()
    except ValueError:
        return False
    return bool(host) and netloc == host


@app.middleware("http")
async def access_control(request: Request, call_next):
    path = request.url.path
    # 静态资源不必每次查会话；登录页需要知道「已登录就送回工作台」
    request.state.user = (
        auth.get_current_user(request) if (not _is_public(path) or path == "/login") else None
    )

    if request.method in UNSAFE_METHODS and not _same_origin(request):
        return JSONResponse({"detail": "跨站请求被拒绝。"}, status_code=403)

    if _is_public(path):
        return await call_next(request)

    user: Optional[Dict[str, Any]] = request.state.user

    if path == "/app":
        if not user:
            return RedirectResponse("/login?next=/app", status_code=302)
        return await call_next(request)

    if path.startswith("/api/"):
        if not user:
            return JSONResponse({"detail": "请先登录。"}, status_code=401)
        return await call_next(request)

    return await call_next(request)


# ==================== 页面 ====================


def _page(name: str) -> FileResponse:
    page = STATIC_DIR / name
    if not page.exists():
        raise HTTPException(500, f"缺少 static/{name}")
    return FileResponse(page, media_type="text/html")


@app.get("/")
def landing() -> FileResponse:
    """首页 = 宣传页：项目介绍、优势。从这里进生成页。"""
    return _page("landing.html")


@app.get("/app")
def app_page() -> FileResponse:
    """生成页：上传视频、看进度、下载成片（需登录）。"""
    return _page("index.html")


@app.get("/features")
def features_page() -> FileResponse:
    """`看看它强在哪`：项目介绍页——优势、原理、间隙示意图。"""
    return _page("features.html")


@app.get("/landing")
def landing_alias() -> RedirectResponse:
    """首页改版前的旧地址，重定向过去。"""
    return RedirectResponse("/", status_code=308)


@app.get("/login")
def login_page(request: Request) -> FileResponse:
    """登录 / 注册页；已登录的人直接送回工作台。"""
    if getattr(request.state, "user", None):
        return RedirectResponse("/app", status_code=302)  # type: ignore[return-value]
    return _page("login.html")


# ==================== 认证接口 ====================


async def _json_body(request: Request) -> Dict[str, Any]:
    try:
        data = json.loads(await request.body() or b"{}")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "请求体不是合法的 JSON。")
    if not isinstance(data, dict):
        raise HTTPException(400, "请求体必须是 JSON 对象。")
    return data


@app.get("/api/auth/config")
def auth_config() -> JSONResponse:
    """登录页据此决定是否显示注册 Tab / 邀请码输入框。"""
    return JSONResponse(
        {
            "allow_registration": bool(WEB_ALLOW_REGISTRATION),
            "require_code": bool(WEB_REGISTRATION_CODE),
            "has_users": db.count_users() > 0,
        }
    )


@app.post("/api/auth/register")
async def register(request: Request) -> JSONResponse:
    if not WEB_ALLOW_REGISTRATION:
        raise HTTPException(403, "本服务已关闭自助注册。")

    key = auth.register_rate_key(request)
    if not auth.allow_request(key, auth.REGISTER_MAX, auth.REGISTER_WINDOW_SEC):
        raise HTTPException(429, "注册过于频繁，请稍后再试。")

    data = await _json_body(request)
    if WEB_REGISTRATION_CODE:
        code = str(data.get("code") or "")
        if not hmac.compare_digest(code, WEB_REGISTRATION_CODE):
            raise HTTPException(403, "邀请码不正确。")

    try:
        user = auth.register_user(str(data.get("username") or ""), str(data.get("password") or ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except auth.UsernameTaken as e:
        raise HTTPException(409, str(e))

    token = auth.create_session(int(user["id"]), request)
    response = JSONResponse(auth.public_user(user), status_code=201)
    auth.set_session_cookie(response, token)
    return response


@app.post("/api/auth/login")
async def login(request: Request) -> JSONResponse:
    key = auth.login_rate_key(request)
    if not auth.allow_request(key, auth.LOGIN_MAX_FAILS, auth.LOGIN_WINDOW_SEC):
        raise HTTPException(429, "尝试次数过多，请 5 分钟后再试。")

    data = await _json_body(request)
    user = auth.authenticate(str(data.get("username") or ""), str(data.get("password") or ""))
    if user is None:
        # 不区分「用户不存在」和「密码错误」
        raise HTTPException(401, "用户名或密码错误。")

    auth.reset_rate(key)
    db.purge_expired_sessions()
    token = auth.create_session(int(user["id"]), request)
    response = JSONResponse(auth.public_user(user))
    auth.set_session_cookie(response, token)
    return response


@app.post("/api/auth/logout")
def logout(request: Request) -> JSONResponse:
    auth.revoke_session(request)
    response = JSONResponse({"ok": True})
    auth.clear_session_cookie(response)
    return response


@app.get("/api/me")
def me(request: Request) -> JSONResponse:
    user = auth.require_user(request)
    return JSONResponse(auth.public_user(user))


# ==================== 管理接口（仅管理员） ====================


def _mask_key(key: str) -> str:
    """只暴露前后几位的打码形式，绝不把完整 key 发回前端。"""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "••••••"
    return f"{key[:6]}…{key[-4:]}"


@app.get("/api/admin/api_key")
def get_api_key(request: Request) -> JSONResponse:
    auth.require_admin(request)
    key = get_dashscope_api_key()
    return JSONResponse({"configured": bool(key), "masked": _mask_key(key)})


@app.post("/api/admin/api_key")
async def set_api_key(request: Request) -> JSONResponse:
    auth.require_admin(request)
    data = await _json_body(request)
    key = str(data.get("api_key") or "").strip()
    if not key:
        raise HTTPException(400, "API Key 不能为空。")
    if len(key) > 512:
        raise HTTPException(400, "API Key 长度异常（超过 512 字符）。")
    persisted = set_dashscope_api_key(key)
    return JSONResponse({"ok": True, "masked": _mask_key(key), "persisted": persisted})


# ==================== TTS 音色库 ====================


@app.get("/api/tts/voices")
def tts_voices(request: Request) -> JSONResponse:
    """音色库列表（登录即可读）。"""
    auth.require_user(request)
    voices = db.list_tts_voices()
    return JSONResponse({"voices": voices, "provider": get_tts_provider()})


@app.post("/api/tts/voices")
async def tts_voice_upsert(request: Request) -> JSONResponse:
    """新增 / 更新音色（仅管理员）。"""
    auth.require_admin(request)
    data = await _json_body(request)
    try:
        voice = db.upsert_tts_voice(
            str(data.get("name") or ""),
            str(data.get("voice_id") or ""),
            note=str(data.get("note") or ""),
            sort=int(data.get("sort") or 0),
            voice_row_id=data.get("id"),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return JSONResponse(voice, status_code=201)


@app.delete("/api/tts/voices/{voice_id}")
def tts_voice_delete(voice_id: int, request: Request) -> JSONResponse:
    """删除音色（仅管理员）。"""
    auth.require_admin(request)
    db.delete_tts_voice(voice_id)
    return JSONResponse({"ok": True})


@app.post("/api/tts/preview")
async def tts_preview(request: Request) -> Response:
    """试听：合成一句短样例并返回音频字节（会真调一次百炼 TTS，消耗少量额度）。"""
    auth.require_user(request)
    data = await _json_body(request)
    voice = str(data.get("voice") or "").strip()
    text = str(data.get("text") or "你好，这是音色试听。").strip()
    speed_raw = data.get("speed")
    speed = None
    if speed_raw not in (None, ""):
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            raise HTTPException(400, "语速必须是数字。")

    from hierad.providers.tts import synthesize_preview

    try:
        audio = synthesize_preview(text, voice, speed=speed)
    except ImportError as e:
        raise HTTPException(500, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"试听合成失败：{type(e).__name__}: {e}")

    return Response(content=audio, media_type="audio/mpeg")


# ==================== 任务 ====================


@app.get("/api/jobs")
def job_history(
    request: Request, limit: int = 20, offset: int = 0, scope: str = "own"
) -> JSONResponse:
    """当前用户的任务历史；管理员加 ?scope=all 可看全部。"""
    user = auth.require_user(request)
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    items = jobs.list_history(user, limit=limit, offset=offset, scope=scope)
    return JSONResponse({"jobs": items, "limit": limit, "offset": offset})


@app.post("/api/jobs")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    voice: str = Form(""),
    speed: str = Form(""),
) -> JSONResponse:
    user = auth.require_user(request)

    # 配音设置：voice 是百炼音色名（如 longwan_v2），speed 是语速倍率
    tts_voice = voice.strip() or None
    tts_speed: Optional[float] = None
    if speed.strip():
        try:
            tts_speed = float(speed)
        except ValueError:
            raise HTTPException(400, "语速不是数字。")
        if not (0.5 <= tts_speed <= 2.0):
            raise HTTPException(400, "语速需在 0.5–2.0 之间。")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"不支持的格式 {suffix or '(无后缀)'}。"
            f"支持：{'、'.join(sorted(ALLOWED_SUFFIXES))}",
        )

    # 先落到 _incoming，探完时长再决定收不收
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = INCOMING_DIR / f"{uuid.uuid4().hex}{suffix}"

    written = 0
    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限。"
                    )
                out.write(chunk)

        if written == 0:
            raise HTTPException(400, "上传的文件是空的。")

        # 不信任前端报的时长，自己探
        try:
            duration = float(get_video_duration(str(tmp_path)))
        except Exception as e:
            raise HTTPException(
                400, f"读不出这个文件的视频信息，可能不是有效视频。（{type(e).__name__}）"
            ) from e

        if duration > MAX_DURATION_SEC:
            raise HTTPException(
                400,
                f"视频长 {duration / 60:.1f} 分钟，超过 {MAX_DURATION_SEC / 60:.1f} 分钟上限。"
                f"请先剪成 3–5 分钟的片段。",
            )
        if duration < MIN_DURATION_SEC:
            raise HTTPException(
                400, f"视频只有 {duration:.1f} 秒，太短了（至少 {MIN_DURATION_SEC:.0f} 秒）。"
            )

        try:
            job = jobs.submit(
                tmp_path,
                file.filename or tmp_path.name,
                duration,
                user_id=int(user["id"]),
                tts_voice=tts_voice,
                tts_speed=tts_speed,
            )
        except jobs.QueueFull as e:
            raise HTTPException(429, str(e)) from e

    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"保存上传文件失败：{type(e).__name__}: {e}") from e

    return JSONResponse(job.to_dict(), status_code=201)


def _not_found() -> HTTPException:
    # 无权访问与不存在都回 404，避免泄露任务是否存在
    return HTTPException(404, "没有这个任务（可能已被清理）。")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> JSONResponse:
    user = auth.require_user(request)
    snapshot = jobs.get_for_user(job_id, user)
    if snapshot is None:
        raise _not_found()
    return JSONResponse(snapshot)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, request: Request, download: int = 0) -> FileResponse:
    user = auth.require_user(request)
    paths = jobs.paths_for_user(job_id, user)
    if paths is None:
        raise _not_found()
    video = paths[0]
    if not video:
        raise HTTPException(404, "这个任务还没有产出视频。")
    path = Path(video)
    if not path.exists():
        raise HTTPException(404, "视频文件不在了。")
    # FileResponse 自带 Range 支持，播放器能拖动进度条
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"AD_{path.name}" if download else None,
    )


@app.get("/api/jobs/{job_id}/srt")
def job_srt(job_id: str, request: Request) -> FileResponse:
    user = auth.require_user(request)
    paths = jobs.paths_for_user(job_id, user)
    if paths is None:
        raise _not_found()
    srt = paths[1]
    if not srt:
        raise HTTPException(404, "这个任务还没有产出字幕。")
    path = Path(srt)
    if not path.exists():
        raise HTTPException(404, "字幕文件不在了。")
    return FileResponse(
        path, media_type="application/x-subrip", filename="AD_字幕.srt"
    )


def _lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return "127.0.0.1"


def main() -> None:
    import uvicorn

    shown = _lan_ip() if HOST in ("0.0.0.0", "::", "") else HOST
    print(f"HierAD Web  →  http://{shown}:{PORT}")
    print(f"数据库      →  {db.DB_PATH}")
    print(f"任务目录    →  {jobs.JOBS_DIR}")
    if HOST in ("0.0.0.0", "::", ""):
        print("模式        →  局域网可访问（首个注册账号成为管理员）")
    print("提示        →  必须单进程运行，不要给 uvicorn 加 --workers")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
