"""HierAD 本机 Web 服务。

启动： python -m webapp.server
打开： http://127.0.0.1:8077

只绑 127.0.0.1（本机自用，不做鉴权）。要给别人用得先加访问控制。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from hierad.config import PROJECT_ROOT
from hierad.preprocess import get_video_duration

from . import jobs

STATIC_DIR = Path(__file__).parent / "static"
INCOMING_DIR = jobs.JOBS_DIR / "_incoming"

HOST = "127.0.0.1"
PORT = 8077

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_DURATION_SEC = 330.0  # 目标片段 3–5 分钟；留 30 秒余量，免得正好 5:00 的被卡掉
MIN_DURATION_SEC = 8.0    # 下限故意放宽，方便拿短片段冒烟测试（正常输入是 3–5 分钟）
CHUNK = 1024 * 1024

app = FastAPI(title="HierAD", docs_url=None, redoc_url=None)

# 样式与脚本（static/app.css, static/app.js），两个页面共用
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    """生成页：上传视频、看进度、下载成片。"""
    return _page("index.html")


@app.get("/landing")
def landing_alias() -> RedirectResponse:
    """首页改版前的旧地址，重定向过去。"""
    return RedirectResponse("/", status_code=308)


# ==================== 任务 ====================


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> JSONResponse:
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
            job = jobs.submit(tmp_path, file.filename or tmp_path.name, duration)
        except jobs.QueueFull as e:
            raise HTTPException(429, str(e)) from e

    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"保存上传文件失败：{type(e).__name__}: {e}") from e

    return JSONResponse(job.to_dict(), status_code=201)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    snapshot = jobs.get(job_id)
    if snapshot is None:
        raise HTTPException(404, "没有这个任务（可能已被清理）。")
    return JSONResponse(snapshot)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = jobs.get_job(job_id)
    if job is None or not job.ad_video:
        raise HTTPException(404, "这个任务还没有产出视频。")
    path = Path(job.ad_video)
    if not path.exists():
        raise HTTPException(404, "视频文件不在了。")
    # FileResponse 自带 Range 支持，播放器能拖动进度条
    return FileResponse(path, media_type="video/mp4", filename=f"AD_{path.name}")


@app.get("/api/jobs/{job_id}/srt")
def job_srt(job_id: str) -> FileResponse:
    job = jobs.get_job(job_id)
    if job is None or not job.ad_srt:
        raise HTTPException(404, "这个任务还没有产出字幕。")
    path = Path(job.ad_srt)
    if not path.exists():
        raise HTTPException(404, "字幕文件不在了。")
    return FileResponse(
        path, media_type="application/x-subrip", filename="AD_字幕.srt"
    )


def main() -> None:
    import uvicorn

    print(f"HierAD Web  →  http://{HOST}:{PORT}")
    print(f"任务目录    →  {jobs.JOBS_DIR}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
