"""HierAD Web 的任务存储与单工作线程队列。

一次只跑一个任务：8GB 显存的 4060 上 Whisper 与 ffmpeg 并发会互相抢显存，
而本机自用没有吞吐需求。

每个任务有独立工作目录（`webapp/jobs/<job_id>/`），这不是洁癖 ——
`hierad/pipeline.py:261` 和 `:292` 会从 work_dir 读断点文件续跑，
共用目录会让第二个视频直接复用第一个的场景切分和 VLM 描述。
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hierad.config import PROJECT_ROOT, WEB_KEEP_JOBS
from hierad.pipeline import run_pipeline
from hierad.preprocess import transcribe_video
from hierad.progress import ProgressReporter, ProgressUpdate

from . import db

JOBS_DIR = PROJECT_ROOT / "webapp" / "jobs"

MAX_QUEUE = 3          # 排队上限（含正在跑的那个）
KEEP_JOB_DIRS = WEB_KEEP_JOBS  # 保留最近 N 个任务目录
PROGRESS_FLUSH_SEC = 1.0
MIN_GAP_SEC = 2.0

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# 异常信息 → 给人看的提示。按顺序匹配，先命中先用。
_FRIENDLY_ERRORS: List[tuple] = [
    (
        "未找到满足",
        "这段视频对白太连续，找不到足够的静音间隙插入解说。"
        "换一段对白之间有停顿的片段试试。",
    ),
    (
        "音色库数据库不存在",
        "TTS 音色解析失败：本机音色库不存在。"
        "请确认 config.yaml 里 tts.provider 设为 dashscope。",
    ),
    (
        "pydub",
        "TTS 混音缺少 pydub，请先执行 pip install pydub。",
    ),
    (
        "AD 视频压制失败",
        "成片压制失败（ffmpeg）。字幕已生成，详见任务目录下的日志。",
    ),
    (
        "No module named 'dashscope'",
        "云端 TTS 缺少 dashscope 包，请先执行 pip install dashscope。",
    ),
]


def _friendly(message: str) -> str:
    for needle, friendly in _FRIENDLY_ERRORS:
        if needle in message:
            return friendly
    return message


@dataclass
class Job:
    job_id: str
    filename: str
    video_path: str
    work_dir: str
    duration_sec: float = 0.0
    status: str = STATUS_QUEUED
    stage: str = "排队中"
    current: Optional[int] = None
    total: Optional[int] = None
    message: str = ""
    error: str = ""
    asr_segments: int = 0
    ad_count: int = 0
    ad_video: Optional[str] = None
    ad_srt: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    user_id: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_flush: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "duration_sec": round(self.duration_sec, 1),
            "status": self.status,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "asr_segments": self.asr_segments,
            "ad_count": self.ad_count,
            "has_video": bool(self.ad_video),
            "has_srt": bool(self.ad_srt),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed_sec": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 1
            ),
        }

    def to_db_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "video_path": self.video_path,
            "work_dir": self.work_dir,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "asr_segments": self.asr_segments,
            "ad_count": self.ad_count,
            "ad_video": self.ad_video,
            "ad_srt": self.ad_srt,
            "tts_voice": self.tts_voice,
            "tts_speed": self.tts_speed,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": time.time(),
        }


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue(maxsize=MAX_QUEUE)
_worker_started = False


# ==================== 对外接口 ====================


class QueueFull(RuntimeError):
    """排队已满，拒绝新任务。"""


# ==================== 持久化 / 行转换 ====================

_persist_warned: set = set()


def _persist(job: Job) -> None:
    """把任务快照写进 SQLite。失败不打断管线，但每个任务只提示一次。"""
    with _lock:
        payload = job.to_db_dict()
    try:
        db.upsert_job(payload)
    except Exception as e:  # sqlite3.Error / OSError
        with _lock:
            already = job.job_id in _persist_warned
            _persist_warned.add(job.job_id)
        if not already:
            print(f"[webapp] 任务 {job.job_id} 持久化失败：{type(e).__name__}: {e}")


def _row_to_job(row: Dict[str, Any]) -> Job:
    return Job(
        job_id=str(row["job_id"]),
        filename=str(row.get("filename") or ""),
        video_path=str(row.get("video_path") or ""),
        work_dir=str(row.get("work_dir") or ""),
        duration_sec=float(row.get("duration_sec") or 0),
        status=str(row.get("status") or STATUS_QUEUED),
        stage=str(row.get("stage") or ""),
        current=row.get("current"),
        total=row.get("total"),
        message=str(row.get("message") or ""),
        error=str(row.get("error") or ""),
        asr_segments=int(row.get("asr_segments") or 0),
        ad_count=int(row.get("ad_count") or 0),
        ad_video=row.get("ad_video"),
        ad_srt=row.get("ad_srt"),
        tts_voice=row.get("tts_voice"),
        tts_speed=row.get("tts_speed"),
        user_id=row.get("user_id"),
        created_at=float(row.get("created_at") or time.time()),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        last_flush=0.0,
    )


def _row_to_api(row: Dict[str, Any], include_owner: bool = False) -> Dict[str, Any]:
    started = row.get("started_at") or row.get("created_at") or time.time()
    finished = row.get("finished_at") or time.time()
    data = {
        "job_id": row["job_id"],
        "filename": row.get("filename") or "",
        "duration_sec": round(float(row.get("duration_sec") or 0), 1),
        "status": row.get("status") or STATUS_ERROR,
        "stage": row.get("stage") or "",
        "current": row.get("current"),
        "total": row.get("total"),
        "message": row.get("message") or "",
        "error": row.get("error") or "",
        "asr_segments": int(row.get("asr_segments") or 0),
        "ad_count": int(row.get("ad_count") or 0),
        "has_video": bool(row.get("ad_video")),
        "has_srt": bool(row.get("ad_srt")),
        "created_at": row.get("created_at"),
        "finished_at": row.get("finished_at"),
        "elapsed_sec": round(max(0.0, finished - started), 1),
    }
    if include_owner:
        data["owner"] = row.get("owner")
    return data


def get_record(job_id: str) -> Optional[Dict[str, Any]]:
    """取内部记录（含 user_id / 产物路径），活跃任务优先。"""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return job.to_db_dict()
    return db.get_job_row(job_id)


def get_for_user(job_id: str, user: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """带归属校验的状态快照；无权访问与不存在都返回 None。"""
    record = get_record(job_id)
    if record is None:
        return None
    if not user.get("is_admin") and record.get("user_id") != user.get("id"):
        return None
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return job.to_dict()
    row = db.get_job_row(job_id)
    return _row_to_api(row) if row else None


def paths_for_user(
    job_id: str, user: Dict[str, Any]
) -> Optional[tuple]:
    """带归属校验的产物路径；返回 (video, srt) 或 None。"""
    record = get_record(job_id)
    if record is None:
        return None
    if not user.get("is_admin") and record.get("user_id") != user.get("id"):
        return None
    return record.get("ad_video"), record.get("ad_srt")


def list_history(
    user: Dict[str, Any], limit: int = 20, offset: int = 0, scope: str = "own"
) -> List[Dict[str, Any]]:
    owner_id = None if (user.get("is_admin") and scope == "all") else user.get("id")
    rows = db.list_jobs(owner_id, limit=limit, offset=offset)
    items: List[Dict[str, Any]] = []
    for row in rows:
        with _lock:
            live = _jobs.get(str(row["job_id"]))
        item = live.to_dict() if live is not None else _row_to_api(
            row, include_owner=bool(user.get("is_admin"))
        )
        if user.get("is_admin"):
            item["owner"] = row.get("owner")
        items.append(item)
    return items


def recover() -> None:
    """启动时把 DB 状态接回内存。

    - running：进程已死，管线状态不可信，标为中断，让用户重提。
    - queued：输入文件还在就直接重新排队。
    """
    message = "服务重启，任务已中断，请重新提交。"
    db.mark_running_interrupted(message)

    queued = db.queued_job_ids()
    if not queued:
        return

    _ensure_worker()
    for job_id in queued:
        row = db.get_job_row(job_id)
        if not row:
            continue
        job = _row_to_job(row)
        if not Path(job.video_path).exists():
            with _lock:
                job.status = STATUS_ERROR
                job.error = "服务重启，任务已中断（找不到输入文件）。"
                job.finished_at = time.time()
            _persist(job)
            continue
        with _lock:
            _jobs[job_id] = job
        try:
            _queue.put_nowait(job_id)
        except queue.Full:
            with _lock:
                job.status = STATUS_ERROR
                job.error = "服务重启后排队已满，任务未能恢复。"
                job.finished_at = time.time()
            _persist(job)


def submit(
    video_path: Path,
    filename: str,
    duration_sec: float,
    user_id: Optional[int] = None,
    tts_voice: Optional[str] = None,
    tts_speed: Optional[float] = None,
) -> Job:
    """登记任务并入队。

    `video_path` 是上传落地的临时文件，这里把它挪进任务目录 —— 必须**先挪再入队**，
    否则工作线程可能在文件就位前就开始跑。
    """
    _ensure_worker()
    job_id = uuid.uuid4().hex[:12]
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    staged = work_dir / f"input{video_path.suffix.lower()}"
    shutil.move(str(video_path), str(staged))

    job = Job(
        job_id=job_id,
        filename=filename,
        video_path=str(staged),
        work_dir=str(work_dir),
        duration_sec=duration_sec,
        user_id=user_id,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
    )
    with _lock:
        _jobs[job_id] = job

    # 先落库再入队：进程在两步之间崩掉时，至少还有一条 queued 记录可恢复。
    _persist(job)

    try:
        _queue.put_nowait(job_id)
    except queue.Full:
        with _lock:
            _jobs.pop(job_id, None)
        db.delete_job(job_id)
        shutil.rmtree(work_dir, ignore_errors=True)
        raise QueueFull(f"排队已满（上限 {MAX_QUEUE}），等前面的任务跑完再提交。")

    return job


def get(job_id: str) -> Optional[Dict[str, Any]]:
    """取任务状态快照：活跃任务走内存，历史任务回落 SQLite。"""
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            return job.to_dict()
    row = db.get_job_row(job_id)
    return _row_to_api(row) if row else None


def get_job(job_id: str) -> Optional[Job]:
    """取内部 Job 对象（仅活跃任务；历史任务请走 get_record）。"""
    with _lock:
        return _jobs.get(job_id)


def queued_count() -> int:
    return _queue.qsize()


# ==================== 工作线程 ====================


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        threading.Thread(target=_worker_loop, name="hierad-worker", daemon=True).start()
        _worker_started = True


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            with _lock:
                job = _jobs.get(job_id)
            if job is not None:
                _run_job(job)
        except Exception:
            # _run_job 自己吞异常，能到这说明是它之外的意外
            traceback.print_exc()
        finally:
            _queue.task_done()


def _run_job(job: Job) -> None:
    work_dir = Path(job.work_dir)
    with _lock:
        job.status = STATUS_RUNNING
        job.started_at = time.time()
        job.stage = "准备中"
    _persist(job)

    reporter = ProgressReporter(
        show=False,  # 不往服务器终端刷 tqdm
        callback=lambda u: _on_progress(job, u),
    )

    try:
        asr_json = work_dir / f"{Path(job.video_path).stem}_asr.json"

        with _lock:
            job.stage = "ASR 语音识别"

        asr_result = transcribe_video(
            video_path=job.video_path,
            output_json_path=str(asr_json),
            work_dir=str(work_dir),
            model_size="base",
            asr_backend="local",
            progress_callback=lambda label: reporter.stage("ASR", message=label),
        )
        with _lock:
            job.asr_segments = len(asr_result)

        run_pipeline(
            video_path=job.video_path,
            whisper_path=str(asr_json),
            work_dir=work_dir,
            skip_annotate=True,  # 无演员库，跳过 InsightFace 人脸标注
            min_gap_sec=MIN_GAP_SEC,
            reporter=reporter,
            show_progress=False,
            pipeline_version="v3",  # Clip→Shot→Scene 三层：旁白有上下文、有故事线
            # burn_ad / enable_tts / translate_zh 用默认 True → 出成片 + 中文 SRT
            # 音色 / 语速来自上传时的「配音设置」；dashscope 下 tts_voice_name 就是百炼音色名
            tts_voice_name=job.tts_voice,
            tts_speed=job.tts_speed,
        )

        _collect_result(job, work_dir)

    except Exception as e:
        message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        with _lock:
            job.status = STATUS_ERROR
            job.error = _friendly(message)
            job.message = ""
            job.finished_at = time.time()
        _write_progress(job)
        traceback.print_exc()

    _prune_old_jobs()


def _collect_result(job: Job, work_dir: Path) -> None:
    """从 run_pipeline 写的 config.json 里取产物路径。

    比猜文件名可靠：hierad/pipeline.py:431 才会把这些键写进去，
    而且**压制失败时不会有 ad_video_path** —— run_pipeline 内部把压制包在
    try/except 里，失败只打印警告不抛异常，所以必须在这里单独判断。
    """
    cfg_path = work_dir / "config.json"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}

    srt = cfg.get("ad_srt_path")
    srt_ok = bool(srt and Path(srt).exists())
    video = cfg.get("ad_video_path")
    video_ok = bool(video and Path(video).exists())

    with _lock:
        job.ad_count = int(cfg.get("ad_segment_count") or 0)
        job.ad_srt = str(srt) if srt_ok else None
        job.finished_at = time.time()

        if video_ok:
            job.ad_video = str(video)
            job.status = STATUS_DONE
            job.stage = "完成"
        else:
            job.status = STATUS_ERROR
            # pipeline 现在会把真实异常写进 config.json 的 ad_error（含 ffmpeg 原始
            # 报错），有就显示出来，别再让用户只看到一句笼统的"压制失败"。
            detail = str(cfg.get("ad_error") or "").strip()
            job.error = (
                f"字幕已生成，但成片压制失败：\n{detail}"
                if detail
                else "字幕已生成，但成片压制失败（AD 视频没有产出）。"
            )
            job.message = f"详见 {work_dir}"

    _write_progress(job)


# ==================== 进度 ====================


def _on_progress(job: Job, u: ProgressUpdate) -> None:
    """ProgressReporter 的回调，跑在工作线程里。"""
    now = time.time()
    with _lock:
        if u.stage:
            job.stage = u.stage
        job.current = u.current
        job.total = u.total
        if u.message:
            job.message = u.message
        due = u.done or (now - job.last_flush) >= PROGRESS_FLUSH_SEC
        if due:
            job.last_flush = now
    if due:
        _write_progress(job)


def _write_progress(job: Job) -> None:
    """落一份进度快照，便于排查（沿用 scripts/e2e_test.py:72 的写法）。"""
    with _lock:
        payload = job.to_dict()
        payload["video_path"] = job.video_path
    try:
        path = Path(job.work_dir) / "progress.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # 进度快照写不了不该影响主流程
    _persist(job)  # 与文件快照同频（1 秒节流），历史列表据此更新


# ==================== 清理 ====================


def _prune_old_jobs() -> None:
    """保留最近 KEEP_JOB_DIRS 个已结束的任务目录，并同步删除 DB 行。

    排队/运行中的任务不参与淘汰；`_incoming` 是上传中转目录，永远不碰。
    """
    try:
        dirs = [
            d
            for d in JOBS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]
    except OSError:
        return

    with _lock:
        active = {
            job_id
            for job_id, job in _jobs.items()
            if job.status in (STATUS_QUEUED, STATUS_RUNNING)
        }
    candidates = [d for d in dirs if d.name not in active]
    try:
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        return

    removed: List[str] = []
    for d in candidates[KEEP_JOB_DIRS:]:
        shutil.rmtree(d, ignore_errors=True)
        with _lock:
            _jobs.pop(d.name, None)
        removed.append(d.name)

    if removed:
        try:
            db.delete_jobs(removed)
        except Exception as e:  # sqlite3.Error / OSError
            print(f"[webapp] 清理任务记录失败：{type(e).__name__}: {e}")
