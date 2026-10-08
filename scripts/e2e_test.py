#!/usr/bin/env python3
"""
HierAD 端到端测试 v2

流程:
  1. ASR
  2. Pipeline: 全片标注 → 场景切分 → Stage1 → Stage2(gap AD) → Stage3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("请先安装 tqdm: pip install tqdm", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hierad.pipeline import run_pipeline
from hierad.preprocess import transcribe_video, load_ad_gap_segments, detect_scene_ranges
from hierad.progress import ProgressReporter, ProgressUpdate, progress_scope

# 作者原服务器目录（MAD-EVAL 数据集）
MAD_VIDEO_DIR = Path("/path/to/data/MAD/MAD-EVAL/Videos")

# 本机片段：用 key 当 --movie 传，如 --movie spiderman4
LOCAL_VIDEOS: Dict[str, Path] = {
    "spiderman4": Path(
        r"D:\Program Files\JiJiDown\Download\《蜘蛛侠4：崭新之日》片段\1.《蜘蛛侠4：崭新之日》片段.mp4"
    ),
}

MOVIE_ACTOR_DB: Dict[str, str] = {
    "1005_Signs": "bdc8c048-2d95-457f-b86c-37e4ccc373c6",
}

DEFAULT_MOVIE = "spiderman4"


def resolve_video(movie: str) -> Path:
    if movie in LOCAL_VIDEOS:
        path = LOCAL_VIDEOS[movie]
    elif Path(movie).suffix:  # 直接传视频路径
        path = Path(movie)
    else:
        path = MAD_VIDEO_DIR / f"{movie}.mp4"
    if not path.exists():
        raise SystemExit(f"视频不存在: {path}")
    return path


def resolve_actor_db(movie_id: str, canonical_characters: Optional[str]) -> Optional[Path]:
    if canonical_characters:
        p = Path(canonical_characters)
        return p if p.exists() else None
    if movie_id in MOVIE_ACTOR_DB:
        p = ROOT / "actor_databases" / MOVIE_ACTOR_DB[movie_id]
        return p if p.exists() else None
    return None


def _write_progress_file(path: Path, update: ProgressUpdate, *, min_interval: float = 2.0) -> None:
    now = update.ts
    state = getattr(_write_progress_file, "_state", None)
    if state is None:
        state = {"last": 0.0, "stage": ""}
        _write_progress_file._state = state  # type: ignore[attr-defined]

    stage_changed = update.stage != state["stage"]
    due = update.done or stage_changed or (now - state["last"]) >= min_interval
    if not due:
        return

    state["last"] = now
    state["stage"] = update.stage
    payload = asdict(update)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_asr(
    video: Path,
    work_dir: Path,
    whisper_path: Optional[str],
    skip_transcribe: bool,
    asr_backend: str,
    asr_url: Optional[str],
    reporter: ProgressReporter,
) -> Path:
    if whisper_path:
        p = Path(whisper_path)
        if not p.exists():
            raise SystemExit(f"ASR 文件不存在: {p}")
        reporter.stage("ASR", message="外部文件", done=True, data={"path": str(p)})
        return p

    out = work_dir / f"{video.stem}_asr.json"
    if skip_transcribe and out.exists():
        reporter.stage("ASR", message="复用缓存", done=True, data={"path": str(out)})
        return out

    def on_asr(label: str) -> None:
        reporter.stage("ASR", message=label)

    with progress_scope(reporter, "ASR", message="Whisper 转录"):
        asr_result = transcribe_video(
            video_path=str(video),
            output_json_path=str(out),
            work_dir=str(work_dir),
            asr_backend=asr_backend,  # type: ignore[arg-type]
            asr_url=asr_url,
            progress_callback=on_asr,
        )
    reporter.stage("ASR", done=True, data={"path": str(out), "segment_count": len(asr_result)})
    tqdm.write(f"[ASR] {len(asr_result)} 段 → {out}")
    return out


def run_preprocess_preview(
    asr_path: Path,
    video: Path,
    annotated: Optional[Path],
    min_gap: float,
    max_gap_duration: Optional[float],
    reporter: ProgressReporter,
) -> Tuple[int, int]:
    with progress_scope(reporter, "预处理", message="场景/Gap 统计"):
        gaps, asr, dur = load_ad_gap_segments(
            str(asr_path), str(video), min_gap_sec=min_gap, margin_sec=0.25
        )
        src = str(annotated) if annotated and annotated.exists() else str(video)
        try:
            scenes = detect_scene_ranges(src)
        except Exception as e:
            tqdm.write(f"[Preview] 场景检测失败: {e}")
            scenes = []

    if max_gap_duration is not None:
        gaps = [(s, e) for s, e in gaps if (e - s) <= max_gap_duration]

    preview = {
        "video_min": round(dur / 60, 1),
        "asr_segments": len(asr.segments),
        "scene_estimate": len(scenes),
        "ad_gaps": len(gaps),
        "min_gap_sec": min_gap,
    }
    reporter.stage("预处理", done=True, data=preview)
    tqdm.write(
        f"[Preview] 视频 {dur/60:.1f}min | ASR {len(asr.segments)} 段 | "
        f"场景 ~{len(scenes)} | AD gap {len(gaps)} (min_gap={min_gap}s)"
    )
    return len(scenes), len(gaps)


def main():
    parser = argparse.ArgumentParser(description="HierAD 端到端测试 v2")
    parser.add_argument("--movie", default=DEFAULT_MOVIE)
    parser.add_argument("--video")
    parser.add_argument("--work-dir")
    parser.add_argument("--whisper")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--asr-backend", choices=["auto", "service", "local"], default="auto")
    parser.add_argument("--asr-url")
    parser.add_argument("--skip-annotate", action="store_true", help="跳过全片人脸椭圆标注")
    parser.add_argument("--annotation-threshold", type=float, default=0.2)
    parser.add_argument("--annotation-frame-interval", type=int, default=5)
    parser.add_argument("--face-gpu", type=int, default=2, help="人脸标注 InsightFace GPU 编号（默认 2）")
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument("--gap-margin", type=float, default=0.25)
    parser.add_argument("--max-gap-duration", type=float)
    parser.add_argument("--max-ad-segments", type=int, help="限制 AD 段数（调试）")
    parser.add_argument("--max-scene-clips", type=int, help="限制场景 clip 数（调试）")
    parser.add_argument("--min-scene-sec", type=float, default=5.0)
    parser.add_argument("--max-scene-sec", type=float, default=90.0)
    parser.add_argument("--scene-threshold", type=float, default=27.0)
    parser.add_argument("--canonical-characters")
    parser.add_argument("--vlm-url")
    parser.add_argument("--vlm-type")
    parser.add_argument("--vlm-endpoint")
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    os.environ["HIERAD_FACE_GPU"] = str(args.face_gpu)

    video = Path(args.video) if args.video else resolve_video(args.movie)
    # --movie 给的是 key 时直接用 key 命名 work_dir；--video 给路径时退回文件名
    movie_id = video.stem if args.video else args.movie
    work_dir = Path(args.work_dir or ROOT / "work_dir" / f"e2e_{movie_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    actor_db_path = resolve_actor_db(movie_id, args.canonical_characters)
    actor_db_str = str(actor_db_path) if actor_db_path else None
    annotated_out = work_dir / f"{video.stem}_annotated.mp4"
    progress_path = work_dir / "progress.json"

    tqdm.write(f"=== HierAD E2E v2: {movie_id} ===")
    tqdm.write(f"视频: {video}")
    tqdm.write(f"工作目录: {work_dir}\n")

    reporter = ProgressReporter(
        show=True,
        desc="HierAD",
        callback=lambda u: _write_progress_file(progress_path, u),
    )

    try:
        asr_path = ensure_asr(
            video=video,
            work_dir=work_dir,
            whisper_path=args.whisper,
            skip_transcribe=args.skip_transcribe,
            asr_backend=args.asr_backend,
            asr_url=args.asr_url,
            reporter=reporter,
        )

        scene_n, gap_n = run_preprocess_preview(
            asr_path, video, annotated_out, args.min_gap, args.max_gap_duration, reporter
        )

        if args.preview_only:
            return

        if gap_n == 0:
            raise SystemExit("无可用 AD 间隙")

        timecodes, ad_scripts = run_pipeline(
            video_path=str(video),
            whisper_path=str(asr_path),
            work_dir=work_dir,
            canonical_characters_path=actor_db_str,
            actor_db_path=actor_db_str,
            skip_annotate=args.skip_annotate,
            annotation_threshold=args.annotation_threshold,
            annotation_frame_interval=args.annotation_frame_interval,
            vlm_url=args.vlm_url,
            vlm_type=args.vlm_type,
            vlm_endpoint=args.vlm_endpoint,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            min_gap_sec=args.min_gap,
            gap_margin_sec=args.gap_margin,
            max_gap_duration_sec=args.max_gap_duration,
            max_ad_segments=args.max_ad_segments,
            max_scene_clips=args.max_scene_clips,
            scene_detect_threshold=args.scene_threshold,
            min_scene_sec=args.min_scene_sec,
            max_scene_sec=args.max_scene_sec,
            show_progress=True,
            reporter=reporter,
        )
    finally:
        reporter.close()

    tqdm.write("\n=== 结果 ===")
    tqdm.write("场景 clip (Stage1): 见 stage1_clip_descriptions.json")
    tqdm.write(f"AD 脚本: {len(ad_scripts)} 段")
    if annotated_out.exists():
        tqdm.write(f"标注视频: {annotated_out}")
    for i, (tc, ad) in enumerate(zip(timecodes, ad_scripts[:5])):
        tqdm.write(f"  [{i}] {tc[0]} - {tc[1]}: {ad[:100]}{'...' if len(ad) > 100 else ''}")
    if len(ad_scripts) > 5:
        tqdm.write(f"  ... 共 {len(ad_scripts)} 段")

    summary = {
        "pipeline_version": "v2",
        "movie": movie_id,
        "video": str(video),
        "asr_path": str(asr_path),
        "annotated_video": str(annotated_out) if annotated_out.exists() else None,
        "work_dir": str(work_dir),
        "scene_clip_estimate": scene_n,
        "ad_segments": len(ad_scripts),
        "gap_count": gap_n,
    }
    summary_path = work_dir / "e2e_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tqdm.write(f"\n输出: {work_dir / 'stage3_ad_scripts.json'}")
    tqdm.write(f"摘要: {summary_path}")
    tqdm.write(f"进度: {progress_path}")


if __name__ == "__main__":
    main()
