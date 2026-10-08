"""
Gap 片段人脸识别 — 为 Stage1 生成角色 hint JSON

仅对 AD 间隙切分后的 clip 做人脸检测/识别，不处理全片。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hierad.progress import progress_iter


def resolve_actor_db_dir(actor_db_path: str) -> Path:
    """解析演员库目录（含 config.json / faiss_index.bin）"""
    p = Path(actor_db_path)
    if not p.exists():
        raise FileNotFoundError(f"演员数据库路径不存在: {p}")

    if p.is_file():
        p = p.parent

    if (p / "config.json").exists():
        return p

    if p.is_dir():
        subdirs = [d for d in p.iterdir() if d.is_dir() and (d / "config.json").exists()]
        if subdirs:
            return subdirs[0]

    raise FileNotFoundError(f"未找到有效演员数据库: {actor_db_path}")


def load_actor_db(actor_db_path: str) -> Any:
    """加载演员向量库，失败返回 None"""
    try:
        from hierad.actor_db import ActorDatabase

        db_dir = resolve_actor_db_dir(actor_db_path)
        db = ActorDatabase(db_dir)
        db.load()
        if not (db_dir / "faiss_index.bin").exists():
            return None
        return db
    except Exception:
        return None


def recognize_faces_in_frame(
    frame: np.ndarray,
    actor_db: Any,
    recognition_threshold: float = 0.4,
    min_det_score: float = 0.35,
    min_match_margin: float = 0.08,
) -> List[Dict[str, Any]]:
    """单帧人脸检测 + 演员识别（显示片内角色名）"""
    from hierad.actor_db.face_processor import detect_faces_in_image

    results: List[Dict[str, Any]] = []
    for face in detect_faces_in_image(frame, min_score=min_det_score):
        embedding = face["embedding"]
        matches = actor_db.search_face(embedding, top_k=2)
        if not matches or matches[0]["similarity"] < recognition_threshold:
            continue
        if (
            len(matches) > 1
            and matches[0]["similarity"] - matches[1]["similarity"] < min_match_margin
        ):
            continue
        meta = matches[0]["metadata"]
        character = actor_db.resolve_face_label(meta)
        results.append({
            "character": character,
            "similarity": float(matches[0]["similarity"]),
            "det_score": float(face["det_score"]),
            "bbox": face["bbox"],
        })
    return results


def _format_hint(char_scores: Dict[str, float]) -> str:
    if not char_scores:
        return ""
    ranked = sorted(char_scores.items(), key=lambda x: x[1], reverse=True)
    return ", ".join(f"{name.upper()} (sim={score:.2f})" for name, score in ranked)


def extract_face_hints_from_clip(
    video_path: str,
    actor_db: Any,
    recognition_threshold: float = 0.4,
    frame_interval: int = 5,
    max_samples: int = 12,
) -> Dict[str, Any]:
    """对单个 gap clip 采样帧并聚合识别结果"""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("需要 opencv-python") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"characters": [], "hint": "", "detections": []}

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    sample_indices: List[int] = []
    step = max(1, frame_interval)
    idx = 0
    while idx < total and len(sample_indices) < max_samples:
        sample_indices.append(idx)
        idx += step
    if not sample_indices:
        sample_indices = [0]

    char_best: Dict[str, float] = {}
    detections: List[Dict[str, Any]] = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        time_sec = frame_idx / fps
        for det in recognize_faces_in_frame(frame, actor_db, recognition_threshold):
            name = det["character"]
            sim = det["similarity"]
            char_best[name] = max(char_best.get(name, 0.0), sim)
            detections.append({
                "time_sec": round(time_sec, 2),
                "character": name,
                "similarity": round(sim, 3),
                "det_score": round(det["det_score"], 3),
            })

    cap.release()
    characters = sorted(char_best.keys(), key=lambda n: char_best[n], reverse=True)
    return {
        "characters": characters,
        "hint": _format_hint(char_best),
        "detections": detections,
    }


def build_face_hints_for_segments(
    segment_paths: List[str],
    timecodes: List[Tuple[str, str]],
    actor_db_path: str,
    recognition_threshold: float = 0.4,
    frame_interval: int = 5,
    max_samples_per_clip: int = 12,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    对所有 gap clip 做人脸识别，返回 JSON 记录与 Stage1 用 hint 字符串列表。

    Returns:
        (records, hint_strings) — hint_strings[i] 为空串表示该段无识别结果
    """
    db = load_actor_db(actor_db_path)
    if db is None:
        return [], [""] * len(segment_paths)

    records: List[Dict[str, Any]] = []
    hints: List[str] = []

    indexed = progress_iter(
        list(enumerate(segment_paths)),
        show=show_progress,
        desc="Gap 人脸 hint",
        unit="seg",
        total=len(segment_paths),
    )
    for i, seg_path in indexed:
        tc = timecodes[i] if i < len(timecodes) else ("", "")
        clip_result = extract_face_hints_from_clip(
            seg_path,
            db,
            recognition_threshold=recognition_threshold,
            frame_interval=frame_interval,
            max_samples=max_samples_per_clip,
        )
        record = {
            "segment_idx": i,
            "timecode": f"{tc[0]} --> {tc[1]}",
            "video_path": seg_path,
            **clip_result,
        }
        records.append(record)
        hints.append(clip_result.get("hint") or "")
        if progress_callback:
            progress_callback(i + 1, len(segment_paths))

    return records, hints


def save_face_hints(records: List[Dict[str, Any]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return output_path


def load_face_hint_strings(hints_path: Path, segment_count: int) -> List[str]:
    """从已保存 JSON 恢复 Stage1 hint 字符串列表"""
    if not hints_path.exists():
        return [""] * segment_count
    with open(hints_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    hints = [""] * segment_count
    for row in records:
        idx = int(row.get("segment_idx", -1))
        if 0 <= idx < segment_count:
            hints[idx] = str(row.get("hint") or "")
    return hints
