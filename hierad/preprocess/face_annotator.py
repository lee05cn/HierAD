"""
人脸标注模块 - 在视频中叠加演员人脸框

依赖: opencv-python, 演员数据库
完整人脸识别需: pip install hierad[actor_db_vectors] (insightface, faiss)
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple

import numpy as np


from .face_hints import load_actor_db, resolve_actor_db_dir, recognize_faces_in_frame


def _load_actor_db(actor_db_path: str) -> Optional[Any]:
    return load_actor_db(actor_db_path)


def _draw_face_labels(
    frame: np.ndarray,
    detections: List[Tuple[List[int], str]],
) -> None:
    """绘制椭圆框与角色名（片内角色名）— 字号需在 VLM 降分辨率后仍可读"""
    import cv2

    h, w = frame.shape[:2]
    # 相对分辨率缩放：1080p 约 scale=1.4；保证短边压到 ~300px 后字高仍 ≥ ~10px
    font_scale = max(1.1, min(2.0, h / 600.0))
    thickness = max(2, int(round(font_scale)))
    for bbox, label in detections:
        x1 = max(0, min(int(bbox[0]), w - 1))
        y1 = max(0, min(int(bbox[1]), h - 1))
        x2 = max(x1 + 1, min(int(bbox[2]), w))
        y2 = max(y1 + 1, min(int(bbox[3]), h))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
        cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (0, 255, 0), max(2, thickness))
        text = label.upper()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = max(4, thickness + 2)
        label_y1 = max(0, y1 - th - pad * 2)
        label_y2 = max(label_y1 + th + pad, y1)
        label_x2 = min(w, x1 + tw + pad * 2)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
        # 黑描边 + 白字，提高降采样后对比度
        tx, ty = x1 + pad, label_y2 - pad
        cv2.putText(
            frame, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )


def annotate_video(
    video_path: str,
    actor_db_path: str,
    output_path: Optional[str] = None,
    recognition_threshold: float = 0.4,
    frame_interval: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    在视频中标注识别到的演员人脸

    Args:
        video_path: 输入视频路径
        actor_db_path: 演员数据库路径（含 config.json，若有 faiss_index.bin 则支持人脸识别）
        output_path: 输出视频路径，默认同目录 _annotated
        recognition_threshold: 人脸识别相似度阈值 (0~1)，默认 0.4
        frame_interval: 每隔多少帧运行一次人脸检测（1=每帧，越大越快）
        progress_callback: (current, total) 进度回调

    Returns:
        {"status": "completed"|"failed", "output_path": "...", "errors": [...], "stats": {...}}
    """
    try:
        import cv2
    except ImportError:
        return {
            "status": "failed",
            "output_path": "",
            "errors": ["需要安装 opencv-python: pip install opencv-python"],
        }

    video_path = Path(video_path)
    if not video_path.exists():
        return {"status": "failed", "output_path": "", "errors": [f"视频不存在: {video_path}"]}

    try:
        db_dir = resolve_actor_db_dir(str(actor_db_path))
    except FileNotFoundError as e:
        return {"status": "failed", "output_path": "", "errors": [str(e)]}

    actor_db = _load_actor_db(str(db_dir))
    if actor_db is None:
        return {"status": "failed", "output_path": "", "errors": [f"无法加载演员数据库: {db_dir}"]}

    has_vectors = (db_dir / "faiss_index.bin").exists()
    if not has_vectors:
        return {
            "status": "failed",
            "output_path": "",
            "errors": [
                "演员数据库无 faiss_index.bin，无法进行人脸识别。"
                "请使用 hierad build-actor-db --with-vectors 构建完整向量库。",
            ],
        }

    try:
        from hierad.actor_db.face_processor import detect_faces_in_image  # noqa: F401
    except ImportError as e:
        return {
            "status": "failed",
            "output_path": "",
            "errors": [
                "需要安装 insightface 等人脸识别依赖: pip install hierad[actor_db_vectors]",
                str(e),
            ],
        }

    output_path = output_path or str(video_path.parent / f"{video_path.stem}_annotated{video_path.suffix}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed", "output_path": "", "errors": ["无法打开视频"]}

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    interval = max(1, frame_interval)
    # 非检测帧继续绘制最近一次结果，覆盖到下一次检测前，保证 VLM 抽帧也能看到标签
    # interval=5 → stale_max=4，即 5 帧窗口内全程有标签（若检测成功）
    stale_max = 0 if interval == 1 else (interval - 1)

    last_detections: List[Tuple[List[int], str]] = []
    last_detect_frame = -10_000
    frame_idx = 0
    detection_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                dets = recognize_faces_in_frame(frame, actor_db, recognition_threshold)
                last_detections = [(d["bbox"], d["character"]) for d in dets]
                last_detect_frame = frame_idx
                detection_count += 1

            if last_detections and (frame_idx - last_detect_frame) <= stale_max:
                _draw_face_labels(frame, last_detections)

            out.write(frame)
            if progress_callback:
                progress_callback(frame_idx + 1, total)
            frame_idx += 1

    finally:
        cap.release()
        out.release()

    return {
        "status": "completed",
        "output_path": output_path,
        "errors": [],
        "stats": {
            "total_frames": frame_idx,
            "detection_passes": detection_count,
            "frame_interval": interval,
            "stale_max_frames": stale_max,
        },
    }
