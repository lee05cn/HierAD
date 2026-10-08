"""
人脸检测与特征提取 - 参考 Actor_Dataset_Construct

使用 InsightFace 提取 512 维人脸向量，可选依赖
GPU: 设置环境变量 HIERAD_FACE_GPU（默认 2），需安装 onnxruntime-gpu
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

_FACE_APP = None
_FACE_GPU: Optional[int] = None


def face_gpu_id() -> int:
    """InsightFace / ONNX Runtime 使用的 GPU 编号（默认 GPU 2，通常较空闲）"""
    return int(os.getenv("HIERAD_FACE_GPU", "2"))


def _build_providers(gpu_id: int) -> Tuple[List[Any], int, str]:
    """返回 (providers, ctx_id, backend_label)"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("需要安装 insightface: pip install insightface onnxruntime-gpu") from None

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers: List[Any] = [
            ("CUDAExecutionProvider", {"device_id": gpu_id}),
            "CPUExecutionProvider",
        ]
        return providers, gpu_id, f"CUDA:{gpu_id}"

    warnings.warn(
        "CUDAExecutionProvider 不可用，InsightFace 将使用 CPU。"
        "请安装 GPU 版: pip uninstall onnxruntime -y && pip install onnxruntime-gpu",
        stacklevel=3,
    )
    return ["CPUExecutionProvider"], -1, "CPU"


def _get_face_processor():
    """延迟加载 InsightFace（进程内单例，避免每帧重复初始化）"""
    global _FACE_APP, _FACE_GPU
    gpu_id = face_gpu_id()
    if _FACE_APP is not None and _FACE_GPU == gpu_id:
        return _FACE_APP

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("需要安装 insightface: pip install insightface onnxruntime-gpu") from None

    providers, ctx_id, backend = _build_providers(gpu_id)
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_thresh=0.5)
    _FACE_APP = app
    _FACE_GPU = gpu_id
    print(f"[InsightFace] backend={backend}, ctx_id={ctx_id}")
    return _FACE_APP


def _imread(path: str) -> Optional[np.ndarray]:
    """支持中文路径的图片读取"""
    try:
        import cv2
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def detect_faces_in_image(
    img: np.ndarray,
    min_score: float = 0.5,
    max_faces: int = 20,
) -> List[Dict[str, Any]]:
    """
    从 BGR 图像（如视频帧）中检测人脸并提取向量

    Args:
        img: OpenCV BGR 格式图像 (H, W, 3)
        min_score: 最低检测分数
        max_faces: 最多返回人脸数

    Returns:
        [{"embedding": ndarray, "det_score": float, "bbox": [x1,y1,x2,y2], ...}, ...]
    """
    app = _get_face_processor()
    results = []
    faces = app.get(img)
    faces = [f for f in faces if f.det_score >= min_score]
    faces.sort(key=lambda f: f.det_score, reverse=True)
    for f in faces[:max_faces]:
        results.append({
            "embedding": f.embedding.astype(np.float32),
            "det_score": float(f.det_score),
            "bbox": f.bbox.astype(int).tolist(),
        })
    return results


def extract_face_embeddings(
    image_paths: List[str],
    max_faces_per_image: int = 1,
    min_score: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    从图片中提取人脸向量

    Args:
        image_paths: 图片路径列表
        max_faces_per_image: 每张图最多取几人脸（取最高分）
        min_score: 最低检测分数

    Returns:
        [{"embedding": ndarray, "det_score": float, "image_path": str, ...}, ...]
    """
    app = _get_face_processor()
    results = []

    for path in image_paths:
        img = _imread(path)
        if img is None:
            continue
        faces = app.get(img)
        faces = [f for f in faces if f.det_score >= min_score]
        faces.sort(key=lambda f: f.det_score, reverse=True)

        for f in faces[:max_faces_per_image]:
            results.append({
                "embedding": f.embedding.astype(np.float32),
                "det_score": float(f.det_score),
                "image_path": path,
                "bbox": f.bbox.astype(int).tolist(),
            })
    return results
