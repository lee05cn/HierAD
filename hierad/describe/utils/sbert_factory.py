"""
SBERT 模型工厂 - 可选依赖 sentence-transformers

用于场景语义聚合，未安装时自动降级为固定窗口
"""

import os
import threading
from typing import Optional, Any

_model: Any = None
_lock = threading.RLock()
_device: Optional[str] = None


def get_sbert_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> Any:
    """
    获取 SBERT 模型（单例，延迟加载）

    Args:
        model_name: HuggingFace 模型名
        device: cuda:0 / cpu，None 时自动检测

    Returns:
        SentenceTransformer 实例

    Raises:
        ImportError: 未安装 sentence-transformers
    """
    global _model, _device
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as e:
        raise ImportError("需要 sentence-transformers: pip install hierad[describe_extras]") from e

    if device is None:
        device = os.environ.get("SBERT_DEVICE", "")
        if not device and torch.cuda.is_available():
            device = "cuda:0"
        elif not device:
            device = "cpu"

    with _lock:
        if _model is not None and _device == device:
            return _model
        _model = SentenceTransformer(model_name, device=device)
        _device = device
        return _model


def has_sbert() -> bool:
    """检查 SBERT 是否可用"""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
