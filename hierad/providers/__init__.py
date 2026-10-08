"""可插拔模型后端。默认 local，可选 dashscope。"""

from .vlm import create_vlm_backend
from .asr import transcribe_with_dashscope
from .tts import synthesize_with_dashscope

__all__ = [
    "create_vlm_backend",
    "transcribe_with_dashscope",
    "synthesize_with_dashscope",
]
