"""
多阶段描述模块 v2

Stage 1: 场景 clip VLM × N + LLM 全局增强
Stage 2: AD gap 定位 + 混合视觉分析
Stage 3: LLM AD 精炼
"""

from .models import (
    ClipDescription,
    DenseDescription,
    ADSegment,
    SceneInfo,
    StoryEvent,
    StoryPhase,
    StoryStructure,
    Character,
    CharacterGraph,
    EnhancedGlobalContext,
    Scene,
)
from .stage1 import Stage1SceneUnderstanding, Stage1Descriptor
from .stage1_v3 import Stage1ShotUnderstanding
from .stage1_enhance import Stage1Enhancer
from .scene_enhance import Stage1SceneEnhancer
from .macro_space import infer_macro_spaces
from .scene_aggregation import aggregate_shots_to_scenes, infer_scene_themes
from .stage2 import Stage2ADAnalysis, Stage2Understanding
from .stage3 import Stage3Refiner
from .vlm_client import VLMClient

__all__ = [
    "ClipDescription",
    "DenseDescription",
    "ADSegment",
    "SceneInfo",
    "StoryEvent",
    "StoryPhase",
    "StoryStructure",
    "EnhancedGlobalContext",
    "Stage1SceneUnderstanding",
    "Stage1Descriptor",
    "Stage1ShotUnderstanding",
    "Stage1Enhancer",
    "Stage1SceneEnhancer",
    "infer_macro_spaces",
    "aggregate_shots_to_scenes",
    "infer_scene_themes",
    "Scene",
    "Stage2ADAnalysis",
    "Stage2Understanding",
    "Stage3Refiner",
    "VLMClient",
]
