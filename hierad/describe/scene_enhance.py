"""
Stage 1b（v3）：Scene 级全局增强（角色图谱 / 事件链 / 故事结构）
"""

from __future__ import annotations

from typing import Dict, List, Optional

from hierad.progress import ProgressReporter

from .models import ClipDescription, Scene, EnhancedGlobalContext
from .stage1_enhance import Stage1Enhancer


class Stage1SceneEnhancer:
    """在 Scene 级构建全局上下文（复用 Stage1Enhancer 的 LLM 逻辑）"""

    def __init__(
        self,
        llm_url: str,
        llm_model: str,
        llm_api_key: Optional[str] = None,
        canonical_mapping: Optional[Dict[str, str]] = None,
    ):
        self.base = Stage1Enhancer(
            llm_url=llm_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            canonical_mapping=canonical_mapping,
        )

    def enhance(
        self,
        shots: List[ClipDescription],
        scenes: List[Scene],
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> EnhancedGlobalContext:
        if reporter:
            reporter.stage("Stage1 增强", message="角色图谱")
        character_graph = self.base._build_character_graph(shots)

        scene_infos = [s.to_scene_info() for s in scenes]

        if reporter:
            reporter.stage("Stage1 增强", message="事件链与故事结构")
        events = self.base._build_events(scene_infos)
        story = self.base._build_story_structure(character_graph, events, len(shots))
        summaries = [s.theme_event[:80] for s in scenes if s.theme_event]
        return EnhancedGlobalContext(
            scenes=scene_infos,
            character_graph=character_graph,
            events=events,
            story_structure=story,
            clip_summaries=summaries,
        )