"""
Stage 1 增强 — 从 N 个 clip 描述用 LLM 构建全局上下文
"""

import re
from typing import List, Optional, Dict

from openai import OpenAI

from hierad.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from hierad.progress import ProgressReporter, progress_iter

from .models import (
    ClipDescription,
    SceneInfo,
    StoryEvent,
    StoryPhase,
    StoryStructure,
    CharacterGraph,
    Character,
    EnhancedGlobalContext,
)
from .prompts import (
    SCENE_SUMMARY_SYSTEM,
    SCENE_SUMMARY_USER,
    CHARACTER_GRAPH_SYSTEM,
    CHARACTER_GRAPH_USER,
    EVENT_CHAIN_SYSTEM,
    EVENT_CHAIN_USER,
    STORY_STRUCTURE_SYSTEM,
    STORY_STRUCTURE_USER,
)
from .utils import (
    parse_scene_summary_output,
    parse_llm_json,
    normalize_character_with_mapping,
    normalize_character_string,
)


_GENERIC_NAME_RE = re.compile(
    r"\b(NO|NONE|LABELED|CHARACTERS|VISIBLE|UNKNOWN|N/A|NA|SILHOUETTE|FIGURE|FIGURES|"
    r"MAN|WOMAN|ELDER|PERSON|PEOPLE|CREATURE|DRAGON|BEAST|ANIMAL|BIRD|HAND|HANDS)\b"
)


def _is_generic_character(name: str) -> bool:
    return bool(_GENERIC_NAME_RE.search(name or ""))


class Stage1Enhancer:
    """LLM 从 Stage1 clip 描述构建全局上下文"""

    def __init__(
        self,
        llm_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
        llm_api_key: Optional[str] = None,
        canonical_mapping: Optional[Dict[str, str]] = None,
        scene_group_size: int = 5,
    ):
        self.client = OpenAI(base_url=llm_url, api_key=llm_api_key or LLM_API_KEY)
        self.llm_model = llm_model
        self.canonical_mapping = canonical_mapping or {}
        self.scene_group_size = scene_group_size

    def _call_llm(self, system: str, user: str, max_tokens: int = 2048) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _build_scenes_from_clips(self, clips: List[ClipDescription]) -> List[SceneInfo]:
        """每 clip 对应一个 scene；可选分组摘要"""
        scenes: List[SceneInfo] = []
        for c in clips:
            scenes.append(SceneInfo(
                scene_id=c.clip_idx,
                start_clip=c.clip_idx,
                end_clip=c.clip_idx,
                start_time=c.timecode_start,
                end_time=c.timecode_end,
                event=c.action[:120] if c.action else "",
                characters=c.characters,
            ))
        return scenes

    def _summarize_scene_groups(
        self,
        scenes: List[SceneInfo],
        clips: List[ClipDescription],
        character_graph: CharacterGraph,
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> None:
        known = set(character_graph.characters.keys())
        groups = [
            scenes[i : i + self.scene_group_size]
            for i in range(0, len(scenes), self.scene_group_size)
        ]
        for group in progress_iter(
            groups,
            show=show_progress and reporter is None and len(groups) > 3,
            reporter=reporter,
            stage="Stage1 增强",
            desc="Stage1 场景摘要",
            unit="grp",
            total=len(groups),
        ):
            if len(group) == 1:
                continue
            lines = []
            for s in group:
                c = clips[s.scene_id] if s.scene_id < len(clips) else None
                text = c.to_text() if c else s.event
                if text:
                    lines.append(f"[clip {s.scene_id}] {text[:120]}")
            if not lines:
                continue
            resp = self._call_llm(
                SCENE_SUMMARY_SYSTEM,
                SCENE_SUMMARY_USER.format(segments="\n".join(lines)),
            )
            parsed = parse_scene_summary_output(resp)
            event = (parsed.get("event") or "").strip()
            if not event or not resp.strip():
                continue
            for s in group:
                s.event = event
                s.progression = parsed.get("progression", "")
                s.progression_list = parsed.get("progression_list", [])
                chars = parsed.get("characters", "")
                s.characters = normalize_character_string(chars, known) if known else chars

    def _build_character_graph(self, clips: List[ClipDescription]) -> CharacterGraph:
        chars: Dict[str, Character] = {}
        lines = [f"[{c.clip_idx}] {c.to_text()[:100]}" for c in clips if c.to_text()]
        chunk_size = 40
        for start in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[start : start + chunk_size])
            resp = self._call_llm(CHARACTER_GRAPH_SYSTEM, CHARACTER_GRAPH_USER.format(descriptions=chunk))
            data = parse_llm_json(resp)
            if data and isinstance(data.get("characters"), list):
                for c in data["characters"]:
                    if isinstance(c, dict):
                        name = (c.get("name") or "").strip().upper()
                        if name:
                            name = normalize_character_with_mapping(name, self.canonical_mapping)
                            if name not in chars:
                                chars[name] = Character(
                                    name=name,
                                    description=(c.get("description") or "").strip(),
                                    role=(c.get("role") or "supporting").lower(),
                                )
        if not chars:
            for c in clips:
                for part in (c.characters or "").split(","):
                    name = part.strip().upper()
                    if name and len(name) > 1:
                        name = normalize_character_with_mapping(name, self.canonical_mapping)
                        if _is_generic_character(name):
                            continue
                        if name not in chars:
                            chars[name] = Character(name=name, first_appearance=c.clip_idx)
        return CharacterGraph(characters=chars, relations=[])

    def _build_events(self, scenes: List[SceneInfo]) -> List[StoryEvent]:
        if not scenes:
            return []
        summaries = "\n".join(f"Clip {s.scene_id}: {s.event}" for s in scenes[:40] if s.event)
        resp = self._call_llm(EVENT_CHAIN_SYSTEM, EVENT_CHAIN_USER.format(scene_summaries=summaries))
        data = parse_llm_json(resp)
        if data and isinstance(data.get("events"), list):
            events = []
            for e in data["events"]:
                if isinstance(e, dict):
                    cid = e.get("clip_id", e.get("scene_id", 0))
                    events.append(StoryEvent(
                        event_id=e.get("id", len(events)),
                        clip_range=(cid, cid),
                        description=(e.get("description") or "").strip(),
                        characters=e.get("characters", []),
                        is_turning_point=e.get("is_turning_point", False),
                        importance=e.get("importance", 1),
                        causes=e.get("caused_by", []),
                    ))
            if events:
                return events
        return [
            StoryEvent(i, (s.scene_id, s.scene_id), s.event[:100] or f"Clip {i}")
            for i, s in enumerate(scenes)
        ]

    def _build_story_structure(
        self,
        character_graph: CharacterGraph,
        events: List[StoryEvent],
        clip_count: int,
    ) -> StoryStructure:
        char_info = "; ".join(
            f"{c.name}: {c.description}" for c in character_graph.characters.values() if c.description
        )
        event_chain = "\n".join(f"[{e.event_id}] {e.description}" for e in events[:15])
        resp = self._call_llm(
            STORY_STRUCTURE_SYSTEM,
            STORY_STRUCTURE_USER.format(character_info=char_info or "N/A", event_chain=event_chain),
        )
        data = parse_llm_json(resp)
        if data and isinstance(data.get("phases"), list):
            phases = []
            for p in data["phases"]:
                if isinstance(p, dict):
                    clip_ids = p.get("clip_ids", p.get("event_ids", [0]))
                    cr = (min(clip_ids), max(clip_ids)) if clip_ids else (0, 0)
                    phases.append(StoryPhase(
                        name=p.get("name", "PHASE"),
                        event_ids=p.get("event_ids", []),
                        clip_range=cr,
                        summary=p.get("summary", ""),
                    ))
            if phases:
                return StoryStructure(
                    setting=data.get("setting", ""),
                    core_conflict=data.get("core_conflict", ""),
                    phases=phases,
                    turning_point_ids=data.get("turning_points", []),
                )
        third = max(clip_count // 3, 1)
        return StoryStructure(phases=[
            StoryPhase("SETUP", clip_range=(0, third)),
            StoryPhase("RISING", clip_range=(third, 2 * third)),
            StoryPhase("RESOLUTION", clip_range=(2 * third, max(clip_count - 1, 0))),
        ])

    def enhance(
        self,
        clips: List[ClipDescription],
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> EnhancedGlobalContext:
        if reporter:
            reporter.stage("Stage1 增强", message="角色图谱")
        character_graph = self._build_character_graph(clips)
        scenes = self._build_scenes_from_clips(clips)
        self._summarize_scene_groups(
            scenes, clips, character_graph, reporter=reporter, show_progress=show_progress
        )
        if reporter:
            reporter.stage("Stage1 增强", message="事件链与故事结构")
        events = self._build_events(scenes)
        story = self._build_story_structure(character_graph, events, len(clips))
        summaries = [c.action[:80] for c in clips if c.action]
        return EnhancedGlobalContext(
            scenes=scenes,
            character_graph=character_graph,
            events=events,
            story_structure=story,
            clip_summaries=summaries,
        )
