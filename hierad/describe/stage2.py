"""
Stage 2: AD 片段定位 + 混合视觉分析（方案 B）

1. 从 ASR gap 确定 AD 时间窗并切 clip
2. 默认复用 Stage1 场景描述；跨场景 / 描述不足时再调 VLM
3. 判定 AD 片段所属 scene
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from hierad.preprocess.asr_overlap import clips_for_interval, find_clip_index_for_time
from hierad.preprocess.gap_segments import dialogue_context_for_gap
from hierad.progress import ProgressReporter, progress_iter

from .models import ADSegment, ClipDescription, EnhancedGlobalContext
from .prompts import STAGE2_AD_SYSTEM, STAGE2_AD_USER, STAGE2_AD_USER_V3
from .utils import parse_stage1_output
from .utils.character_utils import normalize_character_with_mapping, build_system_prompt, scrub_unknown_names
from .vlm_client import VLMClient


def _format_ad_visual(setting: str, characters: str, action: str) -> str:
    """统一成 Setting/Characters/Action，便于 Stage3 解析。"""
    action = (action or "").strip()
    if re.search(r"(?:\*\*)?Characters(?:\*\*)?\s*:", action, re.I) and re.search(
        r"(?:\*\*)?Action(?:\*\*)?\s*:", action, re.I
    ):
        return action
    chars = (characters or "").strip() or "None"
    parts = []
    if (setting or "").strip():
        parts.append(f"**Setting:** {setting.strip()}")
    parts.append(f"**Characters:** {chars}")
    if action:
        parts.append(f"**Action:** {action}")
    return "\n".join(parts)


def _needs_vlm(
    ad_start: float,
    ad_end: float,
    scene_clips: List[ClipDescription],
    scene_indices: List[int],
) -> bool:
    """方案 B：判断是否需对 AD clip 再跑 VLM"""
    ad_dur = ad_end - ad_start
    if len(scene_indices) > 1:
        return True
    if not scene_indices:
        return True
    idx = scene_indices[0]
    if idx >= len(scene_clips):
        return True
    scene = scene_clips[idx]
    scene_dur = max(scene.end_sec - scene.start_sec, 0.1)
    if not scene.action or len(scene.action.strip()) < 15:
        return True
    # AD 窗口远小于场景且场景较长 → 需要更细粒度
    if ad_dur < scene_dur * 0.45 and scene_dur > 12.0:
        return True
    return False


def _global_summary(context: EnhancedGlobalContext) -> str:
    parts = []
    if context.story_structure.setting:
        parts.append(f"Setting: {context.story_structure.setting}")
    if context.story_structure.core_conflict:
        parts.append(f"Conflict: {context.story_structure.core_conflict}")
    for phase in (context.story_structure.phases or [])[:3]:
        if phase.summary:
            parts.append(f"{phase.name}: {phase.summary[:80]}")
    return " | ".join(parts) if parts else "N/A"


def _phase_for_clip(context: EnhancedGlobalContext, clip_idx: int) -> str:
    for phase in context.story_structure.phases or []:
        lo, hi = phase.clip_range
        if lo <= clip_idx <= hi:
            return f"{phase.name}: {phase.summary}" if phase.summary else phase.name
    return ""


def _find_scene_for_clip(scene_contexts: Optional[List[dict]], clip_idx: int) -> Optional[dict]:
    if not scene_contexts:
        return None
    for sc in scene_contexts:
        rng = sc.get("shot_range") or sc.get("clip_range") or []
        if rng and rng[0] <= clip_idx <= rng[1]:
            return sc
    return None


class Stage2ADAnalysis:
    """Stage 2: gap AD 定位 + 混合视觉分析"""

    def __init__(
        self,
        vlm_url: str = "http://localhost:8002",
        vlm_type: str = "qwen2.5-vl",
        vlm_endpoint: str = "/prompted_inference",
        canonical_characters: Optional[Dict[str, str]] = None,
    ):
        self.vlm = VLMClient(vlm_url, vlm_type, vlm_endpoint)
        self.canonical_characters = canonical_characters or {}

    def _normalize_chars(self, characters: str) -> str:
        return scrub_unknown_names(characters, self.canonical_characters)

    def _reuse_from_stage1(
        self,
        scene_clips: List[ClipDescription],
        scene_indices: List[int],
        ad_start: float,
        ad_end: float,
    ) -> Tuple[str, str, str]:
        primary = scene_indices[0] if scene_indices else 0
        clip = scene_clips[primary] if primary < len(scene_clips) else None
        if not clip:
            return "", "", ""
        setting = clip.setting
        characters = clip.characters
        action = clip.action
        if len(scene_indices) > 1:
            extras = []
            for i in scene_indices[1:]:
                if i < len(scene_clips) and scene_clips[i].action:
                    extras.append(scene_clips[i].action[:60])
            if extras:
                action = f"{action} {'; '.join(extras)}"
        return setting, characters, action

    def _vlm_analyze_ad(
        self,
        video_path: str,
        scene_idx: int,
        scene_clips: List[ClipDescription],
        context: EnhancedGlobalContext,
        dialogue_before: str,
        dialogue_after: str,
        scene_contexts: Optional[List[dict]] = None,
    ) -> Tuple[str, str, str]:
        scene = scene_clips[scene_idx] if scene_idx < len(scene_clips) else None
        scene_ctx = scene.to_text()[:200] if scene else "N/A"
        scene_theme = ""
        if scene_contexts:
            sc = _find_scene_for_clip(scene_contexts, scene_idx)
            if sc:
                if sc.get("macro_space"):
                    scene_ctx = f"{scene_ctx} | macro: {sc['macro_space'][:120]}"
                scene_theme = sc.get("theme_event") or ""
        if scene_theme:
            user = STAGE2_AD_USER_V3.format(
                global_summary=_global_summary(context),
                scene_context=scene_ctx,
                scene_theme=scene_theme,
                dialogue_before=dialogue_before or "(None)",
                dialogue_after=dialogue_after or "(None)",
            )
        else:
            user = STAGE2_AD_USER.format(
                global_summary=_global_summary(context),
                scene_idx=scene_idx,
                scene_context=scene_ctx,
                dialogue_before=dialogue_before or "(None)",
                dialogue_after=dialogue_after or "(None)",
            )
        try:
            raw = self.vlm.call(video_path, build_system_prompt(STAGE2_AD_SYSTEM, self.canonical_characters), user)
            parsed = parse_stage1_output(raw)
            return (
                parsed.get("setting", ""),
                parsed.get("characters", ""),
                parsed.get("action", parsed.get("raw", raw)[:200]),
            )
        except Exception:
            if scene:
                return scene.setting, scene.characters, scene.action
            return "", "", ""

    def analyze_ad_segments(
        self,
        ad_manifest: List[dict],
        scene_clips: List[ClipDescription],
        clip_ranges: List[Tuple[float, float]],
        context: EnhancedGlobalContext,
        speech_segments: List[dict],
        scene_contexts: Optional[List[dict]] = None,
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> List[ADSegment]:
        results: List[ADSegment] = []
        for item in progress_iter(
            ad_manifest,
            show=show_progress and reporter is None,
            reporter=reporter,
            stage="Stage2 AD",
            desc="Stage2 AD 分析",
            unit="ad",
            total=len(ad_manifest),
        ):
            ad_idx = int(item["ad_idx"])
            start = float(item["start_sec"])
            end = float(item["end_sec"])
            tc = item.get("timecode", "")
            tc_parts = tc.split(" --> ") if " --> " in tc else ["", ""]
            mid = (start + end) / 2
            scene_indices = clips_for_interval(start, end, clip_ranges)
            primary_scene = find_clip_index_for_time(mid, clip_ranges)
            if primary_scene not in scene_indices:
                scene_indices = [primary_scene] + scene_indices

            d_before, d_after = dialogue_context_for_gap(start, end, speech_segments)

            if _needs_vlm(start, end, scene_clips, scene_indices):
                setting, characters, action = self._vlm_analyze_ad(
                    item["video_path"],
                    primary_scene,
                    scene_clips,
                    context,
                    d_before,
                    d_after,
                    scene_contexts=scene_contexts,
                )
                source = "vlm"
            else:
                setting, characters, action = self._reuse_from_stage1(
                    scene_clips, scene_indices, start, end
                )
                source = "stage1_reuse"

            characters = self._normalize_chars(characters)
            visual = _format_ad_visual(setting, characters, action)

            results.append(ADSegment(
                ad_idx=ad_idx,
                timecode_start=tc_parts[0],
                timecode_end=tc_parts[1] if len(tc_parts) > 1 else "",
                start_sec=start,
                end_sec=end,
                scene_idx=primary_scene,
                scene_indices=scene_indices,
                setting=setting,
                characters=characters,
                visual_description=visual,
                source=source,
                dialogue_before=d_before,
                dialogue_after=d_after,
                video_path=item.get("video_path", ""),
            ))
        return results


# 兼容旧名
Stage2Understanding = Stage2ADAnalysis
