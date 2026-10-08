"""
Stage 1 v3：Clip → Shot 结构化描述

- 固定上下文窗口 = 2，注入前 2 条 Shot 的结构化字段
- 禁止 "Same" / "Unchanged"，每条 Shot 都写完整 setting
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from hierad.progress import ProgressReporter, progress_iter

from .models import ClipDescription
from .prompts import STAGE1_SHOT_SYSTEM, STAGE1_SHOT_USER_FIRST, STAGE1_SHOT_USER
from .stage1 import _parse_fallback, _load_checkpoint, _save_checkpoint
from .utils import parse_stage1_output, normalize_character_with_mapping, build_system_prompt, scrub_unknown_names
from .vlm_client import VLMClient


class Stage1ShotUnderstanding:
    """Stage 1 v3：Clip → Shot（结构化、窗口=2、禁止 Same）"""

    def __init__(
        self,
        vlm_url: str = "http://localhost:8002",
        vlm_type: str = "qwen2.5-vl",
        vlm_endpoint: str = "/prompted_inference",
        llm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        context_window: int = 2,
        canonical_characters: Optional[Dict[str, str]] = None,
    ):
        self.vlm = VLMClient(vlm_url, vlm_type, vlm_endpoint)
        self.context_window = context_window
        self.canonical_characters = canonical_characters or {}

    def _before_shots(self, shots: List[ClipDescription]) -> str:
        if not shots:
            return "(opening — no previous shots)"
        parts = []
        for s in shots[-self.context_window:]:
            fields = []
            if s.setting:
                fields.append(f"setting: {s.setting[:120]}")
            if s.characters:
                fields.append(f"characters: {s.characters[:80]}")
            if s.action:
                fields.append(f"action: {s.action[:120]}")
            if fields:
                parts.append("- " + " | ".join(fields))
        return "\n".join(parts) if parts else "(no previous shot info)"

    def describe_clips(
        self,
        clip_manifest: List[dict],
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> List[ClipDescription]:
        cached = _load_checkpoint(checkpoint_path)
        shots: List[ClipDescription] = []
        for item in progress_iter(
            clip_manifest,
            show=show_progress and reporter is None,
            reporter=reporter,
            stage="Stage1 Shot",
            desc="Stage1 镜头 VLM",
            unit="shot",
            total=len(clip_manifest),
        ):
            idx = int(item["clip_idx"])
            tc = item.get("timecode", "")
            tc_parts = tc.split(" --> ") if " --> " in tc else ["", ""]
            dialogue = item.get("dialogue_overlap", "(None)")
            if idx in cached:
                shots.append(cached[idx])
                continue

            fmt = dict(
                before_shots=self._before_shots(shots),
                dialogue_overlap=dialogue or "(None)",
            )
            user = STAGE1_SHOT_USER_FIRST.format(**fmt) if idx == 0 else STAGE1_SHOT_USER.format(**fmt)

            try:
                raw = self.vlm.call(item["video_path"], build_system_prompt(STAGE1_SHOT_SYSTEM, self.canonical_characters), user)
                parsed = parse_stage1_output(raw)
                if not parsed.get("action") and not parsed.get("setting"):
                    parsed = _parse_fallback(raw)
                chars = scrub_unknown_names(parsed.get("characters", ""), self.canonical_characters)
                shot = ClipDescription(
                    clip_idx=idx,
                    timecode_start=tc_parts[0],
                    timecode_end=tc_parts[1] if len(tc_parts) > 1 else "",
                    start_sec=float(item.get("start_sec", 0)),
                    end_sec=float(item.get("end_sec", 0)),
                    setting=parsed.get("setting", ""),
                    characters=chars,
                    action=parsed.get("action", ""),
                    raw=parsed.get("raw", raw),
                    dialogue_overlap=dialogue,
                    video_path=item.get("video_path", ""),
                )
            except Exception:
                shot = ClipDescription(
                    clip_idx=idx,
                    timecode_start=tc_parts[0],
                    timecode_end=tc_parts[1] if len(tc_parts) > 1 else "",
                    start_sec=float(item.get("start_sec", 0)),
                    end_sec=float(item.get("end_sec", 0)),
                    setting="",
                    characters="",
                    action="",
                    raw="",
                    dialogue_overlap=dialogue,
                    video_path=item.get("video_path", ""),
                )
            shots.append(shot)
            if shot.is_complete():
                _save_checkpoint(checkpoint_path, shots)
        _save_checkpoint(checkpoint_path, shots)
        return shots