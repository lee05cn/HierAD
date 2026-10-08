"""
Scene 聚合：Shot → Scene + theme_event 提炼

聚合规则：macro_space 一致 + 主题连续 + 无硬切分信号。
"""

from __future__ import annotations

import re
from typing import List, Optional

from openai import OpenAI

from hierad.config import LLM_API_KEY
from hierad.progress import ProgressReporter, progress_iter

from .models import ClipDescription, Scene
from .prompts import THEME_EVENT_SYSTEM, THEME_EVENT_USER


_TIME_JUMP_RE = re.compile(
    r"\b(years? later|months? later|weeks? later|days? later|hours? later|"
    r"next (morning|day|night)|back at|meanwhile|elsewhere|earlier|"
    r"the next day|that night|three years|flashback)\b",
    re.I,
)
_NONE_CHARS = {"NONE", "N/A", "NA", "-", ""}


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _macro_key(shot: ClipDescription) -> str:
    return _normalize(shot.macro_space) or _normalize(shot.setting)


def _char_names(shot: ClipDescription) -> set:
    out = set()
    for p in re.split(r"[,;|]", shot.characters or ""):
        p = p.strip().upper()
        if p and p not in _NONE_CHARS:
            out.add(p)
    return out


def _macro_similar(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 1.0 if a == b else 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _same_macro(prev: ClipDescription, cur: ClipDescription) -> bool:
    a, b = _macro_key(prev), _macro_key(cur)
    if a == b:
        return True
    if not a or not b:
        return False
    return _macro_similar(a, b) >= 0.4


def _detect_hard_cut(prev: ClipDescription, cur: ClipDescription) -> bool:
    if not _same_macro(prev, cur):
        return True
    for text in (cur.action or "", cur.dialogue_overlap or ""):
        if _TIME_JUMP_RE.search(text):
            return True
    return False


def _thematically_continuous(prev: ClipDescription, cur: ClipDescription) -> bool:
    gap = cur.start_sec - prev.end_sec
    if gap > 15.0:
        return False
    # 角色信息完整且完全不相交、同时有明显时间间隔时，才视为断裂；
    # 紧邻切镜即使角色不同（如舱内 → 海底）仍属同一 macro_space 的连续叙事。
    a, b = _char_names(prev), _char_names(cur)
    if a and b and not (a & b) and gap > 2.0:
        return False
    return True


def _build_scene(scenes: List[Scene], shots: List[ClipDescription]) -> Scene:
    chars = []
    seen = set()
    for s in shots:
        for p in re.split(r"[,;|]", s.characters or ""):
            p = p.strip()
            if p and p.upper() not in _NONE_CHARS and p not in seen:
                seen.add(p)
                chars.append(p)
    micro = []
    for s in shots:
        st = (s.setting or "").strip()
        if st and st not in micro:
            micro.append(st)
    return Scene(
        scene_id=len(scenes),
        shot_range=(shots[0].clip_idx, shots[-1].clip_idx),
        start_sec=shots[0].start_sec,
        end_sec=shots[-1].end_sec,
        macro_space=(shots[0].macro_space or "").strip(),
        characters=", ".join(chars),
        micro_settings=micro,
        shots=shots,
    )


def aggregate_shots_to_scenes(shots: List[ClipDescription]) -> List[Scene]:
    if not shots:
        return []
    scenes: List[Scene] = []
    cur_shots: List[ClipDescription] = [shots[0]]
    for prev, cur in zip(shots, shots[1:]):
        if _detect_hard_cut(prev, cur) or not _thematically_continuous(prev, cur):
            scenes.append(_build_scene(scenes, cur_shots))
            cur_shots = [cur]
        else:
            cur_shots.append(cur)
    scenes.append(_build_scene(scenes, cur_shots))
    return scenes


def _clean_theme(text: str) -> str:
    text = (text or "").strip()
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("THEME_EVENT") and ":" in line:
            return line.split(":", 1)[1].strip()
    return text


def _fallback_theme(scene: Scene) -> str:
    actions = [s.action.strip() for s in scene.shots if (s.action or "").strip()]
    if not actions:
        return scene.macro_space or ""
    longest = max(actions, key=len)
    return longest[:200]


def infer_scene_themes(
    scenes: List[Scene],
    *,
    llm_url: str,
    llm_model: str,
    llm_api_key: Optional[str] = None,
    reporter: Optional[ProgressReporter] = None,
    show_progress: bool = False,
) -> None:
    client = OpenAI(base_url=llm_url, api_key=llm_api_key or LLM_API_KEY)
    for scene in progress_iter(
        scenes,
        show=show_progress and reporter is None,
        reporter=reporter,
        stage="Scene 主题",
        desc="Scene theme_event",
        unit="scene",
        total=len(scenes),
    ):
        shots_block = "\n".join(
            f"- setting: {(s.setting or '')[:120]}\n"
            f"  characters: {s.characters or 'none'}\n"
            f"  action: {(s.action or '')[:160]}"
            for s in scene.shots[:12]
        )
        user = THEME_EVENT_USER.format(
            macro_space=scene.macro_space or "(none)",
            shots=shots_block,
        )
        theme = ""
        try:
            r = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": THEME_EVENT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=128,
                temperature=0.2,
            )
            theme = _clean_theme(r.choices[0].message.content or "")
        except Exception:
            theme = ""
        if not theme:
            theme = _fallback_theme(scene)
        scene.theme_event = theme