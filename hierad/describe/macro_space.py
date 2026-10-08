"""
Stage 1.5：micro_setting → macro_space 归一化

用轻量 LLM 把每个镜头具体的环境抽象成宏观叙事空间；LLM 不可用时退回 setting 本身。
"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from hierad.config import LLM_API_KEY
from hierad.progress import ProgressReporter, progress_iter

from .models import ClipDescription
from .prompts import MACRO_SPACE_SYSTEM, MACRO_SPACE_USER


def _clean_macro(text: str) -> str:
    text = (text or "").strip()
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("MACRO_SPACE") and ":" in line:
            return line.split(":", 1)[1].strip()
    return text


def _fallback_macro(shot: ClipDescription) -> str:
    return (shot.setting or "").strip()[:120]


def infer_macro_spaces(
    shots: List[ClipDescription],
    *,
    llm_url: str,
    llm_model: str,
    llm_api_key: Optional[str] = None,
    reporter: Optional[ProgressReporter] = None,
    show_progress: bool = False,
) -> List[str]:
    client = OpenAI(base_url=llm_url, api_key=llm_api_key or LLM_API_KEY)
    result: List[str] = []
    prev_macro = ""
    for shot in progress_iter(
        shots,
        show=show_progress and reporter is None,
        reporter=reporter,
        stage="Stage1.5 macro",
        desc="宏观空间归一化",
        unit="shot",
        total=len(shots),
    ):
        user = MACRO_SPACE_USER.format(
            micro_setting=(shot.setting or "(none)")[:200],
            characters=(shot.characters or "(none)")[:120],
            action=(shot.action or "(none)")[:200],
            dialogue=(shot.dialogue_overlap or "(none)")[:160],
            prev_macro=prev_macro or "(none)",
        )
        macro = ""
        try:
            r = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": MACRO_SPACE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=64,
                temperature=0.1,
            )
            macro = _clean_macro(r.choices[0].message.content or "")
        except Exception:
            macro = ""
        if not macro:
            macro = _fallback_macro(shot)
        result.append(macro)
        prev_macro = macro
    return result