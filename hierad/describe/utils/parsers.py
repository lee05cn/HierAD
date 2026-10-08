"""
解析 VLM/LLM 结构化输出 - 无外部依赖
"""

import re
from typing import Dict, Any, Optional


def _clean_character_field(raw_chars: str) -> str:
    """清洗角色字段，分离角色名和外貌描述"""
    if not raw_chars:
        return ""
    APPEARANCE_PATTERNS = [
        r"\bwearing\b", r"\bdressed\s+in\b", r"\bwith\b", r"\bholding\b",
        r"\bhair\b", r"\btop\b", r"\bshirt\b", r"\bjacket\b", r"\bdress\b",
        r"\ban?\s+elderly\b", r"\ba\s+young\b", r"\ba\s+man\b", r"\ba\s+woman\b",
    ]
    parts = re.split(r"[,;]\s*", raw_chars)
    clean_names = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        is_appearance = any(re.search(p, part.lower()) for p in APPEARANCE_PATTERNS)
        if is_appearance:
            continue
        if re.search(r"\b[A-Z][a-zA-Z]*\b", part):
            part = re.sub(r"\s*\([^)]*\)\s*", "", part).strip()
            if part and part.lower() not in ("none", "n/a", "-", ""):
                clean_names.append(part)
    return ", ".join(clean_names)


def parse_stage1_output(text: str) -> Dict[str, str]:
    """
    解析 Stage 1 的结构化输出

    Expected: SETTING: ... / CHARACTERS: ... / ACTION: ...
    Returns: {setting, characters, action, raw}
    """
    result = {"setting": "", "characters": "", "action": "", "raw": text}
    settings, characters, actions = [], [], []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        # 兼容 **Setting:** / Setting: / SETTING:
        key = re.sub(r"[*_\s]+", "", key).upper()
        value = value.strip().strip("*").strip()
        if key == "SETTING" and value:
            settings.append(value)
        elif key == "CHARACTERS" and value:
            characters.append(value)
        elif key == "ACTION" and value:
            actions.append(value)
    result["setting"] = settings[-1] if settings else ""
    raw_characters = ", ".join(characters) if characters else ""
    result["characters"] = _clean_character_field(raw_characters)
    result["action"] = " ".join(actions) if actions else ""
    if not result["setting"] and not result["characters"] and not result["action"]:
        cleaned = text.strip()
        for label in ["SETTING", "CHARACTERS", "ACTION"]:
            cleaned = cleaned.replace(f"{label}:", "").strip()
        result["action"] = cleaned or text.strip()
    return result


def parse_scene_summary_output(text: str) -> Dict[str, Any]:
    """
    解析场景摘要的结构化输出

    Expected: EVENT: ... / PROGRESSION: ... / CHARACTERS: ...
    Returns: {event, progression, progression_list, characters, raw}
    """
    result = {"event": "", "progression": "", "progression_list": [], "characters": "", "raw": text}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().upper(), value.strip()
        if key == "EVENT":
            result["event"] = value
        elif key == "PROGRESSION":
            result["progression"] = value
            if "→" in value:
                result["progression_list"] = [a.strip() for a in value.split("→") if a.strip()]
            elif "->" in value:
                result["progression_list"] = [a.strip() for a in value.split("->") if a.strip()]
            else:
                result["progression_list"] = [a.strip() for a in value.split(",") if a.strip()]
        elif key == "CHARACTERS":
            result["characters"] = value
    if not result["event"] and text:
        result["event"] = text[:200]
    return result


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 响应中解析 JSON"""
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            import json
            return json.loads(m.group())
        except Exception:
            pass
    return None
