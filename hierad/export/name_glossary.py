"""
从演员库提取角色名 → 生成中文译名表 → 翻译时锁定专名一致性。

流程:
1. extract_character_names(actor_db_path) → 英文角色名列表
2. generate_zh_glossary(names, llm) → {English: Chinese}，缓存到 name_glossary_zh.json
3. normalize_english_names(text, glossary) → 翻译前模糊匹配修正 VLM 拼写
4. enforce_glossary_in_zh(zh_text, glossary) → 翻译后后处理，强制中文专名一致
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from hierad.config import get_openai_client, resolve_llm_settings

# 模糊匹配阈值：ratio >= 0.85 视为同一名字的拼写变体
_FUZZY_THRESHOLD = 0.85


def extract_character_names(actor_db_path: Optional[str]) -> List[str]:
    """从演员库 mapping 提取片内角色名（非演员本名）。"""
    if not actor_db_path:
        return []
    p = Path(actor_db_path)
    if p.is_file():
        p = p.parent
    config_path = p / "config.json"
    if not config_path.exists():
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    mapping = config.get("mapping", {})
    names: List[str] = []
    seen = set()
    for alias, canonical in mapping.items():
        # 跳过全大写的 canonical 行和演员本名（alias.upper() == canonical）
        if alias.isupper() or alias.upper() == canonical.upper():
            continue
        # 跳过带 (voice) 等后缀的
        clean = re.sub(r"\s*\(.*\)\s*", "", alias).strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        names.append(clean)
    return names


def generate_zh_glossary(
    names: Sequence[str],
    *,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """用 LLM 一次性翻译角色名列表，返回 {English: Chinese}。"""
    if not names:
        return {}
    _, resolved_model, _ = resolve_llm_settings(url=llm_url, model=llm_model, api_key=llm_api_key)
    client = get_openai_client(api_key=llm_api_key, base_url=llm_url)
    model = llm_model or resolved_model
    name_list = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
    system = (
        "You are a professional film translator. "
        "Translate English character names into their established Chinese translations. "
        "Use the most common/official Chinese transliteration. "
        "Use · (middle dot) for multi-part names: 托尼·斯塔克. "
        "Output one translation per line, format: number. 中文译名"
    )
    user = f"Translate these character names to Chinese:\n{name_list}"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = (r.choices[0].message.content or "").strip()
        # 解析编号列表
        glossary: Dict[str, str] = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # "1. 托尼·斯塔克" 或 "1.托尼·斯塔克"
            m = re.match(r"^(\d+)[.、]\s*(.+)", line)
            if m:
                idx = int(m.group(1)) - 1
                zh = m.group(2).strip()
                if 0 <= idx < len(names) and zh:
                    glossary[names[idx]] = zh
        if glossary:
            return glossary
    except Exception:
        pass
    return {}


def load_or_build_glossary(
    actor_db_path: Optional[str],
    work_dir: Path,
    *,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    加载缓存的 name_glossary_zh.json；若不存在则从演员库构建并缓存。
    """
    cache_path = work_dir / "name_glossary_zh.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    names = extract_character_names(actor_db_path)
    if not names:
        return {}
    glossary = generate_zh_glossary(
        names, llm_url=llm_url, llm_model=llm_model, llm_api_key=llm_api_key
    )
    if glossary:
        cache_path.write_text(
            json.dumps(glossary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return glossary


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def normalize_english_names(text: str, glossary: Dict[str, str]) -> str:
    """
    翻译前：用模糊匹配修正英文文本中角色名的拼写变体。
    例如 "Everheart" → "Everhart"（glossary 里有 Everhart）。
    """
    if not glossary or not text:
        return text
    en_names = sorted(glossary.keys(), key=lambda n: len(n.split()), reverse=True)

    # 1. 多词名：精确/模糊替换（Tony Stark, Obadiah Stane）
    for name in en_names:
        if " " not in name:
            continue
        # 精确（大小写不敏感）
        pattern = re.compile(re.escape(name).replace(r"\ ", r"\s+"), flags=re.I)
        text = pattern.sub(name, text)
        # 模糊：对每个单词部分做变体匹配
        parts = name.split()
        if len(parts) == 2:
            # "Christine Everheart" → "Christine Everhart"
            last_part = parts[-1]
            def fuzzy_last(m: re.Match) -> str:
                word = m.group(0)
                if _fuzzy_ratio(word, last_part) >= _FUZZY_THRESHOLD:
                    return last_part
                return word
            # 只在前面有 first name 的情况下替换 last name
            text = re.sub(
                rf"\b{re.escape(parts[0])}\s+([A-Z][a-zA-Z]+)\b",
                lambda m: f"{parts[0]} {fuzzy_last(re.match(r'([A-Z][a-zA-Z]+)', m.group(1)))}",
                text,
            )

    # 2. 单词名：精确 + 模糊
    for name in en_names:
        if " " in name:
            continue
        if len(name) < 4:
            continue
        # 精确
        text = re.sub(rf"\b{re.escape(name)}\b", name, text, flags=re.I)
        # 模糊
        def fuzzy_word(m: re.Match) -> str:
            word = m.group(0)
            if len(word) < 4:
                return word
            if _fuzzy_ratio(word, name) >= _FUZZY_THRESHOLD:
                return name
            return word
        text = re.sub(r"\b[A-Z][a-zA-Z]{3,}\b", fuzzy_word, text)

    return text


def enforce_glossary_in_zh(zh_text: str, glossary: Dict[str, str]) -> str:
    """
    翻译后后处理：
    1. 替换残留的英文角色名
    2. 对 glossary 中文译名做模糊统一（厄韦哈特 → 厄弗哈特）
    """
    if not glossary or not zh_text:
        return zh_text
    result = zh_text
    # 1. 替换残留的英文角色名（含部分名）
    for en_name, zh_name in sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True):
        result = re.sub(re.escape(en_name), zh_name, result, flags=re.I)
        # 也替换单个 last name（如 Everhart 残留）
        parts = en_name.split()
        if len(parts) > 1:
            last = parts[-1]
            if len(last) >= 4 and last.lower() not in ("the", "and", "guard"):
                zh_last = zh_name.split("·")[-1] if "·" in zh_name else zh_name
                result = re.sub(rf"\b{re.escape(last)}\b", zh_last, result, flags=re.I)

    # 2. 收集 glossary 中文译名的各段，做模糊统一
    zh_segments: Dict[str, str] = {}  # 变体 → 标准段
    for zh_name in glossary.values():
        for seg in zh_name.split("·"):
            seg = seg.strip()
            if len(seg) >= 2:
                zh_segments[seg] = seg

    if not zh_segments:
        return result

    standard_segs = list(zh_segments.keys())

    def fuzzy_zh_replace(m: re.Match) -> str:
        candidate = m.group(0)
        if candidate in standard_segs:
            return candidate
        # 模糊匹配
        best = None
        best_ratio = 0.0
        for std in standard_segs:
            ratio = _fuzzy_ratio(candidate, std)
            if ratio > best_ratio:
                best_ratio = ratio
                best = std
        if best and best_ratio >= 0.80 and len(candidate) >= 2:
            return best
        return candidate

    result = re.sub(r"[\u4e00-\u9fff]{2,8}", fuzzy_zh_replace, result)
    return result
