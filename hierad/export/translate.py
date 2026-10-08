"""
AD 英文脚本 → 中文翻译（对齐 VD-agent translate_by_llm）
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from hierad.config import get_openai_client, resolve_llm_settings

from .name_glossary import (
    enforce_glossary_in_zh,
    normalize_english_names,
)


def _build_glossary_section(glossary: Optional[Dict[str, str]]) -> str:
    if not glossary:
        return ""
    pairs = ", ".join(f"{en}={zh}" for en, zh in glossary.items())
    return (
        f"\n- CHARACTER NAMES: Use EXACTLY these Chinese translations for character names: {pairs}"
        "\n- Do not transliterate character names differently from the glossary above"
    )


_TRANSLATE_SYSTEM = """You are a professional translator specializing in movie audio descriptions.
Your task: Translate English movie descriptions into natural, fluent Chinese.

Requirements:
- Maintain the original text structure and format
- Keep it SHORT and speakable: usually within 25 Chinese characters
- If the English line is long, keep only the single most important visible action and drop redundant detail
- Use short clauses; avoid long attributive chains and trailing descriptions that are hard to finish aloud
- Use natural Chinese expressions appropriate for movie descriptions
- Output one Chinese line for each English line you receive
- Do not add explanations or additional content

Translation guidelines:
- Keep technical terms accurate
- Ensure the tone matches the original description
- Prefer concise, spoken-style Chinese over literal, bookish phrasing, and end each line cleanly
{glossary_section}"""

_TRANSLATE_USER = """Translate the following text into Chinese:

{}"""


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


_LEFT_EN_RE = re.compile(r"(?<![A-Za-z])[a-z]+(?:-[a-z]+)*(?![A-Za-z])")


def _has_leftover_english(text: str) -> bool:
    if not _has_cjk(text):
        return False
    return bool(_LEFT_EN_RE.search(text))


def _strip_leftover_english(text: str) -> str:
    """去掉中文译文里残留的小写英文单词（保留大写专名，如 SINTEL）。"""
    if not _has_cjk(text):
        return text
    cleaned = _LEFT_EN_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:，；：")
    return cleaned


def _is_valid_translation(original: str, translated: str) -> bool:
    if not translated or not translated.strip():
        return False
    t = translated.lower()
    for pattern in (
        "重要提示：每行输入",
        "重要提示：每条输入",
        "不要合并或拆分",
        "important: each input",
        "do not combine or split",
    ):
        if pattern in t:
            return False
    zh = sum(1 for c in translated if "\u4e00" <= c <= "\u9fff")
    en = sum(1 for c in translated if c.isalpha() and ord(c) < 128)
    if zh < 10 and en > 50:
        return False
    ratio = len(translated) / max(len(original), 1)
    if ratio > 5.0 or (ratio < 0.05 and len(original) > 50):
        return False
    if _has_leftover_english(translated):
        return False
    return True


def _retry_single(
    client,
    text: str,
    model: str,
    max_retries: int = 2,
    glossary: Optional[Dict[str, str]] = None,
) -> str:
    text_stripped = text.strip()
    # 翻译前：模糊修正角色名拼写
    if glossary:
        text_stripped = normalize_english_names(text_stripped, glossary)
    is_multiline = "\n" in text_stripped
    last = text_stripped
    system_prompt = _TRANSLATE_SYSTEM.format(
        glossary_section=_build_glossary_section(glossary)
    )
    for attempt in range(max_retries):
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _TRANSLATE_USER.format(text_stripped)},
                ],
                "temperature": 0.3,
                "max_tokens": 3000,
            }
            if not is_multiline:
                params["stop"] = ["\n\n", "\n\n\n"]
            completion = client.chat.completions.create(**params)
            translated = (completion.choices[0].message.content or "").strip()
            if not is_multiline:
                translated = translated.split("\n")[0].strip()
            # 翻译后：强制 glossary 专名一致 + 清理残留小写英文
            if glossary:
                translated = enforce_glossary_in_zh(translated, glossary)
            translated = _strip_leftover_english(translated)
            last = translated or last
            if _is_valid_translation(text_stripped, translated):
                return translated
            if attempt == max_retries - 1:
                zh = sum(1 for c in translated if "\u4e00" <= c <= "\u9fff")
                if zh >= 8:
                    return translated
        except Exception:
            continue
    return last


def translate_texts_to_zh(
    texts: Sequence[str],
    *,
    batch_size: int = 4,
    llm_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    glossary: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    将英文 AD 列表译为中文。已含较多汉字的条目原样返回。
    若提供 glossary（{English: Chinese}），翻译时锁定角色名译名一致性。
    """
    if not texts:
        return []
    _, resolved_model, _ = resolve_llm_settings(url=llm_url, model=llm_model, api_key=llm_api_key)
    client = get_openai_client(api_key=llm_api_key, base_url=llm_url)
    model = llm_model or resolved_model
    system_prompt = _TRANSLATE_SYSTEM.format(
        glossary_section=_build_glossary_section(glossary)
    )
    results: List[str] = []

    for i in range(0, len(texts), batch_size):
        batch = [t.strip() for t in texts[i : i + batch_size]]
        # 翻译前：模糊修正角色名拼写
        if glossary:
            batch = [normalize_english_names(t, glossary) for t in batch]
        # 已是中文则跳过 LLM
        if all(_has_cjk(t) and sum(1 for c in t if "\u4e00" <= c <= "\u9fff") >= max(8, len(t) // 4) for t in batch):
            results.extend(batch)
            continue

        output_list: Optional[List[str]] = None
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _TRANSLATE_USER.format("\n".join(batch))},
                ],
                temperature=0.3,
            )
            raw = (completion.choices[0].message.content or "").strip()
            raw = re.sub(r"\n+", "\n", raw)
            output_list = [line.strip() for line in raw.split("\n") if line.strip()]
            if len(output_list) != len(batch):
                output_list = None
            else:
                invalid = [
                    idx
                    for idx, (o, t) in enumerate(zip(batch, output_list))
                    if not _is_valid_translation(o, t)
                ]
                if len(invalid) > len(batch) / 2:
                    output_list = None
                else:
                    for idx in invalid:
                        output_list[idx] = _retry_single(client, batch[idx], model, glossary=glossary)
        except Exception:
            output_list = None

        if output_list is None:
            output_list = [_retry_single(client, t, model, glossary=glossary) for t in batch]
        # 翻译后：统一 glossary 专名 + 清理残留小写英文
        if glossary:
            output_list = [enforce_glossary_in_zh(t, glossary) for t in output_list]
        output_list = [_strip_leftover_english(t) for t in output_list]
        results.extend(output_list)

    return results
