"""
描述模块工具 - 解耦的辅助函数

- similarity: 余弦相似度计算（无额外依赖）
- sbert_factory: SBERT 模型工厂（可选，需 sentence-transformers）
- parsers: 解析 VLM/LLM 结构化输出
- character_utils: 角色名标准化、模糊匹配
"""

from .similarity import (
    compute_cosine_adjacent_similarities,
    compute_cosine_similarity_matrix,
)
from .parsers import parse_stage1_output, parse_scene_summary_output, parse_llm_json
from .character_utils import (
    normalize_character_with_mapping,
    normalize_character_string,
    fuzzy_match_character,
    build_character_hints,
    build_system_prompt,
    scrub_unknown_names,
)

__all__ = [
    "compute_cosine_adjacent_similarities",
    "compute_cosine_similarity_matrix",
    "parse_stage1_output",
    "parse_scene_summary_output",
    "parse_llm_json",
    "normalize_character_with_mapping",
    "normalize_character_string",
    "fuzzy_match_character",
    "build_character_hints",
    "build_system_prompt",
    "scrub_unknown_names",
]
