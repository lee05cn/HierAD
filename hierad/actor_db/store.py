"""
演员数据库存储

格式: {db_dir}/config.json
  - movie_name, movie_id
  - mapping: {alias -> canonical}

若存在 faiss_index.bin: 支持人脸向量检索
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


class ActorDatabase:
    """演员数据库 - 加载与查询"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._mapping: Dict[str, str] = {}

    def load(self) -> Dict[str, str]:
        """加载 别名 -> 规范名 映射"""
        config_path = self.db_path / "config.json"
        if not config_path.exists():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self._mapping = config.get("mapping", {})
        return self._mapping

    def normalize(self, name: str) -> str:
        """将角色名标准化为规范名（大小写不敏感查找）"""
        if not self._mapping:
            self.load()
        name = name.strip()
        if not name:
            return ""
        if name in self._mapping:
            return self._mapping[name]
        if name.upper() in self._mapping:
            return self._mapping[name.upper()]
        for alias, canonical in self._mapping.items():
            if alias.upper() == name.upper():
                return canonical
        return name.upper()

    @staticmethod
    def _norm_key(s: str) -> str:
        return s.upper().replace(".", "").replace(" ", "")

    def character_display_name(self, canonical: str) -> str:
        """
        将 Faiss 命中的 canonical（演员规范名）转为片内角色名，如 JOAQUIN PHOENIX → Merrill Hess。
        """
        if not self._mapping:
            self.load()
        canonical = (canonical or "").strip().upper()
        if not canonical:
            return "Unknown"

        roles: List[str] = []
        for alias, can in self._mapping.items():
            if can.upper() != canonical:
                continue
            if alias.upper() == canonical or alias.isupper():
                continue
            if self._norm_key(alias) == self._norm_key(canonical):
                continue
            roles.append(alias)

        if not roles:
            return canonical.title()
        # 优先带空格的角色名（Graham Hess），排除与 canonical 同形的演员本名
        with_space = [r for r in roles if " " in r]
        return (with_space or roles)[0]

    def resolve_face_label(self, metadata: Dict[str, Any]) -> str:
        """从检索 metadata 解析应显示的角色名（片内角色，非演员本名）"""
        canonical = (
            metadata.get("canonical")
            or metadata.get("character")
            or metadata.get("actor_name")
            or ""
        )
        # 旧库 metadata["character"] 存的是 JOAQUIN PHOENIX 这类 canonical
        display = self.character_display_name(str(canonical))
        if display.upper() != str(canonical).upper():
            return display
        # 若 metadata 里已有真实角色字段（新库 build 写入）
        role = metadata.get("role_name") or metadata.get("role")
        if role and str(role).upper() != str(canonical).upper():
            return str(role)
        return display

    def role_name_mapping(self) -> Dict[str, str]:
        """别名/演员 canonical -> 片内角色名（大写），供 Stage1 角色字段标准化"""
        if not self._mapping:
            self.load()
        out: Dict[str, str] = {}
        for alias, can in self._mapping.items():
            role = self.character_display_name(can).upper()
            out[alias] = role
            out[alias.upper()] = role
        for can in set(self._mapping.values()):
            role = self.character_display_name(can).upper()
            out[can.upper()] = role
        return out

    @property
    def mapping(self) -> Dict[str, str]:
        """获取完整映射表"""
        if not self._mapping:
            self.load()
        return self._mapping.copy()

    def search_face(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        人脸相似度检索（若存在向量库）

        Returns:
            [{"similarity": float, "metadata": {...}}, ...]
        """
        vector_path = self.db_path / "faiss_index.bin"
        if not vector_path.exists():
            return []
        try:
            from .vector_store import FaceVectorStore
            store = FaceVectorStore(self.db_path)
            store.load()
            return store.search(query_embedding, top_k)
        except ImportError:
            return []
