"""
人脸向量存储 - 参考 Actor_Dataset_Construct

使用 Faiss 存储和检索人脸向量，可选依赖
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


def _check_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("需要安装 faiss: pip install faiss-cpu 或 faiss-gpu") from None


class FaceVectorStore:
    """
    单电影人脸向量库

    存储: {db_dir}/faiss_index.bin, metadata.pkl, id_mapping.json
    """

    def __init__(self, db_dir: Path, dimension: int = 512):
        self.db_dir = Path(db_dir)
        self.dimension = dimension
        self._faiss = _check_faiss()
        self.index = self._faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []
        self.id_to_idx: Dict[str, int] = {}

    def add_faces(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
        face_id_prefix: str = "face",
    ) -> int:
        """添加人脸向量"""
        if len(embeddings) != len(metadata_list):
            raise ValueError("embeddings 与 metadata 数量不匹配")

        arr = np.array(embeddings).astype(np.float32)
        self._faiss.normalize_L2(arr)
        start = self.index.ntotal
        self.index.add(arr)

        for i, meta in enumerate(metadata_list):
            fid = meta.get("face_id", f"{face_id_prefix}_{start + i}")
            self.metadata.append(meta)
            self.id_to_idx[fid] = start + i
        return len(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似人脸"""
        if self.index.ntotal == 0:
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        self._faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results = []
        for s, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "similarity": float(s),
                    "metadata": self.metadata[idx],
                })
        return results

    def save(self) -> bool:
        """保存到 db_dir"""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.db_dir / "faiss_index.bin"))
        with open(self.db_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.db_dir / "id_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.id_to_idx, f, ensure_ascii=False, indent=2)
        return True

    def load(self) -> bool:
        """从 db_dir 加载"""
        idx_path = self.db_dir / "faiss_index.bin"
        if not idx_path.exists():
            return False
        self.index = self._faiss.read_index(str(idx_path))
        if (self.db_dir / "metadata.pkl").exists():
            with open(self.db_dir / "metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
        if (self.db_dir / "id_mapping.json").exists():
            with open(self.db_dir / "id_mapping.json", "r", encoding="utf-8") as f:
                self.id_to_idx = json.load(f)
        return True

    @property
    def count(self) -> int:
        return self.index.ntotal
