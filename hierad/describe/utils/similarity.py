"""
余弦相似度计算 - 无额外依赖，仅 numpy
"""

import numpy as np


def compute_cosine_adjacent_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    计算相邻向量的余弦相似度

    Args:
        embeddings: 形状 (n, dim) 的向量矩阵

    Returns:
        相邻相似度数组，形状 (n-1,)
    """
    if len(embeddings) < 2:
        return np.array([])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    return np.sum(normalized[:-1] * normalized[1:], axis=1)


def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度矩阵

    Args:
        embeddings: 形状 (n, dim) 的向量矩阵

    Returns:
        相似度矩阵，形状 (n, n)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    return np.dot(normalized, normalized.T)
