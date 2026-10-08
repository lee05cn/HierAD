"""
演员数据库 - 与 preprocess 目录平级，业务上属于【预处理最前】步骤。

应在描述流水线（Stage1/2/3）之前构建，供:
  - 角色名映射（run 时 --canonical-characters）
  - 可选人脸向量检索 + preprocess.face_annotator 视频标注

职责:
  - 存储: config.json (角色映射) + Faiss (人脸向量)
  - 构建: TMDB -> 图片爬取 -> InsightFace 人脸向量 -> Faiss
"""

from .store import ActorDatabase
from .build import build_from_tmdb, build_face_vector_db, rebuild_face_vector_db

__all__ = ["ActorDatabase", "build_from_tmdb", "build_face_vector_db", "rebuild_face_vector_db"]
