"""
演员数据库构建脚本 - 参考 Actor_Dataset_Construct

1. build_from_tmdb: 仅元数据（角色名映射）
2. build_face_vector_db: 完整流程 TMDB -> 图片 -> 人脸向量 -> Faiss
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .tmdb import TMDBClient
from .image_crawler import download_actor_images


def build_from_tmdb(
    movie_name: str,
    output_dir: Path,
    tmdb_client: Optional[TMDBClient] = None,
) -> Optional[str]:
    """
    通过 TMDB 构建演员数据库

    Args:
        movie_name: 电影名称
        output_dir: 数据库存储根目录
        tmdb_client: TMDB 客户端，不传则自动创建

    Returns:
        数据库目录路径（UUID 命名），失败返回 None
    """
    client = tmdb_client or TMDBClient()
    results = client.search_movie(movie_name)
    if not results:
        return None

    movie_id = results[0]["id"]
    cast = client.get_movie_credits(movie_id)
    if not cast:
        return None

    db_id = str(uuid.uuid4())
    db_dir = Path(output_dir) / db_id
    db_dir.mkdir(parents=True, exist_ok=True)

    # 规范名 -> 别名列表
    mapping: Dict[str, List[str]] = {}
    for c in cast[:20]:
        name = c.get("name", "").strip()
        character = c.get("character", "").strip()
        if not name:
            continue
        canonical = name.upper()
        if canonical not in mapping:
            mapping[canonical] = []
        aliases = [name, character, canonical] if character else [name, canonical]
        for a in aliases:
            if a and a not in mapping[canonical]:
                mapping[canonical].append(a)

    # 扁平化为 别名 -> 规范名
    flat: Dict[str, str] = {}
    for canonical, aliases in mapping.items():
        for a in aliases:
            if a and a not in flat:
                flat[a] = canonical

    config = {
        "movie_name": movie_name,
        "movie_id": movie_id,
        "mapping": flat,
        "character_count": len(mapping),
    }

    with open(db_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return str(db_dir)


def _refresh_config_in_dir(
    db_dir: Path,
    movie_name: str,
    client: TMDBClient,
    max_actors: int = 20,
) -> bool:
    """刷新已有目录的 config.json（保留目录 UUID）"""
    results = client.search_movie(movie_name)
    if not results:
        return False
    movie_id = results[0]["id"]
    cast = client.get_movie_credits(movie_id)
    if not cast:
        return False

    mapping: Dict[str, List[str]] = {}
    for c in cast[:max_actors]:
        name = c.get("name", "").strip()
        character = c.get("character", "").strip()
        if not name:
            continue
        canonical = name.upper()
        if canonical not in mapping:
            mapping[canonical] = []
        aliases = [name, character, canonical] if character else [name, canonical]
        for a in aliases:
            if a and a not in mapping[canonical]:
                mapping[canonical].append(a)

    flat: Dict[str, str] = {}
    for canonical, aliases in mapping.items():
        for a in aliases:
            if a and a not in flat:
                flat[a] = canonical

    config = {
        "movie_name": movie_name,
        "movie_id": movie_id,
        "mapping": flat,
        "character_count": len(mapping),
    }
    db_dir.mkdir(parents=True, exist_ok=True)
    with open(db_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return True


def _populate_face_vectors(
    db_dir: Path,
    movie_name: str,
    actor_images: Dict[str, List[str]],
) -> None:
    """从已下载图片提取向量写入 Faiss"""
    from .face_processor import extract_face_embeddings
    from .vector_store import FaceVectorStore

    config_path = db_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    mapping = config.get("mapping", {})

    actor_to_role: Dict[str, str] = {}
    for alias, can in mapping.items():
        if alias.isupper() or alias.upper() == can.upper():
            continue
        norm_a = alias.upper().replace(".", "").replace(" ", "")
        norm_c = can.upper().replace(".", "").replace(" ", "")
        if norm_a == norm_c:
            continue
        for actor_name in actor_images:
            if mapping.get(actor_name, "").upper() == can.upper():
                actor_to_role.setdefault(actor_name, alias)

    store = FaceVectorStore(db_dir)
    for actor_name, paths in actor_images.items():
        canonical = mapping.get(actor_name) or mapping.get(actor_name.upper()) or actor_name.upper()
        role_name = actor_to_role.get(actor_name, actor_name)
        faces = extract_face_embeddings(paths, max_faces_per_image=1, min_score=0.8)
        for i, face in enumerate(faces):
            face["actor_name"] = actor_name
            face["canonical"] = canonical
            face["role_name"] = role_name
            face["character"] = role_name
            face["movie_title"] = movie_name
            face["face_id"] = f"{movie_name}_{canonical}_{i}"
        if faces:
            store.add_faces(
                [f["embedding"] for f in faces],
                faces,
                face_id_prefix=f"{movie_name}_{canonical}",
            )
    store.save()


def rebuild_face_vector_db(
    db_dir: Path,
    movie_name: Optional[str] = None,
    tmdb_client: Optional[TMDBClient] = None,
    max_actors: int = 20,
    max_images_per_actor: int = 5,
    with_vectors: bool = True,
) -> Optional[str]:
    """
    在已有目录原地重建演员库（刷新 mapping、重下图片、重建 Faiss）。
    """
    db_dir = Path(db_dir)
    if not db_dir.exists():
        return None

    client = tmdb_client or TMDBClient()
    if movie_name is None:
        config_path = db_dir / "config.json"
        if not config_path.exists():
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            movie_name = json.load(f).get("movie_name")
    if not movie_name:
        return None

    if not _refresh_config_in_dir(db_dir, movie_name, client, max_actors=max_actors):
        return None

    images_dir = db_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for old in images_dir.glob("*"):
        if old.is_file():
            old.unlink()

    for stale in ("faiss_index.bin", "metadata.pkl", "id_mapping.json"):
        p = db_dir / stale
        if p.exists():
            p.unlink()

    actor_images = download_actor_images(
        client,
        movie_name,
        images_dir,
        max_actors=max_actors,
        max_images_per_actor=max_images_per_actor,
    )
    if not actor_images:
        return str(db_dir)

    if not with_vectors:
        return str(db_dir)

    try:
        _populate_face_vectors(db_dir, movie_name, actor_images)
    except ImportError:
        return str(db_dir)

    return str(db_dir)


def build_face_vector_db(
    movie_name: str,
    output_dir: Path,
    tmdb_client: Optional[TMDBClient] = None,
    max_actors: int = 20,
    max_images_per_actor: int = 5,
    with_vectors: bool = True,
) -> Optional[str]:
    """
    构建电影演员人脸向量数据库（参考 Actor_Dataset_Construct）

    流程: TMDB 演员 -> 下载头像 -> InsightFace 提取向量 -> Faiss 存储

    Args:
        movie_name: 电影名称
        output_dir: 数据库存储根目录
        tmdb_client: TMDB 客户端
        max_actors: 最多演员数
        max_images_per_actor: 每位演员最多图片
        with_vectors: 是否构建向量库（需 insightface + faiss）

    Returns:
        数据库目录路径，失败返回 None
    """
    client = tmdb_client or TMDBClient()
    db_path = build_from_tmdb(movie_name, output_dir, client)
    if not db_path:
        return None

    db_dir = Path(db_path)
    images_dir = db_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 1. 下载演员图片
    actor_images = download_actor_images(
        client,
        movie_name,
        images_dir,
        max_actors=max_actors,
        max_images_per_actor=max_images_per_actor,
    )
    if not actor_images:
        return db_path

    # 2. 提取人脸向量并存入 Faiss（可选）
    if not with_vectors:
        return db_path

    try:
        _populate_face_vectors(db_dir, movie_name, actor_images)
    except ImportError:
        return db_path

    return db_path
