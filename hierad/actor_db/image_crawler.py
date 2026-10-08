"""
TMDB 演员图片爬取 - 参考 Actor_Dataset_Construct

从 TMDB 获取演员头像并下载到本地
"""

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .tmdb import TMDBClient


def _validate_image(path: Path) -> bool:
    """验证图片文件"""
    if not path.exists() or path.stat().st_size < 500:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        return (
            header.startswith(b"\xff\xd8\xff")
            or header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"GIF8")
        )
    except Exception:
        return False


def download_actor_images(
    tmdb_client: TMDBClient,
    movie_name: str,
    output_dir: Path,
    max_actors: int = 20,
    max_images_per_actor: int = 5,
    min_resolution: int = 300,
) -> Dict[str, List[str]]:
    """
    为电影演员下载 TMDB 头像

    Args:
        tmdb_client: TMDB 客户端
        movie_name: 电影名称
        output_dir: 输出目录
        max_actors: 最多演员数
        max_images_per_actor: 每位演员最多图片数
        min_resolution: 最小分辨率

    Returns:
        {actor_name: [image_path, ...]}
    """
    results = tmdb_client.search_movie(movie_name)
    if not results:
        return {}

    movie_id = results[0]["id"]
    cast = tmdb_client.get_movie_credits(movie_id)[:max_actors]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_images: Dict[str, List[str]] = {}
    seen_hashes: set = set()

    for actor in cast:
        name = actor.get("name", "").strip()
        person_id = actor.get("id")
        character = actor.get("character", "")
        if not name or not person_id:
            continue

        # 获取演员图片
        profiles = tmdb_client.get_person_images(person_id)
        if not profiles:
            continue
        downloaded = []

        for i, p in enumerate(profiles[:max_images_per_actor * 2]):
            w = p.get("width", 0)
            h = p.get("height", 0)
            if w < min_resolution or h < min_resolution:
                continue

            url = tmdb_client.get_full_image_url(p.get("file_path", ""), "original")
            safe_name = "".join(c for c in name if c.isalnum() or c in " -_")
            fname = f"{person_id}_{safe_name}_{i:02d}.jpg"
            save_path = output_dir / fname

            try:
                r = requests.get(url, timeout=30, stream=True)
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                if _validate_image(save_path):
                    hsh = hashlib.md5(save_path.read_bytes()).hexdigest()
                    if hsh not in seen_hashes:
                        seen_hashes.add(hsh)
                        downloaded.append(str(save_path))
                else:
                    save_path.unlink(missing_ok=True)
            except Exception:
                save_path.unlink(missing_ok=True)

            if len(downloaded) >= max_images_per_actor:
                break
            time.sleep(0.1)

        if downloaded:
            actor_images[name] = downloaded
        time.sleep(0.05)

    return actor_images
