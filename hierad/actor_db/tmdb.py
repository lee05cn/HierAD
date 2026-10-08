"""
TMDB API 客户端 - 仅用于构建演员数据库

本机无法直连外网时，通过 Clash 等代理访问：
  export HIERAD_TMDB_PROXY=http://<你的电脑局域网IP>:7097
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import requests

from hierad.config import TMDB_API_KEY as _CONFIG_TMDB_KEY
from hierad.config import TMDB_PROXY as _CONFIG_TMDB_PROXY


def resolve_tmdb_proxy(explicit: Optional[str] = None) -> str:
    """解析 TMDB 代理 URL（HIERAD_TMDB_PROXY > config.yaml > HTTPS_PROXY）"""
    return (explicit or _CONFIG_TMDB_PROXY or "").strip()


class TMDBClient:
    """TMDB API 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        api_key = api_key or _CONFIG_TMDB_KEY or ""
        if not api_key or api_key == "your_tmdb_api_key_here":
            raise ValueError("TMDB API 密钥未配置，请设置 HIERAD_TMDB_API_KEY 或传入 api_key")

        self.api_key = api_key
        self.proxy_url = resolve_tmdb_proxy(proxy)
        self.base_url = "https://api.themoviedb.org/3"
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        if self.proxy_url:
            self._session.proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url,
            }

    def _request(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{self.base_url}{path}"
        params = params or {}
        params["api_key"] = self.api_key

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(1)
        if last_err:
            return None
        return None

    def search_movie(self, query: str) -> List[Dict]:
        """搜索电影"""
        data = self._request("/search/movie", {"query": query})
        if not data or "results" not in data:
            return []
        return data["results"]

    def get_movie_credits(self, movie_id: int) -> List[Dict]:
        """获取电影演员表"""
        data = self._request(f"/movie/{movie_id}/credits")
        if not data or "cast" not in data:
            return []
        return data["cast"]

    def get_person_images(self, person_id: int) -> List[Dict]:
        """获取演员头像列表"""
        data = self._request(f"/person/{person_id}/images")
        if not data or "profiles" not in data:
            return []
        return data["profiles"]

    def get_full_image_url(self, file_path: str, size: str = "original") -> str:
        """构建完整图片 URL"""
        if not file_path:
            return ""
        base = "https://image.tmdb.org/t/p"
        return f"{base}/{size}{file_path}"


def check_tmdb_connection(
    query: str = "Signs",
    api_key: Optional[str] = None,
    proxy: Optional[str] = None,
    timeout: int = 15,
) -> Tuple[bool, str]:
    """
    检测 TMDB 是否可达。

    Returns:
        (ok, message)
    """
    proxy_url = resolve_tmdb_proxy(proxy)
    key = api_key or _CONFIG_TMDB_KEY or ""
    lines = ["=== TMDB 连接检查 ==="]

    if not key:
        return False, "未配置 HIERAD_TMDB_API_KEY"

    lines.append(f"API Key: {key[:8]}...{key[-4:]}")
    lines.append(f"代理: {proxy_url or '(无，直连)'}")

    session = requests.Session()
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}

    url = "https://api.themoviedb.org/3/search/movie"
    t0 = time.time()
    try:
        r = session.get(url, params={"api_key": key, "query": query}, timeout=timeout)
        elapsed = time.time() - t0
        if not r.ok:
            return False, "\n".join(lines + [f"HTTP {r.status_code}: {r.text[:200]}"])
        data = r.json()
        results = data.get("results", [])
        if not results:
            return False, "\n".join(lines + [f"HTTP 200 但无搜索结果 (query={query!r})"])
        m = results[0]
        lines.append(f"OK  {elapsed:.2f}s  query={query!r}")
        lines.append(f"  → id={m.get('id')}  {m.get('title')} ({m.get('release_date', '')})")
        return True, "\n".join(lines)
    except requests.RequestException as e:
        elapsed = time.time() - t0
        hint = (
            "服务器无法直连外网时，请在 Clash Verge 开启「允许局域网连接」，"
            "并设置: export HIERAD_TMDB_PROXY=http://<你的电脑IP>:7097"
        )
        return False, "\n".join(lines + [f"失败 ({elapsed:.1f}s): {e}", hint])
