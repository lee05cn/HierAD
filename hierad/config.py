"""
HierAD 统一配置

优先级: 环境变量 > config.yaml > 默认值

模型调用默认走本机 HTTP 服务（provider=local）。
设置 llm/vlm/asr/tts.provider=dashscope 时改走阿里云百炼，互不影响。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_DEFAULT_LLM = "qwen-plus"
DASHSCOPE_DEFAULT_VLM = "qwen3-vl-plus"
DASHSCOPE_DEFAULT_ASR = "fun-asr-realtime"
DASHSCOPE_DEFAULT_TTS = "cosyvoice-v3-flash"
DASHSCOPE_DEFAULT_VOICE = "longxiaochun_v3"

# ==================== 默认配置（本机服务） ====================
_DEFAULTS = {
    "LLM_PROVIDER": "local",
    "LLM_API_KEY": "EMPTY",
    "LLM_BASE_URL": "http://127.0.0.1:8014/v1",
    "LLM_MODEL": "llama3-8b",
    "VLM_PROVIDER": "local",
    "VLM_URL": "http://localhost:8002",
    "VLM_TYPE": "qwen2.5-vl",
    "VLM_ENDPOINT": "/prompted_inference",
    "VLM_MODEL": DASHSCOPE_DEFAULT_VLM,
    "VLM_FPS": "1.0",
    "VLM_MAX_FRAMES": "16",
    "ASR_PROVIDER": "local",
    "ASR_URL": "http://127.0.0.1:8001/asr_recognize",
    "ASR_DIARIZE": "false",
    "ASR_MODEL": DASHSCOPE_DEFAULT_ASR,
    "TTS_PROVIDER": "local",
    "TTS_URL": "http://127.0.0.1:8003",
    "TTS_SPK_ID": "",
    "TTS_SPEED": "1.2",
    "TTS_VOICE_NAME": "康辉",
    "TTS_VOICE_ID": "",
    "TTS_MODEL": DASHSCOPE_DEFAULT_TTS,
    "TTS_CLOUD_VOICE": DASHSCOPE_DEFAULT_VOICE,
    "VOICE_DB_PATH": "",
    "TRANSLATE_ZH": "true",
    "TMDB_API_KEY": "",
    "TMDB_PROXY": "",
    "ACTOR_DATABASES_DIR": str(PROJECT_ROOT / "actor_databases"),
    "DASHSCOPE_API_KEY": "",
    "DASHSCOPE_BASE_URL": DASHSCOPE_COMPAT_URL,
}


def _load_yaml_config() -> Dict[str, Any]:
    """可选加载 config.yaml（项目根或当前目录）"""
    for base in (PROJECT_ROOT, Path.cwd()):
        path = base / "config.yaml"
        if path.exists():
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


_YAML = _load_yaml_config()


def _section(name: str) -> Dict[str, Any]:
    data = _YAML.get(name) or {}
    return data if isinstance(data, dict) else {}


def _pick(env_key: str, *yaml_keys: str, default: str = "") -> str:
    env = os.getenv(env_key)
    if env is not None and str(env).strip() != "":
        return str(env).strip()
    cur: Any = _YAML
    for key in yaml_keys:
        if not isinstance(cur, dict):
            cur = None
            break
        cur = cur.get(key)
    if cur is not None and str(cur).strip() != "":
        return str(cur).strip()
    return default


def _norm_provider(value: str) -> str:
    v = (value or "local").strip().lower()
    aliases = {
        "local_http": "local",
        "local_cosyvoice": "local",
        "whisper": "local",
        "service": "local",
        "qwen": "dashscope",
        "bailian": "dashscope",
        "cloud": "dashscope",
    }
    return aliases.get(v, v)


def is_loopback_url(url: Optional[str]) -> bool:
    if not url:
        return True
    host = (urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


# ==================== API 配置 ====================
LLM_PROVIDER = _norm_provider(
    _pick("HIERAD_LLM_PROVIDER", "llm", "provider", default=_DEFAULTS["LLM_PROVIDER"])
)
LLM_API_KEY = _pick("HIERAD_LLM_API_KEY", "llm", "api_key", default=_DEFAULTS["LLM_API_KEY"])
LLM_BASE_URL = _pick("HIERAD_LLM_BASE_URL", "llm", "base_url", default=_DEFAULTS["LLM_BASE_URL"])
LLM_MODEL = _pick("HIERAD_LLM_MODEL", "llm", "model", default=_DEFAULTS["LLM_MODEL"])

VLM_PROVIDER = _norm_provider(
    _pick("HIERAD_VLM_PROVIDER", "vlm", "provider", default=_DEFAULTS["VLM_PROVIDER"])
)
VLM_URL = _pick("HIERAD_VLM_URL", "vlm", "url", default=_DEFAULTS["VLM_URL"])
VLM_TYPE = _pick("HIERAD_VLM_TYPE", "vlm", "type", default=_DEFAULTS["VLM_TYPE"])
VLM_ENDPOINT = _pick("HIERAD_VLM_ENDPOINT", "vlm", "endpoint", default=_DEFAULTS["VLM_ENDPOINT"])
VLM_MODEL = _pick("HIERAD_VLM_MODEL", "vlm", "model", default=_DEFAULTS["VLM_MODEL"])
try:
    VLM_FPS = float(_pick("HIERAD_VLM_FPS", "vlm", "fps", default=_DEFAULTS["VLM_FPS"]))
except (TypeError, ValueError):
    VLM_FPS = 1.0
try:
    VLM_MAX_FRAMES = int(_pick("HIERAD_VLM_MAX_FRAMES", "vlm", "max_frames", default=_DEFAULTS["VLM_MAX_FRAMES"]))
except (TypeError, ValueError):
    VLM_MAX_FRAMES = 16

ASR_PROVIDER = _norm_provider(
    _pick("HIERAD_ASR_PROVIDER", "asr", "provider", default=_DEFAULTS["ASR_PROVIDER"])
)
ASR_URL = _pick("HIERAD_ASR_URL", "asr", "url", default=_DEFAULTS["ASR_URL"])
_asr_diarize = _pick("HIERAD_ASR_DIARIZE", "asr", "diarize", default=_DEFAULTS["ASR_DIARIZE"])
ASR_DIARIZE = str(_asr_diarize).lower() in ("1", "true", "yes", "on")
ASR_MODEL = _pick("HIERAD_ASR_MODEL", "asr", "model", default=_DEFAULTS["ASR_MODEL"])

TTS_PROVIDER = _norm_provider(
    _pick("HIERAD_TTS_PROVIDER", "tts", "provider", default=_DEFAULTS["TTS_PROVIDER"])
)
_tts_yaml = _section("tts")
TTS_URL = _pick("HIERAD_TTS_URL", "tts", "url", default=_DEFAULTS["TTS_URL"])
TTS_SPK_ID = _pick("HIERAD_TTS_SPK_ID", "tts", "spk_id", default=_DEFAULTS["TTS_SPK_ID"])
TTS_VOICE_NAME = _pick("HIERAD_TTS_VOICE_NAME", "tts", "voice_name", default=_DEFAULTS["TTS_VOICE_NAME"])
TTS_VOICE_ID = _pick("HIERAD_TTS_VOICE_ID", "tts", "voice_id", default=_DEFAULTS["TTS_VOICE_ID"])
TTS_MODEL = _pick("HIERAD_TTS_MODEL", "tts", "model", default=_DEFAULTS["TTS_MODEL"])
TTS_CLOUD_VOICE = _pick(
    "HIERAD_TTS_CLOUD_VOICE",
    "tts",
    "cloud_voice",
    default=_DEFAULTS["TTS_CLOUD_VOICE"],
)
VOICE_DB_PATH = (
    os.getenv("HIERAD_VOICE_DB")
    or _tts_yaml.get("voice_db")
    or (_YAML.get("paths") or {}).get("voice_db")
    or _DEFAULTS["VOICE_DB_PATH"]
)
VOICE_NAME = TTS_VOICE_NAME  # alias for voice.resolve
try:
    TTS_SPEED = float(_pick("HIERAD_TTS_SPEED", "tts", "speed", default=_DEFAULTS["TTS_SPEED"]))
except (TypeError, ValueError):
    TTS_SPEED = float(_DEFAULTS["TTS_SPEED"])

_translate = _pick("HIERAD_TRANSLATE_ZH", "export", "translate_zh", default=_DEFAULTS["TRANSLATE_ZH"])
TRANSLATE_ZH = str(_translate).lower() in ("1", "true", "yes", "on")

TMDB_API_KEY = _pick("HIERAD_TMDB_API_KEY", "tmdb", "api_key", default=_DEFAULTS["TMDB_API_KEY"])
TMDB_PROXY = (
    os.getenv("HIERAD_TMDB_PROXY")
    or (_section("tmdb").get("proxy") or "")
    or os.getenv("HTTPS_PROXY")
    or os.getenv("HTTP_PROXY")
    or _DEFAULTS["TMDB_PROXY"]
)

DASHSCOPE_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("HIERAD_DASHSCOPE_API_KEY")
    or (_section("dashscope").get("api_key") or "")
    or _DEFAULTS["DASHSCOPE_API_KEY"]
)
DASHSCOPE_BASE_URL = _pick(
    "HIERAD_DASHSCOPE_BASE_URL",
    "dashscope",
    "base_url",
    default=_DEFAULTS["DASHSCOPE_BASE_URL"],
)

# ==================== 路径配置 ====================
DEFAULT_WORK_DIR = PROJECT_ROOT / "work_dir"
ACTOR_DATABASES_DIR = Path(
    os.getenv("HIERAD_ACTOR_DB_DIR")
    or (_YAML.get("paths") or {}).get("actor_databases")
    or _DEFAULTS["ACTOR_DATABASES_DIR"]
)


def get_llm_provider() -> str:
    return _norm_provider(
        os.getenv("HIERAD_LLM_PROVIDER")
        or (_section("llm").get("provider") or LLM_PROVIDER)
    )


def get_vlm_provider() -> str:
    return _norm_provider(
        os.getenv("HIERAD_VLM_PROVIDER")
        or (_section("vlm").get("provider") or VLM_PROVIDER)
    )


def get_asr_provider() -> str:
    return _norm_provider(
        os.getenv("HIERAD_ASR_PROVIDER")
        or (_section("asr").get("provider") or ASR_PROVIDER)
    )


def get_tts_provider() -> str:
    return _norm_provider(
        os.getenv("HIERAD_TTS_PROVIDER")
        or (_section("tts").get("provider") or TTS_PROVIDER)
    )


def get_dashscope_api_key() -> str:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("HIERAD_DASHSCOPE_API_KEY")
        or DASHSCOPE_API_KEY
        or ""
    ).strip()


def resolve_llm_settings(
    url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Tuple[str, str, str]:
    """按当前 provider 解析 LLM 连接参数。CLI 覆盖优先。"""
    prov = _norm_provider(provider or get_llm_provider())
    if prov == "dashscope":
        base = url or os.getenv("HIERAD_LLM_BASE_URL") or (_section("llm").get("base_url") or "")
        if not base or is_loopback_url(str(base)):
            base = DASHSCOPE_BASE_URL or DASHSCOPE_COMPAT_URL
        mdl = model or os.getenv("HIERAD_LLM_MODEL") or (_section("llm").get("model") or "") or DASHSCOPE_DEFAULT_LLM
        if mdl in ("llama3-8b", "qwen2.5:7b", "qwen2.5-7b"):
            mdl = DASHSCOPE_DEFAULT_LLM
        key = (
            api_key
            or os.getenv("HIERAD_LLM_API_KEY")
            or (_section("llm").get("api_key") or "")
            or get_dashscope_api_key()
        )
        if not key or key == "EMPTY":
            key = get_dashscope_api_key()
        if not key:
            raise ValueError("DashScope LLM 需要 DASHSCOPE_API_KEY 或 llm.api_key")
        return str(base).rstrip("/"), str(mdl), str(key)
    return (
        (url or LLM_BASE_URL).rstrip("/"),
        model or LLM_MODEL,
        api_key or LLM_API_KEY,
    )


def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """获取 OpenAI 兼容的 LLM 客户端（本机或百炼）"""
    from openai import OpenAI

    resolved_url, _, resolved_key = resolve_llm_settings(url=base_url, api_key=api_key)
    return OpenAI(
        api_key=resolved_key,
        base_url=resolved_url,
    )
