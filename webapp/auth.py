"""HierAD Web 的账号、会话与访问控制。

只依赖标准库：PBKDF2-HMAC-SHA256 做密码哈希，随机 token 做会话，
SQLite 存 token 的哈希（不存明文）。定位是局域网团队服务，不追求
公网级方案；但密码不落明文、会话可撤销、登录有基本限流。
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response
from fastapi import status

from hierad.config import WEB_COOKIE_SECURE, WEB_SESSION_HOURS

from . import db

PBKDF2_ITERATIONS = 260_000
SESSION_COOKIE = "hierad_session"

# 限流参数（进程内，够单机/局域网用）
LOGIN_WINDOW_SEC = 300
LOGIN_MAX_FAILS = 10
REGISTER_WINDOW_SEC = 3600
REGISTER_MAX = 10

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-一-鿿]{3,32}$")

_rate_lock = threading.Lock()
_rate_events: Dict[str, List[float]] = {}


class UsernameTaken(Exception):
    """用户名已存在。"""


# ==================== 密码 ====================


def _hash_raw(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def hash_password(password: str, iterations: Optional[int] = None) -> Tuple[str, str, int]:
    """返回 (hash_hex, salt_hex, iterations)。"""
    rounds = int(iterations or PBKDF2_ITERATIONS)
    salt = secrets.token_bytes(16)
    digest = _hash_raw(password, salt, rounds)
    return digest.hex(), salt.hex(), rounds


def verify_password(password: str, salt_hex: str, hash_hex: str, iterations: int) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        rounds = int(iterations)
    except (TypeError, ValueError):
        return False
    actual = _hash_raw(password, salt, rounds)
    return hmac.compare_digest(actual, expected)


def _dummy_verify(password: str) -> None:
    """用户名不存在时也做一次等价开销，缩小时序差异。"""
    _hash_raw(password, b"\x00" * 16, PBKDF2_ITERATIONS)


def validate_username(username: str) -> str:
    value = (username or "").strip()
    if not USERNAME_RE.match(value):
        raise ValueError("用户名需为 3–32 位，可用中英文、数字、下划线、点和短横线。")
    return value


def validate_password(password: str) -> str:
    value = password or ""
    if len(value) < 8:
        raise ValueError("密码至少 8 位。")
    if len(value) > 128:
        raise ValueError("密码不能超过 128 位。")
    if not value.strip():
        raise ValueError("密码不能全是空格。")
    return value


# ==================== 注册 / 登录 ====================


def register_user(username: str, password: str) -> Dict[str, Any]:
    username = validate_username(username)
    password = validate_password(password)
    digest, salt, rounds = hash_password(password)
    try:
        return db.create_user(username, digest, salt, rounds)
    except Exception as exc:  # sqlite3.IntegrityError
        if "UNIQUE" in str(exc).upper() or "users.username" in str(exc):
            raise UsernameTaken("用户名已被占用。") from exc
        raise


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    row = db.get_user_by_username((username or "").strip())
    if row is None:
        _dummy_verify(password or "")
        return None
    if not int(row.get("is_active") or 0):
        _dummy_verify(password or "")
        return None
    if not verify_password(
        password or "",
        str(row.get("salt") or ""),
        str(row.get("password_hash") or ""),
        int(row.get("iterations") or 0),
    ):
        return None
    db.set_last_login(int(row["id"]))
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "is_active": True,
    }


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "is_admin": bool(user.get("is_admin")),
    }


# ==================== 会话 ====================


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int, request: Request) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + WEB_SESSION_HOURS * 3600
    ip = request.client.host if request.client else ""
    user_agent = request.headers.get("user-agent", "")
    db.create_session(_token_hash(token), user_id, expires_at, ip, user_agent)
    return token


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=WEB_SESSION_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=WEB_COOKIE_SECURE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        httponly=True,
        samesite="lax",
        secure=WEB_COOKIE_SECURE,
        path="/",
    )


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return db.get_session_user(_token_hash(token))


def revoke_session(request: Request) -> None:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        db.delete_session(_token_hash(token))


# ==================== FastAPI 依赖 ====================


def require_user(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if not user:
        user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录。"
        )
    return user


def require_admin(request: Request) -> Dict[str, Any]:
    user = require_user(request)
    if not user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限。"
        )
    return user


# ==================== 限流 ====================


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def allow_request(key: str, limit: int, window_sec: float) -> bool:
    """滑动窗口计数：允许则记录一次并返回 True，超限返回 False。"""
    now = time.monotonic()
    with _rate_lock:
        events = [t for t in _rate_events.get(key, []) if now - t < window_sec]
        if len(events) >= limit:
            _rate_events[key] = events
            return False
        events.append(now)
        _rate_events[key] = events
        return True


def reset_rate(key: str) -> None:
    with _rate_lock:
        _rate_events.pop(key, None)


def login_rate_key(request: Request) -> str:
    return f"login:{client_ip(request)}"


def register_rate_key(request: Request) -> str:
    return f"register:{client_ip(request)}"
