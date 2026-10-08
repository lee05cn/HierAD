"""VLM 后端：本机 HTTP（默认）与阿里云百炼。"""

from __future__ import annotations

import base64
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Protocol

import requests

from hierad.config import (
    DASHSCOPE_COMPAT_URL,
    VLM_ENDPOINT,
    VLM_FPS,
    VLM_MAX_FRAMES,
    VLM_MODEL,
    VLM_TYPE,
    VLM_URL,
    get_dashscope_api_key,
    get_vlm_provider,
    is_loopback_url,
)


class VLMBackend(Protocol):
    def call(self, video_path: str, system_prompt: str, user_prompt: str) -> str: ...


class LocalHttpVLM:
    """现有本机 /prompted_inference，传本地 video_path。"""

    def __init__(
        self,
        vlm_url: str = VLM_URL,
        vlm_type: str = VLM_TYPE,
        vlm_endpoint: str = VLM_ENDPOINT,
        timeout: int = 180,
        retries: int = 3,
    ):
        self.vlm_url = (vlm_url or VLM_URL).rstrip("/")
        self.vlm_type = vlm_type or VLM_TYPE
        ep = (vlm_endpoint or "/prompted_inference").strip()
        self.vlm_endpoint = ep if ep.startswith("/") else f"/{ep}"
        self.timeout = timeout
        self.retries = retries

    def call(self, video_path: str, system_prompt: str, user_prompt: str) -> str:
        abs_path = str(Path(video_path).resolve())
        if not Path(abs_path).is_file():
            raise FileNotFoundError(f"视频不存在: {abs_path}")
        data = {
            "video_path": abs_path,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        if self.vlm_type in ("video-xl2", "videollama3"):
            data = {"video_path": abs_path, "prompt": f"{system_prompt}\n\n{user_prompt}"}
            if self.vlm_type == "video-xl2":
                data["user_prompt"] = data["prompt"]

        url = f"{self.vlm_url}{self.vlm_endpoint}"
        last_err: Optional[Exception] = None
        for _ in range(self.retries):
            try:
                r = requests.post(url, json=data, timeout=self.timeout)
                if r.status_code != 200:
                    raise ValueError(f"VLM HTTP {r.status_code}")
                j = r.json()
                if j.get("code", 0) != 0:
                    raise ValueError(f"VLM error: {j.get('msg', j.get('message', 'unknown'))}")
                out = j.get("data") or j.get("result") or j.get("output") or j.get("text")
                if out is not None:
                    return str(out).strip()
                raise ValueError("VLM response missing content field")
            except (requests.RequestException, ValueError) as e:
                last_err = e
                time.sleep(2)
        raise RuntimeError(f"VLM 调用失败: {last_err}")


def _ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return max(float(out.stdout.strip() or 0), 0.1)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 1.0


def _compress_clip(src: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", "scale='min(640,iw)':-2",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
        "-movflags", "+faststart",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _extract_jpeg_frames(src: Path, dest_dir: Path, max_frames: int) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    duration = _ffprobe_duration(str(src))
    fps = min(2.0, max(max_frames / max(duration, 0.1), 0.1))
    pattern = dest_dir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"fps={fps:.4f},scale='min(640,iw)':-2",
        "-q:v", "5",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    frames = sorted(dest_dir.glob("frame_*.jpg"))
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return frames


class DashScopeVLM:
    """百炼 OpenAI 兼容视觉接口：上传 clip（video_url）或抽帧。"""

    def __init__(
        self,
        model: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        timeout: int = 180,
        retries: int = 3,
        max_video_bytes: int = 8 * 1024 * 1024,
    ):
        self.model = model or VLM_MODEL
        self.fps = float(VLM_FPS if fps is None else fps)
        self.max_frames = int(VLM_MAX_FRAMES if max_frames is None else max_frames)
        self.timeout = timeout
        self.retries = retries
        self.max_video_bytes = max_video_bytes

    def _client(self):
        from openai import OpenAI

        key = get_dashscope_api_key()
        if not key:
            raise ValueError("DashScope VLM 需要环境变量 DASHSCOPE_API_KEY")
        return OpenAI(api_key=key, base_url=DASHSCOPE_COMPAT_URL, timeout=self.timeout)

    def _content_from_video(self, video_path: Path) -> list:
        with tempfile.TemporaryDirectory(prefix="hierad_vlm_") as tmp:
            tmp_path = Path(tmp)
            payload = video_path
            try:
                compressed = tmp_path / "clip.mp4"
                _compress_clip(video_path, compressed)
                if compressed.exists() and compressed.stat().st_size > 0:
                    payload = compressed
            except (FileNotFoundError, subprocess.CalledProcessError):
                payload = video_path

            if payload.stat().st_size <= self.max_video_bytes:
                b64 = base64.b64encode(payload.read_bytes()).decode("ascii")
                return [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                        "fps": self.fps,
                    }
                ]

            frames = _extract_jpeg_frames(video_path, tmp_path / "frames", self.max_frames)
            if not frames:
                raise RuntimeError(f"无法从视频抽帧: {video_path}")
            urls = []
            for frame in frames:
                b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
                urls.append(f"data:image/jpeg;base64,{b64}")
            return [{"type": "video", "video": urls, "fps": self.fps}]

    def call(self, video_path: str, system_prompt: str, user_prompt: str) -> str:
        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"视频不存在: {path}")
        visual = self._content_from_video(path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": visual + [{"type": "text", "text": user_prompt}],
            },
        ]
        last_err: Optional[Exception] = None
        client = self._client()
        for _ in range(self.retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body={"enable_thinking": False},
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("DashScope VLM 返回空内容")
                return text
            except Exception as e:
                last_err = e
                time.sleep(2)
        raise RuntimeError(f"DashScope VLM 调用失败: {last_err}")


def create_vlm_backend(
    *,
    provider: Optional[str] = None,
    vlm_url: Optional[str] = None,
    vlm_type: Optional[str] = None,
    vlm_endpoint: Optional[str] = None,
    timeout: int = 180,
    retries: int = 3,
) -> VLMBackend:
    prov = (provider or get_vlm_provider()).lower()
    if prov in ("dashscope", "qwen", "cloud"):
        return DashScopeVLM(timeout=timeout, retries=retries)
    url = vlm_url or VLM_URL
    if prov == "local" and url and not is_loopback_url(url):
        # 仍按本机协议调用自定义 HTTP 服务
        pass
    return LocalHttpVLM(
        vlm_url=url or VLM_URL,
        vlm_type=vlm_type or VLM_TYPE,
        vlm_endpoint=vlm_endpoint or VLM_ENDPOINT,
        timeout=timeout,
        retries=retries,
    )
