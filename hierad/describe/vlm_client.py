"""共享 VLM 客户端。默认本机 HTTP；provider=dashscope 时走百炼。"""

from typing import Optional

from hierad.config import VLM_ENDPOINT, VLM_TYPE, VLM_URL, get_vlm_provider
from hierad.providers.vlm import create_vlm_backend


class VLMClient:
    def __init__(
        self,
        vlm_url: str = "http://localhost:8002",
        vlm_type: str = "qwen2.5-vl",
        vlm_endpoint: str = "/prompted_inference",
        timeout: int = 180,
        retries: int = 3,
        provider: Optional[str] = None,
    ):
        self.vlm_url = (vlm_url or VLM_URL).rstrip("/")
        self.vlm_type = vlm_type or VLM_TYPE
        ep = (vlm_endpoint or VLM_ENDPOINT or "/prompted_inference").strip()
        self.vlm_endpoint = ep if ep.startswith("/") else f"/{ep}"
        self.timeout = timeout
        self.retries = retries
        self.provider = provider or get_vlm_provider()
        self._backend = create_vlm_backend(
            provider=self.provider,
            vlm_url=self.vlm_url,
            vlm_type=self.vlm_type,
            vlm_endpoint=self.vlm_endpoint,
            timeout=timeout,
            retries=retries,
        )

    def call(self, video_path: str, system_prompt: str, user_prompt: str) -> str:
        return self._backend.call(video_path, system_prompt, user_prompt)
