"""Provider 选择与本机默认路径测试。"""

import json

from hierad.config import is_loopback_url, resolve_llm_settings
from hierad.describe.models import ClipDescription
from hierad.describe.stage1 import Stage1SceneUnderstanding
from hierad.describe.vlm_client import VLMClient
from hierad.providers.asr import _sentence_to_seg
from hierad.providers.vlm import LocalHttpVLM, create_vlm_backend


def test_is_loopback_url():
    assert is_loopback_url("http://127.0.0.1:8002")
    assert is_loopback_url("http://localhost:8001/asr_recognize")
    assert not is_loopback_url("https://dashscope.aliyuncs.com/compatible-mode/v1")


def test_resolve_llm_local_keeps_loopback(monkeypatch):
    monkeypatch.setenv("HIERAD_LLM_PROVIDER", "local")
    url, model, key = resolve_llm_settings(
        url="http://127.0.0.1:8016/v1",
        model="qwen2.5-7b",
        api_key="EMPTY",
    )
    assert "127.0.0.1" in url
    assert model == "qwen2.5-7b"
    assert key == "EMPTY"


def test_resolve_llm_dashscope_replaces_loopback(monkeypatch):
    monkeypatch.setenv("HIERAD_LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    url, model, key = resolve_llm_settings(
        url="http://127.0.0.1:8016/v1",
        model="qwen2.5-7b",
        api_key="EMPTY",
    )
    assert "dashscope.aliyuncs.com" in url
    assert model == "qwen-plus"
    assert key == "sk-test"


def test_create_vlm_backend_default_is_local(monkeypatch):
    monkeypatch.delenv("HIERAD_VLM_PROVIDER", raising=False)
    backend = create_vlm_backend()
    assert isinstance(backend, LocalHttpVLM)


def test_local_vlm_posts_video_path(monkeypatch, tmp_path):
    clip = tmp_path / "seg.mp4"
    clip.write_bytes(b"fake")

    class FakeResp:
        status_code = 200

        def json(self):
            return {"code": 0, "data": "SETTING: Room"}

    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr("hierad.providers.vlm.requests.post", fake_post)
    client = VLMClient(provider="local", vlm_url="http://127.0.0.1:8002")
    out = client.call(str(clip), "sys", "user")
    assert out == "SETTING: Room"
    assert captured["url"].endswith("/prompted_inference")
    assert captured["json"]["video_path"] == str(clip.resolve())
    assert "base64" not in json.dumps(captured["json"])


def test_stage1_checkpoint_skips_completed(tmp_path):
    ckpt = tmp_path / "stage1.json"
    ckpt.write_text(
        json.dumps([
            {
                "clip_idx": 0,
                "timecode": "00:00:00.000 --> 00:00:05.000",
                "start_sec": 0,
                "end_sec": 5,
                "setting": "Kitchen",
                "characters": "JOHN",
                "action": "Stirs soup.",
                "video_path": "a.mp4",
            }
        ]),
        encoding="utf-8",
    )
    called = []

    class DummyVLM:
        def call(self, video_path, system_prompt, user_prompt):
            called.append(video_path)
            return "SETTING: Hall\nCHARACTERS: MARY\nACTION: Walks."

    s1 = Stage1SceneUnderstanding.__new__(Stage1SceneUnderstanding)
    s1.vlm = DummyVLM()
    s1.context_window = 3
    s1.canonical_characters = {}
    clips = s1.describe_clips(
        [
            {
                "clip_idx": 0,
                "timecode": "00:00:00.000 --> 00:00:05.000",
                "start_sec": 0,
                "end_sec": 5,
                "video_path": "a.mp4",
                "dialogue_overlap": "(None)",
            }
        ],
        checkpoint_path=str(ckpt),
    )
    assert called == []
    assert clips[0].setting == "Kitchen"
    assert clips[0].is_complete()


def test_dashscope_asr_sentence_parser():
    seg = _sentence_to_seg({"begin_time": 1500, "end_time": 3200, "text": "  Hello "})
    assert seg == {"start": 1.5, "end": 3.2, "text": "Hello"}
    assert _sentence_to_seg({"text": "  "}) is None


def test_clip_description_roundtrip():
    c = ClipDescription(
        clip_idx=1,
        timecode_start="00:00:01.000",
        timecode_end="00:00:02.000",
        start_sec=1,
        end_sec=2,
        setting="Street",
        characters="A",
        action="Runs.",
        raw="SETTING: Street",
    )
    restored = ClipDescription.from_dict(c.to_dict())
    assert restored.clip_idx == 1
    assert restored.setting == "Street"
    assert restored.is_complete()
