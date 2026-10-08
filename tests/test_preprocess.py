"""Preprocess 模块测试"""

import json
import tempfile
from pathlib import Path

import pytest

from hierad.preprocess.cutter import _sec_to_timecode
from hierad.preprocess.asr import ASRResult, segments_from_asr_service_payload, transcribe_with_asr_service
from hierad.preprocess.gap_segments import gaps_from_speech_segments


def test_segments_from_asr_service_payload():
    data = [
        {"start": 0.0, "end": 2.5, "text": " Hello ", "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 5.0, "text": "", "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 7.0, "text": "World", "speaker": 1},
    ]
    segs = segments_from_asr_service_payload(data)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello"
    assert segs[0]["speaker"] == "SPEAKER_00"


def test_transcribe_with_asr_service_mock(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "code": 0,
                "data": [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "A"}],
            }

    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json or {})
        return FakeResp()

    monkeypatch.setattr(
        "hierad.preprocess.asr.requests.post",
        fake_post,
    )
    out = tmp_path / "out.json"
    result = transcribe_with_asr_service(str(wav), output_json_path=str(out), asr_url="http://test/asr")
    assert len(result) == 1
    assert captured.get("diarize") is False
    assert out.exists()


def test_gaps_from_asr_json():
    """ASR JSON 应对白时间计算 AD 间隙，而非直接切对白段"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 10.0, "end": 12.0, "text": "World"},
            ]
        }, f)
        path = f.name
    try:
        asr = ASRResult.from_whisper_json(path)
        speech = [(s["start"], s["end"]) for s in asr.segments]
        gaps = gaps_from_speech_segments(speech, video_duration=20.0, min_gap_sec=2.0)
        assert any(g[0] >= 2.0 and g[1] <= 10.0 for g in gaps)
    finally:
        Path(path).unlink(missing_ok=True)


def test_sec_to_timecode():
    assert _sec_to_timecode(0) == "00:00:00.000"
    assert _sec_to_timecode(65.5) == "00:01:05.500"


def test_asr_result_from_whisper():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hi"},
                {"start": 1.0, "end": 2.0, "text": "Bye"},
            ]
        }, f)
        path = f.name
    try:
        asr = ASRResult.from_whisper_json(path)
        assert len(asr) == 2
        assert asr.segments[0]["text"] == "Hi"
    finally:
        Path(path).unlink(missing_ok=True)
