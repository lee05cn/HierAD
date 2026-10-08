"""AD → SRT / TTS / 压制导出测试"""

from pathlib import Path
from unittest.mock import patch

import pytest

from hierad.export.ad_video import (
    ad_scripts_to_srt,
    burn_ad_subtitles,
    export_ad_video,
    write_srt_file,
)
from hierad.export.tts import build_ad_track, timecode_to_sec


SAMPLE = [
    {
        "ad_idx": 0,
        "start_time": "00:00:00.000",
        "end_time": "00:00:49.210",
        "ad_script": "Tony Stark looks worried as he drives.",
    },
    {
        "ad_idx": 1,
        "start_time": "00:06:02.520",
        "end_time": "00:06:15.790",
        "ad_script": "Hogan tries to calm Tony.\nSecond line.",
    },
    {
        "ad_idx": 2,
        "start_time": "00:09:00.000",
        "end_time": "00:09:05.000",
        "ad_script": "   ",
    },
]


def test_ad_scripts_to_srt_timecode_and_body():
    srt = ad_scripts_to_srt(SAMPLE)
    assert "1\n00:00:00,000 --> 00:00:49,210\nTony Stark looks worried as he drives.\n" in srt
    assert "2\n00:06:02,520 --> 00:06:15,790\nHogan tries to calm Tony.\nSecond line.\n" in srt
    assert "3\n" not in srt


def test_write_srt_file(tmp_path):
    path = write_srt_file(SAMPLE, tmp_path / "ad.srt")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("1\n")
    assert "00:00:00,000 --> 00:00:49,210" in text


def test_timecode_to_sec():
    assert abs(timecode_to_sec("00:06:02.520") - 362.52) < 1e-6
    assert abs(timecode_to_sec("1:02:03,500") - 3723.5) < 1e-6


def test_build_ad_track_places_at_start(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    clip = Sine(440).to_audio_segment(duration=500).set_frame_rate(22050).set_channels(1)
    wav1 = tmp_path / "a.wav"
    wav2 = tmp_path / "b.wav"
    clip.export(wav1, format="wav")
    clip.export(wav2, format="wav")

    entries = [
        {"start_time": "00:00:01.000", "end_time": "00:00:05.000", "ad_script": "a"},
        {"start_time": "00:00:03.000", "end_time": "00:00:08.000", "ad_script": "b"},
    ]
    out = build_ad_track(entries, [str(wav1), str(wav2)], tmp_path / "track.wav", track_len_sec=10.0)
    track = AudioSegment.from_file(out)
    assert abs(len(track) - 10000) < 50
    # 1.0s–1.5s 应有能量；0–0.5s 近似静音
    early = track[0:400].rms
    mid = track[1000:1400].rms
    assert mid > early + 50


def test_fit_clip_to_gap_speeds_up_instead_of_hard_cut(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    from hierad.export.tts import fit_clip_to_gap

    # 2s 音频压进 1.4s gap → 应加速而非砍掉后半句
    clip = Sine(440).to_audio_segment(duration=2000).set_frame_rate(22050).set_channels(1)
    fitted = fit_clip_to_gap(clip, 1400, max_speedup=1.6, fade_ms=0)
    assert len(fitted) <= 1400 + 40
    assert len(fitted) >= 1200  # 不应被裁成极短碎片


def test_export_ad_video_with_tts(tmp_path):
    stage3 = tmp_path / "stage3_ad_scripts.json"
    stage3.write_text(
        __import__("json").dumps(SAMPLE[:2], ensure_ascii=False),
        encoding="utf-8",
    )
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    fake_wav = tmp_path / "ad_audio.wav"
    fake_wav.write_bytes(b"RIFF")

    def fake_burn(video_path, srt_path, output_path, **kwargs):
        assert kwargs.get("ad_wav_path") == fake_wav
        out = Path(output_path)
        out.write_bytes(b"mp4")
        return out

    zh = ["托尼担忧地开车。", "霍根安抚托尼。"]
    with patch("hierad.export.ad_video.prepare_zh_entries", side_effect=lambda entries, **kw: [
            {**e, "ad_script_en": e["ad_script"], "ad_script": zh[i], "ad_script_zh": zh[i]}
            for i, e in enumerate(entries) if (e.get("ad_script") or "").strip()
        ] + [e for e in entries if not (e.get("ad_script") or "").strip()]), \
         patch("hierad.export.ad_video.resolve_voice_id_and_prompt", return_value=("vid-kang", "prompt", "康辉_test")), \
         patch("hierad.export.ad_video.get_video_duration", return_value=100.0), \
         patch("hierad.export.ad_video.synthesize_and_build_track", return_value=(fake_wav, [])) as synth, \
         patch("hierad.export.ad_video.burn_ad_subtitles", side_effect=fake_burn):
        result = export_ad_video(stage3, video, work_dir=tmp_path, enable_tts=True, translate_zh=True)

    assert Path(result["srt_path"]).exists()
    srt = Path(result["srt_path"]).read_text(encoding="utf-8")
    assert "托尼担忧地开车" in srt
    assert result["ad_audio_path"] == str(fake_wav)
    assert result["voice_id"] == "vid-kang"
    assert synth.call_args.kwargs.get("voice_id") == "vid-kang"
    assert synth.call_args.kwargs.get("spk_id") in (None, "")


def test_export_ad_video_tts_failure_falls_back(tmp_path):
    stage3 = tmp_path / "stage3_ad_scripts.json"
    stage3.write_text(
        __import__("json").dumps(SAMPLE[:1], ensure_ascii=False),
        encoding="utf-8",
    )
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    def fake_burn(video_path, srt_path, output_path, **kwargs):
        assert kwargs.get("ad_wav_path") is None
        Path(output_path).write_bytes(b"mp4")
        return Path(output_path)

    with patch("hierad.export.ad_video.prepare_zh_entries", side_effect=lambda entries, **kw: list(entries)), \
         patch("hierad.export.ad_video.resolve_voice_id_and_prompt", return_value=(None, None, None)), \
         patch("hierad.export.ad_video.get_video_duration", return_value=10.0), \
         patch("hierad.export.ad_video.synthesize_and_build_track", side_effect=RuntimeError("down")), \
         patch("hierad.export.ad_video.burn_ad_subtitles", side_effect=fake_burn):
        result = export_ad_video(stage3, video, work_dir=tmp_path, enable_tts=True, translate_zh=False)

    assert "tts_error" in result
    assert Path(result["video_path"]).exists()


def test_resolve_voice_kanghui():
    from hierad.export.voice import resolve_voice_profile
    p = resolve_voice_profile("康辉")
    assert p is not None
    assert "康辉" in p["name"]
    assert p["id"] == "20a29878-9e2c-48b1-bcee-c546c66526d6"


def test_burn_ad_subtitles_missing_video(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        burn_ad_subtitles(tmp_path / "missing.mp4", srt, tmp_path / "out.mp4")
