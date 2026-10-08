"""AD 间隙切分测试"""

import json
import tempfile
from pathlib import Path

import pytest

from hierad.preprocess.gap_segments import (
    merge_speech_intervals,
    gaps_from_speech_segments,
    dialogue_context_for_gap,
    repair_sparse_asr_segments,
    subtract_intervals,
    split_long_gaps,
    filter_gaps_against_blocked,
)


def test_merge_speech_intervals():
    iv = [(0.0, 2.0), (1.5, 4.0), (10.0, 12.0)]
    merged = merge_speech_intervals(iv)
    assert merged == [(0.0, 4.0), (10.0, 12.0)]


def test_gaps_from_speech_segments_signs_example():
    speech = [(180.0, 182.0), (211.0, 213.0), (218.0, 220.0)]
    gaps = gaps_from_speech_segments(speech, video_duration=300.0, min_gap_sec=2.0, margin_sec=0.25)
    assert len(gaps) >= 2
    mid = next(g for g in gaps if g[0] == pytest.approx(182.25))
    assert mid[1] == pytest.approx(210.75)


def test_gaps_no_speech_full_video():
    gaps = gaps_from_speech_segments([], video_duration=100.0, min_gap_sec=2.0)
    assert gaps == [(0.0, 100.0)]


def test_dialogue_context_for_gap():
    speech = [
        {"start": 180.0, "end": 182.0, "text": "Where are they?", "speaker": "A"},
        {"start": 211.0, "end": 213.0, "text": "No!", "speaker": "B"},
    ]
    before, after = dialogue_context_for_gap(182.5, 210.0, speech)
    assert "Where are they?" in before
    assert "No!" in after
    assert "A:" in before


def test_repair_sparse_asr_shrinks_and_blocks_tail():
    segs = [
        {"start": 116.22, "end": 229.58, "text": "Tony Stark. Visionary. Genius. American patriot."},
        {"start": 230.0, "end": 232.0, "text": "Hello there everyone."},
    ]
    repaired, blocked = repair_sparse_asr_segments(segs)
    assert repaired[0][1] - repaired[0][0] <= 8.0
    assert blocked and blocked[0][0] >= repaired[0][1] - 1e-6
    assert blocked[0][1] == pytest.approx(229.58)


def test_filter_gaps_blocks_montage_without_silence():
    gaps = [(120.0, 229.0), (0.0, 12.0)]
    blocked = [(120.0, 229.58)]
    kept = filter_gaps_against_blocked(gaps, blocked, allowed_in_blocked=[], min_gap_sec=3.0)
    assert kept == [(0.0, 12.0)]
    # 静音确认后可保留交集
    kept2 = filter_gaps_against_blocked(
        gaps, blocked, allowed_in_blocked=[(180.0, 188.0)], min_gap_sec=3.0
    )
    assert any(abs(s - 180.0) < 1e-6 and abs(e - 188.0) < 1e-6 for s, e in kept2)


def test_subtract_intervals_punches_holes():
    base = [(0.0, 100.0)]
    holes = [(10.0, 20.0), (50.0, 55.0)]
    out = subtract_intervals(base, holes)
    assert out == [(0.0, 10.0), (20.0, 50.0), (55.0, 100.0)]


def test_split_long_gaps():
    gaps = [(0.0, 30.0), (40.0, 45.0)]
    chunks = split_long_gaps(gaps, max_chunk_sec=12.0, min_gap_sec=3.0)
    assert chunks[0] == (0.0, 12.0)
    assert chunks[-1] == (40.0, 45.0)


def test_load_ad_gap_segments_iron_man_refined():
    from hierad.preprocess.gap_segments import load_ad_gap_segments

    asr = Path("/path/to/HierAD/work_dir/e2e_iron_man_input/input_asr.json")
    vid = "/path/to/data/test/Iron Man/input.mp4"
    if not asr.exists() or not Path(vid).exists():
        pytest.skip("Iron Man 测试数据不存在")

    old, _, _ = load_ad_gap_segments(
        str(asr), vid, min_gap_sec=2.0, repair_sparse=False, silence_refine=False, max_gap_chunk_sec=None
    )
    new, _, _ = load_ad_gap_segments(
        str(asr), vid, min_gap_sec=3.0, repair_sparse=True, silence_refine=True, max_gap_chunk_sec=12.0
    )
    assert len(old) == 5
    # 新逻辑不应再把整段蒙太奇 VO 区切成大量 AD
    assert 5 <= len(new) <= 25


def test_stage3_no_name_when_characters_none():
    from hierad.describe.stage3 import Stage3Refiner, _fallback_from_visual
    from hierad.describe.models import ADSegment

    fb = _fallback_from_visual(
        "**Characters:** None\n**Action:** Two military vehicles drive down a dusty road.",
        max_words=12,
        no_names=True,
    )
    assert "Tony" not in fb

    ref = Stage3Refiner.__new__(Stage3Refiner)
    # 不调 LLM：直接测硬约束路径用 fallback
    seg = ADSegment(
        ad_idx=0,
        start_sec=0.0,
        end_sec=12.0,
        timecode_start="00:00:00.000",
        timecode_end="00:00:12.000",
        scene_idx=0,
        scene_indices=[0],
        setting="Desert",
        characters="None",
        visual_description="**Setting:** Desert\n**Characters:** None\n**Action:** Dusty road with two trucks.",
        source="vlm",
    )
    # monkey: force LLM path failure by not setting client — use refine with mock
    from unittest.mock import MagicMock
    ref.client = MagicMock()
    ref.llm_model = "x"
    ref.before_window = 0
    ref.canonical_characters = {}
    # LLM returns invented name
    ref.client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Tony Stark looks tense in the desert."))
    ]
    out = Stage3Refiner.refine_segment(ref, seg)
    assert "Tony" not in out


def test_stage3_keeps_name_from_visual_when_field_empty():
    """characters 字段为空时，应从 visual 的 Characters 行回填，避免误 scrub。"""
    from unittest.mock import MagicMock

    from hierad.describe.models import ADSegment
    from hierad.describe.stage3 import Stage3Refiner

    ref = Stage3Refiner.__new__(Stage3Refiner)
    ref.client = MagicMock()
    ref.llm_model = "x"
    ref.before_window = 0
    ref.canonical_characters = {}
    ref.client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Tony Stark lies injured on the desert ground."))
    ]
    seg = ADSegment(
        ad_idx=0,
        start_sec=0.0,
        end_sec=8.0,
        timecode_start="00:00:00.000",
        timecode_end="00:00:08.000",
        scene_idx=0,
        scene_indices=[0],
        setting="Desert",
        characters="",
        visual_description=(
            "**Setting:** Desert\n"
            "**Characters:** TONY STARK\n"
            "**Action:** Tony Stark lies injured on the ground."
        ),
        source="vlm",
    )
    out = Stage3Refiner.refine_segment(ref, seg)
    assert "Tony" in out


def test_load_ad_gap_segments_from_json():
    from hierad.preprocess.gap_segments import load_ad_gap_segments

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hi"},
                {"start": 10.0, "end": 12.0, "text": "Bye"},
            ]
        }, f)
        asr_path = f.name

    clip = "/path/to/HierAD-old/data/1005_Signs/segments/0001.mp4"
    if not Path(clip).exists():
        pytest.skip("测试视频不存在")

    gaps, asr, dur = load_ad_gap_segments(
        asr_path, clip, min_gap_sec=1.0, silence_refine=False, max_gap_chunk_sec=None
    )
    Path(asr_path).unlink(missing_ok=True)
    assert dur > 0
    assert len(gaps) >= 1
