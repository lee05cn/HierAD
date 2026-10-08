"""
预处理模块

推荐顺序:
1. 演员库 build-actor-db
2. ASR transcribe
3. 全片人脸椭圆标注 annotate
4. PySceneDetect 场景切分 scene_segments
5. Pipeline Stage1–3
"""

from .asr import (
    ASRResult,
    extract_audio_from_video,
    transcribe_with_asr_service,
    transcribe_with_whisper,
    transcribe_video,
    segments_from_asr_service_payload,
)
from .cutter import cut_by_segments, _sec_to_timecode
from .gap_segments import (
    gaps_from_speech_segments,
    merge_speech_intervals,
    dialogue_context_for_gap,
    load_ad_gap_segments,
    get_video_duration,
)
from .asr_overlap import dialogue_in_interval, find_clip_index_for_time, clips_for_interval
from .scene_segments import (
    detect_scene_ranges,
    build_scene_clip_manifest,
    save_scene_manifest,
    load_scene_manifest,
    merge_short_scenes,
    split_long_scenes,
)
from .face_annotator import annotate_video
from .face_hints import (
    load_actor_db,
    resolve_actor_db_dir,
)

__all__ = [
    "ASRResult",
    "extract_audio_from_video",
    "transcribe_with_asr_service",
    "transcribe_with_whisper",
    "transcribe_video",
    "segments_from_asr_service_payload",
    "cut_by_segments",
    "_sec_to_timecode",
    "gaps_from_speech_segments",
    "merge_speech_intervals",
    "dialogue_context_for_gap",
    "load_ad_gap_segments",
    "get_video_duration",
    "dialogue_in_interval",
    "find_clip_index_for_time",
    "clips_for_interval",
    "detect_scene_ranges",
    "build_scene_clip_manifest",
    "save_scene_manifest",
    "load_scene_manifest",
    "merge_short_scenes",
    "split_long_scenes",
    "annotate_video",
    "load_actor_db",
    "resolve_actor_db_dir",
]
