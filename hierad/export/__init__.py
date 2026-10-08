"""导出：AD → 中文 SRT / CosyVoice TTS → 硬字幕+旁白视频"""

from .ad_video import (
    ad_scripts_to_srt,
    burn_ad_subtitles,
    export_ad_video,
    load_stage3_scripts,
    prepare_zh_entries,
    write_srt_file,
)
from .translate import translate_texts_to_zh
from .tts import (
    build_ad_track,
    synthesize_ad_clips,
    synthesize_and_build_track,
    timecode_to_sec,
)
from .voice import resolve_voice_id_and_prompt, resolve_voice_profile

__all__ = [
    "ad_scripts_to_srt",
    "burn_ad_subtitles",
    "build_ad_track",
    "export_ad_video",
    "load_stage3_scripts",
    "prepare_zh_entries",
    "resolve_voice_id_and_prompt",
    "resolve_voice_profile",
    "synthesize_ad_clips",
    "synthesize_and_build_track",
    "timecode_to_sec",
    "translate_texts_to_zh",
    "write_srt_file",
]
