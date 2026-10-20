"""
Stage3 AD 脚本 → 中文 SRT + CosyVoice（康辉等音色）→ 压制到原始完整输入视频
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from hierad.config import (
    TTS_SPK_ID,
    TTS_VOICE_ID,
    TTS_VOICE_NAME,
    TRANSLATE_ZH,
    get_tts_provider,
)
from hierad.preprocess.gap_segments import get_video_duration

from .translate import translate_texts_to_zh
from .tts import synthesize_and_build_track
from .voice import resolve_voice_id_and_prompt
from .name_glossary import load_or_build_glossary


_TC_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})[.,](\d{1,3})$"
)


def _normalize_srt_timecode(tc: str) -> str:
    """HH:MM:SS.mmm / HH:MM:SS,mmm → SRT 的 HH:MM:SS,mmm"""
    tc = tc.strip().replace(" ", "")
    m = _TC_RE.match(tc)
    if not m:
        raise ValueError(f"无法解析时间码: {tc!r}")
    h = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = m.group(4).ljust(3, "0")[:3]
    return f"{h:02d}:{mm:02d}:{ss:02d},{frac}"


def ad_scripts_to_srt(entries: Sequence[Dict[str, Any]]) -> str:
    """将 stage3 条目列表转为 SRT 文本（正文为 ad_script）。"""
    blocks: List[str] = []
    idx = 0
    for entry in entries:
        text = (entry.get("ad_script") or entry.get("text") or "").strip()
        if not text:
            continue
        start = entry.get("start_time") or entry.get("start")
        end = entry.get("end_time") or entry.get("end")
        if start is None or end is None:
            raise ValueError(f"缺少时间码: {entry!r}")
        idx += 1
        body = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        blocks.append(
            f"{idx}\n{_normalize_srt_timecode(str(start))} --> {_normalize_srt_timecode(str(end))}\n{body}\n"
        )
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def write_srt_file(entries: Sequence[Dict[str, Any]], srt_path: Union[str, Path]) -> Path:
    path = Path(srt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ad_scripts_to_srt(entries), encoding="utf-8")
    return path


def load_stage3_scripts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"stage3 文件应为 JSON 数组: {path}")
    return data


def prepare_zh_entries(
    entries: Sequence[Dict[str, Any]],
    *,
    translate_zh: bool = True,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    glossary: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    将 ad_script 统一为中文：保留 ad_script_en，主字段 ad_script 为中文。
    """
    usable_idx = [
        i for i, e in enumerate(entries)
        if (e.get("ad_script") or e.get("text") or "").strip()
    ]
    en_texts = [
        " ".join(
            line.strip()
            for line in (entries[i].get("ad_script") or entries[i].get("text") or "").splitlines()
            if line.strip()
        )
        for i in usable_idx
    ]
    if translate_zh and en_texts:
        zh_texts = translate_texts_to_zh(
            en_texts,
            llm_url=llm_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            glossary=glossary,
        )
    else:
        zh_texts = en_texts

    zh_map = {i: zh for i, zh in zip(usable_idx, zh_texts)}
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(entries):
        item = dict(e)
        en = (e.get("ad_script") or e.get("text") or "").strip()
        if i in zh_map:
            item["ad_script_en"] = en
            item["ad_script"] = zh_map[i]
            item["ad_script_zh"] = zh_map[i]
        out.append(item)
    return out


def _escape_subtitles_path(path: Path) -> str:
    # ffmpeg 的 filtergraph 要解两层转义：图解析器先把 \\: 还原成 \:，
    # 参数解析器再把 \: 还原成 :。只转一层（\:）在 ffmpeg 8.x 上会把盘符吃掉，
    # 路径后半段被误当作 original_size 选项值。外层单引号同时兜住路径里的空格。
    s = str(path.resolve()).replace("\\", "/")
    s = s.replace("'", "\\'")
    s = s.replace(":", "\\:")
    return f"'{s}'"


def burn_ad_subtitles(
    video_path: Union[str, Path],
    srt_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    ad_wav_path: Optional[Union[str, Path]] = None,
    ffmpeg: str = "ffmpeg",
    font_size: int = 22,
    margin_v: int = 40,
    ad_volume: float = 1.6,
    orig_volume: float = 1.0,
) -> Path:
    """将 AD SRT 硬字幕压制进原始完整视频；可选混入 AD TTS 音轨。"""
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"源视频不存在: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT 不存在: {srt_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = (
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,"
        f"Alignment=2,MarginV={margin_v}"
    )
    vf = f"subtitles={_escape_subtitles_path(srt_path)}:force_style='{style}'"

    if ad_wav_path:
        ad_wav_path = Path(ad_wav_path)
        if not ad_wav_path.exists():
            raise FileNotFoundError(f"AD 音轨不存在: {ad_wav_path}")
        fc = (
            f"[0:a]volume={orig_volume}[a0];"
            f"[1:a]volume={ad_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(ad_wav_path),
            "-vf", vf,
            "-filter_complex", fc,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "copy",
            str(output_path),
        ]

    try:
        # 刻意用 bytes 捕获：本机是 GBK locale，text=True 会在解码 ffmpeg 输出时
        # 抛 UnicodeDecodeError（byte 0x80 等），把真正的 ffmpeg 报错整个吞掉，
        # 最后只剩一句误导性的"需 libass / 需 amix"。
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH") from e

    if proc.returncode != 0:
        raw = proc.stderr or proc.stdout or b""
        err = raw.decode("utf-8", "replace").strip()
        tail = "\n".join(err.splitlines()[-20:]) if err else "(ffmpeg 没有任何输出)"
        raise RuntimeError(
            f"AD 视频压制失败：ffmpeg 退出码 {proc.returncode}。"
            f"（字幕烧制需 libass，混音需 amix）\n{tail}"
        )
    return output_path


def export_ad_video(
    stage3_path: Union[str, Path],
    video_path: Union[str, Path],
    work_dir: Optional[Union[str, Path]] = None,
    *,
    srt_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    font_size: int = 22,
    enable_tts: bool = True,
    translate_zh: Optional[bool] = None,
    actor_db_path: Optional[str] = None,
    tts_url: Optional[str] = None,
    voice_id: Optional[str] = None,
    voice_name: Optional[str] = None,
    prompt_text: Optional[str] = None,
    spk_id: Optional[str] = None,
    speed: Optional[float] = None,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Stage3 → 中文 SRT（默认）+ CosyVoice TTS（默认康辉音色）→ 压制到原始完整输入视频。
    """
    stage3_path = Path(stage3_path)
    video_path = Path(video_path)
    work_dir = Path(work_dir) if work_dir else stage3_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    do_zh = TRANSLATE_ZH if translate_zh is None else translate_zh
    entries_raw = load_stage3_scripts(stage3_path)
    if not entries_raw or not any(
        (e.get("ad_script") or e.get("text") or "").strip() for e in entries_raw
    ):
        raise ValueError("stage3 中没有可用的 ad_script，跳过压制")

    # 从演员库构建中文专名表
    glossary: Dict[str, str] = {}
    if do_zh and actor_db_path:
        try:
            glossary = load_or_build_glossary(
                actor_db_path, work_dir,
                llm_url=llm_url, llm_model=llm_model, llm_api_key=llm_api_key,
            )
            if glossary:
                print(f"中文专名表: {len(glossary)} 个角色名")
        except Exception as e:
            print(f"警告: 专名表构建失败: {e}")

    # 英文 SRT 备份
    write_srt_file(entries_raw, work_dir / "stage3_ad_scripts_en.srt")

    entries = prepare_zh_entries(
        entries_raw,
        translate_zh=do_zh,
        llm_url=llm_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        glossary=glossary,
    )
    zh_json = work_dir / "stage3_ad_scripts_zh.json"
    with open(zh_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    srt_out = Path(srt_path) if srt_path else work_dir / "stage3_ad_scripts.srt"
    write_srt_file(entries, srt_out)

    video_out = (
        Path(output_path)
        if output_path
        else work_dir / f"{video_path.stem}_ad.mp4"
    )

    # 解析音色：本机 CosyVoice 走 VD-agent 库；百炼用系统音色，不查本地 DB
    resolved_voice = voice_id or TTS_VOICE_ID or None
    resolved_prompt = prompt_text
    voice_display = ""
    lookup_key = voice_name or (None if voice_id else TTS_VOICE_NAME)
    if get_tts_provider() != "dashscope":
        if not resolved_voice and lookup_key:
            vid, ptxt, vname = resolve_voice_id_and_prompt(lookup_key)
            resolved_voice = vid
            if resolved_prompt is None:
                resolved_prompt = ptxt
            voice_display = vname or lookup_key
        elif resolved_voice and not voice_display:
            # 仍尝试取 prompt
            if resolved_prompt is None:
                _vid, ptxt, vname = resolve_voice_id_and_prompt(resolved_voice)
                if ptxt:
                    resolved_prompt = ptxt
                voice_display = vname or resolved_voice

    # 有克隆音色时不用 SFT spk_id
    resolved_spk = None if resolved_voice else (spk_id if spk_id is not None else TTS_SPK_ID or None)

    # 百炼云端：voice_name 直接就是百炼音色名（如 longwan_v2），不查本地 VD-agent 库；
    # 本地 CosyVoice 才用上面的 resolved_voice / prompt。dash_voice 为空时走 config 默认音色。
    dash_voice = voice_name if get_tts_provider() == "dashscope" else None

    ad_wav: Optional[Path] = None
    tts_note = ""
    if enable_tts:
        try:
            duration = get_video_duration(str(video_path))
            tts_dir = work_dir / "tts"
            if tts_dir.exists():
                for old in tts_dir.glob("*.wav"):
                    try:
                        old.unlink()
                    except OSError:
                        pass
            if resolved_voice:
                print(f"TTS 音色: {voice_display or resolved_voice} ({resolved_voice})")
            ad_wav, _clips = synthesize_and_build_track(
                entries,
                work_dir,
                track_len_sec=duration,
                tts_url=tts_url,
                voice=dash_voice,
                voice_id=resolved_voice,
                prompt_text=resolved_prompt,
                spk_id=resolved_spk,
                speed=speed,
            )
        except Exception as e:
            tts_note = str(e)
            print(f"警告: AD TTS 失败，仅压制字幕: {e}")
            ad_wav = None

    burn_ad_subtitles(
        video_path,
        srt_out,
        video_out,
        ad_wav_path=ad_wav,
        font_size=font_size,
    )
    result = {
        "srt_path": str(srt_out),
        "srt_en_path": str(work_dir / "stage3_ad_scripts_en.srt"),
        "zh_json_path": str(zh_json),
        "video_path": str(video_out),
        "ad_count": str(sum(1 for e in entries if (e.get("ad_script") or "").strip())),
        "translate_zh": "1" if do_zh else "0",
    }
    if resolved_voice:
        result["voice_id"] = resolved_voice
        if voice_display:
            result["voice_name"] = voice_display
    if dash_voice:
        result["voice_id"] = dash_voice
        result["voice_name"] = dash_voice
    if ad_wav is not None:
        result["ad_audio_path"] = str(ad_wav)
    if tts_note and ad_wav is None:
        result["tts_error"] = tts_note
    return result
