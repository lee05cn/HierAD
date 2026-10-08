"""
主流程编排 v2

预处理: 演员库 → 全片椭圆标注 → ASR → PySceneDetect 场景切分
Stage1: 场景 clip VLM × N → LLM 全局增强
Stage2: gap 定 AD → 混合视觉分析（默认复用 Stage1）
Stage3: LLM AD 精炼
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .config import (
    VLM_ENDPOINT,
    VLM_TYPE,
    VLM_URL,
    resolve_llm_settings,
)
from .progress import ProgressCallback, ProgressReporter
from .preprocess import (
    ASRResult,
    annotate_video,
    load_actor_db,
    load_ad_gap_segments,
    build_scene_clip_manifest,
    save_scene_manifest,
    load_scene_manifest,
    cut_by_segments,
    _sec_to_timecode,
)
from .describe import Stage1SceneUnderstanding, Stage2ADAnalysis, Stage3Refiner
from .export import export_ad_video


def load_canonical_mapping(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
        if not p.exists():
            return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mapping", data) if isinstance(data, dict) else {}


def load_role_mapping(path: Optional[Path]) -> Dict[str, str]:
    """从演员库加载 别名 -> 片内角色名 映射（Stage1 用，非演员本名）"""
    if not path or not path.exists():
        return {}
    p = Path(path)
    if p.is_file():
        p = p.parent
    if (p / "config.json").exists():
        from .actor_db.store import ActorDatabase
        return ActorDatabase(p).role_name_mapping()
    return {}


def ensure_annotated_video(
    video_path: str,
    work_dir: Path,
    actor_db_path: Optional[str],
    skip: bool,
    threshold: float = 0.2,
    frame_interval: int = 5,
    reporter: Optional[ProgressReporter] = None,
) -> str:
    """全片人脸椭圆标注；跳过则使用原片"""
    if skip or not actor_db_path or load_actor_db(actor_db_path) is None:
        return video_path
    out = work_dir / f"{Path(video_path).stem}_annotated.mp4"
    if out.exists():
        if reporter:
            reporter.stage("人脸标注", message="复用已有标注视频", done=True, data={"output": str(out)})
        return str(out)

    def on_frame(current: int, total: int) -> None:
        if not reporter:
            return
        step = max(total // 200, 1)
        if current == 1 or current >= total or current % step == 0:
            reporter.update(current, total)

    if reporter:
        reporter.stage("人脸标注", total=None, message="InsightFace 检测中")

    result = annotate_video(
        video_path=video_path,
        actor_db_path=actor_db_path,
        output_path=str(out),
        recognition_threshold=threshold,
        frame_interval=frame_interval,
        progress_callback=on_frame if reporter else None,
    )
    if result["status"] == "completed":
        if reporter:
            stats = result.get("stats", {})
            reporter.stage(
                "人脸标注",
                current=stats.get("total_frames"),
                total=stats.get("total_frames"),
                done=True,
                data={"output": result["output_path"], **stats},
            )
        return result["output_path"]
    if reporter:
        reporter.stage("人脸标注", message="失败，使用原片", done=True, data={"errors": result.get("errors", [])})
    return video_path


def _clip_descriptions_to_json(clips) -> list:
    return [c.to_dict() if hasattr(c, "to_dict") else {
        "clip_idx": c.clip_idx,
        "timecode": f"{c.timecode_start} --> {c.timecode_end}",
        "start_sec": c.start_sec,
        "end_sec": c.end_sec,
        "setting": c.setting,
        "characters": c.characters,
        "action": c.action,
        "dialogue_overlap": c.dialogue_overlap,
        "video_path": c.video_path,
    } for c in clips]


def _global_context_to_json(ctx) -> dict:
    return {
        "scenes": [
            {
                "scene_id": s.scene_id,
                "clip_range": [s.start_clip, s.end_clip],
                "event": s.event,
                "progression": s.progression,
                "characters": s.characters,
                "macro_space": getattr(s, "macro_space", ""),
                "micro_settings": list(getattr(s, "micro_settings", []) or []),
            }
            for s in ctx.scenes
        ],
        "characters": [
            {"name": c.name, "description": c.description, "role": c.role}
            for c in ctx.character_graph.characters.values()
        ],
        "events": [
            {
                "event_id": e.event_id,
                "clip_range": list(e.clip_range),
                "description": e.description,
            }
            for e in ctx.events
        ],
        "story_structure": {
            "setting": ctx.story_structure.setting,
            "core_conflict": ctx.story_structure.core_conflict,
            "phases": [
                {"name": p.name, "clip_range": list(p.clip_range), "summary": p.summary}
                for p in ctx.story_structure.phases
            ],
        },
    }


def _ad_segments_to_json(segs) -> list:
    return [
        {
            "ad_idx": s.ad_idx,
            "timecode": f"{s.timecode_start} --> {s.timecode_end}",
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "scene_idx": s.scene_idx,
            "scene_indices": s.scene_indices,
            "setting": s.setting,
            "characters": s.characters,
            "visual_description": s.visual_description,
            "source": s.source,
            "dialogue_before": s.dialogue_before,
            "dialogue_after": s.dialogue_after,
        }
        for s in segs
    ]


def run_pipeline(
    video_path: str,
    whisper_path: str,
    work_dir: Path,
    canonical_characters_path: Optional[str] = None,
    actor_db_path: Optional[str] = None,
    annotated_video_path: Optional[str] = None,
    skip_annotate: bool = False,
    annotation_threshold: float = 0.4,
    annotation_frame_interval: int = 5,
    vlm_url: Optional[str] = None,
    vlm_type: Optional[str] = None,
    vlm_endpoint: Optional[str] = None,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    min_gap_sec: float = 3.0,
    gap_margin_sec: float = 0.25,
    max_gap_duration_sec: Optional[float] = None,
    max_ad_segments: Optional[int] = None,
    max_gap_chunk_sec: Optional[float] = 12.0,
    repair_sparse_asr: bool = True,
    silence_refine: bool = True,
    silence_noise_db: float = -25.0,
    max_scene_clips: Optional[int] = None,
    scene_detect_threshold: float = 27.0,
    min_scene_sec: float = 5.0,
    max_scene_sec: float = 90.0,
    show_progress: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    on_stage: Optional[Callable[[str], None]] = None,
    reporter: Optional[ProgressReporter] = None,
    burn_ad: bool = True,
    enable_tts: bool = True,
    translate_zh: bool = True,
    tts_url: Optional[str] = None,
    tts_voice_id: Optional[str] = None,
    tts_voice_name: Optional[str] = None,
    tts_spk_id: Optional[str] = None,
    tts_speed: Optional[float] = None,
    pipeline_version: str = "v3",
) -> Tuple[List[Tuple[str, str]], List[str]]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    scene_clips_dir = work_dir / "scene_clips"
    ad_clips_dir = work_dir / "ad_clips"
    scene_clips_dir.mkdir(exist_ok=True)
    ad_clips_dir.mkdir(exist_ok=True)

    owns_reporter = reporter is None
    if reporter is None and show_progress:
        reporter = ProgressReporter(show=True, callback=on_progress)
    elif reporter is not None and on_progress and reporter.callback is None:
        reporter.callback = on_progress

    def _stage(label: str, *, done: bool = False, message: str = "", **data) -> None:
        if reporter:
            reporter.stage(label, message=message, done=done, data=data)
        if on_stage:
            on_stage(label)

    asr_result = ASRResult.from_whisper_json(whisper_path)
    _stage("加载 ASR", segment_count=len(asr_result.segments), done=True)

    source_video = annotated_video_path or ensure_annotated_video(
        video_path,
        work_dir,
        actor_db_path,
        skip=skip_annotate,
        threshold=annotation_threshold,
        frame_interval=annotation_frame_interval,
        reporter=reporter,
    )
    if skip_annotate:
        _stage("人脸标注", message="已跳过", done=True)

    scene_manifest_path = work_dir / "preprocess_scene_clips.json"
    if scene_manifest_path.exists() and max_scene_clips is None:
        scene_manifest = load_scene_manifest(str(scene_manifest_path))
        _stage(
            "场景切分",
            message="复用缓存",
            done=True,
            clip_count=scene_manifest.get("clip_count", 0),
        )
    else:
        scene_manifest = build_scene_clip_manifest(
            source_video,
            whisper_path,
            str(scene_clips_dir),
            threshold=scene_detect_threshold,
            min_scene_sec=min_scene_sec,
            max_scene_sec=max_scene_sec,
            max_clips=max_scene_clips,
            reporter=reporter,
        )
        save_scene_manifest(scene_manifest, scene_manifest_path)
        _stage("场景切分", done=True, clip_count=scene_manifest.get("clip_count", 0))

    role_mapping = load_role_mapping(
        Path(actor_db_path or canonical_characters_path) if (actor_db_path or canonical_characters_path) else None
    )
    if not role_mapping and canonical_characters_path:
        role_mapping = load_canonical_mapping(Path(canonical_characters_path))

    llm_url_r, llm_model_r, llm_key_r = resolve_llm_settings(
        url=llm_url, model=llm_model, api_key=llm_api_key
    )
    stage1_ckpt = work_dir / "stage1_clip_descriptions.json"
    scene_contexts = None
    if pipeline_version == "v3":
        from .describe import (
            Stage1ShotUnderstanding,
            infer_macro_spaces,
            aggregate_shots_to_scenes,
            infer_scene_themes,
            Stage1SceneEnhancer,
        )
        shot_ckpt = work_dir / "stage1_shot_descriptions.json"
        stage1_v3 = Stage1ShotUnderstanding(
            vlm_url=vlm_url or VLM_URL,
            vlm_type=vlm_type or VLM_TYPE,
            vlm_endpoint=vlm_endpoint or VLM_ENDPOINT,
            llm_url=llm_url_r,
            llm_model=llm_model_r,
            llm_api_key=llm_key_r,
            canonical_characters=role_mapping,
        )
        clip_descs = stage1_v3.describe_clips(
            scene_manifest["clips"],
            reporter=reporter,
            checkpoint_path=str(shot_ckpt),
        )
        _stage("Stage1", done=True, clip_count=len(clip_descs))
        with open(shot_ckpt, "w", encoding="utf-8") as f:
            json.dump(_clip_descriptions_to_json(clip_descs), f, ensure_ascii=False, indent=2)

        macro_spaces = infer_macro_spaces(
            clip_descs,
            llm_url=llm_url_r,
            llm_model=llm_model_r,
            llm_api_key=llm_key_r,
            reporter=reporter,
        )
        for shot, macro in zip(clip_descs, macro_spaces):
            shot.macro_space = macro
        _stage("Stage1.5", done=True, shot_count=len(clip_descs))

        scenes = aggregate_shots_to_scenes(clip_descs)
        infer_scene_themes(
            scenes,
            llm_url=llm_url_r,
            llm_model=llm_model_r,
            llm_api_key=llm_key_r,
            reporter=reporter,
        )
        _stage("Scene 聚合", done=True, scene_count=len(scenes))
        with open(work_dir / "stage1_scenes.json", "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in scenes], f, ensure_ascii=False, indent=2)

        enhancer = Stage1SceneEnhancer(
            llm_url=llm_url_r,
            llm_model=llm_model_r,
            llm_api_key=llm_key_r,
            canonical_mapping=role_mapping,
        )
        global_ctx = enhancer.enhance(clip_descs, scenes, reporter=reporter)
        with open(work_dir / "stage1_global_context.json", "w", encoding="utf-8") as f:
            json.dump(_global_context_to_json(global_ctx), f, ensure_ascii=False, indent=2)

        scene_contexts = [s.to_dict() for s in scenes]
    else:
        stage1 = Stage1SceneUnderstanding(
            vlm_url=vlm_url or VLM_URL,
            vlm_type=vlm_type or VLM_TYPE,
            vlm_endpoint=vlm_endpoint or VLM_ENDPOINT,
            llm_url=llm_url_r,
            llm_model=llm_model_r,
            llm_api_key=llm_key_r,
            canonical_characters=role_mapping,
        )
        clip_descs, global_ctx = stage1.process(
            scene_manifest["clips"],
            reporter=reporter,
            checkpoint_path=str(stage1_ckpt),
        )
        _stage("Stage1", done=True, clip_count=len(clip_descs))

        with open(stage1_ckpt, "w", encoding="utf-8") as f:
            json.dump(_clip_descriptions_to_json(clip_descs), f, ensure_ascii=False, indent=2)
        with open(work_dir / "stage1_global_context.json", "w", encoding="utf-8") as f:
            json.dump(_global_context_to_json(global_ctx), f, ensure_ascii=False, indent=2)

    gaps, _, video_duration = load_ad_gap_segments(
        whisper_path,
        video_path,
        min_gap_sec=min_gap_sec,
        margin_sec=gap_margin_sec,
        repair_sparse=repair_sparse_asr,
        silence_refine=silence_refine,
        silence_noise_db=silence_noise_db,
        max_gap_chunk_sec=max_gap_chunk_sec,
    )
    if max_gap_duration_sec is not None:
        gaps = [(s, e) for s, e in gaps if (e - s) <= max_gap_duration_sec]
    if max_ad_segments is not None:
        gaps = gaps[:max_ad_segments]
    if not gaps:
        raise ValueError(f"未找到满足 min_gap={min_gap_sec}s 的 AD 间隙")

    ad_cut = cut_by_segments(source_video, gaps, str(ad_clips_dir), reporter=reporter)
    ad_manifest = []
    for i, ((start, end), (tc_s, tc_e, path)) in enumerate(zip(gaps, ad_cut)):
        ad_manifest.append({
            "ad_idx": i,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "timecode": f"{tc_s} --> {tc_e}",
            "video_path": path,
        })
    _stage("AD 切分", done=True, ad_count=len(ad_manifest))

    clip_ranges = [(c.start_sec, c.end_sec) for c in clip_descs]

    stage2 = Stage2ADAnalysis(
        vlm_url=vlm_url or VLM_URL,
        vlm_type=vlm_type or VLM_TYPE,
        vlm_endpoint=vlm_endpoint or VLM_ENDPOINT,
        canonical_characters=role_mapping,
    )
    ad_segments = stage2.analyze_ad_segments(
        ad_manifest,
        clip_descs,
        clip_ranges,
        global_ctx,
        asr_result.segments,
        reporter=reporter,
        scene_contexts=scene_contexts,
    )
    _stage("Stage2", done=True, ad_count=len(ad_segments))

    with open(work_dir / "stage2_ad_segments.json", "w", encoding="utf-8") as f:
        json.dump(_ad_segments_to_json(ad_segments), f, ensure_ascii=False, indent=2)

    stage3 = Stage3Refiner(
        llm_url=llm_url_r,
        llm_model=llm_model_r,
        llm_api_key=llm_key_r,
        canonical_characters=role_mapping,
    )
    ad_scripts = stage3.process(ad_segments, global_ctx, reporter=reporter, scene_contexts=scene_contexts)
    _stage("Stage3", done=True, ad_count=len(ad_scripts))

    timecodes = [(s.timecode_start, s.timecode_end) for s in ad_segments]
    stage3_path = work_dir / "stage3_ad_scripts.json"
    with open(stage3_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "ad_idx": s.ad_idx,
                "start_time": s.timecode_start,
                "end_time": s.timecode_end,
                "scene_idx": s.scene_idx,
                "source": s.source,
                "dialogue_before": s.dialogue_before,
                "dialogue_after": s.dialogue_after,
                "visual_description": s.visual_description,
                "ad_script": ad,
            }
            for s, ad in zip(ad_segments, ad_scripts)
        ], f, ensure_ascii=False, indent=2)

    ad_export: Dict[str, str] = {}
    ad_error: str = ""
    if burn_ad and ad_scripts:
        msg = "中文 SRT" + (" + CosyVoice TTS" if enable_tts else "") + "，压制到原始完整输入视频"
        _stage("AD 压制", message=msg)
        try:
            ad_export = export_ad_video(
                stage3_path=stage3_path,
                video_path=video_path,
                work_dir=work_dir,
                enable_tts=enable_tts,
                translate_zh=translate_zh,
                actor_db_path=actor_db_path,
                tts_url=tts_url,
                voice_id=tts_voice_id,
                voice_name=tts_voice_name,
                spk_id=tts_spk_id,
                speed=tts_speed,
                llm_url=llm_url_r,
                llm_model=llm_model_r,
                llm_api_key=llm_key_r,
            )
            _stage("AD 压制", done=True, **ad_export)
        except Exception as e:
            ad_error = f"{type(e).__name__}: {e}"
            _stage("AD 压制", message=f"失败: {ad_error}", done=True)
            # 这里不能只 print：webapp 用单工作线程跑、stdout 常被丢弃，
            # 失败原因必须落到 config.json 和日志文件里才查得到。
            print(f"警告: AD 字幕/TTS 压制失败: {ad_error}")
            try:
                with open(work_dir / "ad_export_error.log", "w", encoding="utf-8") as f:
                    f.write(f"{ad_error}\n\n{traceback.format_exc()}")
            except OSError:
                pass

    config = {
        "pipeline_version": pipeline_version,
        "video_path": video_path,
        "annotated_video_path": source_video,
        "whisper_path": whisper_path,
        "scene_clip_count": len(clip_descs),
        "ad_segment_count": len(ad_segments),
        "video_duration_sec": video_duration,
        "min_gap_sec": min_gap_sec,
        "gap_margin_sec": gap_margin_sec,
        "burn_ad": burn_ad,
        "enable_tts": enable_tts,
        "translate_zh": translate_zh,
    }
    if ad_error:
        config["ad_error"] = ad_error
    if ad_export:
        config["ad_srt_path"] = ad_export.get("srt_path")
        config["ad_video_path"] = ad_export.get("video_path")
        if ad_export.get("ad_audio_path"):
            config["ad_audio_path"] = ad_export["ad_audio_path"]
        if ad_export.get("voice_id"):
            config["tts_voice_id"] = ad_export["voice_id"]
            config["tts_voice_name"] = ad_export.get("voice_name")
    with open(work_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    _stage("保存结果", done=True, **config)
    if owns_reporter and reporter:
        reporter.close()
    return timecodes, ad_scripts
