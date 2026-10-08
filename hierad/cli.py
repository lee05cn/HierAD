"""
HierAD 命令行入口 v2
"""

import argparse
import os
from pathlib import Path

from . import __version__
from .pipeline import run_pipeline
from .preprocess import transcribe_video, annotate_video
from .actor_db import build_from_tmdb, build_face_vector_db, rebuild_face_vector_db
from .actor_db.tmdb import TMDBClient, check_tmdb_connection
from .config import ACTOR_DATABASES_DIR
from .export import export_ad_video


def main():
    parser = argparse.ArgumentParser("hierad", description="HierAD 层次化无障碍视频描述系统 v2")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", help="命令")

    trans_p = sub.add_parser("transcribe", help="视频转字幕（Whisper ASR）")
    trans_p.add_argument("--video", required=True)
    trans_p.add_argument("--output")
    trans_p.add_argument("--model", default="base")
    trans_p.add_argument("--language")
    trans_p.add_argument("--asr-backend", choices=["auto", "service", "local", "dashscope"], default="auto")
    trans_p.add_argument("--asr-url")

    run_p = sub.add_parser("run", help="运行 v2 描述流水线")
    run_p.add_argument("--video", required=True)
    run_p.add_argument("--whisper")
    run_p.add_argument("--work-dir", default="./work_dir")
    run_p.add_argument("--asr-backend", choices=["auto", "service", "local", "dashscope"], default="auto")
    run_p.add_argument("--asr-url")
    run_p.add_argument("--min-gap", type=float, default=3.0, help="最短 AD 间隙（秒），默认 3")

    run_p.add_argument("--gap-margin", type=float, default=0.25)
    run_p.add_argument("--max-gap-duration", type=float, help="丢弃长于此秒数的 gap（少用；优先用 --max-gap-chunk）")
    run_p.add_argument("--max-gap-chunk", type=float, default=12.0, help="过长 gap 切成此长度的多段 AD（默认 12s）")
    run_p.add_argument("--max-ad-segments", type=int)
    run_p.add_argument("--v3", dest="pipeline_version", action="store_const", const="v3", default="v3", help="使用 v3 Clip→Shot→Scene 三层流水线（默认）")
    run_p.add_argument("--v2", dest="pipeline_version", action="store_const", const="v2", help="回退到 v2 单层场景流水线")
    run_p.add_argument("--no-repair-sparse-asr", action="store_true", help="关闭稀疏 ASR 段收缩")
    run_p.add_argument("--no-silence-refine", action="store_true", help="关闭 ffmpeg 静音打孔")
    run_p.add_argument("--silence-noise-db", type=float, default=-25.0, help="silencedetect 噪声阈值 dB")
    run_p.add_argument("--max-scene-clips", type=int)
    run_p.add_argument("--min-scene-sec", type=float, default=5.0)
    run_p.add_argument("--max-scene-sec", type=float, default=90.0)
    run_p.add_argument("--scene-threshold", type=float, default=27.0)
    run_p.add_argument("--canonical-characters")
    run_p.add_argument("--actor-db")
    run_p.add_argument("--skip-annotate", action="store_true")
    run_p.add_argument("--annotation-threshold", type=float, default=0.4)
    run_p.add_argument("--annotation-frame-interval", type=int, default=5)
    run_p.add_argument("--face-gpu", type=int, default=2, help="人脸标注 InsightFace GPU 编号（默认 2）")
    run_p.add_argument("--vlm-url")
    run_p.add_argument("--vlm-type")
    run_p.add_argument("--vlm-endpoint")
    run_p.add_argument("--vlm-provider", choices=["local", "dashscope"], help="默认 local（本机 VLM 服务）")
    run_p.add_argument("--llm-url")
    run_p.add_argument("--llm-model")
    run_p.add_argument("--llm-api-key")
    run_p.add_argument("--llm-provider", choices=["local", "dashscope"], help="默认 local（本机 LLM 服务）")
    run_p.add_argument("--tts-provider", choices=["local", "dashscope"], help="默认 local（本机 CosyVoice）")
    run_p.add_argument(
        "--burn-ad",
        dest="burn_ad",
        action="store_true",
        default=True,
        help="Stage3 后将 AD 硬字幕压制到源视频（默认开启）",
    )
    run_p.add_argument(
        "--no-burn-ad",
        dest="burn_ad",
        action="store_false",
        help="跳过 AD 字幕压制",
    )
    run_p.add_argument(
        "--tts",
        dest="enable_tts",
        action="store_true",
        default=True,
        help="压制时用 CosyVoice 生成 AD 旁白并混音（默认开启）",
    )
    run_p.add_argument(
        "--no-tts",
        dest="enable_tts",
        action="store_false",
        help="仅硬字幕，不生成 AD TTS",
    )
    run_p.add_argument("--tts-url", help="CosyVoice 服务地址（默认 http://127.0.0.1:8003）")
    run_p.add_argument("--tts-voice-id", help="音色库 voice profile UUID")
    run_p.add_argument("--tts-voice-name", help="音色库名称（默认康辉）")
    run_p.add_argument("--tts-spk", help="无音色克隆时的 CosyVoice SFT 说话人")
    run_p.add_argument("--tts-speed", type=float, help="TTS 语速，默认 1.2（>1 更快）")
    run_p.add_argument(
        "--translate-zh",
        dest="translate_zh",
        action="store_true",
        default=True,
        help="导出前将 AD 译为中文（默认开启）",
    )
    run_p.add_argument("--no-translate-zh", dest="translate_zh", action="store_false")

    exp_p = sub.add_parser("export-ad", help="将已有 stage3 AD 转为中文 SRT(+康辉 TTS) 并压制进原始完整输入视频")
    exp_p.add_argument("--work-dir", required=True, help="含 stage3_ad_scripts.json 的工作目录")
    exp_p.add_argument("--video", help="原始完整输入视频（默认读 work-dir/config.json 的 video_path）")
    exp_p.add_argument("--output", help="输出视频路径（默认 work-dir/<stem>_ad.mp4）")
    exp_p.add_argument("--srt", help="输出 SRT 路径（默认 work-dir/stage3_ad_scripts.srt，中文）")
    exp_p.add_argument("--tts", dest="enable_tts", action="store_true", default=True)
    exp_p.add_argument("--no-tts", dest="enable_tts", action="store_false")
    exp_p.add_argument("--tts-url")
    exp_p.add_argument("--tts-voice-id")
    exp_p.add_argument("--tts-voice-name", default=None, help="音色名，默认配置康辉")
    exp_p.add_argument("--tts-spk", help="无音色克隆时的 SFT 说话人")
    exp_p.add_argument("--tts-speed", type=float, help="TTS 语速，默认 1.2")
    exp_p.add_argument("--translate-zh", dest="translate_zh", action="store_true", default=True)
    exp_p.add_argument("--no-translate-zh", dest="translate_zh", action="store_false")
    exp_p.add_argument("--actor-db", help="演员库路径（用于构建中文专名表）")

    ann_p = sub.add_parser("annotate", help="全片人脸椭圆标注")
    ann_p.add_argument("--video", required=True)
    ann_p.add_argument("--actor-db", required=True)
    ann_p.add_argument("--output")
    ann_p.add_argument("--threshold", type=float, default=0.4)
    ann_p.add_argument("--frame-interval", type=int, default=5)
    ann_p.add_argument("--face-gpu", type=int, default=2, help="InsightFace GPU 编号（默认 2）")

    db_p = sub.add_parser("build-actor-db", help="构建演员数据库")
    db_p.add_argument("--movie", required=True)
    db_p.add_argument("--output", default=str(ACTOR_DATABASES_DIR))
    db_p.add_argument("--db-dir", help="在已有目录原地重建（保留 UUID）")
    db_p.add_argument("--with-vectors", action="store_true")

    check_p = sub.add_parser("check-tmdb", help="检测 TMDB API / 代理是否可用")
    check_p.add_argument("--query", default="Signs", help="测试搜索词")
    check_p.add_argument("--proxy", help="代理 URL，如 http://192.168.1.10:7097")

    args = parser.parse_args()

    if args.command == "transcribe":
        if args.asr_backend == "dashscope":
            os.environ["HIERAD_ASR_PROVIDER"] = "dashscope"
        result = transcribe_video(
            video_path=args.video,
            output_json_path=args.output,
            model_size=args.model,
            language=args.language,
            asr_backend=args.asr_backend,
            asr_url=args.asr_url,
        )
        out = args.output or str(Path(args.video).parent / f"{Path(args.video).stem}_asr.json")
        print(f"转录完成: {len(result)} 个片段 → {out}")

    elif args.command == "run":
        os.environ["HIERAD_FACE_GPU"] = str(args.face_gpu)
        if getattr(args, "vlm_provider", None):
            os.environ["HIERAD_VLM_PROVIDER"] = args.vlm_provider
        if getattr(args, "llm_provider", None):
            os.environ["HIERAD_LLM_PROVIDER"] = args.llm_provider
        if getattr(args, "tts_provider", None):
            os.environ["HIERAD_TTS_PROVIDER"] = args.tts_provider
        if args.asr_backend == "dashscope":
            os.environ["HIERAD_ASR_PROVIDER"] = "dashscope"
        work_dir = Path(args.work_dir)
        whisper_path = args.whisper
        if not whisper_path:
            try:
                asr_result = transcribe_video(
                    video_path=args.video,
                    work_dir=str(work_dir),
                    model_size="base",
                    asr_backend=args.asr_backend,
                    asr_url=args.asr_url,
                )
                whisper_path = str(work_dir / f"{Path(args.video).stem}_asr.json")
                print(f"转录完成: {len(asr_result)} 个片段")
            except (ImportError, RuntimeError, FileNotFoundError) as e:
                print(f"错误: 自动转录失败: {e}")
                return
        actor_db = args.actor_db or args.canonical_characters
        _, ads = run_pipeline(
            video_path=args.video,
            whisper_path=whisper_path,
            work_dir=work_dir,
            canonical_characters_path=args.canonical_characters,
            actor_db_path=actor_db,
            skip_annotate=args.skip_annotate,
            annotation_threshold=args.annotation_threshold,
            annotation_frame_interval=args.annotation_frame_interval,
            vlm_url=args.vlm_url,
            vlm_type=args.vlm_type,
            vlm_endpoint=args.vlm_endpoint,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            min_gap_sec=args.min_gap,
            gap_margin_sec=args.gap_margin,
            max_gap_duration_sec=args.max_gap_duration,
            max_ad_segments=args.max_ad_segments,
            max_gap_chunk_sec=args.max_gap_chunk,
            repair_sparse_asr=not args.no_repair_sparse_asr,
            silence_refine=not args.no_silence_refine,
            silence_noise_db=args.silence_noise_db,
            pipeline_version=args.pipeline_version,
            max_scene_clips=args.max_scene_clips,
            scene_detect_threshold=args.scene_threshold,
            min_scene_sec=args.min_scene_sec,
            max_scene_sec=args.max_scene_sec,
            burn_ad=args.burn_ad,
            enable_tts=args.enable_tts,
            translate_zh=args.translate_zh,
            tts_url=args.tts_url,
            tts_voice_id=args.tts_voice_id,
            tts_voice_name=args.tts_voice_name,
            tts_spk_id=args.tts_spk,
            tts_speed=args.tts_speed,
        )
        print(f"完成: {len(ads)} 个 AD 脚本")
        stage1_name = 'stage1_shot_descriptions.json' if args.pipeline_version == 'v3' else 'stage1_clip_descriptions.json'
        print(f"Stage1: {work_dir / stage1_name}")
        if args.pipeline_version == 'v3':
            print(f"Scenes: {work_dir / 'stage1_scenes.json'}")
        print(f"Stage3: {work_dir / 'stage3_ad_scripts.json'}")
        ad_video = work_dir / f"{Path(args.video).stem}_ad.mp4"
        ad_srt = work_dir / "stage3_ad_scripts.srt"
        if args.burn_ad and ad_video.exists():
            print(f"AD SRT(中文): {ad_srt}")
            print(f"AD 视频: {ad_video}")
            ad_wav = work_dir / "ad_audio.wav"
            if ad_wav.exists():
                print(f"AD 音轨: {ad_wav}")

    elif args.command == "export-ad":
        work_dir = Path(args.work_dir)
        stage3 = work_dir / "stage3_ad_scripts.json"
        if not stage3.exists():
            print(f"错误: 未找到 {stage3}")
            return
        video = args.video
        if not video:
            cfg_path = work_dir / "config.json"
            if cfg_path.exists():
                import json
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                video = cfg.get("video_path")
            if not video:
                print("错误: 请用 --video 指定原始完整输入视频，或确保 config.json 含 video_path")
                return
        try:
            result = export_ad_video(
                stage3_path=stage3,
                video_path=video,
                work_dir=work_dir,
                srt_path=args.srt,
                output_path=args.output,
                enable_tts=args.enable_tts,
                translate_zh=args.translate_zh,
                actor_db_path=getattr(args, "actor_db", None),
                tts_url=args.tts_url,
                voice_id=args.tts_voice_id,
                voice_name=args.tts_voice_name,
                spk_id=args.tts_spk,
                speed=args.tts_speed,
                llm_url=getattr(args, "llm_url", None),
                llm_model=getattr(args, "llm_model", None),
                llm_api_key=getattr(args, "llm_api_key", None),
            )
        except Exception as e:
            print(f"错误: {e}")
            return
        print(f"AD SRT(中文): {result['srt_path']}")
        if result.get("srt_en_path"):
            print(f"AD SRT(英文): {result['srt_en_path']}")
        print(f"AD 视频: {result['video_path']}（{result['ad_count']} 条）")
        if result.get("voice_name") or result.get("voice_id"):
            print(f"TTS 音色: {result.get('voice_name') or ''} {result.get('voice_id') or ''}".strip())
        if result.get("ad_audio_path"):
            print(f"AD 音轨: {result['ad_audio_path']}")
        if result.get("tts_error"):
            print(f"TTS 警告: {result['tts_error']}")

    elif args.command == "annotate":
        os.environ["HIERAD_FACE_GPU"] = str(args.face_gpu)
        result = annotate_video(
            video_path=args.video,
            actor_db_path=args.actor_db,
            output_path=args.output,
            recognition_threshold=args.threshold,
            frame_interval=args.frame_interval,
        )
        if result["status"] == "completed":
            print(f"标注完成: {result['output_path']}")
        else:
            for err in result.get("errors", []):
                print(f"  - {err}")

    elif args.command == "build-actor-db":
        os.environ.setdefault("HIERAD_FACE_GPU", "2")
        client = TMDBClient()
        if getattr(args, "db_dir", None):
            path = rebuild_face_vector_db(
                Path(args.db_dir),
                movie_name=args.movie,
                tmdb_client=client,
                with_vectors=getattr(args, "with_vectors", False),
            )
        else:
            path = build_face_vector_db(
                args.movie, Path(args.output), client, with_vectors=getattr(args, "with_vectors", False)
            )
        print(f"演员数据库: {path}" if path else "构建失败")

    elif args.command == "check-tmdb":
        ok, msg = check_tmdb_connection(query=args.query, proxy=args.proxy)
        print(msg)
        if not ok:
            raise SystemExit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
