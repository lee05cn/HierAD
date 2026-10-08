"""用已有中文 SRT 直接生成 TTS 并压制到视频，跳过翻译步骤。

用法:
  python scripts/reburn_from_srt.py [音色名] [--work-dir DIR] [--video PATH] [--srt PATH] [--output PATH]

默认:
  work-dir = work_dir/e2e_wolf_warrior1
  video = /path/to/data/test/Wolf Warrior1/1.mp4
  srt = <work-dir>/stage3_ad_scripts.srt
  output = <work-dir>/<video_stem>_ad.mp4
"""
import argparse
import re
import subprocess
from pathlib import Path

from hierad.export.tts import synthesize_ad_clips, build_ad_track
from hierad.export.ad_video import burn_ad_subtitles


def parse_srt(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", text)
    entries = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 3:
            continue
        tc = lines[1]
        m = re.match(r"(\S+)\s*-->\s*(\S+)", tc)
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        body = "\n".join(lines[2:]).strip()
        entries.append({
            "ad_idx": len(entries),
            "start_time": start,
            "end_time": end,
            "ad_script": body,
        })
    return entries


def get_video_duration(path):
    r = subprocess.run(
        ["ffprobe", "-hide_banner", "-i", str(path)],
        capture_output=True, text=True
    )
    for line in r.stderr.splitlines():
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", line)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0


def resolve_voice_id(name):
    from hierad.export.voice import resolve_voice_id_and_prompt
    vid, ptxt, vname = resolve_voice_id_and_prompt(name)
    return vid, ptxt


def main():
    parser = argparse.ArgumentParser(description="用已有中文 SRT 生成 TTS 并压制视频")
    parser.add_argument("voice", nargs="?", default="康辉", help="音色名（默认康辉）")
    parser.add_argument("--work-dir", default="work_dir/e2e_wolf_warrior1")
    parser.add_argument("--video", help="原始视频路径")
    parser.add_argument("--srt", help="SRT 路径（默认 <work-dir>/stage3_ad_scripts.srt）")
    parser.add_argument("--output", help="输出视频路径")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    srt = Path(args.srt) if args.srt else work_dir / "stage3_ad_scripts.srt"

    if args.video:
        video = Path(args.video)
    else:
        # 尝试从 config.json 读 video_path
        cfg = work_dir / "config.json"
        if cfg.exists():
            import json
            with open(cfg) as f:
                vp = json.load(f).get("video_path")
            video = Path(vp) if vp else None
        else:
            video = None
        if not video:
            video = Path("/path/to/data/test/Wolf Warrior1/1.mp4")

    if args.output:
        output = Path(args.output)
    else:
        output = work_dir / f"{video.stem}_ad.mp4"

    entries = parse_srt(srt)
    print(f"解析到 {len(entries)} 条 AD")
    for i, e in enumerate(entries):
        print(f"  AD#{i+1}: {e['ad_script']}")

    vid, prompt_text = resolve_voice_id(args.voice)
    print(f"音色: {args.voice} ({vid})")

    tts_dir = work_dir / "tts"
    clips = synthesize_ad_clips(
        [e["ad_script"] for e in entries],
        tts_dir,
        voice_id=vid,
        prompt_text=prompt_text,
    )
    print(f"TTS 合成完成: {len(clips)} 个片段")

    dur = get_video_duration(video)
    track = build_ad_track(entries, clips, work_dir / "ad_audio.wav", track_len_sec=dur)
    print(f"AD 音轨: {track}")

    burn_ad_subtitles(video, srt, output, ad_wav_path=track)
    print(f"AD 视频: {output}")


if __name__ == "__main__":
    main()
