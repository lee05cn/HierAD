"""
视频切分 - 按时间区间切分视频（用于 AD 无声间隙片段）

使用 ffmpeg，无额外依赖
"""

import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from hierad.progress import ProgressReporter, progress_iter


def _sec_to_timecode(sec: float) -> str:
    """秒数转 HH:MM:SS.mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def cut_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    ffmpeg: str = "ffmpeg",
) -> bool:
    """切分单个片段"""
    duration = end_sec - start_sec
    cmd = [
        ffmpeg, "-y", "-v", "quiet",
        "-ss", str(start_sec), "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def cut_by_segments(
    video_path: str,
    segments: List[Tuple[float, float]],
    output_dir: str,
    num_workers: int = 1,
    show_progress: bool = False,
    reporter: Optional[ProgressReporter] = None,
) -> List[Tuple[str, str, str]]:
    """
    按片段列表切分视频

    Returns:
        [(start_tc, end_tc, segment_path), ...]
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = []
    indexed = progress_iter(
        list(enumerate(segments)),
        show=show_progress and reporter is None,
        reporter=reporter,
        stage="场景切分",
        desc="ffmpeg 切分",
        unit="seg",
        total=len(segments),
    )
    for i, (start, end) in indexed:
        out_path = output_dir / f"seg_{i:04d}.mp4"
        ok = cut_segment(str(video_path), start, end, str(out_path))
        if ok:
            result.append((_sec_to_timecode(start), _sec_to_timecode(end), str(out_path)))

    if len(result) < len(segments):
        failed = len(segments) - len(result)
        raise RuntimeError(
            f"ffmpeg 切分失败: {failed}/{len(segments)} 个片段未生成。"
            "请检查 ffmpeg 是否可用、视频路径是否有效、磁盘空间是否充足。"
        )
    return result
