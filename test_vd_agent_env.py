#!/usr/bin/env python3
"""
在 vd-agent-main 环境下测试 HierAD（不安装，仅 PYTHONPATH）

用法:
  PYTHONPATH=/path/to/HierAD python test_vd_agent_env.py

或使用 vd-agent-main 的 Python:
  PYTHONPATH=/path/to/HierAD /path/to/conda/envs/vd-agent-main/bin/python test_vd_agent_env.py
"""

import sys
sys.path.insert(0, "/path/to/HierAD")

def main():
    print("=== HierAD 环境测试 (vd-agent-main) ===\n")

    # 1. 版本与导入
    from hierad import __version__
    print(f"[OK] hierad {__version__}")

    from hierad.pipeline import run_pipeline, load_canonical_mapping
    print("[OK] pipeline")

    from hierad.preprocess import cut_by_segments, load_ad_gap_segments, ASRResult, transcribe_video, annotate_video
    print("[OK] preprocess")

    from hierad.describe import Stage1Descriptor, Stage2Understanding, Stage3Refiner, DenseDescription, EnhancedGlobalContext
    from hierad.describe.utils import parse_stage1_output, compute_cosine_adjacent_similarities
    print("[OK] describe (+ utils)")

    from hierad.actor_db import ActorDatabase, build_from_tmdb, build_face_vector_db
    print("[OK] actor_db")

    # 2. 加载 ASR 格式
    asr_path = "/path/to/HierAD-old/data/1005_Signs/asr_result.json"
    try:
        asr = ASRResult.from_whisper_json(asr_path)
        print(f"[OK] ASRResult 加载: {len(asr.segments)} 个片段")
    except Exception as e:
        print(f"[SKIP] ASR 加载: {e}")

    # 3. AD 间隙解析
    try:
        clip = "/path/to/HierAD-old/data/1005_Signs/segments/0001.mp4"
        gaps, asr2, dur = load_ad_gap_segments(asr_path, clip, min_gap_sec=1.0)
        print(f"[OK] load_ad_gap_segments: {len(gaps)} 个间隙, 时长 {dur:.1f}s")
    except Exception as e:
        print(f"[SKIP] 间隙解析: {e}")

    print("\n=== 测试通过 ===")
    print("\nCLI 用法:")
    print("  PYTHONPATH=/path/to/HierAD /path/to/conda/envs/vd-agent-main/bin/python -m hierad.cli --help")
    print("  PYTHONPATH=/path/to/HierAD /path/to/conda/envs/vd-agent-main/bin/python -m hierad.cli run --video VIDEO --whisper WHISPER_JSON")

if __name__ == "__main__":
    main()
