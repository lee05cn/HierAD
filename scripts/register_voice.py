#!/usr/bin/env python3
"""
注册克隆音色到 VD-agent 音色库（零样本克隆）。

用法:
  python scripts/register_voice.py \
    --audio /path/to/李立宏.wav \
    --name "李立宏" \
    --prompt-text "高端的食材往往只需要采用最朴素的烹饪方式"

之后在 HierAD 中使用:
  python -m hierad.cli export-ad --work-dir ... --tts-voice-name "李立宏"
  或在 config.yaml: tts.voice_name: 李立宏
"""

import argparse
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

# 默认路径
DEFAULT_DB = Path("/path/to/VD-agent-master/web_app/web_app.db")
DEFAULT_PROFILES_DIR = Path(
    "/path/to/VD-agent-master/web_app/shared_data/voice_profiles"
)
DEFAULT_MODEL_DIR = Path(
    "/path/to/models/tts/cosyvoice/pretrained_models/CosyVoice2-0.5B"
)


def convert_to_wav(src: Path, dst: Path) -> None:
    """音频转 wav（16kHz mono），用 ffmpeg。"""
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def extract_embedding(audio_path: Path, model_dir: Path) -> "object":
    """用 campplus 提取音色嵌入向量。"""
    import numpy as np
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import onnxruntime
    import librosa

    campplus_path = model_dir / "campplus.onnx"
    if not campplus_path.exists():
        raise FileNotFoundError(f"campplus 模型未找到: {campplus_path}")

    print(f"  加载 campplus 模型: {campplus_path}")
    session = onnxruntime.InferenceSession(
        str(campplus_path), providers=["CPUExecutionProvider"]
    )

    print(f"  加载音频: {audio_path}")
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    speech_16k = torch.from_numpy(audio).float().unsqueeze(0)

    feat = kaldi.fbank(
        speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat_input = feat.unsqueeze(dim=0).cpu().numpy()

    print(f"  提取嵌入向量 (Fbank shape: {feat_input.shape})...")
    ort_inputs = {session.get_inputs()[0].name: feat_input}
    embedding = session.run(None, ort_inputs)[0].squeeze()

    return embedding


def get_audio_duration(audio_path: Path) -> float:
    import librosa

    _, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=0.1)
    full, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return len(full) / sr


def main():
    parser = argparse.ArgumentParser(description="注册克隆音色到 VD-agent 音色库")
    parser.add_argument("--audio", required=True, help="音频文件路径 (wav/mp3/m4a 等)")
    parser.add_argument("--name", required=True, help="音色名称（如 李立宏）")
    parser.add_argument(
        "--prompt-text", required=True,
        help="音频的精确文字稿（音频里说了什么，一字不差）",
    )
    parser.add_argument("--description", default="", help="音色描述（可选）")
    parser.add_argument("--db-path", default=str(DEFAULT_DB), help="音色库数据库路径")
    parser.add_argument(
        "--profiles-dir", default=str(DEFAULT_PROFILES_DIR),
        help="音色文件保存目录",
    )
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR),
        help="CosyVoice 模型目录（含 campplus.onnx）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="同名音色已存在时覆盖",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"错误: 音频文件不存在: {audio_path}")
        sys.exit(1)

    db_path = Path(args.db_path)
    profiles_dir = Path(args.profiles_dir)
    model_dir = Path(args.model_dir)

    # 检查同名
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    existing = con.execute(
        "SELECT id FROM voice_profiles WHERE name = ?", (args.name,)
    ).fetchone()
    if existing and not args.force:
        print(f"错误: 音色名 '{args.name}' 已存在 (id={existing['id']})")
        print(f"  用 --force 覆盖，或换个名字")
        con.close()
        sys.exit(1)
    if existing and args.force:
        # 删除旧的
        old_id = existing["id"]
        old_dir = profiles_dir / old_id
        if old_dir.exists():
            shutil.rmtree(old_dir)
        con.execute("DELETE FROM voice_profiles WHERE id = ?", (old_id,))
        con.commit()
        print(f"已删除旧音色: {args.name} ({old_id})")

    # 生成 UUID 和目录
    voice_id = str(uuid.uuid4())
    voice_dir = profiles_dir / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)

    # 保存音频（统一转 wav 16kHz mono）
    audio_dst = voice_dir / "audio.wav"
    if audio_path.suffix.lower() == ".wav":
        # 已经是 wav，但可能采样率/声道不对，统一转一下
        convert_to_wav(audio_path, audio_dst)
    else:
        convert_to_wav(audio_path, audio_dst)
    print(f"音频已保存: {audio_dst}")

    # 提取嵌入
    print("提取音色嵌入...")
    embedding = extract_embedding(audio_dst, model_dir)
    embedding_path = voice_dir / "embedding.npy"
    import numpy as np
    np.save(str(embedding_path), embedding)
    print(f"嵌入已保存: {embedding_path} (shape: {embedding.shape})")

    # 获取时长和采样率
    duration = get_audio_duration(audio_dst)
    print(f"音频时长: {duration:.2f}s")

    # 写入数据库
    con.execute(
        """INSERT INTO voice_profiles (
            id, name, description, audio_path, embedding_path,
            prompt_text, duration, sample_rate, status, created_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            voice_id,
            args.name,
            args.description,
            str(audio_dst),
            str(embedding_path),
            args.prompt_text,
            duration,
            16000,
            "completed",
            datetime.now(),
            None,
        ),
    )
    con.commit()
    con.close()

    print(f"\n{'='*50}")
    print(f"音色注册成功!")
    print(f"  名称: {args.name}")
    print(f"  ID:   {voice_id}")
    print(f"  状态: completed")
    print(f"  时长: {duration:.2f}s")
    print(f"  文字稿: {args.prompt_text}")
    print(f"{'='*50}")
    print(f"\n在 HierAD 中使用:")
    print(f'  python -m hierad.cli export-ad --work-dir <dir> --tts-voice-name "{args.name}"')
    print(f"  或 config.yaml: tts.voice_name: {args.name}")
    print(f"  或直接用 ID: --tts-voice-id {voice_id}")


if __name__ == "__main__":
    main()
