# HierAD 使用示例

## 最小示例（Python API）

```python
from pathlib import Path
from hierad.pipeline import run_pipeline

# 需先准备：video.mp4 和 whisper_output.json
timecodes, ad_scripts = run_pipeline(
    video_path="/path/to/video.mp4",
    whisper_path="/path/to/whisper_output.json",
    work_dir=Path("./output"),
)
print(f"生成 {len(ad_scripts)} 段 AD 脚本")
```

## 使用演员数据库

```python
# 传入 actor_db 目录，用于角色名标准化
timecodes, ads = run_pipeline(
    video_path="video.mp4",
    whisper_path="asr.json",
    work_dir=Path("./output"),
    canonical_characters_path="./actor_databases/<uuid>",
)
```

## 单独使用各模块

```python
# 加载 ASR
from hierad.preprocess import ASRResult
asr = ASRResult.from_whisper_json("asr.json")

# 加载演员数据库
from hierad.actor_db import ActorDatabase
db = ActorDatabase(Path("./actor_databases/uuid"))
db.load()
db.normalize("Johnny")  # -> "TOMMY WISEAU"

# 描述模块
from hierad.describe import Stage2Understanding
s2 = Stage2Understanding(use_scenedetect=False, use_llm=False)
scenes = s2.aggregate_scenes(descriptions)
```
