# 演员数据库子项目

与 `preprocess` 同级，参考 **Actor_Dataset_Construct** 实现电影演员人脸向量数据库。

## 流程（参考 Actor_Dataset_Construct）

```
TMDB API → 演员列表
    ↓
图片爬取 → 下载演员头像
    ↓
InsightFace → 人脸检测 + 512 维向量
    ↓
Faiss → 向量存储与检索
```

## 目录结构

```
actor_db/
├── __init__.py
├── store.py        # 存储、加载、人脸检索
├── build.py        # 构建脚本（metadata / 完整向量库）
├── tmdb.py         # TMDB API
├── image_crawler.py   # 演员图片下载
├── face_processor.py  # InsightFace 人脸向量（可选）
├── vector_store.py    # Faiss 向量存储（可选）
└── README.md
```

## 存储格式

```
{output_dir}/{uuid}/
├── config.json       # 角色名映射 movie_name, mapping
├── images/           # 演员头像
├── faiss_index.bin   # 人脸向量索引（可选）
├── metadata.pkl      # 人脸元数据（可选）
└── id_mapping.json   # ID 映射（可选）
```

## 使用

### 视频人脸标注

需先构建含向量的数据库，再对视频进行标注：

```bash
# 1. 构建演员数据库（含人脸向量）
hierad build-actor-db --movie "The Room" --output ./actor_databases --with-vectors

# 2. 在视频中标注识别到的演员
hierad annotate --video movie.mp4 --actor-db ./actor_databases/<uuid>
```

### 仅元数据（角色名映射）

```bash
hierad build-actor-db --movie "The Room" --output ./actor_databases
```

### 完整人脸向量库

```bash
pip install hierad[actor_db_vectors]
hierad build-actor-db --movie "The Room" --output ./actor_databases --with-vectors
```

### Python API

```python
from hierad.actor_db import ActorDatabase, build_face_vector_db

# 构建（含向量）
path = build_face_vector_db("The Room", Path("./actor_databases"), with_vectors=True)

# 加载与检索
db = ActorDatabase(path)
db.load()
db.normalize("Johnny")  # 角色名标准化

# 人脸检索（若已构建向量库）
# embedding = ...  # 512 维向量
# results = db.search_face(embedding, top_k=5)
```

## 依赖

| 功能 | 依赖 |
|------|------|
| 元数据 + 图片 | requests |
| 人脸向量 | insightface, faiss-cpu, onnxruntime, opencv-python |
