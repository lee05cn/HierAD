# HierAD

**Hierarchical Audio Description** for video: a modular pipeline that turns a full-length video into **time-aligned AD (audio description) scripts** using a vision-language model (VLM) for dense per-segment understanding, an LLM for global narrative structure, and an LLM again for segment-level refinement.

This repository is a **clean refactor** with clear boundaries between preprocessing, optional actor/face resources, and the multi-stage description stack. It does **not** depend on the legacy HierAD-old codebase.

---

## Table of contents

- [What HierAD does](#what-hierad-does)
- [Pipeline overview](#pipeline-overview)
- [End-to-end workflow](#end-to-end-workflow)
- [Pipeline stages (algorithms)](#pipeline-stages-algorithms)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Work directory outputs](#work-directory-outputs)
- [Development](#development)
- [License](#license)

---

## What HierAD does

1. **Preprocessing (recommended order)**  
   Build an **actor database** (TMDB metadata and optional face embeddings), optionally **annotate faces** on the source video, run **ASR** (Whisper) for timed transcripts, then **split** the video into short clips aligned with Whisper segments.

2. **Description pipeline**  
   **Stage 1**: each clip + local dialogue → structured dense caption (setting / characters / action) via a **VLM**.  
   **Stage 2**: aggregate segments into **scenes** (SBERT / TF-IDF / fixed window), then use an **LLM** for scene summaries, character graph, event chain, and story structure.  
   **Stage 3**: **LLM** refines each segment into a concise AD line using global context (characters, scene, narrative phase).

Optional **canonical character names** from an actor DB improve consistency in Stage 1 when passed via `--canonical-characters`.

---

## Pipeline overview

![HierAD pipeline](docs/pipeline.png)

---

## End-to-end workflow

Think of two layers: **preprocessing first**, then the **main AD pipeline**.

### Phase A — Preprocessing (do this first)

| Step | What | CLI / module |
|------|------|----------------|
| 1 | **Actor database** — cast list from TMDB, alias → canonical name mapping; optional face embeddings (Faiss) for recognition | `hierad build-actor-db` (`hierad.actor_db`) |
| 2 | **Face annotation** (optional) — draw boxes/labels on the source video using the vector DB | `hierad annotate` (`hierad.preprocess.face_annotator`) |
| 3 | **ASR** — Whisper JSON with `segments` (`start`, `end`, `text`) | `hierad transcribe` or automatic inside `hierad run` |
| 4 | **Splitting** — ffmpeg cuts one file per Whisper segment | Inside `run_pipeline` (`hierad.preprocess.cutter`) |

The `actor_db` package lives next to `preprocess/` in the tree, but **logically it is still preprocessing**: you prepare it **before** running the description pipeline, then point `run` at the DB folder or a mapping JSON.

### Phase B — Description pipeline (`run` / `run_pipeline`)

1. Load Whisper segments → **cut** video into `work_dir/segments/seg_XXXX.mp4`.
2. Load **ASR** and align dialogue to each clip’s time range.
3. **Stage 1 (VLM)** — dense descriptions per clip (with optional canonical name mapping).
4. **Stage 2 (LLM + optional SBERT/TF-IDF)** — scenes, characters, events, story phases.
5. **Stage 3 (LLM)** — final AD script per segment.

---

## Pipeline stages (algorithms)

- **Stage 1**  
  Sliding-window context from recent segments; prompts request `SETTING` / `CHARACTERS` / `ACTION`. HTTP POST to your VLM (`HIERAD_VLM_URL` + configurable path, default `/prompted_inference`). Payload shape depends on `HIERAD_VLM_TYPE` (`qwen2.5-vl`, `video-xl2`, `videollama3`).

- **Stage 2 — scene grouping**  
  Priority: **SBERT** adjacent cosine similarity on segment text (with fallbacks) → **TF-IDF** similarity → **fixed-size window**. Then LLM calls fill in scene summaries, a character graph (JSON), an event chain, and high-level story structure.

- **Stage 3**  
  Per-segment LLM refinement with injected context: character hints, scene summary, story phase, progression within the scene, and recent AD lines to avoid repetition.

- **Actor / face (preprocessing)**  
  TMDB → images → InsightFace embeddings → **Faiss** inner-product search on L2-normalized vectors (cosine-like). Used by `annotate` and optionally to build richer DBs.

---

## Project layout

```
hierad/
├── config.py              # Central settings (env + optional config.yaml)
├── cli.py                 # Entry point: transcribe, run, annotate, build-actor-db
├── pipeline.py            # Orchestrates cut → Stage1 → Stage2 → Stage3
├── preprocess/
│   ├── asr.py             # Whisper, ASRResult, time-range query
│   ├── cutter.py          # Load Whisper JSON, ffmpeg segment cutting
│   └── face_annotator.py  # Optional face boxes on video (OpenCV + DB)
├── actor_db/              # Preprocessing data layer (TMDB, build, Faiss)
│   ├── store.py           # ActorDatabase: mapping + optional vector search
│   ├── build.py           # build_from_tmdb, build_face_vector_db
│   ├── tmdb.py            # TMDB API client
│   ├── face_processor.py  # InsightFace detect/embed (optional dep)
│   └── vector_store.py    # Faiss store/load/search
└── describe/
    ├── models.py          # DenseDescription, scenes, events, global context
    ├── prompts.py         # Prompt templates
    ├── stage1.py          # VLM dense description
    ├── stage2.py          # Scene aggregation + LLM global understanding
    ├── stage3.py          # LLM AD refinement
    └── utils/             # Parsers, similarity, SBERT factory, character helpers
```

See also `configs/config.yaml.example` and `examples/README.md` for snippets.

---

## Requirements

### System

- **ffmpeg** — required for audio extraction, video splitting, and encoding. Must be on `PATH`.
- **Python** ≥ 3.9

### Remote or local services

- **VLM HTTP API** — expected to accept JSON with at least `video_path` and prompts; response convention `{ "code": 0, "data": "..." }` or compatible fields (`result` / `output` / `text`). Adjust `HIERAD_VLM_ENDPOINT` if your server uses a different path.
- **OpenAI-compatible LLM HTTP API** — used for Stage 2 and Stage 3 (`base_url` + `model` + optional `api_key`).

### Optional Python extras

| Extra | Install | Purpose |
|--------|---------|---------|
| `asr` | `pip install "hierad[asr]"` | Whisper transcription |
| `actor_db_vectors` | `pip install "hierad[actor_db_vectors]"` | Faiss + InsightFace + OpenCV for vector DB and `annotate` |
| `describe_extras` | `pip install "hierad[describe_extras]"` | Sentence-Transformers + scikit-learn for stronger scene clustering |
| `full` | `pip install "hierad[full]"` | Convenience bundle (see `pyproject.toml`) |
| `dev` | `pip install "hierad[dev]"` | pytest |

For **`config.yaml`** loading, install **PyYAML** (`pip install pyyaml`). Without it, only environment variables and defaults apply.

---

## Installation

```bash
git clone <your-fork-or-url> HierAD
cd HierAD
pip install -e .
```

Install extras as needed, e.g.:

```bash
pip install -e ".[asr,describe_extras]"
```

---

## Configuration

**Precedence:** environment variables → optional `config.yaml` (project root or current working directory) → built-in defaults.

Default **provider is `local`**: VLM/LLM/ASR/TTS talk to the services on this machine (`8002` / `8014` or `8016` / `8001` / `8003`). Cloud (Alibaba DashScope) is opt-in per component and does not replace local endpoints unless you set `provider: dashscope`.

To send only some stages to the cloud:

```yaml
vlm:
  provider: dashscope   # qwen3-vl-flash
llm:
  provider: dashscope   # qwen-plus
asr:
  provider: local       # keep Whisper at :8001
tts:
  provider: local       # keep CosyVoice at :8003
```

```bash
export DASHSCOPE_API_KEY=sk-...
hierad run --video movie.mp4 --vlm-provider dashscope --llm-provider dashscope
```

Install cloud extras with `pip install -e ".[cloud]"`. Stage1 writes `stage1_clip_descriptions.json` after each successful clip so a failed VLM call can resume.

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HIERAD_VLM_URL` | Base URL of the VLM service | `http://localhost:8002` |
| `HIERAD_VLM_PROVIDER` | `local` or `dashscope` | `local` |
| `HIERAD_VLM_TYPE` | `qwen2.5-vl` / `video-xl2` / `videollama3` | `qwen2.5-vl` |
| `HIERAD_VLM_ENDPOINT` | HTTP path for inference, e.g. `/prompted_inference` | `/prompted_inference` |
| `HIERAD_LLM_PROVIDER` | `local` or `dashscope` | `local` |
| `HIERAD_LLM_BASE_URL` | OpenAI-compatible API base URL | `http://127.0.0.1:11436/v1` |
| `HIERAD_LLM_MODEL` | Model name for chat completions | `qwen2.5:7b` |
| `HIERAD_LLM_API_KEY` | API key (use real key for gated endpoints) | `EMPTY` |
| `DASHSCOPE_API_KEY` | 百炼 Key（仅 dashscope provider） | (empty) |
| `HIERAD_ASR_PROVIDER` | `local` or `dashscope` | `local` |
| `HIERAD_TTS_PROVIDER` | `local` or `dashscope` | `local` |
| `HIERAD_TMDB_API_KEY` | TMDB key for `build-actor-db` | (empty) |
| `HIERAD_ACTOR_DB_DIR` | Default output root for actor DBs | `./actor_databases` |

`HTTP_PROXY` / `HTTPS_PROXY` are respected by the TMDB client when set.

### `config.yaml`

Copy `configs/config.yaml.example` to `config.yaml` at the repo root (or your cwd) and edit `vlm`, `llm`, `tmdb`, and `paths` sections. CLI flags still override for a given run when passed explicitly.

---

## CLI reference

```bash
hierad --version
```

### `hierad transcribe`

Extract audio → Whisper → JSON with `segments`.

| Option | Description |
|--------|-------------|
| `--video` | Input video path (required) |
| `--output` | Output JSON path (default: next to video, `*_asr.json`) |
| `--model` | Whisper size: `tiny` / `base` / `small` / `medium` / `large` |
| `--language` | e.g. `en`, `zh`; omit for auto-detect |

### `hierad run`

Full pipeline: split → Stage 1 → 2 → 3.

| Option | Description |
|--------|-------------|
| `--video` | Source video path (required) |
| `--whisper` | Whisper JSON; if omitted, runs transcription (requires `hierad[asr]`) |
| `--work-dir` | Working directory (default: `./work_dir`) |
| `--canonical-characters` | Actor DB directory (with `config.json`) or a JSON mapping file |
| `--vlm-url` | Override VLM base URL |
| `--vlm-type` | Override VLM type |
| `--vlm-endpoint` | Override inference path |
| `--llm-url` | Override LLM base URL |
| `--llm-model` | Override model name |
| `--llm-api-key` | Override API key for this run |

### `hierad build-actor-db`

Search TMDB by movie title, write a UUID-named folder with `config.json` (`mapping`, etc.). With `--with-vectors`, downloads images and builds Faiss index (requires `hierad[actor_db_vectors]`).

| Option | Description |
|--------|-------------|
| `--movie` | Movie title to search (required) |
| `--output` | Parent directory for DB folders (default: `HIERAD_ACTOR_DB_DIR`) |
| `--with-vectors` | Build face embeddings + Faiss |

Set `HIERAD_TMDB_API_KEY` (or `tmdb.api_key` in `config.yaml`) before running.

### `hierad annotate`

Overlay detected faces with labels from the vector DB.

| Option | Description |
|--------|-------------|
| `--video` | Input video (required) |
| `--actor-db` | Path to DB directory containing `faiss_index.bin` (required) |
| `--output` | Output video path (default: `*_annotated` next to input) |
| `--threshold` | Similarity threshold (default: `0.6`) |
| `--frame-interval` | Run detection every N frames (default: `5`) |

---

## Python API

Minimal example:

```python
from pathlib import Path
from hierad.pipeline import run_pipeline

timecodes, ad_scripts = run_pipeline(
    video_path="/path/to/video.mp4",
    whisper_path="/path/to/whisper.json",
    work_dir=Path("./output"),
    canonical_characters_path="/path/to/actor_db_uuid",  # optional
    vlm_url=None,           # optional overrides
    llm_model=None,
)
```

`run_pipeline` returns:

- `timecodes` — list of `(start_tc, end_tc)` strings per segment  
- `ad_scripts` — list of AD strings, same length as segments  

Additional keyword arguments match the CLI overrides (`vlm_type`, `vlm_endpoint`, `llm_url`, `llm_api_key`, etc.).

More examples: `examples/README.md`.

---

## Work directory outputs

After `hierad run`, under `--work-dir` you typically get:

| Path | Content |
|------|---------|
| `config.json` | Original video path, per-segment video paths, timecodes, ASR path |
| `segments/seg_*.mp4` | Short clips |
| `*_asr.json` | Written when transcription runs inside `run` |
| `stage1_dense_descriptions.json` | Per-segment VLM fields |
| `stage2_global_context.json` | Scenes and character list (summary) |
| `stage3_ad_scripts.srt` | Final AD subtitles (**Chinese** by default) |
| `stage3_ad_scripts_en.srt` | English AD subtitles (backup) |
| `stage3_ad_scripts_zh.json` | Stage3 entries with `ad_script` in Chinese |
| `tts/*.wav` | Per-cue CosyVoice AD speech (default voice: 康辉) |
| `ad_audio.wav` | Full-length AD track aligned to source timeline |
| `<stem>_ad.mp4` | Original full input with Chinese hardsubs + TTS mix |

Defaults: translate to Chinese + TTS with voice profile **康辉** (`config.yaml` → `tts.voice_name`). Override with `--tts-voice-name` / `--no-translate-zh`.

```bash
hierad export-ad --work-dir ./work_dir/e2e_iron_man_input
# English SRT/TTS only:
hierad export-ad --work-dir ./work_dir/e2e_iron_man_input --no-translate-zh
```

Requires `ffmpeg` with libass, CosyVoice at `http://127.0.0.1:8003`, VD-agent `web_app.db` voice library, LLM for translation, and `pydub`.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Use `PYTHONPATH=.` if running tests without editable install.

---

## License

MIT (see `pyproject.toml`).
