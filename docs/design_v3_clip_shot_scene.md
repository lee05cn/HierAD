# HierAD v3 设计备忘：Clip → Shot → Scene

> 记录日期：2026-05-28  
> 状态：设计草案（待实现）  
> 参考：shot-by-shot（物理切镜 + AD 上下文）、HierAD v2 现状复盘

---

## 1. 核心思路一句话

**物理上按镜头切 Clip，语义上每条 Clip 产出结构化 Shot 描述，再按「宏观空间 + 主题连续性」把连续 Shot 聚成 Scene；Scene 代表一个小主题事件，由组内 Shot 提炼而成。**

> **Setting 难点**：同一 Scene 内可有多镜切换（如潜水艇内 ↔ 海底 ↔ 残骸），微观 setting 不完全相同，但宏观上同属一个叙事空间（如「泰坦尼克号海底打捞」）。**聚合看 macro_space，AD 描述看 micro_setting。**

---

## 2. 三层概念

```
整部影片
  └─ Scene（叙事场景：同 macro_space 的连续 shot 组 + 主题事件摘要）
       └─ Shot（语义层：单 clip 的结构化理解）
            └─ Clip（物理层：PySceneDetect 切分 + ffmpeg 输出的视频片段）
```

| 层级 | 定义 | 检测 / 产生方式 | 主要字段 |
|------|------|-----------------|----------|
| **Clip** | 摄影机切镜后的物理片段 | PySceneDetect（+ 可选 merge/split）→ ffmpeg 切文件 | `clip_idx`, `start_sec`, `end_sec`, `video_path` |
| **Shot** | 对单个 Clip 的结构化语义描述 | Stage1 VLM（+ 上下文窗口） | `micro_setting`, `macro_space`, `characters`, `action`, `dialogue` |
| **Scene** | 同一宏观叙事空间下的连续剧情单元 | **macro_space 一致 + 主题连续** 的 Shot 聚合 | `scene_id`, `shot_range`, `macro_space`, `theme_event`, `shots[]` |

**与 v2 的对应关系**

| v2 名称 | v3 语义 |
|---------|---------|
| `scene_clip` / `clip_idx` | **Clip**（物理镜头片段） |
| Stage1 `ClipDescription` | **Shot**（结构化描述） |
| Stage1 增强 `SceneInfo`（1 clip = 1 scene） | 改为真正的 **Scene**（多 shot 聚合） |

**与 shot-by-shot 的关系**

- shot-by-shot 的 **Shot** ≈ 本文的 **Clip**（PySceneDetect 切镜）
- shot-by-shot **不做叙事 Scene**；v3 的 **Scene** 是 HierAD 的增量（setting 聚合 + 主题事件）

---

## 3. Stage1：Clip → Shot

### 3.1 固定上下文窗口 = 2

Stage1 处理第 `i` 个 Clip 时，**固定注入前 2 个 Clip 的结构化 Shot 信息**（不是全片，不是可变窗口）。

```
Clip[i-2]  Clip[i-1]  |  Clip[i]  ← 当前待描述
   ↓          ↓       |     ↓
 Shot[i-2]  Shot[i-1] |  VLM → Shot[i]
```

- 首个 Clip：无前置上下文（或仅当前 dialogue）
- 第二个 Clip：注入 1 条 before
- 第三个及以后：注入 2 条 before

### 3.2 上下文注入格式

**Before shots**（最多 2 条）：上一个 / 上上个 Clip 已产出的结构化内容：

```yaml
- setting:   "..."
  action:    "..."
  characters: "..."
  dialogue:  "..."   # 该 clip 时间范围内的 ASR 摘要
```

**Current clip** 额外注入：

```yaml
dialogue: "..."      # 当前 clip 的 ASR / dialogue_overlap
```

设计意图：

- 用 **结构化字段** 替代 v2 的自由文本 `recent_context`，减少 VLM 重复与遗漏
- **Dialogue 仅作上下文**，不要求写入 ACTION，且禁止引用台词（沿用 AD 规则）
- 窗口 = 2：在连贯性与 prompt 长度之间折中（可对比 shot-by-shot 的 5-shot AD 窗口，Stage1 全片扫描用较小窗口即可）

### 3.3 Setting 规则变更（重要）

**v2（废弃）**

```
SETTING: [location/environment, or "Same" if unchanged from previous clip]
```

**v3**

- **每条 Shot 必须写出完整、具体的 setting 描述**
- **禁止使用 `Same` / `Unchanged` / `As before` 等占位**
- 即使环境未变，也要用简短自然语言复述（如 `A dimly lit farmhouse kitchen at night`），而非指代上一镜

原因：

- Scene 聚合依赖 **setting 字符串可比性**；`Same` 无法聚类
- 下游 Scene 主题提炼需要每条 Shot 自洽，不依赖隐式指代

### 3.4 Shot 输出格式（Stage1 VLM）

VLM 输出 **微观环境**；**宏观空间**由 Stage1.5 轻量 LLM 补全（见 §4.1）。

```
SETTING:    [本镜具体环境，禁止 Same；如「潜水艇控制舱内」「海底泰坦尼克号残骸旁」]
CHARACTERS: [标注角色名大写；无标注则用外观+角色]
ACTION:     [本镜视觉动作，20–35 词，现在时]
```

持久化字段映射：

| 字段 | 来源 |
|------|------|
| `micro_setting` | VLM `SETTING` 行 |
| `macro_space` | Stage1.5 LLM 归一化（见下） |
| `characters`, `action`, `dialogue` | 同上 |

---

## 4. Scene 聚合：Shot → Scene

### 4.1 双层 Setting：micro vs macro

**问题示例（泰坦尼克号打捞）**

| Shot | micro_setting（每镜具体） | 是否同一 Scene？ |
|------|---------------------------|------------------|
| S0 | 深海潜水艇控制舱内 | ✅ |
| S1 | 北大西洋海底，泰坦尼克号残骸外 | ✅ 同一次打捞任务 |
| S2 | 潜水器探照灯照向船体断裂处 | ✅ |
| S3 | 控制舱内，船员查看监视器 | ✅ |

微观 setting 在「舱内 / 舱外」间切换，但 **macro_space** 应均为类似：

`North Atlantic seabed — Titanic wreck salvage expedition`

**原则**

| 层级 | 用途 | 谁消费 |
|------|------|--------|
| **micro_setting** | 本镜物理环境，供 AD 准确描述「此刻画面在哪」 | Stage2/3 局部视觉 |
| **macro_space** | 叙事空间 / 任务上下文，供 Scene 边界与主题归纳 | Scene 聚合、Stage1b、Stage3 全局上下文 |

**macro_space 如何产生（Stage1.5，推荐）**

对每条 Shot（或滑动窗口内相邻 Shot），用 **轻量 LLM** 输入：

```yaml
micro_setting, characters, action, dialogue
# + 可选：前 1–2 条的 macro_space
```

输出一行：

```
MACRO_SPACE: [10–20 词，抽象叙事空间，不含机位切换细节]
```

要求 LLM：

- 把「舱内 / 舱外 / 残骸特写」抽象到 **同一任务/同一地理区域**
- 若明显换地点（农场 → 医院）、换时间段（flashback）、换任务线 → 输出 **不同的** macro_space

**备选（无 LLM 时降级）**

- Embedding：`micro_setting + action` 向量，相邻 shot 余弦相似度 > τ 且角色有交集 → 同 macro
- 仅作 MVP fallback，长片准确率不如 LLM

### 4.2 聚合规则（修订）

**主键：macro_space 一致 + 主题连续**

不再用 micro_setting 字符串相等做 Scene 边界。

```
扫描 Shot[0..N]（时间序）:

  若 macro_space[i] == macro_space[i-1]
     且 主题连续（见下）
     且 无硬切分信号
    → 并入当前 Scene

  否则 → 开启新 Scene
```

**主题连续**（软信号，满足多数即可）：

- 时间连续（clip 首尾相接，无大跨度 jump）
- 主要角色集合有交集，或 dialogue 话题延续
- action 在叙事上承接（打捞 ↔ 下潜 ↔ 观察残骸 = 连续）

**硬切分信号**（强制开新 Scene）：

- `macro_space` 变化
- dialogue / action 出现明确时空跳转（如 `"Three years later"`, `"Back at the farm"`）
- 角色群完全更换且 action 无承接（可选 LLM 判定）

**反例：不应合并**

```
macro: "Farmhouse kitchen"  →  "Hospital emergency room"   # 必须拆 Scene
macro: "Cornfield search"   →  "TV news studio"            # 必须拆 Scene
```

**正例：应合并（泰坦尼克）**

```
S0 micro: sub interior      macro: Titanic salvage / North Atlantic seabed
S1 micro: open ocean floor  macro: Titanic salvage / North Atlantic seabed  → 同 Scene
S2 micro: wreck hull close  macro: Titanic salvage / North Atlantic seabed  → 同 Scene
```

### 4.3 Scene 内多 micro_setting 的结构

一个 Scene 可包含多个 micro_setting，这是 **正常情况**，不是聚合失败：

```yaml
scene_id: 7
macro_space: "North Atlantic seabed — Titanic wreck salvage expedition"
theme_event: "The crew descends to the wreck and inspects the hull breach."
shot_range: [42, 48]
micro_settings:                    # 组内去重列表，保留多样性
  - "Deep-sea submersible control room"
  - "Ocean floor beside the Titanic wreck"
  - "Close view of the ship's broken hull"
shots:
  - { clip_idx: 42, micro_setting: "...", action: "..." }
  - ...
```

AD 生成时：**micro_setting 来自当前 Shot**，**theme_event / macro_space 来自所属 Scene**。

### 4.4 Scene 的含义

每个 Scene 代表一个 **小的主题事件**（story beat），不是物理切镜数，而是 **同一 macro_space 下的一段连续叙事**。

Scene 级字段（目标 schema）：

```yaml
scene_id: 0
shot_range: [0, 2]          # inclusive clip_idx
start_time / end_time
macro_space: "..."          # 组内统一的宏观叙事空间
theme_event: "..."          # 从组内 shots 提炼的小主题事件
characters: "..."           # 组内出现角色的并集
micro_settings: ["...", "..."]   # 组内出现过的微观环境（去重）
shots:
  - { clip_idx, micro_setting, characters, action, dialogue }
```

### 4.5 主题事件（theme_event）提炼

Scene 内的 `theme_event` **不能**简单等于某一个 Shot 的 action，需要从 **Scene 内全部 Shot** 归纳：

- 输入：Scene 内各 Shot 的 `{setting, characters, action, dialogue}`
- 输出：一句过去时 / 第三人称的 **小主题摘要**（类似 v2 的 `EVENT`，但在 **Scene 层级**）

示例：

```
Shots: Graham pours water | Caroline watches | Bo pets the dog
theme_event: "In the kitchen, Caroline and Graham tend to the dog while Bo watches."
```

实现路径（待定）：

1. **LLM 批量摘要**（推荐）：每个 Scene 一次调用，输入组内 Shot 列表
2. **规则 fallback**：拼接 actions / 取最长 action（仅作 LLM 失败兜底）

---

## 5. 与下游 Stage 的关系（预览）

| 阶段 | v3 预期用法 |
|------|-------------|
| **Stage1b 增强** | 在 **Scene** 级建角色图谱、事件链、story phases（不再 1 clip = 1 scene，不再固定 5-clip 硬分组） |
| **Stage2 AD** | AD gap 映射到 Clip/Shot；上下文用 gap 附近 Shot + 所属 Scene 的 `theme_event` |
| **Stage3 精炼** | 局部：`Shot.micro_setting` + `action`；全局：`Scene.macro_space` + `theme_event` + `character_graph` + phase |

AD Gap 仍是独立时间单元（ASR 静音），不对齐 Clip 边界；通过 `clip_idx` / 时间戳归属到 Scene。

---

## 6. 待实现清单

- [ ] 术语重命名：`scene_clip` → `shot_clip` / `clip`（物理），`ClipDescription` → `ShotDescription`（可选，分阶段）
- [ ] Stage1 prompt：窗口=2、结构化 before 注入、禁止 Setting Same
- [ ] Stage1.5：`infer_macro_space(shot, prev_macro?)`（LLM）
- [ ] `aggregate_shots_to_scenes(shots) -> List[Scene]`（macro_space + 连续性 + 硬切分）
- [ ] Scene `theme_event` 提炼（LLM + fallback）
- [ ] 重写 `stage1_enhance.py`：输入 Scene 列表而非 per-clip SceneInfo
- [ ] PySceneDetect：评估是否改用 AdaptiveDetector（shot-by-shot / MAD-Eval 配置）

---

## 7. 开放问题

1. **Clip merge/split**（`min_scene_sec=5`, `max_scene_sec=90`）是否保留？过长 Clip 内可能含多个 implicit 切镜，是否影响 Shot 质量？
2. ~~Setting 近似匹配~~ → **已采用 micro/macro 双层**；待调：macro LLM prompt 与硬切分规则阈值
3. **Stage1 全片 Clip VLM vs AD-centric VLM**（shot-by-shot 只对 gap 取上下文）：v3 是否仍全片扫 Clip，还是 Stage2 再按需 VLM？
4. **Thread / Film Grammar**（shot-by-shot DINOv2）：可作为「同 macro 内机位组」辅助，**不替代** macro_space 聚合
5. **Scene 边界 LLM 仲裁**：相邻 shot 对 macro 相近但不确定时，是否用一次二元分类「是否同一叙事 beat？」

---

## 8. 层级对照图

```
时间轴 ───────────────────────────────────────────────────────►

Clip:   |--C0--|--C1--|--C2--|--C3--|--C4--|--C5--|
Shot:    S0     S1     S2     S3     S4     S5
micro:   厨房   厨房   潜艇内  海底   残骸   潜艇内
macro:   农场家  农场家  打捞   打捞   打捞   打捞
Scene:   |_ Scene0 _|__________ Scene1 (Titanic salvage) __________|
         theme_event_0              theme_event_1

AD Gap:              [====gap====]              (ASR 静音，独立定位)
                     └─ 归属 Scene 0，覆盖 C1–C2 等
```
