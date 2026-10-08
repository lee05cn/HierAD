"""
HierAD - 层次化无障碍视频描述系统

架构（解耦、层次清晰）:
  hierad.config      - 统一配置
  hierad.actor_db    - 演员数据库（预处理最前：映射 + 可选人脸向量；与 preprocess 平级）
  hierad.preprocess  - 预处理（人脸标注、ASR、视频切分；应在描述前完成）
  hierad.describe    - 多阶段描述（Stage1/2/3）
  hierad.export      - AD 导出（SRT / 硬字幕视频）
  hierad.pipeline    - 主流程编排（默认从 Whisper + 切分进入；演员库需前置就绪）
"""

__version__ = "0.2.0"
