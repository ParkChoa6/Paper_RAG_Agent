"""全局配置。

约定：这里只放「值」，不放逻辑。
其它模块 `from paper_rag.config import XXX` 取值，不许在别处硬编码。
"""

from pathlib import Path

# ---  路径 ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent  # 项目根目录
DATA_DIR: Path = PROJECT_ROOT / "data"  # 数据总目录
RAW_DIR: Path = DATA_DIR / "raw"  # 原始 PDF 放这
PROCESSED_DIR: Path = DATA_DIR / "processed"  # chunks.json、索引文件放这

# ---- Ollama ----
OLLAMA_HOST: str = "http://localhost:11434"  # 服务地址
GEN_MODEL: str = "qwen2:7b"  # 生成模型
EMB_MODEL: str = "quentinz/bge-large-zh-v1.5:latest"  # 向量模型

# ---- 检索 ----
TOP_K: int = 5  # 每次检索取几段

# ---- 日志 ----
LOG_LEVEL: str = "INFO"
LOG_FILE: Path = DATA_DIR / "logs" / "paper_rag.log"
