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

# ---- PDF 版面过滤（单位：点，1 点 = 1/72 英寸）----
PAGE_MARGIN_X = 60  # 块 x0 小于它 → 边栏水印，丢弃
PAGE_HEADER_Y = 60  # 块 y1 小于它 → 页眉，丢弃
PAGE_FOOTER_Y = 730  # 块 y0 大于它 → 页脚，丢弃

# ---- 标题识别 ----
HEADING_SIZE_GAP = 1.0  # 块主字号 ≥ 正文字号 + 它 → 标题块

# ---- 切分 ----
# 清洗后短于它的块直接丢弃。拦的是「图里的文字」和「公式碎片」：
#   'q(x)'  'd(z)'  'Query Encoder'  'Barack Obama was born in Hawaii.(x)'
# 这个数字没有数据支撑（长度分布是连续的，没有天然断点），
# 是「代价权衡」的结果：丢一个 35 字的碎片几乎不损失知识，
# 留 110 个垃圾 chunk 却是 32% 的索引噪音。以后可以调。
MIN_CHUNK_CHARS = 40
