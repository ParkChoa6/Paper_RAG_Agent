import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.logger import setup

setup()  # ← 入口负责调用

import logging

a = logging.getLogger("paper_rag.ingest")
b = logging.getLogger("paper_rag.index")

a.debug("这条 DEBUG 应该看不到（级别是 INFO）")
a.info("切分出 3000 个 chunk")
b.warning("索引文件不存在，将重建")
b.error("Ollama 连不上")
