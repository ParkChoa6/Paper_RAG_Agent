"""Stage 2 入口：把 chunks.json 里的语料全部向量化，落盘成索引。

和 build_index.py 的分工：
    build_index.py     PDF          → chunks.json   （文本层）
    build_vectors.py   chunks.json  → vectors.npy   （向量层）

为什么分两步、两个脚本：
    两者的【失效条件】不同 ——
        换 PDF / 改切分逻辑  → 重跑 build_index.py
        换嵌入模型           → 只要重跑 build_vectors.py
    合成一个脚本的话，换个嵌入模型也得把 PDF 重新解析一遍，白等。

用法:
    python scripts/build_vectors.py
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import EMB_MODEL, IDS_FILE, PROCESSED_DIR, VECTORS_FILE
from paper_rag.embed import make_client
from paper_rag.index import build, save
from paper_rag.logger import setup

logger = logging.getLogger("paper_rag.build_vectors")


def main() -> None:
    setup()

    logger.info("嵌入模型 → %s", EMB_MODEL)

    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        logger.error("找不到 %s —— 先跑 scripts/build_index.py", chunks_path)
        return

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))["chunks"]
    logger.info("载入 %d 条 chunk ← %s", len(chunks), chunks_path)

    client = make_client()

    t0 = time.time()
    vectors, ids = build(client, EMB_MODEL, chunks)
    dt = time.time() - t0

    save(vectors, ids, EMB_MODEL)

    logger.info("嵌入 %d 条，用时 %.1f s", len(ids), dt)
    logger.info("写入 %s  (%.2f MB)", VECTORS_FILE, VECTORS_FILE.stat().st_size / 1024 / 1024)
    logger.info("写入 %s  (%.1f KB)", IDS_FILE, IDS_FILE.stat().st_size / 1024)


if __name__ == "__main__":
    main()
