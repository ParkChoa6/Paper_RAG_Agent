import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from paper_rag.config import RAW_DIR
from paper_rag.ingest import paragraphs_to_chunks

# doc 是 paragraphs_to_chunks 的第一个参数 —— 调用方负责打开它。
# 库函数不碰磁盘路径，路径是入口的事。
doc = pymupdf.open(RAW_DIR / "rag-original-2005.11401.pdf")

chunks = paragraphs_to_chunks(
    doc,
    paper_id="rag-2005.11401",
    title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    source_path="data/raw/rag-original-2005.11401.pdf",
)

print(f"共 {len(chunks)} 个 chunk\n")  # 期望：387 - 40 = 347
for c in chunks[:5]:
    print(c.format()[:200], "\n")

# 跨页检查：2 Methods 在第 1 页，2.1 Models 在第 2 页
for c in chunks:
    if c.section.startswith("2.1"):
        print(f"✅ 2.1 的第一个 chunk section = {c.section!r}, 页码 = {c.pages()}")
        break
