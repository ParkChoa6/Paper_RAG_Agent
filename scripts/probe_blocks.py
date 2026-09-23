import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf
from paper_rag.config import RAW_DIR

doc = pymupdf.open(RAW_DIR / "rag-original-2005.11401.pdf")
page = doc[0]

blocks = page.get_text("blocks")

for i, block in enumerate(blocks):
    x0, y0, x1, y1, text, *_ = block  # 解包，只要前 5 个
    preview = text[:40]  # 切片：取前 40 字
    preview = preview.replace("\n", " ")  # 换行会让输出乱掉，压成空格
    print(f"[{i}] 字数={len(text):5d}  x={x0:6.1f}~{x1:6.1f}  y={y0:6.1f}~{y1:6.1f}")
    print(f"     {preview}")
