import sys
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import RAW_DIR

doc = pymupdf.open(RAW_DIR / "rag-original-2005.11401.pdf")

print(doc.page_count)  # 打印页数

for page in doc:  # 遍历每一页
    print(f"第 {page.number} 页 | {len(page.get_text())} 字符")

# 打印第一页原始内容
print(doc[0].get_text())
