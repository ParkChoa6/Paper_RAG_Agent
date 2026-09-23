"""探针：扫全文，统计「字号 → 字符数」。

用途：找出全文的「正文字号」（众数），以此决定标题判定的阈值。
不是产品代码，跑完看结果就行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf
from paper_rag.config import RAW_DIR


def scan_size_histogram(doc) -> dict:
    """扫全文，统计每个字号覆盖了多少字符。

    按 span 为单位数 —— 因为块内字号会抖动（9.9/10.0/10.1），
    按块统计会把它们糊在一起，看不出真实分布。

    参数:
        doc: pymupdf 打开的文档

    返回:
        {字号: 字符数}，字号已四舍五入到 1 位小数
    """
    hist = {}  # 累加器：字号 → 累计字符数

    for page in doc:
        d = page.get_text("dict")
        for blk in d["blocks"]:
            if blk["type"] != 0:  # 1 = 图片块，没有 lines
                continue
            for line in blk["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    hist[size] = hist.get(size, 0) + len(span["text"])

    return hist


if __name__ == "__main__":
    doc = pymupdf.open(RAW_DIR / "rag-original-2005.11401.pdf")
    hist = scan_size_histogram(doc)
    total = sum(hist.values())

    print(f"全文总字符数：{total}\n")
    print(f"{'字号':>6}  {'字符数':>9}  {'占比':>7}  分布")
    print("-" * 78)

    for size in sorted(hist):
        n = hist[size]
        bar = "█" * int(n / total * 60)  # 最长 60 格
        print(f"{size:6.1f}  {n:9d}  {n / total * 100:6.1f}%  {bar}")
