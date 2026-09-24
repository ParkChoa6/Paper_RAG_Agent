"""探针：关键词基线 —— 证明「字符串匹配」不够用。

这是一支「反面探针」。它不验证我们写的代码对不对，
它验证的是【我们要解决的问题真的存在】。

做法：给关键词匹配最优条件（把中文问题翻译成英文），
然后看它在真实语料上能不能找到那条真正的答案。

结论会说明为什么 Stage 2 必须用 embedding。

用法:
    python scripts/probe_keyword.py
"""

import json
import re
import sys
from pathlib import Path

# Windows 控制台默认是 gbk，遇到 − （U+2212）、ﬁ 这类字符会直接抛
# UnicodeEncodeError 而不是显示乱码 —— 更难查。统一改成 utf-8。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import PROCESSED_DIR

# ---- 用户的【中文】问题，翻译成英文 ----
# 注意：这一步是关键词方案的前提。翻译本身就得靠模型 ——
# 我们等于先借了一次模型的能力，再来比。
QUERY = "How does RAG incorporate retrieved documents into the generation process"

# 极简停用词。真实的关键词检索会做词干还原、去停用词、TF-IDF 加权，
# 这里只保留最小版本 —— 但已经足够说明问题了。
STOP = {
    "how", "does", "do", "is", "are", "the", "a", "an", "of", "to", "in",
    "into", "and", "or", "with", "for", "on", "by", "that", "this", "it",
    "we", "you", "be", "as", "at", "from", "can",
}

# 那条【真正的答案】
TARGET_ID = "rag-2005.11401#8"


def content_words(text: str) -> set[str]:
    """把文本拆成小写实词集合（去掉停用词和 3 字母以下的短词）。"""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def main() -> None:
    data = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    chunks = data["chunks"]

    qwords = content_words(QUERY)

    print("=" * 72)
    print("关键词基线：给最优条件的一版")
    print("=" * 72)
    print(f"\nquery（已翻译成英文）:\n  {QUERY}")
    print(f"\n拆出的实词 ({len(qwords)} 个): {sorted(qwords)}")

    # ---- 打分：query 实词和 chunk 实词的交集大小 ----
    def hits(text: str) -> set[str]:
        return qwords & content_words(text)

    ranked = sorted(chunks, key=lambda c: -len(hits(c["text"])))

    print()
    print("=" * 72)
    print("匹配结果 Top 5")
    print("=" * 72)
    for i, c in enumerate(ranked[:5], 1):
        h = hits(c["text"])
        print(f"\n{i}. {c['chunk_id']}   命中 {len(h)} 个: {sorted(h)}")
        print(f"   节: {c['section'][:50]}  p{c['page_start']}")
        print(f"   {c['text'][:110]}...")

    # ---- 真正的答案排在哪 ----
    order = [c["chunk_id"] for c in ranked]
    pos = order.index(TARGET_ID) + 1
    target = next(c for c in chunks if c["chunk_id"] == TARGET_ID)
    th = hits(target["text"])

    print()
    print("=" * 72)
    print("那条【真正的答案】排在哪")
    print("=" * 72)
    print(f"\n{TARGET_ID}  →  第 {pos} 名 / 共 {len(chunks)} 条")
    print(f"命中 {len(th)} 个: {sorted(th)}")
    print(f"\n原文:\n  {target['text'][:220]}...")

    # ---- 为什么找不到：把「没重合」的词摆出来 ----
    missing = qwords - content_words(target["text"])
    print()
    print("=" * 72)
    print("为什么找不到")
    print("=" * 72)
    print(f"\nquery 里有、但答案里【一个都没有】的词:")
    print(f"  {sorted(missing)}")
    print()
    print("人话对照:")
    print("  query 说  incorporate（融进）")
    print("  原文说  endow ... with（赋予）        <- 意思一样，字符串毫无关系")
    print()
    print("  唯一稳定命中的词是 'rag' —— 可它是这篇论文的主角，")
    print("  几乎每个 chunk 里都有。命中它等于没命中。")


if __name__ == "__main__":
    main()
