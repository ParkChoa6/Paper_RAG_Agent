"""探针：验收手写的 dot / norm / cosine。

三关，一关比一关狠：

  第 1 关 · 手算得出来的小向量
      用 2 维向量，答案拿纸笔就能算。对不上就是公式写错了。

  第 2 关 · 跟 numpy 对拍
      同样的输入，你的实现和 numpy 出来的数一不一样。
      对拍是工程上最省事的验证手段 —— 不需要你自己知道答案。

  第 3 关 · 真数据 · 中文问题 vs 英文语料
      【这就是这一整章的意义所在】。
      probe_keyword.py 里关键词匹配把真答案排到了第 5 / 第 15 名。
      现在看着向量能不能把它顶到第 1。

用法:
    python scripts/probe_similarity.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import ollama

from paper_rag.config import EMB_MODEL, OLLAMA_HOST, PROCESSED_DIR
from paper_rag.retrieve import cosine, dot, norm

client = ollama.Client(host=OLLAMA_HOST)

LINE = "=" * 72


def embed(text: str) -> list[float]:
    return client.embeddings(model=EMB_MODEL, prompt=text).embedding


def check(label: str, got: float, want: float, tol: float = 1e-6) -> bool:
    ok = abs(got - want) <= tol
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label:38} 得到 {got:+.6f}   期望 {want:+.6f}")
    return ok


def main() -> None:
    passed = 0
    total = 0

    # ══════════════ 第 1 关 · 手算得出来的小向量 ══════════════
    print(LINE)
    print("第 1 关 · 手算得出来的小向量")
    print(LINE)

    print("\n-- dot --")
    total += 2
    passed += check("dot([1,2,3], [4,5,6]) = 4+10+18", dot([1, 2, 3], [4, 5, 6]), 32.0)
    passed += check("dot([1,0], [0,1]) = 0", dot([1, 0], [0, 1]), 0.0)

    print("\n-- norm --")
    total += 2
    passed += check("norm([3,4]) = 5  (勾股)", norm([3, 4]), 5.0)
    passed += check("norm([1,0,0]) = 1", norm([1, 0, 0]), 1.0)

    print("\n-- cosine（三个方向，纸笔可验）--")
    total += 3
    passed += check("同向  cos([1,0],[2,0])", cosine([1, 0], [2, 0]), 1.0)
    passed += check("垂直  cos([1,0],[0,1])", cosine([1, 0], [0, 1]), 0.0)
    passed += check("反向  cos([1,0],[-1,0])", cosine([1, 0], [-1, 0]), -1.0)

    print("\n-- 关键性质：只跟方向有关，跟长度无关 --")
    total += 1
    passed += check(
        "cos([1,1], [100,100]) 仍是 1",
        cosine([1, 1], [100, 100]),
        1.0,
    )

    # ══════════════ 第 2 关 · 跟 numpy 对拍 ══════════════
    print()
    print(LINE)
    print("第 2 关 · 跟 numpy 对拍（用真实向量，1024 维）")
    print(LINE)

    a = embed("Retrieval-Augmented Generation")
    b = embed("I would like a cup of coffee")

    np_dot = float(np.dot(a, b))
    np_norm_a = float(np.linalg.norm(a))
    np_cos = np_dot / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\n  向量维度: {len(a)}")
    print()
    total += 3
    passed += check("dot   vs numpy", dot(a, b), np_dot, tol=1e-4)
    passed += check("norm  vs numpy", norm(a), np_norm_a, tol=1e-4)
    passed += check("cosine vs numpy", cosine(a, b), np_cos, tol=1e-6)

    # ══════════════ 第 3 关 · 中文问题找英文答案 ══════════════
    print()
    print(LINE)
    print("第 3 关 · 真数据 —— 中文问题，英文语料")
    print(LINE)

    data = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
    chunks = {c["chunk_id"]: c for c in data["chunks"]}

    # 挑 5 条候选：一条真答案，四条是关键词匹配会捞上来的
    ref_id = next(c["chunk_id"] for c in data["chunks"] if c["section"] == "References")
    candidate_ids = [
        "rag-2005.11401#8",    # ← 真答案（关键词匹配排第 5 / 加权后排第 15）
        "rag-2005.11401#16",   # ← 关键词匹配的第 1 名
        "rag-2005.11401#53",   # ← 图注
        "rag-2005.11401#222",  # ← 超参数细节
        ref_id,                # ← 参考文献条目（凑数的）
    ]

    QUERY = "RAG 是怎么把检索到的文档融进生成过程的？"
    qvec = embed(QUERY)

    print(f"\n中文 query: {QUERY}\n")
    print("候选（按余弦相似度排序）:\n")

    scored = []
    for cid in candidate_ids:
        c = chunks[cid]
        score = cosine(qvec, embed(c["text"]))
        scored.append((score, cid, c))
    scored.sort(reverse=True, key=lambda t: t[0])

    for rank, (score, cid, c) in enumerate(scored, 1):
        star = "  <-- 真答案" if cid == "rag-2005.11401#8" else ""
        print(f"  {rank}. cos={score:+.4f}  [{cid}]  {c['section'][:28]}{star}")
        print(f"     {c['text'][:88]}...")

    best = scored[0][1]
    print()
    if best == "rag-2005.11401#8":
        print("  >>> 真答案排到了第 1 名。")
    else:
        print(f"  >>> 第 1 名是 {best}，不是真答案 —— 需要检查。")

    # ══════════════ 总结 ══════════════
    print()
    print(LINE)
    print(f"前两关: {passed}/{total} 通过")
    print(LINE)


if __name__ == "__main__":
    main()
