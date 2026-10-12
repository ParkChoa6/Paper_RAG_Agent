"""给【检索器】打分 —— 在同一套题上比不同的找法。

和 eval_embed.py 的分工：
    eval_embed.py      固定「向量检索」，换【嵌入模型】比  ← 换模型时跑
    eval_retrieve.py   固定题和语料，换【检索策略】比      ← 换找法时跑  ← 本文件

两条轴不能混在一起测，否则分数动了你不知道是模型的锅还是策略的锅。

量什么（和 eval_embed 用【同一套定义】，数字才能互相对）：
    Top-1      有多少题，正确段落被排到第 1 位
    MRR        平均倒数排名 —— 第 1 名得 1 分，第 2 名 0.5，第 5 名 0.2
    平均排名    平均要翻几条才找到答案
    recall@k   正确段落有没有出现在前 k 条里（k 默认取 config.TOP_K）

用法:
    python scripts/eval_retrieve.py                          # dense + config.EMB_MODEL
    python scripts/eval_retrieve.py bge-m3                    # 指定模型
    python scripts/eval_retrieve.py bge-m3 qwen3-embedding:0.6b   # 对比多个
    python scripts/eval_retrieve.py --detail bge-m3           # 附每题明细
    python scripts/eval_retrieve.py --kind dense --k 10 bge-m3

── 这是【产品】还是【脚手架】？──

    脚手架。它不产出任何被别处 import 的东西，只负责：
    读题 → 调检索器 → 算指标 → 打表格 → 退出。
    真正可复用的那部分是 paper_rag/retriever.py。

── ★ 一个刻意的设计：按【全量排名】算，不按 top-k 算 ──

    search() 是能只返回 k 条的，但这里让它返回【全部 N 条】。
    理由：排名（rank）这个量在只拿到 top-5 的时候是【截断】的 ——
    正确段落在第 40 名时，你只知道「不在前 5」，算不出 MRR。

    所以这里一次拿全量，recall@k 再从前缀切片算出来。
    代价是每题多排 N 条，对于几百条语料可以忽略。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import EMB_MODEL, EVAL_DIR, PROCESSED_DIR, TOP_K
from paper_rag.embed import make_client
from paper_rag.retriever import DenseRetriever, Retriever


def make_retriever(kind: str, model: str, client, chunks: list[dict]) -> Retriever:
    """按名字造一个检索器。

    Stage 5 会往这里加分支：
        bm25   →  BM25Retriever（下一步）
        rrf    →  RRFRetriever(dense, bm25)
        rerank →  RerankRetriever(rrf, cross-encoder)

    现在只有 dense 一种。★ 一个分支都没有的「工厂函数」是过度设计，
    但【两个】分支就值得了 —— 所以这里等第二步落地时再抽。
    """
    if kind == "dense":
        return DenseRetriever.from_chunks(client, model, chunks)

    raise SystemExit(
        f"还不认识这种检索器: {kind!r}\n"
        f"目前只有 dense。bm25 / rrf / rerank 是 Stage 5 后面几步的事。"
    )


def evaluate(rt: Retriever, queries: list[dict], all_k: int, k: int, detail: bool) -> dict:
    """在一个检索器上跑完整个评测集，返回各项指标。

    参数:
        rt:     检索器
        queries: 评测集，每题有 "q" 和 "targets"（可多个正确答案）
        all_k:  一次取回多少条 —— 传语料总数，也就是拿【全量排名】
        k:      算 recall@k 用的 k
        detail: 打不打每题明细
    """
    print(f"\n{'=' * 74}")
    print(f"检索器: {rt.name}")
    print("=" * 74)

    ranks: list[int] = []
    hits_at_k = 0

    if detail:
        print(f"{'题':<4}{'排名':>6}  {'分数':>9}  问题")
        print("-" * 74)

    t0 = time.time()
    for n, item in enumerate(queries, 1):
        scored = rt.search(item["q"], k=all_k)

        # 每题可能有多个正确答案，取排得最靠前的那个
        rank = None
        best_score = None
        for pos, (cid, score) in enumerate(scored, 1):
            if cid in item["targets"]:
                rank, best_score = pos, score
                break

        if rank is None:
            # 一条都没捞到 —— 记成「比最后一名还差」，不是记 0
            # （记 0 会让 MRR 除零，记 len+1 才对得上 eval_embed 的口径）
            rank = len(scored) + 1
            if detail:
                print(f"{n:<4}{'?':>6}  {'?':>9}  {item['q']}")
            ranks.append(rank)
            continue

        ranks.append(rank)
        hits_at_k += rank <= k

        if detail:
            flag = " OK" if rank == 1 else "   "
            print(f"{n:<4}{rank:>5}{flag}  {best_score:>+9.4f}  {item['q']}")

    dt = time.time() - t0

    n_q = len(queries)
    return {
        "name": rt.name,
        "top1": sum(1 for r in ranks if r == 1) / n_q,
        "mrr": sum(1 / r for r in ranks) / n_q,
        "avg_rank": sum(ranks) / n_q,
        "recall_at_k": hits_at_k / n_q,
        "k": k,
        "ranks": ranks,
        "secs": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None,
                    help="嵌入模型名，不传则用 config.EMB_MODEL")
    ap.add_argument("--kind", default="dense",
                    help="检索器种类（目前只有 dense）")
    ap.add_argument("--k", type=int, default=TOP_K,
                    help=f"算 recall@k 用的 k（默认 config.TOP_K = {TOP_K}）")
    ap.add_argument("--detail", action="store_true", help="打印每题明细")
    args = ap.parse_args()

    models = args.models or [EMB_MODEL]

    eval_set = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = eval_set["queries"]
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]

    print(f"评测集: {len(queries)} 题   语料: {len(chunks)} 条   检索器: {args.kind}")

    client = make_client()
    results = []

    for m in models:
        try:
            t0 = time.time()
            rt = make_retriever(args.kind, m, client, chunks)
            print(f"\n[{m}] 语料向量化 {time.time() - t0:.1f}s → {rt.name}")
            results.append(evaluate(rt, queries, all_k=len(chunks), k=args.k, detail=args.detail))
        except Exception as e:
            print(f"\n[ERROR] {m}: {type(e).__name__}: {e}")

    if not results:
        return

    # ---- 对比表 ----
    print(f"\n{'=' * 74}")
    print("对比")
    print("=" * 74)
    print(f"{'检索器':<38}{'Top-1':>8}{'MRR':>8}{'排名':>8}{f'recall@{args.k}':>11}")
    print("-" * 74)
    for r in results:
        name = r["name"] if len(r["name"]) <= 36 else r["name"][:35] + "…"
        print(
            f"{name:<38}{r['top1'] * 100:>7.0f}%{r['mrr']:>8.3f}"
            f"{r['avg_rank']:>8.1f}{r['recall_at_k'] * 100:>10.0f}%"
        )
    print("-" * 74)
    print("Top-1    = 正确段落排第 1 的题数占比")
    print("MRR      = 平均倒数排名（越高越好）")
    print("排名     = 平均要翻几条才找到答案（越低越好）")
    print(f"recall@{args.k} = 正确段落在前 {args.k} 条里的题数占比")
    print()
    print("⚠️ 这些数字只有在【语料和题都一样】时才能互相比较。")
    print("   加过论文之后，老数字作废 —— 干扰项多了，排名本来就会动。")


if __name__ == "__main__":
    main()
