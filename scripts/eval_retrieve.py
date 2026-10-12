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
    空结果     有几题【一条都没返回】—— 见下面「为什么单列这一项」

用法:
    python scripts/eval_retrieve.py --detail bge-m3
    python scripts/eval_retrieve.py --kind bm25
    python scripts/eval_retrieve.py --kind dense,bm25            # 两种摆一张表
    python scripts/eval_retrieve.py --kind dense,bm25 --expand   # ★ 开查询扩展
    python scripts/eval_retrieve.py --kind dense --k 10 bge-m3

── 这是【产品】还是【脚手架】？──

    脚手架。它不产出任何被别处 import 的东西，只负责：
    读题 → 调检索器 → 算指标 → 打表格 → 退出。
    真正可复用的那部分是 paper_rag/retriever.py。

── ★ 设计一：按【全量排名】算，不按 top-k 算 ──

    search() 是能只返回 k 条的，但这里让它返回【全部】。
    理由：排名（rank）这个量在只拿到 top-5 的时候是【截断】的 ——
    正确段落在第 40 名时，你只知道「不在前 5」，算不出 MRR。

    所以这里一次拿全量，recall@k 再从前缀切片算出来。

── ★ 设计二：为什么要把「空结果」单列一项 ──

    dense 永远返回 min(k, N) 条 —— 它总能给 N 个余弦值。
    BM25 不是：query 里的词一个都没出现过，它就【返回空】。

    语料是英文、题目是中文，于是：

        "静态知识图谱推理的模型按技术分成哪三类？"
            tokenize → []      ← 一个 ASCII 词都没有
            BM25     → []      ← 什么都找不到

    这不是 bug，是 BM25 的【诚实回答】。但必须让它可见 ——
    如果只看汇总的 MRR，你只会看到"BM25 分低"，看不出
    「它在 7 道题上完全失明，在另外 5 道题上其实还行」。

    ★ 平均分会把这两种情况糊成一个数。分开看才知道该往哪修。

── ★ 设计三：--expand ──

    config.EXPAND_ENABLED 默认是 True，也就是说【生产是开着扩展的】，
    而 eval_embed.py 从来没开过。两边量的一直不是同一个 query。

    扩展对两个检索器的意义完全不同：
        dense  加分项（bge-m3 多语言，不扩展也能跨语言）
        BM25   【生死线】（不扩展 = 空 query = 0 分）

    所以它必须是一个能被单独开关的变量，不能写死。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import EMB_MODEL, EVAL_DIR, PROCESSED_DIR, TOP_K
from paper_rag.embed import make_client
from paper_rag.pipeline import expand_query
from paper_rag.retriever import BM25Retriever, DenseRetriever, Retriever


def make_retriever(kind: str, model: str | None, client, chunks: list[dict]) -> Retriever:
    """按名字造一个检索器。

    ★ 注意 bm25 分支【不需要 model 也不需要 client】——
      这是 S5-1 定接口时的那个判断的兑现：
      最不像的那个实现，真的进来。
    """
    if kind == "dense":
        return DenseRetriever.from_chunks(client, model, chunks)
    if kind == "bm25":
        return BM25Retriever.from_chunks(chunks)

    raise SystemExit(
        f"还不认识这种检索器: {kind!r}\n"
        f"目前有 dense / bm25。rrf / rerank 是 Stage 5 后面几步的事。"
    )


def evaluate(
    rt: Retriever,
    queries: list[dict],
    all_k: int,
    k: int,
    *,
    expand: bool,
    detail: bool,
) -> dict:
    """在一个检索器上跑完整个评测集，返回各项指标。

    参数:
        rt:      检索器
        queries: 评测集，每题有 "q" 和 "targets"（可多个正确答案）
        all_k:   一次取回多少条 —— 传语料总数，也就是拿【全量排名】
        k:       算 recall@k 用的 k
        expand:  要不要先做查询扩展（★ 生产是开的，见模块开头设计三）
        detail:  打不打每题明细
    """
    tag = " +expand" if expand else ""
    print(f"\n{'=' * 74}")
    print(f"检索器: {rt.name}{tag}")
    print("=" * 74)

    ranks: list[int] = []
    hits_at_k = 0
    empty = 0

    if detail:
        print(f"{'题':<4}{'排名':>7}  {'分数':>9}  问题")
        print("-" * 74)

    t0 = time.time()
    for n, item in enumerate(queries, 1):
        q = expand_query(item["q"]) if expand else item["q"]
        scored = rt.search(q, k=all_k)

        if not scored:
            empty += 1

        # 每题可能有多个正确答案，取排得最靠前的那个
        rank = None
        best_score = None
        for pos, (cid, score) in enumerate(scored, 1):
            if cid in item["targets"]:
                rank, best_score = pos, score
                break

        if rank is None:
            # ★ 找不到 —— 记成「比最后一名还差」，也就是 all_k + 1。
            #
            # 【不能】写成 len(scored) + 1。
            # scored 有可能是【空的】（BM25 碰上纯中文 query），
            # 那时 len(scored) + 1 == 1 —— 会被当成第 1 名，
            # MRR 白送 1.0 分，而且【不报错】。
            # 这是个安静的错误，正好会偏袒「什么都不返回」的检索器。
            rank = all_k + 1
            if detail:
                mark = "空" if not scored else "?"
                print(f"{n:<4}{mark:>7}  {'—':>9}  {item['q']}")
            ranks.append(rank)
            continue

        ranks.append(rank)
        hits_at_k += rank <= k

        if detail:
            flag = " OK" if rank == 1 else "   "
            print(f"{n:<4}{rank:>6}{flag}  {best_score:>+9.4f}  {item['q']}")

    dt = time.time() - t0

    n_q = len(queries)
    return {
        "name": rt.name + tag,
        "top1": sum(1 for r in ranks if r == 1) / n_q,
        "mrr": sum(1 / r for r in ranks) / n_q,
        "avg_rank": sum(ranks) / n_q,
        "recall_at_k": hits_at_k / n_q,
        "k": k,
        "empty": empty,
        "ranks": ranks,
        "secs": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None,
                    help="嵌入模型名，不传则用 config.EMB_MODEL（--kind bm25 时忽略）")
    ap.add_argument("--kind", default="dense",
                    help="检索器种类，逗号分隔：dense,bm25")
    ap.add_argument("--k", type=int, default=TOP_K,
                    help=f"算 recall@k 用的 k（默认 config.TOP_K = {TOP_K}）")
    ap.add_argument("--expand", action="store_true",
                    help="先做查询扩展（★ 生产默认是开的，所以这个才更接近线上）")
    ap.add_argument("--detail", action="store_true", help="打印每题明细")
    args = ap.parse_args()

    kinds = [x.strip() for x in args.kind.split(",") if x.strip()]
    models = args.models or [EMB_MODEL]

    eval_set = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = eval_set["queries"]
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]

    print(f"评测集: {len(queries)} 题   语料: {len(chunks)} 条   检索器: {', '.join(kinds)}")
    print(f"查询扩展: {'开' if args.expand else '关'}")

    client = make_client()
    results = []

    # bm25 和嵌入模型无关，所以它在 (kind, model) 这个网格里只有一格。
    jobs: list[tuple[str, str | None]] = []
    for kind in kinds:
        if kind == "bm25":
            if args.models:
                print(f"（--kind bm25 忽略模型参数: {args.models}）")
            jobs.append((kind, None))
        else:
            jobs.extend((kind, m) for m in models)

    for kind, m in jobs:
        try:
            t0 = time.time()
            rt = make_retriever(kind, m, client, chunks)
            print(f"\n[{kind}{':' + m if m else ''}] 建索引 {time.time() - t0:.1f}s → {rt.name}")
            results.append(evaluate(
                rt, queries, all_k=len(chunks), k=args.k,
                expand=args.expand, detail=args.detail,
            ))
        except Exception as e:
            print(f"\n[ERROR] {kind} {m}: {type(e).__name__}: {e}")

    if not results:
        return

    # ---- 对比表 ----
    print(f"\n{'=' * 74}")
    print("对比")
    print("=" * 74)
    print(f"{'检索器':<36}{'Top-1':>8}{'MRR':>8}{'排名':>8}{f'recall@{args.k}':>11}{'空结果':>8}")
    print("-" * 74)
    for r in results:
        name = r["name"] if len(r["name"]) <= 34 else r["name"][:33] + "…"
        print(
            f"{name:<36}{r['top1'] * 100:>7.0f}%{r['mrr']:>8.3f}"
            f"{r['avg_rank']:>8.1f}{r['recall_at_k'] * 100:>10.0f}%{r['empty']:>8}"
        )
    print("-" * 74)
    print("Top-1    = 正确段落排第 1 的题数占比")
    print("MRR      = 平均倒数排名（越高越好）")
    print("排名     = 平均要翻几条才找到答案（越低越好）")
    print(f"recall@{args.k} = 正确段落在前 {args.k} 条里的题数占比")
    print("空结果   = 【一条都没返回】的题数 —— 不是排得靠后，是完全失明")
    print()
    print("⚠️ 这些数字只有在【语料和题都一样】时才能互相比较。")
    print("   加过论文之后，老数字作废 —— 干扰项多了，排名本来就会动。")


if __name__ == "__main__":
    main()
