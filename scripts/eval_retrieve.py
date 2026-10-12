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

from paper_rag.config import EMB_MODEL, EVAL_DIR, PROCESSED_DIR, RETRIEVE_POOL, TOP_K
from paper_rag.embed import make_client
from paper_rag.pipeline import expand_query
from paper_rag.rerank import CrossEncoderReranker, RerankedRetriever
from paper_rag.retriever import BM25Retriever, DenseRetriever, Retriever


def make_retriever(
    kind: str,
    model: str | None,
    client,
    chunks: list[dict],
    *,
    pool: int = RETRIEVE_POOL,
    reranker=None,
) -> Retriever:
    """按名字造一个检索器。

    ★ 注意 bm25 分支【不需要 model 也不需要 client】——
      这是 S5-1 定接口时的那个判断的兑现：
      最不像的那个实现，真的进来。

    ★ 而 rerank 分支证明了【另一个】判断：RerankedRetriever 满足 Retriever，
      所以它上面那三行（dense/bm25）【一行都没改】就多了一个成员。
      evaluate() 更是完全不知道有这回事。

    参数:
        reranker: 已经建好的重排器。传进来复用 ——
                  重排模型 2.3GB，每格重建一次要重下/重载。
                  传 None 就自己建一个。
    """
    if kind == "dense":
        return DenseRetriever.from_chunks(client, model, chunks)
    if kind == "bm25":
        return BM25Retriever.from_chunks(chunks)
    if kind == "rerank":
        # ★ 粗排用 dense。为什么不是 bm25？——见模块开头「设计三」和 S5-3 的结论：
        #   BM25 单独只有 32%/54%，当粗排会让候选池质量更差。
        #   要试「bm25 粗排 + rerank」是另一个实验，得单独加一个 kind。
        base = DenseRetriever.from_chunks(client, model, chunks)
        if reranker is None:
            reranker = CrossEncoderReranker.from_chunks(chunks)
        return RerankedRetriever(base, reranker, pool=pool, k=TOP_K)

    raise SystemExit(
        f"还不认识这种检索器: {kind!r}\n"
        f"目前有 dense / bm25 / rerank。rrf 在 S5-3 被实测否决，不做了。"
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
    not_in_list = 0

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
            # ★★ 「找不到」其实是【两种】病，修法完全不同，所以要分开数：
            #     空       —— 检索器一条都没返回（query 一个词都没匹配上）
            #     未进候选 —— 返回了东西，但目标压根【不在它返回的列表里】
            #
            #   对 BM25：未进候选 = 目标段落和 query 零词面重叠（没进倒排表的池子）
            #   对 rerank：未进候选 = 目标不在粗排捞的那 pool 条里
            #              ★ 这一列就是 rerank 的【天花板】：
            #                它没进候选，重排再神也捞不上来。
            #   对 dense：几乎永远是 0（它排全语料，什么都"在列表里"）
            #
            #   三种情况的共同语义是同一句话：
            #   【这个检索器压根没机会找到它】—— 和"排得靠后"是两回事。
            if scored:
                not_in_list += 1
            else:
                empty += 1
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
        "not_in_list": not_in_list,
        "ranks": ranks,
        "secs": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None,
                    help="嵌入模型名，不传则用 config.EMB_MODEL（--kind bm25 时忽略）")
    ap.add_argument("--kind", default="dense",
                    help="检索器种类，逗号分隔：dense,bm25,rerank")
    ap.add_argument("--k", type=int, default=TOP_K,
                    help=f"算 recall@k 用的 k（默认 config.TOP_K = {TOP_K}）")
    ap.add_argument("--expand", action="store_true",
                    help="先做查询扩展（★ 生产默认是开的，所以这个才更接近线上）")
    ap.add_argument("--pool", default=str(RETRIEVE_POOL),
                    help=f"rerank 粗排捞多少条。逗号分隔可做扫描。"
                         f"（默认 config.RETRIEVE_POOL = {RETRIEVE_POOL}）")
    ap.add_argument("--detail", action="store_true", help="打印每题明细")
    args = ap.parse_args()

    kinds = [x.strip() for x in args.kind.split(",") if x.strip()]
    models = args.models or [EMB_MODEL]
    pools = [int(x) for x in str(args.pool).split(",") if x.strip()]

    eval_set = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = eval_set["queries"]
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]

    print(f"评测集: {len(queries)} 题   语料: {len(chunks)} 条   检索器: {', '.join(kinds)}")
    print(f"查询扩展: {'开' if args.expand else '关'}")

    client = make_client()
    results = []

    # ★ 重排器【只建一个】，所有 pool 共用。
    #   模型 2.3GB，建两个对象 = 内存里两份 + 加载两次。
    #   （懒加载在 CrossEncoderReranker 里，所以建对象本身不贵；
    #     真正贵的是 _ensure_model 那一下，第一次 rerank 才触发。）
    reranker = None
    if "rerank" in kinds:
        reranker = CrossEncoderReranker.from_chunks(chunks)
        print(f"重排器: {reranker.name}  ← 模型第一次用到时才加载（首次要下约 2.3GB）")

    # bm25 和嵌入模型无关，所以它在 (kind, model) 这个网格里只有一格。
    # rerank 再乘一个 pool 维度 —— ★ pool 是【系统的一部分】，不是调参旋钮：
    # 同一份权重、同一批题，pool=5 和 pool=20 是两个不同的系统。
    jobs: list[tuple[str, str | None, int | None]] = []
    for kind in kinds:
        if kind == "bm25":
            if args.models:
                print(f"（--kind bm25 忽略模型参数: {args.models}）")
            jobs.append((kind, None, None))
        elif kind == "rerank":
            jobs.extend((kind, m, p) for m in models for p in pools)
        else:
            jobs.extend((kind, m, None) for m in models)

    for kind, m, p in jobs:
        try:
            t0 = time.time()
            rt = make_retriever(kind, m, client, chunks,
                                pool=p if p is not None else RETRIEVE_POOL,
                                reranker=reranker)
            label = kind + (":" + m if m else "") + (f":pool{p}" if p is not None else "")
            print(f"\n[{label}] 建索引 {time.time() - t0:.1f}s → {rt.name}")
            results.append(evaluate(
                rt, queries, all_k=len(chunks), k=args.k,
                expand=args.expand, detail=args.detail,
            ))
        except Exception as e:
            print(f"\n[ERROR] {kind} {m} pool={p}: {type(e).__name__}: {e}")

    if not results:
        return

    # ---- 对比表 ----
    print(f"\n{'=' * 86}")
    print("对比")
    print("=" * 86)
    # ★ 名字列加宽了：重排器的名字长（`dense:...:raw > ce:...:cpu @pool20`），
    #   截断会把两个不同 pool 的行截成一样 —— 那张表就是骗人的。
    print(f"{'检索器':<42}{'Top-1':>7}{'MRR':>8}{'排名':>8}"
          f"{f'recall@{args.k}':>11}{'空':>5}{'未进候选':>9}")
    print("-" * 86)
    for r in results:
        name = r["name"] if len(r["name"]) <= 40 else r["name"][:39] + "…"
        print(
            f"{name:<42}{r['top1'] * 100:>6.0f}%{r['mrr']:>8.3f}"
            f"{r['avg_rank']:>8.1f}{r['recall_at_k'] * 100:>10.0f}%"
            f"{r['empty']:>5}{r['not_in_list']:>9}"
        )
    print("-" * 86)
    print("Top-1    = 正确段落排第 1 的题数占比")
    print("MRR      = 平均倒数排名（越高越好）")
    print("排名     = 平均要翻几条才找到答案（越低越好）")
    print(f"recall@{args.k} = 正确段落在前 {args.k} 条里的题数占比")
    print("空       = 【一条都没返回】的题数（query 一个词都没匹配上）")
    print("未进候选 = 返回了东西，但目标【不在返回的列表里】的题数")
    print("           ★ rerank 的天花板就是这一列：没进粗排候选池的，重排捞不上来")
    print()
    print("⚠️⚠️ 「排名」这一列在【未进候选 > 0】时会被拉高，而且不是变差了：")
    print("   每一道未进候选的题都按 all_k+1 =", len(chunks) + 1, "名计。")
    print("   对 dense 这无所谓（它排全语料，未进候选几乎恒为 0），")
    # ★ 这里原来写死的是一句「rerank@5 有 8 题未进候选」。
    #   两个毛病：
    #     ① 它是【硬编码的】，跑别的配置时就成了假话；
    #     ② 更要命的是它和【正上方那张表】对不上 —— 表是本次真实结果，
    #        尾巴是个常数。**表是活的、解释是死的**，看的人反而会信尾巴。
    #   所以改成从 results 里取本次的真实数字。
    #   ★ 和 S5-5 那条教训是同一个病：别写没量过的断言。
    worst = max(results, key=lambda r: r["not_in_list"])
    if worst["not_in_list"]:
        print(f"   对 BM25 / rerank 就不一样了 —— 本次 {worst['name']} 有 "
              f"{worst['not_in_list']} 题未进候选，")
        print("   单这一项就把它那一列的均值抬高几百名，而它的 Top-1 可能是好的。")
    else:
        print("   对 BM25 / rerank 就不一样了 —— 本次没有未进候选的题，这一列暂时还算干净。")
    print("   ★ 所以：看重排要看 Top-1 和 recall@k，【不要】拿它的「排名」去和 dense 比。")
    print("     想干净地比排名，得只统计「找到了的那些题」—— 那是另一张表，本脚本没做。")
    print()
    print("⚠️ 这些数字只有在【语料、题、以及检索器的配置】都一样时才能互相比较。")
    print("   加过论文之后老数字作废；改了 pool 之后也是 —— pool 是系统的一部分。")


if __name__ == "__main__":
    main()
