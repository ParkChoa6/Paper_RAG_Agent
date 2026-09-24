"""给嵌入模型打分 —— 在我们自己的语料上。

为什么不直接看 MTEB 排行榜：
    排行榜量的是别人的语料、别人的任务。我们真正要回答的是
    「哪个模型最能把【中文问题】对应的【英文段落】排上来」——
    这个问题只能在自己的数据上量。

量什么：
    Top-1     有多少题，正确段落被排到第 1 位
    MRR       平均倒数排名（越接近 1 越好）—— 排第 1 得 1 分，第 2 得 0.5，第 5 得 0.2
    平均排名   平均要翻几条才找到答案
    对比度     正确段落的平均分 − 全体的中位分
              这一项是看【区分度】：分数挤在一起，排序就不可靠

用法:
    python scripts/eval_embed.py                        # 只测 config.EMB_MODEL
    python scripts/eval_embed.py bge-m3                 # 测指定模型
    python scripts/eval_embed.py 模型A 模型B 模型C       # 对比多个
    python scripts/eval_embed.py --detail bge-m3        # 附带每题明细
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import EMB_MODEL, EVAL_DIR, PROCESSED_DIR
from paper_rag.embed import embed_one, embed_texts, make_client
from paper_rag.retrieve import cosine

# ---- query 侧的指令前缀 ----
#
# 有些嵌入模型是【非对称】的：query 要加一句任务说明，文档【不加】。
#   bge-large-zh-v1.5   BGE 家早期版本要求中文指令
#   qwen3-embedding     Qwen 家要求 "Instruct: ...\nQuery: ..."
#   bge-m3              对称，什么都不加（加了反而伤）
#
# 不写这块的后果是【静默掉点】—— 不报错，分数就是低一点。
# qwen3 官方 README 说漏掉会掉 1~5%，具体多少得自己量。
#
# 注意 key 用【子串】匹配，不是 startswith：
#   "qwen3-embedding:0.6b"              → 能配上
#   "quentinz/bge-large-zh-v1.5:latest" → 只有子串匹配才配得上
# 用 startswith 的话第二条会静默漏掉 —— 不报错，只是那一行分数偏低。
QUERY_PREFIX = {
    "qwen3-embedding": (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query.\nQuery: "
    ),
    "bge-large-zh": "为这个句子生成表示以用于检索相关文章：",
}


def prefix_for(model: str) -> str:
    """查这个模型的 query 前缀，没有就返回空串。"""
    for key, p in QUERY_PREFIX.items():
        if key in model:
            return p
    return ""


def evaluate(model: str, queries: list[dict], chunks: list[dict], detail: bool) -> dict:
    """在评测集上跑一个模型，返回各项指标。

    嵌入和 query 前缀的坑都在 paper_rag/embed.py 里 ——
    这个脚本只负责【打分】，不负责怎么嵌。
    """
    client = make_client()

    print(f"\n{'=' * 74}")
    print(f"模型: {model}")
    print("=" * 74)

    # ---- 1. 把语料全部向量化（一次性，之后所有 query 共用）----
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    vecs, failed = embed_texts(client, model, texts)
    dt = time.time() - t0

    ok_idx = [i for i, v in enumerate(vecs) if v is not None]
    dim = len(vecs[ok_idx[0]]) if ok_idx else 0

    qprefix = prefix_for(model)
    print(f"语料 {len(chunks)} 条 → 成功 {len(ok_idx)}，失败 {len(failed)}，{dt:.1f}s，维度 {dim}")
    print(f"query 前缀: {qprefix!r}" if qprefix else "query 前缀: 无（对称模型）")
    if failed:
        print("  塞不下的 chunk:")
        for i in failed[:5]:
            print(f"    {len(texts[i]):5} 字符  {chunks[i]['chunk_id']}  ({chunks[i]['section'][:30]})")
        if len(failed) > 5:
            print(f"    ... 另有 {len(failed) - 5} 条")

    # chunk_id → 向量，方便查
    vec_of = {chunks[i]["chunk_id"]: vecs[i] for i in ok_idx}

    # ---- 2. 逐题检索 ----
    ranks, target_scores, all_scores = [], [], []

    if detail:
        print(f"\n{'题':<4} {'排名':>5}  {'分数':>8}  问题")
        print("-" * 74)

    for n, item in enumerate(queries, 1):
        # 前缀【只加在 query 上】，文档侧不加 —— 这就是「非对称检索」
        qvec = embed_one(client, model, qprefix + item["q"])

        scored = []
        for cid, v in vec_of.items():
            s = cosine(qvec, v)
            scored.append((s, cid))
        scored.sort(reverse=True, key=lambda t: t[0])

        # 每题可能有多个正确答案，取排得最靠前的那个
        rank = None
        for pos, (s, cid) in enumerate(scored, 1):
            if cid in item["targets"]:
                rank, best_score = pos, s
                break

        if rank is None:
            if detail:
                print(f"{n:<4} {'?':>5}  {'?':>8}  {item['q']}")
            ranks.append(len(scored) + 1)
            continue

        ranks.append(rank)
        target_scores.append(best_score)
        all_scores.extend(s for s, _ in scored)

        if detail:
            hit = "OK" if rank == 1 else "  "
            print(f"{n:<4} {rank:>4}{hit}  {best_score:+.4f}  {item['q']}")

    # ---- 3. 汇总 ----
    n_q = len(queries)
    top1 = sum(1 for r in ranks if r == 1) / n_q
    mrr = sum(1 / r for r in ranks) / n_q
    avg_rank = sum(ranks) / n_q
    contrast = statistics.mean(target_scores) - statistics.median(all_scores)

    return {
        "model": model,
        "dim": dim,
        "failed": len(failed),
        "top1": top1,
        "mrr": mrr,
        "avg_rank": avg_rank,
        "target_score": statistics.mean(target_scores),
        "corpus_median": statistics.median(all_scores),
        "contrast": contrast,
        "ranks": ranks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None, help="要测的模型名，不传则用 config.EMB_MODEL")
    ap.add_argument("--detail", action="store_true", help="打印每题明细")
    args = ap.parse_args()

    models = args.models or [EMB_MODEL]

    eval_set = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = eval_set["queries"]
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]

    print(f"评测集: {len(queries)} 题  语料: {len(chunks)} 条")

    results = []
    for m in models:
        try:
            results.append(evaluate(m, queries, chunks, args.detail))
        except Exception as e:
            print(f"\n[ERROR] {m}: {type(e).__name__}: {e}")

    if len(results) < 1:
        return

    # ---- 对比表 ----
    print(f"\n{'=' * 74}")
    print("对比")
    print("=" * 74)
    print(f"{'模型':<34}{'维度':>6}{'塞不下':>7}{'Top-1':>8}{'MRR':>7}{'排名':>7}{'对比度':>9}")
    print("-" * 74)
    for r in results:
        name = r["model"] if len(r["model"]) <= 32 else r["model"][:31] + "…"
        print(
            f"{name:<34}{r['dim']:>6}{r['failed']:>7}"
            f"{r['top1'] * 100:>7.0f}%{r['mrr']:>7.3f}{r['avg_rank']:>7.1f}{r['contrast']:>9.3f}"
        )
    print("-" * 74)
    print("Top-1 = 正确段落排第 1 的题数占比    MRR = 平均倒数排名（越高越好）")
    print("排名  = 平均要翻几条才找到答案（越低越好）")
    print("对比度 = 正确段落平均分 − 全体中位分（越大 = 区分度越好）")


if __name__ == "__main__":
    main()
