"""端到端入口 —— 问一句话，拿一个带引用的答案。

它是【脚手架】，不是产品：只负责把参数喂进 pipeline.ask()，
再把结果打印出来。这里不该有任何判断逻辑 ——
一旦你在这儿写了一句 `if status == ...` 来做业务决策，
那就说明那个决策本该在 paper_rag/ 里。

用法:
    .venv/Scripts/python.exe scripts/ask.py "KGE 模型是怎么分类的？"
    .venv/Scripts/python.exe scripts/ask.py "……" "……"      # 连着问几个
    .venv/Scripts/python.exe scripts/ask.py --no-expand "……"  # 关掉查询扩展
    .venv/Scripts/python.exe scripts/ask.py --quiet "……"      # 只看答案

前置:
    1. Ollama 要起着（ollama serve 或开桌面应用）
       —— 连接被拒 WinError 10061 / ConnectionError 就是它没起
    2. 索引要存在（scripts/build_vectors.py）
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import (
    EMB_MODEL,
    EXPAND_ENABLED,
    GEN_MODEL,
    PROCESSED_DIR,
    RERANK_ENABLED,
    TOP_K,
)
from paper_rag.embed import make_client
from paper_rag.index import load
from paper_rag.pipeline import ask, build_retriever

# 每个 status 对应一个记号。放在这里是因为它【属于展示】，不属于判断 ——
# pipeline 只管算出 status 是什么，怎么显示是入口的事。
MARK = {
    "answered": "OK  ",
    "refused": "拒答",
    "suspect": "!!  ",
    "parse_failed": "!!  ",
}


def show(result: dict, quiet: bool) -> None:
    """把一次 ask() 的结果打印成人能看的样子。"""
    print("=" * 78)
    print(f"问: {result['question']}")

    if result["expanded"] and not quiet:
        print(f"检索用的文本: {result['query']}")

    if not quiet:
        print(f"\n--- 检索到 {len(result['retrieved'])} 条 ---")
        # ★ 检索器名字也印出来。理由和下面那条一样，而且是同一件事的续集：
        #   名字里就写着这次是 pure dense 还是 dense > rerank @pool50 ——
        #   不印的话，「换了检索器」和「没换」在这份输出里长得一模一样。
        print(f"    检索器: {result['retriever']}")
        # ★ 论文标题必须印出来。2026-10-10：模型答错了一条（拿推理那篇的分类法
        #   回答了嵌入模型的问题），而这份输出里【看不出】每条材料来自哪篇 ——
        #   要自己回去翻 chunks.json 才知道。失败能藏这么久，就是因为这个。
        #   出处不只是给模型看的判别器，也是给人看的 —— 两边都得有。
        for i, r in enumerate(result["retrieved"], 1):
            c = r["chunk"]
            print(f"  [{i}] {r['score']:+.4f}  {c['title']}  ({c['section']}，第 {c['page_start']} 页)")
            print(f"      {c['text'][:80]}")
        # ⚠️ 分数的【含义】跟着检索器变：纯 dense 是余弦 ∈[-1,1]，
        #    dense+rerank 是 cross-encoder 的 logit（可负、无上界）。
        #    所以这个数字【只在同一次输出内部】有意义 —— 靠它排序可以，
        #    跨检索器比大小、或者拿它去卡什么阈值，都不行。
        print("    （分数只在本次输出内部可比，用于排序；跨检索器不可比）")

    print(f"\n--- 结果 [{MARK.get(result['status'], result['status'])}] ---")

    if result["status"] == "parse_failed":
        # 不打印答案 —— 因为压根没有解析出来的答案。只报错。
        print(f"  模型输出无法解析：{result['note']}")
        print("  ⚠️ 这不是拒答，是解析失败。别把两者混为一谈。")
        return

    parsed = result["parsed"]
    print(f"  {parsed['answer']}")

    if result["status"] == "suspect":
        print(f"\n  ⚠️ {result['note']}")
        return

    if result["status"] == "refused":
        print("\n  （材料里没有答案，模型按契约拒答了）")
        return

    # 只有 answered 才走到这儿 —— 引用核对的明细
    print(f"\n  引用了: {parsed['used']}")
    for n, quote in parsed.get("quotes", {}).items():
        print(f"    [{n}] {quote[:90]}")

    if result["problems"]:
        print(f"\n  ⚠️ 引用校验发现 {len(result['problems'])} 个问题：")
        for p in result["problems"]:
            print(f"    - {p}")
    else:
        print("\n  ✓ 引用校验通过（每条 quotes 都能在材料原文里找到）")


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端问答：问一句话，拿带引用的答案")
    ap.add_argument("questions", nargs="+", help="一个或多个问题")
    ap.add_argument("--no-expand", action="store_true", help=f"关掉查询扩展（默认 {'开' if EXPAND_ENABLED else '关'}）")
    ap.add_argument("--quiet", action="store_true", help="只看答案，不打印检索明细")
    ap.add_argument("--no-rerank", action="store_true",
                    help=f"关掉 cross-encoder 重排（默认 {'开' if RERANK_ENABLED else '关'}）")
    ap.add_argument("-k", type=int, default=TOP_K, help=f"检索几条（默认 {TOP_K}）")
    args = ap.parse_args()

    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]

    # model=EMB_MODEL 会让 load() 核对索引是不是这个模型建的。
    # 索引不是它建的 → 直接报错，而不是拿不可比的向量算出看起来正常的错结果。
    vectors, ids = load(model=EMB_MODEL)

    print(f"语料 {len(chunks)} 条 · 索引 {vectors.shape} · 嵌入 {EMB_MODEL} · 生成 {GEN_MODEL}")
    print(f"查询扩展: {'开' if not args.no_expand else '关'}   k={args.k}")

    client = make_client()

    # ★★ 检索器在【循环外】建一次，然后每题复用。
    #
    #    为什么这条注释要大写：这个循环以前长这样 ——
    #
    #        for q in args.questions:
    #            result = ask(client, q, ...)
    #
    #    而 ask() 内部每次都会新建一个检索器。检索器里现在裹着一个
    #    cross-encoder，所以【忘了这一条】的后果是每问一句重新构造一次
    #    重排器 —— 而且不报错、不警告，只是"问答突然变得很慢"。
    #
    #    重排模型的【权重】有进程级缓存（paper_rag/rerank.py 的 _load_model），
    #    所以最坏情况不会重下 2.3GB；但"靠缓存兜底"不是设计，是安全网。
    retriever = build_retriever(
        client, EMB_MODEL, vectors, ids, chunks,
        k=args.k,
        enabled=not args.no_rerank,
    )
    print(f"检索器: {retriever.name}")

    for q in args.questions:
        result = ask(
            client,
            q,
            chunks,
            vectors,
            ids,
            k=args.k,
            expand=not args.no_expand,
            # 循环外建好的那个 —— 见上面的大写注释
            retriever=retriever,
            # temperature=0 —— 入口也要可复现。同一个问题两次跑出两个答案，
            # 你就没法判断「改了一行之后变好了还是碰巧」。
            options={"temperature": 0},
        )
        show(result, args.quiet)


if __name__ == "__main__":
    main()
