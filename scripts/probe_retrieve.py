"""探针：验收手写的检索实现。

五关，一关比一关狠：

  第 1 关 · 4 条 2 维向量
      答案纸笔就能算。排序错了会在这一关露馅。

  第 2 关 · 真实索引（237 × 1024），和 numpy 独立实现对拍
      同一份数据、同一个 query，两条完全不同的代码路径。
      数一样 → top_k 是对的。这就是「对拍」。

  第 3 关 · 10 道评测题端到端
      数字必须和 eval_embed.py 报的【对得上】：
          bge-m3 → Top-1 70%、MRR 0.766、平均排名 4.0
      对不上就说明：要么 top_k 错了，要么索引和评测用的不是同一批向量。
      这一关是真正的验收 —— 两个独立脚本，同一组数字。

  第 4 关 · top_k_np 对拍
      换 numpy 实现，契约必须逐位一致：顺序、分数都不能变。
      顺带量速度，以及【线性外推为什么不可靠】。

  第 5 关 · top_k_faiss 对拍
      第三个实现。FAISS 结果是精确的，所以必须和 numpy 一模一样。
      顺便踩一个坑：k > N 时 FAISS 拿 -1 填坑。

「237 条到底要不要上向量库」是【另一个问题】，在 scripts/probe_scale.py。

用法:
    python scripts/probe_retrieve.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from paper_rag.config import EMB_MODEL, EVAL_DIR, TOP_K
from paper_rag.embed import embed_one, make_client
from paper_rag.index import load
from paper_rag.retrieve import top_k

LINE = "=" * 72


def main() -> None:
    client = make_client()

    # ══════════════ 第 1 关 · 纸笔可验 ══════════════
    print(LINE)
    print("第 1 关 · 4 条 2 维向量")
    print(LINE)

    QUERY = [1.0, 0.0]
    VECTORS = [
        [0.0, 1.0],  # A  垂直    → cos = 0.0
        [1.0, 0.0],  # B  同向    → cos = 1.0
        [-1.0, 0.0],  # C  反向    → cos = -1.0
        [0.8, 0.6],  # D  36.87°  → cos = 0.8
    ]
    IDS = ["A", "B", "C", "D"]
    WANT = [("B", 1.0), ("D", 0.8), ("A", 0.0), ("C", -1.0)]

    got = top_k(QUERY, VECTORS, IDS, k=4)
    print(f"\n  query = {QUERY}")
    print(f"  期望顺序: {[w[0] for w in WANT]}")
    print(f"  实际顺序: {[g[0] for g in got]}")

    ok = len(got) == len(WANT) and all(
        g[0] == w[0] and abs(g[1] - w[1]) < 1e-9 for g, w in zip(got, WANT)
    )
    print(f"  [{'OK  ' if ok else 'FAIL'}] 顺序与分数")
    if not ok:
        for g in got:
            print(f"        {g}")
        return

    top2 = top_k(QUERY, VECTORS, IDS, k=2)
    ok2 = [t[0] for t in top2] == ["B", "D"]
    print(f"  [{'OK  ' if ok2 else 'FAIL'}] k=2 只取前两名 → {[t[0] for t in top2]}")

    # ══════════════ 第 2 关 · 和 numpy 对拍 ══════════════
    print()
    print(LINE)
    print("第 2 关 · 真实索引，和 numpy 对拍")
    print(LINE)

    vectors_np, ids = load(model=EMB_MODEL)
    print(f"\n  索引: {vectors_np.shape}  {vectors_np.dtype}  模型 {EMB_MODEL}")

    # 转成 list[list[float]] 给纯 Python 版用 —— 这一步本身也有成本
    t0 = time.time()
    vectors_list = vectors_np.tolist()
    t_conv = time.time() - t0
    print(f"  ndarray → list 转换: {t_conv * 1000:.0f} ms")

    QUERY_TEXT = "RAG 是怎么把检索到的文档融进生成过程的？"
    qvec = embed_one(client, EMB_MODEL, QUERY_TEXT)

    t0 = time.time()
    got = top_k(qvec, vectors_list, ids, k=10)
    t_py = time.time() - t0

    # numpy 独立算一遍
    q = np.asarray(qvec, dtype=np.float32)
    sims = (vectors_np @ q) / (np.linalg.norm(vectors_np, axis=1) * np.linalg.norm(q))
    want_idx = np.argsort(-sims)[:10]
    want = [(ids[i], float(sims[i])) for i in want_idx]

    print(f"\n  query: {QUERY_TEXT}")
    print(f"  纯 Python top_k 用时: {t_py * 1000:.0f} ms\n")
    print(f"  {'名次':<5}{'纯 Python':<26}{'numpy'}")
    print("  " + "-" * 68)
    all_ok = True
    for n, (g, w) in enumerate(zip(got, want), 1):
        same = g[0] == w[0] and abs(g[1] - w[1]) < 1e-4
        all_ok &= same
        flag = "" if same else "   <<< 不一致"
        print(f"  {n:<5}{g[0]:<20}{g[1]:+.6f}    {w[0]:<20}{w[1]:+.6f}{flag}")
    print(f"\n  [{'OK  ' if all_ok else 'FAIL'}] 两套实现的 top-10 完全一致")

    # ══════════════ 第 3 关 · 10 题端到端 ══════════════
    print()
    print(LINE)
    print("第 3 关 · 10 道评测题 —— 数字必须和 eval_embed.py 对上")
    print(LINE)

    queries = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))["queries"]
    N = len(ids)

    ranks = []
    qvecs = []  # 留着给第 4 关计时用
    t0 = time.time()
    for n, item in enumerate(queries, 1):
        qv = embed_one(client, EMB_MODEL, item["q"])
        qvecs.append(qv)
        # k=N → 拿到的就是【完整排名】，顺便就得到了每条的名次
        ranked = top_k(qv, vectors_list, ids, k=N)
        rank = next(
            (pos for pos, (cid, _) in enumerate(ranked, 1) if cid in item["targets"]),
            N + 1,
        )
        ranks.append(rank)
        print(f"  {n:<3} 排名 {rank:>3}{'  OK' if rank == 1 else ''}   {item['q']}")
    t_all = time.time() - t0

    top1 = sum(1 for r in ranks if r == 1) / len(ranks)
    mrr = sum(1 / r for r in ranks) / len(ranks)
    avg = statistics.mean(ranks)

    print()
    print(f"  Top-1 {top1 * 100:.0f}%   MRR {mrr:.3f}   平均排名 {avg:.1f}")
    print("  eval_embed.py 报的:  Top-1 70%   MRR 0.766   平均排名 4.0")

    match = abs(top1 - 0.70) < 1e-9 and abs(mrr - 0.766) < 5e-4 and abs(avg - 4.0) < 1e-9
    print(f"  [{'OK  ' if match else 'FAIL'}] 两个独立脚本的数字一致")

    # ══════════════ 第 4 关 · numpy 版对拍 ══════════════
    print()
    print(LINE)
    print("第 4 关 · numpy 版 —— 契约必须和纯 Python 逐位一致")
    print(LINE)

    # 试调一次，没写就跳过 —— 前 3 关的结果不该被第 4 关拖住
    try:
        from paper_rag.retrieve import top_k_np

        top_k_np(QUERY, np.asarray(VECTORS, dtype=np.float32), IDS, k=2)
    except (ImportError, NotImplementedError):
        print("\n  [SKIP] top_k_np 还没写 —— 2.4 节写完再跑这一关")
        print(LINE)
        return

    # --- 4a. 小向量，顺便压 k >= N 这个边界 ---
    print()
    small = np.asarray(VECTORS, dtype=np.float32)
    a4 = top_k_np(QUERY, small, IDS, k=4)  # k == N，argpartition 容易在这炸
    ok = [t[0] for t in a4] == ["B", "D", "A", "C"]
    print(f"  [{'OK  ' if ok else 'FAIL'}] k=4 = N 的边界     {[t[0] for t in a4]}")

    a2 = top_k_np(QUERY, small, IDS, k=2)
    ok = [t[0] for t in a2] == ["B", "D"]
    print(f"  [{'OK  ' if ok else 'FAIL'}] k=2               {[t[0] for t in a2]}")

    # --- 4b. 真数据，10 题 × top-20：顺序和分数都要一致 ---
    # 分数也要比！只比 id 的话，「顺序对了但分数算错」这种 bug 会溜过去。
    NT = 20
    TOL = 1e-5
    all_same = True
    max_gap = 0.0
    for item, qv in zip(queries, qvecs):
        x = top_k(qv, vectors_list, ids, k=NT)
        y = top_k_np(qv, vectors_np, ids, k=NT)

        if [t[0] for t in x] != [t[0] for t in y]:
            all_same = False
            print(f"\n  [FAIL] 顺序不一致: {item['q'][:34]}")
            print(f"         纯 Python: {[t[0] for t in x][:5]}")
            print(f"         numpy    : {[t[0] for t in y][:5]}")
            continue

        gap = max(abs(a[1] - b[1]) for a, b in zip(x, y))
        max_gap = max(max_gap, gap)
        if gap > TOL:
            all_same = False
            print(f"\n  [FAIL] 分数不一致 最大差 {gap:.2e}: {item['q'][:34]}")

    print(f"\n  [{'OK  ' if all_same else 'FAIL'}] 10 题 × top-20：顺序 + 分数都一致")
    print(f"         最大分数差 {max_gap:.2e}   容差 {TOL:.0e}")

    # --- 4c. 速度 ---
    t0 = time.time()
    for qv in qvecs:
        top_k(qv, vectors_list, ids, k=TOP_K)
    t_py10 = time.time() - t0

    t0 = time.time()
    for qv in qvecs:
        top_k_np(qv, vectors_np, ids, k=TOP_K)
    t_np10 = time.time() - t0

    py_ms = t_py10 / len(qvecs) * 1000
    np_ms = t_np10 / len(qvecs) * 1000

    print()
    print(f"  每次检索（{len(ids)} 条语料）:")
    print(f"      纯 Python : {py_ms:9.2f} ms")
    print(f"      numpy     : {np_ms:9.3f} ms")
    print(f"      提速      : {py_ms / np_ms:9.0f}×")

    # --- 4d. 外推为什么不可靠 ---
    # 成本 = 常数项（Python/numpy 调用开销）+ 线性项（真正的计算）。
    # 小规模时常数项占大头，拿它去乘倍数 = 把常数也放大了同样的倍数。
    # 这里实测一遍，看看外推错多少。
    print()
    print("  线性外推可信吗 —— 实测 vs 外推（合成语料，测【整个函数】）")
    print(f"  {'N':>9}  {'实测':>11}  {'从 237 外推':>14}  {'高估':>8}")
    print("  " + "-" * 50)

    rng = np.random.default_rng(0)
    D = vectors_np.shape[1]
    base = None
    for NN in (len(ids), 1_000, 10_000, 100_000):
        W = rng.standard_normal((NN, D), dtype=np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        fake_ids = [f"x{i}" for i in range(NN)]
        # 小规模多跑几轮，避免测到的全是计时噪声
        reps = max(1, 100_000 // NN)
        batch = qvecs[:5] * reps

        top_k_np(batch[0], W, fake_ids, k=TOP_K)  # 预热
        t0 = time.time()
        for q in batch:
            top_k_np(q, W, fake_ids, k=TOP_K)
        t = (time.time() - t0) / len(batch) * 1000

        if base is None:
            base = t / NN  # 237 条上的「每条成本」—— 老代码就是用它外推的
            guess = t
        else:
            guess = base * NN
        print(f"  {NN:>9,}  {t:8.3f} ms  {guess:11.3f} ms  {guess / t:6.1f}×")

    print("  " + "-" * 50)
    print("  高估的倍数 = 237 条上「常数开销」在总成本里的占比。")
    print("  常数项不该被乘以 422 —— 这就是外推骗人的地方。")

    # ══════════════ 第 5 关 · FAISS 对拍 ══════════════
    print()
    print(LINE)
    print("第 5 关 · FAISS —— 第三个实现，契约还是那一个")
    print(LINE)

    try:
        import faiss
    except ImportError:
        print("\n  [SKIP] 没装 faiss-cpu —— pip install faiss-cpu")
        print(LINE)
        return

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    try:
        from paper_rag.retrieve import top_k_faiss

        top_k_faiss(index, qvecs[0], ids, k=2)
    except NotImplementedError:
        print("\n  [SKIP] top_k_faiss 还没写 —— 2.5 节写完再跑这一关")
        print(LINE)
        return

    # --- 5a. 契约对拍：顺序 + 分数 ---
    # IndexFlatIP 是【精确】索引，算法上就是那行矩阵乘法。
    # 所以这里不是「差不多就行」—— 必须逐位一致，否则说明哪里理解错了。
    NT = 20
    TOL = 1e-5
    all_same = True
    max_gap = 0.0
    for item, qv in zip(queries, qvecs):
        x = top_k_np(qv, vectors_np, ids, k=NT)
        y = top_k_faiss(index, qv, ids, k=NT)

        if [t[0] for t in x] != [t[0] for t in y]:
            all_same = False
            print(f"\n  [FAIL] 顺序不一致: {item['q'][:34]}")
            print(f"         numpy : {[t[0] for t in x][:5]}")
            print(f"         faiss : {[t[0] for t in y][:5]}")
            continue

        gap = max(abs(a[1] - b[1]) for a, b in zip(x, y))
        max_gap = max(max_gap, gap)
        if gap > TOL:
            all_same = False
            print(f"\n  [FAIL] 分数不一致 最大差 {gap:.2e}: {item['q'][:34]}")

    print(f"\n  [{'OK  ' if all_same else 'FAIL'}] 10 题 × top-{NT}：顺序 + 分数都一致")
    print(f"         最大分数差 {max_gap:.2e}   容差 {TOL:.0e}")

    # --- 5b. k > N 的边界：FAISS 用 -1 填坑 ---
    # 前两个实现对 k > N 的处理都是「返回 N 条」（切片 / min(k, N)）。
    # FAISS 不切片，它【补 -1】把数组撑满 k 列。契约在这里分叉了。
    over = len(ids) + 5
    y = top_k_faiss(index, qvecs[0], ids, k=over)
    uniq = len({cid for cid, _ in y})
    ok = len(y) == len(ids) and uniq == len(ids)
    print(f"\n  k={over} > N={len(ids)}:")
    print(f"  [{'OK  ' if ok else 'FAIL'}] 返回 {len(y)} 条，去重后 {uniq} 条"
          f"（期望都是 {len(ids)}）")
    if not ok:
        print("        多出来的就是 -1 填充 —— 拿去索引 ids 会撞上最后一条，不报错。")

    # --- 5c. 速度 ---
    t0 = time.time()
    for qv in qvecs:
        top_k_faiss(index, qv, ids, k=TOP_K)
    t_faiss10 = time.time() - t0
    faiss_ms = t_faiss10 / len(qvecs) * 1000

    print()
    print(f"  每次检索（{len(ids)} 条语料）:")
    print(f"      numpy          : {np_ms:9.3f} ms")
    print(f"      FAISS FlatIP   : {faiss_ms:9.3f} ms")
    ratio = np_ms / faiss_ms
    if ratio > 1.2:
        print(f"      FAISS 快 {ratio:.1f}×")
    elif ratio > 0.8:
        print(f"      基本打平（{ratio:.2f}×）—— 237 条太短，两边都在付调用开销")
    else:
        print(f"      FAISS 反而慢 {1 / ratio:.1f}× —— 索引再快也快不过「不用索引」")

    print()
    print("  想一想：这里比的是【同一件事的两种写法】，快慢只反映实现。")
    print("  真正的问题在另一处 —— 如果索引换成 IVF，结果就不再精确了。")
    print("  那笔账在 scripts/probe_scale.py 里算。")
    print(LINE)


if __name__ == "__main__":
    main()
