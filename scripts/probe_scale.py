"""探针：规模实验 —— 「237 条，到底要不要上向量库？」

这个脚本不教 FAISS 怎么用，它回答的是【什么时候才值得用】。

三个问题，一节一个：
  1. 真实规模（237 条）下，FAISS 比 numpy 暴力快吗？
  2. 放大到 10 万条，加速比怎么变？
  3. IVF 的 nprobe 每调大一档，快多少、丢多少召回？

【为什么必须自己量】
    同一份代码、同一个库，只换数据，结论能从「IVF 没用」翻成
    「IVF 快 100 倍」。基准测试最容易骗人的地方不是代码，是数据。

【合成数据的坑 —— 我踩过】
    1024 维里均匀随机的向量近似两两正交，「最近的邻居」和瞎猜没区别。
    要让 IVF 有意义，数据必须有【簇结构】。而且扰动的模长必须按 sqrt(D) 缩放：

        randn(D) 的模长约等于 sqrt(D) ≈ 32
        直接乘 spread=0.15 → 实际扰动 4.8，比簇心之间的距离还大，簇就没了

    第一次我就是这么写的，于是量出 IVF recall≈0.01，
    差点得出「IVF 在这个项目里没用」的结论 —— 那是我数据的问题。

【已知局限 —— 100k 那两节的计时口径和生产【不一致】】

    bench_all() 一次把 nq 条 query 打包成矩阵一起查（GEMM）：
        x @ q.T        (n, D) @ (D, nq)
    好处是 391 MB 的语料只读一遍，所以它衡量的是【离线批量评测】的场景
    （比如给 1000 道题算 recall）。

    但线上 RAG 是【用户问一句答一句】（GEMV）：
        x @ q          (n, D) @ (D,)
    每问一次都要把整个语料扫一遍，numpy 那边会慢大约 2.8×。

    实测 100k × 1024：
        numpy  GEMV 单条    5.304 ms/query
        numpy  GEMM 批量    1.888 ms/query
        FAISS  批量         4.048 ms/query

    → 所以这一节里 numpy 看起来「不输 FAISS」，有一部分是口径送的。
      真实单条查询下，FAISS 是略快的那个（1.3× 左右）。

    这一段是上一轮基准的教训：**同一份代码、同一台机器，
    只改「一次查几条」，加速比的符号就翻了。比数字更重要的是口径。**

用法:
    python scripts/probe_scale.py
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from paper_rag.config import EMB_MODEL, EVAL_DIR
from paper_rag.embed import embed_one, make_client
from paper_rag.index import load

try:
    import faiss
except ImportError:
    faiss = None

LINE = "=" * 72
K = 5  # 和 config.TOP_K 一致：我们真正在乎的前 5 名


def timeit(fn, budget=0.4):
    """跑够 budget 秒再取平均。

    固定次数的计时在小规模上会被噪声主导 —— 237 条一次才几十微秒，
    调度抖动比计算本身还大。跑满一个时间预算再平均才稳。
    """
    fn()  # 预热：第一次调用要付 import / 缓存 / 线程池启动的钱
    reps = 0
    out = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget and reps < 20_000:
        out = fn()
        reps += 1
    return (time.perf_counter() - t0) / max(reps, 1) * 1000, out


def recall_at_k(truth, got, k=K):
    """前 k 名里命中了几个。

    truth / got 都是 (NQ, k) 的下标数组。
    返回所有 query 的平均命中率，范围 0~1。
    """
    if len(truth) == 0:
        return 0.0
    hit = 0
    for t, g in zip(truth, got):
        hit += len(set(t[:k].tolist()) & set(g[:k].tolist()))
    return hit / (len(truth) * k)


def make_clustered(n, d, spread, nq, seed=0):
    """造有簇结构的语料 + 对应的 query。全部归一化（模长 = 1）。"""
    rng = np.random.default_rng(seed)
    nlist = max(1, int(np.sqrt(n)))

    centers = rng.standard_normal((nlist, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    which = rng.integers(0, nlist, size=n)
    # ★ 关键这一行：除以 sqrt(d)，把「每个分量的标准差」换算成「整支向量的模长」
    noise = rng.standard_normal((n, d)).astype(np.float32) / np.sqrt(d)
    x = centers[which] + spread * noise
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # query = 随机挑一条语料，稍微扰动。正确答案就是它自己 —— 我们知道答案。
    pick = rng.choice(n, size=nq, replace=False)
    qnoise = rng.standard_normal((nq, d)).astype(np.float32) / np.sqrt(d)
    q = x[pick] + spread * 0.5 * qnoise
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return x, q, pick


def bench_all(x, q, truth, label):
    """同一份数据上，把 numpy / Flat / IVF 全跑一遍。"""
    n, d = x.shape
    nq = q.shape[0]

    print()
    print("─" * 72)
    print(f"{label}    {n:,} × {d} 维    {n * d * 4 / 1024 / 1024:.0f} MB    {nq} 条 query")
    print("─" * 72)

    # ---- 基准：numpy 暴力 ----
    def np_brute():
        s = x @ q.T  # (n, D) @ (D, nq) → (n, nq)
        return np.argpartition(-s, K, axis=1)[:, :K].T  # (nq, K)

    t_np, _ = timeit(np_brute)
    print(f"  {'numpy 暴力':<20}{t_np:9.3f} ms/次              recall 1.000   ← 精确解")

    if faiss is None:
        print("  [SKIP] 没装 faiss-cpu —— pip install faiss-cpu")
        return

    # ---- FAISS 精确索引 ----
    index = faiss.IndexFlatIP(d)
    index.add(x)

    def flat():
        _, I = index.search(q, K)
        return I

    t, I = timeit(flat)
    print(f"  {'FAISS IndexFlatIP':<20}{t:9.3f} ms/次   加速 {t_np / t:7.1f}×   "
          f"recall {recall_at_k(truth, I):.3f}")

    # ---- FAISS 近似索引（IVF）----
    nlist = max(1, int(np.sqrt(n)))
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(x)
    ivf.add(x)

    for nprobe in (1, 4, 16, 64, nlist):
        ivf.nprobe = nprobe

        def probe(ivf=ivf):
            _, I2 = ivf.search(q, K)
            return I2

        t, I2 = timeit(probe)
        mark = "" if nprobe < nlist else "   ← 全扫，等于暴力"
        print(f"  {'IVF nprobe=' + str(nprobe):<20}{t:9.3f} ms/次   加速 {t_np / t:7.1f}×   "
              f"recall {recall_at_k(truth, I2):.3f}{mark}")


def main() -> None:
    vectors, ids = load(model=EMB_MODEL)
    d = vectors.shape[1]
    print(f"索引文件: {vectors.shape}  {vectors.dtype}  模型 {EMB_MODEL}")

    # ══════════════ 1 · 真实规模 ══════════════
    # 用我们【自己】的 237 条真语料 + 真实的评测问题。
    # 这一节不需要合成数据，也就没有「数据不真实」的借口。
    client = make_client()
    queries = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))["queries"]
    qv = np.stack([np.asarray(embed_one(client, EMB_MODEL, it["q"]), dtype=np.float32)
                   for it in queries])

    s = vectors @ qv.T
    truth = np.argsort(-s, axis=0)[:K].T  # (nq, K) 精确前 5 名
    bench_all(vectors, qv, truth, "真实语料（我们的 PDF 切出来的）")

    print()
    print("  ▸ 如果这一节看不出 FAISS 的价值，别急着说「FAISS 没用」——")
    print("    换一个规模再问一次。下面把语料放大 400 倍。")

    # ══════════════ 2 · 放大 ══════════════
    # 扰动越小 → 簇内越紧 → IVF 越好找。两个值都跑，看结论稳不稳。
    for spread in (0.15, 0.4):
        x, q, _ = make_clustered(100_000, d, spread, nq=20)
        s = x @ q.T
        truth = np.argsort(-s, axis=0)[:K].T
        bench_all(x, q, truth, f"合成语料  簇内扰动 {spread}")

    # ══════════════ 3 · 结论 ══════════════
    print()
    print(LINE)
    print("怎么读这张表")
    print(LINE)
    print()
    print("  1. IndexFlatIP 快的那几倍，纯粹是 C++ + SIMD + 多线程的功劳。")
    print("     算法和 numpy 版一模一样 —— 所以它的 recall 永远是 1.000。")
    print()
    print("  2. IVF 快的那几十上百倍，是【拿召回换的】。")
    print("     nprobe 调小 = 只搜几个簇 = 快，但答案可能不在那几个簇里。")
    print("     nprobe 调到 nlist 就是全扫 —— 这时它和暴力没有区别，也就不用它了。")
    print()
    print("  3. 决定要不要上向量库的，不是 FAISS 有多快，是【你的 N 有多大】。")
    print("     237 条的时候，numpy 一行矩阵乘法已经是微秒级；")
    print("     索引带来的额外一层（训练、调参、recall 风险、持久化）纯属负债。")
    print()
    print(LINE)


if __name__ == "__main__":
    main()
