"""检索：向量相似度计算。

Stage 2 的第一块。这里【刻意】不依赖任何向量库 ——

  先用最原始的方式把「两个向量有多像」写出来，
  再用最原始的方式把「237 条里哪几条最像」写出来，
  最后才换成 FAISS，并且证明两者结果一致。

不先手写一遍，你永远不知道 FAISS 到底替你做了什么。

前三节（dot / norm / cosine / top_k）【刻意用纯 Python】写，不碰 numpy。
理由：numpy 把循环藏起来了，而现在你要看的正是那个循环。

2.4 节的 top_k_np 才换 numpy —— 只有先看清循环长什么样，
才知道它到底替你省掉了什么、快在哪里。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math

from paper_rag.config import TOP_K

import numpy as np


def dot(a: list[float], b: list[float]) -> float:
    """两个向量的点积。

    定义（n 维）：
        a·b = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]

    也就是：把两个向量【对应位置】的数两两相乘，再把所有乘积加起来。

    长度必须相同 —— 不相同就是 bug，应该直接报错而不是算出一个错答案。
    想一想：这两个向量应该一样长，如果不一样长，说明哪里出了问题？

    参数:
        a: 向量，list[float]
        b: 向量，list[float]，长度必须与 a 相同

    返回:
        float，点积的结果

    提示:
        你需要一个累加器。想想 0A-1 里学过的「累加器惯用法」，
        以及怎么同时遍历两个列表 —— zip 是个好工具。
    """
    if len(a) != len(b):
        raise ValueError("向量长度必须相同")

    result = 0
    for a_i, b_i in zip(a, b):
        result += a_i * b_i
    return result


def norm(a: list[float]) -> float:
    """向量的模长（L2 范数），也就是这支箭的长度。

    定义：
        |a| = √(a[0]² + a[1]² + ... + a[n-1]²)

    也就是：把每个数平方、加起来、再开根号。

    开根号用 math.sqrt()。注意 math 已经 import 好了。

    参数:
        a: 向量，list[float]

    返回:
        float，模长

    提示:
        平方之和可以用 dot(a, a) 直接拿到，试试看对不对。
        想想为什么 dot(a, a) 就等于 Σ aᵢ²。
    """
    return math.sqrt(dot(a, a))


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 —— 两个向量夹角的余弦。

    定义：
        cos(a, b) = (a·b) / (|a| * |b|)

    返回值的含义：

         1.0  方向完全相同  —— 意思一样
         0.0  互相垂直      —— 毫无关系
        -1.0  方向完全相反  —— 意思相反

    参数:
        a: 向量，list[float]
        b: 向量，list[float]

    返回:
        float，范围 [-1, 1]

    提示:
        这一行就能写完 —— 直接用上面两个函数。
        但要先想一个问题：如果 |a| 或 |b| 是 0（零向量），会怎样？
        真实场景会不会出现？要不要防？
    """
    na, nb = norm(a), norm(b)

    return dot(a, b) / (na * nb) if na != 0 and nb != 0 else 0.0


def top_k(
    query_vec: list[float],
    vectors: list[list[float]],
    ids: list[str],
    k: int = TOP_K,
) -> list[tuple[str, float]]:
    """在语料向量里找出和 query_vec 最像的 k 条。【纯 Python 写】

    这是检索的心脏。整个 RAG 的性能问题，最后都会落到这一行上：
    每问一次，就要把 query 和【全部 N 条】语料比一遍。

    参数:
        query_vec: 查询向量，形状 (D,)
        vectors:   语料向量，形状 (N, D) —— vectors[i] 是第 i 条语料
        ids:       语料 id，形状 (N,)     —— ids[i] 对应 vectors[i]
        k:         取前几名

    返回:
        [(chunk_id, 分数), ...]
        按分数【从高到低】排序，长度 = min(k, N)
        分数是余弦相似度，范围 [-1, 1]

    提示（按顺序想，别跳）:
        1. 你已经写好 cosine() 了，直接用。
        2. 要「每条都比一遍」，所以是一个 O(N) 的循环。
           循环里同时拿到 i 和 vectors[i] —— enumerate()。
        3. 【关键】循环里每一步要记两个东西：分数、以及它属于哪个 id。
           只记分数的话，排完序你就不知道是哪个 chunk 了。
           正好配对的两个值 —— 用元组。
        4. 排序：sorted(..., reverse=True, key=?)。
           想清楚 (分数, id) 这个元组顺序，能不能直接用默认排序。
        5. 最后切片取前 k 个。

    自查（写完对着想一遍）:
        - ids 和 vectors 长度不一样会怎样？要不要先断言？
        - k 比 N 大呢？切片会自动处理吗？
        - 两个分数完全一样（打平）时，顺序由什么决定？
    """
    # 自查1: ids 和 vectors 长度不一样会怎样？要不要先断言？
    if len(ids) != len(vectors):
        raise ValueError("ids 和 vectors 的长度必须一致")

    # 收集 (id, 分数) —— 【顺序和对外契约一致】，读的人不用在脑子里做转换
    scored = []
    for i, vec in enumerate(vectors):
        scored.append((ids[i], cosine(query_vec, vec)))

    # key 显式取分数。好处是打平时 sorted 是稳定的（保持原顺序，确定可复现）
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def top_k_np(
    query_vec: list[float],
    vectors,
    ids: list[str],
    k: int = TOP_K,
) -> list[tuple[str, float]]:
    """和 top_k 一模一样，但用 numpy。【契约必须逐位一致】

    同样的输入必须给出同样的输出 —— 顺序、分数、长度都不能有差别。
    这不是「差不多就行」：改了契约，调用方就得跟着改；更糟的是
    【调用方察觉不到】—— 它只是拿到一堆看起来正常的错结果。

    参数:
        query_vec: 查询向量，(D,)
        vectors:   语料向量，np.ndarray，形状 (N, D)，dtype float32
        ids:       语料 id，list[str]，长度 N
        k:         取前几名

    返回:
        [(chunk_id, 分数), ...]，和 top_k 的输出逐位一致

    提示（照顺序想）:
        1. 整个目标：把那个 O(N) 的 Python 循环换成【一次矩阵运算】。
           N 个向量和 1 个 query 的点积，在矩阵语言里是什么？
               (N, D) @ (D,)  →  (N,)
           这一行就是整个检索。
        2. 余弦还差个分母。所有行的模长怎么一次算完？
               np.linalg.norm(vectors, axis=1)
           想清楚 axis=1 是「沿着哪个轴收缩」，为什么不是 axis=0。
        3. 拿到 (N,) 的分数数组，取最大的 k 个下标。两种写法：
               np.argsort(-scores)[:k]              全排     O(N log N)
               np.argpartition(-scores, k)[:k]      部分排   O(N)
           都能用。先写跑得通的那个，再想哪个更合适。
        4. 【argpartition 的坑】它只保证前 k 个是最大的，【内部顺序是乱的】——
           拿到下标后必须再对这 k 个排一次。这一步最容易漏。
        5. 边界：k >= N 时 argpartition 会直接抛错，想好怎么处理。
        6. 下标是 np.int64，拿去索引 list 行不行？不行就转成 int。

    想一想（先别动手，这是下一小节的正题）:
        我们量过 bge-m3 的向量模长【恒等于 1】（0.999999 ~ 1.000001）。
        那么 cos(a, b) = (a·b) / (|a| · |b|) 里的分母等于多少？
        分母能省掉的话，上面第 1、2 步会塌缩成什么？
        （顺带 —— 那正好也是 FAISS 里一个索引类型的名字。）
    """
    N = vectors.shape[0]

    # 【边界防御】k 严格大于 N 时 argpartition 会 ValueError: kth out of bounds。
    # k == N 是合法的（整个数组就是答案）。这里顺手统一成「k 超过 N 就返回全部」，
    # 和 top_k 的切片行为对上 —— 两个实现的契约必须一致。
    k = min(k, N)
    q = np.asarray(query_vec, dtype=np.float32)

    # (N, D) @ (D,) -> (N,)
    scores = vectors @ q

    # 2. 用 argpartition 找最大的 k 个下标（O(N) 复杂度）
    # 写法：直接找正序数组里最大的 k 个
    top_k_idx = np.argpartition(scores, -k)[-k:]

    # 3. 【关键】对挑出来的 k 个下标，按分数降序二次排序
    # 这一步保证了和 top_k 函数的输出顺序完全一致
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

    # 4. 转成原生 Python 类型，严格满足契约
    return [(ids[int(i)], float(scores[i])) for i in top_k_idx]


def top_k_faiss(
    index,
    query_vec,
    ids: list[str],
    k: int = TOP_K,
) -> list[tuple[str, float]]:
    """用 FAISS 索引检索。契约和前两个实现一致。

    【为什么要有第三个实现】
        模块开头就承诺了：先用最原始的方式写出来，最后换成 FAISS，
        并【证明两者结果一致】。这就是那一步。

    FAISS 替你做了什么：
        它就是把 `vectors @ q` + `argpartition` 用 C++ 重写了一遍，
        加上 SIMD 和多线程。算法上【没有任何魔法】——
        这既是它和 numpy 版结果完全一致的原因，
        也是你该亲手验一遍的原因。

    参数:
        index:     faiss.Index，已经 add 过语料向量
        query_vec: 查询向量，(D,)，list 或 ndarray
        ids:       语料 id 列表，【顺序必须和 index.add 时的顺序一致】
        k:         取前几名

    返回:
        [(chunk_id, 分数), ...]，和前两个实现逐位一致

    提示:
        1. index.search(q, k) 返回两个数组：D（分数）和 I（下标）。
           两个都是【二维】的，形状 (nq, k) —— FAISS 支持一次查多条。
           我们只有一条 query，所以要取第 0 行。
        2. 【坑一】I 里装的是【位置下标】，不是 chunk_id。
           要拿它回 ids 里换 —— 这层映射 FAISS 不知道，只有你知道。
        3. 【坑二】候选不足 k 个时，FAISS 用 -1 填充下标。
           直接拿去索引 ids 会拿到 ids[-1]（最后一条）——
           不报错，但结果是错的。要过滤掉。
        4. 【坑三】FAISS 要 (nq, D) 的二维输入，
           传一维 (D,) 会报错。用 reshape(1, -1) 或 q[None, :]。

    想一想（这才是 2.5 的正题）:
        如果 FAISS 的结果和 numpy 版完全一致，那它的价值到底在哪？
        答案不在这个函数里 —— 在 index 的【类型】上。
        IndexFlatIP 是精确的，换一个类型就不是了。换了会怎样？
    """

    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q, k)

    # 过滤掉下标为 -1 的候选（不足 k 个时 FAISS 会填充 -1）
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((ids[int(idx)], float(score)))

    return results
