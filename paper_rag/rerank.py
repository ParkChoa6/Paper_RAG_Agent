"""重排 —— 给一个已经捞回来的候选列表重新排序。

── 为什么这是【另一个抽象】，不是第三个 Retriever ──

    看它们的输入：

        Retriever.search(query)          → 只有 query，自己去全库找
        Reranker.rerank(query, candidates) → query + 【一批已经捞回来的候选】

    重排器【没有能力检索】。它拿不到全库，也不想拿 —— 它只负责在
    一小撮候选里分出谁更相关。所以：

        ★ 把 rerank 塞进 Retriever 接口是【错的抽象】。
          那样每个重排器都得先实现一遍"怎么从 1345 条里捞 20 条"，
          而那件事它根本不干。

    但是——**「检索器 + 重排器」这个组合，应该仍然长得像检索器**：

        RerankedRetriever  implements  Retriever

    这样 eval_retrieve.py 和 pipeline.ask() 【一行都不用改】就能拿它当检索器用。
    这才是 S5-1 那个 Protocol 的回报（见 retriever.py 开头）。

── 为什么重排器手里要攥着 id → text ──

    看 Retriever 的返回：`[(chunk_id, 分数)]` —— 【只有 id，没有正文】。
    而 cross-encoder 要的恰恰是正文：它算的是 `score(query, passage)` 这一对。

    所以正文必须有人拿着。放哪？

        ① 让 search() 连正文一起返回   → 要改 Retriever 契约，
                                           BM25 / dense 全得跟着动，不值
        ② 让 RerankedRetriever 拿着     → 它得为了别人的需要持有全库文本，绕
        ③ 让 Reranker 拿着 ★ 本文件选这个

    ③ 的理由：**重排器是唯一【必须】看到正文的那个东西。**
    把正文给唯一需要它的人，是内聚，不是耦合。

    ⚠️ 代价要说清楚：`CrossEncoderReranker.from_chunks(chunks)` 会把
       1345 条正文全存一份（内存里多一份引用，字符串是共享的，不翻倍）。
       它【不持有向量】，所以不跟 DenseRetriever 抢内存。

── 它在流水线的哪个位置 ──

    query
      │
      ▼
    ┌──────────────────────────────┐
    │  ① Retriever.search(q, pool) │  捞【宽】一点：pool=20 而不是 5
    └──────────────────────────────┘
      │  20 条 (id, 分数)
      ▼
    ┌──────────────────────────────┐
    │  ② Reranker.rerank(q, 20)    │  ← 本文件：20 对 (query, 正文) 逐对打分
    └──────────────────────────────┘
      │  20 条 (id, 【另一个分数】)
      ▼
    ┌──────────────────────────────┐
    │  ③ 取前 k=5 送进 prompt       │
    └──────────────────────────────┘

    ★ ①的 pool 必须 > ③的 k，否则重排【白做】——
      它只能重排你给它的东西，后面的名次永远上不来。见 config.RETRIEVE_POOL。

── 这是【产品】还是【脚手架】？──

    产品。判断可复用（pool 取多大、用哪个模型、device 选什么，都是实测结论），
    而且 CLI 和评测都要走它。不碰 argparse，不 print，只用 logger。
"""

import logging
from typing import Protocol

from paper_rag.config import (
    RERANK_BATCH_SIZE,
    RERANK_DEVICE,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RETRIEVE_POOL,
    TOP_K,
)

logger = logging.getLogger(__name__)

__all__ = ["Reranker", "CrossEncoderReranker", "RerankedRetriever"]


# ────────────────────────────────────────────────────────────────
# 接口
# ────────────────────────────────────────────────────────────────


class Reranker(Protocol):
    """重排器的共同契约。

    和 Retriever 一样用 Protocol（结构化子类型），理由也一样：
    重排器不需要继承谁，只要长得一样。

    ⚠️ 同样【不做任何检查】。跑起来才炸。
    """

    # 表格里显示的名字，例如 "ce:bge-reranker-v2-m3"。
    name: str

    def rerank(
        self, query: str, candidates: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """把候选重排一遍。

        参数:
            query:      查询【文本】（和给 Retriever 的是同一个字符串，
                        ★ 包括查询扩展 —— 见 RerankedRetriever.search 的说明）
            candidates: [(chunk_id, 原分数), ...]，【可以乱序，也可以为空】

        返回:
            [(chunk_id, 新分数), ...]，按新分数【从高到低】。
            长度 = len(candidates) —— 重排【不增不减】，只换顺序。

        ⚠️⚠️ 新分数和原分数【不是一个东西】：
            原分数是余弦 / BM25 分；新分数是 cross-encoder 的 logit（可负、无上界）。
            调用方【不许】拿它和原分数比大小，也不许拿它去套旧的拒答阈值。
            它唯一的用途是【排序】。
        """
        ...


# ────────────────────────────────────────────────────────────────
# 实现：cross-encoder 重排
# ────────────────────────────────────────────────────────────────


# ── 进程级模型缓存 ──────────────────────────────────────────────

# key = (模型名, 设备, 截断长度) → 已经加载好的 CrossEncoder
_MODEL_CACHE: dict[tuple[str, str, int], object] = {}


def _load_model(model_name: str, device: str, max_length: int):
    """加载重排模型，同一个 (模型, 设备, max_length) 在进程里只加载一次。

    ★ 为什么缓存放在【模块级】，不是每个 CrossEncoderReranker 实例自己存一份：

        模型的权重是【只读】的。两个 reranker 指向同一个模型时，
        它们本来就【应该】共用同一份权重 —— 加载两遍不是"更隔离"，
        只是白白多占 2.3GB 内存、多等一次加载。

        这不是优化，是【语义修正】:权重是 (模型名, 设备) 的纯函数，
        和"谁在用它"无关。

    ★ 它真正防的是这个坑 —— 生产入口长这样：

            for q in questions:        # ← 循环
                ask(q, ...)

        如果 ask() 内部每次新建一个 reranker，而权重缓存在【实例】上，
        那就是【每问一句重新加载 2.3GB】。现象是「问答突然变得极慢」，
        而且【不报错、不警告】—— 正是本仓库反复踩的那种安静的错误。

    ⚠️ 但缓存【不能替代依赖注入】。生产入口仍然该在循环外把检索器
       建好再传进去（见 scripts/ask.py 和 pipeline.build_retriever）——
       除了权重，还有 id_to_text 那份引用和对象构造本身。
       缓存只是让「忘了注入」从【灾难】降级成【有点浪费】。
    """
    key = (model_name, device, max_length)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        logger.debug("重排模型 %s 已在缓存里，直接复用", model_name)
        return cached

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "要用 cross-encoder 重排，得先装 sentence-transformers：\n"
            "    .venv/Scripts/python.exe -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "    .venv/Scripts/python.exe -m pip install sentence-transformers\n"
            "★ 这样装的是【CPU 版 torch】（约 200MB）。"
            "别装默认的 CUDA 版（约 2.5GB）—— 那台 4070 的显存要留给 Ollama。"
        ) from e

    logger.info(
        "加载重排模型 %s（device=%s, max_length=%d）—— 第一次会下载约 2.3GB",
        model_name, device, max_length,
    )
    model = CrossEncoder(model_name, device=device, max_length=max_length)
    _MODEL_CACHE[key] = model
    logger.info("重排模型 %s 已就绪（本进程内后续复用）", model_name)
    return model


class CrossEncoderReranker:
    """cross-encoder 重排：把 (query, 正文) 【拼成一个序列】送进模型，出一个分。

    ── 它和 dense 检索的本质区别（这就是"cross"的意思）──

        dense（双塔 / bi-encoder）:
            query → 向量   ┐
                            ├→ 算余弦      ★ query 和正文【从头到尾没见过面】
            正文  → 向量   ┘

        cross-encoder:
            [query + 正文] → 一个数       ★ 在模型的【每一层】里互相 attend

        代价：cross-encoder 不能预计算 —— 正文的表示依赖于 query，
              所以【每条候选都得跑一遍模型】。1345 条全跑一遍是不现实的，
              但只跑 20 条候选就很划算。这就是"先粗排、再精排"的分工。

    ── 为什么 device 默认 cpu ──

        不是因为 cpu 快（它慢）。是因为那台 4070 的 12GB 显存
        【还要给 Ollama 用】—— qwen2:7b 生成时占约 4.5GB，
        再塞一个 568M 的 reranker 进去，早晚撞上显存墙，
        而撞上时的现象是 Ollama 那边超时或 OOM，【不报 rerank 的错】。

        ★ 实测耗时（2026-10-12，单进程独占 CPU，torch 2.14.1+cpu，14 线程，
          max_length=512，同一个问题 + 同一批候选，取 best of N）：

              用本模块的默认值 batch_size=4：
                  pool= 20    4.8s        pool= 50   10.6s        pool=100  21.1s

          ★ 28 题全量跑下来是 **10.6 秒/题**（pool=50），和上面单题的 10.60s
            对上了 —— 所以这一列不是特例，就是平均值。

          ⚠️⚠️ 而【上游默认的 batch_size=32 会慢一倍】：同样 pool=50，32 → 18.9s。
            理由见 config.RERANK_BATCH_SIZE，一句话：
            `predict` 内部按长度【交错】分批，那是为
            「GPU + flash attention ⇒ 没有 padding」调的；
            我们在 CPU 上 padding 要真算，交错等于【每批都按最长的补齐】。
            ★ 这个改动【不改分数】—— 实测全序一致，|Δ| 在 4e-7（float32 舍入）。

          ★ 别改用调 max_length 来提速。那是【截断】，会丢真内容：
            512→256 掉了一题 recall@5；512→128 表面上指标更好，
            但那是噪声（就 2 题，且集中在已怀疑 ground truth 的题上）。
            **一条路省的是白算的部分，另一条省的是真内容 ——
            光看数字分不出来**，见 S5-5 记录 4.6。

          ★ 预热（模型 mmap + 第一次前向）另要 ~9.5 秒，【只付一次】。
            所以生产入口必须把检索器建在循环【外】，见 pipeline.build_retriever。

          哪天还嫌慢：改 config.RERANK_DEVICE（★ 要先重算 Ollama 的显存预算），
          或者重新量 RERANK_BATCH_SIZE / RETRIEVE_POOL ——
          ★ 后两个都和机器/系统绑着，换一个就得重量（见 config 里的说明）。
    """

    def __init__(
        self,
        id_to_text: dict[str, str],
        *,
        model: str = RERANK_MODEL,
        device: str = RERANK_DEVICE,
        max_length: int = RERANK_MAX_LENGTH,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> None:
        """参数:
        id_to_text: chunk_id → 正文。rerank 要用它把 id 换回文本。
        model:      HuggingFace 模型名
        device:     "cpu" / "cuda"
        max_length: 截断长度。512 —— 见下面 ★ 为什么【不】调小。
        batch_size: 一次前向喂几条。★ 这个参数【只影响速度，不影响分数】，
                    而且是这里唯一一个这样的参数。默认 4 而不是上游的 32。
                    完整理由见 config.RERANK_BATCH_SIZE。

        ★ 模型【不在这里加载】。见 _ensure_model。
        ★ batch_size 【不进模型缓存键】—— 它不改模型实例，只改调用方式。
        """
        self.id_to_text = id_to_text
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def name(self) -> str:
        """表格里的名字。device 带上 —— cpu 和 cuda 的耗时差一个量级，
        看表的人有权知道这一行是哪个。"""
        return f"ce:{self.model_name.split('/')[-1]}:{self.device}"

    @classmethod
    def from_chunks(
        cls, chunks: list[dict], **kw
    ) -> "CrossEncoderReranker":
        """从 chunk 列表建。

        ★ 注意它【不需要 client】—— 这是它和 DenseRetriever 一样、
          BM25Retriever 也一样的那个自由：Protocol 只约束 search/rerank，
          构造函数各随各的（见 retriever.BM25Retriever.from_chunks 的说明）。
        """
        return cls({c["chunk_id"]: c["text"] for c in chunks}, **kw)

    def _ensure_model(self):
        """第一次用到时才加载模型。

        ★ 为什么懒加载：
            ① 模型 2.3GB + import sentence_transformers 要好几秒。
               只跑 BM25 的评测不该付这个钱。
            ② 真正的原因是【报错要好懂】——
               sentence-transformers 没装时，import 会抛一堆 torch 的栈，
               看不出"你只是少装了一个包"。这里把它翻译成人话。

        ★ 真正的加载在模块级 _load_model 里，带进程级缓存。
          这里只留一个实例级的小备忘，省掉一次字典查找。
        """
        if self._model is None:
            self._model = _load_model(self.model_name, self.device, self.max_length)
        return self._model

    def rerank(
        self, query: str, candidates: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """逐对打分，按新分数降序返回。

        参数:
            query:      查询文本
            candidates: [(chunk_id, 原分数), ...]

        返回:
            [(chunk_id, rerank 分数), ...]，降序，长度不变。

        ★ 原分数【被丢掉了】。这不是浪费 —— 它已经完成了它的任务
          （决定谁进这 20 条候选），进了之后就没有发言权了。
          真要留着它做加权融合，那是另一个实验，别偷偷混进来。
        """
        if not candidates:
            return []

        # ★ id → 文本查不到时【抛错】，不跳过。
        #   跳过的后果是：这条候选静默消失，而现象只是「答案偶尔变差」。
        #   和 index.build 那条「宁可抛错也不静默跳过」是同一条规矩。
        missing = [cid for cid, _ in candidates if cid not in self.id_to_text]
        if missing:
            raise KeyError(
                f"这些 chunk_id 在 id_to_text 里找不到正文：{missing[:5]}\n"
                f"多半是 reranker 和 retriever 用的【不是同一批 chunks】。"
            )

        model = self._ensure_model()
        pairs = [(query, self.id_to_text[cid]) for cid, _ in candidates]
        # ★ batch_size 必须显式传。上游默认 32，而那是在【GPU + flash attention
        #   ⇒ 没有 padding】的前提下选的（见 predict 内部的 _interleave_sorted_indices）。
        #   我们在 CPU 上跑，padding 是实打实要算的 —— 实测 32 比 4 慢 81%。
        #   ★ 它【不改分数】：predict 是逐对打分，批大小不参与任何数学。
        #     实测 max|Δ分| = 4e-7（float32 舍入），三个查询的【全序】完全一致。
        scores = model.predict(pairs, batch_size=self.batch_size)

        ranked = [(cid, float(s)) for (cid, _), s in zip(candidates, scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# ────────────────────────────────────────────────────────────────
# 组合：检索器 + 重排器 = 还是一个检索器
# ────────────────────────────────────────────────────────────────


class RerankedRetriever:
    """把「先捞宽、再精排」包成一个【普通检索器】。

    它满足 Retriever 协议的【全部】约定 —— 一共就两个：name 和 search。
    （from_chunks 不是约定，只是几个实现顺手加的便利构造器，协议里没有它。）
    所以 eval_retrieve.py 和 pipeline.ask() 拿它当 DenseRetriever 用就行，
    它们【一行都不用改】。

    ── 为什么 pool 和 k 是两个参数，而且 pool 必须 >= k ──

        想象 pool == k == 5：
            粗排捞 5 条，重排【只在这 5 条里调顺序】。
            那么第 6 名永远上不来 —— 而正确答案排第 6 是常态。
            重排的全部价值就在于"把后面好的提上来"，pool == k 等于把这个价值归零。

        ★ 这就是本文件里最容易犯的错，而且【它不报错】：
          数字会动、会好看一点点（因为在 5 条里重排确实能改善一点），
          但它把 rerank 的效果严重低估了，而你不会知道。
          所以 search() 里有一条断言。
    """

    def __init__(
        self,
        base,
        reranker: Reranker,
        *,
        pool: int = RETRIEVE_POOL,
        k: int = TOP_K,
    ) -> None:
        """参数:
        base:     被包住的检索器（任何满足 Retriever 的东西）
        reranker: 重排器（任何满足 Reranker 的东西）
        pool:     粗排捞多少条送给重排器
        k:        最终返回几条

        ★ base / reranker 都没有类型标注成 Retriever / Reranker。
          不是偷懒 —— 是 Protocol 在运行时【本来就不存在】，
          标上去只对读代码的人有意义，运行时一样不做检查。
          真要检查，得自己写 isinstance 或跑 mypy。这一点在
          retriever.py 的 Protocol 注释里已经警告过了。
        """
        if pool < k:
            raise ValueError(
                f"pool={pool} 小于 k={k}：重排器只会看到 {pool} 条，"
                f"却要求它交出 {k} 条 —— 它变不出来。"
            )
        if pool == k:
            logger.warning(
                "pool=%d 和 k=%d 一样大：重排只能在 %d 条里换顺序，"
                "第 %d 名永远上不来。重排的价值大部分来自「把后面的提上来」，"
                "所以这多半不是你想要的那个实验。",
                pool, k, pool, pool + 1,
            )
        self.base = base
        self.reranker = reranker
        self.pool = pool
        self.k = k

    @property
    def depth(self) -> int:
        """这个检索器【最深能排到第几名】= pool。

        ★ 为什么要有这个属性：它不是所有检索器都有意义的 ——
          DenseRetriever 的深度是语料总数（1345），BM25 的深度因题而异
          （候选池大小，实测 0~646）。

          重排器不一样：**它的排序只定义在 pool 条之内。**
          排在 pool 之外的东西，它【不是排得靠后，是压根没有名次】。

          所以评测脚本问它要"全量排名"时，它只能给 pool 条 ——
          这是它的诚实回答，不是它偷懒。见 search() 的说明。
        """
        return self.pool

    @property
    def name(self) -> str:
        """例：`dense:bge-m3:latest:raw > ce:bge-reranker-v2-m3:cpu @pool20`

        ★ pool 必须显示出来。同一个 reranker 配 pool=5 和 pool=20
          是【两个不同的系统】，表格里长得一样的话，那张表就是骗人的。
        """
        return f"{self.base.name} > {self.reranker.name} @pool{self.pool}"

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """① 粗排捞 pool 条 → ② 重排 → ③ 取前 k 条。

        参数:
            query: 查询文本
            k:     取前几名。不传就用构造时那个。

        返回:
            [(chunk_id, rerank 分数), ...]，降序。
            ★ 分数是 cross-encoder 的，不是余弦 —— 见 Reranker.rerank 的警告。
            ★ 长度 = min(k, pool, 粗排捞到的条数)。**可能小于 k。**

        ⚠️ 当 k > pool 时【不报错，只返回 pool 条】—— 这是刻意的：

            评测脚本拿 `k = 语料总数` 来要「全量排名」（见 eval_retrieve.evaluate）。
            对 dense 那是有意义的（它真的排 1345 条），
            对重排器【没有意义】—— 重排器的全量排名只有 pool 深。

            这时正确的回答是"我只能给你 pool 条"，而不是抛错，
            也不是假装排了 1345 条。
            调用方拿到一个短列表就能看出这一点。

        ⚠️ 【query 原样传下去，不做任何加工】
            查询扩展在更上游做过一次（pipeline.ask / eval_retrieve），
            这里拿到什么就传什么。
            重排器【必须看到和粗排完全一样的 query】，否则两边在回答
            两个不同的问题 —— 而这不会报错，只会让分数变得没道理。
        """
        k = self.k if k is None else k

        # ★ 要得比 pool 还深：说一声，然后给 pool 条。
        #   用 debug 不用 warning：这是评测脚本的常规用法，不是异常。
        if k > self.pool:
            logger.debug(
                "要求 %d 条，但重排器只有 %d 条深 —— 返回它有的那么多", k, self.pool
            )
            k = self.pool

        candidates = self.base.search(query, k=self.pool)

        # ★ 粗排返回空（BM25 碰上纯中文 query）就直接返回空，
        #   不要往下走 —— 让 reranker 处理空列表是没意义的开销，
        #   而且「空结果」这个状态要能被调用方看见（见 eval_retrieve.py）。
        if not candidates:
            return []

        return self.reranker.rerank(query, candidates)[:k]
