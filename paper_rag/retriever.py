"""检索器 —— 把「怎么找」抽成一个统一接口，多种实现。

── 为什么要有这一层 ──

    Stage 5 的正题是【比较几种检索方式】：
        纯向量 / 纯 BM25 / 两者融合(RRF) / 融合后再重排

    没有统一接口的话，每加一种，评测脚本就多一个分支；
    分支之间还会互相抄代码。这个仓库已经吃过一次抄代码的亏 ——
    V3 的 `title + text` 写在 index.py 和 eval_embed.py 两处，
    而 ids.json 只记模型名、不记文本方案：改漏一处【不会有任何东西报错】。

    所以：先定接口，再往里塞实现。

── 接口为什么是 search(query: str)，不是 search(query_vec: list[float]) ──

    ★ 这是本文件里唯一一个不能随便改的决定。

    BM25 根本没有 query 向量 —— 它靠词频和倒排表。
    接口要是收向量，BM25 就【实现不了这个接口】，
    只能在外面包一层假的，这个抽象当场失去意义。

    收文本，则每个实现自己决定怎么理解它：
        DenseRetriever   嵌成向量 → 算余弦
        BM25Retriever    分词     → 查倒排表（Stage 5 下一步）
    将来加 rerank / HyDE 也一样。

    另一条路是「接口收向量，BM25 那边传 None」——
    那种接口叫「只对一半实现成立」，是抽象漏了，不是抽象。

── 它在流水线的哪个位置 ──

    build_index.py     PDF          → chunks.json      （离线，跑一次）
    build_vectors.py   chunks.json  → vectors.npy      （离线，跑一次）
                                          │
                                          ▼
    ┌────────────────────────────────────────────────────────────┐
    │  retriever.DenseRetriever.search(query, k)                 │  ← 本文件
    │    query 文本 → 加 query 前缀 → 嵌成向量 → top_k_np         │
    └────────────────────────────────────────────────────────────┘
                    │                            │
                    ▼                            ▼
         pipeline.ask()  【产品】        eval_retrieve.py 【脚手架】

── 这是【产品】还是【脚手架】？──

    产品。三条理由：
      1. 【不是一次性的】CLI 要走它，以后 web 界面要走它，评测也要走它。
         一次性的东西才叫脚手架。
      2. 【含判断】query 前缀加不加、加什么，是实测结论不是格式转换
         （bge-m3 是对称模型，加了反而掉分）。判断可复用，格式不可复用。
      3. 【在 paper_rag/ 里】不碰 argparse，不 print，只用 logger。

    scripts/eval_retrieve.py 才是脚手架：喂参数、算指标、打表格。
"""

import logging
from typing import Protocol

import numpy as np

from paper_rag.bm25 import B, K1, BM25
from paper_rag.config import TOP_K
from paper_rag.embed import embed_one, embed_texts
from paper_rag.retrieve import top_k_np

logger = logging.getLogger(__name__)

__all__ = ["Retriever", "DenseRetriever", "BM25Retriever", "prefix_for"]


# ────────────────────────────────────────────────────────────────
# 接口
# ────────────────────────────────────────────────────────────────


class Retriever(Protocol):
    """所有检索器的共同契约。

    这里用 Protocol 而不是抽象基类（ABC），是因为我们【不需要】它们
    继承同一个父类 —— 只要长得一样就行。这叫「结构化子类型」：

        DenseRetriever 不知道 Retriever 的存在，照样能当 Retriever 用。

    好处是 BM25Retriever 这个类可以完全不知道 dense 那边是怎么写的，
    甚至可以是别人写的第三方类，只要它有个 search(query, k)。

    ⚠️ 这个 Protocol 【不检查任何东西】，它只是一份写下来的约定。
       写错了 Python 不会拦你，跑起来才炸 —— 所以下面那两行注释很重要。
    """

    # 用来在人看的表格里区分是哪个检索器，例如 "dense:bge-m3:latest"。
    # 约定是「策略:参数」，冒号分隔 —— 将来 "bm25:k1=1.5" 也认。
    name: str

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        """给一段【文本】，返回最像的 k 条。契约和 retrieve.top_k* 三个实现一致。

        参数:
            query: 查询文本。★ 是文本不是向量 —— 理由见模块开头。
            k:     取前几名

        返回:
            [(chunk_id, 分数), ...]，按分数【从高到低】，长度 = min(k, N)

        ⚠️ 【分数在不同检索器之间不可比】
            dense 给的是余弦，范围 [-1, 1]；
            BM25 给的是无上界的词频得分，可能 0.0，也可能 37.2。
            两个都「越大越好」，但 0.8 和 0.8 完全不是一回事。

            所以融合（RRF）才要【按排名】融，而不是按分数加权平均 ——
            那一步在 Stage 5 后面，这里先把坑标出来。
        """
        ...


# ────────────────────────────────────────────────────────────────
# query 前缀表
# ────────────────────────────────────────────────────────────────

# 有些嵌入模型是【非对称】的：query 要加一句任务说明，文档【不加】。
#   qwen3-embedding     Qwen 家要求 "Instruct: ...\nQuery: ..."
#   bge-large-zh-v1.5   BGE 家早期版本要求中文指令
#   bge-m3              对称，什么都不加（加了反而伤）
#
# ★ 这张表原来只活在 scripts/eval_embed.py 里，生产路径压根没有前缀这个概念
#   （embed.embed_one 的注释直接写死「query 侧不加任何指令前缀」）。
#   两边今天是【碰巧】一致的 —— 因为生产用的 bge-m3 前缀是空串。
#   哪天把 config.EMB_MODEL 换成 qwen3-embedding，
#   评测会加上前缀、生产不会，而【两边都不报错】。
#   这就是把这张表搬进产品层的原因：让前缀跟着模型走，而不是跟着脚本走。
#
# 注意 key 用【子串】匹配，不是 startswith：
#   "qwen3-embedding:0.6b"              → 能配上
#   "quentinz/bge-large-zh-v1.5:latest" → 只有子串匹配才配得上
# 用 startswith 的话第二条会静默漏掉 —— 不报错，只是那一行分数偏低。
QUERY_PREFIX: dict[str, str] = {
    "qwen3-embedding": (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query.\nQuery: "
    ),
    "bge-large-zh": "为这个句子生成表示以用于检索相关文章：",
}


def prefix_for(model: str) -> str:
    """查这个嵌入模型的 query 前缀，没有就返回空串。

    ★ 遗留疑问（2026-10-12，未解决）：这张表和 embed.embed_one 的注释【互相打架】——
        embed_one 说「bge-large-zh 加官方前缀后 Top-1 从 40% 掉到 30%」，
        这张表说「bge-large-zh-v1.5 要求中文指令」。
        两句必有一句是错的（或者是旧版本/新版本的差别）。
        现在生产用的是 bge-m3，两条都用不上，所以没炸。
        要动 bge-large-zh 之前，先把这两个数字重新量一遍。

    参数:
        model: 嵌入模型名，如 "bge-m3:latest"

    返回:
        前缀字符串；这个模型不需要前缀就返回 ""（空串，不是 None ——
        调用方直接 `prefix + query` 就行，不用判空）
    """
    for key, p in QUERY_PREFIX.items():
        if key in model:
            return p
    return ""


# ────────────────────────────────────────────────────────────────
# 实现一：稠密向量检索
# ────────────────────────────────────────────────────────────────


class DenseRetriever:
    """纯向量检索 —— 把 query 嵌成向量，和全部语料向量比余弦。

    这就是 Stage 2 那条路，只是现在被装进了一个有名字的东西里，
    好和将来的 BM25 / RRF 摆在同一张表上比。

    ⚠️ 【它不拥有语料，只借用】
        向量和 id 由外面传进来（通常是 index.load() 读回来的）。
        这样评测可以拿同一批向量建好几个检索器，不用重嵌一遍。
    """

    def __init__(
        self,
        client,
        model: str,
        vectors,
        ids: list[str],
        *,
        query_prefix: str | None = None,
    ) -> None:
        """参数:
        client:       ollama 客户端
        model:        嵌入模型名 —— 必须和建 vectors 时用的【同一个】
        vectors:      np.ndarray，形状 (N, D)
        ids:          list[str]，长度 N，和 vectors 的行一一对应
        query_prefix: query 侧前缀。不传就按 model 自动查表（prefix_for）。
                      ★ 单独留这个口子，是为了将来能拿它做 A/B ——
                        「前缀到底有没有用」在这个仓库里的说法是矛盾的，
                        要量就得能量两个方向。
        """
        # ★ 这个断言不能省。
        #   retrieve.top_k_np【不会替你查】ids 和 vectors 对不对得上 ——
        #   它拿 argsort 出来的下标直接去索引 ids，错位了它照样返回，
        #   只是返回的 id 属于另一条语料。不报错，结果是错的。
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids 有 {len(ids)} 个，向量有 {vectors.shape[0]} 行 —— 对不上。\n"
                f"这多半是 ids.json 和 vectors.npy 不同步，重建："
                f"python scripts/build_vectors.py"
            )

        self.client = client
        self.model = model
        self.vectors = vectors
        self.ids = ids
        # None 是「没传」，"" 是「明确不要前缀」—— 两者必须分得开，
        # 否则将来没法用它测「强制不加前缀」这一组。
        self.query_prefix = prefix_for(model) if query_prefix is None else query_prefix

    @property
    def name(self) -> str:
        """表格里显示的名字。前缀的存在与否也带上，否则 A/B 两组看不出区别。"""
        tag = "prefix" if self.query_prefix else "raw"
        return f"dense:{self.model}:{tag}"

    @classmethod
    def from_chunks(cls, client, model: str, chunks: list[dict]) -> "DenseRetriever":
        """把一批 chunk 全部向量化，建好检索器。【评测用】

        为什么不复用 index.build()：
            index.build 的文本方案写死在它自己那儿（`c["text"]`），
            而这个方法将来要跟 BM25 共用同一份【分词/预处理】。
            等 Stage 5 走到「两种检索器看同一份文本」那一步时会统一，
            现在先用 embed_texts 直连，保持两边都看得见自己在嵌什么。

        参数:
            client: ollama 客户端
            model:  嵌入模型名
            chunks: chunk 列表，每条至少要有 "chunk_id" 和 "text"

        异常:
            有 chunk 嵌不进去就抛 RuntimeError。★ 不静默跳过 ——
            少一条语料，检索时就永远找不到它，而现象只是「偶尔答不上来」。
        """
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]

        vectors, failed = embed_texts(client, model, texts)

        if failed:
            preview = "\n".join(f"    {len(texts[i]):5} 字符  {ids[i]}" for i in failed[:5])
            raise RuntimeError(
                f"{len(failed)} / {len(texts)} 条嵌不进去，模型 {model}：\n{preview}\n"
                f"（多半是上下文超了。看看是不是该换模型，或者该加 MAX_CHUNK_CHARS 截断）"
            )

        return cls(client, model, np.array(vectors, dtype=np.float32), ids)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        """把 query 嵌成向量，取余弦最高的 k 条。

        参数:
            query: 查询【文本】（会先拼上前缀）
            k:     取前几名

        返回:
            [(chunk_id, 分数), ...]，分数是余弦，范围 [-1, 1]，降序

        ★ 每一步在干嘛：
            ① 拼前缀 —— 骗嵌入模型的，不是给用户看的
            ② embed_one —— 只嵌 1 条，语料向量早就建好了
            ③ top_k_np  —— 一次矩阵乘法 (N,D)@(D,) → (N,)，再取前 k
        """
        # ① 前缀拼在【query】侧，文档侧不拼 —— 这就是「非对称检索」。
        #    bge-m3 对称，所以 self.query_prefix 是 ""，这一行等于没做。
        #    看着像废话，但它是【唯一】一处决定前缀加不加的地方。
        text = self.query_prefix + query

        # ② 语料侧不重嵌 —— 那是离线阶段 build_vectors.py 干的事
        qvec = embed_one(self.client, self.model, text)

        # ③ 契约和 retrieve.top_k / top_k_faiss 逐位一致
        return top_k_np(qvec, self.vectors, self.ids, k=k)


# ────────────────────────────────────────────────────────────────
# 实现二：BM25 字面检索
# ────────────────────────────────────────────────────────────────


class BM25Retriever:
    """BM25 字面检索 —— 词真的出现才算命中。

    ── 这个类证明了 S5-1 那个接口决定是对的 ──

        往下看，这个类里【没有 embed_one、没有向量、没有 numpy】。
        它拿到 query 之后做的事是：分词 → 查倒排表 → 加分数。

        如果 Retriever 接口当初收的是 `query_vec: list[float]`，
        这个类根本写不出来 —— 它没有向量可收。

        ★ 一个接口设计对不对，不是看第一个实现写起来顺不顺，
          是看【最不像的那个实现】能不能进来。

    ── 它的输入和 dense 一样是 chunk["text"] ──

        刻意保持一致，这样「dense vs bm25」的对比里只有【检索方式】一个变量。
        哪天想让 BM25 也看标题，那是另一个实验（标题对 BM25 大概率有用，
        因为标题里的术语密度高）—— 但要单独量，别混进这一组。
    """

    def __init__(
        self,
        texts: list[str],
        ids: list[str],
        *,
        k1: float = K1,
        b: float = B,
    ) -> None:
        """参数:
        texts: 文档正文列表
        ids:   对应的 chunk_id
        k1:    tf 饱和参数（默认 1.5）
        b:     长度归一化强度（默认 0.75）
        """
        self.index = BM25(texts, ids, k1=k1, b=b)

    @property
    def name(self) -> str:
        """★ 把 k1 / b 写进名字里。

        理由和 DenseRetriever 带 `:raw` / `:prefix` 一样：
        将来调参做 A/B 时，两行在表格里必须能分开。
        名字写成 `"bm25"` 的话，`k1=1.5` 和 `k1=2.0` 两组长得一模一样。
        """
        return f"bm25:k1={self.index.k1},b={self.index.b}"

    @classmethod
    def from_chunks(cls, chunks: list[dict], **kw) -> "BM25Retriever":
        """从 chunk 列表建索引。【评测用，不需要客户端】

        参数:
            chunks: chunk 列表，每条至少要有 "chunk_id" 和 "text"
            **kw:   透传给 __init__（k1 / b）

        ★ 注意签名和 DenseRetriever.from_chunks 不同 —— 它不收 client。
          BM25 不需要调模型，所以不需要客户端。
          ★ 接口只约束 search()，不约束构造函数 ——
            这正是 Protocol 而不是 ABC 的好处：构造函数可以各随各的。
        """
        return cls([c["text"] for c in chunks], [c["chunk_id"] for c in chunks], **kw)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        """查倒排表，返回得分最高的 k 条。

        参数:
            query: 查询文本
            k:     取前几名

        返回:
            [(chunk_id, 分数), ...]，分数是 BM25 分，【无上界】。

        ⚠️ 返回长度可能 < k，也可能是空：
            纯中文 query（一个 ASCII 词都没有）→ []
            query 的词在语料里一个都没出现     → []

          ★ 空结果不是 bug，是这个检索器的【诚实回答】：
            "按字面我找不到任何东西。"
            但调用方（评测）必须把它和「找到了但排得靠后」分开记 ——
            混在一起的话，空结果会被当成"排名很靠前"。
        """
        return self.index.search(query, k)
