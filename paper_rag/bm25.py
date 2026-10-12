"""BM25 —— 基于【字面】的检索。纯算法，不依赖本项目任何东西。

── 和稠密向量检索的本质区别 ──

    dense  比的是【意思】。query 和文档可以一个词都不重合。
           「双曲空间」能对上 "hyperbolic space"，因为嵌入模型见过这两种说法。
    BM25   比的是【字面】。词必须真的出现，差一个字母就不算。

    这不是「谁更好」，是【两种不同的失败方式】：
        dense  能跨语言、跨同义词 —— 但也会把「意思近、事实不对」的段落排上来
        BM25   命中就是命中，不会瞎猜 —— 但换个说法就完全找不到

    所以融合（RRF）才有意义：它们的错误【不相关】。
    两个总是一起错的系统融合起来没用；两个错法不同的系统融合才有增益。

── BM25 在算什么 ──

    给每个 (query 词, 文档) 打一个分，加起来：

        score(q, d) = Σ  IDF(t) · ─────────────────────────────
                                  tf + k1·(1 − b + b·|d|/avgdl)

             tf·(k1 + 1)
          = Σ ──────────────────────────  · IDF(t)
               tf + k1·(1 − b + b·|d|/avgdl)

    三个部件各管一件事：

        IDF(t)   这个词有多稀罕。满大街都是的词（the / of）权重趋近 0。
                 ★ 这就是为什么【不需要停用词表】——
                   BM25 用 IDF 自动把 "the" 压到几乎为零，
                   而停用词表是手工的、换语料就得重写。
                   能用统计解决的事，别用手写清单。

        tf       这个词在本文档里出现几次。但【不是线性】——
                 第 1 次出现很重要，第 10 次和第 11 次几乎没差别。
                 k1 控制饱和速度：k1 越大越接近线性，越小饱和越快。
                 这一项的分子分母都在，所以 tf→∞ 时它收敛到 (k1+1)，不会爆。

        b·|d|/avgdl  长度归一化。长文档天然容易命中更多词，
                     不惩罚它的话，长文档会垄断所有结果。
                 b=1 完全归一化，b=0 不归一化。0.75 是几十年的经验值。

    k1=1.5 / b=0.75 是文献里的默认值，也是本文件的值。
    ★ 但它们【不是普适常数】—— 见「不应该怎么写」那一节。

── ★ 中文 query 会怎样 ──

    这个语料是【英文论文】，而评测集是【中文问题】。

    tokenize() 只切 ASCII 字母数字，所以：

        "TransE 的核心思想是什么？"        → ["transe"]
        "静态知识图谱推理的模型按技术分成哪三类？"  → []        ← 一个词都没有

    空 query 意味着 BM25 **什么都找不到** —— 不是"排得靠后"，是"完全无能"。

    ★ 这是【查询扩展】对这两个检索器的意义完全不同：
        dense  扩展是加分项（bge-m3 本来就多语言，不扩展也能跨）
        BM25   扩展是【生死线】（不扩展 = 空 query = 0 分）

    所以评测必须把 expand 当成一个【独立变量】量，不能只量一种。
    （见 scripts/eval_retrieve.py 的 --expand）
"""

import math
import re
from collections import Counter, defaultdict

__all__ = ["tokenize", "BM25", "K1", "B"]

# 文献默认值。改动它们之前先读模块开头 —— 这两个数是在别的语料上量的。
K1 = 1.5
B = 0.75

# ★ 只切 ASCII 字母数字。
#
# 为什么不用 text.split()：
#   "models," 和 "models" 会变成两个不同的词 —— 标点必须切掉，
#   否则 "KGE." 和 "KGE" 匹配不上。
#
# 为什么不用 nltk / jieba：
#   1. 多一个依赖，而这里的活是 20 个字符的正则能干的
#   2. 中文语料才需要分词器，而我们的语料是英文。
#      真加了 jieba，它会兴高采烈地把中文 query 切成词 ——
#      而那些词【在英文语料里一个字都不会出现】，白切。
#      切出来能匹配上才有用，切出来匹配不上不如不切。
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """把文本切成检索用的词。

    转小写 → 切出连续的字母数字串。中文和标点会被丢掉。

    参数:
        text: 任意文本

    返回:
        词列表。★ 可能是【空列表】—— 纯中文的 query 就是这个结果。
        调用方不能假设它非空。

    例:
        "TransE is one of the first KGE models."  → ["transe", "is", "one", "of",
                                                     "the", "first", "kge", "models"]
        "静态知识图谱推理的模型按技术分成哪三类？"    → []
    """
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """BM25 倒排索引 —— 建一次，查多次。

    ── 为什么要「建」──

        原始做法是每来一个 query，就把全部 N 篇文档扫一遍、现算 tf。
        N 篇 × 每篇 L 个词 = O(N·L)，而且这活儿每次查询都在重复。

        倒排索引把这件事【提前做完】：
            term → [(文档号, 出现次数), ...]
        查询时只查 query 里那几个词，O(命中次数)，和语料总大小无关。

        ★ 这和 index.py 把向量【提前算好存盘】是同一个思路：
          把「和 query 无关的那部分」提前算完。
          不同的只是：向量那份要存到磁盘，倒排这份在内存里现建（几秒）。

    ── 它不拥有 ids，只借用 ──

        ids[i] 描述的是第 i 篇文档。向量那边也一样 ——
        行序是唯一的纽带，所以两边的顺序必须来自同一份 chunks 列表。
    """

    def __init__(
        self,
        texts: list[str],
        ids: list[str],
        *,
        k1: float = K1,
        b: float = B,
    ) -> None:
        """建索引。

        参数:
            texts: 文档正文列表
            ids:   对应的 chunk_id，长度必须和 texts 一致
            k1:    tf 饱和参数
            b:     长度归一化强度

        异常:
            ValueError  ids 和 texts 长度对不上。
            ★ 这个检查不能省：错位之后每个 id 都指向另一条语料，
              而且【不会报错】—— 只会安静地返回错误的结果。
              （和 DenseRetriever.__init__ 里那个断言是同一个理由。）
        """
        if len(ids) != len(texts):
            raise ValueError(f"ids 有 {len(ids)} 个，texts 有 {len(texts)} 个 —— 对不上")

        self.k1 = k1
        self.b = b
        self.ids = ids
        self.n = len(texts)

        self.doc_len: list[int] = []
        # term → [(文档下标, 该文档里出现次数), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter = Counter()

        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.doc_len.append(len(toks))
            for term, freq in Counter(toks).items():
                self.postings[term].append((i, freq))
                df[term] += 1

        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

        # BM25 的 IDF（带平滑，保证恒为正）：
        #     IDF(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
        #
        # ★ 里面的 1 + 是【必须】的，不是修饰。
        #   不带它的经典形式是 ln((N − df + 0.5)/(df + 0.5)，
        #   当 df > N/2（词出现在超过一半文档里）时它是【负数】——
        #   一个词越常见、得分越低，低到把总分成负数。
        #   加 1 之后 IDF 永远 ≥ 0，常见词只是"贡献接近 0"，不会倒扣。
        #
        # ★ 注意 IDF 是【全语料共享】的，和 query 无关 ——
        #   所以它完全可以提前算好。这正是倒排索引能省事的地方。
        self.idf: dict[str, float] = {
            t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """返回得分最高的 k 条。

        参数:
            query: 查询文本
            k:     取前几名

        返回:
            [(chunk_id, 分数), ...]，按分数【从高到低】

        ★ 返回长度【可能小于 k，也可能是空】：
            空 query（纯中文）          → []
            query 里的词一个都没出现过   → []
          这和 dense 那边「永远返回 min(k, N) 条」不一样 ——
          因为 dense 总能算出 N 个余弦值，而 BM25 只对【有交集的文档】打分。
          没交集就是没交集，编不出分数来。
        """
        terms = tokenize(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)

        # ★ 用 set 去重：query 里重复的词不该加两次分。
        #   第 2 次出现同样的词，不提供新信息 —— 和 tf 饱和是同一个道理。
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue

            idf = self.idf[term]
            for i, tf in postings:
                # 分母：tf 饱和 + 长度归一化
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom

        # 打平时按文档下标排 —— 保证可复现。
        # （sorted 本来就是稳定的，但显式写出来，读的人不用去查这个保证。）
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self.ids[i], s) for i, s in ranked[:k]]
