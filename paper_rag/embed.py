"""向量化 —— 文本 → 向量。

这一层【只做一件事】：调 Ollama，把一批文本换成一批向量。
它不知道 chunk 是什么，也不知道索引长什么样。

为什么单独拆成模块：
    eval_embed.py（评测模型）和 build_vectors.py（建索引）都要嵌文本。
    代码抄两份的话，将来改一处忘一处 —— 比如 truncate=False 那个坑，
    漏写一次就会让其中一个脚本静默截断，数字全是错的。
"""

import logging

import ollama

from paper_rag.config import OLLAMA_HOST

logger = logging.getLogger(__name__)

# 一次喂多少条给 Ollama。
# 不是越大越好：整批 = 一条 HTTP 请求，一条失败整批就失败。
# 32 是「请求数少」和「失败爆炸半径小」之间的折中。
BATCH = 32


def make_client() -> ollama.Client:
    """建一个 Ollama 客户端。

    返回:
        ollama.Client，指向 config.OLLAMA_HOST
    """
    return ollama.Client(host=OLLAMA_HOST)


def embed_texts(client, model: str, texts: list[str]) -> tuple[list, list[int]]:
    """批量嵌入。返回 (向量列表, 失败的原始下标)。

    某一批失败时，把那一批【拆成单条】重跑 ——
    这样能精确定位是哪几条喂不进去，而不是整批丢掉。
    （实测真有这种：bge-large-zh 上下文 512，Abstract 那条 1606 字塞不下。）

    参数:
        client: ollama 客户端
        model:  模型名
        texts:  待嵌入的文本列表

    返回:
        (vectors, failed_idx)
        vectors 里失败的位置填 None —— 保持下标和 texts 对齐。
        对齐很重要：错位了，后面每一条 [向量 ↔ chunk] 的对应关系都是错的。
    """
    vectors: list = [None] * len(texts)
    failed: list[int] = []

    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        idxs = list(range(start, start + len(batch)))
        try:
            # truncate=False 必须显式写！
            #
            # 实测：ollama 的 embed() 默认【静默截断】——
            #   把 1606 字的 Abstract 喂进去，它不报错，
            #   悄悄只用了前 1480 字，返回的向量跟「只喂前 1480 字」一模一样。
            #   而旧的 embeddings() 接口在同样输入下会报错。
            #
            # 不写这个参数，调用方就会看到「失败 0 条」——
            # 工具撒了谎，我们却以为语料没问题。
            resp = client.embed(model=model, input=batch, truncate=False)
            for i, v in zip(idxs, resp.embeddings):
                vectors[i] = v
        except Exception:
            # 整批失败 —— 拆开逐条跑，找出真正的罪魁
            for i, t in zip(idxs, batch):
                try:
                    vectors[i] = client.embed(model=model, input=t, truncate=False).embeddings[0]
                except Exception:
                    failed.append(i)

    return vectors, failed


def embed_one(client, model: str, text: str) -> list[float]:
    """单条嵌入。query 侧用这个（每次提问只嵌 1 条，不批量）。

    参数:
        client: ollama 客户端
        model:  模型名
        text:   要嵌入的文本

    返回:
        list[float]，长度 = 模型的输出维度

    注意:
        query 侧【不加任何指令前缀】—— bge-m3 是对称模型。
        加了前缀会让 query 和文档落在不同的位置分布上，反而掉分。
        这是实测结论，不是惯例：bge-large-zh 加官方前缀后，Top-1 从 40% 掉到 30%。
    """
    return client.embed(model=model, input=text, truncate=False).embeddings[0]
