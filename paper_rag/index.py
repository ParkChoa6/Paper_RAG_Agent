"""向量索引 —— 建、存、读。

为什么要有索引：
    237 条 chunk 现场嵌入一遍要 2.4 秒，而且每问一次就要重来一遍。
    索引 = 【把这一步提前做完，存到硬盘上】。
    之后每次提问只嵌 1 条 query，而不是 237 条语料。

存什么（两个文件）：
    vectors.npy   形状 (N, D) 的 float32 矩阵，第 i 行 = 第 i 条语料
    ids.json      N 个 chunk_id + 元信息
                  ids[i] 描述的就是 vectors[i] —— 行序是唯一的纽带

为什么分两个文件：
    向量是数字   → 二进制（快、小。0.9 MB；写成 JSON 要 10 MB）
    id 是字符串 → JSON（人能读、git 能 diff、出问题能肉眼查）
    硬塞进一个文件，两边都别扭。

【ids.json 里必须记 model】：
    向量只有在【同一个模型】下才有可比性。
    拿 bge-m3 的 query 去比 bge-large-zh 的语料向量，不会报任何错 ——
    它就是老老实实算出一堆没意义的数。
    所以索引要记住「我是谁生成的」，读的时候对不上就【主动报错】。
"""

import json
import logging

import numpy as np

from paper_rag.config import IDS_FILE, VECTORS_FILE
from paper_rag.embed import embed_texts

logger = logging.getLogger(__name__)


def build(client, model: str, chunks: list[dict]) -> tuple[np.ndarray, list[str]]:
    """把 chunk 列表全部向量化。

    参数:
        client: ollama 客户端
        model:  嵌入模型名
        chunks: chunk 列表，每条至少有 "chunk_id" 和 "text"

    返回:
        (vectors, ids)
        vectors: np.ndarray，形状 (N, D)，dtype=float32
                 【dtype 用 float32 不是 float64】——
                 向量精度本来就只到小数点后几位，float64 白占一倍空间和带宽。
        ids:     list[str]，长度 N，和 vectors 的行一一对应

    异常:
        只要有 chunk 嵌不进去就抛 RuntimeError，【不静默跳过】。
        少一条语料，检索时就永远找不到它，而现象只是「偶尔答不上来」——
        这种 bug 极难排查。宁可在这里响亮地失败。
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

    matrix = np.array(vectors, dtype=np.float32)
    logger.info("向量化完成 → 形状 %s，dtype %s", matrix.shape, matrix.dtype)
    return matrix, ids


def save(vectors: np.ndarray, ids: list[str], model: str) -> None:
    """落盘。

    参数:
        vectors: (N, D) 矩阵
        ids:     N 个 chunk_id
        model:   生成这些向量的模型名 —— 写进元信息，将来用来对账
    """
    if len(ids) != vectors.shape[0]:
        # 行数对不上，存下去就是个错位的索引，而且【读的时候看不出来】
        raise ValueError(f"ids 有 {len(ids)} 个，矩阵有 {vectors.shape[0]} 行，对不上")

    VECTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_FILE, vectors)

    payload = {
        "model": model,
        "count": len(ids),
        "dim": int(vectors.shape[1]),
        "ids": ids,
    }
    IDS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(model: str | None = None) -> tuple[np.ndarray, list[str]]:
    """读回索引。

    参数:
        model: 期望的模型名。传了就【核对】—— 索引不是这个模型建的，直接报错。
               不传则不查（探针里想读旧索引做对比时会用到）。

    返回:
        (vectors, ids)

    异常:
        FileNotFoundError  索引不存在 —— 提示去跑 build_vectors.py
        RuntimeError       模型对不上 —— 说明该重建索引了
    """
    if not VECTORS_FILE.exists() or not IDS_FILE.exists():
        raise FileNotFoundError(f"索引不存在：{VECTORS_FILE}\n先跑：python scripts/build_vectors.py")

    vectors = np.load(VECTORS_FILE)
    payload = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    ids = payload["ids"]

    if len(ids) != vectors.shape[0]:
        raise RuntimeError(f"索引自己就是坏的：ids {len(ids)} 个，矩阵 {vectors.shape[0]} 行")

    if model is not None and payload["model"] != model:
        raise RuntimeError(
            f"索引是 {payload['model']!r} 建的，现在要用 {model!r} 查 —— 向量不可比。\n"
            f"重建：python scripts/build_vectors.py"
        )

    return vectors, ids
