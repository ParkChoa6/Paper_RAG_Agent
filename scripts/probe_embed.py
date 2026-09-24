"""探针：看看 embedding 到底长什么样。

这一支探针不解释原理，只负责把「东西」摆出来 —— 看完了再讲为什么。

问三件事：
  1. 一句话进去，出来的是什么？（维度、数值范围、类型）
  2. 同一句话喂两次，出来的一样吗？（确定性）
  3. 换一句话，出来的东西变了吗？

用法:
    python scripts/probe_embed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

from paper_rag.config import EMB_MODEL, OLLAMA_HOST

client = ollama.Client(host=OLLAMA_HOST)


def embed(text: str) -> list[float]:
    """把一段文本喂给向量模型，拿回一个向量。

    参数:
        text: 任意文本

    返回:
        list[float]，长度 1024
    """
    resp = client.embeddings(model=EMB_MODEL, prompt=text)
    return resp.embedding  # 注意：是 pydantic 对象的属性，不是 resp["embedding"]


def brief(vec: list[float], n: int = 6) -> str:
    """把向量的前 n 个数格式化成一行，方便并排对比。"""
    head = ", ".join(f"{v:+.4f}" for v in vec[:n])
    return f"[{head}, ...]"


# ═══════════ 1. 一句话进去，出来的是什么 ═══════════

TEXT = "Retrieval-Augmented Generation combines parametric and non-parametric memory."

vec = embed(TEXT)

print("=" * 70)
print("1. 一句话进去，出来的是什么")
print("=" * 70)
print(f"\n输入  : {TEXT}")
print(f"字符数: {len(TEXT)}")
print(f"\n输出  : 一个长度 {len(vec)} 的 list")
print(f"类型  : list[{type(vec[0]).__name__}]")
print(f"\n前 6 个数:")
print(f"  {brief(vec)}")
print(f"\n数值范围:")
print(f"  最小 {min(vec):+.4f}")
print(f"  最大 {max(vec):+.4f}")
print(f"  均值 {sum(vec) / len(vec):+.4f}")


# ═══════════ 2. 确定性：同一句话喂两次 ═══════════

print()
print("=" * 70)
print("2. 同一句话喂两次，结果一样吗")
print("=" * 70)

vec_again = embed(TEXT)
same = vec == vec_again
print(f"\n两次完全相同？ {same}")
print(f"第一次: {brief(vec)}")
print(f"第二次: {brief(vec_again)}")


# ═══════════ 3. 换一句话 ═══════════

print()
print("=" * 70)
print("3. 换一句话，输出变了吗")
print("=" * 70)

OTHER = "I would like a cup of coffee and a slice of cake."
vec_other = embed(OTHER)

print(f"\nA: {TEXT[:60]}...")
print(f"   {brief(vec)}")
print(f"\nB: {OTHER}")
print(f"   {brief(vec_other)}")

# 两个向量逐位相减，看看差异有多大
diff = [abs(a - b) for a, b in zip(vec, vec_other)]
print(f"\n两个向量逐位之差:")
print(f"  平均差 {sum(diff) / len(diff):.4f}")
print(f"  最大差 {max(diff):.4f}")


# ═══════════ 4. 算一笔账 ═══════════

print()
print("=" * 70)
print("4. 算一笔账")
print("=" * 70)

n_chunks = 237  # Stage 1 的产出
print(f"\n句子的字符数 : {len(TEXT)}")
print(f"向量的数字个数: {len(vec)}")
print(f"\n我们的语料     : {n_chunks} 个 chunk")
print(f"全部向量化需要 : {n_chunks} x {len(vec)} = {n_chunks * len(vec):,} 个数字")
print(f"按 float32 存 : {n_chunks * len(vec) * 4 / 1024 / 1024:.1f} MB")
