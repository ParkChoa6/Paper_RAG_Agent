# S5-1 · 检索器抽象：先造尺子，再造东西

**日期** 2026-10-12　**阶段** Stage 5 检索质量 · 第 1 小步
**语料** 337 条 / 2 篇　**评测集** 12 题
**状态** ✅ 完成，对拍通过

---

## 0. 这一步在流水线的哪个位置

```
    build_index.py      PDF          → chunks.json      （离线，跑一次）
    build_vectors.py    chunks.json  → vectors.npy      （离线，跑一次）
                                            │
                                            ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  retriever.DenseRetriever.search(query, k)         ★ 新增     │
    │    query 文本 → 加 query 前缀 → 嵌成向量 → top_k_np           │
    └──────────────────────────────────────────────────────────────┘
                       │                            │
                       ▼                            ▼
            pipeline.ask()  【产品】      eval_retrieve.py 【脚手架】
                   ↑ 本次也改成走这条线          ↑ 本次新增
```

| 文件 | 产品 / 脚手架 | 说明 |
|---|---|---|
| `paper_rag/retriever.py` | **产品** | 可复用、含判断（前缀加在哪）、被 CLI / web / 评测三处调用 |
| `scripts/eval_retrieve.py` | 脚手架 | 只负责喂参数、算指标、打表格。没有产出物被 import |
| `paper_rag/pipeline.py` | 产品（改动） | 检索那两行改走 `DenseRetriever` |

**为什么是先做这个，而不是先做 BM25？**

因为没有尺子就量不出 BM25 好不好。这个项目的实际比例是
**「跑通 : 做对 ≈ 1 : 1.3」**（见 `docs/LEARNING_LOG.md`）——
写出来只占三成工作量，剩下七成全是「怎么知道它是对的」。
尺子属于那七成，但它**必须在东西之前到位**，否则你会拿新东西去测一把歪尺子。

---

## 1. 改了什么

### 新增 `paper_rag/retriever.py`

- `Retriever` —— 一个 `Protocol`，规定所有检索器长什么样：有 `name`，有 `search(query, k)`
- `DenseRetriever` —— 第一个实现。封装 `embed_one` + `top_k_np`
- `prefix_for(model)` —— query 前缀表，**从 `scripts/eval_embed.py` 搬进产品层**

### 新增 `scripts/eval_retrieve.py`

评任意检索器：Top-1 / MRR / 平均排名 / recall@k。

### 修改 `paper_rag/pipeline.py`

检索两行改走 `DenseRetriever`：

```python
# 旧
query_vec = embed_one(client, emb_model, query)
hits = top_k_np(query_vec, vectors, ids, k=k)

# 新
hits = DenseRetriever(client, emb_model, vectors, ids).search(query, k=k)
```

---

## 2. 为什么这么写

### 2.1 为什么接口收 `query: str`，而不是 `query_vec: list[float]`

★ **这是整个文件里唯一一个不能随便改的决定。**

BM25 **根本没有 query 向量** —— 它靠词频和倒排表。如果接口收的是向量：

```python
# ❌ 这样写，BM25 就实现不了这个接口
class BM25Retriever:
    def search(self, query_vec, k):   # ← 我需要的是文本，不是向量！
        ...
```

结果是两条烂路二选一：要么在 BM25 外面包一层假的（先嵌个向量再丢掉），
要么在评测脚本里写 `if kind == "bm25": ...` —— 抽象当场作废。

接口收**文本**，每个实现自己决定怎么理解它：

| 实现 | 拿到文本后干什么 |
|---|---|
| `DenseRetriever` | 嵌成向量 → 算余弦 |
| `BM25Retriever` | 分词 → 查倒排表 |
| `RerankRetriever` | 交给 cross-encoder 打分 |

还有一条更容易走错的路：**接口收向量，BM25 那边传 `None`**。
那种接口叫「只对一半实现成立」—— 是**抽象漏了**，不是抽象。

> **一条通用经验**：定接口时，先问「未来最不像的那个实现，需要什么」。
> 让它也能进来，接口才是真的。让最像的那个进来很容易，那叫复制粘贴。

### 2.2 为什么要用 `Protocol`，而不是抽象基类 `ABC`

因为它们**不需要真的继承同一个父类**。`DenseRetriever` 完全不知道
`Retriever` 的存在，照样能当 `Retriever` 用。这叫**结构化子类型**
（structural subtyping）—— 看的是「长什么样」，不是「是谁的儿子」。

好处：将来的 `BM25Retriever` 可以是第三方库里的类，只要有 `search()` 就能用。

⚠️ 代价：`Protocol` **不检查任何东西**，它只是一份写下来的约定。
写错了 Python 不拦你，跑起来才炸。所以接口的 docstring 才是真正的接口。

### 2.3 为什么 query 前缀必须搬进产品层

这一步修掉了一个**潜伏的分歧**：

| | 加不加 query 前缀 |
|---|---|
| 生产路径 `pipeline.ask` | **完全没这个概念**（`embed.embed_one` 注释直接写死「不加前缀」） |
| 评测路径 `eval_embed.py` | 有 `QUERY_PREFIX` 表，按模型加 |

**今天两边数字一样 —— 只因为 `bge-m3` 的前缀是空串。**

哪天把 `config.EMB_MODEL` 换成 `qwen3-embedding`：

- 评测：加上 `"Instruct: ...\nQuery: "` → 分数正常
- 生产：不加 → 静默掉分
- **两边都不会报错。** 你会看到「评测说很好，线上就是不行」

这就是把前缀表搬进 `retriever.py` 的理由：**让前缀跟着模型走，而不是跟着脚本走。**

### 2.4 为什么指标按**全量排名**算，而不是按 top-k 算

`rank`（排名）是个**截断量**：只拿到 top-5 的时候，
正确段落在第 40 名和第 400 名，你看到的都是「不在前 5」——
**算不出 MRR 的差别**。

所以 `eval_retrieve.py` 里 `search()` 一次取回**全部 N 条**，
`recall@k` 再从前缀切片算出来。

代价：每题多排 N 条。337 条可以忽略；将来百万级语料要重新想办法
（见第 7 节「遗留」）。

### 2.5 为什么 `name` 里要带 `:raw` / `:prefix`

前缀这件事在这个仓库里**说法是矛盾的**（见 7.2），所以将来一定要做 A/B。
如果 `name` 写死成 `"dense"`，A/B 两张表在表格里长得一模一样，
你就分不清哪行是哪组 —— **自己把自己的实验做瞎了**。

---

## 3. 不应该怎么写

### 3.1 ❌ 在 `eval_retrieve.py` 里再抄一遍检索循环

**这就是这一步要修的病。**

看旧的 `scripts/eval_embed.py:113-117`：

```python
scored = []
for cid, v in vec_of.items():
    s = cosine(qvec, v)
    scored.append((s, cid))
scored.sort(reverse=True, key=lambda t: t[0])
```

它 **import 了 `cosine`，但没有调用 `retrieve.top_k` / `top_k_np`** ——
手抄了一遍检索。

**后果**：评测量的那份检索，和产品跑的那份，**是两份代码**。
改一处漏一处，**不会有任何东西报错**，你只会看到「评测 0.597，线上感觉不太行」。

> 这个仓库已经中过一模一样的一枪：V3 的 `title + text` 写在
> `index.py:56` 和 `eval_embed.py:82` **两处**，而 `ids.json` 只记**模型名**、
> 不记**文本方案**。改漏一处，静默分歧。

**判断标准**：如果一段逻辑你在两个文件里各写了一遍，
且它们必须保持一致 —— 那它们必须变成**同一个函数调用**。

### 3.2 ❌ 让 `DenseRetriever.__init__` 自己去 load 索引

```python
# ❌
class DenseRetriever:
    def __init__(self, model):
        self.vectors, self.ids = index.load(model)   # 构造函数里碰硬盘
```

**症状一**：评测就没法用同一批向量建多个检索器了 —— 每个都得重读一次盘。
**症状二**：构造函数做 IO，意味着你**没法在不碰硬盘的情况下测这个类**。
**症状三**：`load()` 只认「盘上那一份」索引，将来想让 dense 和 BM25
看同一份语料就没门了。

所以向量和 id 由**外面传进来**（`from_chunks` 只是造它们的一个便捷入口）。

### 3.3 ❌ 把 `name` 写死成 `"dense"`

见 2.5。跑 A/B 时两行同名，实验白做。

### 3.4 ❌ 用 `top_k`（纯 Python 版）而不是 `top_k_np`

```python
from paper_rag.retrieve import top_k      # ❌ 337 条能跑，几万条就废了
```

`top_k` 是 O(N) 的 Python 循环，每条都算一次完整余弦（含两次开根号）。
`top_k_np` 是一次矩阵乘法。

**症状不是「报错」，是「变慢」** —— 慢到你开始想「少测几道题吧」。
**评测就是从这一刻开始烂的。**

> 顺带：这正是 `requirements.txt` 里 faiss 那段注释的道理 ——
> 237 条时暴力检索 0.091ms，占一次问答的 0.24%，**换 FAISS 换不来任何东西**。
> 过早优化和拒绝优化，都是同一个错误的两面。

### 3.5 ❌ 忘记 `ids` 和 `vectors` 的长度断言

```python
if len(ids) != vectors.shape[0]:
    raise ValueError(...)
```

**删掉会怎样**：`retrieve.top_k_np` **不会替你查**。它拿 `argsort` 出来的
下标直接索引 `ids`：

```python
return [(ids[int(i)], float(scores[i])) for i in top_k_idx]
```

长度对不上时它**照样返回**，只是每个 `id` 属于**另一条语料**。
不报错，结果是错的 —— 而且是那种「看起来挺像那么回事」的错。

---

## 4. 怎么验证它真的对了

### 4.1 对拍：换实现必须证明数字没动

`eval_retrieve.py`（走 `DenseRetriever`）vs `eval_embed.py`（旧的独立实现，
**故意没动它**）：

| 题 | 旧：排名 / 分数 | 新：排名 / 分数 |
|---|---|---|
| 1 | 1 / +0.6687 | 1 / +0.6687 |
| 2 | 1 / +0.5948 | 1 / +0.5948 |
| 3 | 13 / +0.5444 | 13 / +0.5444 |
| 4 | 2 / +0.6259 | 2 / +0.6259 |
| 5 | 4 / +0.4792 | 4 / +0.4792 |
| 6 | 1 / +0.6213 | 1 / +0.6213 |
| 7 | 4 / +0.5527 | 4 / +0.5527 |
| 8 | 1 / +0.5765 | 1 / +0.5765 |
| 9 | 16 / +0.4976 | 16 / +0.4976 |
| 10 | 1 / +0.6993 | 1 / +0.6993 |
| 11 | 1 / +0.6519 | 1 / +0.6519 |
| 12 | 33 / +0.4724 | 33 / +0.4724 |

**汇总**：Top-1 50%　MRR 0.597　平均排名 6.5 —— **12/12 逐位一致**。

> ★ **为什么先不改 `eval_embed.py`**
>
> 如果两边同时改，对拍就**什么也证明不了**（两边一起错，数字还是对得上）。
> 先留一个**独立实现**当参照，改完对上了，才有资格去动参照物。
>
> 这是这个项目的核心方法论：**探针和产品必须独立实现，才能互相对拍。**
> `docs/LEARNING_LOG.md` 第八节「探针必须能失败」是同一个道理的另一种说法。

### 4.2 意外的第二重验证：bge-m3 确实归一化了

这次对拍顺手证明了一件本来没打算查的事。

- `eval_embed.py` 用的是**真余弦**：`dot(a,b) / (|a|·|b|)`
- `top_k_np`（`DenseRetriever` 走的就是它）用的是**点积**：`vectors @ q`

**点积只在向量单位长度时才等于余弦。** 而 12 道题的排名和分数
**分毫不差** —— 等于实测确认了 bge-m3 的输出向量模长恒为 1。

> ⚠️ 但这是个**依赖**，不是巧合：换个不归一化的嵌入模型，
> `top_k_np` 会静默给出错误的排序。见 7.3。

### 4.3 端到端

```
$ ./.venv/Scripts/python.exe scripts/ask.py "TransE 的核心思想是什么？"
  [1] +0.5948  Knowledge Graph Embedding: An Overview  (2.1 Distance-based Models，第 8 页)
  ...
  ✓ 引用校验通过
```

产品跑这道题 top-1 是 **+0.5948** —— 和评测里第 2 题**同一个数**。
同一份实现，同一个分数。

### 4.4 回归

```
$ ./.venv/Scripts/python.exe scripts/probe_cite.py
✅ 通过，且诱导有效（1/8 次编了，verify 每次抓住）。
   退出码 = 0
```

Stage 4 没被碰坏。

> 注：`tests/` 目录是**空的**，没装 pytest。
> **这个仓库的「测试」就是 `scripts/probe_*.py`。**

---

## 5. 你要补的概念

### 5.1 `Protocol` 和「结构化子类型」

```python
from typing import Protocol

class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...
```

最小例子：

```python
class Duck(Protocol):
    def quack(self) -> str: ...

class Robot:                      # ★ 注意：没有继承 Duck
    def quack(self) -> str:
        return "beep"

def f(x: Duck) -> None: ...

f(Robot())                        # ✅ 类型检查器认，因为 Robot 有 quack
```

对比 `ABC`：ABC 是**名义子类型**，必须写 `class Robot(Duck)` 才算。

| | 要求 | 适合 |
|---|---|---|
| `ABC` | 必须显式继承 | 你**拥有**所有实现，想强制共享代码 |
| `Protocol` | 只要长得像 | 实现可能来自第三方，或彼此不该耦合 |

### 5.2 为什么排名要用**倒数**平均（MRR）

排第 1 名 → 1 分；第 2 名 → 1/2；第 5 名 → 1/5。

因为**「第 1 名」和「第 2 名」的差距，比「第 20 名」和「第 21 名」大得多**。
直接平均排名会把这个差别抹平。

`recall@k` 和 `MRR` 量的**不是一回事**：

- `recall@5` = 「捞到了没有」—— 位置无所谓
- `MRR` = 「捞得多靠前」—— 对生成质量更敏感，因为送进 prompt 的是前 5 条

本次实测：**recall@5 = 75%，Top-1 = 50%**。
差的这 25% 就是「捞到了，但排在第 2~5 名」。

### 5.3 点积 vs 余弦

```
cos(a,b) = (a·b) / (|a| · |b|)

|a| = |b| = 1  时   →   cos(a,b) = a·b
```

`top_k_np` 省掉的就是那个分母（以及两次开根号）。
省掉**在单位向量上完全等价**，在其余情况**完全是错的**。

### 5.4 工厂函数什么时候才值得写

```python
def make_retriever(kind, model, client, chunks):
    if kind == "dense":
        return DenseRetriever.from_chunks(client, model, chunks)
    raise SystemExit(...)
```

**一个分支的工厂函数是过度设计** —— 直接调构造函数就行。
**两个分支就值得了** —— 因为调用方（评测脚本）不该知道有哪几种检索器。

所以这一步只写了骨架，分支等 BM25 落地时再加。

---

## 6. 自测题

**Q1.** `DenseRetriever.search("TransE 的核心思想是什么？", k=5)` 的 top-1 分数是 +0.5948。
如果我把 `config.EMB_MODEL` 从 `bge-m3:latest` 换成 `qwen3-embedding:0.6b` 并重建索引，
这个分数会**怎么变**？—— 分别说：前缀会加吗？加了会怎样？在哪一行代码决定的？

**Q2.** 如果我把协议改成 `search(self, k)`（**不收 query**，query 在构造函数里传），
`bm25 / rrf / rerank` 三个实现分别会出什么问题？
（提示：RRF 的输入是什么？它需要「同一个 query 的两组结果」。）

**Q3.** `eval_retrieve.py` 用 `all_k=len(chunks)` 拿全量排名。
语料涨到 **100 万条**时，这个设计会出什么问题？
说出**两个**不同的代价，以及各自可能的解法。

**Q4.** `DenseRetriever.__init__` 里删掉那段长度断言，然后故意把
`ids` 传成少一个元素的列表。**具体会发生什么**？
`top_k_np` 会抛异常吗？如果不会，返回的是什么？

**Q5.** 现在 `prefix_for` 在 `retriever.py` 和 `eval_embed.py` 里**各有一份**。
我明明一直在说「不要抄两份」—— 为什么这里**故意**留着两份？什么时候可以合并？

---

## 7. 遗留 / 下一步

### 7.1 `eval_embed.py` 里的 `prefix_for` 还没收掉

**故意的**，理由见 4.1 —— 它现在是独立参照物。
等 Stage 5 走完、新评测稳定了，再单独一步把它迁到 `retriever` 上。

### 7.2 ⚠️ bge-large-zh 的前缀，两份注释互相打架（**未解决**）

| 出处 | 说法 |
|---|---|
| `paper_rag/embed.py:96` | 「bge-large-zh **加**官方前缀后，Top-1 从 40% **掉到** 30%」 |
| `paper_rag/retriever.py` `QUERY_PREFIX` | 「bge-large-zh-v1.5 **要求**中文指令」 |

**必有一句是错的。** 现在生产用 bge-m3，两条都用不上，所以没炸。
动 bge-large-zh 之前必须把这两个数字重新量一遍。

### 7.3 ⚠️ `top_k_np` 用点积冒充余弦

**依赖「嵌入模型的输出向量是单位长度」**，而这件事**没有任何断言在守**。
换一个不归一化的模型，检索会**静默给出错误排序**。

可能的修法（择一，还没定）：
- 在 `index.build` / `load` 时断言 `|v| ≈ 1`，不满足就抛错
- 或者改成先归一化再存

### 7.4 `pipeline.ask()` 的签名还收 `vectors, ids`

等 `BM25Retriever` 落地，`ask()` 就**收不了**这两个参数了 ——
BM25 没有向量。那时签名要改成收一个 `Retriever`。

**现在不改**：还没有第二个实现需要它。**接口要等到有第二个用户时才放宽**，
否则你会照着一个想象出来的需求去设计。

### 7.5 `tests/` 是空的

仓库有个 `tests/` 目录但里面什么都没有，也没装 pytest。
目前「测试」= `scripts/probe_*.py`。

不是这一小步的活，但记一笔 —— 将来要加 CI 的话绕不过去。

---

## 下一步（S5-2）

**BM25 检索器**，以及让 `eval_retrieve.py` 能把它和 dense 摆在同一张表上。

到那时才会第一次真正需要 `make_retriever` 的第二个分支，
也才会第一次面对那个问题：**两种检索器的分数不可比**（余弦 ∈ [-1,1]，BM25 无上界）
—— 这就是 RRF 要按**排名**融合而不是按分数加权的原因。
