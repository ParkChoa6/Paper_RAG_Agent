# Paper RAG Agent

> **一个全部手写的本地论文 RAG。** 7 篇知识图谱论文 → 1345 个段落 → 检索 →
> 生成 → 逐条核对引用。不用 LangChain，不用向量库，零 API 成本。

它回答的不只是「问题」，还有**「你凭什么这么说」**：模型交出的每一条原文引用，
代码都会回到当初喂给它的材料里逐条核对，核对不了的当场报出来。

---

## 目录

- [1. 它现在能做什么](#1-它现在能做什么)
- [2. 架构总览](#2-架构总览)
- [3. 快速开始](#3-快速开始)
- [4. 常用命令](#4-常用命令)
- [5. 实测数字](#5-实测数字)
- [6. 目录结构](#6-目录结构)
- [7. 数据契约](#7-数据契约)
- [8. 已知边界](#8-已知边界)
- [9. 文档地图](#9-文档地图)
- [10. 关于论文 PDF](#10-关于论文-pdf)

---

## 1. 它现在能做什么

```
$ python scripts/ask.py "KGE 模型是按什么标准分类的？"

语料 1345 条 · 索引 (1345, 1024) · 嵌入 bge-m3:latest · 生成 qwen2:7b
检索器: dense>rerank@pool50
==============================================================================
问: KGE 模型是按什么标准分类的？

--- 检索到 5 条 ---
   检索器: dense>rerank@pool50
  [1] +0.87  Knowledge Graph Embedding: An Overview  (Abstract，第 1 页)
      We focus on two main branches of KGE design: ...
  ...（共 5 条，每条都印分数 / 标题 / 章节 / 页码）

--- 结果 [OK  ] ---
  知识图谱嵌入（KGE）模型主要分为两大类…… [1]

  引用了: [1]
    [1] we focus on two main branches of KG embedding (KGE) design ...
  ✓ 引用校验通过（每条 quotes 都能在材料原文里找到）
```

> 上面是**输出格式示意**，不是某次运行的逐字记录（分数和文字会随题变化）。
> 真正跑起来时，最后一行可能是三种之一：
> `✓ 引用校验通过` / `⚠️ 引用校验发现 N 个问题` / `（材料里没有答案，模型按契约拒答了）`。

完整链路是五步：**切块 → 向量化 → 检索（粗排 + 重排）→ 生成 → 核对引用**。
每一步都有对应的探针脚本和实测记录，不是拼起来的黑盒。

---

## 2. 架构总览

### 2.1 离线：把论文变成索引（跑一次，几分钟）

```
  data/raw/*.pdf
        │
        │  scripts/build_index.py          论文登记表：文件名 → paper_id / 标题
        ▼
  paper_rag/ingest.py
        │  跨页家具识别 → 双栏还原 → 标题识别 → 丢参考文献 → 清洗 → 切块
        ▼
  data/processed/chunks.json              1345 条
        │  每条带 chunk_id / paper_id / title / page_start / page_end / section
        │
        │  scripts/build_vectors.py
        ▼
  bge-m3（Ollama）──▶ vectors.npy  (1345, 1024) float32
                      ids.json     {model, count, dim, ids[]}
```

### 2.2 在线：一次问答

```
                          用户问题（中文）
                                │
              ① expand_query()  补上语料里实际用的英文写法
                 ★ 只影响检索，不改 prompt 里的问题
                                ▼
        ┌──────────── DenseRetriever.search() ────────────┐
        │  embed_one → (1024,)                            │
        │  一次矩阵乘法：vectors(N,1024) @ q(1024,)       │
        │  取前 50 条（粗排，pool=50）                     │
        └───────────────────┬─────────────────────────────┘
                            ▼  50 条 (chunk_id, 余弦)
        ┌────────── CrossEncoderReranker.rerank() ────────┐
        │  bge-reranker-v2-m3 逐对打分 (问题, 正文)，CPU   │
        └───────────────────┬─────────────────────────────┘
                            ▼  取前 5
                    按 chunk_id 取回正文   ★ 不是按行号
                            ▼
        ┌────────────── generator.answer() ───────────────┐
        │  build_prompt（材料表头 + 正文 + 契约）          │
        │  qwen2:7b 生成（JSON Schema 约束解码）           │
        │  parse_response（只管形状，不管内容）            │
        └───────────────────┬─────────────────────────────┘
                            ▼  {answer, used[], quotes{}}
        ┌─────────────── cite.verify() ───────────────────┐
        │  每条 quotes 回当初的材料原文里核                │
        └───────────────────┬─────────────────────────────┘
                            ▼
        答案 + 引用 + 问题清单（空列表 = 全部经得起核对）
```

### 2.3 模块分工

| 模块 | 一句话职责 | 产品 / 脚手架 |
|---|---|---|
| [`config.py`](paper_rag/config.py) | 所有常量 + 它们背后的实测决策 | 产品 |
| [`ingest.py`](paper_rag/ingest.py) | PDF → chunk（版面、标题、清洗、切块） | 产品 |
| [`models.py`](paper_rag/models.py) | `Chunk` 数据模型 | 产品 |
| [`embed.py`](paper_rag/embed.py) | 文本 → 向量（Ollama 薄封装） | 产品 |
| [`llm.py`](paper_rag/llm.py) | 文本 → 文本（Ollama 薄封装） | 产品 |
| [`index.py`](paper_rag/index.py) | 向量索引的建 / 存 / 读 | 产品 |
| [`retrieve.py`](paper_rag/retrieve.py) | 相似度原语，手写 → numpy → FAISS 对拍 | 产品 |
| [`retriever.py`](paper_rag/retriever.py) | `Retriever` 接口 + Dense / BM25 实现 | 产品 |
| [`rerank.py`](paper_rag/rerank.py) | cross-encoder 重排，包装成检索器 | 产品 |
| [`generator.py`](paper_rag/generator.py) | prompt 拼装 / 解析 / 生成编排 | 产品 |
| [`cite.py`](paper_rag/cite.py) | 引用真伪核对 | 产品 |
| [`pipeline.py`](paper_rag/pipeline.py) | 把上面全部串起来，只做编排 | 产品 |
| `scripts/*` | 入口、评测、探针 | 脚手架 |

**产品 / 脚手架的分界线**：`paper_rag/` 里不碰 `argparse`、不 `print`、只用 `logger`；
需要做业务判断的逻辑必须在产品里，脚本只负责喂参数和打印。

---

## 3. 快速开始

### 前置

- Python **3.13+**
- [Ollama](https://ollama.com) 已安装并运行（本项目实测基线 **0.40.2**）

### 1. 拉模型

```powershell
ollama pull qwen2:7b          # 生成
ollama pull bge-m3:latest     # 嵌入，1024 维
```

> 重排模型 `BAAI/bge-reranker-v2-m3` 不通过 Ollama 提供，首次运行时会由
> `sentence-transformers` 自动下载（约 2.3 GB），走 CPU。

### 2. 建环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate          # Linux / macOS
pip install -r requirements.txt
```

### 3. 自检

```powershell
python check_env.py
```

四项全过（Python 版本 / 中文编码 / Ollama 与模型 / 两个接口连通）再往下。

### 4. 准备语料

把论文 PDF 放进 `data/raw/`，**文件名必须和 [`scripts/build_index.py`](scripts/build_index.py)
里的 `PAPERS` 登记表一致**（见 [第 10 节](#10-关于论文-pdf)）。

### 5. 建索引（离线，一次）

```powershell
python scripts/build_index.py       # PDF  → chunks.json
python scripts/build_vectors.py     # chunks → vectors.npy + ids.json
```

跑完之后，`data/processed/` 里应该有 `chunks.json`、`vectors.npy`、`ids.json` 三个文件。
换论文 / 改切分逻辑 → 重跑第一个；只换嵌入模型 → 只重跑第二个。

### 6. 提问

```powershell
python scripts/ask.py "KGE 模型是按什么标准分类的？"
python scripts/ask.py --quiet "静态 KG 和时序 KG 的推理区别是什么？"
python scripts/ask.py --no-rerank "……"      # 关掉重排，快 10 倍
```

---

## 4. 常用命令

| 想做什么 | 命令 |
|---|---|
| 环境自检 | `python check_env.py` |
| 重建文本层 | `python scripts/build_index.py` |
| 重建向量层 | `python scripts/build_vectors.py` |
| 问一句 | `python scripts/ask.py "问题"` |
| 连续问几句 | `python scripts/ask.py "问1" "问2"` |
| 比检索策略 | `python scripts/eval_retrieve.py --kind dense,bm25,rerank --expand` |
| 比嵌入模型 | `python scripts/eval_embed.py <模型名>` |
| 端到端跑三个 arm | `python scripts/eval_answer.py` |
| 给判卷表算分 | `python scripts/score_judge.py` |
| 跑某个探针 | `python scripts/probe_cite.py`（共 13 个，见 `scripts/probe_*.py`） |

> ⚠️ 评测脚本里每一个「只影响 query 的开关」（`--expand`、pool、k）都必须在
> 生产配置里也成立。这个仓库在这上面栽过三次，详见
> [`docs/ROADMAP.md`](docs/ROADMAP.md) 规矩 1。

---

## 5. 实测数字

**检索基线**（28 题 / 7 篇 / 1345 条，全部开查询扩展）

| 检索器 | Top-1 | MRR | recall@5 |
|---|---|---|---|
| dense | 50% | 0.605 | 71% |
| **dense + rerank @pool50（= 生产）** | **68%** | **0.763** | **89%** |
| BM25 | 32% | 0.429 | 54% |

**端到端**（同一批 28 题）

| arm | 材料含答案（检索层） | 合规回答（答案质量） | 秒/题 |
|---|---|---|---|
| dense | 71% | 79% | 1.3 |
| **rerank（= 生产）** | **89%** | **79%** | **13.3** |
| oracle（检索满分） | 100% | 82% | 1.3 |

**这三行数字最重要的读法是**：检索涨了 18 点，**答案质量一点没动**；
而且检索满分时生成层也只到 82% —— **瓶颈在生成层，不在检索。**

> ⚠️ 三个必须一起看的限定：
> ① n=28，80% 附近的 95% 区间是 **±15 点**，小于这个的差距别当结论；
> ② **「合规回答」不是「答对」**，它只保证「交付了一个有据可查的答案」；
> ③ 真正回答「对了几题」的判卷表，**目前还空着**（`data/processed/e2e_judge.md`）。
>
> ★ S5-7 之后 dense 那一行是 86%（24/28），生产臂 rerank 仍是 79% —— 修的是
> `parse_failed`，不是这道瓶颈。

---

## 6. 目录结构

```
Paper_RAG_Agent/
├── paper_rag/              产品代码（约 4.1k 行）
│   ├── config.py           常量 + 决策注释（改行为先看这里）
│   ├── ingest.py           PDF 版面还原与切块
│   ├── models.py           Chunk
│   ├── embed.py / llm.py   Ollama 的两个薄封装
│   ├── index.py            向量索引读写
│   ├── retrieve.py         相似度原语
│   ├── retriever.py        Retriever 接口 + Dense / BM25
│   ├── rerank.py           cross-encoder 重排
│   ├── generator.py        prompt / 解析 / 编排
│   ├── cite.py             引用核对
│   ├── pipeline.py         端到端装配
│   └── logger.py           日志
├── scripts/                脚手架（约 4.2k 行）
│   ├── build_index.py      PDF → chunks.json
│   ├── build_vectors.py    chunks.json → vectors.npy
│   ├── ask.py              命令行问答入口
│   ├── eval_*.py           三把尺子（模型 / 策略 / 答案）
│   ├── score_judge.py      判卷表算分
│   └── probe_*.py          13 个探针，每个验证一个具体机制
├── data/
│   ├── raw/                PDF（不进 git）
│   ├── processed/          产物（不进 git）
│   ├── eval/               评测集 queries.json（★ 进 git）
│   └── logs/               日志（不进 git）
├── docs/                   架构、路线、评估、学习记录
├── lessons/                0A 阶段的 Python 练习题
├── check_env.py            不依赖第三方库的环境自检
└── requirements.txt        依赖 + 为什么装它 / 为什么不接线
```

`.gitignore` 的判据只有一条：**别人 clone 下来，能不能自己弄出这个文件？**
能 → 不进 git，把「怎么弄出来」写进 README；不能 → 进。
评测集属于「不能」（得读懂论文才能出题），所以它进仓库。

---

## 7. 数据契约

| 文件 | 形状 | 谁写 | 谁读 |
|---|---|---|---|
| `chunks.json` | `{count, chunks: [...]}` | `build_index.py` | 全链路 |
| `vectors.npy` | `(N, 1024) float32` | `build_vectors.py` | `index.load()` |
| `ids.json` | `{model, count, dim, ids[]}` | `build_vectors.py` | `index.load()` |

两个关键约束：

1. **`ids[i]` 描述的就是 `vectors[i]`，行序是唯一纽带。** 两个文件是分开写的，
   一旦其中一个重跑过而另一个没有，行号就错位了 —— 而**错位不会报错**，
   只会静默返回风马牛不相及的材料。所以生产路径上取正文一律**按 `chunk_id` 查**，
   不按行号。
2. **`ids.json` 必须记模型名。** 拿 bge-m3 的 query 去比另一个模型建的语料向量，
   不会报任何错，只会算出一堆没意义的数。所以 `index.load(model=...)` 会核对，
   对不上直接报错。

`chunk_id` 的格式是 `paper_id#序号`（例：`kg-embedding-overview#1`），
**不承诺跨重建稳定**。

---

## 8. 已知边界

这份清单是给「准备接着做的人」看的，不是自我批评。

| 边界 | 现状 | 影响 |
|---|---|---|
| **评测样本小** | 28 题 | 80% 附近 ±15 点，小差距量不出来 |
| **答案对错未判** | 判卷表空着 | 只知道「有据可查」，不知道「对不对」 |
| **单轮、无 Agent** | Stage 6 未开始 | 没有工具调用循环、没有多轮记忆 |
| **没有 pytest / CI** | 13 个探针靠人眼看输出 | 「看一眼就过」漏过真 bug |
| **查询扩展靠手写表** | `config.EXPANSIONS` | 换一批论文就得重写 |
| **中文问题 / 英文语料** | 靠扩展表 + bge-m3 跨语言 | BM25 一半的题会「完全失明」 |
| **问答约 13 秒/题** | 其中 10.6 秒是 CPU 重排 | 交互体验的硬约束 |

**关于向量库**：本项目**没有用**任何向量库。1345 条上暴力检索只要 0.09 ms，
占一次问答的 0.24%，FAISS 换不来任何东西（实测：0.24% → 0.09%）。
`faiss-cpu` 装了但**故意不接线**，触发条件写在
[`requirements.txt`](requirements.txt) 的注释里：语料涨到几十万 chunk、
需要按元数据预过滤、需要增量增删、或 RAM 装不下时，再接线。

---

## 9. 文档地图

| 文档 | 什么时候读 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 想知道**数据怎么流、接口怎么定、失败怎么分** |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 想知道做到哪一步了、下一步是什么、踩过哪些系统性坑 |
| [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md) | 想知道「这个项目做得怎么样」（2026-09-29 快照 + 复核） |
| [`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md) | 逐日学习记录：概念、踩坑、待复习 |
| [`docs/stages/`](docs/stages/) | 每个小步的完整开发记录（做了什么 / 为什么 / 怎么验证） |

---

## 10. 关于论文 PDF

`data/raw/` 里的 PDF **不进 git**（版权 + 体积），需要自己下载后按下面的
**文件名原样**放进去。文件名是 [`scripts/build_index.py`](scripts/build_index.py)
里 `PAPERS` 登记表的键，改一个字就会「找不到，跳过」。

| 文件名 | 主题 |
|---|---|
| `Knowledge Graph Embedding An Overview.pdf` | KGE 总览 |
| `A Survey of Knowledge Graph Reasoning on Graph Types Static, Dynamic, and Multi-Modal.pdf` | KGR 综述 |
| `Knowledge Graphs Opportunities and Challenges.pdf` | KG 机遇与挑战 |
| `Overview of knowledge reasoning for knowledge graph.pdf` | 知识推理总览 |
| `Multi-hop interpretable meta learning for few-shot temporal knowledge graph completion.pdf` | 时序 KG 补全 |
| `Multi-hop reasoning of knowledge chain over fuzzy spatiotemporal knowledge graphs with box embedding.pdf` | 模糊时空 KG 推理 |
| `Temporal knowledge graph reasoning based on time layering and fuzzy logic rules.pdf` | 时序推理 + 模糊规则 |

七篇均为公开论文。**加新论文时要做三件事**：
① 文件名登记进 `PAPERS`；② 重跑 `build_index.py` + `build_vectors.py`；
③ 给新论文补评测题（否则所有指标都测不到它）。
