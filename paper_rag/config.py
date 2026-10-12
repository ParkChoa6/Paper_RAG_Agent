"""全局配置。

约定：这里只放「值」，不放逻辑。
其它模块 `from paper_rag.config import XXX` 取值，不许在别处硬编码。
"""

from pathlib import Path

# ---  路径 ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent  # 项目根目录
DATA_DIR: Path = PROJECT_ROOT / "data"  # 数据总目录
RAW_DIR: Path = DATA_DIR / "raw"  # 原始 PDF 放这
PROCESSED_DIR: Path = DATA_DIR / "processed"  # chunks.json、索引文件放这

# 评测集：手写的中文问题 + 对应的正确 chunk。
# 它和上面两个目录的区别是【进 git】——
# .gitignore 的判据是「别人 clone 下来能不能自己弄出来」：
#   chunks.json 能（跑 build_index.py 就有）→ 不进
#   评测集不能（得读懂论文才能出题）      → 进
EVAL_DIR: Path = DATA_DIR / "eval"

# ---- 向量索引（Stage 2）----
# 两个文件，靠【行序】绑在一起：ids[i] 描述的就是 vectors[i]。
# 分开存是因为数据类型不同：向量是数字（二进制最省），id 是字符串（JSON 最可读）。
VECTORS_FILE: Path = PROCESSED_DIR / "vectors.npy"  # (N, D) float32 矩阵
IDS_FILE: Path = PROCESSED_DIR / "ids.json"  # 行序 ↔ chunk_id 的对应 + 元信息

# ---- Ollama ----
OLLAMA_HOST: str = "http://localhost:11434"  # 服务地址
GEN_MODEL: str = "qwen2:7b"  # 生成模型
EMB_MODEL: str = "bge-m3:latest"  # 原 "quentinz/bge-large-zh-v1.5:latest"  # 向量模型

# ---- 检索 ----
TOP_K: int = 5  # 每次检索取几段

# ---- 重排（Stage 5 · Rerank）----
# cross-encoder 重排：拿 (query, 正文) 逐对打分，只重排【已经捞回来的】候选。
#
# ★ 为什么本地跑的是 HuggingFace 模型，不是 Ollama 模型 ——
#   2026-10-12 实测：Ollama 0.40.2 【没有 /api/rerank 端点】（直接探到 404，
#   而 /api/embed /api/embeddings 都是 200，所以不是探测方法的问题）。
#   架构上的原因：Ollama 是给「tokens → 文本」的生成式 LLM 设计的，
#   而 reranker 要的是「[query, doc] 对 → 一个标量」。生态里的绕法是
#   llama.cpp --reranking / vLLM / 自己起一个 /rerank 服务。
#   我们选最后一条的等价物：sentence-transformers 直接在进程里跑。
#   代价见下方 requirements.txt 的记账。
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

# ★ 默认 cpu，不是 cuda。理由不是"cpu 快"，是那台 4070 的 12GB 显存
#   还要给 Ollama 用 —— qwen2:7b 生成占约 4.5GB，再挤一个 568M 的 reranker
#   进去，撞上显存墙时的现象是【Ollama 那边超时】，看不到 rerank 的错。
RERANK_DEVICE: str = "cpu"

# 截断长度（query + 正文 拼起来之后的总长，不是正文单独的长度）。
#
# ★ 为什么是 512 而不是更小 —— 2026-10-12 实测（28 题 / pool=50）：
#
#       max_length   Top-1    recall@5   28 题耗时
#              512   67.9%      89.3%      511.8s
#              256   67.9%      85.7%      287.8s
#              128   75.0%      92.9%      164.6s   ← 看着最好，【没采纳】
#
#   ⚠️ 那三行耗时是在 `batch_size=32`（上游默认）下量的，所以【绝对值已过期】——
#      max_length=512 现在只要约 10.6s/题。但这里要的是【相对关系】，它不受影响：
#      截得越短越快。★ 想复现下面的指标对比，用哪个 batch_size 都一样 ——
#      批大小不改分数（见 RERANK_BATCH_SIZE），但【会改这些耗时数字】。
#
#   时间那一列是干净单调的：越短越快（padding 少了）。但指标那一列【不能信】：
#     ① 75.0% vs 67.9% 是 19/28 → 21/28，**就两题**；recall@5 就一题。
#     ② 变的正好是 S5-5 记录 §7.2 里【已经怀疑 ground truth】的 Q3/Q21/Q26。
#     ③ **不单调**：512→256 时 Q3 是 5→6（变差），256→128 却是 6→1（变好）。
#        截断只会丢信息，"多丢一点反而系统性变好"不是一个能站住的机制 ——
#        那是噪声。
#     ④ 我一次试了 3 档、取了最好那档。**在带噪的 28 题上挑冠军，冠军必然偏高。**
#
#   → 所以保持 512。**不为两题去截掉信息。** 同一份数据里那个"更好的数字"，
#     正是最该警惕的东西：它涨在你已经怀疑尺子的地方。
RERANK_MAX_LENGTH: int = 512

# ★ 一次前向喂几条候选 —— 这是唯一一个【只影响速度、不影响分数】的参数。
#
#   2026-10-12 实测（max_length=512，同一个问题、同一批 50 条候选，14 线程）：
#
#       batch_size     1     4    16    32    50
#       耗时        11.7s 10.5s 15.0s 19.0s 26.4s
#                          ↑ 最快                ↑ 上游默认值
#
#   ★ 为什么改它【不冒任何质量风险】，而调 max_length 要冒：
#     `CrossEncoder.predict` 是【逐对打分】，批大小不参与任何数学 ——
#     它只决定"一次 forward 里塞几条、因此要按多长补齐"。
#     实测验证（3 个查询 × 50 条候选）：max|Δ分| = 4e-7（float32 舍入），
#     **全序完全一致**。而 max_length 是【截断】，那是真的丢信息。
#
#   ★ 为什么上游默认 32 反而不对 —— 读它源码读出来的：
#     predict 内部【已经按长度排序】，而且用 _interleave_sorted_indices 把
#     长短【交错】开。它的 docstring 写着前提是
#     「flash attention ⇒ 没有 padding ⇒ 一批的真实计算量 = 各条长度之和」，
#     所以要在【各批之间】均衡。**那是 GPU 的前提。**
#     我们在 CPU 上，padding 实打实要算 —— 于是"每批都混进一条长的"
#     等于"每批都按最长的补齐"。**上游那个优化，对我们是反的。**
#     调小 batch_size 就是缩小交错的窗口；batch_size=1 时完全无 padding。
#
#   ★ 4 而不是 1：再往下掉会丢掉批内向量化的收益（实测 1 比 4 略慢）。
#     ⚠️ 这个最优值【和 CPU 核数有关】。换机器要重量，
#        量法见 S5-5 记录 4.6。
RERANK_BATCH_SIZE: int = 4

# ★ 生产要不要用重排。2026-10-12 实测（28 题 / dense 粗排 / pool 扫描）：
#
#       pool     5     10     20     30     50    100    200
#       Top-1   57%    57%    61%    64%   [68%]   68%    68%
#       recall@5 71%   79%    86%    89%   [89%]   93%    93%
#
#   所以是 True。效果是实的：Top-1 +18 点，recall@5 +18 点。
#
#   ⚠️ 但这个数字是【检索指标】，不是【答案质量】。
#      Top-1 涨 18 点不等于"答对了的题多 18 点" —— 生成那一层【一次都没重跑过】。
#      见 S5-5 记录第 7.1 节，那是本步最大的诚实缺口。
#
#   ⚠️ 打开它会给【每次问答】加一段 CPU 重排。实测（单进程独占 CPU、14 线程、
#      用下面 RERANK_BATCH_SIZE 的值）：
#         pool=20 → 4.8s    pool=50 → 10.6s    pool=100 → 21.1s
#      28 题全量跑下来是 **10.6 秒/题**（pool=50），和单题数对得上。
#      ⚠️ 这串数【换 batch_size 就全变】—— 上游默认 32 时是 10.8/18.3/25.0。
#      完整测量见 S5-5 记录 4.6。
#      好消息是它【完全不碰 Ollama 的显存】，这正是选 cpu 的原因。
RERANK_ENABLED: bool = True

# ★★ 这个数字是 rerank 能不能有用的【前提】，不是调参旋钮。
#
#   粗排先捞 RETRIEVE_POOL 条，重排之后取 TOP_K 条送进 prompt。
#
#   如果 RETRIEVE_POOL == TOP_K == 5，重排就【只能在 5 条里换顺序】，
#   第 6 名永远上不来 —— 而"正确答案排第 6"恰恰是重排最该修的那种情况。
#   实测（2026-10-12，dense / 28 题）：
#       recall@5  = 71%   ← pool=5  时，Top-1 的绝对上限
#       recall@20 = 86%   ← pool=20 时，Top-1 的绝对上限
#   也就是【多捞 15 条，上限抬高 15 个点】。所以 pool 必须明显大于 TOP_K。
#
#   ⚠️⚠️ 但它绝不能【只活在评测脚本里】。
#      config.TOP_K 是被 pipeline.ask 读的生产值；RETRIEVE_POOL 也必须
#      在生产路径上生效，否则评测量的是"捞 50 排 5"，而生产跑的是"捞 5 排 5"。
#      ★ 这是本仓库已经犯过两次的同一类病（S5-1 的 query 前缀、
#        S5-2 的查询扩展开关）。
#      ✅ 2026-10-12（S5-5）：已接。唯一的生产组合点是
#         pipeline.build_retriever()，pipeline.ask 和 scripts/ask.py 都走它。
#      ⚠️ 但只保证"只有一个来源"还不够 —— 见本块末尾那条。

# ★ 取 50 而不是 20/100 —— 2026-10-12 实测（28 题 / dense 粗排 / rerank）：
#
#       pool     5     10     20     30     50    100    200
#       Top-1   57%    57%    61%    64%   [68%]   68%    68%
#       recall@5 71%   79%    86%    89%   [89%]   93%    93%
#       未进候选  7     6      4      3     [2]     1      0
#
#   p50→p200 唯一的收获是 recall@5 从 89% 涨到 93%（就一题，Q13），
#   代价是候选数翻 4 倍、而 Q7 又掉一名。**这是权衡，不是"越大越好"。**
#
#   ★ 曾经这里是个真两难：pool=50 要 18.3 秒/题，pool=20 只要 10.8 秒，
#     7 个 Top-1 点值不值 7.5 秒不好说。**两难是被 RERANK_BATCH_SIZE 解掉的** ——
#     改批大小之后 pool=50 降到 ~10 秒/题，**和原来 pool=20 一样快**。
#     所以别再拿"50 太慢"当理由回退到 20。
RETRIEVE_POOL: int = 50

# ⚠️⚠️ 教训 —— 比上面那个数字重要：
#
#   ★ "只有一个来源"不够，还要"这个来源被回写过"。
#
#   收尾时差点出事：build_retriever() 确实防住了「pool 只活在评测脚本里」，
#   但决策定在 50 的时候，这个常量还写着 20 ——
#   **于是 S5-5 记录和 ROADMAP 说生产是 @pool50，实际跑的是 @pool20。**
#   同一类病换了个入口又进来一次：上次是"两份实现"，这次是"决策没回写"。
#
#   两种都不会报错。所以改完任何评测结论，回头看一眼这里的值对不对。

# ---- 日志 ----
LOG_LEVEL: str = "INFO"
LOG_FILE: Path = DATA_DIR / "logs" / "paper_rag.log"

# ---- PDF 版面过滤 ----
# 这里曾经有三条【绝对坐标】，2026-10-06 一天之内分两次全部删掉。
# 留下的记录，因为它们说的是同一个病的三次发作。
#
#   第一次 · 删 x 阈值 PAGE_MARGIN_X = 60
#       它把双栏论文的左栏整个当成「边栏水印」扔掉，实测丢 82% 的字符。
#       病根不是数字挑错了，是【把「分辨左右栏」错当成了「过滤非正文」】：
#       过滤是丢弃，分栏是排序。混成一步，就必然要用一个阈值去区分
#       「左栏」和「边栏水印」—— 而这两者在坐标上根本没有差别。
#       拆开之后：左右栏 → ingest.find_column_split()（排序）；
#                 竖排水印 → ingest.is_rotated_block()（看文字走向，与尺寸无关）。
#
#   第二次 · 删 y 阈值 PAGE_HEADER_Y=60 / PAGE_FOOTER_Y=730 → 先改成比例
#       730 是照 Letter 纸（高 792）定的，遇到高 663 的页面永远不触发。
#       改成 PAGE_HEADER_RATIO=0.10 之后，尺寸问题没了，但更根本的问题露出来了：
#
#       ★ 页面顶上，「页眉」和「栏顶第一行」占的是同一个 y。
#         实测 kg-reasoning 第 15 页有两个块 y0/H 都是 0.082，
#         x 分别是 42.1 和 305.1 —— 那是左右两栏各自的栏顶小节标题
#         （'E. Inconsistent Reasoning' / 'H. Knowledge Graph Reasoning Application'）。
#         0.10 这条线把两个真标题一起误杀了。
#         任何 y 区间都分不开它们，因为它们在几何上就是同一行。
#
#       能分开它们的只有【跨页重复】：页眉在 23 页里出现 12 页，标题只出现 1 页。
#       → 见下面的「版面家具」。
#
#   ⚠️ 再往前追，LEARNING_LOG 里 PAGE_MARGIN_X = 108 那次是同一个病的第一次发作
#      （当时把 108 调成 60，治的是症状）。三次的共同结构：
#      把一个文档的经验值，写成了一条关于文档的规则。

# ---- 版面家具（页眉 / 页脚 / 授权水印）----
# 判据：一段文本【去掉数字后】在多少页上重复出现。
# 去数字不是顺手，是必须的 —— 页眉最常见的差异就是页码/卷号：
#     '9456 IEEE TRANSACTIONS ON PATTERN ANALYSIS ...'
#     '9458 IEEE TRANSACTIONS ON PATTERN ANALYSIS ...'
# 归一化后是同一个串，才数得出「12/23 页」。不归一化的话每条只出现 1 次，
# 一条家具都找不出来。
FURNITURE_MIN_PAGES = 3  # 至少出现在这么多页上
FURNITURE_MIN_RATIO = 0.30  # 且至少占全文这么多比例
# 实测（2026-10-06）：
#   kg-reasoning 23 页 → 命中 3 条：页脚水印 23/23、奇数页书眉 12/23、偶数页书眉 11/23
#   kg-embedding 51 页 → 命中 2 条：'# Ge et al.' 25/51、'Knowledge ... Overview #' 25/51
# 两条阈值都留了大余量（最低的 11/23 = 48%，离 30% 还很远），换语料应该不用调。

# ---- 分栏检测 ----
# 栏间空隙只可能出现在页面中部这个区间（左右各留 30% 给正文栏本身）。
GUTTER_SEARCH_LO = 0.30
GUTTER_SEARCH_HI = 0.70
# 判成双栏后，左右两边各自至少要有的块数 —— 防止把「块都挤在左边」误判成双栏。
MIN_COLUMN_BLOCKS = 2

# ---- 标题识别 ----
# ★ 2026-10-06 重写。这里原来是一个字号常量：
#       HEADING_SIZE_GAP = 1.0   # 块主字号 ≥ 正文字号 + 它 → 标题块
#   它是在【原来那篇 RAG 论文】上量的（正文 10.1 ↔ 标题 12.0，中间是空的）。
#   换语料后，两篇 KG 论文双双失效 —— 详见 ingest.is_heading_block 的说明。
#   现在改判【文本】：编号 + 结构词，两个都是学术写作的约定，不是排版的约定。

# 超过它就不可能是一行标题 —— 用来把「独立成行的短标题」和
# 「恰好以编号开头的长段落首行」分开。
# 实测：真标题最长 97 字符（kg-embedding 的 '3 Unified Framework for …'），
#       误判样本最短 134 字符（期刊页眉的 ISSN/DOI 那一行）。120 卡在中间。
HEADING_MAX_CHARS = 120

# 没有编号、但属于固定结构的标题词（归一化成小写后再比）。
# 这份名单要短 —— 每加一个词，就多一类「正好等于这个词的正文块」会被吃掉。
# 只放学术论文里位置固定的那几个。
STRUCTURAL_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "references",
        "reference",
        "bibliography",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "appendix",
        "index terms",
    }
)

# ---- 参考文献：入库时丢弃 ----
# 实测（2026-10-06）：参考文献条目占全库 53% 的 chunk
#   kg-reasoning 346/516（67% 的 chunk、45% 的字符）
#   kg-embedding  27/194（14% 的 chunk、40% 的字符）
# 它们对问答没有价值，却在词面上跟任何领域问题高度重叠（一条里塞满
# 模型名/作者名/会议名），检索时会被大量召回、挤掉真正的正文。
#
# ★ 判据挂在【块的 section】上，而不是「遇到 REFERENCES 就停止」——
#   后者在参考文献不位于文末时会静默截断后面整篇。挂在 section 上，
#   遇到 Appendix 之类还能继续正常收录。
REFERENCE_HEADINGS = frozenset({"references", "reference", "bibliography"})

# ---- 拒答 ----
# ★ 2026-10-09：这个字符串以前【只写在 generator 的 prompt 里】，代码里没有它。
#   后果是：调用方没法判断「模型是真拒答了」还是「解析失败被伪装成拒答」——
#   两者都长成 used=[] 的样子。generator 的注释里警告过这个坑（见 parse_response），
#   但没给它一个可比较的值，警告就落不了地。
#   提到这里之后，prompt 和判断逻辑引用的是同一个值，改一处就够。
REFUSAL_TEXT: str = "资料未提及"

# ---- 查询扩展（query expansion）----
# 实测（2026-10-09，337 条语料）：
#     「知识图谱嵌入模型是按什么标准分类的？」  → 正确段落排第 33 名
#     「KGE 模型是按什么标准分类的？」          → 排第 1 名
#   同一件事，只换了一个词。原因是【词表不匹配】：问题用全称，语料通篇是缩写。
#   注意这既不是模型的锅也不是题目的锅 —— 同一个模型能把「双曲空间」
#   对上 "hyperbolic space"（能跨语言），却对不上「知识图谱嵌入」↔ "KGE"
#   （不能跨缩写）。
#
# 这张表是【手写】的，这就是选「查询扩展」而不是「改索引」的代价：
#   换一批论文就得重写一遍。
#   V3（把标题拼进被向量化的文本）不需要这张表，代价是要重建索引 + 重测拒答阈值。
#   两条路哪个对，等入口跑通之后用评测集 A/B，别在这儿争。
EXPAND_ENABLED: bool = True

# 中文全称 → 语料里实际用的英文写法。
# 值里【同时放缩写和全称】：语料两种写法都有，多给一个多一分命中机会。
#
# ── 往这张表里加词，有一条判断标准（2026-10-12 · S5-3 加题时定的）──
#
# ★ 只加【在这份语料里有区分度】的词。判断方法：在语料里数一下它出现多少条 ——
#   出现得越少越该加，出现得越多越不该加。
#
#   为什么：「状态」→ "state"、"损失" → "loss" 这种看着对，其实是【有害】的。
#   它们本身就是高频英文词，IDF 很低，加进去等于往 query 里掺一个
#   「几乎每条文档都匹配」的词 —— 会把一堆不相关的文档捞进候选池，
#   而真正有区分度的那个词被稀释掉了。
#
#   所以：**加 quintuple，不加 state；加 hyperrectangle，不加 operator。**
#   这跟「BM25 不需要停用词表（IDF 自己会压）」是同一件事的两面 ——
#   在语料侧 IDF 帮你压掉常见词，在 query 侧你得自己别把常见词塞进去。
#
#   ⚠️⚠️ 但这是【倾向】，不是铁律 —— 2026-10-12 实测（见 S5-3 记录第 4.6 节）：
#     稀释机制是真的：Q13 的 target 用 'quintuple' 单独搜时 BM25 排第 1，
#     加上 multi-hop / completion / ftmi 之后掉到第 93 —— 两个"看着有用"的词
#     把一个 rank 1 的答案推掉了 92 名。
#     于是按 IDF 裁表（删掉锚点 df > 40 的 18 条）：
#         Top-1 32% → 25%，recall@5 54% → 39%，【空结果 0 → 8】
#     ★ 高频词条是【入口】，不只是噪音：裁掉它们，8 道题在 BM25 里
#       连一条候选都没有 —— 从"排得靠后"直接变成"完全失明"。
#     ★ 一句话：局部为真的机制，整体可以是负的。
#       所以别看到上面这条判据就去裁表 —— 裁之前先按 4.6 的方法重量一遍。
EXPANSIONS: dict[str, str] = {
    "知识图谱嵌入": "KGE knowledge graph embedding",
    "知识图谱推理": "KGR knowledge graph reasoning",
    "链接预测": "link prediction",
    "负采样": "negative sampling",
    "知识图谱": "knowledge graph KG",
    "多模态": "multi-modal",
    "时序": "temporal",
    "静态": "static",
    # ── 2026-10-12 · S5-3 加题时补的：新 5 篇论文的术语 ──
    # 下面每一条都是【先查了语料里的实际写法】才写的，不是猜的。
    "时序知识图谱": "temporal knowledge graph TKG",
    "知识图谱补全": "knowledge graph completion",
    "多跳推理": "multi-hop reasoning",
    "多跳": "multi-hop",
    "补全": "completion",
    "因果推理": "causal inference",
    "元学习": "meta learning",
    "少样本": "few-shot",
    "时空": "spatiotemporal",
    "不确定": "uncertain",
    "模糊隶属度": "fuzzy membership",
    "模糊": "fuzzy",
    "超矩形": "hyperrectangle",
    "五元组": "quintuple",
    "动作空间": "action space",
    "逻辑规则": "logic rule",
    "逻辑约束": "logic constraint",
    "目标函数": "objective function",
    "奖励": "reward",
    "评测指标": "evaluation metric MRR Hits@n",
    # ★ 下面 4 条是 S5-3 跑测之前体检出来的【词表缺口】，不是拟合题目：
    #   体检方法 = 把每道题的扩展结果拿去分词，看有没有仍然是【空】的。
    #   空 = BM25 一个字都进不去 = 这题对 BM25 完全失明。
    #   「知识图谱推理」在表里、而「知识推理」不在 —— 这是个纯粹的漏项。
    "知识推理": "knowledge reasoning",
    "指标": "evaluation metric MRR Hits@n",
    "路径型": "path-based",
    "规则型": "rule-based",
}

# ---- 切分 ----
# 清洗后短于它的块直接丢弃。拦的是「图里的文字」和「公式碎片」：
#   'q(x)'  'd(z)'  'Query Encoder'  'Barack Obama was born in Hawaii.(x)'
# 这个数字没有数据支撑（长度分布是连续的，没有天然断点），
# 是「代价权衡」的结果：丢一个 35 字的碎片几乎不损失知识，
# 留 110 个垃圾 chunk 却是 32% 的索引噪音。以后可以调。
MIN_CHUNK_CHARS = 40
