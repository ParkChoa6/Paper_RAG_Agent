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
EXPANSIONS: dict[str, str] = {
    "知识图谱嵌入": "KGE knowledge graph embedding",
    "知识图谱推理": "KGR knowledge graph reasoning",
    "链接预测": "link prediction",
    "负采样": "negative sampling",
    "知识图谱": "knowledge graph KG",
    "多模态": "multi-modal",
    "时序": "temporal",
    "静态": "static",
}

# ---- 切分 ----
# 清洗后短于它的块直接丢弃。拦的是「图里的文字」和「公式碎片」：
#   'q(x)'  'd(z)'  'Query Encoder'  'Barack Obama was born in Hawaii.(x)'
# 这个数字没有数据支撑（长度分布是连续的，没有天然断点），
# 是「代价权衡」的结果：丢一个 35 字的碎片几乎不损失知识，
# 留 110 个垃圾 chunk 却是 32% 的索引噪音。以后可以调。
MIN_CHUNK_CHARS = 40
