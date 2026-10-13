"""端到端流水线 —— 把 Stage 1~4 装配成一条线。

这是项目里【第一条】「用户问一句 → 拿到带引用的答案」的路径。
在此之前每个 Stage 都有自己的探针，但没有任何一条线把它们串起来 ——
Stage 3 的拒答和 Stage 4 的引用校验，在换语料之后一次都没跑过。

流水线长这样：

    用户问题（中文）
        │
        │  ① expand_query()                    ← 查询扩展（可关）
        ▼
    "知识图谱嵌入模型是按什么标准分类的？ KGE knowledge graph embedding"
        │
        │  ② embed.embed_one()
        ▼
    query 向量 (1024,)
        │
        │  ③ retriever.DenseRetriever.search()   和【全部 337 条】语料比一遍
        ▼
    [(chunk_id, 分数)] × 5
        │
        │  ④ 按 chunk_id 取回 chunk 字典   ★ 不是按行号
        ▼
    [chunk, ...] × 5
        │
        │  ⑤ generator.answer()   拼 prompt → 调模型 → 解析
        ▼
    {"answer": ..., "used": [1, 3], "quotes": {"1": "……原句……"}}
        │
        │  ⑥ cite.verify()   拿 quotes 回原文里核
        ▼
    问题清单 ["[4] quotes[1] 在材料里找不到", ...]    空列表 = 全部经得起核对

── 这是【产品】还是【脚手架】？ ──

    产品。三条理由，缺一条我都会把它放进 scripts/：
      1. 【不是一次性的】CLI 要调它，以后 web 界面要调它，批量评测也要调它。
         一次性的东西才叫脚手架。
      2. 【含判断】这个文件里唯一有技术含量的一行，是区分
         「真拒答」和「空 used」（见 ask() 里的 _judge）。判断可复用，
         输入输出格式不可复用 —— 后者是脚手架的特征。
      3. 【在 paper_rag/ 里】它 import 了四个模块，只做装配，不碰 PDF、
         不碰 HTTP、不碰 print。

    scripts/ask.py 才是脚手架：只负责把命令行参数喂进来、把结果打印出去。

── 和其它模块的分工 ──

    ingest.py      PDF  → chunk           （离线，一次）
    embed.py       文本 → 向量
    index.py       向量 ⇄ 磁盘
    retrieve.py    向量 → 前 k 条         （最底层的原语）
    retriever.py   文本 → 前 k 条         （检索器的统一接口，包住 retrieve）
    generator.py   问题+材料 → 模型 → 结构化结果
    cite.py        结构化结果 → 引用问题清单
    pipeline.py    ↑ 以上全部串起来       ← 本文件，只做编排，不含新逻辑
"""

from paper_rag.cite import verify
from paper_rag.config import (
    EMB_MODEL,
    EXPANSIONS,
    EXPAND_ENABLED,
    GEN_JSON_SCHEMA,
    GEN_MODEL,
    REFUSAL_TEXT,
    RERANK_ENABLED,
    RETRIEVE_POOL,
    TOP_K,
)
from paper_rag.generator import answer
from paper_rag.rerank import CrossEncoderReranker, RerankedRetriever
from paper_rag.retriever import DenseRetriever, Retriever

__all__ = ["expand_query", "build_retriever", "ask"]


def build_retriever(
    client,
    emb_model: str,
    vectors,
    ids: list[str],
    chunks: list[dict],
    *,
    k: int = TOP_K,
    pool: int = RETRIEVE_POOL,
    enabled: bool = RERANK_ENABLED,
) -> Retriever:
    """建一个【生产用的】检索器：粗排捞宽 → （可选）重排 → 取前 k。

    ── 为什么要把它放在产品里，而不是塞进 scripts/ask.py ──

        因为「生产怎么组合检索」这件事，会有【不止一个】调用方：
        scripts/ask.py 现在要，以后的 web 入口要，批量评测也要。

        ★ 各写一份的后果，这个仓库已经吃过【两次】：
            S5-1 —— query 前缀只活在评测脚本里，生产路径压根没这个概念
            S5-2 —— 查询扩展开关在 eval_embed.py 里从来不存在，
                    于是它历史上量的一直是"没扩展"的数字
          两份实现改一处漏一处，而且【不会有任何东西报错】。

        所以组合逻辑只有这一份，调用方各自决定的是【什么时候建】
        （见下面的 ⚠️），不是【怎么建】。

    ── ⚠️⚠️ 什么时候建：循环外，一次 ──

        这个函数每调一次都新建一个对象。重排模型的【权重】有进程级缓存
        （rerank._load_model），所以重复调用不会重复下载 2.3GB ——
        但对象构造、id_to_text 的引用、以及那层包装仍然白做。

        真正要防的是这个形状：

            for q in questions:                 # ← 循环
                ask(client, q, ...)             # ← ask 内部会调本函数

        所以入口应该在循环【外】建好，然后传进 ask(retriever=...)。
        见 scripts/ask.py。

    参数:
        client:    ollama 客户端
        emb_model: 嵌入模型（必须和建索引时一致）
        vectors:   索引向量 (N, D)
        ids:       索引 id 列表 (N,)
        chunks:    全量 chunk 列表（重排器要用它把 id 换回正文）
        k:         最后取几条送进 prompt
        pool:      粗排捞几条。★ 必须 > k，否则重排只能在 k 条里换顺序
        enabled:   False 就只返回粗排（相当于回到 S5-3 的行为）

    返回:
        一个满足 Retriever 的东西 —— 名字和 search() 而已。
        调用方【不需要知道】它背后是纯 dense 还是 dense+rerank。
    """
    base = DenseRetriever(client, emb_model, vectors, ids)
    if not enabled:
        return base

    return RerankedRetriever(
        base,
        CrossEncoderReranker.from_chunks(chunks),
        pool=pool,
        k=k,
    )


def expand_query(question: str) -> str:
    """把问题里出现的【中文全称】补上语料里实际用的【英文写法】。

    为什么需要：中文用户会说「知识图谱嵌入模型」，而语料里通篇是 "KGE"。
    这两个字符串在向量空间里离得很远 —— 实测排第 33 名 vs 第 1 名。

    怎么做：扫描【原问题】，命中几个词，就把它们的英文写法【追加到末尾】。

        为什么不直接替换原文？因为要保留用户的原话 ——
        万一某个词没在表里、但原词本身能对上，替换等于把这条线索丢了。

        为什么扫的是原问题、而不是边扫边改的结果？
        因为「知识图谱」是「知识图谱嵌入」的子串。边扫边改的话，
        第一轮补进去的英文会被第二轮再扫一遍。扫原文 + 只追加，
        两个问题一起消失。

    参数:
        question: 用户原问题

    返回:
        补充过的问题。一个词都没命中就【原样返回】——
        调用方可以靠 `expanded == question` 判断扩展有没有生效。
    """
    # 长词优先：「知识图谱嵌入」必须先于「知识图谱」被匹配到，
    # 否则短词会抢先命中，长词反而被跳过。
    keys = sorted(EXPANSIONS, key=len, reverse=True)

    # (在问题里的位置, 命中的中文词, 对应的英文写法)
    hits: list[tuple[int, str, str]] = []

    for zh in keys:
        pos = question.find(zh)
        if pos < 0:
            continue
        # 被更长的词吃掉了：「知识图谱嵌入」命中后，「知识图谱」就不要再补一遍
        if any(zh in longer for _, longer, _ in hits):
            continue
        hits.append((pos, zh, EXPANSIONS[zh]))

    if not hits:
        return question

    # ★ 按【在问题里出现的先后】排，不是按词长排。
    #   不排的话，「多模态知识图谱」会补成 "knowledge graph KG multi-modal"，
    #   把语料里的固定搭配 "multi-modal knowledge graph" 拆散了 ——
    #   而嵌入模型读的是连续片段，拆开等于白补。
    hits.sort(key=lambda h: h[0])

    return question + " " + " ".join(en for _, _, en in hits)


def _judge(parsed: dict) -> tuple[str, str | None]:
    """拿到模型的结构化输出后，判断这一次属于哪种情况。

    ── 为什么需要这个函数 ──

        generator 的注释里记着一个坑：解析失败被【伪装成拒答】。
        两者看起来都是 used=[]，但药完全不一样：
            真拒答   → 正常，材料里确实没有答案，该跟用户直说
            空 used  → 异常，模型给了答案却一个引用都不交，这条答案不可信

        靠 answer 字段的值区分，而那个值必须和 prompt 里要求的一致。
        → 所以 REFUSAL_TEXT 提到了 config，prompt 和这里引用同一个常量。

    返回:
        (status, note)
        status: "answered" | "refused" | "suspect"
        note:   给 suspect 用的一句话解释，其余为 None
    """
    used = parsed.get("used") or []

    if used:
        return "answered", None

    if parsed.get("answer", "").strip() == REFUSAL_TEXT:
        return "refused", None

    # ★ 这里就是那个坑：有答案、没引用、又不是标准拒答话术。
    #   以前会被当成「本次没有引用，跳过校验」—— 一次解析失败
    #   就这么伪装成了一次正常的拒答，全程无人报错。
    return "suspect", (
        f"模型给了答案却没有任何引用（used=[]），"
        f"而 answer 又不是标准的拒答话术 {REFUSAL_TEXT!r}。"
        f"这【不是】拒答，是模型没按格式交付 —— 这条答案不可信。"
    )


def ask(
    client,
    question: str,
    chunks: list[dict],
    vectors,
    ids: list[str],
    *,
    k: int = TOP_K,
    gen_model: str = GEN_MODEL,
    emb_model: str = EMB_MODEL,
    expand: bool = EXPAND_ENABLED,
    options: dict | None = None,
    retriever: Retriever | None = None,
    gen_schema: bool = GEN_JSON_SCHEMA,
) -> dict:
    """完整走一遍：扩写 → 检索 → 生成 → 校验引用。

    参数:
        client:    ollama 客户端（embed 和生成共用一个）
        question:  用户的原问题
        chunks:    chunks.json 里的 chunks 列表（全量，不只是候选）
        vectors:   index.load() 拿到的向量矩阵 (N, D)
        ids:       index.load() 拿到的 id 列表 (N,)
        k:         检索几条
        gen_model: 生成模型
        emb_model: 嵌入模型 —— 必须和建索引时用的是同一个，
                   否则向量不可比。index.load(model=...) 会替你核对这一条
        expand:    要不要做查询扩展（默认跟 config.EXPAND_ENABLED）
        options:   透传给 Ollama 的生成参数
                   （评测时传 {"temperature": 0} —— 要可复现）
        retriever: 已经建好的检索器。★ 循环里调 ask 时【必须】传 ——
                   不传的话每次都会新建一个（含重排器），白做一堆构造。
                   见 build_retriever 的 ⚠️ 说明。
        gen_schema: 要不要用 JSON Schema 约束解码（默认跟 config.GEN_JSON_SCHEMA）。
                   ★ S5-7 加的。它修的是一整类【解析失败】，和检索无关 ——
                   见 config.GEN_JSON_SCHEMA 和 generator.GEN_SCHEMA。

    返回:
        {
          "question":   str,    # 用户原话，一字不改
          "query":      str,    # 实际送去检索的文本
          "expanded":   bool,   # 扩展有没有真的生效
          "retriever":  str,    # ★ 这次是【哪个检索器】捞的。
                                #   以前没这一项，于是「换了检索器」和
                                #   「没换」在输出里长得一模一样 ——
                                #   和当年不打印论文标题是同一个病。
          "retrieved":  [{"chunk": {...}, "score": float}, ...],
          "parsed":     dict | None,   # 模型输出；解析失败时为 None
          "problems":   [str, ...],    # cite.verify 的问题清单，空 = 通过
          "status":     "answered" | "refused" | "suspect" | "parse_failed",
          "note":       str | None,    # suspect / parse_failed 时的解释
        }

    ⚠️⚠️ retrieved[].score 的【含义取决于检索器】：
        纯 dense        → 余弦相似度，∈ [−1, 1]
        dense + rerank  → cross-encoder 的 logit，可负、无上界
        两者【不能比大小】，也不能拿去套任何按余弦定的阈值。
        判断这次检索好不好，看【排序】和"对的那条在不在里面"，
        不要看分数的绝对值。result["retriever"] 说明它是哪一种。

    ── 两个刻意的设计决定 ──

        1. 【扩展只影响检索，不影响 prompt】
           build_prompt 的契约写着「用户的原问题，一字不改」——
           那个契约是对的。扩展是检索侧的技巧，是拿来骗【嵌入模型】的，
           不该出现在递给生成模型的问题里。
           想改这个行为之前，先想清楚：扩写后的词给生成模型看，
           是帮它、还是给它添乱？

        2. 【解析失败不抛异常，而是变成一个显眼的 status】
           ask() 是给入口用的。入口崩掉对用户毫无意义。
           但【不抛异常】不等于【假装没事】—— parse_failed 和 refused
           是两个完全不同的 status，不允许合并。
    """
    # ── ① 扩写 ──
    query = expand_query(question) if expand else question

    # ── ② + ③ 向量化 + 检索 ──
    # 用 emb_model 而不是 gen_model —— 这是两个完全不同的模型，
    # 一个把文本压成 1024 维向量，一个把 prompt 变成文字。混用不会报错，
    # 只会静默给出垃圾向量（维数可能都碰巧一样）。
    #
    # ★ 2026-10-12：原来这里是 embed_one + top_k_np 两行直调，
    #   现在改走 retriever.DenseRetriever —— 为了和 scripts/eval_retrieve.py
    #   走【同一份】检索实现。原来评测那边是手抄的一份循环，
    #   两边改一处漏一处【不会有任何东西报错】。
    #
    #   顺带修掉一个潜伏的分歧：query 前缀原来只活在评测脚本里，
    #   生产路径压根没有这个概念。今天两边数字一样，只因为 bge-m3
    #   的前缀是空串；换成 qwen3-embedding 就会一边加一边不加。
    #
    # ★ 2026-10-12（S5-4）：检索器改成【可以注入】。
    #   默认不传时行为不变（自己建一个 dense），所以老的调用方不用改。
    #   但生产入口【应该】在循环外建好再传进来 —— 因为现在检索器里
    #   可能裹着一个 cross-encoder，而"每题重建一次"的代价是实打实的。
    if retriever is None:
        retriever = build_retriever(client, emb_model, vectors, ids, chunks, k=k)

    hits = retriever.search(query, k=k)

    # ── ④ chunk_id → chunk 字典 ──
    # ★ 必须按 chunk_id 查，不能按行号。
    #   hits 里的 id 来自 ids.json，chunks 来自 chunks.json ——
    #   两个文件是分开写的。一旦其中一个重跑过，行号就对不上了，
    #   而【行号对不上不会报错】，只会静默返回一段风马牛不相及的材料。
    by_id = {c["chunk_id"]: c for c in chunks}
    retrieved = []
    for cid, score in hits:
        if cid not in by_id:
            raise RuntimeError(
                f"索引里有 {cid!r}，chunks.json 里没有 —— "
                f"两个文件不同步。重建：python scripts/build_vectors.py"
            )
        retrieved.append({"chunk": by_id[cid], "score": score})

    materials = [r["chunk"] for r in retrieved]

    # ── ⑤ 生成 ──
    try:
        parsed = answer(client, question, materials, model=gen_model,
                        options=options, schema=gen_schema)
    except Exception as e:
        # 不抛。但也不许悄悄降级成拒答 —— 见 _judge 的说明。
        return {
            "question": question,
            "query": query,
            "expanded": query != question,
            "retriever": retriever.name,
            "retrieved": retrieved,
            "parsed": None,
            "problems": [],
            "status": "parse_failed",
            "note": f"{type(e).__name__}: {e}",
        }

    # ── ⑥ 校验引用 ──
    # 拒答时模型没交引用，没什么可核的 —— 直接跳过，
    # 因为 answer() 的契约保证了「拒答 ⇒ used=[]」，而 used=[] 时 verify 无事可做。
    status, note = _judge(parsed)
    problems = verify(parsed, materials) if status == "answered" else []

    return {
        "question": question,
        "query": query,
        "expanded": query != question,
        "retriever": retriever.name,
        "retrieved": retrieved,
        "parsed": parsed,
        "problems": problems,
        "status": status,
        "note": note,
    }
