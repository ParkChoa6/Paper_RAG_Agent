"""生成 —— 把检索结果变成一句人话。

这是 Stage 3 的主场，也是整个 RAG 里【唯一一处你直接对模型说话】的地方。

    question ──┐
               ├─▶ build_prompt ──▶ llm.generate ──▶ parse_response ──┐
    chunks ────┘                                                      │
                            ┌─────────────────────────────────────────┘
                            ▼
                     cite.verify ──▶ 问题清单（Stage 4）

三个函数的分工：
    build_prompt    你 → 模型     整个 Stage 3 的核心，所有约束写在这里
    parse_response  模型 → 你     把文本变回程序能用的东西，只管【形状】
    answer          编排三件事    胶水，本身不含逻辑

    Stage 4 的引用核对在 cite.py，不在这里 —— 它变化的原因和这三个都不同。

【硬性要求】不许用 LangChain。全手写一遍，你会看清 RAG 的真实面目
就是一个 for 循环加一次 API 调用。之后再用框架，你是"选择"用，不是"只能"用。
"""

import json
import logging

from paper_rag.config import GEN_JSON_SCHEMA, GEN_MODEL
from paper_rag.llm import generate

logger = logging.getLogger(__name__)

# 表头里论文标题的截断长度。★ 魔法数字，但是【量出来的】，不是拍的。
#
# 为什么必须截断、为什么是 40 —— 见 render_material 的 docstring，那里有同场 A/B：
#   太长（70 或全文）→ 模型干脆不交 quotes
#   太短（20/25）    → 判别词被切断，模型又分不清两篇
# 它成立的隐含前提：判别词出现在标题前 40 字内。换论文时要重新量。
TITLE_MAX = 40


def render_material(number: int, chunk: dict) -> str:
    """渲染【一条材料】—— 和模型看到的完全一致，一字不差。

    ══════════ 为什么这个函数必须存在 ══════════

    ★ 它是 build_prompt() 和 cite.verify() 的【共用函数】。两边必须调同一个，
      这不是"代码复用"的洁癖，是【正确性】的要求：

        build_prompt 给模型看的是：  「表头 + 正文」
        cite.verify  拿来比对的是：  「正文」

      两者的差，正好是表头。模型把表头一起抄进 quotes（它会这么干 ——
      表头就写在材料里面，它分不清哪里算"材料"）时，
      精确匹配就找不到 → 一条【真引用】被报成【伪造】。

      实测（2026-10-09，第一次端到端跑就撞上）：
        模型抄的： "[1] RotatE [94] models entities in the complex vector space…"
        源文本是：       "RotatE [94] models entities in the complex vector space…"
        → verify 报「原文中找不到对应内容（疑似伪造）」

    ── 为什么不是"把 quotes 开头的 [n] 剥掉" ──

      那是【补偿】一个我们自己制造的偏差，不是【消除】它。
      它贴着当前的表头格式写死；表头一改，它立刻失效，而且是【静默失效】——
      真引用又开始被误报，没人会记得去改那个剥离函数。

      抽成共用函数之后，表头怎么改，两边同时改，永远一致。
      ★ 这个预言在 2026-10-10 兑现了：表头加了论文标题（见下），
        cite.verify 一行没动，自动跟上。

    ── ★ 表头里放什么：论文标题是【判别器】，不是装饰 ──

      2026-10-10 实测（端到端，问「知识图谱嵌入模型是按什么标准分类的？」）。
      这道题检索没问题，标准答案就在材料 [2]（那篇的摘要）里：
        "we focus on two main branches of KG embedding (KGE) design:
         1) distance-based methods and 2) semantic matching-based methods"

      材料 [5] 来自【另一篇】（推理综述）：
        "Static KGR models can be categorized into embedding-based,
         path-based, and rule-based models"

      模型用了 [5]，答「嵌入基 / 路径基 / 规则基」—— 张冠李戴，而 verify 报 0 条
      （[5] 那句原文确实存在，引用是真的）。这是 Stage 4 结构上抓不住的那一类。

      隔离实验（同题同参，只换材料）：只给 [2] → 答对；[2]+[5] → 答错。
      **[5] 在场，就把 [2] 盖住了。** 不是"多了条噪音"，是压制。

      然后只改表头，5 个 seed 各跑一遍：

        表头                                判定
        [1] （ABSTRACT，第 1 页）            错 错 错 错 错
        [1] （kg-embedding-overview · …）    混 混 混 混 混   ← 归属修好了，答案还是混
        [1] （Knowledge Graph Embedding: An Overview · …）  对 对 对 对 对

      三级梯度，单调。**判别的关键不是"有没有出处"，是"出处的名字里有没有那两个字"**：
      `kg-embedding-overview` 修好了归属（模型认出 [5] 是推理那篇），
      但只有真标题 "Knowledge Graph **Embedding**" 才让模型敢把 [5] 排除掉。

      ★ 这不是"多加信息总没坏处"。同一个实验里，加 paper_id 是个【半吊子修法】：
        它让模型说对了出处、却照旧把错的分类塞进答案 —— 比原来更难发现。
      ★ 换一篇论文之前先看 title 干不干净：当时两篇各 167/170 块，
        title 每篇只有一个、全对。（section 里倒是有 12 块垃圾，见遗留表。）

    ── ★ 标题【必须截断】，而且 40 这个数字是量出来的 ──

      加【全文】标题会弄坏另一个问题（问「静态和时序推理的区别」，模型交
      used=[3] 但 quotes={} —— 干脆不交引用了）。同场 A/B：

        表头                     Q2（失败 A）      Q3（交没交引用）
        old（只有 section）       错 错 错 错 错    交 交 交 交 交
        title[:40]               对 对 对 对 对    交 交 交 交 交
        title 全文               对 对 对 对 对    空 空 空 空 空

      两个约束把可用区间夹出来（同一场扫截断长度）：
        太短 → 判别词被切断   截断 20/25 时 "Reasoning" 没了 → Q2 错/混
        太长 → 模型不交引用   截断 70 时 Q3 全空
      40 落在两侧都有余量的位置。**这是魔法数字，不假装不是** ——
      它成立的隐含前提是「判别词出现在标题前 40 字内」
      （本文两篇：Embedding 第 16 位、Reasoning 第 30 位）。
      ★ 换论文时要【重新量一遍】，不能假设它还成立。

      ⚠️ 量这个时踩了一个大坑，记在这儿：**seed 跨会话不可复现。**
        同一组参数，不同时间跑，结果不同。有一次测 45/50 两个长度，
        测出完全一样的模式（seed 1 交、2~6 空）—— 两个不同长度给出同一个
        模式，说明那一场里表头根本没起作用，是会话状态在主导，那组数据作废。
      ★ 所以比较方案的规矩是：**所有变体必须在同一个进程里跑完，
        并且要带一个已知基线当"温度计"**。基线的表现要是和上次不一样，
        这一整场作废，不能拿来下结论。

    ── 代价（说清楚，别假装没有）──

      表头从此【也变成可被引用的文本】了。模型抄一句 "2.1 Distance-based Models"
      或标题本身当 quotes 也能通过校验 —— 那是真文本，只是没营养。
      加标题把这个可引用面又扩大了一点。
      这个代价我们认：它要求模型做一件很蠢的事才会触发，
      而现在的 bug 是【必然触发】的。必然的假警报 > 罕见的漏网，
      理由见本文件另一处：用户学会无视警报之后，校验就等于不存在。

    参数:
        number: 材料编号，从 1 开始（就是 used / quotes 里那个数字）
        chunk:  chunks.json 里的一条

    返回:
        str —— 表头 + 换行 + 正文。【不含】材料之间的空行分隔符。
    """
    # 表头三段：编号 / 论文标题 / 章节 + 页码。缺一不可，各有各的用处：
    #   编号      —— used 和 quotes 里的数字，指的就是它
    #   论文标题  —— 判别器。两篇是相邻主题，没有它模型分不清谁是谁（见上面的实测）
    #   章节+页码 —— 给人看的出处，也是 quotes 带表头时能对上的那部分
    #
    # 注意标题是【截断】的（TITLE_MAX），不是全文 —— 全文会弄坏 Q3，理由见 docstring。
    # 给人看的那一侧（scripts/ask.py）印的是全文，不截断：那里不受模型脾气影响。
    return (
        f"[{number}] （{chunk['title'][:TITLE_MAX]} · {chunk['section']}，"
        f"第 {chunk['page_start']} 页）\n"
        f"{chunk['text']}"
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    """把「问题 + 检索到的材料」拼成一段给模型的 prompt。

    ══════════ 这是 Stage 3 的核心 ══════════

    输入:
        question: 用户的问题，例如
                  "RAG 是怎么把检索到的文档融进生成过程的？"
        chunks:   top_k() 的结果，每条是一个 dict（来自 chunks.json）：
                    {
                      "chunk_id":   "kg-embedding-overview#8",
                      "text":       "……正文……",
                      "paper_id":   "kg-embedding-overview",
                      "title":      "Knowledge Graph Embedding: An Overview",
                      "section":    "ABSTRACT",
                      "page_start": 1,
                      "page_end":   1,
                    }
                  ★ 注意里面【有 title】。它不是装饰 —— 表头要拿它做判别器，
                    两篇论文主题相邻时，没有它模型分不清哪条材料属于哪篇。
                    见 render_material 的 docstring。

    返回:
        str —— 完整的 prompt

    ── 必须写进去的三件事，缺一不可 ──

        1. 【材料】chunks 的正文，并且【编号】
        2. 【问题】用户的原问题，一字不改
        3. 【约束】最容易写漏的一块，它决定两件事：
             a. 材料里没有 → 必须说"未提及"，【不许编】
             b. 输出格式 —— 必须是程序能解析的东西

    ── 输出格式（定死，调用方按这个写）──

        {"answer": "……回答……", "used": [1, 3], "quotes": {"1": "……原文原句……"}}

        used 装的是【你在 prompt 里给材料编的编号】，不是 chunk_id。
        quotes 是 Stage 4 加的：模型【逐字抄】的原文，供 cite.verify 核对。
        ★ 必须是【英文原文】，不许翻译 —— 语料是英文、答案是中文，
          这个落差会诱导模型把材料译成中文再当引用交。见下面第 4 条。

    ── 提示（按顺序想，别跳）──

        1. 编号从几开始？
           更重要的是：为什么不让模型直接输出 chunk_id？
           （提示：模型擅长抄一个 1~5 的数字，还是擅长抄
             "rag-2005.11401#8" 这种 18 个字符的字符串？
             它抄错了，你【不会收到任何报错】—— 冒烟测试一切正常。）

        2. 每段材料要不要带上 section / 页码？
           （想一想：Stage 4 要"引用能点回原文"。现在不放，
             到时候还补得回来吗？）

        3. 「拒答」怎么写才叫【强制】？
           Day 0 已经验过：这个模型嘴上答应得很好，转头照样编。
           所以不能只写"请勿编造"—— 得给它一个【可执行的动作】：
           什么条件下、输出什么、放进哪个字段。
           写完之后自查一遍：这一条是"可执行的"还是"一个愿望"？

        4. 材料之间怎么分隔？
           两段材料直接首尾相接，模型会把它们读成【一段连续的话】。

        5. 模型看到的只是一串字符。它分不清哪里是材料、哪里是指令 ——
           除非你用结构明确告诉它。

    ── 自查 ──

        - 把 chunks 换成空列表 [] 会怎样？prompt 还成立吗？
          模型会输出什么？这是拒答的第一道关。
        - f-string 里嵌套引号注意别撞（0A-1 那个坑）。
          要放 JSON 示例的话，考虑用三引号整块写。
    """
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  v0（已废弃，留着做对照）                                          ║
    # ║      parts = ["请根据以下材料回答问题。\n"]                         ║
    # ║      for i, c in enumerate(chunks, 1):                            ║
    # ║          parts.append(f"{i}. {c['text']}")                        ║
    # ║      parts.append(f"\n问题：{question}")                           ║
    # ║  跑出来四枪：格式全丢 / 先拒答再编 / 张冠李戴。见 probe_generate。   ║
    # ╚══════════════════════════════════════════════════════════════════╝

    parts = []

    # ── 1. 角色 + 禁令 ──
    # 放【最前面】。模型对开头和结尾的注意力最重，中间的约束容易被淹。
    parts.append(
        "你是一个严谨的论文问答助手。"
        "你只能依据下面【材料】里写的内容回答问题，"
        "不许使用你自己的任何知识。"
    )
    parts.append("")

    # ── 2. 材料 ──
    # 每段材料给一个 [编号]，这个编号就是 used 里要填的数字。
    # 为什么不用 chunk_id？—— "rag-2005.11401#8" 是 18 个字符，
    # 模型抄错一个字符你【不会收到任何报错】，只会拿到一个不存在的 id。
    # 抄一个 1~5 的数字，它错不了。
    parts.append("## 材料")
    parts.append("")
    for i, c in enumerate(chunks, 1):
        # section 和页码现在【必须】放进去 —— 它们是 Stage 4「引用点回原文」
        # 的唯一依据。等 Stage 4 再想加，prompt 就得重写一遍。
        #
        # ★ 排版走 render_material()，不许在这里手写 ——
        #   cite.verify 拿同一个函数重建这段文本做比对。
        #   在这里多写一行、少写一行，就会造出【两边看到的东西不一样】的 bug，
        #   而那个 bug 的表现是"真引用被报成伪造"，没人会往这里查。
        parts.append(render_material(i, c))
        # 空行是【分隔符】。两段材料首尾相接，模型会读成一段连续的话。
        parts.append("")

    # ── 3. 问题 ──
    parts.append("## 问题")
    parts.append(question)
    parts.append("")

    # ── 4. 约束 —— 最容易写漏的一块 ──
    #
    # 每条都必须是【可执行的动作】，不能是【愿望】。
    #   ❌ "请不要编造"        —— 模型会点头，然后照编（第 3、4 枪）
    #   ✅ "材料里没有 → answer 写成「资料未提及」，used 写成 []"
    #      —— 这是一组【明确的状态】，代码能验
    parts.append("## 回答要求")
    parts.append("1. 只依据【材料】回答。材料里没写的东西，一个字都不要加。")
    parts.append(
        "2. 如果材料里找不到答案，answer 必须【原样】写成「资料未提及」，"
        "并且 used 必须是 []（空数组）。不许推测，不许用自己的知识补充。"
    )
    parts.append(
        "3. used 填你【实际用到】的材料编号，就是上面 [1] [2] 里的数字。"
        "答案里每个结论，你都要能指回某一号材料。"
    )
    # ★ Stage 4 新增。这条的措辞和 cite.verify() 的严格度是【配套】的 ——
    #
    #   校验用【归一化后】比对（转小写 / 标点换空格 / 压空白，见 cite._norm），
    #   不是精确 in。所以 prompt 可以说"逐字照抄"而不必担心标点差异被误报。
    #   哪天校验改成精确 in，这句就要跟着收紧。
    #
    #   两者要一起改。校验收紧了而 prompt 没说 → 真引用被误报成伪造。
    #   这就是「约束的严格度和它宣告的严格度必须一致」——
    #   你没有告诉它规则，却拿规则罚它，那不叫校验，叫钓鱼。
    #
    # ★ 2026-10-09：加了「必须是英文原文，不许翻译」。原因是量出来的，不是猜的 ——
    #   qwen2:7b 被要求交引用时，会【把英文材料忠实翻译成中文】再交上来：
    #     材料原文：BERT is a pretrained bidirectional language model that is built
    #               based on transformer architecture. Since one of the tasks of
    #               BERT pretraining is next-sentence prediction …
    #     模型交的：BERT是2018年提出的语言模型，基于变换器架构，…它以下一个句子
    #               预测任务为目标…
    #   内容句句对得上出处，但字符串在原文里【一个字都找不到】，verify 判它伪造。
    #   那不是模型在撒谎，是【契约本身没法被执行】：answer 用中文写，
    #   却要求它交一句英文原话 —— 没人告诉它这两件事可以并存。
    #   模型有能力做到（正常流程里它就交过英文原句），只是一直没人说。→ 说清楚。
    parts.append(
        "4. quotes 里填【材料原文里的英文原句】—— 从上面材料里逐字复制，"
        "一个字都不要改：标点、大小写、空格全部保持原样。"
        "★ answer 用中文写，但 quotes 【必须是英文原文，不许翻译】。"
        "翻译过来的句子不算引用：代码会拿它回原文里比对，"
        "译文一个词都对不上，会被判定为伪造引用。"
        "键是材料编号（写成字符串），值是你抄下来的那句英文原文。"
        "如果没有用到任何材料（拒答），quotes 写成空对象 {}。"
    )
    parts.append("")

    # ── 5. 输出格式 ──
    # 给一个【字面示例】让模型抄，比用文字描述格式有效得多。
    # "不要 markdown 代码块" 这句要说 —— Day 0 验过这个模型能做到。
    parts.append("## 输出格式")
    parts.append("只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字：")
    parts.append(
        '{"answer": "你的回答（中文）", "used": [1, 3], '
        '"quotes": {"1": "材料1里的英文原句，逐字复制", "3": "材料3里的英文原句，逐字复制"}}'
    )

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  ★ S5-7：把上面那段「输出格式」交给采样器当【语法】。
#
#  这是 build_prompt() 第 5 节那份契约的【机器可读版】，
#  形状和 parse_response() 第 3 步校验的字段【完全一致】 —— 三者是同一件事：
#
#      build_prompt 第 5 节   给【模型】看的字面示例（人话）
#      本常量                 给【采样器】看的语法（它不接受人话）
#      parse_response 第 3 步  给【代码】做的形状校验（事后把关）
#
#  ⚠️⚠️ 三份，就【必须一起改】。这和 render_material 那条是同一条规矩：
#      「守卫的判据和它守的那个操作只要是两套，就会各自漂移」。
#      加一个字段到这里却没加到 prompt 里 → 模型不知道该填，语法又逼着它填，
#      结果是它编一个值；反过来则是模型填了、采样器不认。
#
#  ── 为什么它治得了「JSON 壳没闭合」──
#
#     约束解码（GBNF）把采样器的合法下一步【限死】在语法允许的 token 上。
#     想吐 EOS 就必须站在语法的【接受态】—— 而 `{"answer": "...", "used": [1]`
#     这个状态不接受结束。**模型物理上没法停在半路。**
#
#     这和"在 prompt 里再叮嘱一句别忘了括号"是两回事：
#     后者是【请求】，前者是【保证】。S5-6 §2.3 说过模型嘴上答应得很好转头照旧。
#
#  ── 为什么 quotes 用 additionalProperties 而不是写死键名 ──
#     键是【材料编号】，题题不同（"1"/"2"/"3"…）。写死 {"1": ...} 就等于
#     只允许一条引用 —— 而 prompt 里的示例恰好也是写死 "1" 和 "3" 的。
#     ★ 那个示例是【形状示范】，不是"只能有 1 和 3"。schema 必须比它宽。
GEN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used": {"type": "array", "items": {"type": "integer"}},
        "quotes": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["answer", "used", "quotes"],
}


def parse_response(raw: str) -> dict:
    """把模型输出的【文本】解析成 {"answer": str, "used": list[int]}。

    ── 为什么这件小事值得单独一个函数 ──

        模型给你的是文本，不是 JSON 对象。
        哪怕你千叮咛万嘱咐"只输出 JSON"，它仍然可能给你：

            ```json
            {"answer": "...", "used": [1]}
            ```

        或者前面多一句"好的，以下是我的回答："。
        这些【都不报错】—— 只是 json.loads 会炸；
        或者更糟：不炸，但解析出来个空壳。

    参数:
        raw: llm.generate() 返回的原始文本

    返回:
        {"answer": str, "used": list[int], "quotes": dict[str, str]}

        quotes 是 Stage 4 加的：材料编号（字符串）→ 从原文逐字抄来的句子。
        它【不由本函数核对真伪】—— 那要拿到 chunks 才能做，是 cite.verify 的事。
        本函数只保证它的【形状】：一个「数字串 → 字符串」的对象。
        （键必须是数字串，因为下游 cite.verify 要拿它当材料编号做 int(k)。
          契约写松了，下游就得自己再查一遍 —— 那叫两条保证，不叫一条。）

    ── 要处理的失败情况（一个个写）──

        1. raw 里有 markdown 代码块围栏 ```json ... ```
        2. raw 前后有多余空白 / 寒暄
        3. JSON 合法，但缺字段（没有 used，或者 answer 是空的）
        4. 压根不是 JSON（模型跑偏了，输出了一段散文）

    ── 0A-5 判据的答案（这条已定，写在这里备查）──

        问：「处理方式是否取决于调用方？」

        答：**分两种，界线是「你有没有丢东西」。**

            情况 1、2（围栏 / 寒暄）——【剥包装】，兜底
                JSON 外面那层 ``` 和"好的，以下是"是【包装】。
                剥掉它，内容一个字没动。你没丢任何信息，
                也没有任何东西需要你【编】出来。 → 继续走

            情况 3、4（缺字段 / 不是 JSON）——【抛异常】，绝不兜底
                这时候你没有内容可剥，你只有【空缺】。
                想要返回一个 dict，就必须自己【补】上一个值 ——
                而那个值是你造的，不是模型说的。

        ★ 为什么兜底在这里是【最危险】的选择 ——

          假设缺字段时我们兜底成 {"answer": "", "used": []}，
          而模型真正的拒答是 {"answer": "资料未提及", "used": []}。

          看这两个：
              {"answer": "",             "used": []}   ← 解析失败的兜底
              {"answer": "资料未提及",     "used": []}   ← 真·拒答

          【形状一模一样，语义天差地别。】
          调用方拿到第一个，会以为"模型说资料里没有"。
          实际上模型说了些别的，是【代码替它编了那句拒答】。

          往上追，Stage 4 的引用校验会看到一个 used=[] 的答案，
          判定"本次没有引用，跳过校验" —— 一次解析失败
          就这么【伪装成了一次正常的拒答】，全程无人报错。

        → 一句话：**凡是需要你「补」一个值的，就是在编。**
                    报错很吵，但它诚实；兜底很安静，但它在撒谎。

    ── 这个函数管到哪儿为止 ──

        它校验的是【形状】，不是【语义】。

        它能管：answer 是不是非空字符串、used 是不是整数数组
        它不管：used 里的 [99] 是不是真的存在（那需要 chunks，
                而这个函数根本拿不到 chunks）

        语义词法边界要划在能拿到资料的那一层。这里越界去查，
        就得把 chunks 也传进来 —— 然后这个函数就再也不能被
        "只想解析一段文本"的地方复用了。

    ── 已知漏洞（2026-09-25 决定：只记录，不修）──

        find("{") 默认【第一个左花括号就是 JSON 的开头】。
        如果模型在 JSON 之前先输出了一个花括号：

            '答案里有 {1,2}，JSON 是：{"answer": "A", "used": [1]}'
                          ↑ 从这里切 → json.loads 炸

        更坏的是报错信息：它说"不是合法 JSON"，把矛头指向【模型】，
        而模型输出的后半段完全合法 —— 是【我们切错了位置】。
        这种误导比不报错更难查。

        为什么不修：
            1. 从没观察到模型在 JSON 前输出花括号（探针每次打印原始输出）
            2. 这个坑炸得很响 —— ValueError 里带着原文，不会静默出错
            3. 为一个没观察到的失败模式加一段循环 = 替想象写代码

        真出现了怎么修 —— 逐个试每个 { 的位置：
            obj, _ = json.JSONDecoder().raw_decode(text, idx)
        raw_decode 还会顺手解决"JSON 后面还有废话"的情况。

        ★ 恢复条件：探针里连续两次看到这个报错 → 立刻修。

    ── ★★ 2026-10-13（S5-7）：恢复条件触发了，但【修的不是这个洞】──

        端到端评测里 `parse_failed` 出现 4 次，探针里又 4/4 复现。
        按恢复条件本该在这里加 raw_decode —— 探完才发现【用不上】：

          raw_decode 解决的是「我们【切】错了位置」，
          而实测的病是「文本里【缺】字符」——
          模型自己停了（done_reason='stop'），JSON 的壳少关了一个或多个 `}`。
          没有合法 JSON 可解，试遍每个 { 也解不出来。

        ★ 这就是"恢复条件触发 ≠ 按预写的方案修"。
          预写的是一个【针对当时想象的失败模式】的方案，
          而真实失败模式变了。**先探针，再动手** —— 差一步就白写一个解析器，
          而且新解析器会让"壳缺括号"这种病看起来像已经处理过了。

        真正的修法在 GEN_SCHEMA（约束解码，从源头保证语法闭合）。
        本函数【一行没改】—— 它要的本来就是"合法 JSON"，现在拿到的确实是。

    ── 一句反复出现的话 ──

        "第一个 { 就是 JSON 的开头" 和今天第 4 枪的
        "材料里有 BERT 这个词 ⇒ 材料在讲 BERT"，是同一类东西：
        【代码/推理依赖了、但从没写出来验证过的前提。】
    """
    text = raw.strip()

    # ── 1. 砍包装：从第一个 { 到最后一个 } ──
    # 这一个动作同时解决【两种情况】，别写两段代码：
    #     ```json\n{...}\n```       → 围栏在 JSON 外面，砍掉
    #     好的，以下是我的回答：\n{...}  → 寒暄在 JSON 前面，砍掉
    # 围栏和寒暄是同一类东西 —— 【JSON 之外的多余字符】。
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        raise ValueError(f"模型输出里没有 JSON 对象。原文：{raw[:200]!r}")
    payload = text[i : j + 1]

    # ── 2. 解析 ──
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"不是合法 JSON（{e}）。原文：{raw[:200]!r}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"JSON 顶层不是对象：{obj!r}")

    # ── 3. 校验字段 —— 一律抛异常，理由见上面 docstring ──
    answer, used, quotes = obj.get("answer"), obj.get("used"), obj.get("quotes")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"answer 缺失或为空：{obj!r}")

    if not isinstance(used, list) or not all(isinstance(u, int) for u in used):
        raise ValueError(f"used 不是整数数组：{obj!r}")

    # Stage 4 新增。quotes 缺失时【也抛异常】，同样不兜底成 {}。
    # 理由和上面一模一样：兜底成 {} 就等于替模型声明"我没抄任何原句"，
    # 而它其实说了些别的。缺字段是【空缺】，不是【内容】。
    if not isinstance(quotes, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in quotes.items()
    ):
        raise ValueError(f"quotes 不是「字符串 → 字符串」的对象：{obj!r}")

    # ★ 2026-09-29 补：键必须是【数字串】。
    #
    #   原来只查到"是字符串"就停了，而 cite.verify 拿它做 int(k) ——
    #   契约在这里断了一节：上游说"字符串就行"，下游却假定"得是数字"。
    #
    #   为什么这不算越界：本函数管的是【形状】，而"键长什么样"就是形状。
    #   对照上面 used 那行 —— 它检查了【元素】的类型，这里检查【键】的类型，
    #   本来就该是对称的。
    #   真正属于 verify 的是「[99] 这条材料到底存不存在」，那才需要 chunks。
    #
    #   判据用 int() 【本身】，不用 k.isdigit()。实测：
    #       '²'.isdigit() → True，但 int('²') 抛 ValueError    ← 守不住
    #       ' 2'.isdigit() → False，但 int(' 2') == 2         ← 误伤
    #   isdigit() 既拦不住它该拦的，又会误伤它不该拦的。
    #   ★ 守卫的判据和它守的那个操作只要是两套，就会各自漂移 ——
    #     而漂移出来的是一条"看起来在守卫"的假防线，比不写更危险。
    for k in quotes:
        try:
            int(k)
        except ValueError as e:
            raise ValueError(f"quotes 的键不是材料编号（数字）：{k!r}") from e

    return {"answer": answer.strip(), "used": used, "quotes": quotes}


def answer(
    client,
    question: str,
    chunks: list[dict],
    model: str = GEN_MODEL,
    options: dict | None = None,
    schema: bool = GEN_JSON_SCHEMA,
) -> dict:
    """编排：拼 prompt → 调模型 → 解析。胶水，不含逻辑。

    参数:
        client:   ollama 客户端
        question: 用户的问题
        chunks:   top_k() 的结果
        model:    生成模型
        options:  透传给 Ollama 的生成参数
                  （建议 {"temperature": 0} —— 评测的时候你要可复现）
        schema:   要不要用 GEN_SCHEMA 约束解码（默认跟 config.GEN_JSON_SCHEMA）。
                  ★ 做成参数而不是写死，是为了能 A/B ——
                  而 A/B 的时候【必须把它的值记进评测结果】，
                  否则「评测跑的是哪个配置」又会变成一个查不出来的问题
                  （S5-1 / S5-2 / S5-5 已经栽过三次）。

    返回:
        parse_response() 的结果，{"answer": str, "used": list[int]}
    """
    prompt = build_prompt(question, chunks)
    # ★ schema=False 时【明确传 None】，不是"不传"。
    #   Ollama 的 format 默认值就是"不约束"，两者等价 —— 但写成 None 让
    #   "这次没约束"变成代码里看得见的事实，而不是靠"我漏传了一个参数"。
    raw = generate(client, model, prompt, options=options,
                   format=GEN_SCHEMA if schema else None)
    return parse_response(raw)
