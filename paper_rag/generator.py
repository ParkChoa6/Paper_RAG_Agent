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

from paper_rag.config import GEN_MODEL
from paper_rag.llm import generate

logger = logging.getLogger(__name__)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """把「问题 + 检索到的材料」拼成一段给模型的 prompt。

    ══════════ 这是 Stage 3 的核心 ══════════

    输入:
        question: 用户的问题，例如
                  "RAG 是怎么把检索到的文档融进生成过程的？"
        chunks:   top_k() 的结果，每条是一个 dict（来自 chunks.json）：
                    {
                      "chunk_id":   "rag-2005.11401#8",
                      "text":       "……正文……",
                      "section":    "3.2 Models",
                      "page_start": 4,
                      "page_end":   4,
                      "paper_id":   "rag-2005.11401",
                    }

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
        parts.append(f"[{i}] （{c['section']}，第 {c['page_start']} 页）")
        parts.append(c["text"])
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
    #   校验用精确 in 比对，那 prompt 就必须明说"逐字照抄"。
    #   校验如果容错（归一化、模糊匹配），这句就可以放松。
    #
    #   两者要一起改。校验收紧了而 prompt 没说 → 真引用被误报成伪造。
    #   这就是「约束的严格度和它宣告的严格度必须一致」——
    #   你没有告诉它规则，却拿规则罚它，那不叫校验，叫钓鱼。
    parts.append(
        "4. quotes 里填你【逐字照抄】的原文句子 —— 从上面材料里原样复制，"
        "一个字都不要改：标点、大小写、空格全部保持原样。"
        "代码会拿它去原文里做精确比对，改一个字符就会被判定为伪造引用。"
        "键是材料编号（写成字符串），值是你抄下来的那句话。"
        "如果没有用到任何材料（拒答），quotes 写成空对象 {}。"
    )
    parts.append("")

    # ── 5. 输出格式 ──
    # 给一个【字面示例】让模型抄，比用文字描述格式有效得多。
    # "不要 markdown 代码块" 这句要说 —— Day 0 验过这个模型能做到。
    parts.append("## 输出格式")
    parts.append("只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字：")
    parts.append(
        '{"answer": "你的回答", "used": [1, 3], '
        '"quotes": {"1": "从材料1原样抄来的那句话", "3": "从材料3原样抄来的那句话"}}'
    )

    return "\n".join(parts)


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
) -> dict:
    """编排：拼 prompt → 调模型 → 解析。胶水，不含逻辑。

    参数:
        client:   ollama 客户端
        question: 用户的问题
        chunks:   top_k() 的结果
        model:    生成模型
        options:  透传给 Ollama 的生成参数
                  （建议 {"temperature": 0} —— 评测的时候你要可复现）

    返回:
        parse_response() 的结果，{"answer": str, "used": list[int]}
    """
    prompt = build_prompt(question, chunks)
    raw = generate(client, model, prompt, options=options)
    return parse_response(raw)
