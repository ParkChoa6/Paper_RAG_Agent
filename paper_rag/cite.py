"""引用溯源 —— 模型说的每一句，代码去原文里核。

── 流水线里的位置 ──

    question ──┐
               ├─▶ build_prompt ─▶ llm.generate ─▶ parse_response ─┐
    chunks ────┘                                                     │
                              ┌──────────────────────────────────────┘
                              ▼
                           verify ──┬─▶ []       全部引用经得起核对
                                    └─▶ ["..."]  有问题，逐条列出

    上游：parse_response 已经保证【形状】合法（answer 非空、used 是整数数组）
    本层：只管【内容】—— 那些引用，在原文里真的存在吗

── 为什么单独一个模块（0B-2 判据一）──

    generator.py   会因为「怎么问模型」变化
    cite.py        会因为「怎么核对引用」变化

    两者变化的原因不同 → 两个模块。

── ★ 这一层的定位 —— 它不判断答案对不对，只判断引用真不真 ──

    这是今天两个实验的全部差别：

        实验 A  模型【编造】了一条原文里没有的引用   → 这一层抓得住
        实验 B  模型引用的原文【真实存在】，           → 这一层抓不住
                但它把这段安在了错误的地方

    换句话说：这一层是「有罪推定里的物证鉴定」——
    它能证明"这句话有出处"，不能证明"这个出处该被用在这里"。

    ⚠️ 别把这一层当成幻觉的解药。它是过滤器，不是答案。
       它能挡掉一部分错误，剩下的必须靠别的办法（见 Day 5 第 4 枪）。
"""

import logging
import re

# ★ 2026-10-09：这一行 import 是【故意的】，不是随手加的。
#
#   直觉上，校验层 import 生成层是个坏味道 —— 校验应该独立于被校验的东西。
#   但这里反过来：verify 要回答的问题是「这句话是不是从【模型看到的材料】里抄的」，
#   所以它【必须】知道模型看到的是什么。而"模型看到的材料长什么样"这件事，
#   只有 render_material() 知道。
#
#   不 import 会怎样：自己在这儿再拼一遍。两份拼法迟早漂移，
#   而漂移的表现是「真引用被报成伪造」—— 一个没人会往 import 上查的 bug。
#   2026-10-09 第一次端到端跑，撞的就是这个 bug 的前身。
#
#   校验层和生成层在这里是【一对】，不是两层。build_prompt 里也写着同一句话。
from paper_rag.generator import render_material

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    """把文本压成【只剩词】的形式，用于容错比对。

    做三件事，仅此三件：
        1. 全部转小写          —— "We" 和 "we" 视为同一个词
        2. 所有标点换成空格    —— "top-K." 和 "top-K。" 都变成 "top K"
        3. 连续空白压成一个    —— 换行、多空格、制表符不再影响比对

    没做的事（这些是【判断】，留给你）：
        - 不管数字写法（"5" vs "five"）
        - 不管同义词、改写、语序
        - 不做中文分词

    ★ 用不用它、还要不要更松 —— 那是你的决定，不是这段代码的。
      如果你最后选了别的方案，把它删掉，别留着当摆设。
    """
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def verify(parsed: dict, chunks: list[dict]) -> list[str]:
    """核对模型交出的引用，返回【问题清单】。空列表 = 全部经得起核对。

    ══════════ 这是 Stage 4 的核心，也是你今天的练习 ══════════

    参数:
        parsed: parse_response() 的返回值，形状已保证合法：
                  {
                    "answer": "……回答正文……",
                    "used":   [1, 3],          # 1-based，指向 chunks
                    "quotes": {"1": "原文里的原句", "3": "原文里的原句"},
                  }

        chunks: 【当初喂给模型的那一批材料】，顺序和 prompt 里 [1] [2] 一致。
                就是 build_prompt(question, chunks) 收到的那同一个列表。

                这不是「全部语料」，是「这一次检索出来的 top-k」——
                因为 used 里的编号是 prompt 局部的，不是 chunk_id。

    返回:
        list[str] —— 每条是一个【人能看懂的问题描述】，空列表代表通过。

    ── 为什么返回 list 而不是 bool ──

        bool 只告诉你"有问题"，不告诉你"哪一条、什么问题"。
        调试一个 RAG 系统时，你需要的恰恰是后者 ——
        是模型编了引用？还是它漏交了？还是编号越界了？
        三种病的药不一样。

    ── 要核对的四件事（这是规格，不是步骤）──

        ① 编号越界
           used 里的数字必须在 1..len(chunks) 之间。
           越界 = 模型引用了一条【根本不存在的材料】。

        ② 说了用、但交不出原句
           used 里有 n，quotes 里没有 n。
           = 它声称用了第 n 条材料，却拿不出那句话。

        ③ 交了原句、但没说过要用
           quotes 里有 n，used 里没有 n。
           = 自相矛盾：它给了第 n 条的原句，却不承认用了第 n 条。

        ④ ★ 伪造引用
           quotes[n] 在 chunks[n-1]["text"] 里【找不到】。
           = 这句话不是从材料里抄的，是它编的。
           【这一条是整层存在的理由。】

        另：used 和 quotes 都是空的时候（拒答），必须是【零个问题】。
            拒答不是错误，别把它当成错误报出来。

    ── ★★ 反问二：容错线画在哪儿（今天真正的设计决定）──

        第 ④ 条怎么判「找得到」？原文是：

            We marginalize the latent documents with a top-K approximation.

        模型交出来的是：

            we marginalize the latent documents with a top-K approximation

        （首字母小写了，句号没了）

        这算真引用，还是伪造？三个选项，各有各的代价：

            方案一 · 精确 in
                代价：模型换个标点、多个空格，真引用就被误报成伪造
                      → 假阳性。用户天天看假警报，然后学会无视它。

            方案二 · 归一化后 in（去空白 / 统一标点 / 大小写）
                代价：你得定义"归一化到什么程度"。收得越松，
                      伪造者越容易蒙混过关 → 假阴性。

            方案三 · 模糊匹配（difflib.SequenceMatcher 比阈值）
                代价：阈值是个魔法数字，且【没有明确边界】——
                      "90% 像"到底算不算引用？你答不上来。

        ★ 没有免费的选择。选哪个、为什么，写进代码注释里。

          提示：想清楚【你在防谁】。
                这条校验的对手是一个**会努力模仿格式**的模型（Day 5 学的"穿制服"），
                不是一个只会精确复制的程序。

    ── 自查 ──

        - chunks 是空列表时会怎样？（提示：build_prompt 那边也该有对应行为）
        - quotes 里的值是空字符串 "" 时，上面的判据会给什么结果？合理吗？
        - 你的实现在正常流程上跑一次要多久？它是每个请求都跑的代码。
    """

    problems: list[str] = []

    # ★ 2026-09-29 改：原来是 parsed.get("used", [])，删掉了默认值。
    #
    #   理由：parse_response() 已经保证这两个字段【必然存在】（缺了就抛异常，
    #   见它的 docstring「凡是需要你补一个值的，就是在编」）。
    #   所以默认值永远走不到 —— 除非有人绕过 parse_response 直接调本函数。
    #
    #   真走到那一步会怎样：缺 used → 默认成 [] → 下面"拒答"分支直接放行。
    #   ★ 这是最坏的一种兜底：把【调用方的 bug】伪装成【一次正常的拒答】。
    #     和 parse_response 里那条是同一个判断 ——
    #     凡是需要你"补"一个值的，补出来的那个值就是你在替别人说话。
    #
    #   现在传错东西会 KeyError —— 响，而且直接指向出错的那一行。
    #   那是调用方的程序错误，本来就【不该】变成问题清单里的一条：
    #   问题清单说的是"模型哪里说错了"，不是"你哪里传错了"。
    #   两者混在一起，你会开始去修模型，而 bug 在调用方。
    used: list[int] = parsed["used"]
    quotes: dict[str, str] = parsed["quotes"]

    # 拒答：used 和 quotes 都为空，直接放行
    if not used and not quotes:
        return problems

    used_set = set(used)

    # ★ 2026-10-06 改：把「键 → 编号」的推导【只做一次】。
    #
    #   原来这里是两处各推一次：
    #       quoted_set = {int(k) for k in quotes if k}     ← 用 int(k)
    #       quote_text = quotes.get(str(n), "")            ← 用 str(n)   （在 ④ 里）
    #
    #   键写成 " 1"（前面带个空格）时：int(" 1") == 1 能过，
    #   但 str(1) == "1" ≠ " 1" —— 查不到 → 真引用被报成「提供的原文为空」。
    #   ★ 同一个东西推导两次，两条路径就会各自漂移。
    #     这跟 parse_response 里 isdigit / int 那次是同一个病，只是升了一层。
    #
    #   现在建一张 {编号: 原句} 表，下面 ②③④ 全用它。
    #
    #   依赖：键必须能 int()。这是【上游 parse_response 的契约】
    #   （它已经保证"键是数字串"，见那边的注释）。传错东西这里会 ValueError ——
    #   和上面的 parsed["used"] 一样，是调用方的程序错误，本来就该响。
    quotes_by_n: dict[int, str] = {int(k): v for k, v in quotes.items()}
    quoted_set = set(quotes_by_n)
    # 注：原来那个 `if k` 是死的 —— 空键 int("") 会在 parse_response 就抛错，到不了这里。

    # ★ 2026-10-06：下面四处迭代全部改成 sorted(...)。两个不同的理由，分开说。
    #
    #   ── ① 去重：这是【行为 bug】，量得出来 ──
    #
    #   ① 原来写的是 for n in used，而 ②③ 用的是 used_set。
    #   used 里重复的编号（模型完全可能吐 [9, 9]）会把同一条问题报【两遍】：
    #       实测 used=[9,9] → 改前 2 条，改后 1 条。
    #   同一条问题说一次就够了。②③④ 本来就用的 set，所以只有 ① 有这毛病 ——
    #   ★ 同一个函数里四个循环、两种写法，这本身就是 bug 的温床。
    #
    #   ── ② 定序：这不是 bug，是【别靠实现细节】 ──
    #
    #   先说清楚不是什么：它【不会 flaky】。int 的哈希不受 PYTHONHASHSEED 影响
    #   （随机化的是 str/bytes），所以同一份输入，每次运行给的顺序【一样】。
    #
    #   真正的问题是：那个"一样的顺序"【不是升序】，而是
    #       槽位 = hash(n) % 表长
    #   的先后。实测：
    #       set([3, 11])        → [11, 3]
    #       set([1, 2, 33, 40]) → [40, 1, 2, 33]
    #   看着是乱的。而我们今天看不出乱，只是因为 top-k 就几条、编号都小于表长 8，
    #   槽位恰好就是 n 自己 —— ★ 那是【碰巧】，不是保证。
    #   表长（8/32/128…）是 CPython 的实现细节，语言不保证 set 的迭代顺序。
    #
    #   ⚠️ 我第一版注释在这里写的是"迭代 set 会 flaky"—— 没量就下的结论，是错的。
    #      量完才发现：int 稳定，只是乱序。**量过的和没量过的，是两个不同的结论。**
    #
    #   sorted 之后：同一份输入 → 永远同一份清单，逐字一致，且人能读。
    #   在这么便宜的确定性和可读性面前，"省一次 O(k log k)" 不值得谈。

    # ① 编号越界：used 里的数字必须在 1..len(chunks) 之间
    n_chunks = len(chunks)
    for n in sorted(used_set):
        if n < 1 or n > n_chunks:
            problems.append(f"引用 [{n}] 越界：当前只有 {n_chunks} 条材料")

    # ② 说了用、但交不出原句：used 里有 n，quotes 里没有 n
    for n in sorted(used_set):
        if n not in quoted_set:
            problems.append(f"引用 [{n}]：声称使用但未提供原文")

    # ③ 交了原句、但没说过要用：quotes 里有 n，used 里没有 n
    for n in sorted(quoted_set):
        if n not in used_set:
            problems.append(f"引用 [{n}]：提供了原文但未声明使用")

    # ④ 伪造引用：quotes[n] 在 chunks[n-1]["text"] 里找不到

    for n in sorted(used_set & quoted_set):
        # 越界的编号已经在 ① 里报过了，这里跳过，避免索引错误
        if n < 1 or n > n_chunks:
            continue

        chunk_text = render_material(n, chunks[n - 1])
        # ★ 2026-10-06：原来这里是 quotes.get(str(n), "") —— 换掉。
        #   n 一定在 quoted_set 里，而 quoted_set 就是 quotes_by_n 的键集，
        #   所以这里【必然取得到】，给默认值又是在留后门。（同上，"补一个值就是在编"）
        #   而且 str(n) 和上面建表用的 int(k) 是两条推导路径 —— 就是 bug 的来源。
        quote_text = quotes_by_n[n]

        # 空引用：quotes 里给了空字符串，视为"交不出原句"
        if not quote_text or not quote_text.strip():
            problems.append(f"引用 [{n}]：提供的原文为空")
            continue

        # 归一化后做子串匹配。
        #
        # ★ 2026-10-09：比对的【基线换了】。
        #   原来是 chunks[n-1]["text"]（只有正文），现在是 render_material()
        #   （表头 + 正文，也就是模型实际看到的东西）。
        #
        #   为什么换：模型把表头 "[1] " 一起抄进 quotes 时，正文里当然找不到，
        #   一条真引用被报成伪造。实测第一次端到端跑就撞上。
        #
        #   为什么不是"把 quotes 开头的 [n] 剥掉"：
        #   那是补偿我们自己制造的偏差；表头一改（比如加论文标题），
        #   剥离函数就静默失效。换成"和模型看到的一致"之后，
        #   表头怎么改都自动跟上 —— 因为两边调的是同一个 render_material。
        norm_chunk = _norm(chunk_text)
        norm_quote = _norm(quote_text)

        if norm_quote not in norm_chunk:
            problems.append(f"引用 [{n}]：原文中找不到对应内容（疑似伪造）")

    return problems
