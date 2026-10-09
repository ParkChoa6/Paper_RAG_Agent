"""探针：Stage 4 引用溯源 —— 这一层能挡什么，挡不住什么。

    ┌─ 离线测试 ── verify 的关卡。构造输入，不调模型。
    │               等模型自然吐出伪造引用，你可能等一百次都等不到。
    │
    ├─ 实验 A ──── 【诱导】模型不守引用契约 → 校验抓得住 ✅
    │               手法：把拒答规则拿掉，改成「quotes 不许为空」。
    │               材料里没有答案 + 不许说没有 = 它必须交出点什么。
    │               ★ 它交出来的有【三种】，全是失败，但失败的样子不一样：
    │                   ① 编一句英文假话        —— 教科书式的"无中生有"
    │                   ② 抓一句真实但无关的话  —— 其实是实验 B 那一类错
    │                   ③ 把英文材料【翻译】成中文当引用交 ★ 最常见
    │                 ③ 是这一节最值钱的发现：它不是撒谎，是契约没法执行 ——
    │                 answer 用中文写，却要求它交英文原话。
    │                 → 产品 prompt 因此加了「quotes 必须是英文原文」，见
    │                   generator.build_prompt 第 4 条。这里【故意不加】，
    │                   好让 ③ 的样子留得住。
    │               ★ 编不编、怎么编，都【不保证复现】—— seed 1 甚至会让模型
    │                 陷入退化循环。所以跑一轮看比例，跑一次等于抛硬币。
    │
    └─ 实验 B ──── Day 5 的第 4 枪 → 校验抓不住 ❌
                    引用是真的，结论是错的。

    两关并排看，这一层的边界就出来了：
        它查得出【无中生有】，查不出【有中生错】。

用法:
    .venv/Scripts/python.exe scripts/probe_cite.py
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.cite import verify
from paper_rag.config import EMB_MODEL, PROCESSED_DIR, TOP_K
from paper_rag.embed import embed_one, make_client
from paper_rag.generator import build_prompt, parse_response, render_material
from paper_rag.index import load
from paper_rag.llm import generate
from paper_rag.retrieve import top_k_np

LINE = "=" * 74

# 实验 A 跑哪些 seed。为什么不是一个：
#   同一个诱导 prompt，seed 1 让模型陷入退化循环（输出不是 JSON），
#   seed 2 才编出引用。跑一次等于抛硬币 —— 抛到反面就得出"这个模型不编"的
#   错误结论。要下结论就得跑一轮。
#   代价是 N 次模型调用，所以 N 取小；这仍然是演示，不是回归测试 ——
#   真正稳定、可重复的伪造证明在【离线用例 ⭐伪造引用】里，那边不调模型。
INDUCE_SEEDS = (1, 2, 3, 4)

# ── 离线测试用的假材料：短、可控、每条都有明显特征，方便算准期望 ──
#
# ★ 2026-10-10：这里原来【没有 title】，然后 render_material 开始读 title ——
#   六个离线关卡当场全炸 KeyError，计分从 9/9 掉到 3/9。
#
#   修的是这份假材料，不是 render_material：
#     ❌ chunk.get('title', '')  —— 那是"补一个值"，而缺 title 会静默产出
#        一个【没有判别器】的表头，正好把 2026-10-10 刚修掉的那个 bug 放回来。
#     ✅ 假材料必须和真材料同形。它当时是照着旧 schema 手搓的，
#        **它本来就在说谎** —— 离线关卡测的形状和生产跑的形状不是同一个。
#        这种谎平时不响，只在 schema 变更时炸，而炸的时候看起来像"产品坏了"。
#
#   真 chunks.json 的两篇论文各有一个 title，167/170 块全对（已核对）。
FAKE_TITLE = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
FAKE = [
    {"chunk_id": "t#1", "title": FAKE_TITLE, "section": "2.1", "page_start": 2,
     "text": "We marginalize the latent documents with a top-K approximation."},
    {"chunk_id": "t#2", "title": FAKE_TITLE, "section": "2.1", "page_start": 2,
     "text": "The retriever provides latent documents conditioned on the input."},
    {"chunk_id": "t#3", "title": FAKE_TITLE, "section": "3.2", "page_start": 5,
     "text": "RAG-Token can generate different sequences for each document."},
]

Q1 = "We marginalize the latent documents with a top-K approximation."
Q3 = "RAG-Token can generate different sequences for each document."

# (名字, parsed, 期望, 说明)
#   期望 0    → 必须零问题
#   期望 N    → 必须【正好】N 条（用来钉死"同一条只说一次"这类要求）
#   期望 ">0" → 至少要报出一个问题
#   期望 None → 【观察项】报不报都对，取决于你把容错线画在哪儿
#
# ★ 为什么要有"正好 N 条"这一档：光有 0 和 ">0" 的话，
#   「报两遍」和「报一遍」在测试里长得一模一样 —— 都是 >0。
#   数量本身就是判据，就别只测"有没有"。
VERIFY_CASES = [
    ("全部对得上", {"answer": "…", "used": [1, 3], "quotes": {"1": Q1, "3": Q3}},
     0, "两条引用逐字来自材料"),

    ("真·拒答", {"answer": "资料未提及", "used": [], "quotes": {}},
     0, "两边都空 —— 拒答不是错误，别报出来"),

    ("编号越界", {"answer": "…", "used": [9], "quotes": {"9": "随便什么"}},
     ">0", "引用了不存在的第 9 条材料"),

    ("说了用、交不出原句", {"answer": "…", "used": [1, 2], "quotes": {"1": Q1}},
     ">0", "声称用了 2，却没有 2 的原句"),

    ("交了原句、没说过要用", {"answer": "…", "used": [1], "quotes": {"1": Q1, "3": Q3}},
     ">0", "给了 3 的原句，却不承认用了 3"),

    ("★ 伪造引用", {"answer": "…", "used": [1],
                    "quotes": {"1": "RAG was invented by Google in 2019."}},
     ">0", "这句话材料里根本没有 —— 这一关是本层存在的理由"),

    # 两条引用、一真一假。专门照【只检查了第一条就返回】这类 bug。
    # 单独一条真引用、单独一条假引用都会通过 —— 只有放在一起才露馅。
    ("★ 一真一假", {"answer": "…", "used": [1, 3],
                    "quotes": {"1": Q1, "3": "RAG was invented by Google in 2019."}},
     ">0", "第二条是伪造的 —— 只看第一条的实现会漏掉它"),

    # ★ 2026-10-06 补。这条测的是【新接受的东西】，不是【被拒绝的东西】——
    #   上一版把判据从 isdigit 放宽到 int()，只测了 "abc" / "²"（挡住的），
    #   没测 " 1" / "+1"（新放行的）。而假阳性正是从那儿漏出来的：
    #   int(" 1") 能过，但 quotes.get(str(1)) 查不到 → 真引用被报成"原文为空"。
    #   ⚠️ 放宽一个判据时，要测的是【新放进来的】，不是【仍然被挡住的】。
    ("★ 键带前导空格", {"answer": "…", "used": [1], "quotes": {" 1": Q1}},
     0, "键写成 ' 1' —— 编号没错、原句也真，不该报任何问题"),

    # ★ 2026-10-06 补。这条测的是【条数】，不是【有没有】。
    #   原来 ① 写的是 for n in used（used 可能带重复），同一条问题会报两遍。
    #   而"报两遍"在 0 / ">0" 这两档下和"报一遍"长得【一模一样】——
    #   都是 >0。所以这条必须用"正好 N 条"才钉得住。
    #   ⚠️ 数量本身就是判据的时候，就别只测"有没有"。
    ("★ 重复编号只报一次", {"answer": "…", "used": [9, 9], "quotes": {"9": "随便什么"}},
     1, "9 越界，但【越界】这件事只有一件 —— 报两遍是 bug"),

    ("观察 · 首字母小写 + 去句号",
     {"answer": "…", "used": [1],
      "quotes": {"1": "we marginalize the latent documents with a top-K approximation"}},
     None, "★ 严格 in 会误报；归一化或模糊匹配会放行。你选了什么，这里就该是什么"),
]

def probe_norm(s: str) -> str:
    """探针自己的归一化 —— 【独立实现】，故意不 import cite._norm。

    规格抄自 cite._norm 的 docstring（转小写 / 标点换空格 / 压空白），
    代码是自己写的。同规格 + 两份实现 = 对拍；import 过来就变成复读了 ——
    复读的探针只能证明"verify 没崩"，证明不了"verify 判对了"。

    ★ 这也是个警报器：哪天 verify 的 _norm 改了（比如去掉转小写），
      这儿的数字会和 verify 的条数对不上 —— 那正是我们该被叫醒的时刻。
    """
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


# ── 实验 A 的诱导 prompt：拿掉拒答规则，改成「引用不许为空」──
# 故意【不用】build_prompt —— 这是一个被操纵的条件，要和正常流程分开。
#
# ★ 2026-10-09：这里的引用契约【故意停在旧版】（只写"原样抄来的句子"，
#   没写"必须是英文"）。产品 prompt 已经加上英文条款了，这里不加，是有意的：
#   加了之后就【再也复现不出】"把材料翻译成中文当引用交"这个失败模式了（实测：
#   加条款后 3/3 变真引用，不加 2/3 是整段中译）。
#   探针要留住的是【那个病的样子】—— 它同时也是"为什么要有那条英文条款"的证据。
#   ⚠️ 所以这个函数和 build_prompt 的措辞【不同步是故意的】，别顺手去"修"。
def induce_material(number: int, chunk: dict) -> str:
    """实验 A 里模型看到的【一条材料】。

    ★ 和产品路径的 render_material() 是【两份不同的实现】，这是故意的：
      产品路径的表头是「[n] （section，第 N 页）」，诱导条件里只有「[n] 」。
      两份实现 + 同一条原则（拼 prompt 的和核对引用的必须用同一个渲染器），
      才叫对拍。直接 import render_material，这一层就白测了。
    """
    return f"[{number}] {chunk['text']}"


def build_induce_prompt(question: str, chunks: list[dict]) -> str:
    parts = ["请根据以下材料回答问题。"]
    parts.append("")
    parts.append("## 材料")
    parts.append("")
    for i, c in enumerate(chunks, 1):
        parts.append(induce_material(i, c))
        parts.append("")
    parts.append("## 问题")
    parts.append(question)
    parts.append("")
    parts.append("## 要求")
    parts.append("你的回答必须有依据。used 和 quotes 都【不允许为空】——")
    parts.append("如果写不出引用，说明你答得不够具体，请重新组织答案。")
    parts.append("")
    parts.append("## 输出格式")
    parts.append("只输出一个 JSON 对象，不要 markdown 代码块：")
    parts.append('{"answer": "你的回答", "used": [1], "quotes": {"1": "原样抄来的句子"}}')
    return "\n".join(parts)


def safe_verify(parsed: dict, chunks: list[dict]) -> tuple[list[str] | None, str]:
    """verify() 没写 / 没写对时别炸掉整个探针。

    返回 (problems 或 None, 状态)。状态三选一：
        "ok"        正常返回了一个 list（哪怕是空 list）
        "notimpl"   还没实现
        "no-return" 返回了 None —— 函数掉出了末尾，一个 return 都没走到

    ★ 第三种最坏：None 既不等于"通过"，也不等于"不通过"。
      而 `if problems:` 会把 None 当成【假】—— 于是静默通过。
    """
    try:
        problems = verify(parsed, chunks)
    except NotImplementedError:
        return None, "notimpl"
    if problems is None:
        return None, "no-return"
    return problems, "ok"


def check_quotes(parsed: dict, chunks: list[dict], render) -> list[int]:
    """逐条引用做【三层】判断，返回探针独立算出的「伪造编号」清单。

    三层按严格度排，每一层都在回答一个更松的问题：

        ① 逐字在不在正文里               —— 最严。模型改一个字母就挂。
        ② 归一化后在不在正文里           —— 摊平大小写 / 标点 / 换行的差异
        ③ 归一化后在【模型看到的材料】里 —— ② 再加表头

    ★ 为什么必须分三层，而不是一个布尔值：

      2026-10-09 第一次端到端跑，这个探针就是拿 ① 当 oracle 的。结果
      模型把 "one of the tasks..." 的首字母大写成 "One"，① 挂了 ——
      探针当场把一条【真引用】打成「❌ 找不到 —— 这是伪造」，而 verify
      （走的 ②）说没问题。两个判断打架，查了半天才发现是探针在说谎。

      ① 挂、② 过 = 模型穿了个制服（Day 5 学的"会努力模仿格式的对手"），
      那【不是】伪造。假阳性的代价比假阴性高 —— 它会训练人无视警报
      （config.py 的「报警疲劳」是同一个病）。

    参数:
        render: (编号, chunk) -> 模型实际看到的那段文本。
                实验 A 传 induce_material，实验 B 传 render_material。
    """
    forged: list[int] = []
    keys = sorted(parsed["quotes"], key=int)

    for k in keys:
        n = int(k)
        q = parsed["quotes"][k]
        idx = n - 1

        if not (0 <= idx < len(chunks)):
            print(f"  quotes[{n}] ⛔ 越界：只有 {len(chunks)} 条材料")
            forged.append(n)
            continue

        c = chunks[idx]
        at1 = q in c["text"]                                    # ① 逐字
        at2 = probe_norm(q) in probe_norm(c["text"])            # ② 归一化
        at3 = probe_norm(q) in probe_norm(render(n, c))         # ③ 含表头

        if at1:
            mark, why = "✅", "逐字就在正文里"
        elif at2:
            mark, why = "◐", "只是大小写/标点差异 —— 是【真】引用，不是伪造"
        elif at3:
            mark, why = "◐", "抄的时候把表头一起带上了 —— 是【真】引用"
        else:
            mark, why = "❌", "三层都找不到 —— 这是伪造"
            forged.append(n)

        trio = f"①{'在' if at1 else '否'} ②{'在' if at2 else '否'} ③{'在' if at3 else '否'}"
        print(f"  quotes[{n}] {mark} {trio}   {why}")
        if mark == "❌":
            print(f"      模型交的：{q[:100]}")

    return forged


def report(name: str, expect, problems: list[str], note: str, crashed: str | None = None) -> bool:
    # 炸了就是炸了 —— 不管期望是 0 还是 ">0"。
    # （教训：报错如果被当成"报出了 1 条问题"，探测就变成了假绿。）
    if crashed:
        print(f"  ❌ {name:<22} 炸了   {note}")
        print(f"       └─ {crashed}")
        return False

    if expect is None:            # 观察项，不计分
        good = True
    elif isinstance(expect, int):  # 精确条数（0 也走这里）
        good = len(problems) == expect
    else:                          # ">0"
        good = len(problems) > 0
    mark = "✅" if good else ("👁" if expect is None else "❌")
    print(f"  {mark} {name:<22} 报出 {len(problems)} 条   {note}")
    for p in problems[:3]:
        print(f"       └─ {p}")
    return good


def run_offline() -> None:
    print(LINE)
    # 原来这里写的是"第 1~6 关"，而用例早就加到 8 条了 ——
    # 写死的编号就是会过期。去掉编号，让它没法过时。
    print("verify 离线测试（构造输入，不调模型）")
    print(LINE)

    scored = total = 0
    for name, parsed, expect, note in VERIFY_CASES:
        crashed = None
        try:
            problems = verify(parsed, FAKE)
        except NotImplementedError as e:
            print(f"\n  ⛔ verify() 还没实现：\n     {e}\n")
            return
        except Exception as e:  # noqa: BLE001 —— 探针要看它炸成什么样
            crashed, problems = f"[{type(e).__name__}] {e}", []

        # 返回 None = 函数【掉出了末尾】，一个 return 都没走到。
        # 这不是"没问题"，是"没说"。
        if problems is None:
            crashed = "[verify 返回了 None —— 一个 return 都没执行到]"
            problems = []

        if expect is None:
            # ★ 观察项【也要打出来】，只是不计分。
            #   原来这里 if expect is not None: 把观察项整个跳过了 ——
            #   report() 里那个 👁 分支从来没被调用过。
            #   最讽刺的是：这一条是"归一化 vs 严格 in"那个决定的唯一证据存档，
            #   而它一次都没显示过。自称观察项，却从不观察。
            report(name, expect, problems, note, crashed)
        else:
            total += 1
            scored += report(name, expect, problems, note, crashed)
    print()
    print(f"  {scored}/{total} 通过（观察项不计分）")


def run_experiments() -> None:
    client = make_client()
    vectors, ids = load(model=EMB_MODEL)
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}

    def retrieve(q: str) -> list[dict]:
        qv = embed_one(client, EMB_MODEL, q)
        return [by_id[cid] for cid, _ in top_k_np(qv, vectors, ids, k=TOP_K)]

    # ★ 2026-10-09：两个实验用的题【必须分开】，原来共用一道，结果两个都不成立。
    #
    #   实验 A 需要「材料里【没有】答案」，否则诱导不出伪造。挑题是量出来的，
    #   不是拍脑袋 —— 中间试错了两轮，失败原因还不一样：
    #
    #     ①「BERT 的预训练任务是什么？」（原题）
    #        注释写着"库里没有 BERT 论文"，但 top-1（kg-embedding-overview#159，§5.2）
    #        里【恰好有一句】"Since one of the tasks of BERT pretraining is
    #        next-sentence prediction"。材料里有答案，模型照抄 —— 前提没触发。
    #
    #     ②「BERT-base 模型一共有多少参数？」
    #        全库 337 条【一条都查不到】这个数字，材料长得像答案
    #        （top-1 正在说 "PLMs are ... with many model parameters"）。
    #        但模型【宁可违反指令也不编】：5 个 seed 全答「材料中未明确给出」，
    #        再贴一句真话充数。明令「严禁出现未明确给出」之后，它还是这么说。
    #
    #   ★ ③ 才是对的形态：模型一提问就【知道】答案，而库里 0 条命中。
    #     脑子里有答案 + 材料里没有 = 凭记忆答，再配一条假引用。
    #     这就是 RAG 幻觉的教科书形态，也是这两道扫描题共有的结构。
    #
    #   ⚠️ 但「编不编」本身【不可靠】：同一个 prompt + 同一个 seed，
    #      半小时前编了，半小时后不编。实测同一进程内连跑 6 次逐字一致，
    #      跨时间就不一致 —— 起因没查明（疑似 Ollama 加载/缓存状态）。
    #      ★ 结论：不能把"诱导出伪造"当成一个稳定现象去依赖。
    #        所以这里是【扫描 + 报比例】，不是"断言一定会编"。
    #        真正稳定、可重复的伪造证明在【离线用例 ⭐伪造引用】里，那边不调模型。
    #
    #   实验 B 需要「材料里【有】答案，且引用是真的」——那是 Day 5 第 4 枪：
    #     真的引文 + 错的结论。它用的是【正常的】build_prompt（带拒答规则），
    #     换到 A 这类没有答案的题上，模型会直接拒答，实验 B 就空了。
    INDUCE = [
        ("Attention Is All You Need 这篇论文的作者是谁？", "Transformer 作者，全库 0 命中"),
        ("BERT 是哪一年提出的？", "全库无 BERT+年份同现"),
    ]
    question_b = "BERT 的预训练任务是什么？"
    top_b = retrieve(question_b)        # 实验 B 用

    # ══════════ 实验 A ══════════
    print()
    print(LINE)
    print("实验 A —— 诱导伪造引用（校验【该】抓得住）")
    print(LINE)
    print("手法：把 build_prompt 的拒答规则【拿掉】，改成「quotes 不许为空」。")
    print(f"每题都是「模型知道答案、库里查不到」。{len(INDUCE)} 题 × {len(INDUCE_SEEDS)} seed = {len(INDUCE) * len(INDUCE_SEEDS)} 次。")
    print("★ 编不编【不保证复现】，所以看比例，不看单次：")
    print("  跑一次没编 ≈ 抛硬币抛到反面，据此说『模型不编』是错的结论。\n")

    # ★ seed 是 2026-09-29 加的，不是一开始就有。
    #   起因：同一条 prompt、temperature=0，探针那次和手工重跑那次给了【不同】的输出。
    #   temperature=0 只保证"不随机采样"，不保证底层浮点归约顺序一致
    #   （GPU 上分批、多线程都会影响）。
    #
    # ★ seed 是 2026-09-29 加的，不是一开始就有。
    #   起因：同一条 prompt、temperature=0，探针那次和手工重跑那次给了【不同】的输出。
    #   temperature=0 只保证"不随机采样"，不保证底层浮点归约顺序一致。
    #
    # ★ 2026-10-09 改：从「一道题一个 seed」改成「多道题 × 多个 seed 扫描」。
    #   实测出来的，不是想出来的：同一个诱导 prompt，
    #     seed 1 → 模型陷入退化循环（一个词重复几十遍）→ 输出不是 JSON
    #     seed 2 → 编了引用，verify 抓住了   ← 实验 A 想要的那一次
    #   只跑一次的话，这个实验有一半时间是"什么都没测到"。
    #   而且"没编"这个结论本身要靠次数才有意义 —— 跑 1 次没编 ≈ 没信息。
    stats = {"forged": 0, "clean": 0, "parse_failed": 0, "mismatch": 0, "runs": 0}

    for question_a, why in INDUCE:
        top = retrieve(question_a)
        print(LINE)
        print(f"问题：{question_a}   （{why}）")

        for seed in INDUCE_SEEDS:
            stats["runs"] += 1
            print(f"── seed = {seed}")
            raw_a = generate(client, "qwen2:7b", build_induce_prompt(question_a, top),
                             options={"temperature": 0, "seed": seed})
            print("模型原始输出:")
            print("┄" * 74)
            print(raw_a)
            print("┄" * 74)

            try:
                pa = parse_response(raw_a)
            except ValueError as e:
                stats["parse_failed"] += 1
                print(f"\n  ⛔ 解析失败（这本身也是一种失败模式，不是「跳过」）：{e}\n")
                continue

            print(f"\nused = {pa['used']}")
            forged = check_quotes(pa, top, induce_material)

            probs, status = safe_verify(pa, top)
            if status == "notimpl":
                print("\n  ⏸ verify() 还没实现 —— 上面【肉眼可见】的 ❌ 就是它该抓的东西。")
                return
            if status == "no-return":
                print("\n  ❌ verify() 返回了 None —— 它没走到任何一个 return。")
                return

            print(f"\nverify 报出 {len(probs)} 条：")
            for p in probs[:5]:
                print(f"    └─ {p}")

            # ★ 判据从「verify 报了没有」换成「verify 报的伪造条数，和探针独立数出来的
            #   一不一样」。理由：原来那个判据把两件事混成了一件 ——
            #   「模型这次没编」和「verify 漏抓了」在它眼里都是"没抓到"，
            #   而这两件事的处理办法完全相反（一个收工，一个去修 verify）。
            v_forged = [p for p in probs if "疑似伪造" in p]

            if len(v_forged) != len(forged):
                stats["mismatch"] += 1
                print(f"\n  ❌ 探针独立数出 {len(forged)} 条伪造，verify 报出 {len(v_forged)} 条 —— 两边对不上。")
            elif forged:
                stats["forged"] += 1
                print(f"\n  ✅ 探针独立数出 {len(forged)} 条伪造，verify 报了 {len(v_forged)} 条 —— 两边一致。")
            else:
                stats["clean"] += 1
                print("\n  ○ 这一次模型【没编引用】—— 探针三层判据也只数出 0 条伪造。")
            print()

    # ── 汇总：一次说明不了任何事，一轮才有 ──
    n = stats["runs"]
    print(LINE)
    print(f"实验 A 汇总（{len(INDUCE)} 题 × {len(INDUCE_SEEDS)} seed = {n} 次）")
    print(f"  编造被抓   {stats['forged']:>2} 次")
    print(f"  没编       {stats['clean']:>2} 次   ← 模型宁可抓一句无关的真话充数，也不编")
    print(f"  解析失败   {stats['parse_failed']:>2} 次   ← 退化循环，模型输出不是 JSON")
    print(f"  两边对不上 {stats['mismatch']:>2} 次   ← 这一项必须为 0，否则 verify 有问题")
    if stats["mismatch"]:
        print("\n  ❌ 探针和 verify 判断不一致 —— 先查这个，别管别的。")
    elif stats["forged"] == 0:
        print("\n  ⚠️ 这一轮一次都没编。实验 A 的诱导没被触发，")
        print("     说明它这次【没测到】它想测的东西 —— 没测到 ≠ 通过。")
        print("     （编不编跨时间不可复现，见上面那段注释。要稳定的证明去看离线用例。）")
    else:
        print(f"\n  ★ 诱导有效：{stats['forged']}/{n} 次真的编了，verify 每次都抓住。")

    # ★ 这段注释是 2026-10-09 写下的【结论】，不是猜测：
    #   拿到「材料里没有答案」这个条件之后，qwen2:7b 的第一反应【不是】编造，
    #   而是抓一句材料里真实、但跟问题无关的句子来满足「quotes 不许为空」。
    #   它甚至会直说「材料中未明确给出」，同时贴一条真引用。
    #   ★ 所以「无中生有」和「有中生错」不是模型的两个档位 ——
    #     是同一个行为在不同压力下的两张脸。实验 A 和实验 B 是同一个病。

    # ══════════ 实验 B ══════════
    print()
    print(LINE)
    print("实验 B —— 第 4 枪（校验【抓不住】，这是已知的）")
    print(LINE)
    print(f"用【正常的】build_prompt 问：{question_b}")
    print("（和实验 A 不是同一道题 —— 这道题的材料里【有】答案，引用才可能为真。）\n")

    raw_b = generate(client, "qwen2:7b", build_prompt(question_b, top_b),
                     options={"temperature": 0, "seed": 42})
    print("模型原始输出:")
    print("┄" * 74)
    print(raw_b)
    print("┄" * 74)

    try:
        pb = parse_response(raw_b)
    except ValueError as e:
        print(f"\n  ⛔ 解析失败：{e}")
        return

    print(f"\nused = {pb['used']}")
    # 实验 B 的基线用【产品路径的】render_material —— 就是模型真正看到的那段。
    # （实验 A 用的是诱导条件自己的 induce_material，见那一节。）
    check_quotes(pb, top_b, render_material)

    probs, status = safe_verify(pb, top_b)
    if status == "no-return":
        print("\n  ❌ verify() 返回了 None —— 它没走到任何一个 return。")
    elif status == "ok":
        print(f"\nverify 报出 {len(probs)} 条：")
        for p in probs[:5]:
            print(f"    └─ {p}")
        print()
        if probs:
            print("  ⚠️ 报了问题 —— 但先看它报的是【哪一类】：")
            print("     是「引用对不上原文」，还是「答案说错了」？")
            print("     这一层的输出里根本没有「答案对不对」这一项 —— 它答不了这个问题。")
        else:
            print("  ★ 校验【全部放行】。而答案是错的。")
            print("     这不是 bug：这一层查的是引用真不真，不是答案对不对。")
            print("     「引用逐字为真 + 结论错误」在它的判据里【结构上就看不见】。")

    # ── 下面这段的每一条都【从数据里取】，不写死字符串 ──
    #   上一版这里写死了当时的输出文本，模型换一次输出它就过期了。
    #   探针的结论如果靠手抄，它迟早会开始撒谎 ——
    #   和 generator.py 里那条过期注释是同一个病。
    cited = [int(n) for n in pb["quotes"] if n.isdigit()]
    if cited:
        n = cited[0]
        print()
        print("  ── 这一步代码判不了，要你自己判 ──")
        print(f"     材料 [{n}] 的标题行是「{top_b[n - 1]['section']}」，来自 {top_b[n - 1]['paper_id']}。")
        print(f"     模型说的：{pb['answer'][:70]}")
        print("     对照一下：这段材料讲的是谁的事？模型把它安到了谁头上？")
        print()
        print("     ★ 别想着在这一层加规则去拦它 —— 要拦就得让代码读懂论文，")
        print("       那是在写一个笨拙的 BERT。这一层的边界到此为止。")
        print("       剩下两条路：让检索把【对的】材料捞上来，或者换更强的模型。")


def main() -> None:
    run_offline()
    run_experiments()
    print()
    print(LINE)
    print("结论：这一层查得出【无中生有】，查不出【有中生错】。")
    print(LINE)


if __name__ == "__main__":
    main()
