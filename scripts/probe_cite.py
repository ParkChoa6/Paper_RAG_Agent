"""探针：Stage 4 引用溯源 —— 这一层能挡什么，挡不住什么。

    ┌─ 离线测试 ── verify 的关卡。构造输入，不调模型。
    │               等模型自然吐出伪造引用，你可能等一百次都等不到。
    │
    ├─ 实验 A ──── 【诱导】模型伪造引用 → 校验抓得住 ✅
    │               手法：把拒答规则拿掉，改成「quotes 不许为空」。
    │               材料里没有答案 + 不许说没有 = 它只能编。
    │
    └─ 实验 B ──── Day 5 的第 4 枪 → 校验抓不住 ❌
                    引用是真的，结论是错的。

    两关并排看，这一层的边界就出来了：
        它查得出【无中生有】，查不出【有中生错】。

用法:
    .venv/Scripts/python.exe scripts/probe_cite.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.cite import verify
from paper_rag.config import EMB_MODEL, PROCESSED_DIR, TOP_K
from paper_rag.embed import embed_one, make_client
from paper_rag.generator import build_prompt, parse_response
from paper_rag.index import load
from paper_rag.llm import generate
from paper_rag.retrieve import top_k_np

LINE = "=" * 74

# ── 离线测试用的假材料：短、可控、每条都有明显特征，方便算准期望 ──
FAKE = [
    {"chunk_id": "t#1", "section": "2.1", "page_start": 2,
     "text": "We marginalize the latent documents with a top-K approximation."},
    {"chunk_id": "t#2", "section": "2.1", "page_start": 2,
     "text": "The retriever provides latent documents conditioned on the input."},
    {"chunk_id": "t#3", "section": "3.2", "page_start": 5,
     "text": "RAG-Token can generate different sequences for each document."},
]

Q1 = "We marginalize the latent documents with a top-K approximation."
Q3 = "RAG-Token can generate different sequences for each document."

# (名字, parsed, 期望, 说明)
#   期望 0    → 必须零问题
#   期望 ">0" → 至少要报出一个问题
#   期望 None → 【观察项】报不报都对，取决于你把容错线画在哪儿
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

    ("观察 · 首字母小写 + 去句号",
     {"answer": "…", "used": [1],
      "quotes": {"1": "we marginalize the latent documents with a top-K approximation"}},
     None, "★ 严格 in 会误报；归一化或模糊匹配会放行。你选了什么，这里就该是什么"),
]

# ── 实验 A 的诱导 prompt：拿掉拒答规则，改成「引用不许为空」──
# 故意【不用】build_prompt —— 这是一个被操纵的条件，要和正常流程分开。
def build_induce_prompt(question: str, chunks: list[dict]) -> str:
    parts = ["请根据以下材料回答问题。"]
    parts.append("")
    parts.append("## 材料")
    parts.append("")
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['text']}")
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


def report(name: str, expect, problems: list[str], note: str, crashed: str | None = None) -> bool:
    # 炸了就是炸了 —— 不管期望是 0 还是 ">0"。
    # （教训：报错如果被当成"报出了 1 条问题"，探测就变成了假绿。）
    if crashed:
        print(f"  ❌ {name:<22} 炸了   {note}")
        print(f"       └─ {crashed}")
        return False

    if expect == 0:
        good = len(problems) == 0
    elif expect == ">0":
        good = len(problems) > 0
    else:  # None —— 观察项，不计分
        good = True
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

    # 用 Day 5 第 4 枪那道题：库里没有 BERT 论文，但材料里飘着 "BERT" 这个词
    question = "BERT 的预训练任务是什么？"
    qv = embed_one(client, EMB_MODEL, question)
    top = [by_id[cid] for cid, _ in top_k_np(qv, vectors, ids, k=TOP_K)]

    # ══════════ 实验 A ══════════
    print()
    print(LINE)
    print("实验 A —— 诱导伪造引用（校验【该】抓得住）")
    print(LINE)
    print(f"问题：{question}")
    print("手法：把 build_prompt 的拒答规则【拿掉】，改成「quotes 不许为空」。")
    print("      材料里没有答案 + 不许说没有 = 它只能编。\n")

    # ★ seed 是 2026-09-29 加的，不是一开始就有。
    #   起因：同一条 prompt、temperature=0，探针那次和手工重跑那次给了【不同】的输出。
    #   实测 4 次连跑又是一致的 —— 也就是说它【基本】可复现，但【不是保证】。
    #   temperature=0 只保证"不随机采样"，不保证底层浮点归约顺序一致
    #   （GPU 上分批、多线程都会影响）。
    #   想真正钉死一次实验结果，就得显式固定 seed。
    raw_a = generate(client, "qwen2:7b", build_induce_prompt(question, top),
                     options={"temperature": 0, "seed": 42})
    print("模型原始输出:")
    print("┄" * 74)
    print(raw_a)
    print("┄" * 74)

    try:
        pa = parse_response(raw_a)
    except ValueError as e:
        print(f"\n  ⛔ 解析失败：{e}")
        return

    print(f"\nused = {pa['used']}")
    for n, q in pa["quotes"].items():
        idx = int(n) - 1
        hit = 0 <= idx < len(top) and q in top[idx]["text"]
        print(f"  quotes[{n}] 逐字在材料 {n} 的原文里？ {'✅ 在' if hit else '❌ 找不到 —— 这是伪造'}")
        if not hit:
            print(f"      模型交的：{q[:100]}")

    probs, status = safe_verify(pa, top)
    if status == "notimpl":
        print("\n  ⏸ verify() 还没实现 —— 上面【肉眼可见】的 ❌ 就是它该抓的东西。")
    elif status == "no-return":
        print("\n  ❌ verify() 返回了 None —— 它没走到任何一个 return。")
    else:
        print(f"\nverify 报出 {len(probs)} 条：")
        for p in probs[:5]:
            print(f"    └─ {p}")
        if probs:
            print("\n  ✅ 校验抓住了 —— 这就是这一层存在的意义。")
        else:
            print("\n  ⚠️ 没抓到。要么它没编引用，要么你的容错线太松。回去看上面那条。")

    # ══════════ 实验 B ══════════
    print()
    print(LINE)
    print("实验 B —— 第 4 枪（校验【抓不住】，这是已知的）")
    print(LINE)
    print("用【正常的】build_prompt 问同一个问题。\n")

    raw_b = generate(client, "qwen2:7b", build_prompt(question, top),
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
    for n, q in pb["quotes"].items():
        idx = int(n) - 1
        hit = 0 <= idx < len(top) and q in top[idx]["text"]
        print(f"  quotes[{n}] 在材料 {n} 的原文里？ {'✅ 在' if hit else '❌ 找不到'}")
        print(f"      抄的是：{q[:90]}")

    probs, status = safe_verify(pb, top)
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
        print(f"     材料 [{n}] 的标题行是「{top[n - 1]['section']}」，来自 {top[n - 1]['paper_id']}。")
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
