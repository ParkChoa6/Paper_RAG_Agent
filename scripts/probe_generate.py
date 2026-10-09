"""探针：Stage 3 的验收工具。

四道题，每一道打一枪：

  1. 能答的        材料里有，看它答不答得出、格式对不对
  2. 材料里没有    看它会不会老实拒答
  3. 世界知识陷阱  看它会不会「先拒答再编」（用世界知识补材料里没有的）
  4. 换个论文      看它会不会「张冠李戴」（材料是真的，结论是错的）

【v0 的战绩】（2026-09-25 跑过，留作对照）
    第 1 枪  内容全对，但输出是一段散文 —— used = []，下游什么都做不了
    第 2 枪  拒答了 ✓  ← 但那是运气：问题太荒谬，它自己觉得离谱
    第 3 枪  先写「提供的信息中并未直接提及」，紧接着答出 12 层 / 24 层
             —— 材料里一个字都没有，全是世界知识
    第 4 枪  把 RAG 论文的训练目标（含公式）说成「BERT 的预训练任务」
             —— 引用内容全是真的，字符串校验会全部放行

    → 「拒答」被模型理解成了一种【语气】，不是一个【行为】。

现在跑的是 v1（generator.py）。重点看两件事：
    a. 输出是不是干净的 JSON（parse_response 能不能吃）
    b. 第 3、4 枪有没有被拦住

用法:
    python scripts/probe_generate.py
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.config import EMB_MODEL, PROCESSED_DIR, TOP_K
from paper_rag.embed import embed_one, make_client
from paper_rag.generator import answer, build_prompt, parse_response
from paper_rag.index import load
from paper_rag.retrieve import top_k_np

LINE = "=" * 74

# 四道题，每一道都在打不同的一枪
CASES = [
    ("能答的", "RAG 是怎么把检索到的文档融进生成过程的？"),
    ("材料里没有", "论文作者中午吃了什么？"),
    ("世界知识陷阱", "BERT 模型有多少层？"),
    ("换个论文", "BERT 的预训练任务是什么？"),
]

# parse_response 的验收：喂【构造好的字符串】，不调模型。
# 每条是 (名字, 原始文本, 期望, 期望值)
#   期望 "ok"    → 必须返回期望值那个 dict
#   期望 "raise" → 必须抛 ValueError
#
# 为什么离线测：模型什么时候吐出围栏是【它说了算】的。
# 等它自然发生，你可能等一百次都等不到。构造出来，一次测完。
# ⚠️ Stage 4 给契约加了一个【必填字段】quotes，下面一半的用例当场变红。
#    这不是坏事 —— 这正是探针存在的意义：
#    契约改动会立刻打断所有依赖它的测试，而不是等你上线才发现。
PARSE_CASES = [
    # ── 必须能过：包装可以剥，内容没丢 ──
    ("干净 JSON",       '{"answer": "A", "used": [1, 3], "quotes": {"1": "x"}}',
     "ok", {"answer": "A", "used": [1, 3], "quotes": {"1": "x"}}),
    ("markdown 围栏",   '```json\n{"answer": "A", "used": [1], "quotes": {}}\n```',
     "ok", {"answer": "A", "used": [1], "quotes": {}}),
    ("前面有寒暄",      '好的，以下是我的回答：\n{"answer": "A", "used": [1], "quotes": {}}',
     "ok", {"answer": "A", "used": [1], "quotes": {}}),
    ("后面有废话",      '{"answer": "A", "used": [1], "quotes": {}}\n希望这对你有帮助！',
     "ok", {"answer": "A", "used": [1], "quotes": {}}),
    ("围栏 + 寒暄",     '好的：\n```json\n{"answer": "A", "used": [], "quotes": {}}\n```\n完毕',
     "ok", {"answer": "A", "used": [], "quotes": {}}),
    ("真·拒答",         '{"answer": "资料未提及", "used": [], "quotes": {}}',
     "ok", {"answer": "资料未提及", "used": [], "quotes": {}}),

    # ── 必须炸：内容是空的，补一个值就是在编 ──
    ("缺 used",         '{"answer": "A", "quotes": {}}',
     "raise", None),
    ("缺 quotes",       '{"answer": "A", "used": [1]}',
     "raise", None),          # ← Stage 4 新增，同样的理由
    ("answer 是空串",   '{"answer": "   ", "used": [1], "quotes": {}}',
     "raise", None),
    ("used 不是数组",   '{"answer": "A", "used": 1, "quotes": {}}',
     "raise", None),
    ("used 混了字符串", '{"answer": "A", "used": [1, "2"], "quotes": {}}',
     "raise", None),
    ("quotes 的值不是字符串", '{"answer": "A", "used": [1], "quotes": {"1": 42}}',
     "raise", None),
    ("quotes 的键不是数字", '{"answer": "A", "used": [1], "quotes": {"abc": "x"}}',
     "raise", None),          # ← 2026-09-29 补，见 generator.py 里那段注释
    # ★ 下面这条是【为什么判据用 int() 而不是 isdigit()】的活证据：
    #   '²'.isdigit() → True，isdigit 版会【放行】它，
    #   然后 cite.verify 里的 int('²') 抛 ValueError —— 崩在下游。
    #   没有这条用例，将来有人把判据"简化"回 isdigit()，测试全绿，他永远不会知道。
    #   测试不只是防回归，它还是【设计决定的存档】。
    ("quotes 的键是上标 2", '{"answer": "A", "used": [1], "quotes": {"²": "x"}}',
     "raise", None),
    # ★ 2026-10-06 补：这条测【放宽后新接受的】，不是【被拒绝的】。
    #   判据是 int()，所以 " 1" 和 "+1" 都算合法 —— 契约如此，就得写下来。
    #   上一版只测了拒绝侧，结果假阳性从接受侧漏了出去（见 probe_cite.py）。
    ("键带前导空格", '{"answer": "A", "used": [1], "quotes": {" 1": "x"}}',
     "ok", {"answer": "A", "used": [1], "quotes": {" 1": "x"}}),
    ("压根不是 JSON",   '我认为这段材料没有提到该信息。',
     "raise", None),
    ("顶层是数组",      '[1, 2, 3]',
     "raise", None),

    # ── ★ 已知漏洞：它炸，但炸错了地方（报错怪模型，实际是代码切错）──
    ("★ 已知漏洞",      '答案里有 {1,2}，JSON 是：{"answer": "A", "used": [1], "quotes": {}}',
     "raise", None),
]


def main() -> int:
    client = make_client()
    vectors, ids = load(model=EMB_MODEL)

    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}

    print(f"语料 {len(chunks)} 条   {vectors.shape}")
    # ⚠️ 这行原来写的是"现在跑的是 v0"—— v0 早就废弃了。
    #    同一个错有两处，上次只改了文件头那处，漏了这里。
    #    过期注释最坏的地方不是"错"，是它【读起来像真的】。
    print("现在跑的是 build_prompt 的 v1（generator.py）。v0 的战绩见文件头。\n")

    for n, (label, question) in enumerate(CASES, 1):
        print(LINE)
        print(f"【{n}/{len(CASES)}】{label} —— {question}")
        print(LINE)

        qv = embed_one(client, EMB_MODEL, question)
        hits = top_k_np(qv, vectors, ids, k=TOP_K)
        top = [by_id[cid] for cid, _ in hits]

        print(f"\n检索到 {len(top)} 条   分数 {hits[-1][1]:+.3f} ~ {hits[0][1]:+.3f}")
        for rank, (cid, s) in enumerate(hits, 1):
            print(f"  {rank}. {s:+.3f}  {cid:<24} {by_id[cid]['section'][:26]}")

        # 第一题把【模型实际看到的 prompt】原样打出来
        if n == 1:
            prompt = build_prompt(question, top)
            print(f"\n模型实际收到的 prompt（{len(prompt)} 字符）:")
            print("┄" * 74)
            print(prompt[:900] + ("\n……（后略）" if len(prompt) > 900 else ""))
            print("┄" * 74)

        t0 = time.time()
        try:
            out = answer(client, question, top, options={"temperature": 0})
        except Exception as e:
            print(f"\n[炸了] {type(e).__name__}: {e}\n")
            continue
        dt = time.time() - t0

        print(f"\n模型输出（{dt:.1f}s）:")
        print("┄" * 74)
        print(out["answer"])
        print("┄" * 74)
        print(f"解析出来的 used = {out['used']}\n")

    # 四道题是【演示】（要人眼看），关卡是【断言】（能自己判对错）。
    # 所以退出码只由关卡决定 —— 见 run_parse_gates 的 docstring。
    return 0 if run_parse_gates() else 1


def run_parse_gates() -> bool:
    """第 5~8 关：parse_response 拦不拦得住。离线，不调模型。

    ★ 返回值是 2026-10-10 加的。原来它把 passed/len 打印出来就完了 ——
      和 probe_cite.run_offline 是【一模一样的病】：打印不是返回值，
      于是 18/18 和 11/18 在退出码上长得一模一样（都是 0）。
      查这个病的时候顺手搜了一遍 scripts/，计分的只有这两个，
      所以只修这两处。哪天再添一个"会判对错"的探针，记得给它也留个出口。
    """
    print()
    print(LINE)
    print("第 5~8 关 · parse_response —— 包装能剥，内容不能编")
    print(LINE)

    passed = 0
    for name, raw, expect, want in PARSE_CASES:
        try:
            got = parse_response(raw)
            actual, kind = got, "返回"
        except ValueError as e:
            actual, kind = str(e), "抛错"
        except Exception as e:  # noqa: BLE001 —— 探针就是要看【炸成什么样】
            actual, kind = f"{type(e).__name__}: {e}", "炸错"

        if expect == "ok":
            good = kind == "返回" and actual == want
            detail = f"{actual!r}"
        else:
            good = kind == "抛错"
            detail = "ValueError" if good else f"{kind}: {actual}"

        passed += good
        mark = "✅" if good else "❌"
        print(f"  {mark} {name:<16} 期望 {expect:<6} 实际 {detail}")

    print()
    print(f"  {passed}/{len(PARSE_CASES)} 通过")

    # 已知漏洞那一关是【绿的】—— 因为它确实炸了。
    # 但我们满意的不是"它炸了"，是"它【没有静默出错】"。
    # 响的 bug 可以等，静的 bug 不能等 —— 这就是不修它的全部理由。
    print()
    print("  注：★ 那一关是绿的，但它炸错了地方 ——")
    print("     报错说「不是合法 JSON」，怪的是模型；")
    print("     实际是 find(\"{\") 从错误的位置切的，怪的是代码。")
    print("     详见 generator.parse_response 的「已知漏洞」。")

    print()
    print(LINE)
    print("验收：能答 / 拒答 / 格式 —— 三条都验完了。")
    print(LINE)
    return passed == len(PARSE_CASES)


if __name__ == "__main__":
    sys.exit(main())
