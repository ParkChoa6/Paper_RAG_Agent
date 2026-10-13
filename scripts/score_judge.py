"""把【人填完的】判卷表算成分数 —— 揭盲 + 逐臂答对率 + 配对检验。

── 流水线里的位置 ──

    eval_answer.py ──▶ e2e_judge.md   （盲判表：甲/乙/丙，人填「判定 ___」）
                   └─▶ e2e_key.json   （揭盲钥匙：第 N 题的甲是哪个 arm）

                              │  人填
                              ▼
                       e2e_judge.md（填好的）
                              │
                              ├──▶ score_judge.py  ★ 本文件 —— 算分
                              └──▶ e2e_key.json
                                        │
                                        ▼
                        每臂：对 / 错 / 缺 / 拒 / 编 的计数 + 答对率
                        dense vs rerank 的配对 McNemar

── 产品 / 脚手架 ──

    ★ 脚手架。只有跑评测的时候用，不在生产路径上。
      它【不改任何判定】，只做两件事：把 甲/乙/丙 换回 arm 名、把格子数成数。

── 为什么要有这个文件 ──

    判卷表是 28 题 × 3 臂 = **84 个格子**，全是人手填的。
    填完之后如果靠人眼数：「dense 大概对了 18 题吧」——
    那 84 个格子里积累的注意力，就在最后一步换成了一句估计。

    ★ 人做完的部分（判「这句答得对不对」）机器做不了；
      机器能做的部分（数数、配对）也别让人做。
      这条线画错了，前面 84 格的力气会漏在最后一步。

── 用法 ──

    # 默认读 data/processed/e2e_judge.md + e2e_key.json
    ./.venv/Scripts/python.exe scripts/score_judge.py

    # 也可以指别的（S5-7 那批结果每个都有自己的 _judge/_key）
    ./.venv/Scripts/python.exe scripts/score_judge.py <judge.md> <key.json> [<results.json>]

退出码：
    0  全部格子填好了，且判定词都合法
    1  有格子没填 / 判定词不认识 —— ★ 宁可不给分，也不猜
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_answer as ea  # noqa: E402  —— 复用它的 mcnemar，不另写一份（见下）

# ★ 判定词表。这五个词的定义抄自 eval_answer.write_judge_sheet 印在表头的那段，
#   两边必须一致 —— 表头改了这里不改，就会变成"表上说 `缺`，算分说没这个字"。
VERDICTS = {
    "对": "答案和关键事实一致，且没多编内容",
    "错": "答案和关键事实冲突，或答的是别的东西",
    "缺": "拒答，但材料里其实有答案（该答没答）",
    "拒": "拒答，且材料里确实没有答案（拒得对）",
    "编": "材料里没有，模型却给了答案（幻觉）",
}

# 一题算不算【答对】。只有「对」算。
# ★ 「拒」不算答对，但【也不算答错】—— 它是"正确地不回答"。
#   （把它算成错，等于惩罚一个守规矩的行为；
#     把它算成对，等于让拒答刷分。所以单列一栏，不进答对率的分子。）
CORRECT = "对"

# `判定 甲：____` / `判定 甲：对` / `判定 甲： 对 ` —— 都认。
BLANK_RE = re.compile(r"^判定\s*(\S+?)\s*[：:]\s*(.*)$")


def parse_sheet(text: str) -> tuple[dict[tuple[int, str], str], list[str], list[str]]:
    """把填好的判卷表读成 {(题号, 甲乙丙): 判定词}。

    返回 (已填的, 没填的, 不认识的)。
    ★ 三个都要返回：只返回"已填的"，调用方就分不清
      「这一格是空的」和「这一行我根本没解析到」——
      而这两种情况的处理完全相反（一个去填，一个去修解析器）。
    """
    filled: dict[tuple[int, str], str] = {}
    blank: list[str] = []
    bad: list[str] = []

    qno = 0
    for raw in text.splitlines():
        line = raw.strip()

        # 题号只认 `## 1. xxx` 这种（表头那个 `## ⚠️ ...` 不匹配）
        m = re.match(r"^##\s+(\d+)\.\s", line)
        if m:
            qno = int(m.group(1))
            continue

        m = BLANK_RE.match(line)
        if not m or qno == 0:
            continue

        lab, verdict = m.group(1), m.group(2).strip()
        where = f"第 {qno} 题 {lab}"

        if not verdict or set(verdict) <= {"_", "－", "-", " "}:
            blank.append(where)
        elif verdict not in VERDICTS:
            bad.append(f"{where}：`{verdict}`")
        else:
            filled[(qno, lab)] = verdict

    return filled, blank, bad


def main() -> None:
    base = Path("data/processed")
    judge_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "e2e_judge.md"
    key_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "e2e_key.json"

    if not judge_path.exists() or not key_path.exists():
        print(f"❌ 找不到：{judge_path} / {key_path}")
        sys.exit(1)

    # 结果文件（可选）：只用来查「标注错了」那一类 —— 见文末。
    #
    # ★ 只在文件名真的以 `_judge.md` 结尾时才去推导。
    #   否则 `str.replace` 找不到就【原样返回】，res_path 会等于判卷表自己 ——
    #   然后拿 markdown 去 json.loads，报一个看着像"文件坏了"的错。
    #   （这正是我第一版写的：`j.md` → 推导成 `j.md`。）
    if len(sys.argv) > 3:
        res_path = Path(sys.argv[3])
    elif judge_path.name.endswith("_judge.md"):
        res_path = judge_path.with_name(
            judge_path.name[: -len("_judge.md")] + "_results.json")
    else:
        res_path = None

    filled, blank, bad = parse_sheet(judge_path.read_text(encoding="utf-8"))
    key: dict[str, dict[str, str]] = json.loads(key_path.read_text(encoding="utf-8"))

    n_total = sum(len(v) for v in key.values())
    print("=" * 78)
    print(f"判卷表 {judge_path}")
    print(f"  格子总数 {n_total}    已填 {len(filled)}    空着 {len(blank)}    不认识 {len(bad)}")
    print("=" * 78)

    if blank:
        print("\n⚠️ 还空着的格子（判定词写这几行后面就行）：")
        for w in blank:
            print(f"     · {w}")
    if bad:
        print("\n❌ 判定词不认识（只认 对 / 错 / 缺 / 拒 / 编）：")
        for w in bad:
            print(f"     · {w}")

    if blank or bad:
        print("\n★ 宁可不给分，也不猜 —— 你没填的格子，我替你说『对』就是在编。")
        print("  填完再跑一次。")
        sys.exit(1)

    # ── 揭盲：把 (题号, 甲乙丙) 换回 arm 名 ──
    by_arm: dict[str, dict[int, str]] = {}
    for q_str, labs in key.items():
        q = int(q_str)
        for lab, arm in labs.items():
            by_arm.setdefault(arm, {})[q] = filled[(q, lab)]

    arms = sorted(by_arm)
    n_q = max(max(v) for v in by_arm.values())

    # ── 逐臂计数 ──
    print()
    # ★ `编` 那一栏和后面的答对率之间【必须留空】——
    #   不留的话 ` 1` + `15/28` 会拼成 ` 115/28`，看着像一个数。
    #   这不是排版洁癖：一张会让人读错数的表，比没有表更坏。
    head = f"{'arm':<10}" + "".join(f"{v:>6}" for v in VERDICTS) + "    答对率"
    print(head)
    print("-" * len(head))
    rates: dict[str, float] = {}
    for arm in arms:
        c = {v: 0 for v in VERDICTS}
        for verdict in by_arm[arm].values():
            c[verdict] += 1
        n = len(by_arm[arm])
        rates[arm] = c[CORRECT] / n if n else 0.0
        print(f"{arm:<10}" + "".join(f"{c[v]:>6}" for v in VERDICTS)
              + f"    {c[CORRECT]}/{n} = {rates[arm] * 100:.0f}%")

    print()
    print("对 = 答对（唯一进分子的）    拒 = 拒得对（正确行为，但不算答对）")
    print("缺 = 该答没答    错 = 答错    编 = 材料里没有却答了")

    # ── 配对检验：只有【同一批题、两个臂都有分】才能配对 ──
    if len(arms) >= 2:
        a, b = ("dense", "rerank") if {"dense", "rerank"} <= set(arms) else (arms[0], arms[-1])
        common = sorted(set(by_arm[a]) & set(by_arm[b]))
        ok_a = [by_arm[a][q] == CORRECT for q in common]
        ok_b = [by_arm[b][q] == CORRECT for q in common]

        # ★ mcnemar 返回 (b, c, p)：
        #     b = A 成功而 B 失败  → 只有 a 对
        #     c = B 成功而 A 失败  → 只有 b 对
        #   ★ 不自己再数一遍。同一个量推导两次 = 两条路径会各自漂移
        #     （cite.py 里那个 {int(k)} / str(n) 的 bug 就是这么来的）。
        a_only, b_only, p = ea.mcnemar(ok_a, ok_b)

        print()
        print("=" * 78)
        print(f"★ 配对 McNemar：{a} vs {b}（{len(common)} 题）")
        print("=" * 78)
        print(f"  答对率   {a} {rates[a] * 100:.0f}%   →   {b} {rates[b] * 100:.0f}%"
              f"   （差 {(rates[b] - rates[a]) * 100:+.0f} 点）")
        print(f"  只有 {b} 对：{b_only} 题      只有 {a} 对：{a_only} 题")
        print(f"  p = {p:.3f}   （McNemar，连续性校正）")
        print()
        if b_only + a_only == 0:
            print("  ★ 两臂【一题都没换】—— 这不是「没测出差别」，是【确切地没有差别】。")
        elif p >= 0.05:
            print(f"  ★ p = {p:.3f} ≥ 0.05 —— 没有证据表明两臂答对率不同。")
        else:
            print(f"  ★ p = {p:.3f} < 0.05 —— 两臂有差别。")

        # ★ n=28 的诚实提示。不写这句，上面那两个百分号一定会被读过头。
        print()
        print("  ⚠️ n=28，80% 附近 95% 区间约 ±15 点。"
              "「差 +N 点」在这个样本量上要慎读（见 S5-6 记录 §4.5）。")

    # ── 顺手查一遍「标注错了」那一类 ──
    #   表头那段 ⚠️ 说：「材料里没有答案」标注是【假设】，判卷就是检验它。
    #   检验结果长这样：标注说没有、模型却答了、而且你判了「对」
    #   → 标注错了，该把那条 target 补进 queries.json。★ 这条线索只有人判完才有。
    if res_path is not None and res_path.exists():
        rows_by_arm: dict[str, list[dict]] = json.loads(res_path.read_text(encoding="utf-8"))
        suspects = []
        for arm, verdicts in by_arm.items():
            rows = rows_by_arm.get(arm) or []
            for q, verdict in sorted(verdicts.items()):
                if q - 1 >= len(rows):
                    continue
                r = rows[q - 1]
                if verdict == CORRECT and not r.get("target_in_materials"):
                    suspects.append((q, arm, r))
        if suspects:
            print()
            print("=" * 78)
            print(f"★ 标注可能错了：判了「对」，但标注说「材料里没有答案」（{len(suspects)} 处）")
            print("=" * 78)
            for q, arm, r in suspects:
                print(f"  第 {q} 题 [{arm}] {r['q'][:40]}")
            print()
            print("  → 表头那段 ⚠️ 说的就是这种：targets 是「足够的答案」不是「唯一的答案」。")
            print("    这几条应当把对应段落补进 data/eval/queries.json 的 targets。")
    else:
        print()
        print(f"（没有 {res_path.name if res_path else '配套的 _results.json'}，"
              f"跳过「标注错了」那一步的检查）")


if __name__ == "__main__":
    main()
