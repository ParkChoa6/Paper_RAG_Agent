"""端到端评测 —— 不量检索，量【用户最后拿到的那句话】。

和前面两把尺子的分工：

    eval_embed.py      固定「向量检索」，换【嵌入模型】比
    eval_retrieve.py   固定题和语料，换【检索策略】比      ← 量的是【排名】
    eval_answer.py     固定整条流水线，量【答案对不对】    ← 本文件

前两把尺子量的都是「正确段落排第几」。那是中间量。
用户看不到排名，用户只看到一句话。这两件事【可以脱节】——

    Top-1 从 50% 涨到 68%，不等于「答对的题多了 18 道」。
    涨的那 18 道题里，可能本来靠别的材料也能答对；
    也可能涨了之后模型反而被新材料的干扰带偏。

所以这一步必须单独量，而且必须量在【生成跑完之后】。

═══ 三个 arm，为什么是这三个 ═══

    dense     控制组。★ 只跑生产配置的话，「合规率 85%」没有参照系 ——
              你不知道那 85% 里有多少是检索贡献的、多少是模型本来就行的。
    rerank    生产配置（= config 里的默认值）。
    oracle    ★★ 把【标准答案】直接当检索结果喂下去。检索层 100% 正确。

加了 oracle，一条脱节的流水线才能被【切开】：

    检索层的损失 = 1 − 材料含答案率     （target 压根没进那 5 条材料）
    生成层的损失 = oracle 的失败率        （材料里有答案，模型还是没答对）

    没有 oracle，你看到「rerank 合规率 85%」只能干瞪眼：
    剩下那 15% 是检索没捞到？还是捞到了模型没用好？修法完全不同。
    有了 oracle，第二个问题被单独量出来了。

★ oracle 靠什么塞进去？—— `ask()` 的参数是 `retriever`，而 `Retriever`
  是个 Protocol（S5-1 定的），只要求 `name` 和 `search` 两个东西。
  所以这里【手搓一个假的】就能接进去，`ask()` 一个字都不用改。
  这是那个 Protocol 判断的第三次兑现：
      dense → bm25        一行没改
      + rerank            一行没改
      → 一个作弊的假货     还是不用改

═══ 量什么（★ 客观列和主观列必须分开）═══

    ── 客观，脚本自己算 ──

    status 分布      answered / refused / suspect / parse_failed
    引用有问题        status=answered 但 cite.verify 报出了问题
    材料含答案        ★ 那座【桥】—— target 有没有被送进 prompt
    引对材料          材料里有答案时，模型引用的编号指向的是不是含答案那条

    ── 主观，必须人来判 ──

    「答对了吗」

    ★★ 为什么不用 qwen2:7b 自己判：
       ① 自评偏好（self-preference）—— 同一个模型判自己的答案会系统性偏高；
       ② 更根本的：那样你就有了【两把没校准过的尺子】——
          一把是它答的，一把是它判的，两个都不知道准不准。
          S5-1（query 前缀）、S5-2（扩展开关）、S5-5（MRR 定义）
          已经栽过三次「手搓第二把尺子」，这次连模型都别信。

    ★ 所以脚本产出两张东西：
       data/processed/e2e_results.json   机器可算的全部原始结果
       data/processed/e2e_judge.md       ★ 判卷表 —— 给人填的
       data/processed/e2e_key.json       揭盲用的对照表（先别看）

    ★★ 判卷表是【盲】的：同一道题的 rerank 版和 dense 版
       随机编成「甲 / 乙」，标注顺序逐题打乱。
       不盲的话，你会在看到答案之前就知道哪个是"新方法"，
       然后不自觉地宽容它 —— 这不是道德问题，是人脑的默认行为。

用法:
    ./.venv/Scripts/python.exe scripts/eval_answer.py                    # 三个 arm 全跑
    ./.venv/Scripts/python.exe scripts/eval_answer.py --arms rerank
    ./.venv/Scripts/python.exe scripts/eval_answer.py --limit 3          # 冒烟
    ./.venv/Scripts/python.exe scripts/eval_answer.py --out D:/tmp/x     # 换输出前缀
    ⚠️ --out 给的若是【不带目录的裸名字】，产物落在【当前目录】而不是
       data/processed/ 下。要指定目录就写全（data/processed/xxx）。

    ★ S5-7 加的两个开关（A/B 用；默认值都是生产值）:
    ./.venv/Scripts/python.exe scripts/eval_answer.py --arms dense --no-gen-schema
    ./.venv/Scripts/python.exe scripts/eval_answer.py --arms dense --gen-seed 42

前置: Ollama 起着 + 索引存在（scripts/build_vectors.py）

── 这是【产品】还是【脚手架】？──

    脚手架。它不产出任何被别处 import 的东西。
    真正可复用的那条链是 paper_rag/pipeline.py 的 ask() —— 本文件
    【一行判断逻辑都不该有】，全部调 ask() 然后数结果。
    （唯一的例外是 OracleRetriever，见它的 docstring：那是评测专用的作弊器，
      不该进 paper_rag/。）
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag import cite, generator
from paper_rag.config import (
    EMB_MODEL,
    EVAL_DIR,
    GEN_JSON_SCHEMA,
    GEN_MODEL,
    PROCESSED_DIR,
    TOP_K,
)
from paper_rag.embed import make_client
from paper_rag.index import load
from paper_rag.pipeline import ask, build_retriever, expand_query

# temperature=0 —— 评测必须可复现。
# 同一个问题两次跑出两个答案，你就没法判断「改了一行之后是变好了还是碰巧」。
# ★ 这一条在【生成】这半边比在检索那半边重要得多：
#   检索是确定性的，生成不是。忘了传，整张表都是噪声。
#
# ⚠️⚠️ 2026-10-13（S5-7）**差点写下一个错的警告，这里是改对之后的版本。**
#
#    第一版警告写的是：「temperature=0 【并没有】让它可复现 ——
#    同一配置跑两遍，28 题里 7 题答案实质不同。」
#    那是**从一对运行**（A vs A2）读出来的。后来把基线补到 8 次运行，
#    才发现：**那 7 题全部来自 8 次里唯一的一次异常运行。**
#    另外 21 个配对全在 0~2/28。
#
#    现在实测到的本底噪声（8 次无约束运行）：
#       解析失败    0 题   —— 8/8 次都恰好是 [Q6, Q13]，完全确定
#       材料含答案  0 题   —— 8/8 次都是 20
#       合规回答    1 题   —— 22 或 23
#       答案文本    1 / 28 —— 而且永远是 Q3
#
#    ★ 所以：`temperature=0` 是【相当可复现】的，单遍 A/B 大体可信；
#      但 1 题的差别永远不算数。**别再用一对运行去判断"什么是噪声"。**
#    ★ 那次异常运行的成因没查清（代码、prompt、检索都排除了）。
#      它恰好长得像"默认温度下的采样"（见 S5-7 记录 §4.4 / §7.1）。
#    → `--gen-seed` 试过了，**没用**：不同 seed 之间的差异和同 seed 一样小。
GEN_OPTIONS = {"temperature": 0}


class OracleRetriever:
    """作弊检索器：直接把【标准答案】当检索结果交出去。

    ★ 它满足 Retriever 协议 —— 一共就两个约定：name 和 search。
      所以 ask() 完全不知道自己在跟一个假货说话。见模块开头。

    ★ 为什么要 pad（补足到 k 条）：
      有些题只有 1 个 target，直接只喂 1 条材料的话，prompt 里材料数
      和别的 arm 就不一样了（别的 arm 都是 k=5 条）。
      那 prompt 本身就变了，比的就不是同一件事了。
      所以：target 排最前，剩下的空位用 dense 的结果补齐（去掉重复的）。

    ⚠️ 它【不是】生产代码。真上线不可能有这个东西 ——
      它存在的唯一目的是「把生成层单独拎出来量」。
      这就是它待在 scripts/ 而不是 paper_rag/ 的原因。
    """

    def __init__(self, by_id: dict, key_to_targets: dict, pad, k: int):
        self.by_id = by_id
        self.key_to_targets = key_to_targets
        self.pad = pad
        self.k = k

    @property
    def name(self) -> str:
        return f"ORACLE(标准答案直喂)@k{self.k}"

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        k = self.k if k is None else min(k, self.k)
        targets = self.key_to_targets.get(query)
        if targets is None:
            # ★ 找不到 key 就【大声炸】。悄悄返回空的话，
            #   表现出来的样子是"生成层答不出来"，你会去查模型 ——
            #   而真正的问题是这个假检索器没对上号。
            raise KeyError(
                f"OracleRetriever 没有 {query!r} 的答案。"
                f"（key 必须和 ask() 实际送进来的 query 完全一致 —— "
                f"注意 expand=True 时送进来的是【扩展后】的文本）"
            )

        hits = [(cid, 0.0) for cid in targets if cid in self.by_id]
        # 补足到 k 条，用 dense 的结果当填充。★ 别拿分数当信号 —— 0.0 是占位。
        for cid, _score in self.pad.search(query, k=self.k + len(targets)):
            if len(hits) >= k:
                break
            if cid not in targets and cid in self.by_id:
                hits.append((cid, 0.0))
        return hits[:k]


def build_arms(names: list[str], client, chunks, vectors, ids, targets_map, k):
    """按名字建检索器，一次建好，全程复用。

    ★★ 建在【循环外】。这条在 S5-5 量过：重排器每题重建一次的代价是
       每题多付一次构造 + 可能在第一次前向上付 ~9.5s 预热。
       而且它不报错，只是"评测突然变得很慢"。
    """
    by_id = {c["chunk_id"]: c for c in chunks}
    out = {}

    # dense 和 rerank 的底座是同一个 DenseRetriever —— 但【对象要分开】。
    # 共用一个对象不是不行，是两个 arm 的意义就变成"同一个索引上换了排序"，
    # 而这里刻意要让它们各自独立、互不干扰。
    dense_rt = None
    for name in names:
        if name == "dense":
            dense_rt = build_retriever(client, EMB_MODEL, vectors, ids, chunks,
                                       k=k, enabled=False)
            out["dense"] = dense_rt
        elif name == "rerank":
            out["rerank"] = build_retriever(client, EMB_MODEL, vectors, ids, chunks,
                                            k=k, enabled=True)
        elif name == "oracle":
            if dense_rt is None:
                dense_rt = build_retriever(client, EMB_MODEL, vectors, ids, chunks,
                                           k=k, enabled=False)
            out["oracle"] = OracleRetriever(by_id, targets_map, dense_rt, k)
        else:
            raise SystemExit(f"不认识这个 arm: {name!r}（有 dense / rerank / oracle）")
    return out


def run_arm(arm: str, retriever, queries, client, chunks, vectors, ids, *,
            k, limit, gen_schema):
    """在一个 arm 上跑完整个评测集，返回每题一行原始记录。"""
    print(f"\n{'=' * 78}")
    print(f"arm = {arm}    检索器: {retriever.name}    JSON schema 约束: {gen_schema}")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for n, item in enumerate(queries[:limit] if limit else queries, 1):
        q = item["q"]
        targets = set(item["targets"])

        t1 = time.time()
        r = ask(
            client, q, chunks, vectors, ids,
            k=k,
            expand=True,          # ★ 生产是开着的，评测必须跟生产同一条路
            options=GEN_OPTIONS,
            retriever=retriever,
            gen_schema=gen_schema,
        )
        secs = time.time() - t1

        material_ids = [m["chunk"]["chunk_id"] for m in r["retrieved"]]
        # ★ 那座【桥】：标准答案有没有被送进 prompt。
        target_in = any(cid in targets for cid in material_ids)

        # 模型引用的编号指向的材料里，有没有含答案的那条。
        # ★ used 是【材料编号，从 1 开始】—— 见 generator.build_prompt。
        #   所以这里要 -1 转成下标。写错了不会报错，只会静默全判 False。
        used = (r["parsed"] or {}).get("used") or []
        cited_target = any(
            0 < u <= len(material_ids) and material_ids[u - 1] in targets
            for u in used
        )

        row = {
            "arm": arm,
            "n": n,
            "q": q,
            "targets": item["targets"],
            "retriever": r["retriever"],
            # ★ S5-7：把【生成侧的开关】也记进每一行。
            #   它不改检索、不改题、不报错 —— 和 S5-1 的 query 前缀、
            #   S5-2 的扩展开关是同一类"只影响一处的开关"。
            #   前两次都是【事后才发现两份结果量的不是一回事】，这次先把话说在前面。
            "gen_schema": gen_schema,
            "gen_options": dict(GEN_OPTIONS),
            "status": r["status"],
            "n_problems": len(r["problems"]),
            "problems": r["problems"],
            "note": r["note"],
            "answer": (r["parsed"] or {}).get("answer", ""),
            "used": used,
            # ★ quotes 必须留下来。光看 problems 里那句「疑似伪造」，
            #   你分不清两种病：
            #     ① 模型【编】了一句原文里没有的话  → 幻觉，要收紧 prompt
            #     ② 模型把英文材料【翻译/改写成中文】再交 → 契约没被执行，
            #        要改的是「answer 中文而 quotes 必须英文」这条要求的传达
            #   两者的药完全不同，而 problems 文本把它们糊成一句。
            "quotes": (r["parsed"] or {}).get("quotes", {}),
            # ★ query 也留下来：确认三个 arm 送进检索的是不是同一段文本
            #   （S5-2 的教训 —— 评测和生产量不同 query，全程不报错）。
            "query": r["query"],
            "materials": [
                {"chunk_id": m["chunk"]["chunk_id"],
                 "title": m["chunk"]["title"],
                 "is_target": m["chunk"]["chunk_id"] in targets}
                for m in r["retrieved"]
            ],
            "target_in_materials": target_in,
            "cited_target": cited_target,
            "secs": round(secs, 2),
        }
        rows.append(row)

        mark = {"answered": "OK ", "refused": "拒答", "suspect": "!! ",
                "parse_failed": "!! "}.get(r["status"], "?? ")
        flag = "桥通" if target_in else "桥断"
        print(f"  [{n:>2}] {mark} {flag}  引用{'有问题' if r['problems'] else '通过'}  "
              f"{secs:>5.1f}s  {q[:36]}")

    print(f"  ── {arm} 跑完，{time.time() - t0:.0f}s")

    # ★ 实时落盘：这轮要跑十几分钟，中途被测者杀掉不该全丢。
    return rows


def summarize(rows: list[dict], n_q: int) -> dict:
    """把原始记录数成指标。★ 只数，不判断。"""
    answered = [r for r in rows if r["status"] == "answered"]
    clean = [r for r in answered if r["n_problems"] == 0]
    with_target = [r for r in rows if r["target_in_materials"]]
    cited_ok = [r for r in with_target if r["cited_target"]]
    return {
        "n": len(rows),
        "answered": len(answered),
        "answered_clean": len(clean),
        "refused": sum(1 for r in rows if r["status"] == "refused"),
        "suspect": sum(1 for r in rows if r["status"] == "suspect"),
        "parse_failed": sum(1 for r in rows if r["status"] == "parse_failed"),
        "cite_problems": sum(1 for r in answered if r["n_problems"] > 0),
        "target_in_materials": len(with_target),
        "cited_target": len(cited_ok),
        "secs_total": round(sum(r["secs"] for r in rows), 1),
        "secs_per_q": round(sum(r["secs"] for r in rows) / max(len(rows), 1), 1),
    }


def print_table(summs: dict[str, dict]) -> None:
    print(f"\n{'=' * 100}")
    print("端到端对比")
    print("=" * 100)
    print(f"{'配置':<10}{'合规回答':>9}{'拒答':>6}{'可疑':>6}{'解析失败':>9}"
          f"{'引用有问题':>11}{'材料含答案':>11}{'引对材料':>9}{'秒/题':>8}")
    print("-" * 100)
    for arm, s in summs.items():
        n = s["n"]
        pct = lambda x: f"{x / n * 100:.0f}%"
        # 「引对材料」的分母不是 n —— 是【材料里真有答案】的题数。
        # 材料里压根没有答案的题，"没引对"是检索的锅，不该算到模型头上。
        cited_pct = (f"{s['cited_target'] / s['target_in_materials'] * 100:.0f}%"
                     if s["target_in_materials"] else "—")
        print(f"{arm:<10}{pct(s['answered_clean']):>9}{s['refused']:>6}{s['suspect']:>6}"
              f"{s['parse_failed']:>9}{s['cite_problems']:>11}"
              f"{pct(s['target_in_materials']):>11}{cited_pct:>9}{s['secs_per_q']:>8.1f}")
    print("-" * 100)
    print("合规回答   = status=answered 且引用校验全过。★ 这【不】等于「答对了」——")
    print("             它只说明「交付了一个有据可查的答案」。对不对要人判。")
    print("材料含答案 = 标准答案被送进了 prompt 的题数占比 ← ★ 检索层的成绩")
    print("引对材料   = 材料里真有答案时，模型引用的编号指向该条的占比")
    print("             （分母是「材料含答案」的题数，不是总题数）")
    print()
    print("★ 怎么用这张表做分解：")
    print("    1 − 材料含答案率  = 【检索层】丢掉的题")
    print("    oracle 那行       = 【生成层】的天花板（检索满分时能做到多少）")
    print("    生产那行和 oracle 的差 = 检索层拖累的部分")
    print("    oracle 自己没做到的   = 生成层的锅，修检索没用")
    print()
    print("⚠️ 「合规回答」这个列【偏高】—— 模型完全可以引用一段真材料，")
    print("   然后从中推出一个错的结论。引用校验只保证引文没伪造，")
    print("   不保证结论正确。真正的那一列在判卷表里，得人来填。")


def classify_citations(rows: list[dict], chunks: dict) -> list[tuple]:
    """把每一条【引用校验失败】的引文，回语料里查一遍，分开是哪种病。

    ══════════ ★ 2026-10-13（S5-8）：这一步的定位变了，说清楚 ══════════

      写它的时候（S5-6，见那份记录 4.6），`cite.verify` 报出来的都是同一句话
      「原文中找不到对应内容（疑似伪造）」，背后至少是【三种】不同的东西：

          假警报   引文【完全正确】（多抄了 "[N] "、或 PDF 丢失空格、或连字符断行）
          引错编号  内容真实存在，只是编号数错了
          幻觉     引文在【全语料】里都不存在

      **有损压缩不可逆** —— 所以当时在这里补了一层事后分析。

    ★ S5-8 把前两种【收进了 verify 自己】：
        前缀三兄弟（表头 / PDF 丢空格 / PDF 断行）在 cite._norm 里直接消掉；
        引错编号变成 verify 的一种独立报错：「编号错配，不是伪造」。
      所以现在这一段对【新跑出来的】结果，主要只剩一件事可做 —— 见下。

    ★ 判据仍然用【全语料】而不是【本次材料】，这条【没变】，而且现在更重要了：
      verify 只有本次材料，它说「疑似伪造」的意思是
      「这句话不在我给你的这批材料里」。但那句话可能【在语料库里】——
      那是"模型凭记忆引用"，不是"模型编话"，两者的药不一样。
      这个区分 verify 做不到（它拿不到全语料），只能在这里做。

    ⚠️ 它读的是【存下来的】 problems 字符串，不是现算的 verify 输出。
       所以对 S5-8 之前的旧结果文件，它照样能拆 —— 那些文件里存的是旧文案。
       新旧混着看时，记得文案已经变了：新文案里「编号错配」是 verify 自己报的。

    返回 [(题号, 引用编号, 类别), ...]
    """
    norm_all = [cite._norm(c["text"]) for c in chunks.values()]
    out = []
    for r in rows:
        if not r["n_problems"]:
            continue
        mats = [chunks[m["chunk_id"]] for m in r["materials"] if m["chunk_id"] in chunks]
        for p in r["problems"]:
            m = re.search(r"引用 \[(\d+)\]", p)
            if not m:
                continue
            i = int(m.group(1))
            q = r.get("quotes", {}).get(str(i), "")
            # 先试着剥掉模型可能多抄的那个 "[N] " 前缀
            stripped = re.sub(r"^\[\d+\]\s*", "", q.strip())
            if not (1 <= i <= len(r["materials"])):
                out.append((r["n"], i, "越界"))
                continue
            rendered = generator.render_material(i, mats[i - 1]) if i <= len(mats) else ""
            if cite._norm(stripped) in cite._norm(rendered):
                # ★ S5-8 改名：原来叫「前缀」。现在 _norm 已经能看穿前缀/丢空格/断行
                #   三兄弟，所以走到这一支的真正含义是 ——
                #   「引文确实在被引的那条材料里，是【我们旧版的校验器报错了】」。
                out.append((r["n"], i, "假警报"))
            elif any(cite._norm(stripped) in v for v in norm_all):
                out.append((r["n"], i, "引错编号"))
            else:
                out.append((r["n"], i, "幻觉"))
    return out


def print_citation_taxonomy(rows_by_arm: dict, chunks: dict) -> None:
    print(f"\n{'=' * 100}")
    print("★ 引用校验失败 —— 拆成四种病")
    print("=" * 100)
    for arm, rows in rows_by_arm.items():
        detail = classify_citations(rows, chunks)
        if not detail:
            print(f"{arm:<8} 没有引用失败")
            continue
        counts = {}
        for _, _, kind in detail:
            counts[kind] = counts.get(kind, 0) + 1
        s = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{arm:<8} {s:<44} 明细={[(n, i, k) for n, i, k in detail]}")
    print("-" * 100)
    print("假警报   = 引文确实在被引的那条材料里 ← ★ 这是【我们的校验器】的锅，不是模型的")
    print("           （旧版把表头 '[N] ' / PDF 丢空格 / PDF 断行都误报成伪造 —— 见 S5-8）")
    print("引错编号 = 内容真实存在，编号数错了")
    print("幻觉    = 引文在【整个语料】里都不存在")
    print()
    print("⚠️ 别把这三类合成一个『引用失败率』。它们修法不同，合成之后就再也拆不开了。")


def mcnemar(a_ok: list[bool], b_ok: list[bool]) -> tuple[int, int, float]:
    """配对 McNemar 检验 —— 同一个题集上，两个配置谁更好。

    ★ 为什么【必须】是配对的：
      这不是"两批独立的题各测一次"，是【同一批 28 题】各测两次。
      同一个配置在哪几题上容易挂，是题的性质，不是配置的性质 ——
      把两批当独立样本算，会系统性地低估 / 高估显著性。

    返回 (b, c, p)：
        b = A 成功而 B 失败的题数
        c = B 成功而 A 失败的题数
        p = 双侧 p 值（1 自由度卡方，连续性校正）
    """
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi2 / 2))


def print_noise_floor(rows_by_arm: dict[str, list[dict]], summs: dict) -> None:
    """★ 噪声地板 —— n 这么小的时候，多大的差才算差。

    ★★ 这一栏不是装饰。没有它，上面那张表会诱导人把 3 个点的差
       当成"变好了" —— 而在 n=28 上，3 个点连方向都定不下来。

        S5-5 说过「能算样本量」。这里就是它的兑现：
       【在 reporting 之前，先告诉大家这台仪器的分辨率是多少。】

    ★ 用的是【配对】检验（McNemar），不是两组独立比例的比较 ——
      见 mcnemar 的 docstring。
    """
    n = next(iter(summs.values()))["n"]
    print(f"\n{'=' * 100}")
    print(f"★ 噪声地板 —— n={n} 时，多大的差才算差")
    print("=" * 100)

    # 单个比率的正态近似半宽。p=0.5 是最坏情况（方差最大）。
    for p in (0.5, 0.8, 0.9):
        se = 1.96 * math.sqrt(p * (1 - p) / n)
        print(f"    某条线在 {p*100:.0f}% 附近时，单独的 95% 区间是 ±{se*100:.0f} 点")
    print("    ⚠️ 两条线各自的区间【叠在一起】不代表它们没差别（那是配对数据），")
    print("       但反过来：**差 5 个点以内，别急着说谁好。**")
    print()

    arms = list(rows_by_arm.keys())
    if len(arms) < 2:
        return

    def clean_flags(rows):
        return [r["status"] == "answered" and r["n_problems"] == 0 for r in rows]

    def bridge_flags(rows):
        return [r["target_in_materials"] for r in rows]

    print(f"{'配对比较':<28}{'b':>4}{'c':>4}{'p':>9}   结论")
    print("-" * 100)
    # ★ 比【所有配对】，不是只比相邻的。
    #   第一版写的是 range(len(arms)-1) —— 那只比 dense→oracle 和 oracle→rerank，
    #   【恰好漏掉了 dense→rerank】，而那一对才是本步真正要看的东西
    #   （生产和控制组）。一个只比邻居的循环，会让最重要的比较根本不出现在表上，
    #   而且它看起来完全正常。
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            for label, fn in (("合规回答", clean_flags), ("材料含答案", bridge_flags)):
                a, b_ = arms[i], arms[j]
                x, y = fn(rows_by_arm[a]), fn(rows_by_arm[b_])
                bb, cc, p = mcnemar(x, y)
                verdict = ("★ 可分辨" if p < 0.05 else
                           "（分不出来 —— 落在噪声里）" if p > 0.20 else
                           "（弱信号，别下结论）")
                print(f"{label + ': ' + a + ' → ' + b_:<28}{bb:>4}{cc:>4}{p:>9.3f}   {verdict}")
    print("-" * 100)
    print("b = 左边成功、右边失败的题数；c = 右边成功、左边失败的题数")
    print("p < 0.05 才算「这个差异是真的」；p > 0.20 基本可以说「这俩没法区分」。")
    print()
    print("★★ 这条比上面所有数字都重要：")
    print("   28 题能稳定分辨的是【两个配置在哪几道上不同】，")
    print("   不是【精确到个位数的百分比】。写进文档时要写区间，不要写单点。")


def write_judge_sheet(rows_by_arm: dict[str, list[dict]], queries: list[dict],
                      path: Path, key_path: Path, seed: int = 20261013) -> None:
    """生成【盲判】表：同一题的多个 arm 随机编成 甲/乙/丙。

    ★ 打乱顺序的种子写死 —— 判卷表要能重现。
      种子随机会让「同一份结果生成两次得到两张不同的表」，
      你就没法确认自己填的是哪一张。
    """
    rng = random.Random(seed)
    arms = list(rows_by_arm.keys())
    labels = "甲乙丙丁"

    lines = [
        "# 端到端判卷表（盲）",
        "",
        "每个格子独立判，**不要**回头猜哪个是哪个 arm。",
        "判定用词（填在每条的「判定」后面）：",
        "",
        "- `对`   —— 答案和「关键事实」一致，且没有多编内容",
        "- `错`   —— 答案和关键事实冲突，或者答的是别的东西",
        "- `缺`   —— 拒答，但材料里其实有答案（该答没答）",
        "- `拒`   —— 拒答，且材料里确实没有答案（拒得对）",
        "- `编`   —— 材料里没有，模型却给了答案（幻觉）",
        "",
        "> 判完再打开 `e2e_key.json` 对号入座。",
        "",
        "## ⚠️ 「材料里有/没有答案」这个标注，自己是有误差的",
        "",
        "它判的是「**targets 里列的**那几条在不在材料里」，不是"
        "「材料里有没有**任何**能回答这个问题的内容」。",
        "",
        "`targets` 是「**足够**的答案」，不一定是「**唯一**含答案的段落」。"
        "所以标着「材料里没有答案」的时候，真相有两种：",
        "",
        "- 材料里**真**没有 → 模型还答了，那是**幻觉**，判 `编`；",
        "- 材料里有**别的**能答的段落（只是没被列进 targets）→ 模型没编，"
        "这一格标错了，判 `对`，**并且顺手把这条 target 补进 queries.json**。",
        "",
        "★ 别把这个标注当结论。它是**一条待检验的假设**，判卷就是检验它的过程。",
        "（这也是为什么判卷必须人来做 —— 脚本没有能力区分上面这两种。）",
        "",
        "---",
        "",
    ]

    key = {}
    for i, item in enumerate(queries):
        q = item["q"]
        # 关键事实：why 里带着原文那句英文答案
        lines.append(f"## {i + 1}. {q}")
        lines.append("")
        lines.append(f"**关键事实**：{item['why']}")
        lines.append("")

        present = [(a, rows_by_arm[a][i]) for a in arms
                   if i < len(rows_by_arm[a])]
        # ★ 逐题重新打乱。全局固定的顺序也会泄漏 ——
        #   判到第 5 题你就会发现"甲总是同一个 arm"。
        rng.shuffle(present)

        for j, (arm, row) in enumerate(present):
            lab = labels[j]
            key.setdefault(str(i + 1), {})[lab] = arm
            if row["target_in_materials"]:
                at = [str(n + 1) for n, m in enumerate(row["materials"]) if m["is_target"]]
                bridge = f"材料里有答案（在 [{'、'.join(at)}]）"
            else:
                bridge = "材料里**没有**答案"
            lines.append(f"**{lab}**（{bridge}｜{row['status']}"
                         f"{'｜引用有问题' if row['n_problems'] else ''}）")
            lines.append("")
            ans = row["answer"].strip() or "（空）"
            lines.append(f"> {ans}")
            lines.append("")
            # ★ 引文也给判卷人看：它是「模型有没有真去读材料」的直接证据，
            #   而且引用校验报的「疑似伪造」到底是编的还是翻译的，
            #   只有看到原文才能判。见 run_arm 里 quotes 那段注释。
            if row.get("quotes"):
                lines.append("引文：")
                for n, quote in row["quotes"].items():
                    lines.append(f"- `[{n}]` {quote}")
                lines.append("")
            if row["problems"]:
                lines.append("引用校验：")
                for p in row["problems"]:
                    lines.append(f"- {p}")
                lines.append("")
            lines.append(f"判定 {lab}：____")
            lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端评测：量答案，不量排名")
    ap.add_argument("--arms", default="dense,rerank,oracle",
                    help="逗号分隔：dense,rerank,oracle（默认全跑）")
    ap.add_argument("-k", type=int, default=TOP_K, help=f"检索几条（默认 {TOP_K}）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--out", default=None,
                    help="输出前缀（默认 data/processed/e2e）。"
                         "⚠️ 裸名字会落在【当前目录】，不是 data/processed/ —— 要目录就写全")
    ap.add_argument("--seed", type=int, default=20261013, help="判卷表打乱种子")
    # ★ S5-7：生成侧的开关也要能 A/B —— 但【默认值 = 生产值】。
    #   这就挡住了"评测量 A、生产跑 B"那个老病：想量 B 必须【明说】。
    ap.add_argument("--gen-schema", action=argparse.BooleanOptionalAction,
                    default=GEN_JSON_SCHEMA,
                    help="用 JSON Schema 约束解码（默认跟 config；--no-gen-schema 做对照）")
    # ★ S5-7 §7.1 的实验工具：temperature=0 实测【不可复现】，试试固定 seed 管不管用。
    #   不传 = 现状（不设种子）。传了就进 options。
    ap.add_argument("--gen-seed", type=int, default=None,
                    help="给生成设一个固定 seed（试能不能压住跨运行抖动）")
    args = ap.parse_args()

    if args.gen_seed is not None:
        GEN_OPTIONS["seed"] = args.gen_seed

    arms = [x.strip() for x in args.arms.split(",") if x.strip()]

    eval_set = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = eval_set["queries"]
    if args.limit:
        queries = queries[:args.limit]
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    vectors, ids = load(model=EMB_MODEL)

    out_prefix = Path(args.out) if args.out else (PROCESSED_DIR / "e2e")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"评测集: {len(queries)} 题 · 语料: {len(chunks)} 条 · k={args.k} · 生成: {GEN_MODEL}")
    print(f"arm: {', '.join(arms)}    输出前缀: {out_prefix}")
    print(f"JSON schema 约束: {args.gen_schema}"
          f"{'  ← = config.GEN_JSON_SCHEMA（生产值）' if args.gen_schema == GEN_JSON_SCHEMA else '  ⚠️ 与生产不同（对照组）'}")

    client = make_client()

    # oracle 的 key 必须是 ask() 实际送进去的那个 query（扩展后的）。
    # ★ 这里用 expand_query 现算，而不是复用 ask() 内部的 ——
    #   ask() 的契约是「query = expand_query(question) if expand」，
    #   同一份实现，所以算出来必然一致。不一致的话 OracleRetriever 会炸。
    targets_map = {expand_query(item["q"]): item["targets"] for item in queries}

    retrievers = build_arms(arms, client, chunks, vectors, ids, targets_map, args.k)
    for a, rt in retrievers.items():
        print(f"  [{a}] {rt.name}")

    rows_by_arm: dict[str, list[dict]] = {}
    for a in arms:
        rows_by_arm[a] = run_arm(
            a, retrievers[a], queries, client, chunks, vectors, ids,
            k=args.k, limit=0, gen_schema=args.gen_schema,
        )
        # 每跑完一个 arm 就落一次盘 —— 后面的 arm 挂了，前面的还在。
        (out_prefix.parent / f"{out_prefix.name}_results.json").write_text(
            json.dumps(rows_by_arm, ensure_ascii=False, indent=2), encoding="utf-8")

    summs = {a: summarize(rows, len(queries)) for a, rows in rows_by_arm.items()}
    print_table(summs)
    print_noise_floor(rows_by_arm, summs)
    print_citation_taxonomy(rows_by_arm, {c["chunk_id"]: c for c in chunks})

    write_judge_sheet(rows_by_arm, queries,
                      out_prefix.parent / f"{out_prefix.name}_judge.md",
                      out_prefix.parent / f"{out_prefix.name}_key.json",
                      seed=args.seed)
    print(f"\n判卷表: {out_prefix.parent / (out_prefix.name + '_judge.md')}")
    print(f"揭盲表: {out_prefix.parent / (out_prefix.name + '_key.json')}  ← 判完再看")
    print(f"原始结果: {out_prefix.parent / (out_prefix.name + '_results.json')}")
    # ★ 判卷是【人】做的（脚本分不出"材料里其实有答案"和"模型编了"），
    #   但数格子和配对检验是【机器】做的。别让人去做机器能做的部分 ——
    #   84 个格子攒出来的注意力，不该在最后一步换成一句"大概对吧"。
    print(f"★ 填完判卷表后算分: ./.venv/Scripts/python.exe scripts/score_judge.py "
          f"{(out_prefix.parent / (out_prefix.name + '_judge.md')).as_posix()}")

    # 汇总也落一份，方便直接贴进阶段记录。
    (out_prefix.parent / f"{out_prefix.name}_summary.json").write_text(
        json.dumps(summs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
