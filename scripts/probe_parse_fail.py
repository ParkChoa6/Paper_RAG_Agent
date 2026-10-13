"""探针：S5-6 里那 4 次 `parse_failed`，到底是【哪种坏】。

══════════ 为什么要专门探一次 ══════════

S5-6 的 `e2e_results.json` 里 `parse_failed` 那几行，只留下了一句话：

    ValueError: 不是合法 JSON（Expecting ',' delimiter: line 1 column 510）
                。原文：'{"answer": "……", "used": [1], "quotes": {"1": "Knowledge fu'

★ 那 `原文` 是【我们自己截的 200 字符】（见 parse_response 的 raise）。
  错误位置在 **509**，在截断点之外 ——
  **最该看的那一段，恰好没留下来。**

★ 这正是 S5-6 §4.6 那条教训的又一次现身：
  「判定结果」不能替代「原始证据」。第一次是 quotes 没存，
  这一次是 raw 没存全。同一条规矩，隔了一周又踩一次。

所以本探针只做一件事：**把那 4 次重新跑出来，并把全文留下。**

══════════ 它要回答的问题 ══════════

`parse_response` 的 docstring 里预写了一个修法（逐个试 `{` 的位置 + `raw_decode`），
那是给「第一个 `{` 不是 JSON 开头」准备的。**但实测的报错根本不是那一种。**

先看症状，两种猜测：

    Q6          报「模型输出里没有 JSON 对象」 —— 那是 `j < i`，
                即【全文一个 `}` 都没有】。看着像【被截断了】。
    Q13 / Q24   报「Expecting ',' delimiter」在 char 509 / 981 ——
                在字符串【里面】。看着像【引文里有个没转义的 `"`】。

★ 两种猜测导向【完全不同的修法】：
    截断 → 去查 num_predict / 让 quotes 短一点（生成侧）
    转义 → 去写一个容错解析器（解析侧）
  猜错方向 = 白写一段代码，而且新代码会让【另一种】坏得更隐蔽。

══════════ 怎么才能不猜 ══════════

★ Ollama 的响应里【自带证据】，不用推理：

    done_reason   "stop"   → 模型自己停下（输出是完整的）
                  "length" → 撞到了 num_predict 上限，【被砍了】
    eval_count    实际生成了几个 token

  一条 `done_reason == "length"`，比任何间接论证都硬。

★ 还有一条：报错位置那 60 个字符必须打出来。
  未转义的 `"` 就藏在那里，一眼可见 —— 但前提是你得把它打出来。

用法:
    ./.venv/Scripts/python.exe -u scripts/probe_parse_fail.py
    ./.venv/Scripts/python.exe -u scripts/probe_parse_fail.py --repeat 3
    ./.venv/Scripts/python.exe -u scripts/probe_parse_fail.py --arms dense --q 6

前置: Ollama 起着 + 索引存在
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ★ 复用评测脚本里的检索器构造 —— 包括那个作弊的 OracleRetriever。
#   抄一份的话，探针量的就不是评测里坏掉的那一次了
#   （S5-1 / S5-2 各栽过一次"两份实现量不同的东西"）。
import eval_answer as ea

from paper_rag.config import EMB_MODEL, EVAL_DIR, GEN_MODEL, PROCESSED_DIR, TOP_K
from paper_rag.embed import make_client
from paper_rag.generator import build_prompt, parse_response
from paper_rag.index import load
from paper_rag.pipeline import expand_query

LINE = "=" * 82

# 实测坏掉的那 4 次（臂, 题号）—— 题号从 1 开始，和评测表一致
DEFAULT_CASES = [("dense", 6), ("dense", 13), ("dense", 24), ("oracle", 24)]

OPTIONS = {"temperature": 0}

# ★ 候选修法之二：把【契约本身】交给解码器当语法。
#   比裸 "json" 紧得多 —— 裸 json 只要求"是 JSON"，
#   模型仍然可以花一大堆 token 去写我们根本不认识的字段；
#   这份 schema 钉死了三个字段的名字和类型。
GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used": {"type": "array", "items": {"type": "integer"}},
        "quotes": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["answer", "used", "quotes"],
}


def window(text: str, pos: int, w: int = 60) -> str:
    """把报错位置前后各 w 个字符挖出来。★ 未转义的引号就藏在这一段里。"""
    a, b = max(0, pos - w), min(len(text), pos + w)
    return f"...{text[a:pos]}▶{text[pos:b]}..."


def diagnose(raw: str) -> dict:
    """走一遍 parse_response，把失败点定位到字符上。"""
    out = {
        "len_raw": len(raw),
        "n_lbrace": raw.count("{"),
        "n_rbrace": raw.count("}"),
        "n_dquote": raw.count('"'),
        "parses": False,
        "error": None,
        "at": None,
        "window": None,
    }
    try:
        parse_response(raw)
        out["parses"] = True
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    # json.JSONDecodeError 带着 pos —— 那就是病灶的位置
    import json as _json

    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j < i:
        out["at"] = "没有完整的 {...} 外壳（j < i）→ 尾部被砍掉了"
        out["window"] = raw[-160:]
        return out

    payload = raw[i : j + 1]
    try:
        _json.loads(payload)
    except _json.JSONDecodeError as e:
        out["at"] = f"char {e.pos}（payload 内）"
        out["window"] = window(payload, e.pos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="查出 parse_failed 到底是哪种坏")
    ap.add_argument("--arms", default="dense,oracle",
                    help="用哪几个 arm 重跑（dense / oracle；rerank 太慢，默认不用）")
    ap.add_argument("--q", default="", help="题号，逗号分隔（默认用实测坏掉的那 4 次）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每题跑几遍。★ 大于 1 才能看出这坏是【偶发】还是【必现】")
    ap.add_argument("-k", type=int, default=TOP_K)
    ap.add_argument("--out", default=None, help="原始证据落盘路径")
    # ★ 候选修法。Ollama 的 format="json" 是【语法约束解码】：
    #   采样器被 GBNF 语法顶着，物理上【没法】停在一个不合法的 JSON 上 ——
    #   想吐 EOS 就必须先把括号闭上。这正是"模型忘了闭关"这个病的对症药。
    #   （而 docstring 里预写的 raw_decode 修法对这个病【无效】：
    #     没有合法 JSON 可解，缺的是字符，不是切点。）
    ap.add_argument("--format", default="", choices=["", "json", "schema"],
                    help='"json"=裸 JSON 语法；"schema"=按契约的 JSON Schema；留空=现状')
    args = ap.parse_args()

    if args.q:
        idxs = [int(x) for x in args.q.split(",") if x.strip()]
        arms = [x.strip() for x in args.arms.split(",") if x.strip()]
        cases = [(a, n) for a in arms for n in idxs]
    else:
        cases = DEFAULT_CASES

    eval_set = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    queries = eval_set["queries"]
    chunks = json.loads((PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    vectors, ids = load(model=EMB_MODEL)

    client = make_client()

    need = sorted({a for a, _ in cases})
    targets_map = {expand_query(it["q"]): it["targets"] for it in queries}
    retrievers = ea.build_arms(need, client, chunks, vectors, ids, targets_map, args.k)

    print(LINE)
    print(f"重跑 {len(cases)} 个（臂, 题）组合 × {args.repeat} 遍 · 生成: {GEN_MODEL}")
    print(f"format = {args.format or '（不传 = 现状）'}")
    print(LINE)

    records = []
    for arm, n in cases:
        item = queries[n - 1]
        question = item["q"]

        # ★ 完全照 ask() 的路子走，一步都不省 —— 只有这样才能说是"同一次"。
        #   唯一的不同在本行【下面】：绕过 generator.answer()，
        #   直接调 client.generate，为的是拿到 done_reason。
        query = expand_query(question)
        hits = retrievers[arm].search(query, k=args.k)
        by_id = {c["chunk_id"]: c for c in chunks}
        materials = [by_id[cid] for cid, _ in hits]
        prompt = build_prompt(question, materials)

        print(f"\n{'─' * 82}")
        print(f"arm={arm}  第 {n} 题  {question}")
        print(f"  材料数={len(materials)}  prompt 字符数={len(prompt)}  "
              f"材料正文合计={sum(len(m['text']) for m in materials)} 字")

        for rep in range(1, args.repeat + 1):
            kw = {}
            if args.format == "json":
                kw["format"] = "json"
            elif args.format == "schema":
                kw["format"] = GEN_SCHEMA
            # ★ 约束解码会【硬炸】：Ollama 检测到重复就 500。
            #   不接住的话整个探针跑一半就死了，而且看不到已经拿到的证据。
            try:
                resp = client.generate(model=GEN_MODEL, prompt=prompt,
                                       options=OPTIONS, **kw)
            except Exception as e:
                print(f"\n  第 {rep} 遍：")
                print(f"    ❌❌ 连生成都没跑完：{type(e).__name__}: {e}")
                records.append({
                    "arm": arm, "n": n, "q": question, "rep": rep,
                    "done_reason": None, "eval_count": None,
                    "prompt_eval_count": None, "n_materials": len(materials),
                    "prompt_chars": len(prompt), "raw": "",
                    "len_raw": 0, "n_lbrace": 0, "n_rbrace": 0, "n_dquote": 0,
                    "parses": False,
                    "error": f"GENERATE_FAILED {type(e).__name__}: {e}",
                    "at": "连生成都没跑完（约束解码把模型憋死了）",
                    "window": None,
                })
                continue
            raw = resp.response
            d = diagnose(raw)

            print(f"\n  第 {rep} 遍：")
            print(f"    done_reason={getattr(resp, 'done_reason', '?')!r}   "
                  f"eval_count={getattr(resp, 'eval_count', '?')}   "
                  f"prompt_eval_count={getattr(resp, 'prompt_eval_count', '?')}")
            print(f"    输出 {d['len_raw']} 字符 · "
                  f"{{x{d['n_lbrace']} }}x{d['n_rbrace']} \"x{d['n_dquote']}")
            if d["parses"]:
                print("    ✅ 解析通过 —— ★ 这一次【没重现】")
            else:
                print(f"    ❌ {d['error']}")
                print(f"       病灶: {d['at']}")
                if d["window"]:
                    print(f"       {d['window']!r}")

            records.append({
                "arm": arm, "n": n, "q": question, "rep": rep,
                "done_reason": getattr(resp, "done_reason", None),
                "eval_count": getattr(resp, "eval_count", None),
                "prompt_eval_count": getattr(resp, "prompt_eval_count", None),
                "n_materials": len(materials),
                "prompt_chars": len(prompt),
                "raw": raw,           # ★ 全文。这一行就是本探针存在的理由。
                **{k: v for k, v in d.items() if k != "raw"},
            })

    out_path = Path(args.out) if args.out else (
        PROCESSED_DIR / f"probe_parse_fail{'_fmt' + args.format if args.format else ''}.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 汇总：把 4 种"坏"分开数，不合成一个数（S5-6 §4.6 的老规矩）──
    print(f"\n{LINE}")
    print("★ 汇总 —— 坏法分开数")
    print(LINE)
    tally = {}
    lens = {}
    done = {}
    for r in records:
        if r["parses"]:
            key = "✅ 解析通过"
        elif (r["error"] or "").startswith("GENERATE_FAILED"):
            key = "❌❌ 生成阶段就炸了"
        else:
            key = f"❌ {r['error'].split('（')[0][:32]}"
        tally[key] = tally.get(key, 0) + 1
        lens.setdefault(key, []).append(r["eval_count"] or 0)
        done.setdefault(key, set()).add(r["done_reason"])
    for key, cnt in sorted(tally.items(), key=lambda x: -x[1]):
        avg = sum(lens[key]) / len(lens[key])
        print(f"  {key:<34} x{cnt}   eval_count 均值 {avg:.0f}   "
              f"done_reason={sorted(done[key])}")
    print()
    print("★ done_reason='length' ⇒ 被 num_predict 砍了，是【生成侧】的锅。")
    print("★ 病灶窗口里那个未转义的 \" ⇒ 【解析侧】的锅，两种病药不同。")
    print(f"\n原始证据（全文）: {out_path}")


if __name__ == "__main__":
    main()
