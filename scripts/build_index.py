"""Stage 1 入口：把 data/raw/ 下的 PDF 全部切成 chunks，落盘成 JSON。

这是「离线阶段」的入口。跑一次，之后 Stage 2/3/4 都直接读这个 JSON，
不用再解析 PDF。

用法:
    python scripts/build_index.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from paper_rag.config import LOG_FILE, PROCESSED_DIR, RAW_DIR
from paper_rag.ingest import paragraphs_to_chunks
from paper_rag.logger import setup

# 脚本的 __name__ 永远是 "__main__"，不管文件叫 build_index.py 还是别的。
# 写 getLogger(__name__) 的话，日志里每条都印 "__main__"，分不清来源。
# 所以脚本里显式起名；库文件里才用 __name__（那时它等于模块路径，正好合适）。
logger = logging.getLogger("paper_rag.build_index")

# ---- 论文元信息表 ----
# 文件名 → (paper_id, 标题)
# 为什么写死：paper_id 和标题是「人对论文的命名」，不是 PDF 里能可靠读出来的。
# 文件里那个标题块可能是两行、可能带脚注符号，提取出来不一定干净。
# 手工登记一张表，最可控。论文多了再换成一个 papers.yaml。
#
# ★ 2026-10-06 语料重置：RAG 那篇（rag-original-2005.11401.pdf）已从 data/raw/ 移除，
#   语料转向知识图谱（KG）。原来那条登记也一并删掉 ——
#   留着它只会让 build_index 每次打一条 "找不到 xxx，跳过" 的警告，
#   而那种警告看多了就没人看了（这就是"报警疲劳"的起点）。
PAPERS = {
    "Knowledge Graph Embedding An Overview.pdf": (
        "kg-embedding-overview",
        "Knowledge Graph Embedding: An Overview",
    ),
    "A Survey of Knowledge Graph Reasoning on Graph Types Static, Dynamic, and Multi-Modal.pdf": (
        "kg-reasoning-survey",
        "A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal",
    ),
    # ---- 2026-10-12 扩语料 2 篇 → 7 篇 ----
    # 加这 5 条之前踩了一次坑，见 main() 里那段「反方向也要查」。
    "Knowledge Graphs Opportunities and Challenges.pdf": (
        "kg-opportunities-challenges",
        "Knowledge Graphs: Opportunities and Challenges",
    ),
    "Overview of knowledge reasoning for knowledge graph.pdf": (
        "kg-reasoning-overview",
        "Overview of knowledge reasoning for knowledge graph",
    ),
    "Multi-hop interpretable meta learning for few-shot temporal knowledge graph completion.pdf": (
        "tkg-meta-learning",
        "Multi-hop interpretable meta learning for few-shot temporal knowledge graph completion",
    ),
    "Multi-hop reasoning of knowledge chain over fuzzy spatiotemporal knowledge graphs with box embedding.pdf": (
        "tkg-box-embedding",
        "Multi-hop reasoning of knowledge chain over fuzzy spatiotemporal knowledge graphs with box embeddings",
    ),
    "Temporal knowledge graph reasoning based on time layering and fuzzy logic rules.pdf": (
        "tkg-time-layering",
        "Temporal knowledge graph reasoning based on time layering and fuzzy logic rules",
    ),
}


def main() -> None:
    # 必须在最前面：setup() 配的是「根 logger」，
    # 配好之后，全项目任何 getLogger(...) 发的消息都会流向控制台 + 文件。
    # 晚于第一条日志调用就没用了 —— basicConfig 只在根 logger 还没有 handler 时生效。
    setup()

    # 用 %-占位而不是 f-string：字符串只在真的要输出时才拼。
    # 把级别调到 WARNING 后，这些 line 的拼接成本是 0。
    logger.info("日志文件 → %s", LOG_FILE)
    logger.info("chunk 输出目录 → %s", PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 反方向也要查，而且是【报错】不是警告 ----
    #
    # 下面的循环只遍历 PAPERS，所以「raw/ 里躺着一个没登记的 PDF」
    # 这件事它【根本看不到】—— 那篇论文就是不存在，一句提示都没有。
    #
    # ★ 2026-10-12 真踩到了：一次丢了 5 篇 PDF 进 raw/，忘了登记，
    #   重建索引后还是 337 条 / 2 篇，而【日志干干净净】。
    #   注意「找不到 xxx，跳过」那条警告是【另一个方向】的
    #   （登记了但文件没了），它救不了这种情况。
    #
    # 丢论文这件事不会自愈：检索永远找不到它，现象只是「有些问题答不上来」，
    # 而你会以为是模型不行。所以在这里响亮地失败 —— 和 index.build()
    # 那边「宁可抛错也不静默跳过」是同一条规矩。
    known = set(PAPERS)
    loose = sorted(p.name for p in RAW_DIR.glob("*.pdf") if p.name not in known)
    if loose:
        raise RuntimeError(
            f"data/raw/ 里有 {len(loose)} 个 PDF 【没有登记】，它们不会被解析：\n"
            + "\n".join(f"    {n}" for n in loose)
            + "\n\n补进 scripts/build_index.py 的 PAPERS 表"
            "（键必须和文件名逐字一致）。\n"
            "如果是有意不放进来，把文件移出 data/raw/。"
        )

    all_chunks = []

    for filename, (paper_id, title) in PAPERS.items():
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            # 原来这里是 print("⚠️ ...") —— 现在不用 emoji 了，
            # 级别本身就是「有多严重」的表达，而且它能被过滤、能被检索。
            logger.warning("找不到 %s，跳过", pdf_path)
            continue

        logger.info("正在解析 %s ...", filename)
        doc = pymupdf.open(pdf_path)

        chunks = paragraphs_to_chunks(
            doc,
            paper_id=paper_id,
            title=title,
            source_path=str(pdf_path.relative_to(Path.cwd())),
        )
        doc.close()

        logger.info("  → %d 个 chunk", len(chunks))
        all_chunks.extend(chunks)

    # ---- 落盘 ----
    # Chunk 对象不能直接 json.dump，得先转成 dict —— 这就是 to_dict() 的用途
    out_path = PROCESSED_DIR / "chunks.json"
    payload = {
        "count": len(all_chunks),
        "chunks": [c.to_dict() for c in all_chunks],
    }

    # ensure_ascii=False：中文不转义成 \uXXXX，文件人能直接读
    # indent=2：格式化，方便肉眼检查
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_kb = out_path.stat().st_size / 1024
    logger.info("写入 %s  (%d chunks, %.0f KB)", out_path, len(all_chunks), size_kb)


if __name__ == "__main__":
    main()
