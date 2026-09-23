import unicodedata
import re
from paper_rag.config import (
    PAGE_MARGIN_X,
    PAGE_HEADER_Y,
    PAGE_FOOTER_Y,
    HEADING_SIZE_GAP,
    MIN_CHUNK_CHARS,
)
from paper_rag.models import Chunk

# PyMuPDF 的 span["flags"] 是个位域，bit 4 表示粗体。
# 这是 PyMuPDF 的语义，不是本项目的配置，所以留在这里而不是 config.py。
_BOLD: int = 1 << 4  # = 16

# 同一字号在 PDF 里的浮点抖动上限（10.0 可能存成 9.9 / 10.1）。
_SIZE_JITTER: float = 0.3


def clean_text(text: str) -> str:
    """清洗单个块的文本。

    做三件事，顺序不能变：
      1. NFKC 规范化 —— 连字 ﬁ 变 fi，全角变半角
      2. 接合行尾断词 —— "down-\\nstream" 变 "downstream"
         （行内的连字符，如 retrieval-augmented，不受影响）
      3. 压掉排版换行 —— 剩下的 "\\n" 换成空格
         （块内的换行是"一行写不下"，不是"段落结束"）

    参数:
        text: 一个块的原始文本（末尾带 \\n）

    返回:
        清洗后的文本（单行、首尾无空白）
    """

    # 1. NFKC 规范化
    text = unicodedata.normalize("NFKC", text)

    # 2. 接合行尾断词
    text = re.sub(r"-\n([a-zA-Z])", r"\1", text)

    # 3. 压掉排版换行
    text = re.sub(r"\s+", " ", text)

    return text.strip()  # 去掉首尾空白


def block_text(blk: dict) -> str:
    """把一个 dict 块的文本拼成字符串 —— **必须保留行内换行**。

    两层 join，缺一不可：

      内层 "".join    span 之间**不加**任何东西。
                      空格是字符，已经在 span["text"] 里了；
                      你加空格会把 "Retrieval-" 拆成 "Retrieval - "。

      外层 "\\n".join  行与行之间**必须补换行**。
                      get_text("dict") 的 span["text"] **不含换行符** ——
                      换行是结构信息（lines 是个列表），不是字符。
                      不补的话：
                        · "knowledge" + "in" 粘成 "knowledgein"
                        · clean_text 的接断词规则永远等不到 "\\n"，静默失效

    参数:
        blk: 一个 dict 块

    返回:
        块的原始文本，行间带 \\n（还没清洗，交给 clean_text）
    """
    return "\n".join(
        "".join(span["text"] for span in line["spans"]) for line in blk["lines"]
    )


def is_body_block(x0: float, y0: float, x1: float, y1: float) -> bool:
    """判断一个块是不是正文。

    靠坐标剔除三类东西：边栏水印、页眉、页脚。
    阈值从 config 读，不许硬编码。

    参数:
        x0, y0, x1, y1: 块的包围盒坐标（页面单位：点）

    返回:
        True 表示是正文块
    """

    if x0 < PAGE_MARGIN_X:
        return False  # 边栏水印
    if y1 < PAGE_HEADER_Y:
        return False  # 页眉
    if y0 > PAGE_FOOTER_Y:
        return False  # 页脚
    return True


def _size_histogram(blks) -> dict:
    """统计一组 dict 块里「字号 → 字符数」。

    内部工具，find_body_size 和 block_main_size 共用。
    下划线开头的名字是 Python 约定：外部别调，随时可能改。

    参数:
        blks: list[dict]，page.get_text("dict")["blocks"]

    返回:
        {字号: 字符数}
    """
    hist = {}  # 累加器
    for blk in blks:
        if blk["type"] != 0:  # 1 = 图片块，没有 lines
            continue
        for line in blk["lines"]:
            for span in line["spans"]:
                size = round(span["size"], 1)  # 抹掉浮点抖动：9.8999… → 9.9
                hist[size] = hist.get(size, 0) + len(span["text"])
    return hist


def find_body_size(doc) -> float:
    """扫全文，返回「正文字号」—— 覆盖字符数最多的那个字号。

    为什么扫全文而不是看第 1 页：见 config.py 里 PAGE_MARGIN_X 的教训。
    为什么数「字符」而不是数「块」：正文块少但每块上千字，
    脚注/图表标注块多但每块几十字。按块数统计会被小字反超。

    参数:
        doc: pymupdf 打开的文档

    返回:
        float，正文字号（本文 ≈ 10.0）
    """
    hist = {}
    for page in doc:
        page_hist = _size_histogram(page.get_text("dict")["blocks"])
        for size, n in page_hist.items():
            hist[size] = hist.get(size, 0) + n

    return max(hist, key=hist.get)  # 取"字符数最多"的那个字号


def _main_span(blk: dict) -> dict:
    """取一个 dict 块的"主 span"—— 覆盖字符数最多的那个 span。

    为什么不跟 find_body_size 共用一个循环：
    两者问的问题不同。find_body_size 问「全文哪个字号最常见」，
    只按字号分组（不同字体的 10.0 要合并计数）；
    这里问「这个块主要由哪个 span 构成」，要按 (字号,字体,粗细) 一起分组。
    口径不同，不强行合并。

    参数:
        blk: 一个 dict 块

    返回:
        span 字典，含 text / size / font / flags
    """
    tally = {}  # (字号, 字体, flags) → 字符数
    best = None
    best_n = -1

    for line in blk["lines"]:
        for span in line["spans"]:
            key = (round(span["size"], 1), span["font"], span["flags"])
            n = tally.get(key, 0) + len(span["text"])
            tally[key] = n
            if n > best_n:  # 这个 key 反超了，记住它代表的 span
                best_n, best = n, span

    return best


def block_main_size(blk: dict) -> float:
    """取一个 dict 块的主字号。

    参数:
        blk: 一个 dict 块

    返回:
        float，这个块的主字号
    """
    return round(_main_span(blk)["size"], 1)


def is_heading_block(blk: dict, body_size: float) -> bool:
    """判断一个 dict 块是不是小标题。

    两个信号，任一成立即判为标题（OR）：

      信号一 · 字号明显大于正文
          size >= body_size + HEADING_SIZE_GAP
          注意是「明显大于」不是「不等于」—— 全文 6% 的字符是脚注/
          图表标注的 4~9pt 小字，用 != 会把它们全判成标题。

      信号二 · 字号跟正文一样，但加粗
          |size - body_size| <= _SIZE_JITTER 且 flags 带粗体位
          为什么需要它：排版区分层级用「字号 × 字重」两个维度。
          本文的一级标题两个都用（12pt + 粗体），二级标题只用字重
          （10pt + 粗体，字号跟正文一模一样）。只看字号必然漏。

    已知假阳性：论文首页的作者名也是粗体，会被判成标题。
    影响有限 —— 它们紧接着就被 "Abstract" 覆盖掉了。

    参数:
        blk: dict 块（调用方保证 blk["type"] == 0）
        body_size: find_body_size() 的结果

    返回:
        True 表示这是标题块
    """
    main = _main_span(blk)
    size = round(main["size"], 1)

    if size >= body_size + HEADING_SIZE_GAP:
        return True  # 信号一：大字

    if abs(size - body_size) <= _SIZE_JITTER and main["flags"] & _BOLD:
        return True  # 信号二：同字号 + 粗体

    return False


def paragraphs_to_chunks(doc, paper_id, title, source_path) -> list[Chunk]:
    """把整篇 PDF 切成 chunk 列表。

    流程（顺序不能变）：
      对每一页的每一个块：
        1. 跳过图片块（type != 0）
        2. 坐标过滤 → is_body_block()，不是正文就丢
        3. 清洗文本 → clean_text()
        4. 是标题吗？（is_heading_block）
             是 → 更新 current_section，然后 continue（不产出 Chunk！）
             否 → 产出一个 Chunk

    current_section 是跨块、跨页的累加状态。

    参数:
        doc: pymupdf 打开的文档
        paper_id: str，如 "rag-2005.11401"
        title: str，论文标题
        source_path: str，源文件路径

    返回:
        list[Chunk]
    """
    body_size = find_body_size(doc)  # 全文正文字号，只算一次 —— 别放进循环里
    chunks = []  # 产出累加器
    seq = 0  # chunk 序号累加器（这篇论文内的流水号）

    # ═══════════ 空 ① ═══════════
    # current_section 就定义在这一层（函数体里，页循环之外）。
    # 初值给什么？想想：第一个标题出现之前的那些正文块，属于哪一节？
    # 又：Chunk.__init__ 里有 `if not section: raise ValueError("section is empty")`

    current_section = "正文开头"

    for page_no, page in enumerate(doc, start=1):  # start=1 → 页码从 1 开始
        for blk in page.get_text("dict")["blocks"]:
            if blk["type"] != 0:  # 图片块没有 lines
                continue

            x0, y0, x1, y1 = blk["bbox"]
            if not is_body_block(x0, y0, x1, y1):  # 水印 / 页眉 / 页脚
                continue

            # 先把块拼成带换行的字符串，再清洗
            text = clean_text(block_text(blk))

            # ② 标题块 —— 必须排在长度过滤【之前】！
            #    标题天然很短（'Abstract' 8 字、'4 Results' 9 字），
            #    先过滤长度会把它们全杀掉，current_section 再也更新不了。
            #    长度过滤是针对【正文块】的规则，作用域不能套到标题上。
            if is_heading_block(blk, body_size):
                # 标题块：不产出 Chunk，只更新标签
                current_section = text
                continue  # 跳过后面的产出

            # ③ 太短的丢弃 —— 拦的是图里的文字和公式碎片。
            #    这个判断同时挡掉了空串（len("") = 0 < 40）。
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # ═══════════ 空 ③ ═══════════
            seq += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}#{seq}",  # 用 paper_id 和 seq 拼出一个 ID
                    text=text,
                    paper_id=paper_id,
                    title=title,
                    source_path=source_path,
                    page_start=page_no,  # 一个块只在一页上，所以两个页码相同
                    page_end=page_no,
                    section=current_section,  # ← 空① 那个变量
                )
            )

    return chunks
