import logging
import unicodedata
import re
from paper_rag.config import (
    REFERENCE_HEADINGS,
    FURNITURE_MIN_PAGES,
    FURNITURE_MIN_RATIO,
    GUTTER_SEARCH_LO,
    GUTTER_SEARCH_HI,
    MIN_COLUMN_BLOCKS,
    HEADING_MAX_CHARS,
    STRUCTURAL_HEADINGS,
    MIN_CHUNK_CHARS,
)
from paper_rag.models import Chunk

# 库文件用 __name__（这里等于 "paper_rag.ingest"，正好说明来源）。
# 脚本里不能用它 —— 脚本的 __name__ 永远是 "__main__"，见 build_index.py 的说明。
logger = logging.getLogger(__name__)

# 横排从左到右的文字，line["dir"] 就是这个向量。
# dir 是「文字走向的单位向量」，不是坐标 —— 竖排是 (0, ±1)，RTL 是 (-1, 0)。
_HORIZONTAL: tuple = (1.0, 0.0)
_DIR_TOL: float = 0.01  # 浮点抖动容差

# 学术论文的章节编号：'1' / '1.1' / '1.1.2' / 'A.' / 'IV.'
# ★ 编号后面必须跟一个【字母开头】的词 —— 这一个字符挡掉了大量公式碎片：
#     '2. (3)'            → 后面是括号，不匹配
#     '1 0 vx 0 1 vy ...' → 后面是数字，不匹配
#     '2 (⟨h,r,t⟩+ ...)'  → 后面是括号，不匹配
# 这是学术写作的约定，跟具体哪篇论文、哪种排版都无关。
_HEADING_NUM = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+[A-Za-z]")


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


def is_rotated_block(blk: dict) -> bool:
    """块里的文字被旋转过吗？只要有一行是，整块就算。

    判据是 line["dir"] —— 文字走向的单位向量，不是坐标：
        横排从左到右 → (1, 0)     竖排 → (0, ±1)     RTL → (-1, 0)
    ★ 这个判据和页面尺寸无关，这正是它比原来那条 `x0 < PAGE_MARGIN_X` 强的地方。

    它替掉的正是原来想抓的「边栏水印」：
      实测 KG Embedding 那篇有 333 个旋转块，全部是图 1 时间轴的竖排标签
      （'2013'、'• TransE[7]'，块宽 5~7 点）—— 该丢的；
      而两篇论文的【所有正文块】dir 都是 (1, 0) —— 不会误伤。
      KG Reasoning 那篇旋转块数为 0，也就是这个判据在它身上不产生任何作用。

    参数:
        blk: dict 块（调用方保证 blk["type"] == 0，有 lines）

    返回:
        True 表示这个块的文字被旋转/竖排过
    """
    for line in blk["lines"]:
        dx, dy = line["dir"]
        if abs(dx - _HORIZONTAL[0]) > _DIR_TOL or abs(dy - _HORIZONTAL[1]) > _DIR_TOL:
            return True
    return False


_DIGITS = re.compile(r"\d+")


def _norm_for_repeat(text: str) -> str:
    """把文本归一化成「跨页重复」的判据形态：数字全换成 #，转小写。

    去数字不是顺手，是必须的 —— 页眉最常见的差异就是页码/卷号：
        '9456 IEEE TRANSACTIONS ON PATTERN ANALYSIS ...'
        '9458 IEEE TRANSACTIONS ON PATTERN ANALYSIS ...'
    归一化后是同一个串，才数得出「出现在 12/23 页」。
    不归一化的话，每一条都只出现 1 次，一条家具都找不出来。

    参数:
        text: clean_text() 之后的文本

    返回:
        归一化后的字符串（只用于比较，不用于展示）
    """
    return _DIGITS.sub("#", text.lower()).strip()


def find_furniture(doc) -> set:
    """通读全文，找出「版面家具」—— 跨页重复出现的页眉、页脚、授权水印。

    为什么必须通读一遍才知道：
      单看一页，「页眉」和「栏顶第一行」在几何上完全是同一行东西
      （y 相同、都在顶端）。区分它们的唯一依据是【它在别的页面上还出现过没有】——
      这是文档级的信息，任何逐块、逐页的判据都拿不到。

    实测（2026-10-06，两篇 KG 论文）：
      kg-reasoning 23 页 → 3 条：页脚 'Authorized licensed use...' 23/23、
                          奇数页书眉 12/23、偶数页书眉 11/23（12+11=23，正好互补）
      kg-embedding 51 页 → 2 条，各 25/51
    对比之前按 y 区间丢：kg-embedding 结果完全一致，
    kg-reasoning 少丢 1010 字符 —— 而那 1010 字符正是被 y 区间误杀的栏顶标题。

    参数:
        doc: pymupdf 打开的文档

    返回:
        set[str]，归一化后的家具文本。调用方用 _norm_for_repeat(文本) 去查。
    """
    pages_of = {}  # 归一化文本 → 出现在哪些页（set）
    for page_no, page in enumerate(doc, start=1):
        for blk in page.get_text("dict")["blocks"]:
            if blk["type"] != 0 or is_rotated_block(blk):
                continue
            text = clean_text(block_text(blk))
            if not text:
                continue
            pages_of.setdefault(_norm_for_repeat(text), set()).add(page_no)

    n_pages = doc.page_count
    return {
        key
        for key, pages in pages_of.items()
        if len(pages) >= FURNITURE_MIN_PAGES
        and len(pages) >= FURNITURE_MIN_RATIO * n_pages
    }


# ★★ 2026-10-06 这里原本有四个函数，全部删除 ★★
#       _size_histogram / find_body_size / _main_span / block_main_size
#   它们服务的是旧的「按字号+字重判标题」那套机制。那套机制被换掉之后
#   这四个就成了死代码 —— 而死代码最坏的地方不是占地方，
#   是它读起来像还在生效的东西，下次改这里的人会以为字号判据还在。
#
#   旧机制为什么被换掉，见 is_heading_block 的说明。


def is_heading_block(text: str) -> bool:
    """判断一段文本是不是小节标题 —— **纯看文本，不看任何排版特征**。

    两个判据，任一成立即判为标题（OR）：

      信号一 · 学术编号
          '1 Introduction' / '2.1 Distance-based Models'
          'A. Task Formulation' / 'IV. TEMPORAL KGR MODEL'
          见 _HEADING_NUM。

      信号二 · 结构词
          'Abstract' 'References' 'Acknowledgments' 'Appendix' …
          见 config.STRUCTURAL_HEADINGS。
          这一类没有编号，但它们在学术写作里是固定的结构标记。

    ★ 为什么把「字号 / 字体 / 粗体」整套判据都扔掉了：
      实测两篇 KG 论文（2026-10-06），旧机制两个信号对它们双双失效 ——
        · kg-reasoning 用【斜体】标标题（Times-Italic），字号和正文一样(10.0)，
          一级标题 'I. INTRODUCTION' 甚至【比正文小】(8.0pt)。两个信号都不响，
          全篇 502 个 chunk 的 section 全是作者列表。
        · kg-embedding 的二级标题是 9.0pt，跟正文(10.0)差 1.0，
          超过容差 _SIZE_JITTER=0.3，全部漏掉；
          同时数学字体 CMBX10 报 bold，公式碎片被误判成标题反过来污染 section。
      病根和 PAGE_MARGIN_X 那几次是同一个：
      【排版特征是每篇论文自己的约定，换个出版社就失效】。
      而编号和结构词是【学术写作的约定】，跨论文稳定。

    长度上限 HEADING_MAX_CHARS 的作用，是把「独立成行的短标题」和
    「恰好以编号开头的长段落首行」分开。实测真标题最长 97 字符，
    误判样本最短 134 字符，120 卡在中间。

    已知假阳性（实测 63 个命中里 4 个）：公式/图表的碎块恰好以
    「数字 空格 字母」开头，如 '1 Shy'、'0 Or,2 . . . 0'。
    代价有限 —— 它们本来也会被 MIN_CHUNK_CHARS 丢掉，
    只是会把紧随其后的 chunk 的 section 污染成这几个字符串。

    参数:
        text: clean_text() 之后的块文本

    返回:
        True 表示是标题块
    """
    if len(text) > HEADING_MAX_CHARS:
        return False  # 太长，不可能是标题行

    if _HEADING_NUM.match(text):
        return True  # 信号一：学术编号

    # 归一化后再比：统一大小写、去掉尾部的句点（'REFERENCES.' → 'references'）
    return _norm_for_repeat(text).strip(".") in STRUCTURAL_HEADINGS  # 信号二：结构词


def find_column_split(blocks: list, page_width: float):
    """找这一页的栏间空隙，返回分界 x；单栏返回 None。

    做法：在页面中部【逐点扫竖线】，数有多少块横跨它。
    双栏排版在栏间必然留出一段没有块横跨的空白 —— 那就是空隙。
    反过来，整页宽的图/表那种页面，中部的每条竖线都有块跨过去 → 找不到空隙
    → 正确地判成单栏。这就是「按页判定」的意思：同一篇论文可以有的页双栏、
    有的页单栏（KG Embedding 的 p4/p5/p15 就是整页大图），逐页判才都对。

    为什么扫的是 30%~70% 这一段：
      空隙只可能在页面中部。往两边扩没意义 —— 页边距也是"没有块横跨"的空白，
      而且比栏间空隙宽得多，一扩进来就会被选成"最长空白段"。

    参数:
        blocks: 已经过滤完的正文块（同一页）
        page_width: 当页宽度（点）

    返回:
        float 分界 x；或 None 表示按单栏处理
    """
    # 至少得能分出两个像样的栏，否则不折腾
    if len(blocks) < 2 * MIN_COLUMN_BLOCKS:
        return None

    lo = int(page_width * GUTTER_SEARCH_LO)
    hi = int(page_width * GUTTER_SEARCH_HI)
    if hi - lo < 2:
        return None  # 页面窄到没地方扫，别硬判

    # 每个 x 上，是否有块横跨。全都没有 → 这个 x 在空隙里。
    free_x = [
        x
        for x in range(lo, hi)
        if all(not (b["bbox"][0] < x < b["bbox"][2]) for b in blocks)
    ]
    if not free_x:
        return None  # 中部每条竖线都有块跨过 → 不是双栏

    # 取【最长的一段连续空白】的中点。
    # 为什么要连续：栏间空隙是一段有宽度的空白。若某页恰好有两处零星空隙，
    # 取最长的那个更可能是真正的栏间；取全部空白的首尾中点会被边距带偏。
    best = cur = [free_x[0]]
    for x in free_x[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [x]
    if len(cur) > len(best):
        best = cur

    split = (best[0] + best[-1]) / 2

    # 左右两边都得有足够的块 —— 防止「所有块都挤在左边、右边空一大片」
    # 被当成双栏（那其实是单栏 + 大片页边距）。
    left = sum(1 for b in blocks if b["bbox"][2] <= split)
    right = sum(1 for b in blocks if b["bbox"][0] >= split)
    if min(left, right) < MIN_COLUMN_BLOCKS:
        return None

    return split


def order_blocks(blocks: list, split) -> list:
    """把一页的块排成【人读的顺序】。

    为什么需要它：PDF 里块的先后是【生成 PDF 时的写入顺序】，不是阅读顺序。
    双栏论文尤其明显 —— 可能出现「左栏第 3 段」后面直接接「右栏第 1 段」。

    单栏（split is None）：
        按 (y, x) 从上到下、同一行从左到右。

    双栏：
        栏优先 —— 左栏读完再读右栏。
        但【横跨栏间空隙的块】（整页宽的图、表、跨栏公式）会把页面切成若干条横带，
        跨栏块排在它所属的那条带之后：
            带1左栏 → 带1右栏 → 跨栏块 → 带2左栏 → 带2右栏 → …
        这是学术排版的标准阅读顺序（LaTeX 的 figure* 就是这个语义）。

    参数:
        blocks: 已经过滤完的正文块（同一页）
        split: find_column_split() 的结果

    返回:
        排好序的块列表
    """

    def key(b):
        return (b["bbox"][1], b["bbox"][0])  # (上边界 y, 左边界 x)

    if split is None:
        return sorted(blocks, key=key)

    spans = sorted((b for b in blocks if b["bbox"][0] < split < b["bbox"][2]), key=key)
    left = [b for b in blocks if b["bbox"][2] <= split]
    right = [b for b in blocks if b["bbox"][0] >= split]

    out = []
    prev = float("-inf")  # 上一条带的底
    for s in spans:
        cut = s["bbox"][1]  # 跨栏块的上边界 = 这条带的底
        out += sorted((b for b in left if prev <= b["bbox"][1] < cut), key=key)
        out += sorted((b for b in right if prev <= b["bbox"][1] < cut), key=key)
        out.append(s)
        prev = cut

    # 最后一条带：最后一个跨栏块之下
    out += sorted((b for b in left if b["bbox"][1] >= prev), key=key)
    out += sorted((b for b in right if b["bbox"][1] >= prev), key=key)
    return out


def paragraphs_to_chunks(doc, paper_id, title, source_path) -> list[Chunk]:
    """把整篇 PDF 切成 chunk 列表。

    流程（顺序不能变）：
      开头：通读全文一遍 → find_furniture()，得到「版面家具」的文本集合。
            这一步必须在页循环【之前】—— 它要的是文档级视野，见该函数说明。
      然后对每一页：
        1. 收齐文字块：跳过图片块（type != 0）、旋转块、以及命中家具的块
        2. 检测分栏 → find_column_split()
        3. 排成阅读顺序 → order_blocks()
        4. 按顺序逐块：
             清洗文本 → clean_text()
             是标题吗？（is_heading_block）
               是 → 更新 current_section，然后 continue（不产出 Chunk！）
               否 → 太短就丢（MIN_CHUNK_CHARS），否则产出一个 Chunk

    ★ 为什么 2、3 必须排在 4 之前：
      current_section 是【沿阅读顺序】推进的状态。顺序错了，
      一个 chunk 会被打上它后面那一节的标签 —— section 元数据就废了。
      而 section 在「绝不砍」清单上，Stage 3 的 prompt 和 Stage 4 的引用都要用它。

    current_section 是跨块、跨页的累加状态。

    参数:
        doc: pymupdf 打开的文档
        paper_id: str，如 "kg-embedding-overview"
        title: str，论文标题
        source_path: str，源文件路径

    返回:
        list[Chunk]
    """
    furniture = find_furniture(doc)  # 全文版面家具 —— 它要通读全文，见该函数说明
    chunks = []  # 产出累加器
    seq = 0  # chunk 序号累加器（这篇论文内的流水号）
    n_reference = 0  # 被丢掉的参考文献块数，最后记一条日志

    # ═══════════ 空 ① ═══════════
    # current_section 就定义在这一层（函数体里，页循环之外）。
    # 初值给什么？想想：第一个标题出现之前的那些正文块，属于哪一节？
    # 又：Chunk.__init__ 里有 `if not section: raise ValueError("section is empty")`

    current_section = "正文开头"

    for page_no, page in enumerate(doc, start=1):  # start=1 → 页码从 1 开始
        # ① 先把这一页的正文块收齐 —— 分栏检测要看【全页】的块分布，
        #    边收边处理是看不出栏间空隙的。
        #    （这里和下面循环里各算了一次 clean_text，是重复劳动。留着，
        #      因为干净；真正的开销在 get_text("dict") 解 PDF，那个只跑一次。）
        blocks = []
        for blk in page.get_text("dict")["blocks"]:
            if blk["type"] != 0:  # 图片块没有 lines
                continue
            if is_rotated_block(blk):  # 竖排水印 / 图里的竖轴标签
                continue
            text = clean_text(block_text(blk))
            if not text or _norm_for_repeat(text) in furniture:
                continue  # 空块 / 跨页重复的页眉页脚
            blocks.append(blk)

        # ② 这一页是单栏还是双栏？（按页判 —— 同一篇里两种都可能出现）
        # ③ 排成阅读顺序
        for blk in order_blocks(blocks, find_column_split(blocks, page.rect.width)):
            # 先把块拼成带换行的字符串，再清洗
            text = clean_text(block_text(blk))

            # ② 标题块 —— 必须排在长度过滤【之前】！
            #    标题天然很短（'Abstract' 8 字、'4 Results' 9 字），
            #    先过滤长度会把它们全杀掉，current_section 再也更新不了。
            #    长度过滤是针对【正文块】的规则，作用域不能套到标题上。
            if is_heading_block(text):
                # 标题块：不产出 Chunk，只更新标签
                current_section = text
                continue  # 跳过后面的产出

            # ③ 太短的丢弃 —— 拦的是图里的文字和公式碎片。
            #    这个判断同时挡掉了空串（len("") = 0 < 40）。
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # ④ 参考文献条目丢弃（全库 53%）。
            #    判据挂在 section 上，不是「遇到 REFERENCES 就停止」——
            #    后者在参考文献不位于文末时会静默截断后面整篇。
            #    计数并记日志：一次丢掉几百块，必须让人看得见，不能静默。
            if _norm_for_repeat(current_section).strip(".") in REFERENCE_HEADINGS:
                n_reference += 1
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

    if n_reference:
        logger.info("  丢弃 %d 个参考文献块（占产出的 %.0f%%）",
                    n_reference, n_reference / (len(chunks) + n_reference) * 100)

    return chunks
