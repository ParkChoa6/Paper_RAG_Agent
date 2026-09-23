"""本领域的数据模型。

这里的东西只描述"数据长什么样"，不做任何计算、不读文件、不调模型。
"""


class Chunk:
    """
    1.page_start / page_end 是 1-based
    2.chunk_id 不承诺跨重建稳定

    """

    def __init__(
        self,
        chunk_id,
        text,  # 段落文本
        paper_id,  # 论文ID
        title,  # 论文标题
        source_path,  # 源文件路径
        page_start,
        page_end,
        section,  # 段落所属的章节
    ):
        # 页码检查
        if not isinstance(page_start, int) or not isinstance(page_end, int):
            raise TypeError("页码必须是整数类型")
        if page_start < 1 or page_end < 1:
            raise ValueError("页码必须为正整数")
        if page_start > page_end:
            raise ValueError(f"起始页 ({page_start}) 不能大于结束页 ({page_end})")
        # 非空检查
        if not text:  # 段落文本不能为空
            raise ValueError("text is empty")
        if not paper_id:  # 论文ID不能为空
            raise ValueError("paper_id is empty")
        if not title:  # 论文标题不能为空
            raise ValueError("title is empty")
        if not source_path:  # 源文件路径不能为空
            raise ValueError("source_path is empty")
        if not section:  # 段落所属的章节不能为空
            raise ValueError("section is empty")
        # 赋值
        self.chunk_id = chunk_id  # 段落ID
        self.text = text
        self.paper_id = paper_id
        self.title = title
        self.source_path = source_path
        self.page_start = page_start
        self.page_end = page_end
        self.section = section

    def pages(self):
        if self.page_start == self.page_end:
            return f"第{self.page_start}页"
        return f"第{self.page_start}-{self.page_end}页"

    def format(self):
        return f"[{self.chunk_id}] {self.title} | {self.pages()} | {self.section} | \n---\n {self.text} \n---"

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "paper_id": self.paper_id,
            "title": self.title,
            "source_path": self.source_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
        }


if __name__ == "__main__":
    c1 = Chunk(
        "1#1",
        "this is a text sample!",
        1,
        "this is title sample",
        "data/raw/sample.pdf",
        1,
        1,
        "section sample",
    )
    c2 = Chunk(
        "1#2",
        "this is a text sample!",
        1,
        "this is title sample",
        "data/raw/sample2.pdf",
        1,
        1,
        "section sample",
    )
    c3 = Chunk(
        "1#3",
        "this is a text sample!",
        1,
        "this is title sample",
        "data/raw/sample3.pdf",
        1,
        1,
        "section sample",
    )
    print(c1.format())
    print(c2.format())
    print(c3.format())
