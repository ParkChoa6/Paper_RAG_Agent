class Chunk:
    def __init__(self, text, paper_id, page_start, page_end, section):
        self.text = text
        self.paper_id = paper_id
        self.page_start = page_start
        self.page_end = page_end
        self.section = section

    def pages(self):
        if self.page_start == self.page_end:
            return f"第{self.page_start}页"
        return f"第{self.page_start}-{self.page_end}页"

    def format(self):
        return f"[{self.paper_id}] {self.pages()} * {self.section} --- {self.text}"

    def char_count(self):
        return len(self.text)


if __name__ == "__main__":

    list_of_chunks = [
        Chunk("第一段...", "paper_001", 1, 2, "section1"),
        Chunk("第二段...", "paper_002", 2, 3, "section2"),
        Chunk("第三段...", "paper_003", 2, 2, "section5"),
        Chunk("第四段...", "paper_004", 3, 3, "section6"),
    ]

    for chunk in list_of_chunks:
        print(chunk.format())
    print(f"Summary text length: {sum(chunk.char_count() for chunk in list_of_chunks)}")
