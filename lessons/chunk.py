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
    print("！！！ chunk.py 的顶层代码被执行了 ！！！")
