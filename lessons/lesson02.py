def format_pages(page_start, page_end):
    if page_start == page_end:
        return f"第{page_start}页"
    else:
        return f"第{page_start}-{page_end}页"


def format_chunk(chunk):
    return f"[{chunk['paper_id']}] {format_pages(chunk['page_start'], 
    chunk['page_end'])} * {chunk['section']} --- {chunk['text']}"


def count_chars(chunks):
    return sum(len(chunk["text"]) for chunk in chunks)


chunks = [
    {
        "text": "第一段...",
        "paper_id": "paper_001",
        "page_start": 1,
        "page_end": 2,
        "section": "section1",
    },
    {
        "text": "第二段...",
        "paper_id": "paper_002",
        "page_start": 2,
        "page_end": 3,
        "section": "section2",
    },
    {
        "text": "第三段...",
        "paper_id": "paper_003",
        "page_start": 2,
        "page_end": 2,
        "section": "section5",
    },
    {
        "text": "第四段...",
        "paper_id": "paper_004",
        "page_start": 3,
        "page_end": 3,
        "section": "section6",
    },
]

for chunk in chunks:  # 迭代 chunks
    print(format_chunk(chunk))

print(f"Summary text length: {count_chars(chunks)}")
