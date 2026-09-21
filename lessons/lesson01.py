
# chunk = {"text": "注意力机制的核心是...", "page": 3}
# print(chunk["text"])   # 注意力机制的核心是...
# print(chunk["page"])   # 3

chunks = [
    {"text": "第一段...", "paper_id": "paper_001","page_start": 1, "page_end": 2,"section": "section1"},
    {"text": "第二段...", "paper_id": "paper_002","page_start": 2, "page_end": 3,"section": "section2"},
    {"text": "第三段...", "paper_id": "paper_003","page_start": 2, "page_end": 2,"section": "section5"},
    {"text": "第四段...", "paper_id": "paper_004","page_start": 3, "page_end": 3,"section": "section6"},
]



total_length = 0
for chunk in chunks:  # 迭代 chunks
    
    page_range = (
    f"第{chunk['page_start']}页"
    if chunk['page_start'] == chunk['page_end']
    else f"第{chunk['page_start']}-{chunk['page_end']}页")
    
    print(f"[{chunk['paper_id']}] {page_range} * {chunk['section']} --- {chunk['text']}")
    total_length += len(chunk["text"])
print(f"Summary text length: {total_length}")