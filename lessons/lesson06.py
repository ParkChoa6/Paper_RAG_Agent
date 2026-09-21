import json


def save_chunks(chunks, path):
    """把 chunks 存成 JSON 文件。

    参数:
        chunks: list[dict] —— 每个 dict 是一个 chunk
        path: 输出文件路径（str）

    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_chunks(path):
    """从 JSON 文件读回 chunks。

    参数:
        path: JSON 文件路径（str）

    返回:
        list[dict]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
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
    ]
    save_chunks(chunks, "./chunks.json")
    loaded = load_chunks("./chunks.json")  # 读一次，存起来
    print(loaded)
    print(type(loaded))
    print(type(loaded[0]))
