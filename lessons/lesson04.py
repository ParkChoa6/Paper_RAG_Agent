from chunk import Chunk

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
