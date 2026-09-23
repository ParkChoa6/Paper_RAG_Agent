import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_rag.ingest import clean_text, is_body_block

if __name__ == "__main__":
    # 测试
    print(clean_text("down-\nstream\n"))  # down-stream → downstream
    print(clean_text("ﬁrst\n"))  # 连字 → first
    print(clean_text("Hello\nWorld\n"))  # 换行 → 空格

    # 拿上面输出的真坐标来测：
    print(is_body_block(170.8, 99.8, 441.6, 117.0))  # [0] 标题 → 应该 True
    print(is_body_block(10.9, 212.3, 37.6, 560.0))  # [11] 水印 → 应该 False
