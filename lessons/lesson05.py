class PDFLoadError(Exception):
    """PDF 加载失败。"""


def check_pdf_name(path):
    """检查路径名是否以 .pdf 结尾。

    参数:
        path: 文件路径(str)

    返回:
        path 本身

    异常:
        PDFLoadError —— path 不以 ".pdf" 结尾
    """
    if not path.lower().endswith(".pdf"):
        raise PDFLoadError(f"不是PDF文件: {path}")
    return path


if __name__ == "__main__":
    print(check_pdf_name("paper.pdf"))
    try:
        check_pdf_name("paper.doc")
    except PDFLoadError as e:
        print(e)
