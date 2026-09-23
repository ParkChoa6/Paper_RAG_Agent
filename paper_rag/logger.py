"""日志配置。

本模块只提供 setup()，被 import 时不产生任何副作用 ——
调用它的责任在入口脚本（scripts/）。
"""

import logging
import sys

from paper_rag.config import DATA_DIR, LOG_FILE, LOG_LEVEL


def setup(level=None):
    """配置根 logger，同时输出到控制台和文件。

    参数:
        level: 日志级别字符串，如 "DEBUG" / "INFO"。不传则用 config.LOG_LEVEL。

    返回:
        None
    """
    level = level or LOG_LEVEL
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # 建目录，调用时才执行

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)-7s] %(name)s |  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )
