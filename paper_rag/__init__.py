"""Paper RAG Agent —— 论文检索增强问答。

分层（依赖方向只能从上往下）：

    config / llm     基础设施 —— 谁都可以用，它自己不依赖任何模块
    ingest           文档接入 —— PDF 变成 chunks
    index            向量化与检索 —— chunks 与向量之间的双向转换
    generator        生成与引用 —— chunks + query 变成带引用的答案

scripts/ 里的入口脚本负责把这些串起来；这个包里的文件只提供定义，
不在被 import 的时候产生任何副作用。
"""
