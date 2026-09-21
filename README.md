# Paper RAG Agent

论文检索增强问答：把 PDF 论文切块、向量化，提问时检索相关段落，
让本地大模型基于这些段落回答，并给出来源引用。

## 环境要求

- Python 3.13+
- [Ollama](https://ollama.com) 已安装并运行

## 快速开始

### 1. 拉模型

    ollama pull qwen2:7b
    ollama pull quentinz/bge-large-zh-v1.5:latest

### 2. 建虚拟环境

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1        # Windows
    # source .venv/bin/activate          # Linux/macOS

### 3. 装依赖

    pip install -r requirements.txt

### 4. 改配置 ← 唯一需要你确认的地方

打开 `paper_rag/config.py`，确认 `GEN_MODEL` 和 `EMB_MODEL`
跟你 `ollama list` 里的模型名一致。

### 5. 自检

    python check_env.py

四项全过，环境就绪。

## 目录结构

（简短说明 paper_rag / scripts / data / docs / lessons 各是干什么的）

## 样例数据

`data/raw/` 里的样例论文来自 XXX，许可：XXX。

## 开发记录

见 `docs/LEARNING_LOG.md`。
