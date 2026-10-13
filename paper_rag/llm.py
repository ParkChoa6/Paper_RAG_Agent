"""文本生成 —— 文本 → 文本。

和 embed.py 是兄弟，但方向相反：

    embed.py   文本 → 向量    为了【找得到】
    llm.py     文本 → 文本    为了【答得出】

两个都是「薄薄一层 Ollama 封装」—— 不知道 chunk 是什么，
不知道 prompt 该怎么拼，不知道什么叫"引用"。

为什么单独拆成模块（0B-2 判据二）：
    它【被所有人依赖，但它谁都不依赖】。
    generator.py 依赖它，Stage 6 的 Agent 也会依赖它，
    而它只依赖 config 和 ollama。
    业务知识一旦长进这一层，Stage 6 就没法复用它了。
"""

import logging

import ollama

from paper_rag.config import OLLAMA_HOST

logger = logging.getLogger(__name__)


def make_client() -> ollama.Client:
    """建一个 Ollama 客户端。

    返回:
        ollama.Client，指向 config.OLLAMA_HOST

    注意:
        embed.py 里有一个【一模一样】的函数。这是故意的，理由见文件末尾。
    """
    return ollama.Client(host=OLLAMA_HOST)


def generate(
    client,
    model: str,
    prompt: str,
    options: dict | None = None,
    format=None,
) -> str:
    """单轮生成：给一段 prompt，拿一段文本回来。

    参数:
        client:   ollama 客户端
        model:    模型名（config.GEN_MODEL）
        prompt:   完整的 prompt 字符串
        options:  透传给 Ollama 的生成参数，比如
                    {"temperature": 0, "num_predict": 512, "seed": 42}
        format:   ★ S5-7：透传给 Ollama 的 `format` —— 【约束解码】。
                  传一个 JSON Schema（dict）或 "json"（裸语法），
                  采样器就被限死在合法输出上；传 None = 不约束。
                  ⚠️ 参数名就叫 format（和 Ollama 的关键字【同名】），
                     这是故意的：想找"哪次调用约束了解码"，grep format 就够。
                     代价是它在本函数里遮住了内置 format() —— 本函数不用那个。
                  ⚠️ 别传裸 "json"：实测会触发 Ollama 的重复保护直接 500。
                     理由见 config.GEN_JSON_SCHEMA 的注释。

    返回:
        str —— 模型输出的【原文】

    注意:
        这一层不做任何解析、不做任何校验。模型说什么就原样返回什么。
        【判断是上层的事。】
        这一层多写一行"帮它清理一下输出"，你就再也说不清
        到底是模型没按格式答，还是你替它圆过去了。

        ★ format 不违反这条：它不是"替它清理输出"，
          它是【在模型开口之前就限死它的嘴】——
          走过来说的话依然是模型自己说的，一个字没被改。
    """
    resp = client.generate(model=model, prompt=prompt, options=options or {},
                           format=format)
    return resp.response


# ---------------------------------------------------------------------------
# 关于 make_client 的两份拷贝
#
# 你可能想说：DRY —— 重复的代码该抽出去。
#
# 但抽到哪？
#   抽到 embed.py，让 llm.py 去 import 它？
#       → llm 就【假依赖】了 embed。生成压根不关心向量的事。
#         而且 embed 依赖 ollama，llm 依赖 embed，链条白长一节。
#   抽到独立的 client.py？
#       → 为了一行 `ollama.Client(host=...)` 多一个模块。
#         模块本身也是负债：多一个文件、多一次 import、多一个要维护的名字。
#
# 现在的选择是：【先重复，等第三处出现再抽】。
# 如果 Stage 6 造 Agent 的时候又要第三份，那一刻再抽 —— 那时你才知道
# 它到底该住哪、该长什么样。
#
# 这不是懒，是【推迟一个信息不足的决定】。
# 现在抽，你是在猜；等第三处出现，你是在看。
# ---------------------------------------------------------------------------
