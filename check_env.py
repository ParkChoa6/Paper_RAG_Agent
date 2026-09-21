"""环境自检脚本 —— 不依赖任何第三方库，装 venv 之前就能跑。

用法：python check_env.py（在任何目录都可以）

它检查四件事：
    1. Python 版本
    2. 中文编码是否正常（这台机器上有坑，见下方注释）
    3. Ollama 服务是否活着、两个模型是否都在
    4. 生成接口和向量接口能不能真的调通
"""

import json
import sys
import urllib.error
import urllib.request
from paper_rag.config import OLLAMA_HOST as OLLAMA, GEN_MODEL, EMB_MODEL

# Windows 上 Python 默认用系统区域编码（简体中文机器上是 gbk）来读写 stdin/stdout。
# 结果就是：程序逻辑全对，打印出来的中文却是乱码。
# 强制切到 utf-8，之后的输出才可信。
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")


def post(path, payload):
    """向 Ollama 发一个 POST 请求，返回解析后的 JSON。"""
    body = json.dumps(payload).encode("utf-8")  # 显式 utf-8，不靠默认值
    req = urllib.request.Request(
        OLLAMA + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=" * 52)
    print("1) Python")
    print("=" * 52)
    print(f"版本   : {sys.version.split()[0]}")
    print(f"解释器 : {sys.executable}")
    print(f"输出编码: {sys.stdout.encoding}")

    print()
    print("=" * 52)
    print("2) 中文编码")
    print("=" * 52)
    print("中文测试：你好，世界 —— 如果你看到的是方块或问号，编码没修好")

    print()
    print("=" * 52)
    print("3) Ollama 服务与模型")
    print("=" * 52)
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"[失败] 连不上 Ollama（{e}）")
        print("       去开始菜单启动 Ollama，或运行 ollama app.exe")
        return 1

    names = [m["name"] for m in tags.get("models", [])]
    print(f"[成功] 服务在线，共 {len(names)} 个模型")
    for n in names:
        print(f"       - {n}")

    for label, model in (("生成模型", GEN_MODEL), ("向量模型", EMB_MODEL)):
        hit = any(n == model or n.startswith(model.split(":")[0] + ":") for n in names)
        print(
            f"[{'成功' if hit else '失败'}] {label} {model} {'在' if hit else '不在'}"
        )
        if not hit:
            print(f"       补装：ollama pull {model}")

    print()
    print("=" * 52)
    print("4) 接口连通性")
    print("=" * 52)

    try:
        out = post(
            "/api/generate",
            {
                "model": GEN_MODEL,
                "prompt": "只回答两个字：收到",
                "stream": False,
                "options": {"num_predict": 16},
            },
        )
        secs = out.get("total_duration", 0) / 1e9
        print(f"[成功] 生成接口，耗时 {secs:.1f}s")
        print(f"       模型回答：{out.get('response', '').strip()}")
    except Exception as e:
        print(f"[失败] 生成接口：{e}")
        return 1

    try:
        out = post(
            "/api/embeddings", {"model": EMB_MODEL, "prompt": "这是一段测试文本"}
        )
        emb = out.get("embedding", [])
        print(f"[成功] 向量接口，维度 {len(emb)}")
        print(f"       单条向量占用内存：{len(emb) * 4 / 1024:.1f} KB（float32）")
    except Exception as e:
        print(f"[失败] 向量接口：{e}")
        return 1

    print()
    print("=" * 52)
    print("体检结束 —— 四项全过就可以开始写代码了")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    sys.exit(main())
