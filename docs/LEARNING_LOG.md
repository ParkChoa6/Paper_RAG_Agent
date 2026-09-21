# 学习记录

> 每节课后更新。记录**你实际掌握的内容**、踩过的坑、以及提问时没答上来的地方。
>
> 「待复习」尤其重要 —— 那是下一轮的重点。

---

## 课程列表

| 课 | 标题 | 状态 |
|---|---|---|
| 0A-1 | dict 与 list —— RAG 的数据单元 | ✅ 已完成 |
| 0A-2 | 函数与返回值 | ✅ 已完成 |
| 0A-3 | 类与对象 | ✅ 已完成 |
| 0A-4 | 模块与 import | ✅ 已完成 |
| 0A-5 | 异常处理 | ✅ 已完成 |
| 0A-6 | 文件读写与 JSON | ✅ 已完成 |
| 0A-7 | 虚拟环境与依赖 | ✅ 已完成 |
| 0B-1 | Ollama 命令行跑通 | ✅ 已完成（Day 0） |
| 0B-2 | 项目目录结构设计 | ✅ 已完成 |
| 0B-3 | 配置管理 | ✅ 已完成 |
| 0B-4 | 日志 | ✅ 已完成 |
| 0B-5 | git 与 `.gitignore` | ✅ 已完成 |
| 1 | Stage 1 · 文档接入 | ⬜ 下一步 |

---

## ⚡ 下次从这里继续

**Day 2 ✅ · 0B 全部完成 —— 下次从这里继续：Day 3 · Stage 1 文档接入**

- **0B 五项全过**：Ollama 跑通 ✅ / 目录结构 ✅ / 配置管理 ✅ / 日志 ✅ / git 与 `.gitignore` ✅
- **0B 验收已通过**：克隆到全新目录 `C:\MyTemp\pyra_clone` → 建 venv → 装依赖 → `check_env.py` **四项全过**，全程没改任何代码
- **远程仓库**：`https://github.com/ParkChoa6/Paper_RAG_Agent`（Private）
- **下一步：Stage 1 文档接入**（PDF → chunks）
  1. 用 PyMuPDF 把一页 PDF 读成文字（先跑通单栏；**双栏/公式只记录问题，不解决**）
  2. 切分成 chunks（复用 `lessons/chunk.py` 里 `Chunk` 的字段设计）
  3. **元数据设计**（不许砍 —— 三大支柱之一）paper_id / page_start / page_end / section
  4. 写进 `paper_rag/ingest.py`，入口放 `scripts/build_index.py`
- **已知待办**：`data/processed/` 没有进仓库（空目录 git 不跟踪）—— 由**程序自己 mkdir**，不加 `.gitkeep`（理由见 0B-5 记录）
- **七天冲刺进度**：Day 1 ✅ → Day 2 ✅（0A-7 + 0B 全部）→ **Day 3：Stage 1**

---

## Day 0 · 环境体检（2026-09-20）

七天冲刺正式开始前的一次体检。产出：`check_env.py`（项目根目录，随时可重跑）。

### 通过的项

| 项 | 结果 |
|---|---|
| Python | 3.13.9 ✅ |
| GPU | RTX 4070 SUPER 12GB / 驱动 610.88 ✅ |
| 磁盘 | C: 剩余 1079 GB ✅ |
| Ollama | 0.33.2，服务已启动 ✅ |
| 模型 | `qwen2:7b`（4.4GB）、`quentinz/bge-large-zh-v1.5`（651MB）都在 ✅ |
| 生成接口 | `/api/generate` 通，0.2s ✅ |
| 向量接口 | `/api/embeddings` 通，**维度 1024** ✅ |

### 体检查出的三个坑

**坑 1 · 默认解释器是 Anaconda base，且 pip 装在用户目录**

- `python` → `C:\WorkAndStudy\anaconda\python.exe`
- `python -m pip` → 却来自 `C:\Users\蒋进平\AppData\Roaming\Python\Python313\site-packages`

两个不同的地方。这意味着现在 `pip install` 装到哪，取决于调用方式 —— **这是 0A-7「虚拟环境与依赖」要解决的核心问题**，不是现在改，是留到那节课当作真实案例。

**坑 2 · Windows 上 Python 默认用 gbk 读写中文**

- 实测：`sys.stdin.encoding == 'gbk'`，`sys.stdout.encoding == 'gbk'`
- 后果：**程序逻辑全对，打印出来的中文是乱码**
- 这个坑在 Stage 1（解析中文论文）和 Stage 3（打印中文回答）一定会再撞上
- 两种解法：环境变量 `PYTHONUTF8=1`（一劳永逸）或代码里 `sys.stdout.reconfigure(encoding="utf-8")`（`check_env.py` 里用的是后者）

**坑 3 · `qwen2:7b` 的世界知识不可靠 —— 这是今天最有价值的发现**

用干净的 UTF-8 通道问了两次，两次都错，而且错得还不一样：

| 提问 | 回答 |
|---|---|
| 什么是RAG？ | "RAG是**风险评估矩阵**（Risk and Assessment Matrix）" |
| 什么是RAG？（换个问法） | "RAG的全称是**Risk, Availability, and Governance**" |

**但这不是坏消息，恰恰相反** —— 它精确地解释了 RAG 为什么存在：

> 不要把希望寄托在模型的世界知识上。**事实必须由你从检索里喂给它**，模型只负责组织和表达。

同时它给两条设计结论：

- Stage 3 的**拒答不能靠模型自觉**，必须在 prompt 里强制约束
- Stage 4 的**引用校验是刚需** —— 这个模型会一本正经地编造，且语气毫无破绽（两次都答得很自信）

### 意外的好消息

探查指令遵循能力时，要求「严格只输出 JSON、不要 markdown 代码块」：

```
{"answer": "你的回答", "citations": [1, 2]}
```

**干净通过**。说明 Stage 4 的「结构化引用」在这个模型上可行 —— 原计划里这一块的风险解除了。

拒答能力测试也通过（喂了强指令时会回答"资料未提及"），但注意：**这是被模板逼出来的，不是它自觉**。真实 prompt 怎么设计，是 Stage 3 的正课。

---

## 记录模板

```markdown
## 课程 X-Y · 标题

**日期**：YYYY-MM-DD
**产出**：`lessons/xxx.py`

### 学到什么
-

### 踩过的坑
-

### 待复习
-

### 验收
- [ ] 通过 / 未通过
```

---

## 课程 0A-1 · dict 与 list —— RAG 的数据单元

**日期**：2026-09-18
**产出**：`lessons/lesson01.py`

### 学到什么

- **`list[dict]` 是 RAG 的原子结构**：用 dict 描述一个实体（一段文本 + 它的出身），用 list 装一串
- **元数据设计**：`paper_id` / `page_start` / `page_end` / `section` —— 这是 Stage 4「引用原文」能实现的唯一前提
- **f-string 引号嵌套**：外层双引号、内层单引号
- **累加器模式**：初值写在循环外，累加写在循环内
- **`d[k]` vs `d.get(k)`**：前者缺键抛 `KeyError`，后者返回 `None`
  →「**让它炸** 还是 **静默兜底**，是一个主动的设计选择」，不是随手挑一个
- **三类错误的区别**：

  | 错误 | 何时发生 | 怪谁 |
  |---|---|---|
  | `SyntaxError` | 代码解析时，一行都没跑 | 代码写错了 |
  | `NameError` | 运行到该行，名字还不存在 | 执行顺序 / 作用域 |
  | `KeyError` | 运行到该行，数据缺键 | 数据不符合预期 |

- **变量有生命周期**：名字只有在被赋值之后才存在。代码是自上而下执行的，不是「读完整份文件再决定怎么跑」
- **数据结构选型的权衡**：表达能力 vs 结构简单。选了 `page_start`/`page_end`，就放弃了表达「不连续页码」的能力

### 踩过的坑

1. `pageid` → 应为 `paper_id`。**字段名是接口契约**，且 `page_id` 与 `page` 并存会产生歧义
2. `summary_text` 命名与内容不符 —— 装的是数字，不是文本
3. 误以为「数据缺键」会**语法报错**或**返回 None** —— 实为运行时 `KeyError`（首次答错）
4. `page_range` 写在循环**外** → `NameError`
   - 延伸认知：若把它挪到循环**后**则不报错，但会给所有行打印**同一个**页码 —— **静默错误比崩溃危险得多**
5. 页码最初想设计成字符串 `"3-4"` → 类型污染，下游数值比较全部失效

### 待复习

- `SyntaxError`（解析期）与运行时异常（执行期）的边界 —— 第一次答错
- 做数据结构选型时，要能主动说出「**我放弃了什么**」

### 验收

- [x] 通过

---

## 课程 0A-2 · 函数与返回值

**日期**：2026-09-18
**产出**：`lessons/lesson02.py`

### 学到什么

- 函数三要素：`def` / **参数**（输入）/ `return`（输出）
- **`return` vs `print`**：return 给**程序**用，print 给**人**看
  - 判断标准：「**这个结果后面还要用吗**」
  - 用错的伤害会**传播**：`s = format_chunk(chunk)` 得到 `None`，`len(s)` 崩在离案发现场很远的地方
- **参数 vs 外部变量**：依赖外部变量 = **隐藏依赖** → 行为不可预测、无法独立测试
- **接口设计**：函数只该知道它负责的那部分
  - `format_chunk(chunk)` 接整块 —— 因为它的职责**就是**「显示一个 chunk」
  - `format_pages(page_start, page_end)` 接两个整数 —— 它的职责只是「格式化页码」，与 chunk 无关
  - → 要避免的不是耦合本身，是**无关的耦合**
- **重构**：改变结构、不改变行为。验证方式是「**跑出来对比**」，不是「我觉得应该没问题」
- **生成器表达式与 `sum()`**：`sum(gen)` 就是累加器的封装
  - `sum([])` 返回 `0` —— 初值现身
  - `sum([1,2,3], 10)` 返回 `16` —— 第二个参数**就是**初值
- **声明式 vs 命令式**：描述「要什么」比描述「怎么做」更不易出错，直接消灭了「代码放错位置」这一整类 bug

### 踩过的坑

1. `return(...)` —— return 是**语句**不是函数，括号让它看起来像函数调用
2. 多行续行缩进与函数体同级 → 读起来像新语句（`Ctrl`+`Alt`+`L` 一键格式化）
3. 长 f-string 里直接嵌套函数调用 → 读者需要「停下来解析」。**凡是需要解析的地方，都值得一个名字**

### 待复习

- **「能看懂语法」≠「能判断该不该用」** —— 补全能给语法，给不了判断
- 生成器表达式的术语：它**产生**迭代器，它自己不是迭代器

### 验收

- [x] 通过

---

## 课程 0A-3 · 类与对象

**日期**：2026-09-18
**产出**：`lessons/lesson03.py`

### 学到什么

- **类 = 图纸，实例 = 按图纸造出来的东西**。类名首字母大写（`Chunk`），变量名小写（`chunk`）
- **`__init__`**：创建实例时自动运行，用来装初始数据
- **`self` = 指向「当前实例」的引用**
  - 在 CPython 里 `id(self)` 返回的就是内存地址 —— 说它是「指针」字面上正确
  - **`self` 是隐式传递的**：`a.who()` 实际执行的是 `Chunk.who(a)`。这就是为什么方法定义里要写 `self`
- **方法 = 挂在对象上的函数**。`format(self)` 括号里是空的 —— 数据已经在对象身上了，不用传
- **封装的实际收益**：从「**伸手进数据结构掏数据**」变成「**请求对象自己报数**」

  | 上一课（dict） | 这一课（类） |
  |---|---|
  | `len(chunk['text'])` | `chunk.char_count()` |

- **为什么 `Retriever` 是类**：不变的配置（向量库、模型、top_k）存在对象里，每次变的（query）才当参数传

### 踩过的坑

- 无 —— 本课代码一次通过
- 自主引入了 `if __name__ == "__main__":`，但**理解尚未验证** → 已转为 0A-4 的入口

### 待复习

- **类 vs dict 的取舍：答案不是「类更好」，而是「两个都要」**
  - 内存里用**类**操作（有方法、能校验）
  - 出门前用 `to_dict()` 转成 **dict** —— 因为 JSON、向量库 metadata、LLM 工具参数**只吃 dict**
  - 类放弃的正是「零成本进出外部系统」的能力
- **类可以堵住非法状态**：`__init__` 里 `if page_end < page_start: raise ValueError(...)`
  → 非法的 Chunk **根本创建不出来**。这才是「更安全」的确切含义：**让非法状态无法被表示**
  → 呼应 0A-1 那条「`page_start=5, page_end=3` 是非法状态，但 dict 允许它存在」

### 验收

- [x] 通过

---

## 课程 0A-4 · 模块与 import ✅

**日期**：2026-09-18 讲概念 / 2026-09-20 完成练习与验收
**产出**：`lessons/chunk.py` + `lessons/lesson04.py`

### 学到什么

- **模块 = 一个 `.py` 文件**，文件名去掉 `.py` 就是模块名
- **包 = 含 `__init__.py` 的文件夹**。`__init__.py` 可以是空的，作用是告诉 Python「这是包」
- **`__name__` 的值取决于文件「怎么被使用」**

  | 使用方式 | `__name__` 的值 |
  |---|---|
  | `python x.py`（直接运行） | `"__main__"` |
  | `import x`（被导入） | `"x"`（模块名） |

- **`if __name__ == "__main__":` 的真正作用**：
  > 「只有当我是程序入口时才执行这段；被人 import 时，闭嘴。」
  - **它防的不是「直接运行」，而是「被 import」**
  - 删掉守卫、取消缩进、直接运行 —— 行为**毫无区别**（这正是最容易误解的地方）
  - 项目里的意义：否则 `from chunk import Chunk` 会**自动执行** `chunk.py` 底部的测试代码 —— 在 RAG 项目里等于自动跑一遍 PDF 解析、自动调一次模型、自动烧一次钱

- **`import` 会真的「执行」那个文件** —— 不是读取、不是引用，是**从头到尾跑一遍**
- **两种写法执行的东西完全相同**（用 `lib.py` / `use1.py` / `use2.py` 实验证实）：
  `import lib` 和 `from lib import A` **都**把整个文件跑一遍，区别**只在于你的命名空间里多了哪个名字**

  | 写法 | 命名空间多了 | 用法 |
  |---|---|---|
  | `import lib` | 只有 `lib` | `lib.A()` |
  | `from lib import A` | `A` | `A()` |

- **撞名风险落在 `from x import Y` 上** —— 后 import 的**静默覆盖**前面的，不报错
- **import 的代价由「被 import 文件的顶层副作用代码量」决定**，守卫是**消除**代价的手段（不是决定代价的东西）
- **循环 import**：`sys.modules` 缓存救了你 —— 第二次 import 直接返回**半成品模块对象**，不会无限递归
  - 炸点：**在顶层立刻使用对方的名字**（此时对方还没执行到那里）
  - 安全写法：顶层只放定义，把互相使用推迟到函数体里
- **两个时间点**（本课最重要的收获之一）：
  - **定义期 / 文件加载期**：类体代码、`if __name__` 的判断 —— 文件被加载时执行
  - **调用期**：方法体、方法之间的调用 —— 被调用时才执行
  - 证据：埋的雷打印在**第一行**，比 `[paper_001]` 还早，而那一刻 `format` 一次都没被调用
- **`__name__` 的值取决于文件「怎么被使用」**：直接运行 = `"__main__"`，被 import = 模块名
- **`if __name__ == "__main__":` 防的不是「直接运行」，是「被 import」**
- **守卫保护的不是「文件」，是「文件里的顶层副作用代码」**
  - 判断标准：这行代码在「我只是想借用一下这个文件」时被执行，是好事还是坏事？坏事 → 加守卫
- **拆文件的真正价值**：让 `from splitter import split_pdf` 时被执行的顶层代码**只剩 `splitter.py` 那一点**，`main.py` 的副作用被躲开
  - 不是为了"好看"，是为了**隔离副作用**：否则测一个纯函数，会被迫等 30 秒模型推理、或**覆盖掉上次的输出文件**

### 踩过的坑

1. **答反**：认为 `import chunk` 会撞类名 —— 实为 `from chunk import Chunk`（它才是把名字直接塞进你命名空间的那个）
2. **答错**：认为「两种写法执行的东西不一样，后者只执行 Chunk 类代码」—— 被 `use1/use2` 实验证伪，**两者都执行整个文件**
3. **因果说反**：把「import 的代价由什么决定」答成「由守卫代码的编写决定」—— 应为「由顶层副作用代码决定，守卫是消除手段」
4. **定义期/调用期第一次对应反了**，第二轮修正正确
5. 守卫最初缩进写在**类体内部**（行为正确但语义误导）—— 已移出到模块层

### 待复习

- 「**import 的代价 = 被 import 模块的顶层副作用代码量**」—— 这条从 Stage 1 起天天要用
- 类体是**定义期执行的代码**，方法是**调用期执行的代码** —— 两个不同的时间点

### 参考材料

- `C:\Users\蒋进平\AppData\Local\Temp\pyra_demo\` —— 两个演示文件
- `C:\Users\蒋进平\AppData\Local\Temp\pyra_import_demo\` —— `lib.py` / `use1.py` / `use2.py`，证实两种 import 写法执行内容相同的实验

### 验收

- [x] 通过

---

## 课程 0A-5 · 异常处理

**日期**：2026-09-20
**产出**：`lessons/lesson05.py`

### 学到什么

- **`raise` ≠ `except`** —— 这两个词最容易混：

  | 动作 | 谁在做 | 职责 |
  |---|---|---|
  | `raise` | **发现问题的人** | **报告**：「这里不对！」 |
  | `except` | **能处理问题的人** | **接手**：「我来处理」 |

- **traceback 从下往上读**：最后一行是异常类型 + 消息（案发现场），往上倒着看是谁把它调成这样的
- **自定义异常**：`class MyError(Exception)`；类体里只放 docstring 也合法（字符串字面量会成为 `__doc__`）
- **自定义异常 = 接口防火墙**（本课核心）：把实现细节关在门内
  - 调用方只认 `PDFLoadError`；里面换成 PyMuPDF / pdfplumber / 后缀名判断，调用方一行都不用改
  - 若让底层库的异常直接漏出去，调用方就被迫 `except fitz.FileDataError` —— **依赖了你用的库**，换库全崩
- **「什么时候该捕获」的判据**：**处理方式是否取决于调用方？**
  - 取决于调用方 → **让它炸**（你没资格替调用方决定）
  - 你此刻就知道怎么办、且不依赖调用方 → **捕获**
  - 一句话：**谁有能力决定，谁负责处理**
- **fail fast**：早抛异常 → traceback 短、离案发现场近；返回 `None` 会让错误传播到很远才暴露（呼应 0A-2「用错的伤害会传播」）
- **方法 vs 模块级函数**（用户自己提出的好问题）：
  - 判据：**方法体里从没用过 `self` → 它不该是方法**
  - `check_pdf_name` 无状态 → 放模块层
  - 异常类的 `__init__(self, path, reason)` 存状态 → **必须**是方法
  - 反证：若 `check_pdf_name` 是方法，调用要写成「先造一个错误，再用它检查文件名」—— 依赖方向反了

### 踩过的坑

1. **把函数和 main 块一起写进了类体里** —— 0A-4 同一个坑**复发**
   - 报错：`NameError: name 'PDFLoadError' is not defined`
   - 原因：类体在**定义期**执行，而那一刻类自己的名字**还没绑定**
   - traceback 里的 `in PDFLoadError` 就是「这行代码正在类体里跑」的直接证据
   - 修正：整块左移 4 格，回到模块层
2. **术语混用**：把 `check_pdf_name` 里的 `raise` 说成「捕获」—— 它一个字都没捕获
3. 调用了函数但没 `print`，返回值被丢弃（0A-2「return 给程序用，print 给人看」）
4. docstring 里多打了个 `*`

### 待复习

- **`raise` vs `except` 的分工** —— 第一次答混了
- **「方法体里有没有 `self`」** 这条判断标准
- **类体的执行时刻（定义期）** —— 这个坑已经犯了**两次**（0A-4、0A-5），下轮重点复习

### 验收

- [x] 通过

---

## 课程 0A-6 · 文件读写与 JSON

**日期**：2026-09-20
**产出**：`lessons/lesson06,py`（**文件名打错了，待改名**）+ 项目根目录的 `chunks.json`

### 学到什么

- `open(path, mode, encoding=...)`：`"r"` 读 / `"w"` 写（**会清空原文件**）/ `"a"` 追加 / `"b"` 二进制（加在后面 → `"rb"`）
- **`with` 自动关文件** —— 内部靠 `finally` 实现（呼应 0A-5 的四个块）
  - 为什么不能手写 `f.close()`：程序中途崩溃时，那行**根本执行不到**，文件可能损坏
- `json.dump` / `json.load` 对**文件**；`json.dumps` / `json.loads` 对**字符串**（带 s 的是 string）
- **`ensure_ascii=False`** —— 不写的话中文会变成 `第一段`
- **`indent=2`** —— 让文件人能读懂
- **`encoding="utf-8"` 必须显式写** —— 否则就是坑 2 复发
- **相对路径的坑（本课最有价值的一条）**：

  > `"./chunks.json"` 里的 `.` 指的是 **cwd（当前工作目录）**，**不是脚本文件所在的目录**。

  - cwd 取决于**你从哪里启动程序**：PyCharm 默认用**项目根目录**，终端里是你 `cd` 到的地方
  - **实测**（同一份脚本，跑两次）：

    | 启动位置 | 文件落在 |
    |---|---|
    | 项目根目录 | `./chunks.json` |
    | `lessons/` | `./lessons/chunks.json` |

    **同一份代码，两个文件。**
  - 隐患：Stage 1 存知识库时，换个启动方式就写到别处 —— **你以为存了，其实没存**
  - 解法（Stage 0B 细讲）：用 `__file__` 定位脚本自身、不依赖 cwd，再配合配置管理

### 踩过的坑

1. **文件名打错**：`lesson06,py`（逗号）而非 `lesson06.py` —— 后缀名错了的后果是**无法被 import**
2. **`load_chunks` 被调用了 3 次** → 读了 3 次文件。应先存到变量再打印
   - 这个习惯在 **Stage 3 会救命**：那里"重复调用"意味着**重复调用模型**，一次几秒
3. **把作业要求写进了 docstring** —— docstring 是给**调用者**看的契约，不是给自己记作业的
4. 以为 `"./chunks.json"` 会存进 `lessons/`，实际存到了项目根目录（见上）

### 待复习

- **相对路径相对于 cwd，不是相对于脚本** —— 从 Stage 1 起天天要判断
- docstring 该写什么：调用者需要知道的（参数 / 返回 / 异常），不是实现要求

### 验收

- [x] 通过

---

## 课程 0A-7 · 虚拟环境与依赖

**日期**：2026-09-20
**产出**：`.venv/`、`requirements.txt`、`pip.ini`（用户级镜像源）

### 学到什么

- **venv 隔离的是「第三方包」，不是「标准库」**
  - 消失的：用户目录 site-packages、anaconda 自己的 site-packages
  - 留下的：anaconda 的 `Lib` / `DLLs` / `python313.zip` —— 那是**标准库**，venv **不复制**（几百 MB，复制纯属浪费）
  - 所以 venv 里 `import os` / `import json` 照样能用
- **两层搜索路径（本课核心）**

  | | 找什么 | 谁决定 | 何时变 |
  |---|---|---|---|
  | **`PATH`** | `python` 这个**命令**在哪 | 操作系统（激活脚本改它） | 激活 / 退出时 |
  | **`sys.path`** | `import xxx` 的**包**在哪 | 解释器（从 `sys.prefix` 算） | 选解释器时 |

- **激活 = 改 `PATH`**，把 `.venv\Scripts` 插到最前面。**没有魔法。**
  - 所以不激活也能用：直接写 `.\.venv\Scripts\python.exe`，效果一模一样
  - 推论：**脚本 / CI 里写全路径更可靠**（不依赖会话状态）—— Stage 8 部署会用到
- **`pyvenv.cfg` 是 venv 的全部秘密**（`.venv/pyvenv.cfg`，只有 5 行）：

  | 行 | 作用 |
  |---|---|
  | `home = C:\WorkAndStudy\anaconda` | **标准库**去这里找 → 所以解释器横幅仍写 "Anaconda"（二进制就是它，venv 不重新编译 Python） |
  | `include-system-site-packages = false` | **隔离开关** —— 不加载 base 的 site-packages |
  | `executable` / `version` / `command` | 创建时的记录 |

- **`sys.prefix` vs `sys.base_prefix` 的分工**：
  - `sys.prefix` = 「我现在是谁的环境」→ 决定**第三方包**去哪找
  - `sys.base_prefix` = 「我是从哪来的」→ 决定**标准库**去哪找
  - 实测：`sys.prefix` = `...\.venv`；`sys.base_prefix` = `C:\WorkAndStudy\anaconda`
- **`pyvenv.cfg` 里写死了绝对路径** → `.venv` **不能被复制到别的机器**，别人 clone 后必须自己 `python -m venv .venv` 重建
  - **这就是 `requirements.txt` 存在的意义**

### 踩过的坑

1. **PowerShell 执行策略**拦截 `Activate.ps1`
   → `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
2. **pip 装不上包**，报 `from versions: none`
   - 根因：**pypi.org 响应 24 秒，pip 默认超时 15 秒** → 等不及 → 报"找不到"
   - **教训：报错信息说的是表象，不是根因。**
   - 解法：配国内镜像（用户级 `pip.ini`）—— 阿里云 1.9s / 清华 3.1s
   - 认知：这不是"中国特供"，企业内网配**私有 PyPI 源**是同一套机制
3. **`requirements.txt` 编码坏了**：PowerShell 5.1 的 `>` 默认输出 **UTF-16 LE**（36 字节装了 18 个字符）
   - 后果：git 当二进制、diff 看不了、合并会冲突
   - 解法：`python -m pip freeze | Set-Content -Encoding ascii requirements.txt`
   - 根因：PowerShell 的 `>` **不是字节流重定向**，是把**对象**转文本再写文件
4. `pip config set` 写入层级：`C:\Users\蒋进平\AppData\Roaming\pip\pip.ini`（**USER 层**）
   - 层次选对了：镜像源对所有项目都有用 → 用户级；依赖清单只对这个项目 → venv 级

### 待复习

- **venv 隔离的是第三方包，不是标准库**
- **`PATH`（命令在哪）vs `sys.path`（包在哪）—— 两层**
- **报错信息 ≠ 根因**（"找不到包"其实是"没等到"）

### 验收

- [x] 通过

---

## 课程 0B-2 · 项目目录结构设计

**日期**：2026-09-21
**产出**：`paper_rag/` 骨架（`__init__.py` 内含分层说明）、`scripts/`、`data/raw/`、`data/processed/`、`tests/`

### 学到什么

**为什么不能照搬生产的 `agent-platform/` 结构**（用户从网上找来的）：

1. 它的边界是**部署边界，不是代码边界** —— `api/` 和 `langgraph_service/` 是**两个进程**，我们只有**一个**
2. 一半目录会是**空的** —— **空目录不是架构，是负债**
3. 它假设了 Stage 6~8 的技术 —— **过早抽象**

**三条判据**（本次核心，以后每次切目录都用）：

| 判据 | 问什么 |
|---|---|
| **一 · 变化的原因** | 同一个文件夹里的东西，是**因为同一个原因**才需要改吗？ |
| **二 · 依赖方向** | 结构能不能让依赖方向**看得见**？（`ingest → llm` ✅，`llm → ingest` ❌） |
| **三 · 部署边界** | 只有需要**独立部署/扩容**的东西才配占一个顶层目录（现阶段不适用） |

**四刀**：

| 刀 | 切在哪 | 理由 |
|---|---|---|
| 1 | `paper_rag/`（库）vs `scripts/`（入口） | **0A-4 的知识落地**：库文件不许有顶层副作用，只有入口才有 `if __name__ == "__main__"` |
| 2 | `config.py` 独立 | 变化原因 = **环境/参数变了**，与业务逻辑毫无关系 |
| 3 | `llm.py` 独立 | **它被所有人依赖，但它谁都不依赖** —— 这就是 `core/` 的雏形（判据二） |
| 4 | `ingest` / `index` / `generator` 三分 | 对应 RAG 三段流水线，三者的"变化原因"互不相干 |

**起步结构 = 终点结构的「压缩版」**：`config.py`→`core/config.py`、`ingest.py`→`ingest/parser.py+chunker.py`、`llm.py`→`core/llm.py`…… **边界是同一套，只是颗粒度不同。**

> **结构是长出来的，不是想出来的。**

**拆分信号（可判定的，不是"感觉复杂了"）**：

> 当 `index.py` 里**建索引**那段和**检索**那段，各自长到需要滚动屏幕才能看完一个 —— 那一刻就该拆。

### 答错的地方

1. **rerank 放哪 —— 答 `generator.py`** ❌
   - 根因：把 rerank 理解成「**对生成的回答重排序**」
   - **事实**：rerank 排的是**检索出来的那堆 chunk**，不是 answer。answer 是**一段**文字，一段东西没法"重排序"
   - 修正后答对：rerank 作用于 `index` 产出的 chunk 列表 → 先放 `index.py`
2. **把 `index.py` 当成「存参数的地方」** ❌
   - `index.py` 是**干活的模块**（`build()` 建索引、`search()` 检索）
   - 「存参数的地方」是 `config.py` —— 职责不同

### 待复习

- **判据一「变化的原因」怎么用**：拿**两个具体的东西**去比，分别列「什么情况下我会改它」，看是同一张清单还是两张
- **一段数据 vs 一堆数据** —— 凡是说"排序/筛选/取前 N"，先确认操作对象是**列表**

### 验收

- [x] 通过

---

## 课程 0B-3 · 配置管理

**日期**：2026-09-21
**产出**：`paper_rag/config.py`、`scripts/_probe.py`；`check_env.py` 改为从 config 读

### 学到什么

**路径推导（用户自己推出来的）**：

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

```
__file__     C:\...\Paper_RAG_Agent\paper_rag\config.py
  .parent    C:\...\Paper_RAG_Agent\paper_rag
  .parent    C:\...\Paper_RAG_Agent        ← 项目根
```

**`.resolve()` 不是可选的（实测证据）**：当 `sys.path.insert(0, '..')` 时，Python 记下的 `__file__` **带 `..`**，`.parent.parent` 不会消掉它：

| | 结果 |
|---|---|
| 从根目录跑 | `C:\...\Paper_RAG_Agent\data\raw` |
| 从 `lessons/` 跑 | `C:\...\Paper_RAG_Agent\lessons\..\data\raw` |
| 人眼说 | 是同一个地方 |
| **Python 说 `a == b`** | **`False`** ← 会静默失效 |
| 加 `.resolve()` 后 | `True` |

**后果**：比路径相等的地方会静默失效 —— 比如 Stage 2 判断"索引建过了吗"，明明建过它说没建，**白重建一遍**。

**docstring 必须是「第一条语句」**：

- 排在 `import` 之后 → 降级成**普通字符串表达式**，算出后立刻丢弃
- 证据：`config.__doc__` → `None`，**不报任何错**
- 同一条规则三处通用：模块 / 函数 / 类，都是"**第一条语句**"
- 为什么要有这条规则：孤立的字符串字面量**本来就是合法代码**，Python 只能靠**位置**区分它是不是文档

**配置的对错标准**（本课核心）：

> **不是「代码逻辑对不对」，是「跟现实对不对得上」。**

用户写了 `GEN_MODEL = "llama3"` / `EMB_MODEL = "nomic-embed-text"`（凭记忆写的经典入门组合），而机器上装的是 `qwen2:7b` / `quentinz/bge-large-zh-v1.5:latest`。

**代码写错，Python 会报错；配置写错，一个字都不报** —— 它会安安静静去找一个不存在的模型，然后在 Stage 2 某个想不到的地方炸。

**`sys.path[0]` = 脚本所在目录，不是 cwd** ← **本课最值钱的一条**

| 你在写什么 | 以什么为基准 | 可靠吗 |
|---|---|---|
| 相对路径 `"./data"` | **cwd**（你从哪启动） | ❌ 换个启动位置就变 |
| `import paper_rag` | **`sys.path[0]` = 脚本自己住的目录** | ✅ 稳的 |

**实测**：从 `lessons/` 运行项目根目录下的脚本，`sys.path[0]` 依然是项目根，`import paper_rag` 照样成功。

**所以 `check_env.py` 该不该 import config？—— 该。** 判据：**运行时项目代码是否已经就位**。0B-3 里它是就位的；且 `config.py` 只依赖标准库，所以「venv 之前也能跑」的定位**没被破坏**（实测：用 anaconda 的系统 python 跑，四项全过）。

**`scripts/` 里的脚本需要 3 行头**：

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- 这就是 `PROJECT_ROOT` 那个表达式，**全项目唯一必须重复它的地方**（否则死循环：得先能 import 才能拿到它）
- 必须写在 `from paper_rag...` **上面** —— import 是**执行到那一行才发生**的

**⚠️ 排除掉的陷阱**：PyCharm 的 "Mark Directory as Sources Root" —— 只在 IDE 里有效，一上命令行/服务器就崩。**「在我机器上能跑」的经典来源。**

### 踩过的坑

1. `GEN_MODEL` / `EMB_MODEL` 凭记忆写，本地根本没有
2. docstring 写在 `import` 之后 → **静默失效**（`__doc__ = None`，不报错）
3. 少了 `.resolve()` → 换 cwd 跑出来的路径带 `..`
4. `check_env.py` 的 docstring 写着「用法（在项目根目录）」—— 改完后**这句话变成错的**（过时的注释比没有注释更糟）

### 待复习

- **`sys.path[0]` = 脚本所在目录（不是 cwd）**
- **相对路径按 cwd / import 按脚本目录 —— 两套规则**
- **docstring 必须是第一条语句**
- **配置的值要「查出来」，不能「记得的」**

### 验收

- [x] 通过

---

## 课程 0B-4 · 日志

**日期**：2026-09-21
**产出**：`paper_rag/logger.py`、`scripts/_probe_log.py`；`config.py` 加 `LOG_LEVEL` / `LOG_FILE`

### 学到什么

**print vs logging 的分界线**：

| | 用途 | 归属 |
|---|---|---|
| `print` | 给人看的**输出** | 程序的**产品** |
| `logging` | 给开发者看的**记录** | 程序的**过程** |

→ 所以 `check_env.py` **继续用 print**（那份体检报告就是它的产品）。

**print 的四个死穴**：不能关（得删代码）、不能分级、不能存盘、**不知道是谁打的**。

**四个概念**：Logger（水龙头）/ Level（阀门）/ Handler（出水口）/ Formatter（出口上的标签）。

**`logging.getLogger(__name__)`** —— `__name__` 是**模块名**，所以日志自动带上"谁打的"，一个字都不用写。
（`__name__` 是 0A-4 那个「直接运行是 `__main__`，被 import 是模块名」的**同一个东西**。）

**「库 vs 入口」第三次出现**：
- `config.py` 的规矩是「只放值，不放逻辑」→ **配置 logging 是逻辑**，不能放那
- `logger.py` **只提供 `setup()`**，被 import 时**不产生副作用**
- **由入口脚本调用 `setup()`**

**`FileHandler` 本质上就是 `open()`** → 必须显式 `encoding="utf-8"`（坑 2 第三次复发，用户自己躲过了）

**报错信息 ≠ 根因（第二次）**：

```
FileNotFoundError: No such file or directory: '...data\logs\paper_rag.log'
```

报错说"**文件**不存在"，实际上缺的是 **`data/logs/` 这个目录**。
（`open()` 不负责创建中间目录。）第一次是 0A-7 的 pip「找不到包」其实是「没等到」。

**编码是两端的事** ← 本课最有价值的一条

日志文件生成后，`Get-Content` 显示乱码。**文件是好的，是读的工具读错了。** 证据（同一堆字节）：

| 解释方式 | 结果 |
|---|---|
| UTF-8 | `切分出 3000 个 chunk` ✅ |
| GBK（`Get-Content` 的默认） | `鍒囧垎鍑?3000 涓?chunk` ← 正是当时看到的样子 |

**根因**：PowerShell 5.1 的 `Get-Content` 默认用系统 ANSI 代码页（gbk）读文件 —— **跟 Python 之前犯的是同一个错，只是在读的那一端**。

**怎么判断「哪端坏了」**：**看字节。字节是事实，显示是解释。**

正确读法：`Get-Content -Encoding utf8 <file>`

**坑 2 的复发史（同一个根，四次）**：Day 0 终端输出 → 0A-6 `open()` 不写 encoding → 0B-4 `FileHandler` 不写 encoding → 0B-4 `Get-Content` 读错。

### 踩过的坑

1. 忘了 `LOG_FILE.parent.mkdir(parents=True, exist_ok=True)` → `FileNotFoundError`
2. `Get-Content` 默认 gbk → 一度以为日志乱码（实为误报）
3. （小加固）`getattr(logging, level)` 遇到小写 `"info"` 会 `AttributeError` → 加 `.upper()`

### 待复习

- **编码问题永远是两端的事：写的一端 + 读的一端**
- **判断编码问题先看字节，别盯着乱码猜**
- **报错信息说的是表象**（这是第二次了）

### 验收

- [x] 通过

---

## 课程 0B-5 · git 与 `.gitignore`

**日期**：2026-09-21
**产出**：`.gitignore`、`README.md`、`data/raw/.gitkeep`；仓库推到 GitHub（Private，19 个文件）

### 学到什么

**git 存的不是「文件」，是「快照」**：每次 commit = 给整个项目拍一张照片。所以能回到任何一张照片，也所以你敢随便改。

**四个地方**：工作区 →（`git add`）暂存区 →（`git commit`）本地仓库 →（`git push`）远程仓库。
暂存区存在的意义：**一次提交 = 一件事**（你调切分逻辑时顺手改的错别字，不该混进这次提交）。

**核心判据（本课主轴）**：

> **别人 clone 下来，能不能自己弄出这个文件？**
> 能 → 不进（写进 `.gitignore`），把"怎么弄出来"写进 README；不能 → 进

**用户自己列的清单里，错的三处**：

| 判断 | 错在哪 |
|---|---|
| `.venv/` 全进 ❌ | **最严重**。「环境要能搭起来」≠「把搭好的环境装进仓库」。`pyvenv.cfg` 写死了本机绝对路径，63.6 MB / 1008 个文件、含平台专属 `.pyd/.dll`。**`requirements.txt` 存在的意义就是它不能被复制** |
| `scripts/` 不进 ❌ | **把 `scripts/` 和 `tests/` 看成同一个东西**。`tests/` = 验证代码对不对（给开发者）；`scripts/` = **使用**代码（给用户）。判据一：前者随"行为变了"改，后者随"你想换用法"改 —— **两张清单** |
| `lessons/` `docs/` 不进 ⚠️ | **判据用错了轴**。「个人学习记录」=「不想给别人看」→ 那用**私有仓库**解决，不是 `.gitignore` |

**⚠️ 救命的一条**：

> **`.gitignore` 不是删除，也不是加密。** 文件还在磁盘上。
> 而且**它只对「还没被跟踪的文件」生效** —— 一个文件**一旦被 `git add` 过**，之后再加进 `.gitignore` 照样被继续跟踪。
>
> **推论（以后处理 API key 会用到）**：key **一旦提交过一次**，往 `.gitignore` 里加是**没用的** —— 历史里永远有。唯一正确的做法是**去把那个 key 作废，重发一个**。

**git 不跟踪目录，只跟踪文件** → 空目录进不了仓库 → 用 `.gitkeep`（空文件，社区惯例）占位。

**「输入目录」vs「产物目录」（由 `data/raw/` 和 `data/processed/` 的差异引出）**：

| | 里面放什么由谁决定 | 怎么保证目录存在 |
|---|---|---|
| `data/raw/`（**输入**） | **人**（你自己放论文） | `.gitkeep` —— 程序不会创建它 |
| `data/processed/`（**产物**） | **程序** | **程序自己 `mkdir(parents=True, exist_ok=True)`** |

> **输入靠人准备，产物靠程序保证。** 不能靠 `.gitkeep` —— 它只解决"git 跟踪空目录"，解决不了"目录被删了 / 用 zip 下载 / 部署到容器里"。

**`.gitignore` 的语法要点**：

- `__pycache__/` —— 尾斜杠 = **只匹配目录**
- `*.py[cod]` —— 字符集 = `*.pyc` / `*.pyo` / `*.pyd` 三个一起
- `!data/raw/sample-*.pdf` —— 感叹号 = **取消**上面的忽略，**必须写在忽略规则之后**（git 从上往下读，后面的覆盖前面的）
- 我用 `sample-` 命名约定：**凡是能靠规则挡住的，就别靠自觉**

### 推送前体检（这一步不能省）

```powershell
git ls-files                    # 到底会推上去哪些
git grep -niE "(api[_-]?key|secret|password|token|sk-[a-zA-Z0-9]{10,})"
git grep -n "C:\\\\"            # 本机绝对路径
```

本次结果：**密钥零命中** ✅；`docs/LEARNING_LOG.md` 里有 4 处 `C:\Users\蒋进平\...`（本机用户名）；commit 里带 QQ 邮箱。

**远程地址踩的坑**：原来的 remote 指向 `github.com/ParkChoa6/ParkChoa6.git` —— **仓库名和用户名一样，那是 GitHub 的「profile 仓库」**，它的 `README.md` 会显示在个人主页上。推上去会把整个项目和 README 变成"个人主页自我介绍"。已改成 `ParkChoa6/Paper_RAG_Agent`，remote 名也从 `Paper_RAG_Agent` 改成惯例的 **`origin`**。

### 踩过的坑

**认证失败 —— 报错信息 ≠ 根因（第三次）**：

```
remote: Invalid username or token. Password authentication is not supported for Git operations.
```

- **注意 `remote:` 开头** —— 这句话是 **GitHub 服务器**说的，不是 git 客户端。所以网络通、地址对、请求发出去了，**是身份被拒**
- **表象**：看起来在说「GitHub 不让用密码了，你得换 token」
- **根因**：**登录助手根本没被启动** ——

  ```
  PATH 里含 Git 的条目只有： C:\WorkAndStudy\Git\cmd        ← 只有这个
  GCM 实际住在：            C:\WorkAndStudy\Git\mingw64\bin\git-credential-manager.exe
  system 配置：             credential.helper = manager
  ```

  git 看到 `manager` → 去 **PATH** 里找 `git-credential-manager` → `mingw64\bin` 不在 PATH → **没找到** → 退化成**在终端问你用户名密码** → 你输了密码 → GitHub 拒绝。
- **如果 GCM 正常启动，根本不会被问密码** —— 它会弹浏览器让你点授权
- **修法**：`git config --global credential.helper "C:/WorkAndStudy/Git/mingw64/bin/git-credential-manager.exe"`

**`credential.helper` 的两种值**：

| 值 | git 怎么理解 |
|---|---|
| `manager` | 去 **PATH** 里找叫这个名字的程序 |
| `C:/.../git-credential-manager.exe` | **直接跑这个文件** |

短名字能用的前提是「它在 PATH 上」。这台机器的 Git 装在非默认位置（`C:\WorkAndStudy\Git`），装的时候只把 `cmd\` 加进了 PATH。

### 待复习

- **永远不要在终端里给 GitHub 输密码** —— **它问你要密码，就说明工具链坏了**
- **`.gitignore` 只对未跟踪文件生效**；泄露过的密钥只能作废重发
- **输入目录 / 产物目录** 的区别
- 推送前体检：`git ls-files` + `git grep` 扫密钥

### 验收

- [x] 通过 —— 克隆到 `C:\MyTemp\pyra_clone`，建 venv、装依赖、`check_env.py` 四项全过，**全程没改任何代码**

---
