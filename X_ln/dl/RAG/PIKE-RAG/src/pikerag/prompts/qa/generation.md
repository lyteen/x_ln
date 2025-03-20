Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\qa\generation.py`

好的，我们重新开始分析这段代码。

**1. 模块导入 (Import Statements):**

```python
from typing import Dict, List, Tuple
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.json_parser import parse_json
```

**描述:** 这一部分导入了代码中需要使用的类型提示和类。
- `typing`: 用于类型提示，提高代码可读性和可维护性。`Dict`, `List`, `Tuple` 分别表示字典、列表和元组类型。
- `pikerag.prompts`: 导入了 `BaseContentParser`, `CommunicationProtocol`, `MessageTemplate` 这三个类，这些类很可能用于构建和处理与语言模型交互的消息提示。
- `pikerag.utils.json_parser`: 导入了 `parse_json` 函数，这个函数可能用于安全地解析 JSON 字符串。

**2. 默认系统提示 (DEFAULT_SYSTEM_PROMPT):**

```python
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant on question answering."
```

**描述:**  定义了一个字符串常量，作为默认的系统提示。这个提示会被用于指示语言模型扮演的角色，即一个“乐于助人的AI问答助手”。

**3. 生成式问答模板 (generation_qa_template):**

```python
generation_qa_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to give your answer to the given question.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.
{{
    "answer": <a string. Your answer.>,
    "rationale": <a string. Rationale behind your answer.>
}}

# Question
{content}

Let's think step by step.
""".strip()),
    ],
    input_variables=["content"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)
```

**描述:**  定义了一个 `MessageTemplate` 对象，用于构建与语言模型交互的消息。
- `template`: 定义了消息的结构，包含一个 "system" 消息和一个 "user" 消息。
    - "system" 消息使用 `{system_prompt}` 占位符，会被替换为系统提示。
    - "user" 消息包含任务描述、输出格式要求和一个 `{content}` 占位符，会被替换为实际的问题内容。
- `input_variables`:  指定了模板中需要用户提供的变量，这里是 "content"，即问题内容。
- `partial_variables`:  指定了模板中可以预先设置的变量，这里是 "system_prompt"，被设置为 `DEFAULT_SYSTEM_PROMPT`。

**如何使用:**  这个模板可以用来格式化发送给语言模型的问题，要求模型以 JSON 格式返回答案和理由。
**演示:**  你可以通过将具体的问题内容传递给 `MessageTemplate` 对象，来生成实际的提示。

**4. 带参考的生成式问答模板 (generation_qa_with_reference_template):**

```python
generation_qa_with_reference_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to answer a question referring to a given context, if any.
For answering the Question at the end, you need to first read the context provided, then give your final answer.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.
{{
    "answer": <A string. Your Answer.>,
    "rationale": <A string. Rationale behind your choice>
}}

# Context, if any
{context_if_any}

# Question
{content}{yes_or_no_limit}

Let's think step by step.
""".strip()),
    ],
    input_variables=["content", "context_if_any", "yes_or_no_limit"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)
```

**描述:**  类似于 `generation_qa_template`，但这个模板用于处理需要参考上下文信息的问答任务。
- `template`:  增加了 `{context_if_any}` 占位符，用于插入参考上下文。 另外还有一个 `{yes_or_no_limit}`占位符，用于约束模型的输出。
- `input_variables`: 除了 "content" 之外，还包括 "context_if_any" 和 "yes_or_no_limit"，分别表示参考上下文和是否需要限制为“是”或“否”的回答。

**如何使用:**  这个模板用于当问题需要根据提供的上下文进行回答时。
**演示:** 将问题内容、上下文信息和是否需要“是/否”回答的指示传递给 `MessageTemplate` 对象，生成提示。

**5.  GenerationQaParser 类:**

```python
class GenerationQaParser(BaseContentParser):
    def encode(
        self, content: str, references: List[str]=[], context_len_limit: int=80000, **kwargs,
    ) -> Tuple[str, dict]:
        # Construct `yes_or_no_limit` instruction.
        # TODO: update the logic when "question_type" enabled.
        answer_labels = kwargs.get("answer_labels", [])
        if len(answer_labels) == 1 and answer_labels[0] in ["yes", "no"]:
            yes_or_no_limit = """ Your answer shall be "Yes" or "No"."""
        else:
            yes_or_no_limit = ""

        # Construct reference contexts.
        context_if_any = ""
        for context in list(set(references)):
            context_if_any += f"\n{context}\n"
            if len(context_if_any) >= context_len_limit:
                break

        return content, {
            "yes_or_no_limit": yes_or_no_limit,
            "context_if_any": context_if_any,
        }

    def decode(self, content: str, **kwargs) -> Dict[str, str]:
        try:
            output = parse_json(content)
        except Exception as e:
            print(f"[GenerationQaParser] Content: {content}\nException: {e}")
            return {  # TODO
                "answer": "parsing error",
                "rationale": "parsing error",
            }

        for key, value in output.items():
            output[key] = str(value)
        return output
```

**描述:**  定义了一个 `GenerationQaParser` 类，继承自 `BaseContentParser`。这个类的作用是将输入内容编码成适合模板的格式，并将语言模型的输出解码成可用的数据结构。

- `encode` 方法:
    - 接收问题内容 (`content`)、参考文本列表 (`references`)、上下文长度限制 (`context_len_limit`) 和其他关键字参数 (`kwargs`)。
    - 根据 `kwargs` 中的 `answer_labels` 判断是否需要添加 "yes_or_no_limit" 指示。
    - 将 `references` 中的文本拼接成 `context_if_any` 字符串，并限制其长度不超过 `context_len_limit`。
    - 返回原始问题内容和包含 "yes_or_no_limit" 和 "context_if_any" 的字典，这个字典会被用于填充 `generation_qa_with_reference_template` 模板。

- `decode` 方法:
    - 接收语言模型的输出字符串 (`content`)。
    - 尝试使用 `parse_json` 函数解析输出字符串。
    - 如果解析成功，将 JSON 对象中的所有值转换为字符串，并返回一个字典，包含 "answer" 和 "rationale"。
    - 如果解析失败，则返回一个包含 "parsing error" 的字典。

**如何使用:**  `GenerationQaParser` 用于预处理输入问题和上下文，并解析语言模型的输出，使其易于使用。

**6. CommunicationProtocol 对象:**

```python
generation_qa_protocol = CommunicationProtocol(
    template=generation_qa_template,
    parser=GenerationQaParser(),
)


generation_qa_with_reference_protocol = CommunicationProtocol(
    template=generation_qa_with_reference_template,
    parser=GenerationQaParser(),
)
```

**描述:**  定义了两个 `CommunicationProtocol` 对象，分别对应于不带参考和带参考的问答任务。
- `CommunicationProtocol` 接收一个 `MessageTemplate` 对象和一个 `BaseContentParser` 对象。
- 它将模板和解析器组合在一起，形成一个完整的通信协议，用于与语言模型进行交互。

**如何使用:**  `CommunicationProtocol` 对象可以用来方便地发送问题给语言模型，并解析模型的输出。 它封装了模板格式化和输出解析的细节。

**总结:**

这段代码定义了一套用于构建和处理与语言模型进行问答交互的工具。它包括：
- 两个消息模板，分别用于不带参考和带参考的问答任务。
- 一个解析器，用于预处理输入和解析语言模型的输出。
- 两个通信协议，将模板和解析器组合在一起，形成完整的交互流程。

这些工具可以用于构建各种问答系统，例如知识库问答、阅读理解等。 通过灵活地组合模板和解析器，可以定制化与语言模型的交互方式，以满足不同的应用场景。
