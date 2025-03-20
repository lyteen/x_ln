Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\tagging\atom_question_tagging.py`

好的，我们重新开始。

**1. `atom_question_tagging_template`:**

```python
from pikerag.prompts import MessageTemplate

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant good at content understanding and asking question."

atom_question_tagging_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to extract as many questions as possible that are relevant and can be answered by the given content. Please try to be diverse and avoid extracting duplicated or similar questions. Make sure your question contain necessary entity names and avoid to use pronouns like it, he, she, they, the company, the person etc.

# Output Format
Output your answers line by line, with each question on a new line, without itemized symbols or numbers.

# Content
{content}

# Output:
""".strip()),
    ],
    input_variables=["content"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)
```

**描述:**  这段代码定义了一个 `MessageTemplate`，用于构建一个用于提取问题的提示。 它包括一个系统提示和一个用户提示。 用户提示说明了任务、期望的输出格式和要处理的内容。 `input_variables` 指定需要提供哪些输入（在本例中为 `content`）。 `partial_variables` 允许预先设置提示的某些部分（在本例中为 `system_prompt`）。

**如何使用:** 你可以使用这个模板来创建一个字符串，该字符串可以发送到语言模型。 你需要提供 `content` 的值。 例如:
```python
template_string = atom_question_tagging_template.format(content="这是一个关于苹果公司的段落。")
print(template_string)
```

**2. `AtomQuestionParser`:**

```python
from typing import List, Tuple
from pikerag.prompts import BaseContentParser

class AtomQuestionParser(BaseContentParser):
    def encode(self, content: str, **kwargs) -> Tuple[str, dict]:
        title = kwargs.get("title", None)
        if title is not None:
            content = f"Title: {title}. Content: {content}"
        return content, {}

    def decode(self, content: str, **kwargs) -> List[str]:
        questions = content.split("\n")
        questions = [question.strip() for question in questions if len(question.strip()) > 0]
        return questions
```

**描述:**  `AtomQuestionParser` 类负责编码和解码内容。 `encode` 方法接受内容，如果提供了 `title`，则将其添加到内容中。 `decode` 方法接受包含问题的字符串，将其拆分为行，并返回一个问题的列表。

**如何使用:** 这个类用于预处理发送到语言模型的内容 (使用 `encode`)，以及后处理从语言模型接收到的内容 (使用 `decode`)。

```python
parser = AtomQuestionParser()
encoded_content, _ = parser.encode(content="这是关于苹果公司的段落。", title="苹果公司")
print(f"编码后的内容: {encoded_content}")

decoded_questions = parser.decode(content="苹果公司是做什么的？\n苹果公司的总部在哪里？")
print(f"解码后的问题: {decoded_questions}")
```

**3. `atom_question_tagging_protocol`:**

```python
from pikerag.prompts import CommunicationProtocol

atom_question_tagging_protocol = CommunicationProtocol(
    template=atom_question_tagging_template,
    parser=AtomQuestionParser(),
)
```

**描述:**  `CommunicationProtocol` 将 `MessageTemplate` 和 `BaseContentParser` 组合在一起。 它提供了一种方便的方式来定义与语言模型的交互方式。

**如何使用:**  你可以使用 `CommunicationProtocol` 对象将内容发送到语言模型并解析结果。

```python
#  这是一个简化的示例，需要实际与语言模型的交互
#  请注意，这需要一个语言模型来实际处理模板。

# 假设我们有一个名为 `llm` 的函数，它接受字符串并返回模型的响应
# def llm(prompt: str) -> str:
#   # 实际调用语言模型
#   pass

# prompt = atom_question_tagging_protocol.template.format(content="这是一篇关于AI的文章")
# response = llm(prompt)
# questions = atom_question_tagging_protocol.parser.decode(response)
# print(f"提取出的问题: {questions}")
```

总而言之，这段代码定义了一个用于从文本内容中提取问题的协议。 它使用 `MessageTemplate` 来格式化提示，使用 `AtomQuestionParser` 来编码和解码内容，并使用 `CommunicationProtocol` 来组合两者。
