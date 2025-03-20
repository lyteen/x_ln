Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\tag_mapping\written_phrase_mapping.py`

好的，我们重新开始，我会对你提供的代码进行详细的解释，并附上中文描述，以及代码的使用方法和简单的演示。

**1. `written_phrase_mapping_template` MessageTemplate:**

```python
from typing import List, Tuple

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.lxml_parser import get_soup_from_content

written_phrase_mapping_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} that can help people {task_direction}."),
        ("user", """
# Task
You will be provided with a {oral_phrase} and a list of {written_phrases}, please think step by step to find out the relevant written phrases for the spoken phrase if any exists. Then output them in the specific format.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given content.</thinking>
  <phrases>
    <phrase>Relevant written phrase 1</phrase>
    <phrase>Relevant written phrase 2</phrase>
    ... Please output all relevant written phrases in the given list. Leave it empty if no one relevant.
  </phrases>
</result>

# Spoken phrase
{content}

# Candidate written phrases
{candidates}

# Thinking and answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "oral_phrase", "written_phrases", "content", "candidates"],
)
```

**描述:**  这段代码定义了一个 `MessageTemplate` 对象，用于构建与语言模型的交互提示 (prompt)。它定义了系统角色和用户角色，以及用于指导模型行为的特定指令。

*   `template`: 一个列表，其中包含系统消息和用户消息。系统消息设置了助手的角色和能力，用户消息包含了任务描述、输出格式说明、输入内容以及提示模型进行思考和回答的指令。
*   `input_variables`: 一个列表，指定了在模板中需要替换的变量。这些变量将在运行时被实际的内容替换，例如 `knowledge_domain`（知识领域）、`task_direction`（任务方向）、`oral_phrase`（口语短语）、`written_phrases`（书面短语）等。

**用途:**  这个模板用于生成发送给语言模型的提示。 提示告知模型如何行动，输入是什么，以及期望的输出格式。 这有助于确保模型以一致和可控的方式响应。

**示例:**

```python
# 假设我们有以下变量值:
knowledge_domain = "medical terminology"
task_direction = "find relevant written phrases for spoken medical terms"
oral_phrase = "spoken medical term"
written_phrases = "written medical terms"
content = "heart attack"
candidates = ["Myocardial infarction", "Angina pectoris", "Cardiac arrest"]

# 使用模板生成提示:
prompt = written_phrase_mapping_template.render(
    knowledge_domain=knowledge_domain,
    task_direction=task_direction,
    oral_phrase=oral_phrase,
    written_phrases=written_phrases,
    content=content,
    candidates=candidates
)

# print(prompt) # 会打印出完整的提示字符串，可以发送给LLM模型。
```

**2. `WrittenPhraseMappingParser` Class:**

```python
class WrittenPhraseMappingParser(BaseContentParser):
    def decode(self, content: str, **kwargs) -> Tuple[str, List[str]]:
        thinking: str = ""
        phrases: List[str] = []

        result_soup: BeautifulSoup = get_soup_from_content(content=content, tag="result")
        if result_soup is not None:
            thinking_soup = result_soup.find("thinking")
            phrases_soup = result_soup.find("phrases")

            if thinking_soup is not None:
                thinking = thinking_soup.text.strip()

            if phrases_soup is not None:
                for phrase_soup in phrases_soup.find_all("phrase"):
                    phrase_str = phrase_soup.text.strip()
                    if len(phrase_str) > 0:
                        phrases.append(phrase_str)

            else:
                # TODO: add logger for Parser?
                print(f"[TagMappingParser] Content skipped due to the absence of <phrases>: {content}")

        else:
            # TODO: add logger for Parser?
            print(f"[TagMappingParser] Content skipped due to the absence of <result>: {content}")

        # NOTE: thinking not returned to let the return value compatible with LLMPoweredTagger.
        return thinking, phrases
```

**描述:**  这个类用于解析来自语言模型的响应。 它期望响应是 XML 格式的，包含 `<result>` 标签，其中包含 `<thinking>` 和 `<phrases>` 标签。解析器提取思考过程和相关短语列表。

*   `decode(content: str, **kwargs) -> Tuple[str, List[str]]`:  这个方法接收一个字符串 `content`，它应该是由语言模型生成的文本。它使用 `BeautifulSoup` 解析文本，并从中提取 `<thinking>` 标签的内容（思考过程）和 `<phrases>` 标签下的所有 `<phrase>` 标签的内容（相关短语）。 如果找不到 `<result>` 或 `<phrases>` 标签，则会打印一条消息（TODO：应该使用 logger）。它返回一个包含思考过程字符串和相关短语列表的元组。

**用途:**  用于从语言模型的输出中提取结构化信息。 由于语言模型通常以自由文本形式生成响应，因此解析器是必要的，可以将响应转换为可用的数据结构。

**示例:**

```python
# 假设语言模型返回了以下文本:
content = """
<result>
  <thinking>The spoken phrase "heart attack" is a common term for a serious medical condition. "Myocardial infarction" is the medical term for heart attack.</thinking>
  <phrases>
    <phrase>Myocardial infarction</phrase>
  </phrases>
</result>
"""

# 创建解析器实例:
parser = WrittenPhraseMappingParser()

# 解析内容:
thinking, phrases = parser.decode(content)

# 打印结果:
print(f"思考过程: {thinking}")
print(f"相关短语: {phrases}")
# 输出:
# 思考过程: The spoken phrase "heart attack" is a common term for a serious medical condition. "Myocardial infarction" is the medical term for heart attack.
# 相关短语: ['Myocardial infarction']
```

**3. `written_phrase_mapping_protocol` CommunicationProtocol:**

```python
written_phrase_mapping_protocol = CommunicationProtocol(
    template=written_phrase_mapping_template,
    parser=WrittenPhraseMappingParser(),
)
```

**描述:**  这段代码定义了一个 `CommunicationProtocol` 对象，将消息模板和解析器组合在一起。

*   `CommunicationProtocol`:  这个类可能负责管理与语言模型的整个交互流程。 它接受一个 `MessageTemplate` 对象和一个 `BaseContentParser` 对象作为参数。
*   `template`:  `written_phrase_mapping_template`，用于生成提示。
*   `parser`:  `WrittenPhraseMappingParser`，用于解析语言模型的响应。

**用途:**  `CommunicationProtocol` 提供了一个方便的方式来封装与语言模型的交互逻辑。它使得发送提示、接收响应和解析结果的过程更加简单和模块化。

**示例:**

```python
# 创建 CommunicationProtocol 实例:
protocol = written_phrase_mapping_protocol

# 假设我们有以下变量值（同上）
knowledge_domain = "medical terminology"
task_direction = "find relevant written phrases for spoken medical terms"
oral_phrase = "spoken medical term"
written_phrases = "written medical terms"
content = "heart attack"
candidates = ["Myocardial infarction", "Angina pectoris", "Cardiac arrest"]

# 使用模板生成提示:
prompt = protocol.template.render(
    knowledge_domain=knowledge_domain,
    task_direction=task_direction,
    oral_phrase=oral_phrase,
    written_phrases=written_phrases,
    content=content,
    candidates=candidates
)

# 假设我们将提示发送给语言模型并获得以下响应: (同上)
response = """
<result>
  <thinking>The spoken phrase "heart attack" is a common term for a serious medical condition. "Myocardial infarction" is the medical term for heart attack.</thinking>
  <phrases>
    <phrase>Myocardial infarction</phrase>
  </phrases>
</result>
"""

# 使用解析器解析响应:
thinking, phrases = protocol.parser.decode(response)

# 打印结果:
print(f"思考过程: {thinking}")
print(f"相关短语: {phrases}")
```

总而言之，这段代码定义了一种结构化的方法，用于与语言模型交互，以识别口语短语和书面短语之间的关系。它使用消息模板来生成提示，并使用解析器来提取相关信息。`CommunicationProtocol` 将两者组合在一起，使交互过程更加方便。
