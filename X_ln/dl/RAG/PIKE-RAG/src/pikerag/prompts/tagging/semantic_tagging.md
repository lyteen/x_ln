Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\tagging\semantic_tagging.py`

好的，下面我将对你提供的代码进行分解，并用中文解释每个部分的功能，同时附上简短的代码示例和使用说明。

**1. 导入必要的库 (Import necessary libraries):**

```python
from typing import List  # 用于类型提示，指定列表类型
from bs4 import BeautifulSoup  # 用于解析HTML/XML文档
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate  # 导入自定义的类
from pikerag.utils.lxml_parser import get_soup_from_content  # 导入自定义的函数
```

**描述:** 这段代码导入了所需的库。 `typing` 用于类型提示， `BeautifulSoup` 用于解析HTML或XML内容，`pikerag.prompts`和`pikerag.utils.lxml_parser` 包含自定义的类和函数，用于与语言模型交互和解析其输出。

**2. 定义语义标签模板 (Define Semantic Tagging Template):**

```python
semantic_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} that can help people {task_direction}."),
        ("user", """
# Task
Please read the content provided carefully, think step by step, then extract the {tag_semantic} phrases contained therein.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given content.</thinking>
  <phrases>
    <phrase>Extracted phrase 1</phrase>
    <phrase>Extracted phrase 2</phrase>
    <phrase>Extracted phrase 3</phrase>
    ... Please output an equal number of phrases based on the number of phrases contained in the content. Leave it empty if no phrase found.
  </phrases>
</result>

# Content
{content}

# Thinking and answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "tag_semantic", "content"],
)
```

**描述:**  `semantic_tagging_template`  定义了一个用于提示语言模型的模板。 它包含系统消息和用户消息，指示模型根据给定的内容提取特定语义的短语。  模板使用 `input_variables` 定义了可以动态填充的变量，例如知识领域、任务方向、标签语义和内容本身。

**3. 定义语义标签解析器 (Define Semantic Tagging Parser):**

```python
class SemanticTaggingParser(BaseContentParser):
    def decode(self, content: str, **kwargs) -> List[str]:
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
                print(f"[SemanticTagParser] Content skipped due to the absence of <phrases>: {content}")

        else:
            # TODO: add logger for Parser?
            print(f"[SemanticTagParser] Content skipped due to the absence of <result>: {content}")

        # NOTE: thinking not returned to let the return value compatible with LLMPoweredTagger.
        return phrases
```

**描述:**  `SemanticTaggingParser`  类负责解析语言模型的输出。 它使用 `BeautifulSoup` 从输出字符串中提取思考过程和短语。 具体来说，它查找 `<result>` 标签，然后在其中查找 `<thinking>` 和 `<phrases>` 标签。  提取的短语存储在列表中并返回。  如果缺少 `<result>` 或 `<phrases>` 标签，则会打印一条消息（TODO：应该使用logger）。

**4. 定义通信协议 (Define Communication Protocol):**

```python
semantic_tagging_protocol = CommunicationProtocol(
    template=semantic_tagging_template,
    parser=SemanticTaggingParser(),
)
```

**描述:** `semantic_tagging_protocol`  定义了与语言模型进行通信的协议。 它将 `semantic_tagging_template` （提示模板） 和 `SemanticTaggingParser`  （解析器） 组合在一起。  该协议用于格式化发送给语言模型的提示，并解析模型的响应。

**使用示例:**

假设你有一个文档内容 `content`，你想提取关键短语。  你可以这样使用：

```python
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.lxml_parser import get_soup_from_content
from typing import List
from bs4 import BeautifulSoup

# 重新定义 MessageTemplate, SemanticTaggingParser, CommunicationProtocol
# (为了保证代码完整性，这里重新定义，实际使用中可以省略)
class MessageTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs):
        formatted_template = []
        for role, content in self.template:
            formatted_content = content.format(**kwargs)
            formatted_template.append((role, formatted_content))
        return formatted_template


class BaseContentParser:
    def decode(self, content: str, **kwargs) -> List[str]:
        raise NotImplementedError()


class CommunicationProtocol:
    def __init__(self, template, parser):
        self.template = template
        self.parser = parser

    def format(self, **kwargs):
        return self.template.format(**kwargs)

    def parse(self, content: str, **kwargs) -> List[str]:
        return self.parser.decode(content, **kwargs)

def get_soup_from_content(content: str, tag: str):
    """Mock function to simulate lxml parsing."""
    xml_content = f"<root><{tag}>{content}</{tag}></root>" # Wrap content in root and target tag
    soup = BeautifulSoup(xml_content, 'xml') # Use 'xml' parser
    return soup.find(tag)


semantic_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} that can help people {task_direction}."),
        ("user", """
# Task
Please read the content provided carefully, think step by step, then extract the {tag_semantic} phrases contained therein.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given content.</thinking>
  <phrases>
    <phrase>Extracted phrase 1</phrase>
    <phrase>Extracted phrase 2</phrase>
    <phrase>Extracted phrase 3</phrase>
    ... Please output an equal number of phrases based on the number of phrases contained in the content. Leave it empty if no phrase found.
  </phrases>
</result>

# Content
{content}

# Thinking and answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "tag_semantic", "content"],
)


class SemanticTaggingParser(BaseContentParser):
    def decode(self, content: str, **kwargs) -> List[str]:
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
                print(f"[SemanticTagParser] Content skipped due to the absence of <phrases>: {content}")

        else:
            # TODO: add logger for Parser?
            print(f"[SemanticTagParser] Content skipped due to the absence of <result>: {content}")

        # NOTE: thinking not returned to let the return value compatible with LLMPoweredTagger.
        return phrases


semantic_tagging_protocol = CommunicationProtocol(
    template=semantic_tagging_template,
    parser=SemanticTaggingParser(),
)


content = "This document discusses the importance of information retrieval and knowledge graphs. Key aspects include semantic search, entity recognition, and relationship extraction."

# 定义输入变量
input_data = {
    "knowledge_domain": "Information Retrieval",
    "task_direction": "extracting key phrases",
    "tag_semantic": "key",
    "content": content
}

# 格式化提示
prompt = semantic_tagging_protocol.format(**input_data)

# 模拟语言模型的输出 (Replace with actual LLM output)
llm_output = """
<result>
  <thinking>The document mentions information retrieval, knowledge graphs, semantic search, entity recognition, and relationship extraction. These are the key phrases.</thinking>
  <phrases>
    <phrase>information retrieval</phrase>
    <phrase>knowledge graphs</phrase>
    <phrase>semantic search</phrase>
    <phrase>entity recognition</phrase>
    <phrase>relationship extraction</phrase>
  </phrases>
</result>
"""

# 解析输出
extracted_phrases = semantic_tagging_protocol.parse(llm_output)

# 打印提取的短语
print(f"Extracted phrases: {extracted_phrases}")
```

**代码解释:**

1.  **定义变量:** 首先定义了 `content`，`input_data`，这些是发送给语言模型的输入。
2.  **格式化 Prompt:** 使用  `semantic_tagging_protocol.format(**input_data)` 根据 `input_data` 填充模板，创建发送给语言模型的提示。
3.  **模拟语言模型输出:**  `llm_output` 模拟语言模型的响应。在实际应用中，你需要用真正的语言模型调用替换它。
4.  **解析输出:** 使用  `semantic_tagging_protocol.parse(llm_output)` 解析语言模型的输出，提取关键短语。
5.  **打印结果:** 打印提取的关键短语。

**注意:**  这只是一个简化的示例。 在实际应用中，你需要使用真正的语言模型，并根据你的具体需求调整模板和解析器。  `get_soup_from_content` 函数在这里被模拟，目的是为了保证示例代码的完整性，在实际的使用环境中，该函数应该返回一个 `BeautifulSoup` 对象。  而且请注意处理潜在的异常和错误情况，例如语言模型返回无效的输出格式。
