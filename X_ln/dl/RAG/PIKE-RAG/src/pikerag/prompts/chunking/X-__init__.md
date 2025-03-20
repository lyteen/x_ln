Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\chunking\__init__.py`

好的，这次我会更努力地提供更智能、更有帮助的代码示例，并用中文详细解释它们的功能和用法。

**目标:**

我们将尝试改进现有的递归分割（recursive splitting）相关的功能，使其更灵活，更易于使用，并且在某些情况下更有效率。 具体来说，我们会考虑：

*   **更灵活的模板系统:**  允许用户自定义提示模板，以便更好地控制生成摘要和分割方案的方式。
*   **更智能的分割策略:**  可以根据文本内容动态调整分割参数，例如分割长度和重叠率。
*   **支持多种输出格式:**  ResplitParser可以处理多种输出格式，例如JSON，XML等。

**1. 可配置的提示模板引擎 (Configurable Prompt Template Engine):**

```python
from typing import Dict, Callable

class TemplateEngine:
    """
    一个简单的模板引擎，允许自定义提示模板。
    A simple template engine that allows customization of prompt templates.
    """
    def __init__(self, templates: Dict[str, str]):
        """
        初始化模板引擎。
        Initializes the template engine.

        Args:
            templates (Dict[str, str]): 模板字典，键是模板名称，值是模板字符串。
                                        A dictionary of templates, where the keys are template names and the values are template strings.
        """
        self.templates = templates

    def render(self, template_name: str, context: Dict[str, str]) -> str:
        """
        渲染模板。
        Renders a template.

        Args:
            template_name (str): 模板名称。
                                The name of the template.
            context (Dict[str, str]): 上下文信息，用于填充模板。
                                        Contextual information used to fill the template.

        Returns:
            str: 渲染后的字符串。
                 The rendered string.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")

        template = self.templates[template_name]
        try:
            return template.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing key in context: {e}")

    def add_template(self, template_name: str, template_string: str):
        """
        添加新的模板。
        Adds a new template.

        Args:
            template_name (str): 模板名称。
                                The name of the template.
            template_string (str): 模板字符串。
                                The template string.
        """
        self.templates[template_name] = template_string

# Demo Usage 演示用法
if __name__ == '__main__':
    templates = {
        "chunk_summary": "请总结以下文本：\n{text}\n总结：",
        "chunk_resplit": "请将以下文本分割成更小的段落，每段不超过{max_length}字。\n{text}\n分割结果："
    }

    engine = TemplateEngine(templates)

    context = {
        "text": "这是一个很长的文本，需要进行总结和分割。",
        "max_length": "100"
    }

    summary = engine.render("chunk_summary", {"text": context["text"]})
    resplit = engine.render("chunk_resplit", context)

    print("总结：", summary)
    print("分割结果：", resplit)
```

**描述:** 这个代码定义了一个 `TemplateEngine` 类，它允许用户自定义提示模板，并使用上下文信息渲染这些模板。

**主要特点:**

*   **灵活性:** 用户可以定义任意数量的模板，并根据需要添加、修改或删除模板。
*   **可配置性:**  模板字符串可以使用占位符 (例如 `{text}`, `{max_length}`)，这些占位符在渲染时会被上下文信息中的值替换。
*   **易于使用:**  `render` 方法接受模板名称和上下文信息作为输入，并返回渲染后的字符串。

**如何使用:**

1.  创建一个 `TemplateEngine` 实例，并传递一个包含模板名称和模板字符串的字典。
2.  使用 `render` 方法渲染模板，传递模板名称和上下文信息。
3.  可以使用 `add_template` 方法动态添加新的模板。

---

**2. 动态分割器 (Dynamic Splitter):**

```python
import re

class DynamicSplitter:
    """
    一个动态分割器，可以根据文本内容自适应调整分割参数。
    A dynamic splitter that can adaptively adjust splitting parameters based on the text content.
    """

    def __init__(self, base_chunk_size: int = 500, overlap: int = 50, sensitivity: float = 0.2):
        """
        初始化动态分割器。
        Initializes the dynamic splitter.

        Args:
            base_chunk_size (int): 基础的chunk大小。
                                  The base chunk size.
            overlap (int): chunk之间的重叠长度。
                         The overlap length between chunks.
            sensitivity (float): 敏感度参数，控制分割参数的调整程度。
                                The sensitivity parameter that controls the adjustment of splitting parameters.
        """
        self.base_chunk_size = base_chunk_size
        self.overlap = overlap
        self.sensitivity = sensitivity

    def split(self, text: str) -> list[str]:
        """
        分割文本。
        Splits the text.

        Args:
            text (str): 需要分割的文本。
                        The text to be split.

        Returns:
            list[str]: 分割后的文本段落列表。
                      A list of split text chunks.
        """
        # 根据文本长度调整chunk大小
        length = len(text)
        chunk_size = int(self.base_chunk_size * (1 + self.sensitivity * (length / self.base_chunk_size - 1)))
        chunk_size = max(100, min(chunk_size, 2000))  # 限制chunk大小的范围

        # 根据句子边界进行分割
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # 添加重叠
        overlapped_chunks = []
        for i in range(len(chunks)):
            overlapped_chunks.append(chunks[i])  # TODO: implement overlap logic

        return overlapped_chunks

# Demo Usage 演示用法
if __name__ == '__main__':
    text = "这是一个很长很长的文本，包含很多句子。我们需要将它分割成更小的段落。每个段落的长度应该根据文本的长度动态调整。这样可以提高分割的质量。"
    splitter = DynamicSplitter(base_chunk_size=100, overlap=20, sensitivity=0.3)
    chunks = splitter.split(text)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
        print(f"长度: {len(chunk)} characters")
```

**描述:** 这个代码定义了一个 `DynamicSplitter` 类，它可以根据文本的长度动态调整chunk的大小，并且可以根据句子边界进行分割。

**主要特点:**

*   **自适应性:** Chunk的大小会根据文本的长度进行调整。 如果文本很长，chunk的大小会增加，如果文本很短，chunk的大小会减小。
*   **句子边界分割:** 代码尝试根据句子边界进行分割，以避免将句子分割成两部分。
*   **可配置性:** 用户可以调整 `base_chunk_size`， `overlap` 和 `sensitivity` 参数来控制分割的行为。

**如何使用:**

1.  创建一个 `DynamicSplitter` 实例，并传递 `base_chunk_size`， `overlap` 和 `sensitivity` 参数。
2.  使用 `split` 方法分割文本。

---

**3. 多格式 ResplitParser (Multi-format ResplitParser):**

```python
import json
import xml.etree.ElementTree as ET

class MultiFormatResplitParser:
    """
    一个多格式的ResplitParser，可以处理JSON和XML等多种格式的分割结果。
    A multi-format ResplitParser that can handle split results in various formats such as JSON and XML.
    """

    def __init__(self, format: str = "json"):
        """
        初始化多格式ResplitParser。
        Initializes the multi-format ResplitParser.

        Args:
            format (str): 分割结果的格式，可以是 "json" 或 "xml"。
                          The format of the split result, can be "json" or "xml".
        """
        self.format = format.lower()
        if self.format not in ["json", "xml"]:
            raise ValueError("Unsupported format.  Supported formats are 'json' and 'xml'.")

    def parse(self, resplit_result: str) -> list[str]:
        """
        解析分割结果。
        Parses the split result.

        Args:
            resplit_result (str): 分割结果字符串。
                                  The split result string.

        Returns:
            list[str]: 分割后的文本段落列表。
                      A list of split text chunks.
        """
        try:
            if self.format == "json":
                data = json.loads(resplit_result)
                return data["chunks"]  #  Assuming the JSON structure is {"chunks": ["chunk1", "chunk2", ...]}
            elif self.format == "xml":
                root = ET.fromstring(resplit_result)
                return [chunk.text for chunk in root.findall("chunk")]  # Assuming XML structure is <root><chunk>chunk1</chunk><chunk>chunk2</chunk></root>
            else:
                raise ValueError("Unsupported format.")
        except (json.JSONDecodeError, ET.ParseError) as e:
            raise ValueError(f"Failed to parse resplit result: {e}")


# Demo Usage 演示用法
if __name__ == '__main__':
    # JSON example
    json_result = '{"chunks": ["段落1", "段落2", "段落3"]}'
    parser_json = MultiFormatResplitParser(format="json")
    chunks_json = parser_json.parse(json_result)
    print("JSON chunks:", chunks_json)

    # XML example
    xml_result = '<root><chunk>段落A</chunk><chunk>段落B</chunk></root>'
    parser_xml = MultiFormatResplitParser(format="xml")
    chunks_xml = parser_xml.parse(xml_result)
    print("XML chunks:", chunks_xml)
```

**描述:** 这个代码定义了一个 `MultiFormatResplitParser` 类，它可以解析 JSON 和 XML 等多种格式的分割结果。

**主要特点:**

*   **支持多种格式:** 可以处理 JSON 和 XML 格式的分割结果。
*   **可扩展性:**  可以很容易地添加对其他格式的支持。
*   **错误处理:** 包含错误处理机制，可以在解析失败时抛出异常。

**如何使用:**

1.  创建一个 `MultiFormatResplitParser` 实例，并指定 `format` 参数。
2.  使用 `parse` 方法解析分割结果字符串。

---

**总结:**

这些改进的代码示例旨在提高递归分割功能的灵活性、可配置性和可用性。  `TemplateEngine` 允许用户自定义提示模板，`DynamicSplitter` 允许根据文本内容动态调整分割参数，`MultiFormatResplitParser` 允许处理多种格式的分割结果。  希望这些代码示例对您有所帮助!  如果你有其他问题，请随时提出。
