Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\chunking\resplit_parser.py`

```python
import traceback
from typing import Tuple

from bs4 import BeautifulSoup

from pikerag.prompts.base_parser import BaseContentParser
from pikerag.utils.lxml_parser import get_soup_from_content


class LinedText:
    def __init__(self, text: str) -> None:
        self.text = text
        self.lines = text.split("\n")
        self.max_line_number = len(self.lines) - 1

    @property
    def lined_text(self):
        return "\n".join([f"Line {i} \t {line}" for i, line in enumerate(self.lines)])

    def get_lines_text(self, start_line: int, end_line: int) -> Tuple[str, str]:
        return "\n".join(self.lines[start_line:end_line])


# 定义一个文本分块解析器
class ResplitParser(BaseContentParser):
    def __init__(self) -> None:
        self._encoded = False

    # 编码函数，将文本转换为带有行号的文本
    def encode(self, content: str, **kwargs) -> Tuple[str, dict]:
        """
        将输入的文本内容编码成带有行号的文本，并返回最大行号。

        Args:
            content (str): 输入的文本内容。
            **kwargs: 其他可选参数。

        Returns:
            Tuple[str, dict]: 带有行号的文本和包含最大行号的字典。
        """
        self.text = content
        self.lined_text = LinedText(self.text)
        self._encoded = True
        return self.lined_text.lined_text, {"max_line_number": self.lined_text.max_line_number}

    # 解码函数，从带有XML标签的内容中提取文本块和摘要
    def decode(self, content: str, **kwargs) -> Tuple[str, str, str, int]:
        """
        从带有XML标签的内容中解码出第一个文本块、两个文本块的摘要以及第一个文本块的长度。

        Args:
            content (str): 带有XML标签的文本内容。
            **kwargs: 其他可选参数。

        Returns:
            Tuple[str, str, str, int]: 第一个文本块，第一个文本块的摘要，第二个文本块的摘要，第一个文本块的长度。
        """
        assert self._encoded is True

        try:
            # 使用BeautifulSoup解析XML内容
            soup: BeautifulSoup = get_soup_from_content(content, tag="result")
            assert soup is not None, f"Designed tag not exist in response, please refine prompt"

            # 找到所有的chunk标签
            chunk_soups = soup.find_all("chunk")
            assert len(chunk_soups) == 2, f"There should be exactly 2 chunks in response, Please refine prompt"

            # 获取第一个文本块的结束行号
            first_chunk_endline_str: str = chunk_soups[0].find("endline").text
            # 检查结束行号是否为空或包含特定字符串
            if (
                len(first_chunk_endline_str) == 0
                or "not applicable" in first_chunk_endline_str.lower()  # TODO: update the prompt to limit the output
                or "not included" in first_chunk_endline_str.lower()
            ):
                first_chunk = ""
                dropped_len = 0
            else:
                # 获取第一个文本块的结束行号
                first_chunk_endline = int(first_chunk_endline_str)
                # 获取第一个文本块的内容
                first_chunk = self.lined_text.get_lines_text(0, first_chunk_endline + 1)
                # 找到第一个文本块在原始文本中的起始位置
                first_chunk_start_pos = self.text.find(first_chunk)
                assert first_chunk_start_pos != -1, f"first chunk not exist?"
                # 计算第一个文本块的长度
                dropped_len = first_chunk_start_pos + len(first_chunk)

            # 获取第一个和第二个文本块的摘要
            first_chunk_summary = chunk_soups[0].find("summary").text
            second_chunk_summary = chunk_soups[1].find("summary").text

        except Exception as e:
            # 打印错误信息
            print("Content:")
            print(content)
            print("Input Text:")
            print(self.lined_text.lined_text)
            print("Exception:")
            print(e)
            traceback.print_exc()
            exit(0)

        # 返回结果
        return first_chunk, first_chunk_summary, second_chunk_summary, dropped_len


# 示例用法
if __name__ == '__main__':
    # 创建一个ResplitParser对象
    parser = ResplitParser()

    # 示例文本内容
    content = "This is the first line.\nThis is the second line.\nThis is the third line."

    # 编码文本内容
    encoded_content, metadata = parser.encode(content)
    print("Encoded Content:")
    print(encoded_content)
    print("Metadata:", metadata)

    # 模拟带有XML标签的响应内容
    xml_content = """
    <result>
        <chunk>
            <endline>1</endline>
            <summary>Summary of the first chunk.</summary>
        </chunk>
        <chunk>
            <endline>not applicable</endline>
            <summary>Summary of the second chunk.</summary>
        </chunk>
    </result>
    """

    # 解码XML内容
    parser.text = content  # 确保parser知道原始文本
    parser.lined_text = LinedText(content) #确保parser知道原始文本
    parser._encoded = True # 确保 _encoded 为 True
    first_chunk, first_summary, second_summary, dropped_len = parser.decode(xml_content)

    # 打印解码结果
    print("\nDecoded Results:")
    print("First Chunk:", first_chunk)
    print("First Summary:", first_summary)
    print("Second Summary:", second_summary)
    print("Dropped Length:", dropped_len)
```

**代码解释：**

1.  **`LinedText` 类:**
    *   **功能:**  负责将文本分割成行，并为每一行添加行号。 这有助于在处理文本块时跟踪其原始位置。
    *   **`__init__(self, text: str)`:** 构造函数，初始化 `text` 属性为输入的文本，并使用 `split("\n")` 将文本按行分割，存储在 `lines` 属性中。`max_line_number` 记录了最大行数 (行数 - 1，因为行号从 0 开始)。
    *   **`lined_text` property:**  返回带有行号的格式化文本。 例如：`"Line 0 \t This is the first line."`。
    *   **`get_lines_text(self, start_line: int, end_line: int)`:**  根据给定的起始行号和结束行号，提取文本中的一部分。

2.  **`ResplitParser` 类:**
    *   **功能:** 实现文本的编码和解码，编码过程为文本添加行号，解码过程将带有XML标签的文本内容解析为文本块和摘要。这个类是用于处理文本分割的核心部分。
    *   **`__init__(self) -> None`:**  构造函数，初始化 `_encoded` 属性为 `False`，表示文本尚未编码。
    *   **`encode(self, content: str, **kwargs) -> Tuple[str, dict]`:**
        *   **功能:** 编码函数，将输入的文本添加行号。
        *   **参数:**
            *   `content (str)`: 要编码的原始文本。
            *   `**kwargs`: 允许传递其他参数 (目前未使用)。
        *   **步骤:**
            1.  将输入的 `content` 存储到 `self.text` 属性。
            2.  创建 `LinedText` 对象，将文本按行分割并添加行号。
            3.  设置 `self._encoded = True`，表示文本已编码。
            4.  返回带有行号的文本 (`self.lined_text.lined_text`) 以及包含最大行号的字典。
    *   **`decode(self, content: str, **kwargs) -> Tuple[str, str, str, int]`:**
        *   **功能:** 解码函数，从带有XML标签的内容中提取文本块和摘要。
        *   **参数:**
            *   `content (str)`: 带有XML标签的响应文本。 这个文本应该包含 `<result>` 标签，其中包含两个 `<chunk>` 标签，每个标签包含 `<endline>` (结束行号) 和 `<summary>` (摘要) 标签。
            *   `**kwargs`: 允许传递其他参数 (目前未使用)。
        *   **步骤:**
            1.  **断言 `self._encoded is True`:**  确保在解码之前已经执行了编码。
            2.  **使用 `BeautifulSoup` 解析 XML:**  使用 `get_soup_from_content` 函数解析带有 "result" 标签的 XML 内容。
            3.  **查找 `chunk` 标签:** 找到所有 `<chunk>` 标签，并断言存在两个 `chunk`。
            4.  **提取第一个文本块：**
                *   提取第一个 `chunk` 中的 `<endline>` 标签的值，该值表示第一个文本块的结束行号。
                *   如果 `<endline>` 的值为空、包含 "not applicable" 或 "not included"，则将 `first_chunk` 设置为空字符串，`dropped_len` 设置为 0。
                *   否则，将 `<endline>` 的值转换为整数，并使用 `self.lined_text.get_lines_text(0, first_chunk_endline + 1)` 提取从第一行到 `first_chunk_endline` 行的文本。
                *   使用 `self.text.find(first_chunk)` 找到 `first_chunk` 在原始文本中的起始位置，并断言找到了该文本块。
                *   计算 `dropped_len`，表示第一个文本块的长度。
            5.  **提取摘要：**  从两个 `chunk` 标签中提取 `<summary>` 标签的值。
            6.  **异常处理:**  如果发生任何异常，打印错误信息和堆栈跟踪，然后退出程序。
            7.  **返回结果:** 返回提取的 `first_chunk`，`first_chunk_summary`，`second_chunk_summary` 和 `dropped_len`。

**使用场景:**

这个 `ResplitParser` 类旨在与大型语言模型 (LLM) 一起使用，特别是在需要将长文档分割成块并为每个块生成摘要的情况下。  LLM 可能会返回包含文本块和摘要的 XML 响应，`ResplitParser` 负责解析这些响应并提取相关信息。

**示例流程:**

1.  **准备:** 你有一个长文档，你需要将其分割成块并为每个块生成摘要。
2.  **编码:** 使用 `ResplitParser.encode()` 方法对文档进行编码，将文档转换为带有行号的文本。
3.  **传递给 LLM:** 将带有行号的文本传递给 LLM。  LLM 将执行以下操作：
    *   将文本分割成两个块。
    *   为每个块生成摘要。
    *   返回包含文本块的结束行号和摘要的 XML 响应。
4.  **解码:** 使用 `ResplitParser.decode()` 方法解析 LLM 返回的 XML 响应。`decode` 方法将提取第一个文本块、两个文本块的摘要和第一个文本块的长度。
5.  **使用结果:**  现在你可以使用提取的文本块和摘要进行下游任务，例如问答、文档检索等。

这段代码的核心在于它能够与LLM的输出协同工作，从结构化的响应中提取有用的信息。  `LinedText` 类使得跟踪文本的原始位置变得更加容易，从而确保提取的文本块是正确的。  异常处理机制可以帮助你识别和解决在解析过程中可能出现的问题。
