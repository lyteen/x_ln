Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\chunking\recursive_splitter.py`

Alright, let's try this again with a focus on creating smaller, more modular pieces of code and providing explanations in Chinese.

**1. 改进的 `BaseContentParser`:**

```python
class ImprovedBaseContentParser:
    """
    一个简单的内容解析器，用于提取文本内容.
    A simple content parser for extracting text content.
    """

    def parse(self, text: str) -> str:
        """
        直接返回输入文本.  对于更复杂的解析，可以扩展这个方法.
        Directly returns the input text.  For more complex parsing, you can extend this method.
        """
        return text

    def format(self, text: str) -> str:
        """
        格式化输出文本，使其更易读. 可以根据需要进行定制.
        Formats the output text for better readability. Can be customized as needed.
        """
        return text.strip()
```

**描述:**

这个 `ImprovedBaseContentParser` 类提供了一个基本的文本解析功能。  它简单地返回输入的文本，但是您可以扩展 `parse` 方法来处理更复杂的任务，比如提取特定部分的内容或者进行数据清洗。`format`方法则用于格式化输出，例如去除首尾空格。

**2. 改进的 `ResplitParser` (更健壮的解析):**

```python
import re

class ImprovedResplitParser:
    """
    用于解析 resplit 任务输出的解析器。  更健壮的处理逻辑.
    A parser for parsing the output of resplit tasks.  More robust handling logic.
    """

    def parse(self, text: str) -> dict:
        """
        解析文本，提取 "endline" 和 "summary" 信息.
        Parses the text to extract "endline" and "summary" information.

        Args:
            text: 包含 resplit 结果的文本. Text containing the resplit results.

        Returns:
            一个包含 "chunk1" 和 "chunk2" 的字典，每个 chunk 包含 "endline" 和 "summary".
            A dictionary containing "chunk1" and "chunk2", each containing "endline" and "summary".
        """
        try:
            result = {}

            # 提取第一个 chunk 的信息 (Extract information for the first chunk)
            chunk1_match = re.search(r"<chunk>\s*<endline>(.*?)</endline>\s*<summary>(.*?)</summary>\s*</chunk>", text, re.DOTALL)
            if chunk1_match:
                result["chunk1"] = {
                    "endline": int(chunk1_match.group(1).strip()),  # 转换为整数 (Convert to integer)
                    "summary": chunk1_match.group(2).strip()
                }
            else:
                raise ValueError("未找到第一个 chunk 的信息 (Information for the first chunk not found)")

            # 提取第二个 chunk 的信息 (Extract information for the second chunk)
            chunk2_match = re.search(r"<chunk>\s*<summary>(.*?)</summary>\s*</chunk>", text[chunk1_match.end():] if chunk1_match else text, re.DOTALL)
            if chunk2_match:
                result["chunk2"] = {
                    "summary": chunk2_match.group(1).strip()
                }
            else:
                raise ValueError("未找到第二个 chunk 的信息 (Information for the second chunk not found)")

            return result

        except Exception as e:
            print(f"解析错误: {e}.  原始文本: {text}")  # 打印错误信息和原始文本 (Print error message and original text)
            return {}  # 或者抛出异常 (or raise the exception)

    def format(self, data: dict) -> str:
        """
        格式化解析结果为字符串. Formats the parsed results into a string.
        """
        if not data:
            return "解析失败 (Parsing failed)"

        formatted_text = ""
        if "chunk1" in data:
            formatted_text += f"Chunk 1:\n"
            formatted_text += f"  Endline: {data['chunk1'].get('endline', 'N/A')}\n"
            formatted_text += f"  Summary: {data['chunk1'].get('summary', 'N/A')}\n"
        if "chunk2" in data:
            formatted_text += f"Chunk 2:\n"
            formatted_text += f"  Summary: {data['chunk2'].get('summary', 'N/A')}\n"

        return formatted_text
```

**描述:**

这个 `ImprovedResplitParser` 类用于解析 LLM 执行 resplit 任务的输出。

**主要改进:**

*   **更健壮的正则表达式:** 使用 `re.DOTALL` 允许 `.` 匹配换行符，可以处理更复杂的输出格式。
*   **错误处理:**  包含 `try...except` 块来捕获解析错误，并提供更有用的错误消息。  这使得调试更容易。
*   **更清晰的结构:**  代码更清晰地组织，更易于阅读和维护。
*   **更强的容错性:** `format` 方法包含 `get` 方法，保证即使某个字段缺失，也不会报错。
*   **使用正则匹配第二个chunk：** 通过正则匹配`<chunk>\s*<summary>(.*?)</summary>\s*</chunk>`来提取第二个chunk的内容

**3. MessageTemplate 和 CommunicationProtocol (保持不变，但可以使用新的解析器):**

```python
from typing import List, Tuple, Dict

class MessageTemplate:
    """
    表示消息模板.  Represents a message template.
    """

    def __init__(self, template: List[Tuple[str, str]], input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def render(self, **kwargs) -> List[Dict[str, str]]:
        """
        使用给定的变量渲染模板.  Renders the template using the given variables.
        """
        messages = []
        for role, content in self.template:
            # 替换变量 (Replace variables)
            for var in self.input_variables:
                content = content.replace("{" + var + "}", str(kwargs.get(var, "")))  # 默认值为空字符串 (Default to empty string)
            messages.append({"role": role, "content": content})
        return messages

class CommunicationProtocol:
    """
    定义与语言模型的通信协议.  Defines the communication protocol with the language model.
    """

    def __init__(self, template: MessageTemplate, parser: ImprovedBaseContentParser): # 使用 ImprovedBaseContentParser
        self.template = template
        self.parser = parser

    def generate_message(self, **kwargs) -> List[Dict[str, str]]:
        """
        从模板生成消息.  Generates a message from the template.
        """
        return self.template.render(**kwargs)

    def parse_response(self, response: str) -> str:
        """
        解析语言模型的响应.  Parses the response from the language model.
        """
        parsed_content = self.parser.parse(response)
        return self.parser.format(parsed_content)
```

**描述:**

`MessageTemplate` 和 `CommunicationProtocol` 的定义基本保持不变。  关键在于 `CommunicationProtocol` 现在使用我们改进后的 `ImprovedBaseContentParser` 或 `ImprovedResplitParser` 。

**4. 更新后的模板 (Templates) - 可选的，但可以使提示语更清晰:**

```python
chunk_summary_template = MessageTemplate(
    template=[
        ("system", "你是一个擅长文档摘要的助手 (You are a helpful assistant good at document summarization)."),
        ("user", """
# 原始文本来源 (Source of the original text)

原始文本来自 {filename} (The original text comes from {filename})。

# 原始文本 (Original text)

"部分原始文本" (partial original text):
{content}

# 任务 (Task)

你的任务是总结上面的 "部分原始文本" (Your task is to summarize the above "partial original text")

# 输出 (Output)

输出应该包含摘要，不要添加任何冗余信息 (The output should contain the summary, do not add any redundant information).
""".strip()),
    ],
    input_variables=["filename", "content"],
)


chunk_summary_refinement_template = MessageTemplate(
    template=[
        ("system", "你是一个擅长摘要精炼的助手 (You are a helpful AI assistant good at summary refinement)."),
        ("user", """
# 原始文本来源 (Source of the original text)

原始文本来自 {filename} (The original text comes from {filename})。

# 原始文本 (Original text)

"部分原始文本" 的概括 (generalization of "partial original text"):
{summary}

"部分原始文本" (partial original text):
{content}

# 任务 (Task)

你的任务是总结上面的 "部分原始文本" (Your task is to summarize the above "partial original text")

# 输出 (Output)

输出应该包含摘要，不要添加任何冗余信息 (The output should contain the summary, do not add any redundant information).
""".strip()),
    ],
    input_variables=["filename", "summary", "content"],
)


chunk_resplit_template = MessageTemplate(
    template=[
        ("system", "你是一个擅长文档分块的助手 (You are a helpful AI assistant good at document chunking)."),
        ("user", """
# 原始文本来源 (Source of the original text)

原始文本来自 {filename} (The original text comes from {filename})。

# 原始文本 (Original text)

"部分原始文本" 的 "第一部分" 的概括 (generalization of "the first part" of "partial original text"):
{summary}

"部分原始文本" (partial original text):
{content}

# 任务 (Task)

你的任务:
1. 理解 "部分原始文本" 的 "第一部分" 的概括和 "部分原始文本" 本身 (Understand the generalization of "the first part" of "partial original text" and the "partial original text");
2. 分析 "部分原始文本" 的结构，严格地将其分为 "第一部分" 和 "第二部分"，不能丢失任何内容 (Analyse the structure of "partial original text", Split the "partial original text" strictly into "the first part" and "the second part", no content can be missing).
3. 提供 "第一部分" 的 "结束行号" (Provide the "end line number" of "the first part")，注意 "第一部分" 定义为：从第 "0" 行到 "结束行号" + 1 行的所有内容，不允许为空 (pay attention that "the first part" is defined as: all the content of "partial original text" from Line "0" to Line "end line number" + 1, where empty is not allowed)。注意这里的最大行号是 {max_line_number} (Please note that here the maximum line number is {max_line_number}).
4. 总结 "第一部分" (Summarize "the first part")。
5. 对于 "第二部分" (For "the second part")，考虑上下文和 "第一部分" 的主要内容来总结 "第二部分" (considering the context and summarizing the main content of "the first part", please note that the content of "the first part" is defined as: all "partial original text" content after Line "end line number" + 1)。

# 输出 (Output)

输出应严格遵循以下格式，不要添加任何冗余信息 (The output should strictly follow the format below, do not add any redundant information)。

思考过程 (Thinking): 根据任务要求，仔细分析上面的 "部分原始文本" 的结构，思考如何合理地将其分成两部分，并输出你的思考过程 (According to the task requirements, carefully analyze the structure of the above "partial original text", think about how to reasonably split it into two parts, and output your thinking process)。

<result>
<chunk>
  <endline>结束行号 (end line number), 一个非负数表示 "第一部分" 的结束行。"第一部分" 将包含这一行 (a non-negative number indicates the end line of "the first part". The first part will include this line)。</endline>
  <summary>"第一部分" 的摘要 (A summary of the "first part")。以 "本部分的主要内容是 (The main content of this part is)" 开头 (Starting with "The main content of this part is")。可以参考 "部分原始文本" 的概括 (It can be referred to the generalization of "partial original text")</summary>
</chunk>
<chunk>
  <summary>结合上下文和 "第一部分" 的概括来总结 "第二部分" 的主要内容 (Combine the context and the generalization of "the first part" to summarize the main content of "the second part")。以 "本部分的主要内容是 (The main content of this part is)" 开头 (Starting with "The main content of this part is").</summary>
</chunk>
</result>
""".strip()),
    ],
    input_variables=["filename", "summary", "content", "max_line_number"],
)
```

**描述:**

这些模板定义了发送给 LLM 的提示语。  我已经将英文注释翻译成中文，并稍微调整了格式，使其更清晰。

**5. 实例化协议 (Instantiating the Protocols):**

```python
chunk_summary_protocol = CommunicationProtocol(
    template=chunk_summary_template,
    parser=ImprovedBaseContentParser(), # 使用改进的解析器
)

chunk_summary_refinement_protocol = CommunicationProtocol(
    template=chunk_summary_refinement_template,
    parser=ImprovedBaseContentParser(), # 使用改进的解析器
)

chunk_resplit_protocol = CommunicationProtocol(
    template=chunk_resplit_template,
    parser=ImprovedResplitParser(), # 使用改进的解析器
)
```

**描述:**

这里我们创建了 `CommunicationProtocol` 的实例，并使用我们改进后的解析器。

**6. 示例使用 (Example Usage):**

```python
# 示例数据 (Example data)
filename = "example.txt"
content = """
This is the first line.
This is the second line.
This is the third line.
This is the fourth line.
This is the fifth line.
"""
summary = "A short summary of the first few lines."
max_line_number = 4

# Resplit 示例 (Resplit example)
resplit_message = chunk_resplit_protocol.generate_message(filename=filename, summary=summary, content=content, max_line_number=max_line_number)
print("Resplit 消息 (Resplit Message):", resplit_message)

# 模拟 LLM 响应 (Simulate LLM response)
llm_response = """
Thinking: I think the first two lines form a coherent unit.

<result>
<chunk>
  <endline>1</endline>
  <summary>The main content of this part is the first two lines of the text.</summary>
</chunk>
<chunk>
  <summary>The main content of this part is the remaining lines after the first two.</summary>
</chunk>
</result>
"""

# 解析 LLM 响应 (Parse LLM response)
parsed_resplit_response = chunk_resplit_protocol.parse_response(llm_response)
print("解析后的 Resplit 响应 (Parsed Resplit Response):", parsed_resplit_response)

# Summary 示例 (Summary example)
summary_message = chunk_summary_protocol.generate_message(filename=filename, content=content)
print("Summary 消息 (Summary Message):", summary_message)

# 模拟 LLM 响应 (Simulate LLM response)
llm_summary_response = "The text describes a short example with multiple lines."

# 解析 LLM 响应 (Parse LLM response)
parsed_summary_response = chunk_summary_protocol.parse_response(llm_summary_response)
print("解析后的 Summary 响应 (Parsed Summary Response):", parsed_summary_response)
```

**描述:**

这段代码演示了如何使用我们定义的 `CommunicationProtocol` 和 `MessageTemplate` 来生成消息，并解析 LLM 的响应。  它模拟了 LLM 的输出，并使用 `parse_response` 方法来提取有用的信息。

**总结:**

This revised response provides:

*   **Smaller, more modular code:** Each class has a clear purpose.
*   **Improved robustness:** The `ResplitParser` now handles errors better.
*   **Clearer structure:** The code is organized for readability.
*   **Chinese explanations:**  Comments and descriptions are in Chinese.
*   **Example usage:**  The example code demonstrates how to use the components.

This approach makes the code easier to understand, maintain, and extend.  The improved `ResplitParser` will be more resilient to variations in LLM output.  Remember to adapt the example usage to your specific needs and LLM.