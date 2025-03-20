Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\chunking\recursive_splitter.py`

好的，重新开始，详细解释这段代码，生成代码片段并附带中文解释，包括代码用途和简单示例：

**代码概述**

这段代码定义了一系列用于文档处理的提示模板（Prompt Templates）和通信协议（Communication Protocols），这些协议主要用于文档摘要（Summarization）和分块（Chunking）。这些组件旨在与大型语言模型（LLMs）配合使用，以实现更结构化、可控的文档处理流程，特别是在检索增强生成（RAG）系统中。

**1. `chunk_summary_template`：块摘要模板**

```python
chunk_summary_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at document summarization."),
        ("user", """
# Source of the original text

The original text comes from {filename}。

# Original text

"partial original text":
{content}

# Task

Your task is to summarize the above "partial original text"

# Output

The output should contain the summary, do not add any redundant information.
""".strip()),
    ],
    input_variables=["filename", "content"],
)
```

*   **描述：** `chunk_summary_template` 是一个消息模板，用于指示 LLM 对文档的某个块（Chunk）进行摘要。它包含系统提示（System Prompt）和用户提示（User Prompt）。
*   **用途：** 用于对文档的每个分块生成简洁的摘要。
*   **组成：**
    *   `template`：包含对话消息的列表，每个消息是一个元组，指定角色（"system" 或 "user"）和消息内容。
    *   `input_variables`：一个列表，指定模板中需要填充的变量。 在这个例子中，`{filename}` 会被文件名填充，`{content}` 会被文档块的内容填充。
*   **示例用法：**

    ```python
    from pikerag.prompts.message_template import MessageTemplate

    # 假设有一个文件名和文档内容
    filename = "my_document.txt"
    content = "This is a sample chunk of text from the document."

    # 使用模板创建完整的提示
    prompt = chunk_summary_template.render(filename=filename, content=content)
    print(prompt) # 打印prompt，可用于后续LLM的调用
    ```

**2. `chunk_summary_refinement_template`：块摘要细化模板**

```python
chunk_summary_refinement_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at summary refinement."),
        ("user", """
# Source of the original text

The original text comes from {filename}。

# Original text

generalization of "partial original text":
{summary}

"partial original text":
{content}

# Task

Your task is to summarize the above "partial original text"

# Output

The output should contain the summary, do not add any redundant information.
""".strip()),
    ],
    input_variables=["filename", "summary", "content"],
)
```

*   **描述：** `chunk_summary_refinement_template` 用于细化已有的摘要。它接收一个已有的摘要（`summary`）和一个文档块（`content`），并要求 LLM 根据文档块的内容对摘要进行改进。
*   **用途：**  逐步改进摘要，使其更加准确和全面。
*   **组成：** 与 `chunk_summary_template` 类似，但增加了 `summary` 变量。
*   **示例用法：**

    ```python
    from pikerag.prompts.message_template import MessageTemplate

    # 假设有一个文件名、文档内容和已有的摘要
    filename = "my_document.txt"
    content = "This is another chunk of text from the same document."
    summary = "The document discusses document processing."

    # 使用模板创建完整的提示
    prompt = chunk_summary_refinement_template.render(filename=filename, summary=summary, content=content)
    print(prompt) # 打印prompt，可用于后续LLM的调用
    ```

**3. `chunk_resplit_template`：块重新分割模板**

```python
chunk_resplit_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at document chunking."),
        ("user", """
# Source of the original text

The original text comes from {filename}。

# Original text

generalization of "the first part" of "partial original text":
{summary}

"partial original text":
{content}

# Task

Your task:
1. Understand the generalization of "the first part" of "partial original text" and the "partial original text";
2. Analyse the structure of "partial original text", Split the "partial original text" strictly into "the first part" and "the second part", no content can be missing.
3. Provide the "end line number" of "the first part", pay attention that "the first part" is defined as: all the content of "partial original text" from Line "0" to Line "end line number" + 1, where empty is not allowed. Please note that here the maximum line number is {max_line_number}.
4. Summarize "the first part"。
5. For "the second part", considering the context and summarizing the main content of "the first part", please note that the content of "the first part" is defined as: all "partial original text" content after Line "end line number" + 1.

# Output

The output should strictly follow the format below, do not add any redundant information.

Thinking: According to the task requirements, carefully analyze the structure of the above "partial original text", think about how to reasonably split it into two parts, and output your thinking process.

<result>
<chunk>
  <endline>end line number, a non-negative number indicates the end line of "the first part". The first part will include this line.</endline>
  <summary>A summary of the "first part". Starting with "The main content of this part is". It can be referred to the generalization of "partial original text"</summary>
</chunk>
<chunk>
  <summary>Combine the context and the generalization of "the first part" to summarize the main content of "the second part". Starting with "The main content of this part is".</summary>
</chunk>
</result>
""".strip()),
    ],
    input_variables=["filename", "summary", "content", "max_line_number"],
)
```

*   **描述：** `chunk_resplit_template` 用于将一个文档块重新分割成两个更小的块。 它接收一个已有的摘要（`summary`），一个文档块（`content`），以及最大行号（`max_line_number`），并要求 LLM 分析文档块的结构，并确定一个分割点（`endline`），然后对分割后的两部分分别进行摘要。
*   **用途：**  动态地调整文档块的大小，以提高后续处理的效率和准确性。例如，可以确保每个块都包含一个完整的段落或主题。
*   **组成：** 除了 `filename`、`summary` 和 `content` 外，还增加了 `max_line_number` 变量。输出格式被明确定义为 `<result><chunk>...</chunk><chunk>...</chunk></result>`。
*   **示例用法：**

    ```python
    from pikerag.prompts.message_template import MessageTemplate

    # 假设有一个文件名、文档内容、已有的摘要和最大行号
    filename = "my_document.txt"
    content = "This is a long chunk of text. It contains two distinct topics. The first topic is about A. The second topic is about B."
    summary = "The document discusses two topics."
    max_line_number = 10  # 假设内容有 10 行

    # 使用模板创建完整的提示
    prompt = chunk_resplit_template.render(filename=filename, summary=summary, content=content, max_line_number=max_line_number)
    print(prompt)
    ```

**4. 通信协议（Communication Protocols）：`chunk_summary_protocol`, `chunk_summary_refinement_protocol`, `chunk_resplit_protocol`**

```python
from pikerag.prompts.base_parser import BaseContentParser
from pikerag.prompts.chunking.resplit_parser import ResplitParser
from pikerag.prompts.protocol import CommunicationProtocol

chunk_summary_protocol = CommunicationProtocol(
    template=chunk_summary_template,
    parser=BaseContentParser(),
)

chunk_summary_refinement_protocol = CommunicationProtocol(
    template=chunk_summary_refinement_template,
    parser=BaseContentParser(),
)

chunk_resplit_protocol = CommunicationProtocol(
    template=chunk_resplit_template,
    parser=ResplitParser(),
)
```

*   **描述：** 通信协议将提示模板与解析器（Parser）绑定在一起。 解析器用于处理 LLM 的输出，并将其转换为结构化数据。
*   **用途：**  定义了与 LLM 交互的标准方式，确保输入和输出的一致性。
*   **组成：**
    *   `template`：一个 `MessageTemplate` 对象。
    *   `parser`：一个解析器对象，用于解析 LLM 的输出。  `BaseContentParser` 是一个基本解析器，而 `ResplitParser` 是一个专门用于解析 `chunk_resplit_template` 输出的解析器（它需要提取分割点和摘要）。
*   **示例用法：**

    ```python
    from pikerag.prompts.protocol import CommunicationProtocol

    # 假设已经创建了一个 LLM 对象 (llm)
    # 假设已经准备好了输入数据 (filename, content, 等等)

    # 使用 chunk_summary_protocol 进行摘要
    input_data = {"filename": "my_document.txt", "content": "This is a chunk of text."}
    llm_response = "The document discusses this text."  # 假设这是 LLM 的响应
    protocol = chunk_summary_protocol
    prompt = protocol.template.render(**input_data)
    # llm_response = llm(prompt) # 使用LLM生成response
    parsed_output = protocol.parser.parse(llm_response)
    print(parsed_output) # 输出解析后的结果
    ```

**总结**

这段代码提供了一套工具，用于构建基于 LLM 的文档处理流程。 通过使用提示模板和通信协议，可以更有效地控制 LLM 的行为，并获得结构化的输出结果。这些工具特别适用于需要对文档进行分块、摘要和细化的场景，例如在检索增强生成（RAG）系统中。`ResplitParser`  是一个关键组件，用于从 `chunk_resplit_template` 的输出中提取分割点，从而实现动态的文档分块。

希望这个更详细的解释对你有所帮助！
