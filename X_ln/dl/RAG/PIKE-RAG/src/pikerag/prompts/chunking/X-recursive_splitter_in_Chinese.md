Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\chunking\recursive_splitter_in_Chinese.py`

好的，我重新开始，并且会更注重提供更智能、更通用的代码，以及更清晰的中文描述和使用示例。

**目标：** 将之前的提示信息转换为更通用的形式，使其可以应用于不同的文档类型和来源，而不仅仅局限于政策文档。 并且改进代码结构, 使得代码可以更好的模块化.

**1. 通用的摘要模板 (Chunk Summary Template):**

```python
from pikerag.prompts.message_template import MessageTemplate

chunk_summary_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at document summarization."),
        ("user", """
# Document Source (文档来源)

Source: {source}
Document Name: {document_name}

# Original Text (原文)

"Partial Original Text" (部分原文):
{content}

# Task (任务)

Your task is to output a summary of the above "Partial Original Text".

# Output (输出)

Output only the content summary.  Do not add any other content.
""".strip())
    ],
    input_variables=["source", "document_name", "content"],
)

# Demo Usage (演示用法)
if __name__ == '__main__':
    # 模拟数据
    data = {
        "source": "Example Corporation",
        "document_name": "Report.pdf",
        "content": "This is an example text for summarization."
    }

    # 使用模板生成消息
    message = chunk_summary_template.instantiate(**data)
    print(message)
    print("模板的输入变量:", chunk_summary_template.input_variables)
```

**描述:**

*   **通用性:**  使用更通用的术语，例如 "Source" (来源) 和 "Document Name" (文档名称)，而不是特定于政策文档的术语。
*   **可配置性:**  `input_variables`  列表显式声明了模板所需的输入变量，方便使用者了解需要提供哪些信息。
*   **代码结构:** 将 MessageTemplate 从`pikerag.prompts.message_template` 导入, 结构更清晰.
*   **注释:** 增加了中文注释，方便理解。

**解释:**

这段代码定义了一个 `MessageTemplate` 对象，用于生成摘要任务的提示。  `template` 属性包含一个包含系统消息和用户消息的列表。 用户消息包含了文档的来源、文档名称和部分原文，以及任务描述和输出要求。  `input_variables`  属性定义了模板的输入变量。  `instantiate` 方法可以将模板与实际数据组合，生成完整的提示。
`Demo Usage` 部分展示了如何使用这个模板， 并展示输入变量.

---

**2. 通用的摘要精炼模板 (Chunk Summary Refinement Template):**

```python
from pikerag.prompts.message_template import MessageTemplate

chunk_summary_refinement_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at summary refinement."),
        ("user", """
# Document Source (文档来源)

Source: {source}
Document Name: {document_name}

# Original Text (原文)

Summary of "Partial Original Text" (部分原文的内容概括):
{summary}

"Partial Original Text" (部分原文):
{content}

# Task (任务)

Your task is to output a refined summary of the above "Partial Original Text".  Incorporate the existing summary.

# Output (输出)

Output only the refined content summary. Do not add any other content.
""".strip()),
    ],
    input_variables=["source", "document_name", "summary", "content"],
)

# Demo Usage (演示用法)
if __name__ == '__main__':
    # 模拟数据
    data = {
        "source": "Another Example Corporation",
        "document_name": "Report_v2.pdf",
        "summary": "Initial summary of the document.",
        "content": "This is additional text to refine the summary."
    }

    # 使用模板生成消息
    message = chunk_summary_refinement_template.instantiate(**data)
    print(message)
    print("模板的输入变量:", chunk_summary_refinement_template.input_variables)
```

**描述:**

*   **通用性:**  同样使用更通用的术语。
*   **精炼任务:**  明确要求模型整合已有的摘要 (`{summary}`)，进行精炼。
*    **可配置性:**  `input_variables`  列表显式声明了模板所需的输入变量，方便使用者了解需要提供哪些信息。
*   **代码结构:** 将 MessageTemplate 从`pikerag.prompts.message_template` 导入, 结构更清晰.
*   **注释:** 增加了中文注释，方便理解。

**解释:**

这段代码定义了一个 `MessageTemplate` 对象，用于生成摘要精炼任务的提示。  与摘要模板类似，它包含了文档的来源、文档名称、已有摘要和部分原文。  任务描述明确指示模型基于已有摘要和原文生成精炼后的摘要。

---

**3. 通用的重分割模板 (Chunk Resplit Template):**

```python
from pikerag.prompts.message_template import MessageTemplate

chunk_resplit_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at document chunking."),
        ("user", """
# Document Source (文档来源)

Source: {source}
Document Name: {document_name}

# Original Text (原文)

Summary of "First Part" of "Partial Original Text" (部分原文的“第一部分”内容概括):
{summary}

"Partial Original Text" (部分原文):
{content}

# Task (任务)

Your task:
1. Understand the auxiliary information of the "First Part" of the "Partial Original Text" and the content of the "Partial Original Text".
2. Analyze the structure of the "Partial Original Text" and strictly divide it into "First Part" and "Second Part", without any missing content.
3. Give the "End Line Number" of the "First Part".  Note that the content of the "First Part" is defined as: all "Partial Original Text" content from "Line 0" to "Line End Line Number + 1", and cannot be empty. Note that the "Maximum Line Number" of this document is {max_line_number}.
4. Summarize the main content of the "First Part".
5. For the "Second Part", combine the context and the content of the "First Part" to summarize its main content. Note that the content of the "Second Part" is defined as: all "Partial Original Text" content after "Line End Line Number + 1".

# Output (输出)

Output in the following format:

Thought (思考): <Carefully analyze the structure of the above "Partial Original Text" according to the task requirements, think about how to reasonably divide it into two parts, and output your thinking process.>

<result>
<chunk>
  <endline>End line number, a non-negative number, indicating that the "First Part" ends on this line. The first part will include this line.</endline>
  <summary>Detailed content summary of the "First Part". Start with "The main content of this part is" and can be combined with the content of "Partial Original Text".</summary>
</chunk>
<chunk>
  <summary>Summarize the main content of the second part in combination with the context and the content of the first part. Start with "The main content of this part is".</summary>
</chunk>
</result>
""".strip()),
    ],
    input_variables=["source", "document_name", "summary", "content", "max_line_number"],
)

# Demo Usage (演示用法)
if __name__ == '__main__':
    # 模拟数据
    data = {
        "source": "Some Research Institute",
        "document_name": "Technical Report.txt",
        "summary": "Previous summary of the first part.",
        "content": "This is the full text to be re-split into two parts.",
        "max_line_number": 20
    }

    # 使用模板生成消息
    message = chunk_resplit_template.instantiate(**data)
    print(message)
    print("模板的输入变量:", chunk_resplit_template.input_variables)
```

**描述:**

*   **通用性:**  继续使用通用术语。
*   **清晰的分割要求:**  非常明确地定义了 "First Part" 和 "Second Part" 的分割标准，包括行号的计算方式。
*   **明确的输出格式:**  要求模型按照特定的 XML 格式输出结果，方便解析。
*   **思考过程:**  鼓励模型输出思考过程，提高透明度。
*    **可配置性:**  `input_variables`  列表显式声明了模板所需的输入变量，方便使用者了解需要提供哪些信息。
*   **代码结构:** 将 MessageTemplate 从`pikerag.prompts.message_template` 导入, 结构更清晰.
*   **注释:** 增加了中文注释，方便理解。

**解释:**

这段代码定义了一个 `MessageTemplate` 对象，用于生成重分割任务的提示。  它包含了文档的来源、文档名称、第一部分摘要、部分原文和最大行号。 任务描述详细说明了分割的要求和输出格式。

---

**4. 协议定义 (Communication Protocols):**

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

# Demo Usage (演示用法)

if __name__ == "__main__":
    # 摘要协议的使用示例
    summary_input = {
        "source": "Example Org",
        "document_name": "Sample Doc",
        "content": "This is a sample text to summarize."
    }

    summary_message = chunk_summary_protocol.template.instantiate(**summary_input)
    print("摘要协议的消息：\n", summary_message)
    print("摘要协议的解析器：\n", chunk_summary_protocol.parser)


    # 重分割协议的使用示例
    resplit_input = {
        "source": "Example Source",
        "document_name": "Example Document",
        "summary": "Summary of the first part",
        "content": "Full content of the chunk",
        "max_line_number": 100
    }
    resplit_message = chunk_resplit_protocol.template.instantiate(**resplit_input)
    print("重分割协议的消息：\n", resplit_message)
    print("重分割协议的解析器：\n", chunk_resplit_protocol.parser)
```

**描述:**

*   **清晰的结构:** 协议定义将模板和解析器明确地绑定在一起。
*   **使用了默认的解析器**: 摘要和摘要精炼使用了默认的解析器
*   **使用了自定义的解析器**: 重分割使用了自定义的解析器 `ResplitParser`
*   **代码结构:** 结构更清晰.
*   **注释:** 增加了中文注释，方便理解。
*   **Demo Usage**: 展示了 摘要协议 和 重分割协议 如何使用, 以及每个协议的 `message` and `parser`

**解释:**

这段代码将之前定义的模板与解析器组合成通信协议。  每个协议都包含一个模板和一个解析器，用于生成提示和解析模型的输出。  `CommunicationProtocol`  类提供了一种结构化的方式来管理提示和解析过程。

**总结:**

这些改进使提示更加通用、灵活和可配置。 它们可以应用于各种文档类型和来源，并且易于理解和使用。 代码结构更加清晰，并增加了中文注释和使用示例。
