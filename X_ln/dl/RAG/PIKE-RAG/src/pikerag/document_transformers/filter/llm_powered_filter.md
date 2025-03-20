Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\filter\llm_powered_filter.py`

Okay, restarting to provide a more detailed explanation, code snippets with descriptions in Chinese, and a usage demo. Let's break down the `LLMPoweredFilter` class.

**1. `LLMPoweredFilter` Class Definition:**

```python
from typing import Any, List, Sequence, Tuple

from tqdm import tqdm

from langchain_core.documents import Document, BaseDocumentTransformer

from pikerag.llm_client import BaseLLMClient
from pikerag.prompts import CommunicationProtocol
from pikerag.utils.logger import Logger


class LLMPoweredFilter(BaseDocumentTransformer):
    NAME = "LLMPoweredFilter"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        filter_protocol: CommunicationProtocol,
        llm_config: dict = {},
        logger: Logger = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self._llm_client = llm_client
        self._llm_config = llm_config

        self._filter_protocol: CommunicationProtocol = filter_protocol

        self.logger = logger
```

**Description (描述):**

*   **Imports (导入):** 导入必要的库，包括类型提示 (`typing`)、进度条 (`tqdm`)、Langchain 的文档处理类 (`langchain_core.documents`)、自定义的 LLM 客户端 (`pikerag.llm_client`)、通信协议 (`pikerag.prompts`) 和日志记录器 (`pikerag.utils.logger`).
*   **`LLMPoweredFilter` Class (类定义):** 定义一个名为 `LLMPoweredFilter` 的类，它继承自 `BaseDocumentTransformer`。这个类用于过滤文档，基于 LLM 的判断来决定哪些文档应该保留。
*   **`NAME` (名称):**  定义一个常量 `NAME`，用于标识这个类的名称。
*   **`__init__` (构造函数):**  初始化 `LLMPoweredFilter` 类的实例。接收以下参数：
    *   `llm_client`: 一个 `BaseLLMClient` 实例，用于与 LLM 交互。
    *   `filter_protocol`: 一个 `CommunicationProtocol` 实例，定义了与 LLM 交互的格式（提示语等）。
    *   `llm_config`: 一个字典，包含传递给 LLM 客户端的配置信息。
    *   `logger`: 一个 `Logger` 实例，用于记录日志。
    *   `**kwargs`: 额外的关键字参数。
    在构造函数中，初始化了 LLM 客户端、LLM 配置、过滤协议和日志记录器。`super().__init__()` 调用父类的构造函数。

**Translation (中文翻译):**

*   这个类 `LLMPoweredFilter` 继承自 Langchain 的 `BaseDocumentTransformer`，它的作用是使用大型语言模型 (LLM) 来过滤文档。
*   构造函数接收一个 LLM 客户端、一个通信协议、LLM 配置和一个日志记录器作为参数。这些参数用于配置 LLM 的行为和记录日志。

**Usage Demo (使用示例):**

```python
from pikerag.llm_client import OpenAIClient  # 假设有 OpenAIClient
from pikerag.prompts import FilterProtocol  # 假设有 FilterProtocol

# 假设你已经有了一个 OpenAIClient 和一个 FilterProtocol
# 并且已经设置好了 OpenAI 的 API 密钥
llm_client = OpenAIClient(api_key="YOUR_OPENAI_API_KEY")  # 替换为你的 OpenAI API 密钥
filter_protocol = FilterProtocol(prompt_template="Is this document relevant to the query: {query}? Document: {document}") # 替换为你的 prompt
llm_config = {"model_name": "gpt-3.5-turbo"}

# 创建 LLMPoweredFilter 实例
filter_instance = LLMPoweredFilter(
    llm_client=llm_client,
    filter_protocol=filter_protocol,
    llm_config=llm_config
)

# 注意: 实际使用需要实现 OpenAIClient 和 FilterProtocol
```

**2. `_get_filter_info` Method:**

```python
    def _get_filter_info(self, content: str, **metadata) -> Tuple[str, bool]:
        messages = self._filter_protocol.process_input(content, **metadata)

        # Call client for filtering info
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)

        return self._filter_protocol.parse_output(content=response, **metadata)
```

**Description (描述):**

*   **`_get_filter_info` Method (方法定义):**  定义一个私有方法 `_get_filter_info`，它接收文档的内容 (`content`) 和元数据 (`metadata`) 作为输入。
*   **`process_input` (处理输入):** 使用 `filter_protocol` 的 `process_input` 方法，将文档内容和元数据转换为 LLM 可以理解的消息格式。
*   **`generate_content_with_messages` (调用 LLM):** 使用 `llm_client` 的 `generate_content_with_messages` 方法，将消息传递给 LLM，并获取 LLM 的响应。LLM 的配置信息通过 `llm_config` 传递。
*   **`parse_output` (解析输出):** 使用 `filter_protocol` 的 `parse_output` 方法，解析 LLM 的响应，提取过滤信息（`filter_info`）和相关性判断结果（`related`）。
*   **Return (返回):** 返回一个元组，包含过滤信息和相关性判断结果。

**Translation (中文翻译):**

*   这个方法负责与 LLM 交互，判断文档是否相关。
*   它首先使用 `filter_protocol` 将文档内容转换为 LLM 可以理解的格式。
*   然后，它调用 LLM 客户端，将消息传递给 LLM 并获取响应。
*   最后，它使用 `filter_protocol` 解析 LLM 的响应，提取过滤信息和相关性判断结果。

**Usage Demo (使用示例):**

```python
# 假设你已经有了一个 filter_instance (LLMPoweredFilter 的实例)
content = "This is a document about cats and dogs."
metadata = {"source": "example.com"}

filter_info, related = filter_instance._get_filter_info(content, **metadata)

print(f"过滤信息: {filter_info}")
print(f"是否相关: {related}")
```

**3. `transform_documents` Method:**

```python
    def transform_documents(self, documents: Sequence[Document], keep_unrelated: bool = False, **kwargs: Any) -> Sequence[Document]:
        ret_docs: List[Document] = []
        for idx, doc in tqdm(enumerate(documents), desc="Filtering Documents", total=len(documents)):
            content = doc.page_content
            metadata = doc.metadata

            filter_info, related = self._get_filter_info(content, **metadata)
            if self.logger is not None:
                self.logger.debug(
                    f"{idx + 1}/{len(documents)} document -- related? {related}, kept? {keep_unrelated or related}",
                    tag=self.NAME,
                )

            if not keep_unrelated and not related:
                continue

            # TODO: could there be multiple filter conditions? Set it as list of filter_info?
            metadata.update({"filter_info": filter_info, "related": related})
            ret_docs.append(doc)
        return ret_docs
```

**Description (描述):**

*   **`transform_documents` Method (方法定义):** 定义一个方法 `transform_documents`，它接收一个文档序列 (`documents`) 和一个 `keep_unrelated` 布尔值作为输入。`keep_unrelated` 决定是否保留不相关的文档。
*   **Initialization (初始化):** 创建一个空列表 `ret_docs`，用于存储过滤后的文档。
*   **Loop (循环):** 遍历输入文档序列，使用 `tqdm` 显示进度条。
*   **Extract Content and Metadata (提取内容和元数据):** 从每个文档中提取内容 (`page_content`) 和元数据 (`metadata`).
*   **Get Filter Info (获取过滤信息):** 调用 `_get_filter_info` 方法，获取过滤信息和相关性判断结果。
*   **Logging (日志记录):** 如果提供了日志记录器，则记录每个文档的处理结果。
*   **Filtering (过滤):** 根据 `keep_unrelated` 和 `related` 的值，决定是否保留当前文档。如果 `keep_unrelated` 为 `False` 且 `related` 为 `False`，则跳过当前文档。
*   **Update Metadata (更新元数据):** 将过滤信息和相关性判断结果添加到文档的元数据中。
*   **Append to Result (添加到结果):** 将保留的文档添加到 `ret_docs` 列表中。
*   **Return (返回):** 返回过滤后的文档列表。

**Translation (中文翻译):**

*   这个方法是 `LLMPoweredFilter` 类的核心，它负责实际的文档过滤操作。
*   它遍历输入的文档序列，对于每个文档，它调用 `_get_filter_info` 方法获取过滤信息和相关性判断结果。
*   然后，它根据 `keep_unrelated` 参数和相关性判断结果，决定是否保留当前文档。
*   如果文档被保留，则将其添加到结果列表中，并更新其元数据。

**Usage Demo (使用示例):**

```python
from langchain_core.documents import Document

# 假设你已经有了一个 filter_instance (LLMPoweredFilter 的实例)
# 假设你已经有一些文档
documents = [
    Document(page_content="This is a document about cats.", metadata={"source": "cat.com"}),
    Document(page_content="This is a document about dogs.", metadata={"source": "dog.com"}),
    Document(page_content="This is a document about unrelated topics.", metadata={"source": "other.com"}),
]

# 过滤文档，只保留相关的文档
filtered_documents = filter_instance.transform_documents(documents, keep_unrelated=False)

# 打印过滤后的文档
for doc in filtered_documents:
    print(f"文档内容: {doc.page_content}")
    print(f"文档元数据: {doc.metadata}")
```

**Complete Demo (完整示例):**

```python
from typing import Any, List, Sequence, Tuple

from tqdm import tqdm

from langchain_core.documents import Document, BaseDocumentTransformer

# 假设的 LLM 客户端和通信协议 (需要替换为实际的实现)
class BaseLLMClient:
    def generate_content_with_messages(self, messages, **kwargs):
        # 模拟 LLM 的响应
        if "cats" in messages[0]["content"]:
            return "Relevant: True, Filter Info: About Cats"
        elif "dogs" in messages[0]["content"]:
            return "Relevant: True, Filter Info: About Dogs"
        else:
            return "Relevant: False, Filter Info: Unrelated"

class CommunicationProtocol:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def process_input(self, content, **metadata):
        return [{"role": "user", "content": self.prompt_template.format(document=content, query="cats and dogs")}]

    def parse_output(self, content, **metadata):
        if "True" in content:
            return content.split(", Filter Info: ")[1], True
        else:
            return content.split(", Filter Info: ")[1], False

class Logger:
    def debug(self, message, tag):
        print(f"[{tag}] {message}")

class LLMPoweredFilter(BaseDocumentTransformer):
    NAME = "LLMPoweredFilter"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        filter_protocol: CommunicationProtocol,
        llm_config: dict = {},
        logger: Logger = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self._llm_client = llm_client
        self._llm_config = llm_config

        self._filter_protocol: CommunicationProtocol = filter_protocol

        self.logger = logger

    def _get_filter_info(self, content: str, **metadata) -> Tuple[str, bool]:
        messages = self._filter_protocol.process_input(content, **metadata)

        # Call client for filtering info
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)

        return self._filter_protocol.parse_output(content=response, **metadata)

    # TODO: create new interface like "TextSplitter" for filtering?
    def transform_documents(self, documents: Sequence[Document], keep_unrelated: bool = False, **kwargs: Any) -> Sequence[Document]:
        ret_docs: List[Document] = []
        for idx, doc in tqdm(enumerate(documents), desc="Filtering Documents", total=len(documents)):
            content = doc.page_content
            metadata = doc.metadata

            filter_info, related = self._get_filter_info(content, **metadata)
            if self.logger is not None:
                self.logger.debug(
                    f"{idx + 1}/{len(documents)} document -- related? {related}, kept? {keep_unrelated or related}",
                    tag=self.NAME,
                )

            if not keep_unrelated and not related:
                continue

            # TODO: could there be multiple filter conditions? Set it as list of filter_info?
            metadata.update({"filter_info": filter_info, "related": related})
            ret_docs.append(doc)
        return ret_docs

# 创建 LLM 客户端、通信协议和日志记录器
llm_client = BaseLLMClient()
filter_protocol = CommunicationProtocol(prompt_template="Is this document relevant to the query: cats and dogs? Document: {document}")
logger = Logger()

# 创建 LLMPoweredFilter 实例
filter_instance = LLMPoweredFilter(
    llm_client=llm_client,
    filter_protocol=filter_protocol,
    llm_config={},
    logger=logger
)

# 创建文档
documents = [
    Document(page_content="This is a document about cats.", metadata={"source": "cat.com"}),
    Document(page_content="This is a document about dogs.", metadata={"source": "dog.com"}),
    Document(page_content="This is a document about unrelated topics.", metadata={"source": "other.com"}),
]

# 过滤文档
filtered_documents = filter_instance.transform_documents(documents, keep_unrelated=False)

# 打印过滤后的文档
for doc in filtered_documents:
    print(f"文档内容: {doc.page_content}")
    print(f"文档元数据: {doc.metadata}")
```

**Important Notes (重要提示):**

*   The provided code includes a *dummy* implementation of `BaseLLMClient` and `CommunicationProtocol`.  You'll need to replace these with your actual LLM client (e.g., using `openai` library) and a properly designed prompt protocol.
*   Make sure you have the `langchain_core` and `tqdm` libraries installed (`pip install langchain-core tqdm`).
*   Adjust the prompt in the `CommunicationProtocol` to suit your specific filtering needs. The prompt is crucial for the LLM to understand what kind of documents you want to keep.

This detailed explanation, along with the code snippets and usage examples, should provide a comprehensive understanding of the `LLMPoweredFilter` class. Remember to replace the dummy implementations with your actual LLM client and communication protocol for real-world use.
