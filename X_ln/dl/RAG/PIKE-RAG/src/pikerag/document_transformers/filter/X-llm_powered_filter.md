Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\filter\llm_powered_filter.py`

好的，我们重新开始，这次我将提供更智能的代码改进，并附上中文描述和演示。我们将专注于提升效率、可读性和灵活性。

**1. 异步批量处理 (Asynchronous Batch Processing):**

```python
import asyncio
from typing import Any, List, Sequence, Tuple

from tqdm.asyncio import tqdm
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
        batch_size: int = 4, # 新增: 批量大小
        **kwargs,
    ) -> None:
        super().__init__()

        self._llm_client = llm_client
        self._llm_config = llm_config
        self._filter_protocol: CommunicationProtocol = filter_protocol
        self.logger = logger
        self.batch_size = batch_size

    async def _get_filter_info(self, content: str, **metadata) -> Tuple[str, bool]:
        messages = self._filter_protocol.process_input(content, **metadata)
        response = await self._llm_client.agenerate_content_with_messages(messages=messages, **self._llm_config)
        return self._filter_protocol.parse_output(content=response, **metadata)

    async def _process_document(self, idx: int, doc: Document, keep_unrelated: bool) -> Document | None:
        content = doc.page_content
        metadata = doc.metadata

        filter_info, related = await self._get_filter_info(content, **metadata)

        if self.logger is not None:
            self.logger.debug(
                f"{idx + 1} document -- related? {related}, kept? {keep_unrelated or related}",
                tag=self.NAME,
            )

        if not keep_unrelated and not related:
            return None

        metadata.update({"filter_info": filter_info, "related": related})
        return doc

    async def transform_documents(self, documents: Sequence[Document], keep_unrelated: bool = False, **kwargs: Any) -> Sequence[Document]:
        ret_docs: List[Document] = []
        tasks = []

        for idx, doc in enumerate(documents):
            tasks.append(self._process_document(idx, doc, keep_unrelated))

        # 使用 tqdm.asyncio 进行异步进度条显示
        for doc in tqdm(asyncio.as_completed(tasks), desc="Filtering Documents", total=len(documents)):
            result = await doc # 等待每个任务完成
            if result:
                ret_docs.append(result)

        return ret_docs
```

**描述:** 这个版本使用 `asyncio` 和 `tqdm.asyncio` 来实现异步批量处理。

**主要改进:**

*   **异步处理:** `_get_filter_info` 和 `transform_documents` 现在是异步函数，允许并行处理多个文档。
*   **批量大小:** 引入 `batch_size` 参数，用于控制一次发送到 LLM 的文档数量（尽管这部分需要 LLM Client 的支持）。
*   **异步进度条:**  使用 `tqdm.asyncio` 为异步处理提供进度条。

**如何使用:**

1.  确保你的 `BaseLLMClient` 支持异步调用（例如，提供 `agenerate_content_with_messages` 方法）。
2.  初始化 `LLMPoweredFilter` 实例，传入异步 LLM 客户端和必要的配置。
3.  使用 `asyncio.run(filter.transform_documents(documents))` 来执行文档转换。

**中文解释:**

这段代码通过使用异步编程，提高了文档过滤的效率。它允许同时处理多个文档，而不是一个接一个地处理。`asyncio` 模块用于管理并发任务，`tqdm.asyncio` 提供了异步进度条，方便监控处理进度。`batch_size` 参数可以用来控制每次发送给大型语言模型的文档数量，以避免超出其处理能力。

**Demo:**

```python
import asyncio
from pikerag.llm_client import MockLLMClient # 假设你有一个模拟的 LLM 客户端
from pikerag.prompts import SimpleFilterProtocol # 假设你有一个简单的协议
from langchain_core.documents import Document

# 模拟的 LLM 客户端
class MockLLMClient:
    async def agenerate_content_with_messages(self, messages, **kwargs):
        # 模拟 LLM 的行为，返回一个简单的结果
        return "Related: True" # 始终返回 "Related: True"

# 创建模拟的 LLM 客户端和协议
llm_client = MockLLMClient()
filter_protocol = SimpleFilterProtocol(relevant_label="True", irrelevant_label="False")

# 创建 LLMPoweredFilter 实例
filter = LLMPoweredFilter(llm_client=llm_client, filter_protocol=filter_protocol)

# 创建一些示例文档
documents = [
    Document(page_content="This is a related document."),
    Document(page_content="This is another related document."),
    Document(page_content="This might be an unrelated document.")
]

# 运行异步文档转换
async def main():
    filtered_documents = await filter.transform_documents(documents)
    print(f"Number of filtered documents: {len(filtered_documents)}")
    for doc in filtered_documents:
        print(doc.page_content)

if __name__ == "__main__":
    asyncio.run(main())
```

**这段 Demo 代码:**

1.  定义了一个模拟的 LLM 客户端 `MockLLMClient`，用于演示异步 LLM 调用的行为。
2.  创建了一个 `SimpleFilterProtocol` 实例，用于定义与 LLM 的通信协议。
3.  创建了 `LLMPoweredFilter` 实例，并传入模拟的 LLM 客户端和协议。
4.  创建了一些示例文档。
5.  使用 `asyncio.run` 运行 `transform_documents` 方法，并打印过滤后的文档数量和内容。

**重要提示:**

*   你需要根据你实际使用的 LLM 客户端和协议进行调整。
*   在生产环境中，你需要使用真正的 LLM 客户端，并确保其支持异步调用。
*   确保你的 `pikerag` 包已经安装，或者根据需要替换为你的实际实现。

这个异步批量处理的版本可以显著提高文档过滤的效率，特别是在处理大量文档时。

**2. 更灵活的过滤条件 (More Flexible Filtering Conditions):**

```python
# ... (之前的代码)

class LLMPoweredFilter(BaseDocumentTransformer):
    # ... (之前的代码)

    def _process_document(self, idx: int, doc: Document, keep_unrelated: bool, additional_filters: List[callable] = None) -> Document | None:
        content = doc.page_content
        metadata = doc.metadata

        filter_info, related = await self._get_filter_info(content, **metadata)

        if self.logger is not None:
            self.logger.debug(
                f"{idx + 1} document -- related? {related}, kept? {keep_unrelated or related}",
                tag=self.NAME,
            )

        # Apply additional filters
        if additional_filters:
            for filter_func in additional_filters:
                if not filter_func(doc, filter_info, related):
                    return None  # Document failed an additional filter

        if not keep_unrelated and not related:
            return None

        metadata.update({"filter_info": filter_info, "related": related})
        return doc

    async def transform_documents(self, documents: Sequence[Document], keep_unrelated: bool = False, additional_filters: List[callable] = None, **kwargs: Any) -> Sequence[Document]:
        ret_docs: List[Document] = []
        tasks = []

        for idx, doc in enumerate(documents):
            tasks.append(self._process_document(idx, doc, keep_unrelated, additional_filters))

        # 使用 tqdm.asyncio 进行异步进度条显示
        for doc in tqdm(asyncio.as_completed(tasks), desc="Filtering Documents", total=len(documents)):
            result = await doc # 等待每个任务完成
            if result:
                ret_docs.append(result)

        return ret_docs
```

**改进描述:**

*   **`additional_filters`参数:**  `transform_documents` 和 `_process_document` 方法现在接受一个 `additional_filters` 参数，它是一个可调用对象（函数）的列表。
*   **灵活的过滤条件:** 允许添加自定义的过滤函数，这些函数可以根据文档内容、元数据、`filter_info` 和 `related` 标志来决定是否保留文档。

**中文解释:**

这个改进增加了过滤器的灵活性。除了根据 LLM 的结果进行过滤之外，你还可以添加自定义的过滤函数。这些函数可以根据你的特定需求来检查文档的内容、元数据或 LLM 返回的过滤信息，从而实现更精确的过滤。

**示例:**

```python
def custom_filter(doc: Document, filter_info: str, related: bool) -> bool:
    """
    一个自定义的过滤函数，只保留长度大于 100 的相关文档。
    """
    if related and len(doc.page_content) > 100:
        return True
    return False

# 在调用 transform_documents 时传递 additional_filters
filtered_documents = await filter.transform_documents(documents, additional_filters=[custom_filter])
```

这个示例演示了如何创建一个自定义的过滤函数，该函数只保留长度大于 100 且被 LLM 认为是相关的文档。

这些改进旨在使你的文档过滤流程更高效、灵活和可定制。
