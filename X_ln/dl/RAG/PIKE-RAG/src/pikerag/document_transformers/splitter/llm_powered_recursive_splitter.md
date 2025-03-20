Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\splitter\llm_powered_recursive_splitter.py`

Okay, I will restart and provide a fresh explanation of the code with Chinese descriptions and a demo.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy  # 用于创建对象的深拷贝，避免修改原对象
from typing import Callable, Iterable, List, Tuple  # 类型提示，提高代码可读性

from tqdm import tqdm  # 进度条显示

from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter  # 用于文本分割
from langchain_core.documents import Document  # Langchain 文档对象

from pikerag.llm_client import BaseLLMClient  # 自定义的 LLM 客户端接口
from pikerag.prompts import CommunicationProtocol  # 自定义的通信协议
from pikerag.utils.logger import Logger  # 自定义的日志记录器


class LLMPoweredRecursiveSplitter(TextSplitter):  # 继承自 Langchain 的 TextSplitter
    NAME = "LLMPoweredRecursiveSplitter"  # 定义类的名称

    def __init__(
        self,
        llm_client: BaseLLMClient,  # LLM 客户端实例
        first_chunk_summary_protocol: CommunicationProtocol,  # 用于获取第一个 chunk 摘要的通信协议
        last_chunk_summary_protocol: CommunicationProtocol,  # 用于获取最后一个 chunk 摘要的通信协议
        chunk_resplit_protocol: CommunicationProtocol,  # 用于重新分割 chunk 的通信协议
        llm_config: dict = {},  # LLM 配置
        chunk_size: int = 4000,  # chunk 的大小
        chunk_overlap: int = 200,  # chunk 之间的重叠大小
        length_function: Callable[[str], int] = len,  # 计算文本长度的函数
        keep_separator: bool = False,  # 是否保留分隔符
        add_start_index: bool = False,  # 是否添加起始索引
        strip_whitespace: bool = True,  # 是否去除空白字符
        logger: Logger = None,  # 日志记录器
        **kwargs,  # 其他参数
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator, add_start_index, strip_whitespace)  # 调用父类的构造函数

        self._base_splitter = RecursiveCharacterTextSplitter(  # 创建一个基本的递归字符分割器
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            **kwargs,
        )

        self._llm_client = llm_client  # LLM 客户端
        self._llm_config = llm_config  # LLM 配置

        self._first_chunk_summary_protocol: CommunicationProtocol = first_chunk_summary_protocol  # 第一个 chunk 摘要协议
        self._last_chunk_summary_protocol: CommunicationProtocol = last_chunk_summary_protocol  # 最后一个 chunk 摘要协议
        self._chunk_resplit_protocol: CommunicationProtocol = chunk_resplit_protocol  # chunk 重新分割协议

        self.logger = logger  # 日志记录器

    def _get_first_chunk_summary(self, text: str, **kwargs) -> str:
        # 获取第一个 chunk 的摘要
        # 1. 首先，使用基本的分割器将文本分割成多个 chunk
        chunks = self._base_splitter.split_text(text)
        # 2. 找到第一个 chunk 在原始文本中的起始位置
        first_chunk_start_pos = text.find(chunks[0])
        # 3. 获取用于生成摘要的文本内容 (从文本开始到第一个 chunk 的结束位置)
        text_for_summary = text[:first_chunk_start_pos + len(chunks[0])]

        # 4. 格式化消息模板 (使用第一个 chunk 的内容)
        messages = self._first_chunk_summary_protocol.process_input(content=text_for_summary, **kwargs)

        # 5. 调用 LLM 客户端生成摘要
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)

        # 6. 解析 LLM 客户端的响应，获取摘要
        return self._first_chunk_summary_protocol.parse_output(content=response, **kwargs)

    def _resplit_chunk_and_generate_summary(
        self, text: str, chunks: List[str], chunk_summary: str, **kwargs,
    ) -> Tuple[str, str, str, str]:
        # 重新分割 chunk 并生成摘要
        # 1. 检查 chunk 的数量是否大于等于 2 (至少需要两个 chunk 才能进行重新分割)
        assert len(chunks) >= 2, f"When calling this function, input chunks length should be no less than 2!"
        # 2. 获取用于重新分割的文本内容 (前两个 chunk 的内容)
        text_to_resplit = text[:len(chunks[0]) + len(chunks[1])]

        # 3. 格式化消息模板 (使用前两个 chunk 的内容和当前的摘要)
        kwargs["summary"] = chunk_summary
        messages = self._chunk_resplit_protocol.process_input(content=text_to_resplit, **kwargs)

        # 4. 调用 LLM 客户端生成摘要
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)

        # 5. 解析 LLM 客户端的响应，获取重新分割后的 chunk 和新的摘要
        return self._chunk_resplit_protocol.parse_output(content=response, **kwargs)

    def _get_last_chunk_summary(self, chunk: str, chunk_summary: str, **kwargs) -> str:
        # 获取最后一个 chunk 的摘要
        # 1. 格式化消息模板 (使用最后一个 chunk 的内容和当前的摘要)
        kwargs["summary"] = chunk_summary
        messages = self._last_chunk_summary_protocol.process_input(content=chunk, **kwargs)

        # 2. 调用 LLM 客户端生成摘要
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)

        # 3. 解析 LLM 客户端的响应，获取摘要
        return self._last_chunk_summary_protocol.parse_output(content=response, **kwargs)

    def split_text(self, text: str, metadata: dict) -> List[str]:
        # 分割文本 (将文本分割成多个 chunk)
        docs = self.create_documents(texts=[text], metadatas=[metadata])
        return [doc.page_content for doc in docs]

    def create_documents(self, texts: List[str], metadatas: List[dict], **kwargs) -> List[Document]:
        # 创建文档 (将文本和元数据封装成 Langchain Document 对象)
        # 1. 检查文本和元数据的长度是否一致
        if len(texts) != len(metadatas):
            error_message = (
                f"Input texts and metadatas should have same length, "
                f"{len(texts)} texts but {len(metadatas)} metadatas are given."
            )
            if self.logger is not None:
                self.logger.error(error_message, tag=self.NAME)
            raise ValueError(error_message)

        ret_docs: List[Document] = []
        # 2. 遍历文本和元数据，并将它们封装成 Document 对象
        for text, metadata in zip(texts, metadatas):
            ret_docs.extend(self.split_documents([Document(page_content=text, metadata=metadata)], **kwargs))
        return ret_docs

    def split_documents(self, documents: Iterable[Document], **kwargs) -> List[Document]:
        # 分割文档 (将 Langchain Document 对象分割成多个 Document 对象)
        ret_docs: List[Document] = []
        # 1. 遍历文档列表
        for idx, doc in tqdm(enumerate(documents), desc="Splitting Documents", total=len(documents)):  # 使用 tqdm 显示进度条
            text = doc.page_content  # 获取文档的内容
            metadata = doc.metadata  # 获取文档的元数据

            text = text.strip()  # 去除文本两端的空白字符
            chunk_summary = self._get_first_chunk_summary(text, **metadata)  # 获取第一个 chunk 的摘要
            chunks = self._base_splitter.split_text(text)  # 使用基本的分割器将文本分割成多个 chunk
            while True:  # 循环处理 chunk
                if len(chunks) == 1:  # 如果只有一个 chunk，则表示已经处理到最后一个 chunk
                    # 1. 获取最后一个 chunk 的摘要
                    chunk_summary = self._get_last_chunk_summary(chunks[0], chunk_summary, **metadata)
                    # 2. 创建新的元数据 (包含摘要)
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    # 3. 创建新的 Document 对象 (包含最后一个 chunk 和新的元数据)
                    ret_docs.append(Document(page_content=chunks[0], metadata=chunk_meta))

                    if self.logger is not None:
                        self.logger.debug(
                            msg=(
                                f"{len(ret_docs)}th chunk added (length: {len(chunks[0])}),"
                                f" the last chunk of current document."
                            ),
                            tag=self.NAME,
                        )

                    break  # 结束循环

                else:  # 如果有多个 chunk，则需要重新分割
                    # 1. 重新分割 chunk 并生成摘要
                    chunk, chunk_summary, next_summary, dropped_len = self._resplit_chunk_and_generate_summary(
                        text, chunks, chunk_summary, **metadata,
                    )

                    if len(chunk) == 0:  # 如果重新分割后的第一个 chunk 为空，则跳过
                        if self.logger is not None:
                            self.logger.debug(msg=f"Skip empty re-split first chunk", tag=self.NAME)

                        chunk_summary = next_summary  # 更新摘要
                        chunks = [chunks[0] + chunks[1]] + chunks[2:]  # 合并前两个 chunk
                        continue  # 继续循环

                    # 2. 创建新的元数据 (包含摘要)
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    # 3. 创建新的 Document 对象 (包含重新分割后的第一个 chunk 和新的元数据)
                    ret_docs.append(Document(page_content=chunk, metadata=chunk_meta))

                    if self.logger is not None:
                        self.logger.debug(msg=f"{len(ret_docs)}th chunk added (length: {len(chunk)}).", tag=self.NAME)

                    # 4. 更新信息，处理剩余的文本
                    text = text[dropped_len:].strip()  # 更新文本
                    chunk_summary = next_summary  # 更新摘要
                    chunks = self._base_splitter.split_text(text)  # 重新分割文本

        return ret_docs  # 返回分割后的 Document 对象列表


# 示例用法
if __name__ == "__main__":
    from pikerag.llm_client import MockLLMClient # 使用MockLLMClient 作为示例，避免真实调用LLM
    from pikerag.prompts import FirstChunkSummaryProtocol, LastChunkSummaryProtocol, ChunkResplitProtocol

    # 1. 创建一个 MockLLMClient
    llm_client = MockLLMClient()

    # 2. 创建通信协议 (需要根据实际情况定义)
    first_chunk_summary_protocol = FirstChunkSummaryProtocol(input_template="Summarize the first part: {content}", output_template="Summary: {summary}")
    last_chunk_summary_protocol = LastChunkSummaryProtocol(input_template="Summarize the last part: {content} with previous summary: {summary}", output_template="Summary: {summary}")
    chunk_resplit_protocol = ChunkResplitProtocol(input_template="Resplit the text: {content} based on previous summary: {summary}", output_template="Chunk: {chunk}\nSummary: {summary}\nNext Summary: {next_summary}\nDropped Length: {dropped_len}")

    # 3. 创建 LLMPoweredRecursiveSplitter 实例
    splitter = LLMPoweredRecursiveSplitter(
        llm_client=llm_client,
        first_chunk_summary_protocol=first_chunk_summary_protocol,
        last_chunk_summary_protocol=last_chunk_summary_protocol,
        chunk_resplit_protocol=chunk_resplit_protocol,
        chunk_size=1000,  # 设置 chunk 大小
        chunk_overlap=50,  # 设置 chunk 重叠大小
    )

    # 4. 准备要分割的文本
    text = "This is a long text that needs to be split into smaller chunks. " * 20

    # 5. 准备元数据
    metadata = {"source": "example_document.txt"}

    # 6. 分割文本
    documents = splitter.create_documents(texts=[text], metadatas=[metadata])

    # 7. 打印分割后的 chunk
    for i, doc in enumerate(documents):
        print(f"Chunk {i + 1}: {doc.page_content[:50]}...")  # 只打印前50个字符
        print(f"Metadata: {doc.metadata}") # 打印metadata信息
```

**代码解释:**

*   **`LLMPoweredRecursiveSplitter` 类:** 这是核心类，它继承自 `TextSplitter`，并使用 LLM 来辅助文本分割。它接收 LLM 客户端和几个通信协议作为参数。
*   **`__init__` 方法:** 初始化函数，设置 LLM 客户端，通信协议，以及基本的文本分割参数，比如 `chunk_size` 和 `chunk_overlap`。
*   **`_get_first_chunk_summary`，`_resplit_chunk_and_generate_summary`，`_get_last_chunk_summary` 方法:** 这些方法使用 LLM 客户端和定义的通信协议来生成第一个 chunk，中间 chunk以及最后一个chunk的摘要，并协助重新分割chunk。
*   **`split_documents` 方法:**  这是主要的处理函数。 它首先使用基本的 `RecursiveCharacterTextSplitter` 将文档分割成 chunk。 然后，它循环处理这些 chunk，使用 LLM 辅助重新分割和生成摘要。  每个 chunk 的摘要都被添加到 chunk 的元数据中。
*   **示例用法:**  代码的最后一部分展示了如何使用 `LLMPoweredRecursiveSplitter` 类。  它首先创建一个 `MockLLMClient` (为了方便演示，实际使用时需要替换成真正的 LLM 客户端)，然后创建通信协议实例，并最终创建 `LLMPoweredRecursiveSplitter` 实例。  然后，它准备要分割的文本和元数据，并调用 `create_documents` 方法进行分割。  最后，它打印分割后的 chunk (只打印前 50 个字符) 和元数据。

**运行这段代码的步骤:**

1.  **安装依赖:**  确保你已经安装了必要的依赖包，例如 `langchain`， `tqdm`，以及 `pikerag` (如果 `pikerag` 是一个单独的库，你需要先安装它)。 你可以使用 `pip` 来安装这些依赖包。

    ```bash
    pip install langchain tqdm
    # pip install pikerag  (如果 pikerag 是一个单独的库)
    ```

2.  **替换 `MockLLMClient` (可选):**  如果 `MockLLMClient` 只是一个用于演示的模拟 LLM 客户端，你可能需要将其替换成真正的 LLM 客户端。 替换方式取决于你使用的 LLM 服务 (例如 OpenAI, Azure OpenAI, 等等)。 你还需要配置相应的 API 密钥和其他认证信息。 并且需要修改`FirstChunkSummaryProtocol`等等protocol定义，来实现LLM Client的调用和信息解析。

3.  **运行代码:**  保存代码到一个 Python 文件 (例如 `splitter_example.py`)，然后运行它。

    ```bash
    python splitter_example.py
    ```

**预期输出:**

这段代码会输出分割后的 chunk (每个 chunk 只显示前 50 个字符) 以及每个 chunk 的元数据，包括 LLM 生成的摘要。 由于我们使用的是 `MockLLMClient`，所以摘要的内容可能是模拟的，而不是真实的 LLM 生成的。 你会看到 `tqdm` 提供的进度条，显示文档分割的进度。

**总结:**

`LLMPoweredRecursiveSplitter` 类是一个强大的工具，可以帮助你更智能地分割文本。 它使用 LLM 来辅助分割，可以生成每个 chunk 的摘要，并根据摘要来调整分割策略。 这可以提高后续信息检索的准确性。

我希望这个更详细的解释对你有所帮助!
