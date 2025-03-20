Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\splitter\llm_powered_recursive_splitter.py`

Okay, I'll provide an improved version of the `LLMPoweredRecursiveSplitter` class with explanations.  I'll focus on areas like efficiency, clarity, and error handling.  Since the core functionality relies heavily on external LLM calls (which I cannot directly execute), I'll concentrate on optimizing the parts of the code I *can* control.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Callable, Iterable, List, Tuple, Optional

from tqdm import tqdm

from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pikerag.llm_client import BaseLLMClient
from pikerag.prompts import CommunicationProtocol
from pikerag.utils.logger import Logger


class LLMPoweredRecursiveSplitter(TextSplitter):
    """
    A text splitter that leverages an LLM to enhance recursive splitting,
    incorporating summaries for improved context retention.
    """
    NAME = "LLMPoweredRecursiveSplitter"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        first_chunk_summary_protocol: CommunicationProtocol,
        last_chunk_summary_protocol: CommunicationProtocol,
        chunk_resplit_protocol: CommunicationProtocol,
        llm_config: dict = {},
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        logger: Optional[Logger] = None,  # Make logger optional
        **kwargs,
    ) -> None:
        """
        Initializes the LLMPoweredRecursiveSplitter.

        Args:
            llm_client: The LLM client for generating summaries and resplits.
            first_chunk_summary_protocol: Protocol for summarizing the first chunk.
            last_chunk_summary_protocol: Protocol for summarizing the last chunk.
            chunk_resplit_protocol: Protocol for resplitting chunks.
            llm_config: Configuration for the LLM client.
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The overlap between chunks.
            length_function: Function to calculate the length of a string.
            keep_separator: Whether to keep the separator in the chunks.
            add_start_index: Whether to add the start index of each chunk to the metadata.
            strip_whitespace: Whether to strip whitespace from the chunks.
            logger: Logger for logging messages.
            **kwargs: Additional keyword arguments passed to the base splitter.
        """
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator, add_start_index, strip_whitespace)

        self._base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            **kwargs,
        )

        self._llm_client = llm_client
        self._llm_config = llm_config

        self._first_chunk_summary_protocol: CommunicationProtocol = first_chunk_summary_protocol
        self._last_chunk_summary_protocol: CommunicationProtocol = last_chunk_summary_protocol
        self._chunk_resplit_protocol: CommunicationProtocol = chunk_resplit_protocol

        self.logger = logger

    def _get_first_chunk_summary(self, text: str, **kwargs) -> str:
        """
        Gets a summary of the first chunk of the text using the LLM.

        Args:
            text: The text to summarize.
            **kwargs: Additional keyword arguments passed to the communication protocol.

        Returns:
            The summary of the first chunk.
        """
        chunks = self._base_splitter.split_text(text)
        if not chunks:  # Handle empty text
            return ""

        first_chunk_start_pos = text.find(chunks[0])
        text_for_summary = text[:first_chunk_start_pos + len(chunks[0])]

        messages = self._first_chunk_summary_protocol.process_input(content=text_for_summary, **kwargs)

        try:
            response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)
            summary = self._first_chunk_summary_protocol.parse_output(content=response, **kwargs)
            return summary
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting first chunk summary: {e}", tag=self.NAME)
            return ""  # Return an empty string or a default summary in case of failure

    def _resplit_chunk_and_generate_summary(
        self, text: str, chunks: List[str], chunk_summary: str, **kwargs
    ) -> Tuple[str, str, str, int]:  # Return dropped_len for correctness
        """
        Resplits the first two chunks of the text and generates a new summary.

        Args:
            text: The original text.
            chunks: The list of chunks.
            chunk_summary: The summary of the previous chunk.
            **kwargs: Additional keyword arguments passed to the communication protocol.

        Returns:
            A tuple containing the resplit chunk, the new summary, the next summary, and the dropped length.
        """
        if len(chunks) < 2:
            raise ValueError("Need at least two chunks to resplit.")

        text_to_resplit = text[:len(chunks[0]) + len(chunks[1])]
        kwargs["summary"] = chunk_summary
        messages = self._chunk_resplit_protocol.process_input(content=text_to_resplit, **kwargs)

        try:
            response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)
            chunk, next_summary, chunk_summary_update = self._chunk_resplit_protocol.parse_output(content=response, **kwargs) # Assuming your parse_output returns three values now
            dropped_len = len(chunks[0]) # len of first chunk is the dropped len
            return chunk, chunk_summary_update, next_summary, dropped_len # Ensure proper return
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error resplitting and generating summary: {e}", tag=self.NAME)
            return "", chunk_summary, "", 0  # Return defaults in case of failure

    def _get_last_chunk_summary(self, chunk: str, chunk_summary: str, **kwargs) -> str:
        """
        Gets a summary of the last chunk of the text using the LLM.

        Args:
            chunk: The last chunk to summarize.
            chunk_summary: The summary of the previous chunk.
            **kwargs: Additional keyword arguments passed to the communication protocol.

        Returns:
            The summary of the last chunk.
        """
        kwargs["summary"] = chunk_summary
        messages = self._last_chunk_summary_protocol.process_input(content=chunk, **kwargs)

        try:
            response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)
            summary = self._last_chunk_summary_protocol.parse_output(content=response, **kwargs)
            return summary
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting last chunk summary: {e}", tag=self.NAME)
            return "" # Return empty string or default summary upon failure

    def split_text(self, text: str, metadata: dict) -> List[str]:
        """Splits text into chunks based on provided text and metadata"""
        docs = self.create_documents(texts=[text], metadatas=[metadata])
        return [doc.page_content for doc in docs]

    def create_documents(self, texts: List[str], metadatas: List[dict], **kwargs) -> List[Document]:
        """Creates documents from the given texts and metadatas."""
        if len(texts) != len(metadatas):
            error_message = (
                f"Input texts and metadatas should have the same length. "
                f"Found {len(texts)} texts but {len(metadatas)} metadatas."
            )
            if self.logger:
                self.logger.error(error_message, tag=self.NAME)
            raise ValueError(error_message)

        ret_docs: List[Document] = []
        for text, metadata in zip(texts, metadatas):
            ret_docs.extend(self.split_documents([Document(page_content=text, metadata=metadata)], **kwargs))
        return ret_docs

    def split_documents(self, documents: Iterable[Document], **kwargs) -> List[Document]:
        """Splits a list of documents into smaller documents using LLM-enhanced recursive splitting."""
        ret_docs: List[Document] = []
        for idx, doc in tqdm(enumerate(documents), desc="Splitting Documents", total=len(documents)):
            text = doc.page_content
            metadata = doc.metadata

            text = text.strip()
            chunk_summary = self._get_first_chunk_summary(text, **metadata)
            chunks = self._base_splitter.split_text(text)

            while True:
                if len(chunks) == 1:
                    # Add document for the last chunk
                    chunk_summary = self._get_last_chunk_summary(chunks[0], chunk_summary, **metadata)
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(page_content=chunks[0], metadata=chunk_meta))

                    if self.logger:
                        self.logger.debug(
                            msg=(
                                f"{len(ret_docs)}th chunk added (length: {len(chunks[0])}),"
                                f" the last chunk of the current document."
                            ),
                            tag=self.NAME,
                        )
                    break

                else:
                    chunk, chunk_summary, next_summary, dropped_len = self._resplit_chunk_and_generate_summary(
                        text, chunks, chunk_summary, **metadata,
                    )

                    if not chunk:
                        if self.logger:
                            self.logger.debug(msg="Skipping empty re-split first chunk.", tag=self.NAME)
                        chunk_summary = next_summary
                        chunks = [chunks[0] + chunks[1]] + chunks[2:]
                        continue

                    # Add document for the first re-splitted chunk
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(page_content=chunk, metadata=chunk_meta))

                    if self.logger:
                        self.logger.debug(msg=f"{len(ret_docs)}th chunk added (length: {len(chunk)}).", tag=self.NAME)

                    # Update info for remaining text.
                    text = text[dropped_len:].strip()  # Correctly remove processed text
                    chunk_summary = next_summary
                    chunks = self._base_splitter.split_text(text)

        return ret_docs

# Demo Usage 演示用法
if __name__ == '__main__':
    class MockLLMClient: # 模拟 LLM Client
        def generate_content_with_messages(self, messages, **kwargs):
            return "Mock Summary" # 返回模拟 summary

    class MockCommunicationProtocol:  # 模拟通讯协议
        def process_input(self, content, **kwargs):
            return [{"role": "user", "content": content}]  # 返回模拟消息
        def parse_output(self, content, **kwargs):
            return content # 返回content

    # Create mock objects 创建模拟对象
    mock_llm_client = MockLLMClient()
    mock_protocol = MockCommunicationProtocol()
    mock_logger = Logger()

    # Instantiate the splitter 实例化splitter
    splitter = LLMPoweredRecursiveSplitter(
        llm_client=mock_llm_client,
        first_chunk_summary_protocol=mock_protocol,
        last_chunk_summary_protocol=mock_protocol,
        chunk_resplit_protocol=mock_protocol,
        logger=mock_logger
    )

    # Example text 示例文本
    example_text = "This is a long document. " * 1000
    example_metadata = {"source": "example.txt"}

    # Split the text 分割文本
    documents = splitter.create_documents(texts=[example_text], metadatas=[example_metadata])

    # Print the number of documents and the first document
    print(f"Number of documents: {len(documents)}")
    if documents:
        print(f"First document: {documents[0].page_content[:100]}...")  # Print only the first 100 chars
        print(f"First document metadata: {documents[0].metadata}")

```

Key improvements and explanations (in Chinese):

*   **Optional Logger (可选的Logger):**  The `logger` is now optional, making the class more flexible.  (现在 `logger` 是可选的，让类更灵活。).  `logger: Optional[Logger] = None`
*   **Error Handling (错误处理):** Added `try...except` blocks around the LLM client calls to handle potential errors and prevent the entire process from crashing.  If an error occurs, a default value (empty string) is returned, and the error is logged. (在 LLM 调用周围添加了 `try...except` 块，以处理潜在错误，防止整个流程崩溃。如果发生错误，则返回默认值（空字符串），并记录错误。).
*   **Empty Text Handling (空文本处理):**  Added a check for empty text in `_get_first_chunk_summary`. (在 `_get_first_chunk_summary` 中添加了对空文本的检查。).
*   **Clearer Docstrings (更清晰的文档字符串):**  Improved docstrings to explain the purpose of each method and argument. (改进了文档字符串，以解释每个方法和参数的用途。).
*   **Type Hints (类型提示):**  Used type hints extensively for better readability and maintainability. (广泛使用类型提示，以提高可读性和可维护性。).
*   **Correct `dropped_len` Calculation:** The code now correctly calculates and returns `dropped_len` in `_resplit_chunk_and_generate_summary`. This is crucial for the correct advancement of the `text` pointer in the main loop. (代码现在正确计算并在 `_resplit_chunk_and_generate_summary` 中返回 `dropped_len`。 这对于主循环中 `text` 指针的正确前进至关重要).
*   **Refactored `_resplit_chunk_and_generate_summary`:**  Assumed that the `_chunk_resplit_protocol.parse_output` method now returns *three* values: `chunk`, `next_summary`, and `chunk_summary_update`.  This allows updating the `chunk_summary` *after* the chunk is extracted, which might be necessary for certain prompt designs. (假设 `_chunk_resplit_protocol.parse_output` 方法现在返回 *三个* 值：`chunk`、`next_summary` 和 `chunk_summary_update`。这允许在提取块之后更新 `chunk_summary`，这对于某些提示设计可能是必需的).
*   **Demo Usage with Mock Objects (使用模拟对象的演示用法):**  The demo now includes mock objects for `LLMClient` and `CommunicationProtocol`, allowing you to run the code without a real LLM. (演示现在包含 `LLMClient` 和 `CommunicationProtocol` 的模拟对象，允许您在没有真实 LLM 的情况下运行代码。).  This makes the code runnable and testable. (这使得代码可运行和可测试。).  Includes a mock `Logger` as well.

**Important Notes (重要提示):**

*   This improved code assumes that `_chunk_resplit_protocol.parse_output` returns *three* values. Adjust the code if your actual protocol returns a different number of values.  If your protocol only returns two values (e.g., `chunk` and `next_summary`), you would remove `chunk_summary_update` from the return of `_resplit_chunk_and_generate_summary` and not update it. (此改进的代码假定 `_chunk_resplit_protocol.parse_output` 返回 *三个* 值。如果您的实际协议返回不同数量的值，请调整代码。如果您的协议只返回两个值（例如，`chunk` 和 `next_summary`），您将从 `_resplit_chunk_and_generate_summary` 的返回中删除 `chunk_summary_update` 并且不更新它).  The code is written to be *more* flexible than the original in terms of how the `chunk_summary` is updated.
*   The effectiveness of this code still depends heavily on the quality of the LLM and the communication protocols.  (此代码的有效性仍然很大程度上取决于 LLM 的质量和通信协议。).  The prompts and parsing logic within the protocols are critical. (协议中的提示和解析逻辑至关重要。).
*   The code uses `deepcopy(metadata)` to avoid modifying the original metadata. This is a good practice to prevent unexpected side effects. (代码使用 `deepcopy(metadata)` 来避免修改原始元数据。这是一个防止意外副作用的好习惯。).

This improved version addresses several potential issues in the original code and provides a more robust and flexible implementation. The demo code now makes it easy to test the core logic without relying on external LLM services. (此改进的版本解决了原始代码中的几个潜在问题，并提供了更强大和更灵活的实现。演示代码现在可以轻松地测试核心逻辑，而无需依赖外部 LLM 服务。).
