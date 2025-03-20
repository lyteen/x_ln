Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\splitter\recursive_sentence_splitter.py`

Okay, restarting with a fresh explanation of the code you provided, broken down into smaller, more digestible parts, along with Chinese explanations and usage examples.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import List, Optional

import spacy
import spacy.tokens
from tqdm import tqdm

from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document


LANG2MODELNAME = {
    "en": "en_core_web_lg",
    "zh": "zh_core_web_lg",
}
```

**描述:** 这部分代码导入了必要的库，并定义了一个字典 `LANG2MODELNAME`，它将语言代码（如 "en" 和 "zh"）映射到 Spacy 模型的名称。

**Chinese (中文描述):** 这段代码导入了像 `copy`, `typing`, `spacy`, `tqdm` 和 `langchain` 等库， 这些库将在后面的文本分割任务中使用。 `LANG2MODELNAME` 字典定义了两种语言（英语 "en" 和中文 "zh"）对应的 Spacy 预训练模型的名称。

```python
class RecursiveSentenceSplitter(TextSplitter):
    NAME = "RecursiveSentenceSplitter"

    def __init__(
        self,
        lang: str = "en",
        nlp_max_len: int = 4000000,
        num_parallel: int = 4,
        chunk_size: int = 12,
        chunk_overlap: int = 4,
        **kwargs,
    ):
        """
        Args:
            lang (str): "en" for English, "zh" for Chinese.
            nlp_max_len (int):
            num_workers (int):
            chunk_size (int): number of sentences per chunk.
            chunk_over_lap (int): number of sentence overlap between two continuous chunks.
        """
        super().__init__(chunk_size, chunk_overlap)
        self._stride: int = self._chunk_size - self._chunk_overlap
        self._num_parallel: int = num_parallel

        self._load_model(lang, nlp_max_len)

        return
```

**描述:**  这段代码定义了一个名为 `RecursiveSentenceSplitter` 的类，它继承自 `langchain.text_splitter.TextSplitter`。  `__init__` 方法初始化类的实例。它接受语言、Spacy 处理的最大长度、并行处理的数量、每个块的句子数和块重叠的句子数等参数。  `_stride` 计算步幅，`_num_parallel` 存储并行处理的数量，并且调用 `_load_model` 方法来加载 Spacy 模型。

**Chinese (中文描述):**  这段代码定义了一个名为 `RecursiveSentenceSplitter` 的类, 这个类继承了 `langchain` 中的 `TextSplitter` 类， 用于递归地将文本分割成句子。 `__init__` 方法是类的构造函数，用于初始化类的实例。
*   `lang`: 指定使用的语言，例如 "en" 代表英语， "zh" 代表中文.
*   `nlp_max_len`:  指定 Spacy 模型可以处理的最大文本长度.
*   `num_parallel`:  指定并行处理的进程数量.
*   `chunk_size`:  指定每个文本块包含的句子数量.
*   `chunk_overlap`:  指定相邻文本块之间重叠的句子数量.
构造函数中计算了 `_stride` (步幅，即两个相邻文本块的起始位置的距离) 和 `_num_parallel` (并行处理数量)， 并且调用了 `_load_model` 方法来加载对应的 Spacy 模型。

```python
    def _load_model(self, language: str, nlp_max_len: int) -> None:
        assert language in LANG2MODELNAME, f"Spacy model not specified for language: {language}."

        model_name = LANG2MODELNAME[language]
        try:
            self._nlp = spacy.load(model_name)
        except:
            spacy.cli.download(model_name)
            self._nlp = spacy.load(model_name)
        self._nlp.max_length = nlp_max_len
        return
```

**描述:**  `_load_model` 方法加载指定语言的 Spacy 模型。  它首先检查该语言是否在 `LANG2MODELNAME` 字典中。  然后，它尝试加载 Spacy 模型。如果模型未找到，它会下载该模型。最后，它设置 Spacy 模型的最大长度。

**Chinese (中文描述):**  `_load_model` 方法用于加载指定语言的 Spacy 模型。
*   首先，它会检查指定的语言是否在 `LANG2MODELNAME` 字典中，如果不在，则会抛出一个错误。
*   然后，它会根据语言代码从 `LANG2MODELNAME` 字典中获取对应的 Spacy 模型名称。
*   接着，它会尝试使用 `spacy.load` 方法加载 Spacy 模型。如果加载失败（例如，模型尚未下载），则会使用 `spacy.cli.download` 方法下载该模型，然后再加载。
*   最后，它会将 Spacy 模型的最大文本长度设置为 `nlp_max_len`。

```python
    def _nlp_doc_to_texts(self, doc: spacy.tokens.Doc) -> List[str]:
        sents = [sent.text.strip() for sent in doc.sents]
        sents = [sent for sent in sents if len(sent) > 0]

        segments: List[str] = []
        for i in range(0, len(sents), self._stride):
            segment = " ".join(sents[i : i + self._chunk_size])
            segments.append(segment)
            if i + self._chunk_size >= len(sents):
                break

        return segments
```

**描述:** `_nlp_doc_to_texts` 方法将 Spacy Doc 对象转换为文本段列表。  它首先提取文档中的句子并删除前导和尾随空格。  然后，它以指定的步幅遍历句子，将句子连接成段，并将这些段添加到列表中。

**Chinese (中文描述):** `_nlp_doc_to_texts` 方法用于将 Spacy 的 `Doc` 对象转换为文本段落（`segments`）的列表。
*   首先，它从 `Doc` 对象中提取句子，并使用 `strip()` 方法删除每个句子开头和结尾的空白字符。
*   然后，它会过滤掉空字符串的句子。
*   接下来，它使用 `_stride` 作为步长，将句子连接成段落。
*   `segment = " ".join(sents[i : i + self._chunk_size])` 这行代码将从 `sents` 列表中索引 `i` 开始的 `_chunk_size` 个句子连接成一个段落，并用空格分隔句子。
*   `segments.append(segment)` 将生成的段落添加到 `segments` 列表中。
*   `if i + self._chunk_size >= len(sents): break` 这行代码用于判断是否已经处理完所有的句子，如果是，则跳出循环。
*   最后，该方法返回包含文本段落的 `segments` 列表。

```python
    def split_text(self, text: str) -> List[str]:
        doc = self._nlp(text)
        segments = self._nlp_doc_to_texts(doc)
        return segments
```

**描述:**  `split_text` 方法将文本分割成段落。  它首先使用 Spacy 模型处理文本以创建一个 Doc 对象。  然后，它调用 `_nlp_doc_to_texts` 方法将 Doc 对象转换为段落列表。

**Chinese (中文描述):** `split_text` 方法用于将输入的文本分割成文本段落的列表。
*   首先，它使用 Spacy 模型 (`self._nlp`) 处理输入的文本，创建一个 `Doc` 对象。
*   然后，它调用 `_nlp_doc_to_texts` 方法，将 `Doc` 对象转换为文本段落的列表。
*   最后，该方法返回包含文本段落的列表。

```python
    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        pbar = tqdm(total=len(texts), desc="Splitting texts by sentences")
        num_workers = min(len(texts), self._num_parallel)
        for idx, doc in enumerate(self._nlp.pipe(texts, n_process=num_workers, batch_size=32)):
            segments = self._nlp_doc_to_texts(doc)
            for segment in segments:
                documents.append(Document(page_content=segment, metadata=deepcopy(_metadatas[idx])))
            pbar.update(1)
        pbar.close()

        return documents
```

**描述:** `create_documents` 方法根据文本列表创建 `langchain_core.documents.Document` 对象列表。 它接受文本列表和可选的元数据列表。  它使用 Spacy 模型并行处理文本。  对于每个文本，它将文本分割成段落，并为每个段落创建一个 Document 对象，并将元数据复制到 Document 对象中。

**Chinese (中文描述):** `create_documents` 方法用于从文本列表创建 `langchain_core.documents.Document` 对象列表。
*   `texts`:  要分割的文本列表。
*   `metadatas`:  可选的元数据列表，与文本列表一一对应。如果未提供元数据，则使用空字典 `[{}]` 作为默认值。
*   `_metadatas = metadatas or [{}] * len(texts)`：如果 `metadatas` 为 `None`，则创建一个长度与 `texts` 相同的列表，其中每个元素都是一个空字典。
*   `documents = []`：创建一个空列表，用于存储 `Document` 对象。
*   `pbar = tqdm(total=len(texts), desc="Splitting texts by sentences")`：创建一个 `tqdm` 进度条，用于显示处理进度。
*   `num_workers = min(len(texts), self._num_parallel)`：确定用于并行处理的进程数量，取 `texts` 的长度和 `self._num_parallel` 中的较小值。
*   `self._nlp.pipe(texts, n_process=num_workers, batch_size=32)`：使用 Spacy 模型并行处理文本列表。 `n_process` 参数指定并行处理的进程数量， `batch_size` 参数指定每个批次处理的文本数量。
*   循环遍历每个 `doc` 对象，使用 `self._nlp_doc_to_texts(doc)` 方法将 `doc` 对象分割成文本段落，并将每个文本段落包装成一个 `Document` 对象，然后添加到 `documents` 列表中。  `deepcopy(_metadatas[idx])` 确保元数据被正确复制，避免多个 `Document` 对象共享相同的元数据对象。
*   `pbar.update(1)`：更新进度条。
*   `pbar.close()`：关闭进度条。
*   最后，该方法返回包含 `Document` 对象的列表。

**Example Usage (用法示例):**

```python
from langchain_core.documents import Document

# Create an instance of the RecursiveSentenceSplitter
splitter = RecursiveSentenceSplitter(lang="zh", chunk_size=3, chunk_overlap=1)

# Example text
text = "这是一个示例文本。它包含多个句子。每个句子都很重要。我们需要将它分割成更小的块。这些块将用于信息检索。"

# Split the text into documents
documents = splitter.create_documents([text], metadatas=[{"source": "example"}])

# Print the documents
for doc in documents:
    print(doc)
```

**Chinese (中文示例):**

```python
from langchain_core.documents import Document

# 创建 RecursiveSentenceSplitter 的实例
splitter = RecursiveSentenceSplitter(lang="zh", chunk_size=3, chunk_overlap=1)

# 示例文本
text = "这是一个示例文本。它包含多个句子。每个句子都很重要。我们需要将它分割成更小的块。这些块将用于信息检索。"

# 将文本分割成文档
documents = splitter.create_documents([text], metadatas=[{"source": "示例"}])

# 打印文档
for doc in documents:
    print(doc)
```

这个例子展示了如何使用 `RecursiveSentenceSplitter` 类将中文文本分割成 `Document` 对象。  `chunk_size` 设置为 3，表示每个块包含 3 个句子，`chunk_overlap` 设置为 1，表示块之间重叠 1 个句子。 `metadatas` 参数用于为每个 `Document` 对象添加元数据。
