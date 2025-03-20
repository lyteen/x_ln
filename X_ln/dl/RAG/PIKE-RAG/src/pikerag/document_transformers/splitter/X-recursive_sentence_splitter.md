Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\splitter\recursive_sentence_splitter.py`

好的，我们重新开始，这次我会更注重代码的改进和更清晰的中文解释。

**1. 优化的`RecursiveSentenceSplitter`:**

```python
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


class RecursiveSentenceSplitter(TextSplitter):
    """
    递归句子分割器，用于将文本分割成句子块。

    旨在提高效率，支持多语言，并能更好地处理长文本。
    """

    NAME = "RecursiveSentenceSplitter"

    def __init__(
        self,
        lang: str = "en",
        nlp_max_len: int = 4000000,
        num_parallel: int = 4,
        chunk_size: int = 12,
        chunk_overlap: int = 4,
        use_tqdm: bool = True,  # 新增参数：是否使用tqdm
        **kwargs,
    ):
        """
        初始化函数。

        Args:
            lang (str): 语言类型，"en"为英语，"zh"为中文。
            nlp_max_len (int): SpaCy模型处理的最大文本长度。
            num_parallel (int): 并行处理的数量。
            chunk_size (int): 每个文本块包含的句子数量。
            chunk_overlap (int): 相邻文本块之间的句子重叠数量。
            use_tqdm (bool): 是否使用tqdm显示进度条。
        """
        super().__init__(chunk_size, chunk_overlap)
        self._stride: int = self._chunk_size - self._chunk_overlap
        self._num_parallel: int = num_parallel
        self._use_tqdm: bool = use_tqdm  # 保存tqdm的使用标志

        self._load_model(lang, nlp_max_len)

        return

    def _load_model(self, language: str, nlp_max_len: int) -> None:
        """
        加载SpaCy模型。

        Args:
            language (str): 语言类型。
            nlp_max_len (int): SpaCy模型处理的最大文本长度。
        """
        assert language in LANG2MODELNAME, f"Spacy model not specified for language: {language}."

        model_name = LANG2MODELNAME[language]
        try:
            self._nlp = spacy.load(model_name)
        except OSError:  # 明确捕获OSError
            spacy.cli.download(model_name)
            self._nlp = spacy.load(model_name)
        self._nlp.max_length = nlp_max_len
        return

    def _nlp_doc_to_texts(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        将SpaCy文档对象转换为文本块列表。

        Args:
            doc (spacy.tokens.Doc): SpaCy文档对象。

        Returns:
            List[str]: 文本块列表。
        """
        sents = [sent.text.strip() for sent in doc.sents]
        sents = [sent for sent in sents if len(sent) > 0]

        segments: List[str] = []
        for i in range(0, len(sents), self._stride):
            segment = " ".join(sents[i : i + self._chunk_size])
            segments.append(segment)
            if i + self._chunk_size >= len(sents):
                break

        return segments

    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成文本块。

        Args:
            text (str): 输入文本。

        Returns:
            List[str]: 文本块列表。
        """
        doc = self._nlp(text)
        segments = self._nlp_doc_to_texts(doc)
        return segments

    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        """
        将文本列表转换为文档对象列表。

        Args:
            texts (List[str]): 文本列表。
            metadatas (Optional[List[dict]]): 元数据列表。

        Returns:
            List[Document]: 文档对象列表。
        """
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        # 根据self._use_tqdm的设定决定是否显示进度条
        pbar = tqdm(total=len(texts), desc="Splitting texts by sentences", disable=not self._use_tqdm)
        num_workers = min(len(texts), self._num_parallel)
        for idx, doc in enumerate(self._nlp.pipe(texts, n_process=num_workers, batch_size=32)):
            segments = self._nlp_doc_to_texts(doc)
            for segment in segments:
                documents.append(Document(page_content=segment, metadata=deepcopy(_metadatas[idx])))
            pbar.update(1)
        pbar.close()

        return documents


# Demo Usage 演示用法
if __name__ == '__main__':
    # 创建 RecursiveSentenceSplitter 实例，禁用tqdm
    splitter = RecursiveSentenceSplitter(lang="zh", chunk_size=8, chunk_overlap=2, num_parallel=2, use_tqdm=False)

    # 示例文本列表
    texts = [
        "这是一个很长的中文句子。它包含了很多信息。我们希望将它分割成更小的块。",
        "这是另一个文本。它也很长，需要分割。",
        "第三个文本，同样需要处理。"
    ]

    # 创建文档对象
    documents = splitter.create_documents(texts)

    # 打印结果
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
```

**主要改进和说明：**

*   **更清晰的注释:**  使用更详细的注释来解释代码的各个部分。
*   **Error Handling (错误处理):**  使用 `try...except OSError` 块来处理 SpaCy 模型下载可能出现的错误。 这使得代码更加健壮。
*   **`use_tqdm` 参数:** 添加了一个 `use_tqdm` 参数，允许用户控制是否显示进度条。 在非交互式环境或需要减少输出时，这很有用。
*   **Progress Bar Control (进度条控制):**  使用 `disable` 参数来控制 `tqdm` 进度条的显示。
*   **明确捕获`OSError`** 更精确的错误处理.
*   **示例用法：** 增加了一个简单的示例用法，展示如何使用 `RecursiveSentenceSplitter`。

**中文解释：**

这段代码实现了一个递归句子分割器。  它可以将一段长文本分割成更小的文本块，每个文本块包含一定数量的句子，并且相邻的文本块之间有一定的重叠。  这个分割器使用 SpaCy 库来进行句子分割，并支持中文和英文。  `use_tqdm=False`关闭进度条，适合在没有终端输出的环境中使用。

**如何使用：**

1.  创建 `RecursiveSentenceSplitter` 类的实例，并指定语言、文本块大小、重叠大小、并行处理数量，以及是否显示进度条。
2.  调用 `create_documents` 方法，传入文本列表和可选的元数据列表。  该方法将返回一个文档对象列表，每个文档对象包含一个文本块和相应的元数据。

这种结构允许更细粒度的控制，并提高了在各种环境中的可用性。 明确的错误处理可以确保代码在各种情况下都能正常运行。
