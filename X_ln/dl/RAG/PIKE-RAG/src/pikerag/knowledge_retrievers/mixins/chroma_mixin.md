Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\knowledge_retrievers\mixins\chroma_mixin.py`

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from chromadb.api.models.Collection import GetResult
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


ChromaMetaType = Union[str, int, float, bool]


def _check_ids_and_documents(ids: Optional[List[str]], documents: List[Document]) -> Optional[List[str]]:
    """
    检查提供的 ID 列表和文档列表是否一致。

    Args:
        ids: 文档 ID 列表 (可选).
        documents: 文档列表.

    Returns:
        如果提供了 ID，则返回原始 ID 列表；否则返回 None.
        如果 ID 数量和文档数量不匹配，会抛出异常.
    """
    if ids is None or len(ids) == 0:
        return None

    assert len(ids) == len(documents), f"{len(ids)} ids provided with {len(documents)} documents!"
    return ids

# Demo 使用示例
if __name__ == '__main__':
    from langchain_core.documents import Document
    # Example Usage 例子用法
    documents = [Document(page_content="This is document 1", metadata={"author": "Alice"}),
                Document(page_content="This is document 2", metadata={"author": "Bob"})]
    ids = ["doc1", "doc2"]

    checked_ids = _check_ids_and_documents(ids, documents)
    print(f"检查后的 IDs: {checked_ids}")

    documents = [Document(page_content="This is document 1", metadata={"author": "Alice"}),
                Document(page_content="This is document 2", metadata={"author": "Bob"})]
    ids = None

    checked_ids = _check_ids_and_documents(ids, documents)
    print(f"检查后的 IDs: {checked_ids}")
```

**解释:**

*   `_check_ids_and_documents(ids, documents)`:  这个函数检查 `ids` 列表（如果提供）的长度是否与 `documents` 列表的长度匹配。如果 `ids` 为 `None` 或空列表，则返回 `None`。如果长度不匹配，则引发 `AssertionError`。  主要用于确保在将文档添加到向量数据库时，提供的id与文档一一对应.

```python
def _documents_match(docs: List[Document], ids: Optional[List[str]], vector_store: Chroma) -> bool:
    """
    验证提供的文档是否与 Chroma 数据库中存储的文档匹配。

    Args:
        docs: 文档列表.
        ids: 文档 ID 列表 (可选).
        vector_store: Chroma 向量数据库实例.

    Returns:
        如果文档匹配，则返回 True；否则返回 False.
        如果向量数据库中的文档数量与提供的文档数量不匹配，或者文档内容或元数据不匹配，则返回 False.
    """
    if vector_store._collection.count() != len(docs):
        print(
            "[ChromaDB Loading Check] Document quantity not matched! "
            f"{vector_store._collection.count()} in store but {len(docs)} provided."
        )
        return False

    for idx in np.random.choice(len(docs), 3):
        content_in_doc: str = docs[idx].page_content
        meta_in_doc: dict = docs[idx].metadata
        if ids is not None:
            res = vector_store.get(ids=ids[idx])
            if len(res) == 0 or len(res["documents"]) == 0:
                print(f"[ChromaDB Loading Check] No data with id {ids[idx]} exist!")
                return False
            content_in_store = res["documents"][0]
            meta_in_store =res["metadatas"][0]
        else:
            doc_in_store = vector_store.similarity_search(query=content_in_doc, k=1)[0]
            content_in_store = doc_in_store.page_content
            meta_in_store = doc_in_store.metadata

        if content_in_store != content_in_doc:
            print(
                "[ChromaDB Loading Check] Document Content not matched:\n"
                f"  In store: {content_in_store}\n"
                f"  In Doc: {content_in_doc}"
            )
            return False

        for key, value in meta_in_doc.items():
            if key not in meta_in_store:
                print(f"[ChromaDB Loading Check] Metadata {key} in doc but not in store!")
                return False

            if isinstance(value, float):
                if abs(value - meta_in_store[key]) > 1e-9:
                    print(f"[ChromaDB Loading Check] Metadata {key} not matched: {value} v.s. {meta_in_store[key]}")
                    return False
            elif meta_in_store[key] != value:
                print(f"[ChromaDB Loading Check] Metadata {key} not matched: {value} v.s. {meta_in_store[key]}")
                return False

    return True

# Example Usage (Requires Chroma setup) 需要 Chroma 环境
if __name__ == '__main__':
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings  # 需要安装 openai 包

    # 1. 初始化 Chroma (确保 Chroma 数据库已运行)
    persist_directory = "db" # 指定持久化目录，不指定则是内存数据库，进程结束即消失
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # 2. 创建一些文档
    documents = [
        Document(page_content="This is document 1 about cats.", metadata={"author": "Alice"}),
        Document(page_content="This is document 2 about dogs.", metadata={"author": "Bob"}),
    ]
    ids = ["doc1", "doc2"]

    # 3. 先将文档添加到向量数据库中
    vector_store.add_documents(documents=documents, ids=ids)

    # 4. 检查文档是否匹配
    match = _documents_match(documents, ids, vector_store)
    print(f"文档是否匹配: {match}")

    # 5. 清理向量数据库，方便下次运行
    vector_store.delete_collection()

```

**解释:**

*   `_documents_match(docs, ids, vector_store)`: 此函数验证提供的文档是否与 Chroma 数据库中的文档匹配。首先，它检查向量数据库中的文档数量是否与提供的文档数量匹配。 然后，它随机选择三个文档，并检查其内容和元数据是否与向量数据库中的相应文档匹配。如果任何检查失败，该函数返回 `False`。 使用`np.random.choice`随机抽取一定数量的doc进行验证，避免验证所有文档导致时间过长。
*   函数利用`vector_store.get(ids=ids[idx])` 通过id从数据库中获取文档，或者利用`vector_store.similarity_search(query=content_in_doc, k=1)[0]` 通过相似度搜索获取文档.
*   对于浮点类型的metadata，需要用容差进行比较`abs(value - meta_in_store[key]) > 1e-9`， 避免浮点数精度问题导致的误判.

```python
def load_vector_store(
    collection_name: str,
    persist_directory: str,
    embedding: Embeddings=None,
    documents: List[Document]=None,
    ids: List[str]=None,
    exist_ok: bool=True,
    metadata: dict=None,
) -> Chroma:
    """
    加载或创建 Chroma 向量数据库。

    Args:
        collection_name: 集合名称.
        persist_directory: 持久化目录.
        embedding: 嵌入模型 (可选).
        documents: 文档列表 (可选).
        ids: 文档 ID 列表 (可选).
        exist_ok: 如果集合已存在，是否可以继续 (默认: True).
        metadata: 集合元数据 (可选).

    Returns:
        Chroma 向量数据库实例.
    """
    vector_store = Chroma(collection_name, embedding, persist_directory, collection_metadata=metadata)

    if documents is None or len(documents) == 0:
        return vector_store

    assert exist_ok or vector_store._collection.count() == 0, f"Collection {collection_name} already exist!"

    ids = _check_ids_and_documents(ids, documents)

    if _documents_match(documents, ids, vector_store):
        print(f"Chroma DB: {collection_name} loaded.")
        return vector_store

    vector_store.delete_collection()

    # Direct using of vector_store.add_documents() will raise InvalidCollectionException.
    print(f"Start to build up the Chroma DB: {collection_name}")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_metadata=metadata,
    )
    print(f"Chroma DB: {collection_name} Building-Up finished.")
    return vector_store

# Example Usage (Requires Chroma and OpenAI setup)  需要 Chroma 和 OpenAI 环境
if __name__ == '__main__':
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings  # 需要安装 openai 包
    import os

    # 1. 定义参数
    collection_name = "my_collection"
    persist_directory = "db"
    embedding = OpenAIEmbeddings()
    documents = [
        Document(page_content="This is document 1 about apples.", metadata={"author": "Alice"}),
        Document(page_content="This is document 2 about bananas.", metadata={"author": "Bob"}),
    ]
    ids = ["doc1", "doc2"]
    exist_ok = True
    metadata = {"purpose": "testing"}

    # 2. 加载或创建向量数据库
    vector_store = load_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding=embedding,
        documents=documents,
        ids=ids,
        exist_ok=exist_ok,
        metadata=metadata,
    )

    # 3. 使用向量数据库进行一些操作 (例如，相似性搜索)
    results = vector_store.similarity_search(query="fruit", k=1)
    print(f"相似性搜索结果: {results[0].page_content}")

    # 4.  清理数据库 (可选，如果希望删除数据)
    vector_store.delete_collection()

```

**解释:**

*   `load_vector_store(...)`: 此函数加载或创建 Chroma 向量数据库。 如果 `documents` 为 `None` 或为空，则该函数返回一个空的 Chroma 向量数据库。 否则，它会检查集合是否已经存在。 如果 `exist_ok` 为 `False` 并且集合已经存在，则该函数会引发异常。 如果集合不存在，则该函数使用提供的文档创建一个新的集合。  如果已经存在集合，并且文档匹配，则直接加载。 如果文档不匹配，则删除重建。函数利用`Chroma.from_documents`从文档创建数据库。
*    该函数包装了Chroma的初始化和创建过程，简化了向量数据库的加载和创建。并且增加了数据一致性的校验功能。

```python
class ChromaMixin:
    """
    Chroma 向量数据库集成的 Mixin 类。
    """
    def _init_chroma_mixin(self):
        """
        初始化 ChromaMixin 类的实例。
        """
        self.retrieve_k: int = self._retriever_config.get("retrieve_k", 4)
        self.retrieve_score_threshold: float = self._retriever_config.get("retrieve_score_threshold", 0.5)

    def _get_doc_with_query(
        self, query: str, store: Chroma, retrieve_k: int=None, score_threshold: float=None,
    ) -> List[Tuple[Document, float]]:
        """
        使用给定的查询从向量数据库中检索文档。

        Args:
            query: 查询字符串.
            store: Chroma 向量数据库实例.
            retrieve_k: 要检索的文档数量 (可选).
            score_threshold: 相关性得分阈值 (可选).

        Returns:
            文档和相关性得分的列表。
        """
        if retrieve_k is None:
            retrieve_k = self.retrieve_k
        if score_threshold is None:
            score_threshold = self.retrieve_score_threshold

        infos: List[Tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query=query,
            k=retrieve_k,
            score_threshold=score_threshold,
        )

        filtered_docs = [(doc, score) for doc, score in infos if score >= score_threshold]
        sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)

        return sorted_docs

    def _get_infos_with_given_meta(
        self, store: Chroma, meta_name: str, meta_value: Union[ChromaMetaType, List[ChromaMetaType]],
    ) -> Tuple[List[str], List[str], List[Dict[str, ChromaMetaType]]]:
        """
        根据给定的元数据从向量数据库中检索文档信息。

        Args:
            store: Chroma 向量数据库实例.
            meta_name: 元数据名称.
            meta_value: 元数据值或元数据值列表.

        Returns:
            文档 ID 列表、文档内容列表和文档元数据列表。
        """
        if isinstance(meta_value, list):
            filter = {meta_name: {"$in": meta_value}}
        else:
            filter = {meta_name: meta_value}

        results: GetResult = store.get(where=filter)
        ids, chunks, metadatas = results["ids"], results["documents"], results["metadatas"]
        return ids, chunks, metadatas

    def _get_scoring_func(self, store: Chroma):
        """
        获取向量数据库的相关性得分函数。

        Args:
            store: Chroma 向量数据库实例.

        Returns:
            相关性得分函数.
        """
        return store._select_relevance_score_fn()

# Example Usage (Requires Chroma and OpenAI setup)  需要 Chroma 和 OpenAI 环境
if __name__ == '__main__':
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings  # 需要安装 openai 包
    from langchain_chroma import Chroma

    # 1. 初始化 Chroma (确保 Chroma 数据库已运行)
    persist_directory = "db" # 指定持久化目录
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # 2. 创建一些文档并添加到向量数据库中
    documents = [
        Document(page_content="This is document 1 about apples.", metadata={"author": "Alice", "category": "fruit"}),
        Document(page_content="This is document 2 about bananas.", metadata={"author": "Bob", "category": "fruit"}),
        Document(page_content="This is document 3 about cars.", metadata={"author": "Charlie", "category": "vehicle"}),
    ]
    ids = ["doc1", "doc2", "doc3"]
    vector_store.add_documents(documents=documents, ids=ids)

    # 3. 创建 ChromaMixin 的实例 (需要一个假的 _retriever_config)
    class MyClass(ChromaMixin):
        def __init__(self):
            self._retriever_config = {"retrieve_k": 2, "retrieve_score_threshold": 0.7}
            self._init_chroma_mixin() # 初始化 ChromaMixin
    mixin = MyClass()

    # 4. 使用 _get_doc_with_query 进行查询
    query = "fruit"
    results = mixin._get_doc_with_query(query, vector_store)
    print(f"查询结果: {[r[0].page_content for r in results]}")

    # 5. 使用 _get_infos_with_given_meta 根据元数据进行检索
    meta_name = "category"
    meta_value = "fruit"
    ids, chunks, metadatas = mixin._get_infos_with_given_meta(vector_store, meta_name, meta_value)
    print(f"根据元数据检索结果: {chunks}")

    # 6. 获取评分函数
    scoring_func = mixin._get_scoring_func(vector_store)
    print(f"评分函数: {scoring_func}") # 输出函数对象

    # 7.  清理数据库 (可选，如果希望删除数据)
    vector_store.delete_collection()
```

**解释:**

*   `ChromaMixin`: 这是一个 Mixin 类，旨在简化与 Chroma 向量数据库的集成。 它提供了一组方法来查询、检索和管理向量数据库中的文档。
*   `_init_chroma_mixin(self)`: 从`self._retriever_config` 获取 `retrieve_k` 和 `retrieve_score_threshold`的值， 并赋给类的成员变量。
*   `_get_doc_with_query(self, query: str, store: Chroma, retrieve_k: int=None, score_threshold: float=None)`:  使用提供的查询从向量数据库中检索文档。  它使用`store.similarity_search_with_relevance_scores` 方法来执行相似性搜索，并返回满足指定相关性得分阈值的文档。
*   `_get_infos_with_given_meta(self, store: Chroma, meta_name: str, meta_value: Union[ChromaMetaType, List[ChromaMetaType]])`:  根据给定的元数据从向量数据库中检索文档信息。  它使用 `store.get(where=filter)` 方法来执行元数据过滤，并返回匹配的文档 ID、内容和元数据。其中，`filter`构造了Chroma查询语句。
*   `_get_scoring_func(self, store: Chroma)`: 获取向量数据库的相关性评分函数。 它使用 `store._select_relevance_score_fn()` 方法来检索评分函数。

总的来说，这段代码提供了一系列函数和类，用于简化 Chroma 向量数据库的使用。 它们涵盖了文档加载、数据验证、查询和元数据检索等常见任务.  `ChromaMixin` 将 Chroma 向量数据库的功能封装成一个可重用的组件，可以方便地集成到其他类中。
