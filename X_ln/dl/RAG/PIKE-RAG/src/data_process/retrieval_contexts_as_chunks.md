Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\retrieval_contexts_as_chunks.py`

好的，现在重新开始，并按照您的要求进行代码解释和示例展示。

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from typing import Dict, List, Literal

import jsonlines
import numpy as np


def read_testing_suite(filepath: str) -> List[dict]:
    """
    读取测试数据集。

    Args:
        filepath (str): JSONL 文件路径.

    Returns:
        List[dict]: 测试数据集列表，每个元素是一个字典.
    """
    with jsonlines.open(filepath, "r") as reader:
        testing_suite = [data for data in reader]
    return testing_suite


# 示例用法：
# file_path = "data/hotpotqa/dev_500.jsonl"  # 假设存在这个文件
# data = read_testing_suite(file_path)
# print(f"读取了 {len(data)} 条数据.")
# 这段代码读取一个JSONL文件，该文件包含测试数据，每一行代表一个问答对。


def get_chunks_from_testing_suite(testing_suite: List[dict]) -> Dict[str, List[str]]:
    """
    从测试数据集中提取文本块（chunks），按照标题进行分组。

    Args:
        testing_suite (List[dict]): 测试数据集列表.

    Returns:
        Dict[str, List[str]]: 字典，键是标题，值是文本块列表.
    """
    chunks_by_title = {}
    for qa in testing_suite:
        for retrieval_context in qa["metadata"]["retrieval_contexts"]:
            title = retrieval_context["title"]
            contents = retrieval_context["contents"]
            if title not in chunks_by_title:
                chunks_by_title[title] = []
            chunks_by_title[title].append(contents)
    chunks_by_title = {title: list(set(chunks)) for title, chunks in chunks_by_title.items()} # 去重

    chunk_count = [len(lst) for _, lst in chunks_by_title.items()]
    print(
        f"{len(chunks_by_title)} titles in total. "
        f"{sum(chunk_count)} chunks in total. "
        f"Chunk count: {min(chunk_count)} ~ {max(chunk_count)}, avg: {np.mean(chunk_count)}"
    )
    return chunks_by_title

# 示例用法：
# file_path = "data/hotpotqa/dev_500.jsonl"  # 假设存在这个文件
# data = read_testing_suite(file_path)
# chunks = get_chunks_from_testing_suite(data)
# print(f"提取了 {len(chunks)} 个标题.")

# 这段代码遍历测试数据集，提取每个问答对中的检索上下文，然后按标题对文本块进行分组。 它还打印了有关文本块数量的一些统计信息。

if __name__ == "__main__":
    data_dir = "data"
    datasets = ["hotpotqa", "two_wiki", "musique"]
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        input_path = os.path.join(dataset_dir, "dev_500.jsonl")
        output_path = os.path.join(dataset_dir, "dev_500_retrieval_contexts_as_chunks.jsonl")

        print(f"\n#### Dataset: {dataset}")
        testing_suite = read_testing_suite(input_path)
        chunks_by_title = get_chunks_from_testing_suite(testing_suite)
        chunk_dicts: List[Dict[Literal["chunk_id", "title", "content"], str]] = [
            {
                "chunk_id": f"{title}-{cidx}-{len(chunks)}",
                "title": title,
                "content": chunk,
            }
            for title, chunks in chunks_by_title.items()
            for cidx, chunk in enumerate(chunks)
        ]

        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(chunk_dicts)

        counter = {}
        for qa in testing_suite:
            count = len(qa["metadata"]["retrieval_contexts"])
            counter[count] = counter.get(count, 0) + 1
        print("Retrieval Contexts:", counter)

        counter = {}
        for qa in testing_suite:
            count = len(qa["metadata"]["supporting_facts"])
            counter[count] = counter.get(count, 0) + 1
        print("Supporting Facts:", counter)

# 这部分是主程序，它遍历指定的数据集，读取测试数据集，提取文本块，并将它们保存到一个新的JSONL文件中。 它还计算并打印了有关检索上下文和支持事实的一些统计信息。

# 例如，在控制台中运行此代码，如果 data 目录下有 hotpotqa, two_wiki, musique 三个目录，每个目录中都有 dev_500.jsonl 文件，
# 那么它会处理这些文件，生成相应的 chunk 文件，并打印相关的统计数据。
```

**代码关键部分的详细解释：**

1.  **`read_testing_suite(filepath: str) -> List[dict]` 函数:**
    *   **功能:** 从指定的 JSONL 文件中读取测试数据集。
    *   **参数:**
        *   `filepath (str)`:  JSONL 文件的路径。
    *   **返回值:**
        *   `List[dict]`:  一个列表，其中每个元素是一个字典，代表测试数据集中的一个问答对。
    *   **示例:**
        ```python
        import jsonlines
        # 假设存在一个名为 'test_data.jsonl' 的文件，其中包含 JSONL 格式的数据
        file_path = 'test_data.jsonl'
        try:
            test_suite = read_testing_suite(file_path)
            print(f"成功读取了 {len(test_suite)} 条数据。")
            # 可以进一步处理 test_suite 中的数据
        except FileNotFoundError:
            print(f"错误：文件 '{file_path}' 未找到。")

        # 创建一个示例 'test_data.jsonl' 文件
        sample_data = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the highest mountain?", "answer": "Everest"}
        ]

        with jsonlines.open('test_data.jsonl', mode='w') as writer:
            writer.write_all(sample_data)
        ```
        *   这段代码演示了如何使用`read_testing_suite`函数读取JSONL文件。同时，为了方便演示，创建了一个小的jsonl文件

2.  **`get_chunks_from_testing_suite(testing_suite: List[dict]) -> Dict[str, List[str]]` 函数:**
    *   **功能:**  从测试数据集中提取文本块（chunks），并按标题进行分组。  它还计算并打印一些关于文本块数量的统计信息。
    *   **参数:**
        *   `testing_suite (List[dict])`:  测试数据集列表，由 `read_testing_suite` 函数返回。
    *   **返回值:**
        *   `Dict[str, List[str]]`:  一个字典，其中键是标题，值是与该标题关联的文本块的列表。  去除了重复的文本块。
    *   **示例:**
        ```python
        # 假设 test_suite 已经通过 read_testing_suite 读取
        # 且数据格式满足该函数的需求
        # 构造一个示例测试数据集
        test_suite = [
            {"metadata": {"retrieval_contexts": [{"title": "Article A", "contents": "Content 1"},
                                                   {"title": "Article A", "contents": "Content 2"}]}},
            {"metadata": {"retrieval_contexts": [{"title": "Article B", "contents": "Content 3"}]}},
            {"metadata": {"retrieval_contexts": [{"title": "Article A", "contents": "Content 1"}]}}  # 重复的文本块
        ]
        chunks = get_chunks_from_testing_suite(test_suite)
        print(chunks)
        # 预期输出（顺序可能不同）:
        # {'Article A': ['Content 1', 'Content 2'], 'Article B': ['Content 3']}

        ```
        *   这个例子创建了一个示例测试数据集，其中包含不同的标题和内容，并展示了如何使用 `get_chunks_from_testing_suite` 函数来提取和组织这些内容。 请注意，重复的内容已被删除。

3.  **`if __name__ == "__main__":` 代码块:**
    *   **功能:**  这是主程序入口。它遍历指定的数据集，读取测试数据，提取文本块，并将它们保存到新的 JSONL 文件中。  它还计算并打印关于检索上下文和支持事实的一些统计信息。
    *   **主要步骤:**
        1.  定义数据集列表 `datasets` 和数据目录 `data_dir`。
        2.  遍历每个数据集。
        3.  构建输入文件路径 `input_path` 和输出文件路径 `output_path`。
        4.  使用 `read_testing_suite` 函数读取测试数据集。
        5.  使用 `get_chunks_from_testing_suite` 函数从测试数据集中提取文本块。
        6.  创建一个列表 `chunk_dicts`，其中包含字典，每个字典表示一个文本块，包含 `chunk_id`、`title` 和 `content`。
        7.  使用 `jsonlines.open` 函数将 `chunk_dicts` 列表写入到输出文件 `output_path` 中。
        8.  计算并打印关于检索上下文和支持事实的统计信息。
    *   **示例:**
        *   假设您已经按照类似 HotpotQA 的格式组织了数据，并且在 `data/hotpotqa/` 目录下有一个名为 `dev_500.jsonl` 的文件，此文件包含类似以下结构的JSONL数据：

        ```json
        {"question": "...", "answer": "...", "metadata": {"retrieval_contexts": [{"title": "Title A", "contents": "Content A"}, {"title": "Title B", "contents": "Content B"}], "supporting_facts": [...]}}
        {"question": "...", "answer": "...", "metadata": {"retrieval_contexts": [{"title": "Title C", "contents": "Content C"}], "supporting_facts": [...]}}
        ```

        *   运行代码后，会在相同的 `data/hotpotqa/` 目录下生成一个新的文件 `dev_500_retrieval_contexts_as_chunks.jsonl`，其中包含以下格式的数据:

        ```json
        {"chunk_id": "Title A-0-2", "title": "Title A", "content": "Content A"}
        {"chunk_id": "Title B-0-1", "title": "Title B", "content": "Content B"}
        {"chunk_id": "Title C-0-1", "title": "Title C", "content": "Content C"}
        ```

        *   此外，控制台会打印出有关数据集中标题数量、文本块数量以及检索上下文和支持事实分布的统计信息。

        *   **运行前注意事项：**

            *   请确保已安装 `jsonlines` 和 `numpy` 库：`pip install jsonlines numpy`
            *   如果 `data` 目录或其中的子目录不存在，你需要手动创建它们。
            *   为了运行此代码，你需要将示例JSONL数据放置在 `data/hotpotqa/dev_500.jsonl`、`data/two_wiki/dev_500.jsonl` 和 `data/musique/dev_500.jsonl` （如果你的 datasets 列表包含这些数据集）文件中，或者修改 `datasets` 变量以匹配你实际的数据集目录。
        ```bash
        mkdir -p data/hotpotqa
        mkdir -p data/two_wiki
        mkdir -p data/musique
        #  将你的 JSONL 文件放在这些目录中
        ```

**总结:**

这段代码的主要功能是从 JSONL 格式的测试数据集中提取文本块，并将它们组织成另一种 JSONL 格式的文件。 它还提供了一些统计信息，可以帮助您了解数据集的结构。  这对于预处理问答数据集以进行检索增强生成 (RAG) 或其他自然语言处理任务非常有用。
