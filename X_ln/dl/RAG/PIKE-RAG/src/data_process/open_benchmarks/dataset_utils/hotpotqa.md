Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\hotpotqa.py`

好的，现在重新开始。我对之前的回答格式进行了调整，使其更清晰和易于理解。以下是对代码片段的分解和中文解释，并附带使用示例。

**1. 下载原始数据 (`download_raw_data`)**

```python
import requests

def download_raw_data(raw_filepath: str, split: str) -> None:
    """
    从指定URL下载原始数据并保存到文件中。

    参数:
    raw_filepath (str): 保存原始数据的文件路径。
    split (str): 数据集分割的名称 ("train" 或 "dev")。
    """
    url = split2url[split]  # 根据split选择对应的URL
    with requests.get(url) as response: # 发起网络请求
        with open(raw_filepath, "wb") as fout: # 以二进制写入模式打开文件
            for chunk in response.iter_content(chunk_size=1024): # 迭代响应内容，每次读取1024字节
                fout.write(chunk) # 将数据块写入文件
    return

# Demo Usage 演示用法
if __name__ == '__main__':
    # 创建一个临时的测试文件
    test_filepath = "test_data.json"
    download_raw_data(test_filepath, "train")
    print(f"数据已下载到：{test_filepath}")

    # 清理测试文件 (可选)
    import os
    os.remove(test_filepath)
    print(f"已删除测试文件：{test_filepath}")
```

**描述:** 这个函数的作用是从指定的URL下载HotpotQA数据集的原始JSON数据，并将其保存到本地文件中。 它使用`requests`库来发起HTTP请求，并使用二进制写入模式 (`"wb"`) 将数据写入文件。`split` 参数用于指定下载哪个数据集分割 (训练集或开发集)。

**如何使用:**  指定要保存文件的路径 (`raw_filepath`) 以及要下载的数据集分割 (`split`)，然后调用此函数。 首次运行时，它会从网络下载数据。后续运行会直接使用本地文件（如果存在）。

**2. 加载原始数据 (`load_raw_data`)**

```python
import json
import os

def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    """
    从指定文件中加载原始JSON数据。如果文件不存在，则先下载。

    参数:
    dataset_dir (str): 数据集目录。
    split (str): 数据集分割的名称 ("train" 或 "dev")。

    返回值:
    List[dict]: 包含JSON数据的Python字典列表。
    """
    raw_dir = os.path.join(dataset_dir, "raw") # 构建raw目录
    os.makedirs(raw_dir, exist_ok=True) # 创建raw目录，如果存在则不报错
    raw_filepath = os.path.join(raw_dir, f"{split}.json") # 构建文件路径
    if not os.path.exists(raw_filepath): # 如果文件不存在，则下载
        download_raw_data(raw_filepath, split)

    with open(raw_filepath, "r", encoding="utf-8") as fin: # 以utf-8编码读取文件
        dataset = json.load(fin) # 将JSON数据加载到Python字典
    return dataset

# Demo Usage 演示用法
if __name__ == '__main__':
    # 设置数据集目录和分割
    dataset_dir = "temp_dataset" # 临时目录
    split = "train"

    # 加载原始数据
    data = load_raw_data(dataset_dir, split)
    print(f"已加载 {split} 分割的数据，共有 {len(data)} 个样本。")

    # 打印第一个样本的键 (key)
    if data:
        print(f"第一个样本的键：{data[0].keys()}")

    # 清理临时目录 (可选)
    import shutil
    shutil.rmtree(dataset_dir, ignore_errors=True)
    print(f"已删除临时目录：{dataset_dir}")
```

**描述:**  这个函数从本地文件加载HotpotQA数据集的原始JSON数据。如果文件不存在，它会首先调用 `download_raw_data` 函数下载数据。它使用 `json.load` 将JSON数据解析为Python字典列表。

**如何使用:**  指定数据集的目录 (`dataset_dir`) 和要加载的数据集分割 (`split`)，然后调用此函数。 它会返回一个包含数据集的Python字典列表。

**3. 获取索引句子 (`get_idx_sentence`)**

```python
from typing import List, Tuple, Optional

def get_idx_sentence(
    title: str, idx: int, original_contexts: List[Tuple[str, List[str]]], verbose: bool=False,
) -> Optional[str]:
    """
    从给定的上下文中，根据标题和索引获取句子。

    参数:
    title (str): 文档标题。
    idx (int): 句子索引。
    original_contexts (List[Tuple[str, List[str]]]): 上下文列表，每个元素是一个元组，包含标题和句子列表。
    verbose (bool): 是否打印详细的调试信息。

    返回值:
    Optional[str]: 找到的句子，如果未找到则返回 None。
    """
    for item_title, sentences in original_contexts: # 遍历上下文列表
        if item_title == title and idx < len(sentences): # 找到匹配的标题和索引
            return sentences[idx] # 返回对应的句子

    if verbose: # 如果verbose为True，则打印调试信息
        print(f"######## Indexed sentence not found ########")
        print(f"title: {title}, idx: {idx}")
        for item_title, sentences in original_contexts:
            if item_title != title:
                continue
            for i, sentence in enumerate(sentences):
                print(f"  {i}: {sentence}")
            print()
    return None # 未找到句子，返回None

# Demo Usage 演示用法
if __name__ == '__main__':
    # 创建一个示例上下文
    contexts = [
        ("Document1", ["Sentence 1", "Sentence 2", "Sentence 3"]),
        ("Document2", ["Sentence A", "Sentence B"]),
    ]

    # 尝试获取句子
    sentence = get_idx_sentence("Document1", 1, contexts)
    print(f"获取到的句子：{sentence}")

    # 尝试获取不存在的句子
    sentence = get_idx_sentence("Document3", 0, contexts, verbose=True)
    print(f"获取不存在的句子：{sentence}")
```

**描述:**  这个函数在给定的上下文中查找特定标题和索引对应的句子。上下文是一个列表，其中每个元素都是一个元组，包含文章标题和句子列表。 如果找不到匹配的句子，该函数返回 `None`。 `verbose` 参数控制是否打印详细的调试信息。

**如何使用:**  提供标题 (`title`)、句子索引 (`idx`) 和上下文列表 (`original_contexts`)，然后调用此函数。

**4. 获取支持事实 (`get_supporting_facts`)**

```python
from typing import List, Tuple, Optional, Dict, Literal

def get_supporting_facts(
    supporting_fact_tuples: List[Tuple[str, int]],
    context_tuples: List[Tuple[str, List[str]]],
) -> Optional[List[Dict[Literal["type", "title", "contents"], str]]]:
    """
    从上下文中提取支持事实的内容。

    参数:
    supporting_fact_tuples (List[Tuple[str, int]]): 支持事实的元组列表，每个元组包含标题和句子索引。
    context_tuples (List[Tuple[str, List[str]]]): 上下文列表，每个元素是一个元组，包含标题和句子列表。

    返回值:
    Optional[List[Dict[Literal["type", "title", "contents"], str]]]: 支持事实列表，每个元素是一个字典，包含类型、标题和内容。如果任何支持事实未找到，则返回 None。
    """
    supporting_facts: List[dict] = [] # 初始化支持事实列表
    for title, sent_idx in supporting_fact_tuples: # 遍历支持事实元组
        content: Optional[str] = get_idx_sentence(title, sent_idx, context_tuples) # 获取句子内容
        if content is None: # 如果句子内容未找到，则返回None
            return None
        supporting_facts.append( # 添加支持事实到列表
            {
                "type": "wikipedia",
                "title": title,
                "contents": content,
            }
        )
    return supporting_facts # 返回支持事实列表

# Demo Usage 演示用法
if __name__ == '__main__':
    # 创建示例支持事实和上下文
    supporting_facts_tuples = [
        ("Document1", 0),
        ("Document2", 1),
    ]
    contexts = [
        ("Document1", ["Sentence 1", "Sentence 2"]),
        ("Document2", ["Sentence A", "Sentence B", "Sentence C"]),
    ]

    # 获取支持事实
    supporting_facts = get_supporting_facts(supporting_facts_tuples, contexts)
    print(f"获取到的支持事实：{supporting_facts}")
```

**描述:**  这个函数根据提供的支持事实元组列表，从给定的上下文中提取支持事实的内容。 支持事实元组包含文档标题和句子索引。 该函数返回一个字典列表，其中每个字典包含支持事实的类型、标题和内容。 如果任何支持事实的内容找不到，则该函数返回 `None`。

**如何使用:**  提供支持事实元组列表 (`supporting_fact_tuples`) 和上下文元组列表 (`context_tuples`)，然后调用此函数。

**5. 格式化原始数据 (`format_raw_data`)**

```python
import uuid
from typing import Dict, List, Optional, Literal, Tuple

# 假设已定义函数 infer_question_type，此处省略

def format_raw_data(raw: dict) -> Optional[dict]:
    """
    将原始数据格式化为模型所需的格式。

    参数:
    raw (dict): 原始数据字典。

    返回值:
    Optional[dict]: 格式化后的数据字典，如果支持事实未找到，则返回 None。
    """
    # Step 1: extract supporting facts contents from retrieval contexts.
    # Skip sample if supporting fact not found. Currently there is one error case in `dev` split with `_id`:
    # 5ae61bfd5542992663a4f261
    supporting_facts = get_supporting_facts(raw["supporting_facts"], raw["context"]) # 获取支持事实
    if supporting_facts is None: # 如果支持事实未找到，则返回None
        return None

    # Step 2: re-format to fit dataset protocol.
    answer_labels: List[str] = [raw["answer"]] # 提取答案标签
    # 假设已经实现了 infer_question_type
    def infer_question_type(answer_labels: List[str]) -> str:
        if "yes" in answer_labels or "no" in answer_labels:
            return "yesno"
        return "extractive"
    qtype: str = infer_question_type(answer_labels) # 推断问题类型

    formatted_data = { # 创建格式化后的数据字典
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["_id"],
            "supporting_facts": supporting_facts,
            "retrieval_contexts": [
                {
                    "type": "wikipedia",
                    "title": title,
                    "contents": "".join(sentences),
                }
                for title, sentences in raw["context"]
            ],
            "original_type": raw["type"],
            "original_level": raw["level"],
        },
    }

    return formatted_data # 返回格式化后的数据

# Demo Usage 演示用法
if __name__ == '__main__':
    # 创建一个示例原始数据
    raw_data = {
        "_id": "123",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "supporting_facts": [("France", 0)],
        "context": [("France", ["Paris is the capital.", "France is in Europe."])],
        "type": "bridge",
        "level": "hard",
    }

    # 格式化原始数据
    formatted_data = format_raw_data(raw_data)
    print(f"格式化后的数据：{formatted_data}")
```

**描述:**  这个函数将原始HotpotQA数据格式化为更适合模型训练的格式。它从原始数据中提取问题、答案标签、支持事实和检索到的上下文，并将其组织成一个字典。它还推断问题的类型。如果无法找到支持事实，则该函数返回 `None`。

**如何使用:**  提供原始数据字典 (`raw`)，然后调用此函数。

总而言之，这段代码提供了一套完整的工具，用于下载、加载和预处理HotpotQA数据集。这些函数可以帮助你准备数据，以便用于训练和评估问答模型。 重要的是要理解每个函数的作用以及如何使用它们来处理数据。
