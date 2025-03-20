Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\two_wiki.py`

Okay, restarting. Let's break down the provided Python code, which focuses on data processing for a question answering dataset, likely HotpotQA or a similar task. I'll give explanations of each key part, followed by code snippets with descriptions in Chinese and usage demos.

**1. Downloading Raw Data ( `download_raw_data` function):**

```python
import os
import subprocess
import zipfile

default_name: str = "data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46"
dropbox_url: str = f"https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/{default_name}&e=1"

def download_raw_data(raw_dir: str) -> None:
    """
    Downloads the raw dataset from a Dropbox URL, renames it, and extracts its contents.
    从Dropbox URL下载原始数据集，重命名并解压缩其内容。
    """
    command = f'wget -q -P {raw_dir} {dropbox_url}' # 构建 wget 命令，-q 表示静默模式，-P 指定下载目录
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # 执行 wget 命令
    os.rename(f"{raw_dir}/{default_name}", f"{raw_dir}/2wiki.zip") # 重命名下载的文件
    with zipfile.ZipFile(f"{raw_dir}/2wiki.zip", "r") as zip_ref: # 打开 zip 文件
        zip_ref.extractall(raw_dir) # 解压缩到指定目录
    return

# Demo Usage 演示用法
if __name__ == '__main__':
    temp_dir = "temp_raw_data"
    os.makedirs(temp_dir, exist_ok=True)
    download_raw_data(temp_dir)
    print(f"Data downloaded and extracted to: {temp_dir}")
    # Clean up after demo (optional)
    # import shutil
    # shutil.rmtree(temp_dir)
```

**描述:**  此函数负责从给定的 Dropbox URL 下载原始数据集 ZIP 文件。  它使用 `wget` 命令（需要在系统上安装）静默下载文件，然后将其重命名为 "2wiki.zip"。 最后，它使用 `zipfile` 模块将 ZIP 文件的内容提取到指定的 `raw_dir` 目录中。

**如何使用:**  你需要提供一个目录路径 `raw_dir` 作为参数。 该函数将在该目录中下载、重命名和提取数据集。  确保 `wget` 已安装在你的系统上。

**2. Loading Raw Data ( `load_raw_data` function):**

```python
import json
import os
from typing import List

def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    """
    Loads the raw dataset from a JSON file.
    从JSON文件加载原始数据集。
    """
    raw_dir = os.path.join(dataset_dir, "raw") # 构建 raw 数据目录的路径
    os.makedirs(raw_dir, exist_ok=True) # 创建 raw 数据目录，如果不存在
    raw_filepath = os.path.join(raw_dir, f"{split}.json") # 构建 JSON 文件路径
    if not os.path.exists(raw_filepath): # 检查文件是否存在
        download_raw_data(raw_dir) # 如果文件不存在，则下载数据

    with open(raw_filepath, "r") as fin: # 打开 JSON 文件
        dataset = json.load(fin) # 加载 JSON 数据

    return dataset

# Demo Usage 演示用法
if __name__ == '__main__':
    temp_dir = "temp_dataset"
    os.makedirs(os.path.join(temp_dir, "raw"), exist_ok=True)

    # Create a dummy json file for testing
    dummy_data = [{"question": "What is the capital of France?", "answer": "Paris"}]
    with open(os.path.join(temp_dir, "raw", "train.json"), "w") as f:
        json.dump(dummy_data, f)

    loaded_data = load_raw_data(temp_dir, "train")
    print(f"Loaded data: {loaded_data}")

    # Clean up after demo (optional)
    import shutil
    shutil.rmtree(temp_dir)
```

**描述:** 此函数加载存储在 JSON 文件中的原始数据集。  它首先构建原始数据目录和 JSON 文件路径。 如果 JSON 文件不存在，它会调用 `download_raw_data` 函数来下载数据。  然后，它打开 JSON 文件并使用 `json.load` 加载数据。

**如何使用:** 你需要提供数据集目录 `dataset_dir` 和数据集分割（例如 "train"、"dev"）`split` 作为参数。 该函数将返回一个 Python 字典列表，其中每个字典表示数据集中的一个示例。

**3. Loading Title to QID Mapping ( `load_title2qid` function):**

```python
import os
from typing import Dict
import jsonlines

def load_title2qid(dataset_dir: str, split: str=None) -> Dict[str, str]:
    """
    Loads a mapping from Wikipedia article titles to question IDs.
    加载从维基百科文章标题到问题ID的映射。
    """
    raw_dir = os.path.join(dataset_dir, "raw") # 构建 raw 数据目录的路径

    title2qid: Dict[str, str] = {} # 初始化 title2qid 字典
    with jsonlines.open(f"{raw_dir}/id_aliases.json", "r") as fin: # 打开 jsonlines 文件
        for line in fin: # 遍历每一行
            qid, aliases = line["Q_id"], line["aliases"] # 获取问题 ID 和别名
            for alias in aliases: # 遍历别名
                title2qid[alias] = qid # 将别名映射到问题 ID

    return title2qid

# Demo Usage 演示用法
if __name__ == '__main__':
    temp_dir = "temp_dataset"
    os.makedirs(os.path.join(temp_dir, "raw"), exist_ok=True)

    # Create a dummy id_aliases.json file for testing
    dummy_data = [{"Q_id": "Q123", "aliases": ["France", "French Republic"]}]
    with jsonlines.open(os.path.join(temp_dir, "raw", "id_aliases.json"), "w") as f:
        for item in dummy_data:
            f.write(item)

    title_to_qid = load_title2qid(temp_dir)
    print(f"Title to QID mapping: {title_to_qid}")

    # Clean up after demo (optional)
    import shutil
    shutil.rmtree(temp_dir)
```

**描述:**  此函数加载一个从维基百科文章标题到相应问题 ID 的映射。 它从 `id_aliases.json` 文件（以 JSON Lines 格式存储）读取数据。  此文件包含问题 ID 和与该问题相关的维基百科文章标题的别名列表。

**如何使用:** 你需要提供数据集目录 `dataset_dir` 作为参数。  该函数返回一个字典，其中键是维基百科文章标题（或别名），值是对应的问题 ID。

**4. Formatting Raw Data ( `format_raw_data` function):**

```python
import uuid
from typing import Dict, List, Optional

from data_process.dataset_utils.hotpotqa import get_supporting_facts
from data_process.utils.question_type import infer_question_type

def format_raw_data(raw: dict) -> Optional[dict]:
    """
    Formats the raw data into a standardized format.
    将原始数据格式化为标准化格式。
    """
    # Step 1: extract supporting facts contents from retrieval contexts. 提取支持事实的内容
    supporting_facts = get_supporting_facts(raw["supporting_facts"], raw["context"])
    if supporting_facts is None: # 如果没有支持事实，则返回 None
        return None

    # Step 2: re-format to fit dataset protocol. 重新格式化以适应数据集协议
    answer_labels: List[str] = [raw["answer"]] # 获取答案标签
    qtype: str = infer_question_type(answer_labels) # 推断问题类型

    formatted_data = {
        "id": uuid.uuid4().hex, # 生成唯一 ID
        "question": raw["question"], # 问题
        "answer_labels": answer_labels, # 答案标签
        "question_type": qtype, # 问题类型
        "metadata": {
            "original_id": raw["_id"], # 原始 ID
            "original_type": raw["type"], # 原始类型
            "supporting_facts": supporting_facts, # 支持事实
            "retrieval_contexts": [ # 检索上下文
                {
                    "type": "wikipedia", # 类型
                    "title": title, # 标题
                    "contents": "".join(sentences), # 内容
                }
                for title, sentences in raw["context"] # 遍历上下文
            ],
            "reasoning_logics": [ # 推理逻辑
                {
                    "type": "wikidata", # 类型
                    "title": title, # 标题
                    "section": section, # 章节
                    "contents": content, # 内容
                }
                for title, section, content in raw["evidences"] # 遍历证据
            ],
        }
    }

    return formatted_data

# Demo Usage 演示用法
if __name__ == '__main__':
    # Create a dummy raw data sample
    dummy_raw_data = {
        "_id": "123",
        "type": "comparison",
        "question": "Which is bigger, France or Germany?",
        "answer": "Germany",
        "supporting_facts": [["France", 0], ["Germany", 0]],
        "context": [["France", ["France is a country.", "It is in Europe."]], ["Germany", ["Germany is also a country.", "It is next to France."]]],
        "evidences": [["France", "Geography", "France is big."], ["Germany", "Geography", "Germany is bigger."]]
    }

    formatted_data = format_raw_data(dummy_raw_data)
    print(f"Formatted data: {formatted_data}")

    # In a real scenario, you'd need to have implementations for
    # get_supporting_facts and infer_question_type to make this demo fully functional.
```

**描述:**  此函数采用原始数据集的单个样本（以字典形式）并将其格式化为标准化格式。  它提取支持事实，推断问题类型，并构建一个包含原始数据以及一些元数据的新字典。 这个函数依赖于 `get_supporting_facts` 和 `infer_question_type` 这两个外部函数，你需要确保它们可用。

**如何使用:**  你需要提供一个原始数据样本 `raw` 作为参数。 该函数返回一个格式化的数据样本（字典）。  如果无法提取支持事实，则返回 `None`。 此外，你需要定义或导入 `get_supporting_facts` 和 `infer_question_type` 函数，以便此函数能够正常工作。
