Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\popqa.py`

好的，现在重新开始，针对你提供的代码进行分析和注释，并附带中文解释和使用示例：

**1. 导入必要的库:**

```python
import json
from typing import Dict, List, Optional
import uuid
from datasets import Dataset, load_dataset
from tqdm import tqdm
from data_process.utils.question_type import infer_question_type
```

*   `json`: 用于处理 JSON 格式的数据，例如读取和写入 JSON 文件。
*   `typing`:  用于类型提示，提高代码的可读性和可维护性。`Dict`、`List`、`Optional` 分别表示字典、列表和可选类型。
*   `uuid`: 用于生成唯一的 ID，例如在创建新的数据样本时。
*   `datasets`:  Hugging Face `datasets` 库，用于加载和处理各种数据集。
*   `tqdm`:  用于显示循环的进度条，提供更友好的用户体验。
*   `data_process.utils.question_type`:  一个自定义模块，用于推断问题的类型。（假设这个模块存在，实际应用中可能需要根据你的项目结构进行调整）

**2. `load_raw_data` 函数:**

```python
def load_raw_data(dataset_dir: str, split: str) -> Dataset:
    dataset: Dataset = load_dataset("akariasai/PopQA", split=split)
    return dataset
```

*   **功能:**  加载原始的 PopQA 数据集。
*   **参数:**
    *   `dataset_dir`:  数据集的目录 (在这个函数中未使用，但保留参数可能是为了将来扩展)。
    *   `split`:  数据集的划分 (例如: "train", "validation", "test")。
*   **返回值:**  Hugging Face `Dataset` 对象，包含指定划分的数据。
*   **实现:**  使用 `load_dataset` 函数从 Hugging Face Hub 加载 "akariasai/PopQA" 数据集。

```python
# 示例用法：
# 假设你想加载 PopQA 数据集的 "train" 划分。
# 你可以这样调用函数：
# train_dataset = load_raw_data("", "train")
# print(train_dataset)  # 打印数据集信息
```

**3. `format_raw_data` 函数:**

```python
def format_raw_data(raw: dict) -> Optional[dict]:
    # Skip raw if subject item not exist.
    if raw["subj"] is None:
        return None

    # Re-format to fit dataset protocol.
    answer_labels: List[str] = json.loads(raw["possible_answers"])
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["id"],
            "supporting_facts": [
                {
                    "type": "wikidata",
                    "title": raw["subj"],
                    "section": raw["prop"],
                    "contents": raw["obj"],
                }
            ],
        }
    }

    return formatted_data
```

*   **功能:**  将原始的 PopQA 数据样本格式化为更易于使用的格式。
*   **参数:**  `raw`: 原始的 PopQA 数据样本 (一个字典)。
*   **返回值:**  格式化后的数据样本 (一个字典)，如果 `raw["subj"]` 为 `None`，则返回 `None`。
*   **实现:**
    1.  **检查 `raw["subj"]` 是否为 `None`:** 如果是，则跳过该样本，返回 `None`。
    2.  **解析 `answer_labels`:** 将 `raw["possible_answers"]` 字段 (一个 JSON 字符串) 解析为字符串列表。
    3.  **推断问题类型:** 调用 `infer_question_type` 函数，根据 `answer_labels` 推断问题类型。
    4.  **创建格式化后的数据字典:**
        *   `id`:  使用 `uuid.uuid4().hex` 生成唯一的 ID。
        *   `question`:  直接从 `raw["question"]` 复制问题。
        *   `answer_labels`:  使用解析后的 `answer_labels`。
        *   `question_type`:  使用推断出的 `qtype`。
        *   `metadata`:  包含原始 ID 和支持事实的元数据。支持事实包括 Wikidata 实体 (`raw["subj"]`) 的标题、属性 (`raw["prop"]`) 和内容 (`raw["obj"]`)。

```python
# 示例用法：
# 假设你有一个原始的 PopQA 数据样本 raw_data。
# formatted_data = format_raw_data(raw_data)
# if formatted_data:
#     print(formatted_data)  # 打印格式化后的数据
# else:
#     print("Skipped sample due to missing subject.")
```

**4. `extract_title2qid` 函数:**

```python
def extract_title2qid(split: str, dump_path: str) -> None:
    raw_data = load_raw_data("", split)

    wikidata_title2qid: Dict[str, str] = {}
    for raw in tqdm(raw_data, total=len(raw_data), desc=f"Processing PopQA/{split}"):
        title = raw["subj"]
        qid = raw["s_uri"].split("/")[-1]
        wikidata_title2qid[title] = qid

    with open(dump_path, "w", encoding="utf-8") as fout:
        json.dump(wikidata_title2qid, fout, ensure_ascii=False)

    return
```

*   **功能:**  从 PopQA 数据集中提取 Wikidata 实体的标题和 QID 的映射关系，并将映射关系保存到 JSON 文件中。
*   **参数:**
    *   `split`:  数据集的划分 (例如: "train", "validation", "test")。
    *   `dump_path`:  用于保存映射关系的 JSON 文件的路径。
*   **返回值:**  无 (`None`)。
*   **实现:**
    1.  **加载原始数据:** 使用 `load_raw_data` 函数加载指定划分的数据。
    2.  **创建空字典 `wikidata_title2qid`:** 用于存储标题和 QID 的映射关系。
    3.  **遍历数据集:**  使用 `tqdm` 显示进度条，遍历数据集中的每个样本。
    4.  **提取标题和 QID:**
        *   从 `raw["subj"]` 中获取 Wikidata 实体的标题。
        *   从 `raw["s_uri"]` 中提取 QID (假设 QID 是 URI 的最后一个部分，例如 "http://www.wikidata.org/entity/Q123" 中的 "Q123")。
    5.  **添加到字典:** 将标题和 QID 添加到 `wikidata_title2qid` 字典中。
    6.  **保存到 JSON 文件:** 使用 `json.dump` 函数将 `wikidata_title2qid` 字典保存到指定的 JSON 文件中。`ensure_ascii=False` 确保可以正确保存非 ASCII 字符。

```python
# 示例用法：
# 假设你想从 PopQA 数据集的 "train" 划分中提取标题和 QID 的映射关系，并将结果保存到 "title2qid.json" 文件中。
# extract_title2qid("train", "title2qid.json")
```

**5. `load_title2qid` 函数:**

```python
def load_title2qid(dataset_dir: str, split: str) -> Dict[str, str]:
    dataset = load_raw_data("", split)

    title2qid = {}
    for raw in dataset:
        if raw["subj"] is not None:
            title = raw["subj"]
            qid = raw["s_uri"].split("/")[-1]
            title2qid[title] = qid

    return title2qid
```

*   **功能:**  加载 PopQA 数据集，并返回一个包含 Wikidata 实体的标题和 QID 映射的字典。  与 `extract_title2qid` 不同，此函数直接返回字典，不保存到文件。
*   **参数:**
    *   `dataset_dir`: 数据集目录 (在这个函数中未使用，但保留可能是为了将来扩展)。
    *   `split`: 数据集划分 (例如 "train", "validation", "test")。
*   **返回值:** 一个字典，其中键是 Wikidata 实体的标题，值是对应的 QID。
*   **实现:**
    1. **加载原始数据:** 使用 `load_raw_data` 函数加载指定划分的数据。
    2. **创建空字典 `title2qid`:** 用于存储标题和 QID 的映射关系。
    3. **遍历数据集:** 遍历数据集中的每个样本。
    4. **检查 `raw["subj"]` 是否为 `None`:**  如果为 `None`，则跳过该样本。
    5. **提取标题和 QID:**
        * 从 `raw["subj"]` 中获取 Wikidata 实体的标题。
        * 从 `raw["s_uri"]` 中提取 QID (假设 QID 是 URI 的最后一个部分)。
    6. **添加到字典:**  将标题和 QID 添加到 `title2qid` 字典中。
    7. **返回字典:** 返回 `title2qid` 字典。

```python
# 示例用法：
# 假设你想加载 PopQA 数据集的 "train" 划分，并获取标题和 QID 的映射关系。
# title2qid_map = load_title2qid("", "train")
# print(title2qid_map)  # 打印标题和 QID 的映射关系
```

**总结:**

这段代码主要用于加载、格式化和处理 PopQA 数据集。 它提供了以下功能：

*   加载原始数据。
*   将原始数据格式化为更易于使用的格式。
*   提取 Wikidata 实体的标题和 QID 的映射关系，并保存到 JSON 文件中或直接返回。

这些函数可以帮助你更轻松地访问和使用 PopQA 数据集，用于知识图谱问答等任务。
