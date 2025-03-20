Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\popqa.py`

好的，现在我重新开始，提供更智能、更模块化的代码，并附带中文描述和示例：

**目标：优化PopQA数据处理流程**

原始代码主要完成了PopQA数据集的加载、格式化，并提取了实体标题到QID的映射。 我们可以从以下几个方面进行改进：

*   **更强的类型提示和验证：** 确保数据的一致性和避免运行时错误。
*   **更灵活的格式化：** 允许自定义格式化策略，以便适应不同的下游任务。
*   **更好的错误处理：** 增加对异常情况的处理，例如网络错误或数据损坏。
*   **更快的处理速度：** 利用多线程或异步处理来加速数据加载和转换。
*   **增加单元测试:** 确保代码的正确性。

**1. 改进的数据加载 (Data Loading):**

```python
from datasets import load_dataset
from typing import Optional, Dict
import json

def load_popqa_data(split: str) -> Optional[Dict]:
    """
    从Hugging Face Datasets加载PopQA数据集。

    Args:
        split: 数据集分割（例如 "train", "validation", "test"）。

    Returns:
        一个包含数据集的字典，或者在发生错误时返回 None。
    """
    try:
        dataset = load_dataset("akariasai/PopQA", split=split)
        return dataset
    except Exception as e:
        print(f"加载PopQA数据集失败: {e}")
        return None

# 示例：加载训练集
if __name__ == '__main__':
    train_data = load_popqa_data("train")
    if train_data:
        print(f"成功加载训练集，包含 {len(train_data)} 个样本。")
        print(f"第一个样本：{train_data[0]}") #打印第一个样本查看数据格式
    else:
        print("训练集加载失败。")

```

**描述:**  `load_popqa_data` 函数封装了数据集的加载过程。  使用 `try...except` 块来处理加载过程中可能出现的异常情况，例如网络连接问题。函数返回 `Dataset` 对象或 `None`。

*   **中文:** `load_popqa_data` 函数封装了数据集的加载过程，使用 `try...except` 块来处理加载过程中可能出现的异常情况，例如网络连接问题。 函数返回 `Dataset` 对象或 `None`，方便后续代码进行判断和处理。 如果加载成功，会打印数据集的大小和第一个样本，方便用户快速了解数据。

**2. 改进的数据格式化 (Data Formatting):**

```python
import uuid
from typing import List, Dict, Optional

from data_process.utils.question_type import infer_question_type  # 假设存在

def format_popqa_sample(raw: Dict, custom_fields: Optional[Dict] = None) -> Optional[Dict]:
    """
    格式化PopQA数据集中的单个样本。

    Args:
        raw: 原始样本字典。
        custom_fields: 额外的字段，用于自定义格式化过程。

    Returns:
        格式化后的样本字典，或者在缺少必要信息时返回 None。
    """
    if raw["subj"] is None:
        print("跳过样本，因为缺少 'subj' 字段。")
        return None

    try:
        answer_labels: List[str] = json.loads(raw["possible_answers"])
        qtype: str = infer_question_type(answer_labels)

        formatted_data = {
            "id": str(uuid.uuid4()),  # 使用字符串类型的 UUID
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

        # 添加自定义字段
        if custom_fields:
            formatted_data.update(custom_fields)

        return formatted_data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"格式化样本时发生错误: {e}")
        return None


# 示例：格式化单个样本
if __name__ == '__main__':
    # 假设 raw_sample 是从 load_popqa_data 获取的
    raw_sample = {
        "id": 123,
        "question": "Who is the wife of Barack Obama?",
        "possible_answers": '["Michelle Obama", "Hillary Clinton"]',
        "subj": "Barack Obama",
        "prop": "spouse",
        "obj": "Michelle Obama",
        "s_uri": "http://www.wikidata.org/entity/Q76",
    }

    formatted_sample = format_popqa_sample(raw_sample, custom_fields={"dataset": "PopQA"})
    if formatted_sample:
        print("格式化后的样本:", formatted_sample)
    else:
        print("样本格式化失败。")
```

**描述:** `format_popqa_sample` 函数现在包含以下改进：

*   **更强的类型提示：**  使用 `typing` 模块中的 `Dict`、`List` 和 `Optional` 来提供更清晰的类型信息。
*   **异常处理：** 使用 `try...except` 块来处理可能出现的异常，例如 `json.JSONDecodeError` 和 `KeyError`。
*   **自定义字段：**  允许用户通过 `custom_fields` 参数添加额外的字段，从而使格式化过程更加灵活。
*   **UUID：** 使用字符串类型的 UUID，更常见且易于处理。

*   **中文:** `format_popqa_sample` 函数现在可以处理样本格式化过程中可能出现的各种错误，例如JSON解析错误和键值错误。 可以通过 `custom_fields` 参数来添加额外的字段，使得格式化过程更加灵活，以适应不同的任务需求。 使用字符串类型的 UUID，更常见且易于处理。

**3. 改进的实体到QID映射提取 (Entity to QID Mapping Extraction):**

```python
from datasets import Dataset
from tqdm import tqdm
import json
from typing import Dict

def extract_title_to_qid(dataset: Dataset, dump_path: str) -> None:
    """
    从PopQA数据集提取实体标题到QID的映射，并保存到JSON文件中。

    Args:
        dataset: PopQA 数据集对象。
        dump_path: JSON文件的保存路径。
    """
    wikidata_title2qid: Dict[str, str] = {}
    try:
        for raw in tqdm(dataset, total=len(dataset), desc="Processing PopQA"):
            if raw["subj"] is not None:
                title = raw["subj"]
                qid = raw["s_uri"].split("/")[-1]
                wikidata_title2qid[title] = qid

        with open(dump_path, "w", encoding="utf-8") as fout:
            json.dump(wikidata_title2qid, fout, ensure_ascii=False, indent=4)  # 添加缩进以提高可读性

        print(f"成功提取实体标题到QID的映射并保存到: {dump_path}")

    except Exception as e:
        print(f"提取实体标题到QID的映射失败: {e}")

# 示例：提取映射
if __name__ == '__main__':
    train_data = load_popqa_data("train")
    if train_data:
        extract_title_to_qid(train_data, "title2qid.json")
    else:
        print("无法提取实体标题到QID的映射，因为训练集加载失败。")

```

**描述:**  `extract_title_to_qid` 函数：

*   接受数据集对象作为输入，而不是分割字符串。
*   包含完整的错误处理。
*   使用 `json.dump` 的 `indent` 参数使输出的 JSON 文件更具可读性。

*   **中文:**  `extract_title_to_qid` 函数接受数据集对象作为输入，而不是分割字符串，这样更灵活，方便使用不同来源的数据。 包含完整的错误处理，确保程序在出现问题时能够正常运行。 使用 `json.dump` 的 `indent` 参数使输出的 JSON 文件更具可读性，方便用户查看和调试。

**4.  单元测试 (Unit Testing):**

```python
import unittest
from unittest.mock import patch
from io import StringIO
import json
# 导入上述定义的函数
from data_process import (load_popqa_data, format_popqa_sample, extract_title_to_qid)

class TestDataProcessing(unittest.TestCase):
    # Mock一个可以成功加载的数据集
    def mock_load_popqa_data_success(self, split):
        # 创建模拟数据集，可以根据你的实际情况进行调整
        mock_dataset = [
            {
                "id": 123,
                "question": "Who is the wife of Barack Obama?",
                "possible_answers": '["Michelle Obama", "Hillary Clinton"]',
                "subj": "Barack Obama",
                "prop": "spouse",
                "obj": "Michelle Obama",
                "s_uri": "http://www.wikidata.org/entity/Q76",
            },
            {
                "id": 456,
                "question": "What is the capital of France?",
                "possible_answers": '["Paris", "London"]',
                "subj": "France",
                "prop": "capital",
                "obj": "Paris",
                "s_uri": "http://www.wikidata.org/entity/Q142",
            }
        ]
        return mock_dataset

    def test_load_popqa_data_success(self):
        # 使用mock加载数据集
        with patch('data_process.load_dataset', self.mock_load_popqa_data_success):
            dataset = load_popqa_data("train")
            self.assertIsNotNone(dataset)
            self.assertEqual(len(dataset), 2)

    def test_format_popqa_sample_success(self):
        raw_sample = {
            "id": 123,
            "question": "Who is the wife of Barack Obama?",
            "possible_answers": '["Michelle Obama", "Hillary Clinton"]',
            "subj": "Barack Obama",
            "prop": "spouse",
            "obj": "Michelle Obama",
            "s_uri": "http://www.wikidata.org/entity/Q76",
        }

        formatted_sample = format_popqa_sample(raw_sample, custom_fields={"dataset": "PopQA"})
        self.assertIsNotNone(formatted_sample)
        self.assertEqual(formatted_sample["question"], "Who is the wife of Barack Obama?")
        self.assertEqual(formatted_sample["metadata"]["original_id"], 123)
        self.assertEqual(formatted_sample["metadata"]["supporting_facts"][0]["title"], "Barack Obama")
        self.assertIn("dataset", formatted_sample)

    def test_format_popqa_sample_missing_subj(self):
        raw_sample = {
            "id": 123,
            "question": "Who is the wife of Barack Obama?",
            "possible_answers": '["Michelle Obama", "Hillary Clinton"]',
            "subj": None, #缺少subj
            "prop": "spouse",
            "obj": "Michelle Obama",
            "s_uri": "http://www.wikidata.org/entity/Q76",
        }

        formatted_sample = format_popqa_sample(raw_sample)
        self.assertIsNone(formatted_sample)  # 应该返回None

    def test_extract_title_to_qid_success(self):
       # 使用mock加载数据集
        with patch('data_process.load_dataset', self.mock_load_popqa_data_success):
            dataset = load_popqa_data("train")
            # 使用StringIO来模拟一个文件
            with patch('builtins.open', return_value=StringIO()) as mock_open:
                extract_title_to_qid(dataset, "mock_title2qid.json")

                # 验证open是否被调用，以及是否使用了正确的参数
                mock_open.assert_called_once_with("mock_title2qid.json", "w", encoding="utf-8")

                # 从StringIO获取写入的内容
                mock_file = mock_open.return_value
                mock_file.seek(0)  # 回到文件开始位置
                written_content = mock_file.read()

                # 验证写入的内容
                expected_data = {
                    "Barack Obama": "Q76",
                    "France": "Q142",
                }
                self.assertEqual(json.loads(written_content), expected_data) #验证写入的内容

# 请注意替换 'data_process' 为你的实际模块名
if __name__ == '__main__':
    unittest.main()
```

**描述:**
此代码添加了单元测试，以确保各个函数的正常运行。
- 使用mock模拟load_dataset函数，以便测试能够在不需要访问网络的情况下运行。
- 实现了load_popqa_data, format_popqa_sample和extract_title_to_qid的测试用例。
- 增加了对format_popqa_sample函数缺少subj字段时返回None的测试用例。
- 使用StringIO和patch模拟文件操作，以便测试extract_title_to_qid函数，并验证写入文件的内容。

**中文:**
此代码添加了单元测试，以确保各个函数的正常运行。
- 使用mock模拟load_dataset函数，以便测试能够在不需要访问网络的情况下运行，提高了测试效率。
- 实现了load_popqa_data, format_popqa_sample和extract_title_to_qid的测试用例，覆盖了主要的功能。
- 增加了对format_popqa_sample函数缺少subj字段时返回None的测试用例，保证了在数据异常情况下程序的健壮性。
- 使用StringIO和patch模拟文件操作，以便测试extract_title_to_qid函数，并验证写入文件的内容，保证了文件操作的正确性。

**如何运行单元测试:**

1.  确保你已经安装了 `unittest` 模块（通常Python自带）。

2.  将上述代码保存为 `test_data_process.py` (假设你的原始代码文件是 `data_process.py`).

3.  在命令行中，导航到包含 `test_data_process.py` 文件的目录，然后运行：

    ```bash
    python -m unittest test_data_process.py
    ```

**总结:**

通过这些改进，代码变得更健壮、更灵活和更易于维护。  单元测试提供了额外的保证，确保代码在各种情况下都能正常运行。
