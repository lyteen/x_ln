Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\triviaqa.py`

好的，让我们重新开始，并提供更智能的、模块化的代码，并附带中文描述和示例。

**目标:** 改进原始代码，使其更易于阅读、维护和扩展，并提供更清晰的错误处理和数据验证。

**1. 更健壮的`load_raw_data`函数:**

```python
from datasets import Dataset, load_dataset
from typing import Optional

def load_raw_data(dataset_name: str = "mandarjoshi/trivia_qa", subset: str = "rc", split: str = "train") -> Optional[Dataset]:
    """
    加载原始数据集.

    Args:
        dataset_name (str): 数据集名称. 默认为 "mandarjoshi/trivia_qa".
        subset (str): 数据集子集. 默认为 "rc".
        split (str): 数据集划分 (例如 "train", "validation", "test"). 默认为 "train".

    Returns:
        Dataset: 加载的数据集. 如果加载失败，则返回 None.
    """
    try:
        dataset: Dataset = load_dataset(dataset_name, subset, split=split)
        print(f"成功加载数据集 '{dataset_name}', 子集 '{subset}', 划分 '{split}'. 数据集大小: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None

# 示例用法
if __name__ == '__main__':
    train_dataset = load_raw_data(split="train")
    if train_dataset:
        print(f"数据集的第一个样本: {train_dataset[0]}")
```

**描述:**

*   **更加灵活:**  现在可以自定义`dataset_name`、`subset`和`split`。
*   **错误处理:** 使用`try...except`块捕获`load_dataset`可能引发的异常，例如网络问题或数据集不存在。 如果加载失败，则返回`None`。
*   **日志记录:**  打印成功或失败的消息，方便调试。
*   **类型提示:**  使用类型提示，提高代码可读性。

**中文描述:**  这个函数负责从Hugging Face Hub加载原始的TriviaQA数据集。  它接受数据集的名称、子集和划分作为参数。  它使用 `try...except` 块来处理加载过程中可能出现的错误，并返回加载的数据集或者 `None` 如果加载失败。  最后，它会打印一条消息，指示数据集是否已成功加载。

---

**2.  用于提取 Bing 搜索结果的辅助函数:**

```python
from typing import List, Dict

def extract_bing_search_results(raw: dict) -> List[Dict]:
    """
    从原始数据中提取 Bing 搜索结果.

    Args:
        raw (dict): 包含原始数据的字典.

    Returns:
        List[Dict]: 包含格式化后的 Bing 搜索结果的列表.
    """
    bing_search_results = []
    try:
        for title, url, description, contents, rank in zip(
            raw["search_results"]["title"],
            raw["search_results"]["url"],
            raw["search_results"]["description"],
            raw["search_results"]["search_context"],
            raw["search_results"]["rank"],
        ):
            bing_search_results.append(
                {
                    "type": "BingSearch",
                    "title": title,
                    "url": url,
                    "description": description,
                    "contents": contents,
                    "rank": rank,
                }
            )
    except KeyError as e:
        print(f"提取 Bing 搜索结果时出错: 缺少键 '{e}'")
        return []
    except TypeError as e:
        print(f"提取 Bing 搜索结果时出错: 数据类型不匹配: {e}")
        return []
    return bing_search_results
```

**描述:**

*   **明确的参数和返回值:**  接受 `raw` 字典作为输入，并返回格式化的 Bing 搜索结果列表。
*   **错误处理:**  添加了`try...except`块来捕获`KeyError`（如果原始数据中缺少某些键）和`TypeError`（如果数据的类型不正确），并打印错误消息。 如果发生错误，则返回一个空列表。
*   **更易读:**  循环逻辑保持不变，但现在封装在一个单独的函数中。

**中文描述:** 这个函数负责从原始数据字典中提取 Bing 搜索结果并将它们格式化成一个列表。 它使用 `try...except` 块来处理可能出现的 `KeyError`（缺少键）和 `TypeError`（类型错误），并返回一个格式化的结果列表，或者一个空列表如果发生错误。

---

**3. 改进的`format_raw_data`函数:**

```python
import uuid
from typing import List, Optional, Dict
from data_process.utils.question_type import infer_question_type

def format_raw_data(raw: dict) -> Optional[dict]:
    """
    格式化原始数据，使其符合数据集协议.

    Args:
        raw (dict): 原始数据字典.

    Returns:
        Optional[dict]: 格式化后的数据字典. 如果格式化失败，则返回 None.
    """
    try:
        # 1. 提取 Bing 搜索结果
        bing_search_results = extract_bing_search_results(raw)

        # 2. 提取 Wikipedia 标题
        wikipedia_titles = raw.get("entity_pages", {}).get("title", [])  # 使用 .get() 避免 KeyError
        wikipedia_contexts = [{"type": "wikipedia", "title": title} for title in wikipedia_titles]

        # 3. 获取答案标签和问题类型
        answer_labels: List[str] = raw["answer"]["aliases"]
        qtype: str = infer_question_type(answer_labels)

        # 4. 构建格式化后的数据
        formatted_data = {
            "id": uuid.uuid4().hex,
            "question": raw["question"],
            "answer_labels": answer_labels,
            "question_type": qtype,
            "metadata": {
                "original_id": raw["question_id"],
                "retrieval_contexts": wikipedia_contexts + bing_search_results,
            },
        }
        return formatted_data

    except KeyError as e:
        print(f"格式化数据时出错: 缺少键 '{e}'")
        return None
    except Exception as e:
        print(f"格式化数据时发生意外错误: {e}")  # 捕获所有其他异常
        return None

# 示例用法 (需要一个 'raw' 字典)
if __name__ == '__main__':
    # 模拟一个 'raw' 数据字典 (需要替换为真实数据)
    raw_data = {
        "question": "...",
        "answer": {"aliases": ["..."]},
        "question_id": "...",
        "entity_pages": {"title": ["..."]},
        "search_results": {
            "title": ["...", "..."],
            "url": ["...", "..."],
            "description": ["...", "..."],
            "search_context": ["...", "..."],
            "rank": [1, 2],
        },
    }
    formatted_data = format_raw_data(raw_data)

    if formatted_data:
        print(f"格式化后的数据: {formatted_data}")
    else:
        print("数据格式化失败.")
```

**描述:**

*   **模块化:**  调用`extract_bing_search_results`函数，将Bing搜索结果的提取逻辑分离出来。
*   **更健壮:**  使用`raw.get("entity_pages", {}).get("title", [])`来安全地访问嵌套的字典键，避免`KeyError`。 如果`entity_pages`或`title`不存在，则返回一个空列表。
*   **清晰的错误处理:** 使用`try...except`块捕获`KeyError`和其他可能的异常，例如`TypeError`，并返回`None`如果格式化失败。
*   **详细的错误消息:**  打印更具体的错误消息，方便调试。
*   **类型提示:** 使用类型提示，提高代码可读性。
*  **问题类型推断依赖前置:** 明确 `infer_question_type` 的位置。
*  **避免直接访问键，使用 `get` 方法:** 避免因键不存在而引发的 `KeyError`。

**中文描述:**  这个函数负责将原始数据格式化为符合数据集协议的格式。 它首先调用 `extract_bing_search_results` 函数来提取 Bing 搜索结果。  然后，它使用 `raw.get()` 方法安全地访问 `entity_pages` 和 `title` 键，避免 `KeyError`。 接下来，它提取答案标签和使用 `infer_question_type` 函数推断问题类型。 最后，它构建一个格式化后的数据字典并返回它，或者返回 `None` 如果格式化失败。

---

**总结:**

这些改进使代码更易于阅读、调试和维护。  错误处理更加健壮，并且代码现在可以更好地处理各种输入数据。  模块化设计允许您轻松地修改或扩展各个组件。 示例用法演示了如何使用这些函数，您需要替换示例数据以进行真实的测试。
