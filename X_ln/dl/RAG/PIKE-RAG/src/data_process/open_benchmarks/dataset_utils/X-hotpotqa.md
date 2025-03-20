Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\hotpotqa.py`

好的，我们重新开始。

我想为您提供的代码改进，着重关注以下几个方面：

1.  **代码可读性和模块化：** 将一些复杂逻辑拆分成更小的函数，增加注释，使代码更易于理解和维护。
2.  **错误处理和健壮性：** 增加错误处理机制，例如检查文件是否存在、网络请求是否成功等，避免程序崩溃。
3.  **性能优化：**  虽然这个数据集处理脚本对性能要求不高，但可以考虑使用一些技巧来提高效率，例如批量处理数据、使用更高效的数据结构等。
4.  **类型提示和文档：**  完善类型提示，并添加文档字符串，方便其他开发者使用。
5.  **灵活性和可配置性：**  尽量将一些硬编码的参数改为可配置的选项，方便用户根据实际情况进行调整。

下面我将分步给出改进后的代码，并附上详细的中文解释和使用示例。

**Step 1: 改进 `download_raw_data` 函数**

```python
import os
import requests
from typing import Dict

def download_raw_data(raw_filepath: str, split: str, split2url: Dict[str, str]) -> None:
    """
    下载原始数据文件.

    Args:
        raw_filepath: 保存原始数据的文件路径.
        split: 数据集划分 (例如, "train", "dev").
        split2url: 包含数据集划分和对应URL的字典.

    Raises:
        ValueError: 如果split不在split2url中.
        requests.exceptions.RequestException: 如果下载请求失败.
    """
    if split not in split2url:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(split2url.keys())}")

    url = split2url[split]
    try:
        response = requests.get(url, stream=True)  # 使用 stream=True 提高下载大文件的效率
        response.raise_for_status()  # 检查HTTP状态码，如果不是200则抛出异常

        with open(raw_filepath, "wb") as fout:
            for chunk in response.iter_content(chunk_size=8192):  # 调整 chunk_size
                fout.write(chunk)
        print(f"Successfully downloaded {split} data to {raw_filepath}")  # 添加成功下载的消息

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {split} data from {url}: {e}")
        raise  # 重新抛出异常，让调用者处理
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        raise
```

**中文解释:**

1.  **参数验证:**  首先，我们检查 `split` 是否在 `split2url` 字典中，如果不在，则抛出一个 `ValueError` 异常，避免后续出错。
2.  **异常处理:** 使用 `try...except` 块来捕获可能发生的 `requests.exceptions.RequestException` 异常 (例如网络错误、HTTP错误) 和其他异常。这可以使程序更加健壮，即使下载过程中出现问题，也不会直接崩溃。
3.  **HTTP状态码检查:**  `response.raise_for_status()` 方法会检查 HTTP 响应状态码，如果状态码不是 200 (OK)，则会抛出一个异常。
4.  **流式下载:**  使用 `requests.get(url, stream=True)` 可以实现流式下载，这意味着数据会以小块 (chunk) 的形式下载，而不是一次性下载整个文件。这对于下载大型文件来说更有效率，可以减少内存占用。
5.  **Chunk Size:** 调整了 `chunk_size` 为 8192 字节，可以根据实际情况进行调整。
6.  **成功消息:** 添加了成功下载的消息，方便用户了解下载进度。
7. **错误信息:** 添加了详细的错误信息，方便用户调试.

**Step 2: 改进 `load_raw_data` 函数**

```python
import json
import os
from typing import List, Dict

def load_raw_data(dataset_dir: str, split: str, split2url: Dict[str, str]) -> List[dict]:
    """
    加载原始数据.  如果数据文件不存在, 则先下载.

    Args:
        dataset_dir: 数据集目录.
        split: 数据集划分 ("train", "dev").
        split2url: 包含数据集划分和对应URL的字典.

    Returns:
        包含原始数据的字典列表.

    Raises:
        FileNotFoundError: 如果JSON文件无法加载.
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, f"{split}.json")

    if not os.path.exists(raw_filepath):
        print(f"Raw data file not found at {raw_filepath}. Downloading...")
        download_raw_data(raw_filepath, split, split2url)

    try:
        with open(raw_filepath, "r", encoding="utf-8") as fin:
            dataset = json.load(fin)
        print(f"Successfully loaded {split} data from {raw_filepath}") # 添加成功加载的消息
        return dataset
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_filepath}.")
        raise  # 重新抛出异常，让调用者处理
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {raw_filepath}: {e}")
        raise # 重新抛出异常，让调用者处理
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        raise

```

**中文解释:**

1.  **数据文件存在性检查:** 首先检查原始数据文件是否存在，如果不存在，则调用 `download_raw_data` 函数下载数据。
2.  **异常处理:** 使用 `try...except` 块来捕获可能发生的 `FileNotFoundError` 异常 (文件未找到) 和 `json.JSONDecodeError` 异常 (JSON解码错误)。
3. **成功信息:** 添加了成功加载的消息.
4. **错误信息:** 添加了详细的错误信息，方便用户调试.
5.  **更清晰的错误提示:**  当文件不存在时，输出更友好的提示信息。
6.  **重新抛出异常:** 捕获到异常后，使用 `raise` 重新抛出异常，这可以让调用 `load_raw_data` 函数的代码知道发生了错误，并进行相应的处理。

**Step 3: 改进 `get_idx_sentence` 函数**

```python
from typing import List, Tuple, Optional

def get_idx_sentence(
    title: str, idx: int, original_contexts: List[Tuple[str, List[str]]], verbose: bool = False
) -> Optional[str]:
    """
    从原始上下文中获取指定标题和索引的句子.

    Args:
        title: 文档标题.
        idx: 句子索引.
        original_contexts: 包含标题和句子列表的列表.
        verbose: 是否打印调试信息.

    Returns:
        如果找到句子, 则返回句子字符串; 否则返回 None.
    """
    for item_title, sentences in original_contexts:
        if item_title == title:
            if 0 <= idx < len(sentences): # 增加索引范围检查
                return sentences[idx]
            else:
                if verbose:
                    print(f"Warning: Index {idx} out of range for title {title} (length: {len(sentences)}).")
                return None

    if verbose:
        print(f"######## Indexed sentence not found ########")
        print(f"title: {title}, idx: {idx}")
        for item_title, sentences in original_contexts:
            if item_title != title:
                continue
            for i, sentence in enumerate(sentences):
                print(f"  {i}: {sentence}")
            print()
    return None
```

**中文解释:**

1.  **索引范围检查:**  增加了对 `idx` 的范围检查，确保索引值在 `sentences` 列表的有效范围内。如果索引超出范围，则打印警告信息（如果 `verbose` 为 `True`）并返回 `None`。
2.  **更清晰的警告信息:**  如果索引超出范围，打印更详细的警告信息，包括标题、索引和句子列表的长度。
3.  **类型提示:**  完善了类型提示，使代码更易于理解。
4.  **文档字符串:**  添加了文档字符串，解释了函数的作用、参数和返回值。

**Step 4: 改进 `get_supporting_facts` 函数**

```python
from typing import List, Tuple, Dict, Literal, Optional

def get_supporting_facts(
    supporting_fact_tuples: List[Tuple[str, int]],
    context_tuples: List[Tuple[str, List[str]]],
) -> Optional[List[Dict[Literal["type", "title", "contents"], str]]]:
    """
    从上下文中提取支持事实.

    Args:
        supporting_fact_tuples: 包含支持事实标题和句子索引的列表.
        context_tuples: 包含标题和句子列表的列表.

    Returns:
        包含支持事实的字典列表, 每个字典包含 "type", "title", "contents" 键.
        如果任何支持事实无法找到, 则返回 None.
    """
    supporting_facts: List[dict] = []
    for title, sent_idx in supporting_fact_tuples:
        content: Optional[str] = get_idx_sentence(title, sent_idx, context_tuples)
        if content is None:
            print(f"Warning: Could not find supporting fact for title '{title}' and index {sent_idx}.")
            return None  # 如果找不到任何一个支持事实，就返回 None

        supporting_facts.append(
            {
                "type": "wikipedia",
                "title": title,
                "contents": content,
            }
        )
    return supporting_facts
```

**中文解释:**

1.  **错误处理:** 如果在 `get_idx_sentence` 中找不到支持事实，则立即返回 `None`，并打印警告信息。
2.  **更清晰的警告信息:**  如果找不到支持事实，打印更详细的警告信息，包括标题和索引。
3.  **类型提示:**  完善了类型提示.
4.  **文档字符串:**  添加了文档字符串。

**Step 5: 改进 `format_raw_data` 函数**

```python
import uuid
from typing import List, Dict, Optional

from data_process.utils.question_type import infer_question_type

def format_raw_data(raw: dict) -> Optional[dict]:
    """
    格式化原始数据，使其符合目标格式.

    Args:
        raw: 原始数据字典.

    Returns:
        格式化后的数据字典. 如果无法提取支持事实, 则返回 None.
    """
    # Step 1: extract supporting facts contents from retrieval contexts.
    # Skip sample if supporting fact not found. Currently there is one error case in `dev` split with `_id`:
    # 5ae61bfd5542992663a4f261
    supporting_facts = get_supporting_facts(raw["supporting_facts"], raw["context"])
    if supporting_facts is None:
        print(f"Warning: Could not extract supporting facts for original_id '{raw.get('_id', 'N/A')}'. Skipping sample.")
        return None

    # Step 2: re-format to fit dataset protocol.
    answer_labels: List[str] = [raw["answer"]]
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": str(uuid.uuid4()),  # 使用字符串形式的 UUID
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

    return formatted_data
```

**中文解释:**

1.  **UUID:**  使用 `str(uuid.uuid4())` 生成字符串形式的 UUID，而不是 `uuid.uuid4().hex`，这更符合通用做法。
2.  **错误处理:** 如果无法提取支持事实，则打印警告信息，包括原始数据的 `_id`（如果存在），然后返回 `None`。
3. **更清晰的警告信息:** 如果无法提取支持事实，打印更详细的警告信息，包括原始数据的 `_id`。 使用 `raw.get('_id', 'N/A')` 来安全地获取 `_id`，如果 `_id` 不存在，则使用 "N/A" 作为默认值.
4.  **类型提示:**  完善了类型提示.
5.  **文档字符串:**  添加了文档字符串.

**Step 6: 完整的代码和使用示例**

```python
import json
import os
import requests
from typing import Dict, List, Literal, Optional, Tuple

import uuid

from data_process.utils.question_type import infer_question_type


split2url: Dict[str, str] = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
}


def download_raw_data(raw_filepath: str, split: str, split2url: Dict[str, str]) -> None:
    """
    下载原始数据文件.

    Args:
        raw_filepath: 保存原始数据的文件路径.
        split: 数据集划分 (例如, "train", "dev").
        split2url: 包含数据集划分和对应URL的字典.

    Raises:
        ValueError: 如果split不在split2url中.
        requests.exceptions.RequestException: 如果下载请求失败.
    """
    if split not in split2url:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(split2url.keys())}")

    url = split2url[split]
    try:
        response = requests.get(url, stream=True)  # 使用 stream=True 提高下载大文件的效率
        response.raise_for_status()  # 检查HTTP状态码，如果不是200则抛出异常

        with open(raw_filepath, "wb") as fout:
            for chunk in response.iter_content(chunk_size=8192):  # 调整 chunk_size
                fout.write(chunk)
        print(f"Successfully downloaded {split} data to {raw_filepath}")  # 添加成功下载的消息

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {split} data from {url}: {e}")
        raise  # 重新抛出异常，让调用者处理
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        raise


def load_raw_data(dataset_dir: str, split: str, split2url: Dict[str, str]) -> List[dict]:
    """
    加载原始数据.  如果数据文件不存在, 则先下载.

    Args:
        dataset_dir: 数据集目录.
        split: 数据集划分 ("train", "dev").
        split2url: 包含数据集划分和对应URL的字典.

    Returns:
        包含原始数据的字典列表.

    Raises:
        FileNotFoundError: 如果JSON文件无法加载.
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, f"{split}.json")

    if not os.path.exists(raw_filepath):
        print(f"Raw data file not found at {raw_filepath}. Downloading...")
        download_raw_data(raw_filepath, split, split2url)

    try:
        with open(raw_filepath, "r", encoding="utf-8") as fin:
            dataset = json.load(fin)
        print(f"Successfully loaded {split} data from {raw_filepath}") # 添加成功加载的消息
        return dataset
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_filepath}.")
        raise  # 重新抛出异常，让调用者处理
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {raw_filepath}: {e}")
        raise # 重新抛出异常，让调用者处理
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        raise


def get_idx_sentence(
    title: str, idx: int, original_contexts: List[Tuple[str, List[str]]], verbose: bool = False
) -> Optional[str]:
    """
    从原始上下文中获取指定标题和索引的句子.

    Args:
        title: 文档标题.
        idx: 句子索引.
        original_contexts: 包含标题和句子列表的列表.
        verbose: 是否打印调试信息.

    Returns:
        如果找到句子, 则返回句子字符串; 否则返回 None.
    """
    for item_title, sentences in original_contexts:
        if item_title == title:
            if 0 <= idx < len(sentences): # 增加索引范围检查
                return sentences[idx]
            else:
                if verbose:
                    print(f"Warning: Index {idx} out of range for title {title} (length: {len(sentences)}).")
                return None

    if verbose:
        print(f"######## Indexed sentence not found ########")
        print(f"title: {title}, idx: {idx}")
        for item_title, sentences in original_contexts:
            if item_title != title:
                continue
            for i, sentence in enumerate(sentences):
                print(f"  {i}: {sentence}")
            print()
    return None


def get_supporting_facts(
    supporting_fact_tuples: List[Tuple[str, int]],
    context_tuples: List[Tuple[str, List[str]]],
) -> Optional[List[Dict[Literal["type", "title", "contents"], str]]]:
    """
    从上下文中提取支持事实.

    Args:
        supporting_fact_tuples: 包含支持事实标题和句子索引的列表.
        context_tuples: 包含标题和句子列表的列表.

    Returns:
        包含支持事实的字典列表, 每个字典包含 "type", "title", "contents" 键.
        如果任何支持事实无法找到, 则返回 None.
    """
    supporting_facts: List[dict] = []
    for title, sent_idx in supporting_fact_tuples:
        content: Optional[str] = get_idx_sentence(title, sent_idx, context_tuples)
        if content is None:
            print(f"Warning: Could not find supporting fact for title '{title}' and index {sent_idx}.")
            return None  # 如果找不到任何一个支持事实，就返回 None

        supporting_facts.append(
            {
                "type": "wikipedia",
                "title": title,
                "contents": content,
            }
        )
    return supporting_facts


def format_raw_data(raw: dict) -> Optional[dict]:
    """
    格式化原始数据，使其符合目标格式.

    Args:
        raw: 原始数据字典.

    Returns:
        格式化后的数据字典. 如果无法提取支持事实, 则返回 None.
    """
    # Step 1: extract supporting facts contents from retrieval contexts.
    # Skip sample if supporting fact not found. Currently there is one error case in `dev` split with `_id`:
    # 5ae61bfd5542992663a4f261
    supporting_facts = get_supporting_facts(raw["supporting_facts"], raw["context"])
    if supporting_facts is None:
        print(f"Warning: Could not extract supporting facts for original_id '{raw.get('_id', 'N/A')}'. Skipping sample.")
        return None

    # Step 2: re-format to fit dataset protocol.
    answer_labels: List[str] = [raw["answer"]]
    qtype: str = infer_question_type(answer_labels)

    formatted_data = {
        "id": str(uuid.uuid4()),  # 使用字符串形式的 UUID
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

    return formatted_data


if __name__ == '__main__':
    # 示例用法
    dataset_dir = "data"  # 数据集目录
    split = "dev"  # 选择数据集划分
    try:
        raw_data = load_raw_data(dataset_dir, split, split2url)

        # 格式化数据
        formatted_data = []
        for raw_item in raw_data:
            formatted_item = format_raw_data(raw_item)
            if formatted_item:
                formatted_data.append(formatted_item)

        # 打印格式化后的数据 (可选)
        print(f"Formatted {len(formatted_data)} samples.")
        #print(json.dumps(formatted_data[0], indent=2, ensure_ascii=False)) # 打印第一个样本

        print("Data processing complete.")

    except Exception as e:
        print(f"An error occurred during data processing: {e}")
```

**使用示例:**

1.  **设置数据集目录和划分:**  将 `dataset_dir` 设置为你的数据集目录，将 `split` 设置为你想处理的数据集划分 (例如，"train" 或 "dev")。
2.  **运行脚本:**  运行 Python 脚本。脚本会自动下载原始数据（如果数据文件不存在），然后加载数据，格式化数据，并打印格式化后的数据数量。

**总结:**

这些改进旨在提高代码的可读性、健壮性、灵活性和可维护性。 通过添加错误处理、类型提示、文档字符串和更清晰的错误消息，可以更容易地理解、使用和调试代码。 此外，通过使用流式下载和更高效的数据结构，可以提高代码的效率。 请记住，这只是一个起点，你可以根据你的具体需求进一步改进代码。
