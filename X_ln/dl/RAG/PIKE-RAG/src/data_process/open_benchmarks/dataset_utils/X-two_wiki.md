Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\two_wiki.py`

好的，我们重新开始。这次我会更注重代码的可读性、模块化，并且提供更清晰的中文解释和示例。

**1. 改进的 `download_raw_data` 函数 (下载原始数据):**

```python
import os
import subprocess
import zipfile

def download_raw_data(raw_dir: str, dropbox_url: str, default_name: str) -> None:
    """
    下载并解压原始数据。

    Args:
        raw_dir (str): 保存原始数据的目录。
        dropbox_url (str): Dropbox URL 地址.
        default_name (str): 下载的文件名.

    Returns:
        None
    """
    os.makedirs(raw_dir, exist_ok=True)  # 确保目录存在

    zip_filename = "2wiki.zip"
    zip_filepath = os.path.join(raw_dir, zip_filename)

    # 下载文件
    download_command = f'wget -q -P {raw_dir} {dropbox_url}'
    subprocess.run(download_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.rename(f"{raw_dir}/{default_name}", zip_filepath)

    # 解压文件
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
    except zipfile.BadZipFile:
        print(f"警告：下载的 ZIP 文件损坏，请检查 URL 或重新下载。文件路径：{zip_filepath}")
        os.remove(zip_filepath)  # 删除可能损坏的文件
        return

    print(f"原始数据已下载并解压到：{raw_dir}")

# 演示用法
if __name__ == '__main__':
    raw_dir = "./raw_data"  # 本地目录
    default_name = "data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46" # default name
    dropbox_url= f"https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/{default_name}&e=1"

    download_raw_data(raw_dir, dropbox_url, default_name)
```

**描述:**

这个函数负责从 Dropbox 下载原始数据，并解压缩到指定的目录。

**改进:**

*   **错误处理:** 增加了 `try...except` 块来处理 ZIP 文件损坏的情况，并给出友好的提示。
*   **清晰的目录处理:** 使用 `os.makedirs(raw_dir, exist_ok=True)` 确保目录存在，如果存在也不会报错。
*   **模块化:**  将下载和解压操作封装在一个函数中，使代码更易于维护和重用。
*   **可配置的参数:** 接受 `raw_dir`, `dropbox_url` 和 `default_name` 作为参数，使其更灵活。
*   **中文注释:**  添加了中文注释，方便理解代码的含义。

**中文解释:**

这段代码首先检查指定的目录是否存在，如果不存在就创建它。然后，它使用 `wget` 命令从 Dropbox 下载文件，并将文件名修改为 `2wiki.zip`。  接下来，它尝试解压 ZIP 文件。 如果解压过程中发生错误（比如文件损坏），会捕获 `BadZipFile` 异常，打印警告信息，并删除可能损坏的文件。最后打印成功下载解压的文件目录.

**2. 改进的 `load_raw_data` 函数 (加载原始数据):**

```python
import json
import os
from typing import List

def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    """
    从 JSON 文件加载原始数据。

    Args:
        dataset_dir (str): 数据集根目录。
        split (str): 数据集分割（例如，"train"，"dev"）。

    Returns:
        List[dict]: 包含原始数据的字典列表。
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    raw_filepath = os.path.join(raw_dir, f"{split}.json")

    if not os.path.exists(raw_filepath):
        print(f"错误：找不到原始数据文件：{raw_filepath}。请先下载数据。")
        return [] # Or raise an Exception: raise FileNotFoundError(f"...")

    try:
        with open(raw_filepath, "r", encoding="utf-8") as fin:
            dataset = json.load(fin)
    except FileNotFoundError:
        print(f"错误：找不到原始数据文件：{raw_filepath}")
        return []
    except json.JSONDecodeError:
        print(f"错误：JSON 文件解码失败：{raw_filepath}。文件可能损坏。")
        return []
    except Exception as e:
        print(f"加载原始数据时发生未知错误: {e}")
        return []

    print(f"已加载 {split} 数据集，包含 {len(dataset)} 个样本。")
    return dataset

# 演示用法
if __name__ == '__main__':
    dataset_dir = "./data" # 数据集目录
    split = "train"
    dataset = load_raw_data(dataset_dir, split)

    if dataset:
        print(f"第一个样本的键：{dataset[0].keys()}")
```

**描述:**

这个函数负责从指定的 JSON 文件加载原始数据。

**改进:**

*   **更健壮的错误处理:**  添加了更多的 `try...except` 块来处理 `FileNotFoundError` (文件不存在), `JSONDecodeError` (JSON 解码失败)和通用异常。
*   **更好的错误提示:**  提供更清晰的错误提示信息，帮助用户诊断问题。
*   **UTF-8 编码:** 显式指定使用 UTF-8 编码打开文件，以处理包含中文等非 ASCII 字符的数据。
*   **空列表返回:** 如果加载失败，返回一个空列表，避免后续代码出错。 更好的做法是抛出异常 `raise FileNotFoundError(f"...")`
*   **更详细的信息:**  在加载成功时打印数据集的大小。

**中文解释:**

这段代码首先构建原始数据文件的完整路径。 然后，它检查文件是否存在。如果文件不存在，它会打印错误消息并返回空列表。如果文件存在，它尝试以 UTF-8 编码打开文件，并使用 `json.load` 函数加载 JSON 数据。如果在加载过程中发生错误（比如文件不存在、JSON 格式错误或未知错误），它会捕获相应的异常，打印错误消息，并返回空列表。如果加载成功，它会打印加载的数据集的大小，并返回加载的数据。

**3. 改进的 `load_title2qid` 函数 (加载标题到 QID 的映射):**

```python
import jsonlines
import os
from typing import Dict

def load_title2qid(dataset_dir: str) -> Dict[str, str]:
    """
    从 JSON Lines 文件加载标题到 QID 的映射。

    Args:
        dataset_dir (str): 数据集根目录。

    Returns:
        Dict[str, str]: 标题到 QID 的字典。
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    aliases_filepath = os.path.join(raw_dir, "id_aliases.json")

    title2qid: Dict[str, str] = {}

    try:
        with jsonlines.open(aliases_filepath, "r") as fin:
            for line in fin:
                qid, aliases = line.get("Q_id"), line.get("aliases", [])  # Use .get() for safety
                if qid:
                    for alias in aliases:
                        title2qid[alias] = qid
    except FileNotFoundError:
        print(f"错误：找不到 id_aliases.json 文件：{aliases_filepath}")
    except jsonlines.JSONLDecodeError:
        print(f"错误：JSON Lines 文件解码失败：{aliases_filepath}。文件可能损坏。")
    except Exception as e:
        print(f"加载 title2qid 时发生未知错误: {e}")


    print(f"已加载 title2qid 映射，包含 {len(title2qid)} 个条目。")
    return title2qid


# 演示用法
if __name__ == '__main__':
    dataset_dir = "./data"
    title2qid = load_title2qid(dataset_dir)

    if title2qid:
        print(f"部分映射：{list(title2qid.items())[:5]}")  # Print the first 5 items
```

**描述:**

这个函数负责从 `id_aliases.json` 文件加载标题到 QID 的映射。

**改进:**

*   **更安全的字典访问:** 使用 `.get()` 方法安全地访问 JSON 对象的键，避免 `KeyError` 异常。
*   **更好的错误处理:**  增加了 `try...except` 块来处理 `FileNotFoundError` 和 `jsonlines.JSONLDecodeError` 异常。
*   **更详细的信息:**  在加载成功时打印映射的大小。
*   **使用 `jsonlines`:**  正确使用 `jsonlines` 库来处理 JSON Lines 格式的文件.
*   **默认别名列表:** 使用 `line.get("aliases", [])`，如果 `aliases` 键不存在，则使用空列表作为默认值，避免出错.

**中文解释:**

这段代码首先构建 `id_aliases.json` 文件的完整路径。 然后，它尝试使用 `jsonlines.open` 函数打开文件。 对于文件中的每一行，它提取 QID 和别名列表，并将其添加到 `title2qid` 字典中。如果在加载过程中发生错误（比如文件不存在、JSON Lines 格式错误或未知错误），它会捕获相应的异常并打印错误消息。最后，它打印加载的映射的大小并返回该映射。

**4.  改进的 `format_raw_data` 函数 (格式化原始数据):**

```python
import uuid
from typing import Dict, List, Optional

from data_process.dataset_utils.hotpotqa import get_supporting_facts
from data_process.utils.question_type import infer_question_type

def format_raw_data(raw: dict) -> Optional[dict]:
    """
    格式化原始数据以符合数据集协议。

    Args:
        raw (dict): 原始数据字典。

    Returns:
        Optional[dict]: 格式化后的数据字典，如果处理失败则返回 None。
    """

    try:
        # Step 1: 提取支持事实内容
        supporting_facts = get_supporting_facts(raw["supporting_facts"], raw["context"])
        if supporting_facts is None:
            print(f"警告：无法提取支持事实，跳过样本。原始 ID：{raw.get('_id', 'N/A')}")  # 使用 .get()
            return None

        # Step 2: 格式化答案标签和问题类型
        answer_labels: List[str] = [raw["answer"]]
        qtype: str = infer_question_type(answer_labels)

        # Step 3: 构建格式化后的数据
        formatted_data = {
            "id": uuid.uuid4().hex,
            "question": raw["question"],
            "answer_labels": answer_labels,
            "question_type": qtype,
            "metadata": {
                "original_id": raw.get("_id", "N/A"),  # 使用 .get()
                "original_type": raw.get("type", "N/A"),  # 使用 .get()
                "supporting_facts": supporting_facts,
                "retrieval_contexts": [
                    {
                        "type": "wikipedia",
                        "title": title,
                        "contents": "".join(sentences),
                    }
                    for title, sentences in raw["context"]
                ],
                "reasoning_logics": [
                    {
                        "type": "wikidata",
                        "title": title,
                        "section": section,
                        "contents": content,
                    }
                    for title, section, content in raw["evidences"]
                ],
            },
        }

        return formatted_data

    except KeyError as e:
        print(f"错误：缺少必需的键：{e}。原始 ID：{raw.get('_id', 'N/A')}") # 使用 .get()
        return None
    except Exception as e:
        print(f"格式化数据时发生未知错误：{e}。原始 ID：{raw.get('_id', 'N/A')}") # 使用 .get()
        return None

# 演示用法 (你需要模拟 raw 数据)
if __name__ == '__main__':
    # 模拟原始数据 (你需要用真实数据替换)
    raw_data = {
        "_id": "sample_id",
        "type": "bridge",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 0]],
        "context": [("Paris", ["Paris is the capital of France."])],
        "evidences": []
    }

    formatted = format_raw_data(raw_data)

    if formatted:
        print("格式化成功:")
        print(formatted)
    else:
        print("格式化失败.")
```

**描述:**

这个函数负责将原始数据格式化为符合数据集协议的格式。

**改进:**

*   **更全面的错误处理:** 添加了 `try...except` 块来处理 `KeyError` 异常（缺少必需的键）和未知异常。
*   **更安全的字典访问:** 使用 `.get()` 方法安全地访问 JSON 对象的键，如果键不存在，则返回一个默认值，避免程序崩溃。  在日志中包含原始 ID，方便调试.
*   **更好的错误提示:** 提供更清晰的错误提示信息，包括原始 ID。
*   **提前跳过坏样本:** 如果无法提取支持事实，则提前跳过该样本，并打印警告消息。
*   **更清晰的结构:** 将格式化过程分解为多个步骤，使代码更易于阅读和理解。
*   **模拟演示数据:**  提供了一个模拟的原始数据，用于演示函数的用法。

**中文解释:**

这段代码首先尝试从原始数据中提取支持事实内容。 如果提取失败，它会打印警告消息并返回 `None`。然后，它格式化答案标签并推断问题类型。 接下来，它构建格式化后的数据字典，包括 ID、问题、答案标签、问题类型和元数据（原始 ID、原始类型、支持事实、检索上下文和推理逻辑）。如果在格式化过程中发生任何错误（比如缺少必需的键或未知错误），它会捕获相应的异常，打印错误消息，并返回 `None`。 最后，它返回格式化后的数据字典。

**总结:**

这些改进版本更健壮、更易于理解和维护。  它们包含了更全面的错误处理、更清晰的错误提示、更安全的数据访问方法、更模块化的结构和更详细的注释。  此外，它们还提供了演示用法，方便用户理解函数的用法。重要的是添加了中文注释方便理解。
