Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\musique.py`

好的，我们重新开始。 这次，我会专注于提供更模块化、更易于理解的代码，并附上清晰的中文注释和更详细的解释。 我将把代码分解成更小的片段，并提供一个完整的示例，以便您可以轻松地将其整合到您的项目中。

**1. 更健壮的下载函数 (More Robust Download Function):**

```python
import os
import subprocess
import requests
import zipfile

def download_raw_data(raw_dir: str, zipfile_id: str) -> None:
    """
    下载并解压原始数据.

    Args:
        raw_dir (str): 存储原始数据的目录.
        zipfile_id (str): Google Drive 文件 ID.
    """
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "musique.zip")

    # 使用 requests 下载，避免依赖 gdown
    download_url = f"https://drive.google.com/uc?id={zipfile_id}&export=download"  # Updated URL
    try:
        print(f"开始下载文件从 {download_url} 到 {output_path}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # 检查下载是否成功
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): # 8KB chunks
                f.write(chunk)

        print(f"成功下载到 {output_path}")

        # 解压文件
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

        print(f"成功解压到 {raw_dir}")

        # 移动文件 (如果需要)
        data_dir = os.path.join(raw_dir, "data")
        if os.path.exists(data_dir):  # 确保 "data" 目录存在
            for filename in os.listdir(data_dir):
                source = os.path.join(data_dir, filename)
                destination = os.path.join(raw_dir, filename)
                os.rename(source, destination)
            os.rmdir(data_dir)  # 删除空的 "data" 目录
            print("成功移动文件并删除 'data' 目录")
        else:
            print("'data' 目录不存在，跳过移动步骤")

    except requests.exceptions.RequestException as e:
        print(f"下载出错: {e}")
    except zipfile.BadZipFile as e:
        print(f"解压出错: {e}")
    except Exception as e:
        print(f"其他错误: {e}")



# Demo Usage: 演示用法
if __name__ == '__main__':
    raw_dir = "temp_raw_data"  # 用于存储下载数据的临时目录
    zipfile_id = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h" # Your zipfile id

    download_raw_data(raw_dir, zipfile_id)
    print(f"数据已下载并解压到: {raw_dir}")

```

**描述:**

*   **不再依赖 `gdown`:**  使用 `requests` 库直接从 Google Drive 下载文件，避免了对外部库的依赖，使代码更易于部署和维护。
*   **错误处理:** 增加了 `try...except` 块，可以捕获下载和解压过程中可能出现的异常，例如网络错误或损坏的 ZIP 文件。 这样可以使代码更加健壮，并提供有用的错误信息。
*   **流式下载:** 使用 `response.iter_content` 进行流式下载，可以处理大型文件，而不会将整个文件加载到内存中。
*   **更清晰的输出:** 提供了更详细的输出，指示下载和解压过程的状态。
*   **更正后的 Google Drive URL:** 使用了正确的 Google Drive 下载 URL 格式。

**2. 更清晰的数据加载函数 (More Readable Data Loading Function):**

```python
import os
import jsonlines
from typing import List, Dict

def load_raw_data(dataset_dir: str, split: str) -> List[Dict]:
    """
    加载原始数据从 JSONL 文件.

    Args:
        dataset_dir (str): 包含原始数据的目录.
        split (str): 数据集分割 (例如, "train", "dev", "test").

    Returns:
        List[Dict]: 原始数据的列表，其中每个元素都是一个字典.
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True) # 确保目录存在
    raw_filepath = os.path.join(raw_dir, f"musique_ans_v1.0_{split}.jsonl")

    if not os.path.exists(raw_filepath):
        print(f"文件 {raw_filepath} 不存在，尝试下载...")
        download_raw_data(raw_dir, zipfile_id)  # 确保 `download_raw_data` 函数可用
        if not os.path.exists(raw_filepath):
            raise FileNotFoundError(f"下载后文件 {raw_filepath} 仍然不存在.")

    dataset = []
    try:
        with jsonlines.open(raw_filepath, "r") as reader:
            for data in reader:
                dataset.append(data)
        print(f"成功加载 {len(dataset)} 条数据从 {raw_filepath}")

    except FileNotFoundError:
        print(f"文件 {raw_filepath} 未找到.")
        return []  # 返回空列表而不是引发异常
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [] # 返回空列表而不是引发异常

    return dataset

# Demo Usage
if __name__ == '__main__':
    dataset_dir = "temp_dataset" # Replace with your dataset directory
    split = "train"
    raw_data = load_raw_data(dataset_dir, split)

    if raw_data:
        print(f"加载了 {len(raw_data)} 条数据.")
        # 打印第一条数据的键
        if raw_data:
            print("第一条数据的键:", raw_data[0].keys())

    else:
        print("未能加载数据.")
```

**描述:**

*   **更清晰的错误处理:**  使用 `try...except` 块来处理文件未找到或读取错误的情况。 这样可以避免程序崩溃，并提供有用的错误信息。  如果文件不存在，会尝试下载数据。
*   **明确的文件路径:**  更清晰地构建文件路径，使代码更易于理解。
*   **详细的日志记录:**  提供了更详细的日志记录，指示文件是否已成功加载以及加载了多少条记录。
*   **空列表返回:**  如果文件未找到或读取失败，则返回一个空列表，而不是引发异常。 这样可以使代码更易于使用，并避免在调用代码中进行额外的错误处理。
*   **检查下载后文件是否存在:**  在下载后再次检查文件是否存在，确保下载过程成功。
*   **类型提示 (Type Hints):**  使用了类型提示，使代码更易于阅读和理解。

**3. 更灵活的数据格式化函数 (More Flexible Data Formatting Function):**

```python
import copy
import uuid
from typing import Dict, List, Literal, Optional

from data_process.utils.question_type import infer_question_type  # 假设这个函数存在

def format_raw_data(raw: Dict) -> Optional[Dict]:
    """
    格式化原始数据，使其符合目标数据协议.

    Args:
        raw (Dict): 原始数据字典.

    Returns:
        Optional[Dict]: 格式化后的数据字典，如果格式化失败则返回 None.
    """
    try:
        # Step 1: Extract contents of Retrieved Contexts and Supporting Facts
        retrieval_contexts: List[Dict[Literal["type", "title", "contents"], str]] = [
            {
                "type": "wikipedia",
                "title": paragraph["title"],
                "contents": paragraph["paragraph_text"],
            }
            for paragraph in raw.get("paragraphs", []) # Use .get() with default empty list
        ]

        supporting_facts: List[Dict[Literal["type", "title", "contents"], str]] = []
        for item in raw.get("question_decomposition", []): # Use .get()
            try:
                supporting_facts.append(copy.deepcopy(retrieval_contexts[item["paragraph_support_idx"]]))
            except IndexError:
                print(f"IndexError: paragraph_support_idx {item.get('paragraph_support_idx', 'N/A')} 超出 retrieval_contexts 范围.")
                continue  # Skip this item if index is out of range

        # Step 2: Extract Answer Labels
        answer_labels: List[str] = [raw["answer"]] + raw.get("answer_aliases", []) # Use .get()

        # Step 3: Infer Question Type
        qtype: str = infer_question_type(answer_labels)

        # Step 4: Create Formatted Data
        formatted_data = {
            "id": str(uuid.uuid4()), # Generate UUID as string
            "question": raw["question"],
            "answer_labels": answer_labels,
            "question_type": qtype,
            "metadata": {
                "original_id": raw["id"],
                "supporting_facts": supporting_facts,
                "retrieval_contexts": retrieval_contexts,
            }
        }

        return formatted_data

    except KeyError as e:
        print(f"KeyError: 缺少键 {e}. 无法格式化数据.")
        return None # Return None instead of raising exception
    except Exception as e:
        print(f"格式化数据时出错: {e}")
        return None  # Return None instead of raising exception

# Demo Usage
if __name__ == '__main__':
    # Create a dummy raw data dictionary
    dummy_raw_data = {
        "id": "123",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "answer_aliases": ["City of Lights"],
        "paragraphs": [
            {"title": "Paris", "paragraph_text": "Paris is the capital of France."},
            {"title": "France", "paragraph_text": "France is a country in Europe."}
        ],
        "question_decomposition": [
            {"paragraph_support_idx": 0}
        ]
    }

    formatted_data = format_raw_data(dummy_raw_data)

    if formatted_data:
        print("格式化后的数据:", formatted_data)
    else:
        print("数据格式化失败.")
```

**描述:**

*   **更强的错误处理:**  使用 `try...except` 块来处理 `KeyError` 和其他可能发生的异常。 如果在格式化过程中发生错误，函数将返回 `None`，而不是引发异常。
*   **使用 `.get()` 方法:** 使用 `.get()` 方法来访问字典中的值，以避免 `KeyError`。 如果键不存在，`.get()` 方法将返回 `None` 或指定的默认值。 这可以使代码更加健壮，并避免在数据集中存在缺少键时出现问题。
*   **默认值:**  为 `.get()` 方法提供默认值，例如 `[]` 对于列表。 这样可以确保即使缺少某个键，代码也可以继续运行，而不会引发异常。
*   **更清晰的变量名:**  使用了更清晰的变量名，使代码更易于理解。
*   **UUID 作为字符串:**  将 UUID 生成为字符串，使其更易于序列化和存储。
*   **IndexError 处理:** 增加了对 `IndexError` 的处理，当 `paragraph_support_idx` 超出 `retrieval_contexts` 范围时，跳过当前项，避免程序崩溃。
*    **类型提示 (Type Hints):** 使用了类型提示，使代码更易于阅读和理解。

**总结:**

这些改进使代码更加健壮、易于理解和维护。 错误处理得到了改进，代码使用了更清晰的变量名和更详细的日志记录。 此外，代码现在可以处理缺少键的情况，并且可以避免在数据集中存在缺少键时出现问题。 代码结构更模块化，方便您根据自己的需要进行修改和扩展。

希望这些改进对您有所帮助！ 让我知道您是否需要任何进一步的修改或解释。
