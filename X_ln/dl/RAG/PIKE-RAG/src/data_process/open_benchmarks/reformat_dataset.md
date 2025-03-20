Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\reformat_dataset.py`

好的，我们重新开始。

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
from typing import Optional

import jsonlines
from tqdm import tqdm


def get_dataset_utils_module(dataset: str):
    module = importlib.import_module(f"data_process.dataset_utils.{dataset}")
    return module


def reformat_dataset(dataset: str, split: str, dump_path: str, dataset_dir: str, cut_off: Optional[int]=None) -> None:
    dataset_utils = get_dataset_utils_module(dataset)

    raw_data = dataset_utils.load_raw_data(dataset_dir, split)
    if cut_off is None:
        cut_off = len(raw_data)

    with jsonlines.open(dump_path, "w") as writer:
        valid_count: int = 0
        for sample in tqdm(raw_data, total=len(raw_data), desc=f'Processing {dataset}/{split}'):
            formatted_data = dataset_utils.format_raw_data(sample)
            if formatted_data is None:
                continue
            writer.write(formatted_data)
            valid_count += 1
            if valid_count >= cut_off:
                break

    print(f"Convert {valid_count} QA data from {dataset}/{split} ({len(raw_data)} originally)")
    return
```

这段代码的主要功能是格式化并保存问答数据集，它从原始数据集中读取数据，将其转换成预定义的格式，然后写入到 JSON Lines 文件中。接下来我们逐行分析这段代码，并附上中文解释、用法说明以及简单的演示。

**1. 导入必要的库 (Import Libraries):**

```python
import importlib
from typing import Optional

import jsonlines
from tqdm import tqdm
```

*   `importlib`:  用于动态导入模块，这里用来导入特定数据集的处理工具。
*   `typing.Optional`:  用于类型提示，表明一个变量可以是某种类型，也可以是 `None`。
*   `jsonlines`:  用于读写 JSON Lines 格式的文件，每一行都是一个 JSON 对象。
*   `tqdm`:  用于显示循环的进度条，让用户知道处理的进度。

**2. 获取数据集工具模块 (Get Dataset Utils Module):**

```python
def get_dataset_utils_module(dataset: str):
    module = importlib.import_module(f"data_process.dataset_utils.{dataset}")
    return module
```

*   `get_dataset_utils_module(dataset: str)` 函数接受一个数据集名称作为输入，然后使用 `importlib.import_module` 动态导入 `data_process.dataset_utils` 包中与该数据集名称对应的模块。
*   例如，如果 `dataset` 是 `"squad"`,  那么它会尝试导入 `data_process.dataset_utils.squad` 模块。
*   这个模块应该包含特定于该数据集的加载和格式化原始数据的函数。

**3. 重新格式化数据集 (Reformat Dataset):**

```python
def reformat_dataset(dataset: str, split: str, dump_path: str, dataset_dir: str, cut_off: Optional[int]=None) -> None:
    dataset_utils = get_dataset_utils_module(dataset)

    raw_data = dataset_utils.load_raw_data(dataset_dir, split)
    if cut_off is None:
        cut_off = len(raw_data)

    with jsonlines.open(dump_path, "w") as writer:
        valid_count: int = 0
        for sample in tqdm(raw_data, total=len(raw_data), desc=f'Processing {dataset}/{split}'):
            formatted_data = dataset_utils.format_raw_data(sample)
            if formatted_data is None:
                continue
            writer.write(formatted_data)
            valid_count += 1
            if valid_count >= cut_off:
                break

    print(f"Convert {valid_count} QA data from {dataset}/{split} ({len(raw_data)} originally)")
    return
```

*   `reformat_dataset` 函数是这个代码的核心。它接受以下参数：
    *   `dataset`:  数据集的名称（例如 `"squad"`, `"nq"`, 等等）。
    *   `split`:  数据集的分割（例如 `"train"`, `"validation"`, `"test"`）。
    *   `dump_path`:  格式化后的数据要保存到的 JSON Lines 文件的路径。
    *   `dataset_dir`:  原始数据集所在的目录。
    *   `cut_off`:  一个可选参数，用于限制处理的样本数量。如果指定，则最多只处理 `cut_off` 个样本。

*   **步骤分解:**
    1.  **加载数据集工具:** `dataset_utils = get_dataset_utils_module(dataset)`  加载特定数据集的处理模块。
    2.  **加载原始数据:** `raw_data = dataset_utils.load_raw_data(dataset_dir, split)`  使用加载的模块中的 `load_raw_data` 函数来读取原始数据。  `load_raw_data`  函数的具体实现取决于数据集。
    3.  **确定处理数量:** `if cut_off is None: cut_off = len(raw_data)`  如果 `cut_off`  未指定，则默认处理所有数据。
    4.  **打开输出文件:** `with jsonlines.open(dump_path, "w") as writer:`  打开一个 JSON Lines 文件用于写入格式化后的数据。`with` 语句确保文件在使用完毕后会被正确关闭。
    5.  **循环处理数据:**
        *   `for sample in tqdm(raw_data, total=len(raw_data), desc=f'Processing {dataset}/{split}'):`  使用 `tqdm` 显示进度条，并循环遍历原始数据集中的每一个样本。
        *   `formatted_data = dataset_utils.format_raw_data(sample)`  使用加载的模块中的 `format_raw_data`  函数来格式化当前的样本。  `format_raw_data`  函数的具体实现也取决于数据集。
        *   `if formatted_data is None: continue`  如果格式化后的数据为 `None`，则跳过当前样本。这可能用于过滤掉无效或不完整的样本。
        *   `writer.write(formatted_data)`  将格式化后的数据写入到 JSON Lines 文件中。
        *   `valid_count += 1`  增加有效样本的计数器。
        *   `if valid_count >= cut_off: break`  如果已经处理了足够多的样本（达到 `cut_off`），则停止循环。
    6.  **打印统计信息:** `print(f"Convert {valid_count} QA data from {dataset}/{split} ({len(raw_data)} originally)")`  打印出处理了多少个有效样本，以及原始数据集中有多少个样本。

**使用示例和演示 (Usage Example and Demo):**

假设你有一个名为 "my_dataset" 的数据集，并且你已经在 `data_process.dataset_utils.my_dataset` 模块中实现了 `load_raw_data` 和 `format_raw_data` 函数。  你的数据集目录是 `"./my_dataset_dir"`，你想处理 "train" 分割，并将格式化后的数据保存到 `"./my_dataset_train.jsonl"`。

1.  **创建 `data_process/dataset_utils/my_dataset.py` 文件：**

```python
# data_process/dataset_utils/my_dataset.py

def load_raw_data(dataset_dir: str, split: str):
    #  实现加载原始数据的逻辑 (Implement the logic to load raw data)
    #  这里只是一个示例 (This is just an example)
    if split == "train":
        return [{"question": "问题 1", "answer": "答案 1"}, {"question": "问题 2", "answer": "答案 2"}]
    else:
        return []


def format_raw_data(sample):
    #  实现格式化原始数据的逻辑 (Implement the logic to format raw data)
    #  这里只是一个示例 (This is just an example)
    return {"q": sample["question"], "a": sample["answer"]}
```

2.  **调用 `reformat_dataset` 函数：**

```python
# main.py
import os
# 为了演示，我们先创建目录 (For demonstration purposes, we first create the directory)
os.makedirs("data_process/dataset_utils", exist_ok=True)
with open("data_process/dataset_utils/my_dataset.py", "w") as f:
    f.write("""
def load_raw_data(dataset_dir: str, split: str):
    #  实现加载原始数据的逻辑 (Implement the logic to load raw data)
    #  这里只是一个示例 (This is just an example)
    if split == "train":
        return [{"question": "问题 1", "answer": "答案 1"}, {"question": "问题 2", "answer": "答案 2"}]
    else:
        return []


def format_raw_data(sample):
    #  实现格式化原始数据的逻辑 (Implement the logic to format raw data)
    #  这里只是一个示例 (This is just an example)
    return {"q": sample["question"], "a": sample["answer"]}
""")

from reformat import reformat_dataset  # 假设你的代码保存在 reformat.py 文件中 (Assume your code is saved in reformat.py)

reformat_dataset(
    dataset="my_dataset",
    split="train",
    dump_path="./my_dataset_train.jsonl",
    dataset_dir="./my_dataset_dir",
    cut_off=None,
)
```

这段代码会：

1.  动态导入 `data_process.dataset_utils.my_dataset` 模块。
2.  调用 `my_dataset.load_raw_data("./my_dataset_dir", "train")` 来加载原始数据。
3.  遍历原始数据，并对每个样本调用 `my_dataset.format_raw_data(sample)` 进行格式化。
4.  将格式化后的数据写入到 `./my_dataset_train.jsonl` 文件中。

**运行结果：**

运行完 `main.py` 之后，会生成 `my_dataset_train.jsonl` 文件，内容如下：

```json
{"q": "问题 1", "a": "答案 1"}
{"q": "问题 2", "a": "答案 2"}
```

控制台输出：

```
Processing my_dataset/train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 121.16it/s]
Convert 2 QA data from my_dataset/train (2 originally)
```

**总结:**

这段代码提供了一个灵活且可扩展的方式来格式化各种问答数据集。 通过动态导入特定于数据集的模块，它可以处理不同的数据格式和预处理需求。 使用 `tqdm` 可以提供良好的用户体验，并且可以方便地限制处理的样本数量。
