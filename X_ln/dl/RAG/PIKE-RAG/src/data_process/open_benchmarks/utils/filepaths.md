Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\filepaths.py`

Alright, restarting with a fresh explanation and demo examples for the provided code.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, Literal, Optional

from data_process.utils.stats import FILE_TYPES_TO_DOWNLOAD, SOURCE_TYPES_TO_DOWNLOAD


def get_dataset_dir(root_dir: str, dataset: str) -> str:
    """
    获取数据集目录的路径。

    Args:
        root_dir: 根目录路径。
        dataset: 数据集名称。

    Returns:
        数据集目录的完整路径。
    """
    return os.path.join(root_dir, dataset)

# 示例用法 
if __name__ == '__main__':
    root_directory = "/path/to/my/data"
    dataset_name = "my_dataset"
    dataset_directory = get_dataset_dir(root_directory, dataset_name)
    print(f"数据集目录：{dataset_directory}")  # 输出：数据集目录：/path/to/my/data/my_dataset

```

**描述:** `get_dataset_dir` 函数用于构建数据集的路径。它接收根目录和数据集名称，然后使用 `os.path.join` 将它们连接起来，返回完整的数据集目录路径。

**如何使用:**  传入根目录和数据集名称，即可获得数据集的完整路径。这在组织和访问数据集时非常有用。

```python
def get_split_filepath(root_dir: str, dataset: str, split: str, sample_num: Optional[int]) -> str:
    """
    获取数据集分割文件（如train/val/test）的路径。

    Args:
        root_dir: 根目录路径。
        dataset: 数据集名称。
        split: 数据集分割名称（如train, val, test）。
        sample_num: 如果存在，表示采样数量，否则为None。

    Returns:
        分割文件的完整路径。
    """
    if sample_num is None:
        filepath = os.path.join(root_dir, dataset, f"{split}.jsonl")
    else:
        filepath = os.path.join(root_dir, dataset, f"{split}_{sample_num}.jsonl")
    return filepath

# 示例用法
if __name__ == '__main__':
    root_directory = "/path/to/my/data"
    dataset_name = "my_dataset"
    split_name = "train"
    filepath = get_split_filepath(root_directory, dataset_name, split_name, None)
    print(f"训练集文件路径：{filepath}")  # 输出：训练集文件路径：/path/to/my/data/my_dataset/train.jsonl

    sample_num = 1000
    sampled_filepath = get_split_filepath(root_directory, dataset_name, split_name, sample_num)
    print(f"训练集抽样文件路径：{sampled_filepath}") # 输出：训练集抽样文件路径：/path/to/my/data/my_dataset/train_1000.jsonl
```

**描述:** `get_split_filepath` 函数用于构建数据集分割文件的路径。它接收根目录、数据集名称、分割名称和可选的样本数量。根据样本数量是否存在，构建不同的文件名。

**如何使用:**  传入根目录、数据集名称、分割名称和可选的样本数量，即可获得分割文件的完整路径。这方便了读取不同分割的数据。

```python
def get_document_dir(root_dir: str) -> str:
    """
    获取文档目录的路径，并创建必要的子目录结构。

    Args:
        root_dir: 根目录路径。

    Returns:
        文档目录的完整路径。
    """
    doc_dir = os.path.join(root_dir, "documents")

    # Create dirs for each source type and file type.
    for source_type in SOURCE_TYPES_TO_DOWNLOAD:
        for file_type in FILE_TYPES_TO_DOWNLOAD:
            dir = os.path.join(doc_dir, source_type, file_type)
            if not os.path.exists(dir):
                os.makedirs(dir)

    return doc_dir

# 假设 FILE_TYPES_TO_DOWNLOAD 和 SOURCE_TYPES_TO_DOWNLOAD 已定义
FILE_TYPES_TO_DOWNLOAD = ["pdf", "html"]
SOURCE_TYPES_TO_DOWNLOAD = ["wikipedia", "arxiv"]

# 示例用法
if __name__ == '__main__':
    root_directory = "/path/to/my/data"
    document_directory = get_document_dir(root_directory)
    print(f"文档目录：{document_directory}")  # 输出：文档目录：/path/to/my/data/documents

    # 此时会在 /path/to/my/data/documents 下创建 wikipedia/pdf, wikipedia/html, arxiv/pdf, arxiv/html 目录（如果不存在）

```

**描述:** `get_document_dir` 函数用于获取文档目录的路径，并根据预定义的 `SOURCE_TYPES_TO_DOWNLOAD` 和 `FILE_TYPES_TO_DOWNLOAD` 创建必要的子目录结构。

**如何使用:**  传入根目录，即可获得文档目录的完整路径，并且该函数还会自动创建用于存储不同类型文档的子目录。

```python
def get_doc_location_filepath(root_dir: str) -> str:
    """
    获取文档位置信息文件的路径。

    Args:
        root_dir: 根目录路径。

    Returns:
        文档位置信息文件的完整路径。
    """
    filepath = os.path.join(root_dir, "doc_title_type_to_location.json")
    return filepath

# 示例用法
if __name__ == '__main__':
    root_directory = "/path/to/my/data"
    doc_location_filepath = get_doc_location_filepath(root_directory)
    print(f"文档位置文件路径：{doc_location_filepath}")  # 输出：文档位置文件路径：/path/to/my/data/doc_title_type_to_location.json
```

**描述:** `get_doc_location_filepath` 函数用于构建文档位置信息文件的路径。

**如何使用:** 传入根目录，即可获得文档位置信息文件的完整路径。

```python
def get_title_status_filepath(root_dir: str) -> str:
    """
    获取标题状态信息文件的路径。

    Args:
        root_dir: 根目录路径。

    Returns:
        标题状态信息文件的完整路径。
    """
    filepath = os.path.join(root_dir, "wiki_title_type_to_validation_status.json")
    return filepath

# 示例用法
if __name__ == '__main__':
    root_directory = "/path/to/my/data"
    title_status_filepath = get_title_status_filepath(root_directory)
    print(f"标题状态文件路径：{title_status_filepath}")  # 输出：标题状态文件路径：/path/to/my/data/wiki_title_type_to_validation_status.json
```

**描述:** `get_title_status_filepath` 函数用于构建标题状态信息文件的路径。

**如何使用:** 传入根目录，即可获得标题状态信息文件的完整路径。

```python
def title_to_filename_prefix(title: str) -> str:
    """
    将标题转换为文件名前缀，替换掉"/"。

    Args:
        title: 文档标题。

    Returns:
        文件名前缀。
    """
    return title.replace("/", " ")

# 示例用法
if __name__ == '__main__':
    title = "My/Awesome/Document"
    filename_prefix = title_to_filename_prefix(title)
    print(f"文件名前缀：{filename_prefix}")  # 输出：文件名前缀：My Awesome Document
```

**描述:** `title_to_filename_prefix` 函数将标题转换为文件名前缀，将标题中的 "/" 替换为空格，以避免文件名非法字符。

**如何使用:** 传入文档标题，即可获得用于文件名的有效前缀。

```python
def get_download_filepaths(title: str, source_type: str, document_dir: str) -> Dict[Literal["pdf", "html"], str]:
    """
    获取指定标题和来源类型的文档下载路径。

    Args:
        title: 文档标题。
        source_type: 文档来源类型（如wikipedia, arxiv）。
        document_dir: 文档根目录。

    Returns:
        包含pdf和html文件路径的字典。
    """
    filename_prefix = title_to_filename_prefix(title)
    filepaths = {
        filetype: os.path.join(document_dir, source_type, filetype, f"{filename_prefix}.{filetype}")
        for filetype in FILE_TYPES_TO_DOWNLOAD
    }
    return filepaths

# 假设 FILE_TYPES_TO_DOWNLOAD 已定义
FILE_TYPES_TO_DOWNLOAD = ["pdf", "html"]

# 示例用法
if __name__ == '__main__':
    title = "My/Awesome/Document"
    source_type = "wikipedia"
    root_directory = "/path/to/my/data"
    document_dir = get_document_dir(root_directory)
    download_filepaths = get_download_filepaths(title, source_type, document_dir)
    print(f"下载文件路径：{download_filepaths}")
    # 输出:
    # 下载文件路径：{'pdf': '/path/to/my/data/documents/wikipedia/pdf/My Awesome Document.pdf', 'html': '/path/to/my/data/documents/wikipedia/html/My Awesome Document.html'}
```

**描述:** `get_download_filepaths` 函数用于构建特定标题、来源类型的文档下载路径。  它使用 `title_to_filename_prefix` 获取文件名前缀，并根据预定义的 `FILE_TYPES_TO_DOWNLOAD` 构建不同文件类型的完整路径。

**如何使用:**  传入文档标题、来源类型和文档根目录，即可获得包含 PDF 和 HTML 文件路径的字典。  这方便了下载和存储不同格式的文档。
