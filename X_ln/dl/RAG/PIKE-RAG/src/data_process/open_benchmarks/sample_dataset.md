Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\sample_dataset.py`

好的，现在我将重新开始，并对提供的代码进行详细解释，生成每个小代码片段并附上简短的中文描述，并在描述中包含代码的使用方式和简单演示。

**1. 导入必要的库 (Import Libraries):**

```python
import importlib
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from data_process.utils.filepaths import get_doc_location_filepath, get_download_filepaths, get_title_status_filepath
from data_process.utils.io import dump_to_json_file, load_from_json_file, dump_to_jsonlines, load_from_jsonlines
from data_process.utils.stats import SOURCE_TYPES_TO_DOWNLOAD
from data_process.utils import wikidata, wikipedia
```

**描述:** 这部分代码导入了程序所需的所有库。包括用于文件操作、数据类型定义、数值计算、进度条显示以及自定义工具模块的库。

**2. 加载缓存 (Load Caches):**

```python
def load_caches(document_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, bool]]]:
    location_path = get_doc_location_filepath(document_dir)
    validation_path = get_title_status_filepath(document_dir)

    def _load(filepath: str) -> dict:
        mapping = load_from_json_file(filepath)
        if mapping is None:
            mapping = {source_type: {} for source_type in SOURCE_TYPES_TO_DOWNLOAD}
        return mapping

    return _load(location_path), _load(validation_path)
```

**描述:** `load_caches` 函数用于加载已存在的文档位置和验证状态的缓存信息。它从指定路径读取JSON文件，并返回两个字典：`title_to_location` 和 `title_to_validation`。如果文件不存在，则创建一个空的字典。

**使用方式:**  在程序开始时调用此函数，以避免重复下载已存在的文档。
**示例:**
```python
document_directory = "my_documents"
location_cache, validation_cache = load_caches(document_directory)
print(f"加载的文档位置缓存：{location_cache}")
print(f"加载的文档验证缓存：{validation_cache}")
```

**3. 保存缓存 (Dump Caches):**

```python
def dump_caches(document_dir: str, location_mapping: dict, validation_mapping: dict) -> None:
    dump_to_json_file(get_doc_location_filepath(document_dir), location_mapping)
    dump_to_json_file(get_title_status_filepath(document_dir), validation_mapping)
    return
```

**描述:** `dump_caches` 函数用于将文档位置和验证状态的缓存信息保存到JSON文件中。

**使用方式:** 在程序运行过程中或结束时调用此函数，以保存最新的缓存信息。
**示例:**
```python
document_directory = "my_documents"
location_cache = {"wikipedia": {"Example Title": {"text": "path/to/text.txt"}}}
validation_cache = {"wikipedia": {"Example Title": True}}
dump_caches(document_directory, location_cache, validation_cache)
```

**4. 获取待下载的标题 (Get Titles to Download):**

```python
def get_titles_to_download(sample: dict) -> Dict[str, List[str]]:
    """Download only the supporting facts related documents if exist, else retrieval contexts related documents."""
    metadata = sample["metadata"]
    titles_by_type = {source_type: set() for source_type in SOURCE_TYPES_TO_DOWNLOAD}
    if "supporting_facts" in metadata:
        for supporting_fact in metadata["supporting_facts"]:
            titles_by_type[supporting_fact["type"]].add(supporting_fact["title"])
    else:
        for retrieval_context in metadata["retrieval_contexts"]:
            titles_by_type[retrieval_context["type"]].add(retrieval_context["title"])

    return {key: list(value_set) for key, value_set in titles_by_type.items()}
```

**描述:** `get_titles_to_download` 函数从数据样本中提取需要下载的文档标题。优先下载 supporting facts 相关的文档，如果不存在，则下载 retrieval contexts 相关的文档。

**使用方式:**  对于每个数据样本，调用此函数以确定需要下载哪些文档。
**示例:**
```python
sample_data = {
    "metadata": {
        "supporting_facts": [
            {"type": "wikipedia", "title": "Example Title 1"},
            {"type": "wikidata", "title": "Example Title 2"}
        ],
        "retrieval_contexts": [
            {"type": "wikipedia", "title": "Example Title 3"}
        ]
    }
}
titles = get_titles_to_download(sample_data)
print(f"需要下载的标题：{titles}") # {'wikipedia': ['Example Title 1'], 'wikidata': ['Example Title 2']}
```

**5. 尝试下载 (Try Download):**

```python
def try_download(
    titles_by_type: Dict[str, List[str]],
    title2qid: Dict[str, str],
    title_to_location: Dict[str, Dict[str, Dict[str, str]]],
    title_to_validation: Dict[str, Dict[str, bool]],
    document_dir: str,
) -> Tuple[bool, int]:
    cache_update_count: int = 0
    newly_download_count: int = 0

    # Skip this download if there is a title cached as invalid (to download).
    for source_type, titles in titles_by_type.items():
        for title in titles:
            if title in title_to_validation[source_type] and title_to_validation[source_type][title] is False:
                return False, cache_update_count, newly_download_count

    for source_type, titles in titles_by_type.items():
        # Filter out titles if already exists in title_to_location cache.
        remaining_titles = [title for title in titles if title not in title_to_location[source_type]]
        dump_filepaths_list = [get_download_filepaths(title, source_type, document_dir) for title in remaining_titles]

        # Filter out titles if file already exists in path (although not cached).
        indexes_to_remove: List[int] = []
        for idx, title in enumerate(remaining_titles):
            dump_filepaths = dump_filepaths_list[idx]
            files_exist: bool = True
            for filepath in dump_filepaths.values():
                if not os.path.exists(filepath):
                    files_exist = False
                    break
            if files_exist is True:
                title_to_validation[source_type][title] = True
                title_to_location[source_type][title] = dump_filepaths
                indexes_to_remove.append(idx)
        if len(indexes_to_remove) > 0:
            remaining_titles = [title for idx, title in enumerate(remaining_titles) if idx not in indexes_to_remove]
            dump_filepaths_list = [path_list for idx, path_list in enumerate(dump_filepaths_list) if idx not in indexes_to_remove]
            cache_update_count += len(indexes_to_remove)

        if len(remaining_titles) == 0:
            continue

        if source_type == "wikipedia":
            success, title2valid = wikipedia.download_all_titles(remaining_titles, dump_filepaths_list)

        elif source_type == "wikidata":
            success, title2valid = wikidata.download_all_titles(remaining_titles, dump_filepaths_list, title2qid)

        else:
            raise ValueError(f"Unsupported source_type {source_type}!")

        # Update mappings.
        title_to_validation[source_type].update(title2valid)
        cache_update_count += len(title2valid)

        if success is True:
            title_to_location[source_type].update(
                {title: dump_filepaths for title, dump_filepaths in zip(remaining_titles, dump_filepaths_list)}
            )
            newly_download_count += len(remaining_titles)
        else:
            return False, cache_update_count, newly_download_count

    return True, cache_update_count, newly_download_count
```

**描述:** `try_download` 函数尝试下载指定的文档标题。它首先检查缓存中是否存在无效的标题，如果存在则跳过下载。然后，它过滤掉已经存在于缓存中的标题，并下载剩余的标题。根据 `source_type` (wikipedia 或 wikidata) 调用相应的下载函数。最后，更新缓存并返回下载结果。

**使用方式:** 调用此函数以下载文档，并更新缓存信息。
**示例:**
```python
titles_to_download = {"wikipedia": ["Example Title 1"], "wikidata": ["Example Title 2"]}
title_to_qid = {"Example Title 2": "Q123"}
document_directory = "my_documents"
location_cache, validation_cache = load_caches(document_directory)

success, updated_count, downloaded_count = try_download(
    titles_to_download, title_to_qid, location_cache, validation_cache, document_directory
)

print(f"下载是否成功: {success}")
print(f"缓存更新数量: {updated_count}")
print(f"新下载数量: {downloaded_count}")
```

**6. 数据集采样 (Sample Datasets):**

```python
def sample_datasets(
    dataset: str, split: str, sample_size_list: List[int], random_seed: int, document_dir: str,
    split_path_func: Callable[[Optional[int]], str], cache_every_updates: int=20,
) -> None:
    # set random seed for each dataset.
    np.random.seed(random_seed)

    # get raw split data.
    raw_split_path = split_path_func(sample_num=None)
    split_data: List[dict] = load_from_jsonlines(raw_split_path)

    # load title2qid dict for some datasets.
    title2qid: Dict[str, str] = {}
    if dataset in ["two_wiki", "popqa"]:
        dataset_dir: str = os.path.dirname(raw_split_path)
        module = importlib.import_module(f"data_process.dataset_utils.{dataset}")
        title2qid = module.load_title2qid(dataset_dir, split)

    # truncate sample size list to display tqdm correctly
    for i in range(len(sample_size_list)):
        if sample_size_list[i] > len(split_data):
            sample_size_list = sample_size_list[:i+1]
            break

    # Read location mapping and validation mapping from disk to avoid duplicated downloads.
    title_to_location, title_to_validation = load_caches(document_dir)
    newly_updated: int = 0

    # lists to save qualified samples and selected indexes
    chosen_samples: List[dict] = []
    chosen_indexes: Set[int] = set()
    remaining_indexes: Set[int] = set(range(len(split_data)))

    # sample valid samples for each sample size
    for sample_size in tqdm(sample_size_list, total=len(sample_size_list), desc=f"Sampling {dataset}/{split}"):

        acc_updated: int = 0
        acc_downloaded: int = 0
        download_bar_desc: str = f"Downloading for size: {sample_size} " + "(Status updated: {}, Newly downloaded: {})"
        download_pbar = tqdm(total=sample_size - len(chosen_samples), desc=download_bar_desc.format(0, 0))
        while len(chosen_samples) < sample_size and len(remaining_indexes) > 0:
            # Sample indexes for remaining index list.
            num_to_sample = min(sample_size - len(chosen_samples), len(remaining_indexes))
            newly_sampled_indexes = np.random.choice(list(remaining_indexes), size=num_to_sample, replace=False)
            remaining_indexes -= set(newly_sampled_indexes)

            # TODO: replace the for-loop below with an async-one if possible.

            # Check the validation of the newly sampled one, download files if valid.
            for idx in newly_sampled_indexes:
                sample = split_data[idx]
                titles_by_type: Dict[str, List[str]] = get_titles_to_download(sample)
                success, updated, downloaded = try_download(
                    titles_by_type, title2qid, title_to_location, title_to_validation, document_dir,
                )
                acc_updated += updated
                acc_downloaded += downloaded

                if success is True:
                    chosen_samples.append(sample)
                    chosen_indexes.add(idx)
                    download_pbar.set_description_str(desc=download_bar_desc.format(acc_updated, acc_downloaded))
                    download_pbar.update(1)

                newly_updated += updated
                if newly_updated >= cache_every_updates:
                    dump_caches(document_dir, title_to_location, title_to_validation)
                    newly_updated = 0
        download_pbar.close()

        # Save the chosen samples and name it with the number of samples it contains till now.
        assert len(chosen_samples) == len(chosen_indexes), "Samples could not be duplicated in one set!"
        dump_path: str = split_path_func(sample_num=len(chosen_samples))
        dump_to_jsonlines(dump_path, chosen_samples)

    dump_caches(document_dir, title_to_location, title_to_validation)
    return
```

**描述:** `sample_datasets` 函数从原始数据集中采样数据，并下载相关的文档。它首先加载数据集，然后根据指定的采样大小进行采样。对于每个样本，它调用 `try_download` 函数下载相关的文档，并更新缓存。最后，将采样后的数据保存到文件中。

**使用方式:** 调用此函数以采样数据集，并下载相关的文档。
**示例:**
```python
dataset_name = "my_dataset"
split_name = "train"
sample_sizes = [100, 200, 300]
random_seed = 42
document_directory = "my_documents"

def split_path_function(sample_num: Optional[int]) -> str:
    if sample_num is None:
        return f"data/{dataset_name}/{split_name}.jsonl"  # 原始数据路径
    else:
        return f"data/{dataset_name}/{split_name}_{sample_num}.jsonl" # 采样后数据路径

sample_datasets(dataset_name, split_name, sample_sizes, random_seed, document_directory, split_path_function)
```

希望这个详细的解释和示例能够帮助你理解代码的功能和使用方式。
