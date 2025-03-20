Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\sample_dataset.py`

好的，现在我们来改进您提供的代码。目标是使其更高效、更易读，并减少潜在的错误。我将逐步介绍改进后的代码片段，并附上中文解释。

**1. 并发下载（使用 `asyncio` 和 `aiohttp`）:**

```python
import asyncio
import aiohttp

async def download_title(session: aiohttp.ClientSession, title: str, source_type: str, title2qid: Dict[str, str], document_dir: str) -> Tuple[str, bool, Dict]:
    """异步下载单个文档"""
    dump_filepaths = get_download_filepaths(title, source_type, document_dir)
    files_exist = all(os.path.exists(filepath) for filepath in dump_filepaths.values())

    if files_exist:
        return title, True, dump_filepaths  # 已经存在，跳过

    try:
        if source_type == "wikipedia":
            success, _ = await wikipedia.download_title_async(session, title, dump_filepaths)
        elif source_type == "wikidata":
            success, _ = await wikidata.download_title_async(session, title, dump_filepaths, title2qid)
        else:
            raise ValueError(f"Unsupported source_type {source_type}!")

        return title, success, dump_filepaths if success else {}
    except Exception as e:
        print(f"Error downloading {title} ({source_type}): {e}")
        return title, False, {}

async def process_titles(
    titles_by_type: Dict[str, List[str]],
    title2qid: Dict[str, str],
    title_to_location: Dict[str, Dict[str, Dict[str, str]]],
    title_to_validation: Dict[str, Dict[str, bool]],
    document_dir: str
) -> Tuple[int, int]:
    """异步处理多个标题下载"""
    cache_update_count = 0
    newly_download_count = 0
    async with aiohttp.ClientSession() as session:  # 创建一个全局的会话
        tasks = []
        for source_type, titles in titles_by_type.items():
            for title in titles:
                if title in title_to_validation[source_type] and title_to_validation[source_type][title] is False:
                    continue  # 跳过无效标题
                if title in title_to_location[source_type]:
                    continue  # 跳过已经下载的标题

                task = asyncio.create_task(download_title(session, title, source_type, title2qid, document_dir))
                tasks.append(task)

        results = await asyncio.gather(*tasks)  # 并发执行所有下载任务

        for title, success, dump_filepaths in results:
            source_type = next(st for st, titles in titles_by_type.items() if title in titles)
            title_to_validation[source_type][title] = success
            cache_update_count += 1
            if success:
                title_to_location[source_type][title] = dump_filepaths
                newly_download_count += 1

    return cache_update_count, newly_download_count
```

**描述:**

*   **`download_title`**:  这是一个异步函数，负责下载单个文档。它使用 `aiohttp.ClientSession` 来发起 HTTP 请求，并根据 `source_type` 调用相应的下载函数（`wikipedia.download_title_async` 或 `wikidata.download_title_async`）。如果下载成功，它会返回标题、成功标志和文件路径信息。如果下载失败，它会返回标题、失败标志和一个空字典。

*   **`process_titles`**:  这个异步函数负责处理多个标题的下载。它首先创建一个 `aiohttp.ClientSession` 对象，用于管理多个并发的 HTTP 连接。然后，它遍历 `titles_by_type` 字典，为每个需要下载的标题创建一个 `asyncio.Task` 对象，并将所有任务添加到 `tasks` 列表中。最后，它使用 `asyncio.gather` 函数并发执行所有下载任务，并将结果存储在 `results` 列表中。

    对于每个下载结果，它会更新 `title_to_validation` 和 `title_to_location` 字典，并增加相应的计数器。

**2.  修改 `try_download` 函数:**

```python
def try_download(
    titles_by_type: Dict[str, List[str]],
    title2qid: Dict[str, str],
    title_to_location: Dict[str, Dict[str, Dict[str, str]]],
    title_to_validation: Dict[str, Dict[str, bool]],
    document_dir: str,
) -> bool:
    """尝试下载文档"""
    loop = asyncio.get_event_loop()
    cache_update_count, newly_download_count = loop.run_until_complete(
        process_titles(titles_by_type, title2qid, title_to_location, title_to_validation, document_dir)
    )
    return newly_download_count > 0
```

**描述:**

*   `try_download` 函数现在使用 `asyncio.get_event_loop()` 获取事件循环，并使用 `loop.run_until_complete()` 运行 `process_titles` 函数，直到它完成。  如果下载了新的文档，则返回 `True`，否则返回 `False`。

**3.  修改 `sample_datasets` 函数:**

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
    newly_updated: 0

    # lists to save qualified samples and selected indexes
    chosen_samples: List[dict] = []
    chosen_indexes: Set[int] = set()
    remaining_indexes: Set[int] = set(range(len(split_data)))

    # sample valid samples for each sample size
    for sample_size in tqdm(sample_size_list, total=len(sample_size_list), desc=f"Sampling {dataset}/{split}"):

        downloaded_count: int = 0
        download_bar_desc: str = f"Downloading for size: {sample_size} " + "(Newly downloaded: {})"
        download_pbar = tqdm(total=sample_size - len(chosen_samples), desc=download_bar_desc.format(0))
        while len(chosen_samples) < sample_size and len(remaining_indexes) > 0:
            # Sample indexes for remaining index list.
            num_to_sample = min(sample_size - len(chosen_samples), len(remaining_indexes))
            newly_sampled_indexes = np.random.choice(list(remaining_indexes), size=num_to_sample, replace=False)
            remaining_indexes -= set(newly_sampled_indexes)

            # Check the validation of the newly sampled one, download files if valid.
            for idx in newly_sampled_indexes:
                sample = split_data[idx]
                titles_by_type: Dict[str, List[str]] = get_titles_to_download(sample)
                success = try_download(
                    titles_by_type, title2qid, title_to_location, title_to_validation, document_dir,
                )

                if success is True:
                    chosen_samples.append(sample)
                    chosen_indexes.add(idx)
                    downloaded_count += 1
                    download_pbar.set_description_str(desc=download_bar_desc.format(downloaded_count))
                    download_pbar.update(1)

                if downloaded_count >= cache_every_updates:
                    dump_caches(document_dir, title_to_location, title_to_validation)
                    downloaded_count = 0
        download_pbar.close()

        # Save the chosen samples and name it with the number of samples it contains till now.
        assert len(chosen_samples) == len(chosen_indexes), "Samples could not be duplicated in one set!"
        dump_path: str = split_path_func(sample_num=len(chosen_samples))
        dump_to_jsonlines(dump_path, chosen_samples)

    dump_caches(document_dir, title_to_location, title_to_validation)
    return
```

**描述:**

*   `sample_datasets` 函数现在使用 `try_download` 函数来下载文档，并更新进度条的描述。

**需要注意的是:**

*   您需要在 `wikipedia.py` 和 `wikidata.py` 中实现异步下载函数 `download_title_async`。
*   确保安装了 `aiohttp` 库：`pip install aiohttp`

**总结:**

这些改进主要集中在：

*   **并发下载:** 使用 `asyncio` 和 `aiohttp` 提高下载速度。
*   **代码结构:** 简化代码结构，提高可读性。
*   **错误处理:** 增加错误处理，提高代码的健壮性。

希望这些改进能帮助您更好地处理数据下载任务！
