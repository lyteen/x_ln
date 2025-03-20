Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\wikipedia.py`

Alright, let's break down this code piece by piece with explanations and examples. This code is designed to download Wikipedia pages in various formats (HTML, PDF, Markdown) and save them to disk. It includes both synchronous and asynchronous implementations for improved efficiency.

**1. Initialization and Setup:**

```python
import asyncio
import requests
from typing import Any, Callable, Dict, List, Tuple
import urllib.parse

import aiohttp
import wikipediaapi
from wikipediaapi import WikipediaPage, WikipediaPageSection

from data_process.utils.io import dump_bytes_to_file, dump_texts_to_file, async_dump_bytes_to_file


WIKI_WIKI = wikipediaapi.Wikipedia('Microsoft Research Asia PIKE-RAG', 'en')
```

**Code Description (中文描述):**

*   **Imports (导入):** 导入必要的库。`asyncio` 用于异步编程, `requests` 用于同步HTTP请求, `aiohttp` 用于异步HTTP请求, `wikipediaapi` 是一个用于访问 Wikipedia API 的 Python 库。`typing` 用于类型提示，`urllib.parse` 用于URL编码。
*   **`WIKI_WIKI` Instance (实例):**  创建 `wikipediaapi.Wikipedia` 的一个实例。这指定了用户代理（"Microsoft Research Asia PIKE-RAG"）和语言（"en" 表示英语）。

**How it's used (如何使用):**

This section sets up the environment and creates a Wikipedia API object. This is the foundation for all subsequent operations.

**2. Synchronous Page Retrieval:**

```python
def get_raw_bytes(url: str) -> bytes:
    with requests.get(url) as response:
        assert response.status_code == 200, (
            "Url must be accessible since the given page is checked to be valid.\n"
            f"Response {response.status_code} to url: {url}"
        )
        ret = response.content
    return ret


def get_html_bytes(page: WikipediaPage) -> bytes:
    url = page.fullurl
    return get_raw_bytes(url)


def get_pdf_bytes(page: WikipediaPage) -> bytes:
    parsed_title = urllib.parse.quote(page.title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{parsed_title}"
    return get_raw_bytes(url)
```

**Code Description (中文描述):**

*   **`get_raw_bytes(url: str) -> bytes`:**  一个辅助函数，用于从给定的 URL 获取原始字节数据。它使用 `requests` 库发起一个 GET 请求，并检查响应状态码是否为 200 (成功)。如果不是，则引发断言错误。
*   **`get_html_bytes(page: WikipediaPage) -> bytes`:**  获取给定 Wikipedia 页面的 HTML 内容。它使用 `page.fullurl` 获取页面的完整 URL，然后调用 `get_raw_bytes` 下载 HTML 内容。
*   **`get_pdf_bytes(page: WikipediaPage) -> bytes`:**  获取给定 Wikipedia 页面的 PDF 版本。 它首先使用 `urllib.parse.quote` 对页面标题进行 URL 编码，然后构造 PDF 文件的 API URL，最后调用 `get_raw_bytes` 下载 PDF 内容。

**How it's used (如何使用):**

These functions provide the core logic for fetching data from Wikipedia using synchronous HTTP requests.  `get_raw_bytes` is the base function, while `get_html_bytes` and `get_pdf_bytes` build upon it to retrieve specific content types.

**Simple Demo (简单演示):**

```python
# Assuming you have a WikipediaPage object named 'page'
# html_content = get_html_bytes(page)
# pdf_content = get_pdf_bytes(page)
# You would then save these to files.
```

**3. Markdown Text Extraction:**

```python
def _extract_markdown_texts(sections: List[WikipediaPageSection], level: int) -> str:
    texts = ""
    for section in sections:
        title_prefix = "#" * level
        texts += f"{title_prefix} **{section.title}**\n\n"
        texts += f"{section.text}\n\n"
        texts += _extract_markdown_texts(section.sections, level + 1)
    return texts


def get_markdown_texts(page: WikipediaPage) -> str:
    texts = f"# **{page.title}**\n\n"
    texts += f"{page.summary.strip()}\n\n"
    texts += _extract_markdown_texts(page.sections, level=2)
    return texts
```

**Code Description (中文描述):**

*   **`_extract_markdown_texts(sections: List[WikipediaPageSection], level: int) -> str`:**  一个递归函数，用于从 Wikipedia 页面的章节中提取 Markdown 文本。 它遍历所有章节，为每个章节标题添加适当的 Markdown 标题前缀（`#` 的数量取决于章节的层级），然后将章节的文本添加到结果字符串中。 递归调用自身来处理嵌套的子章节。
*   **`get_markdown_texts(page: WikipediaPage) -> str`:** 获取给定 Wikipedia 页面的 Markdown 文本。 它首先添加页面的标题和摘要，然后调用 `_extract_markdown_texts` 来提取所有章节的 Markdown 文本。

**How it's used (如何使用):**

These functions convert a Wikipedia page into a Markdown-formatted string. This can be useful for creating easily readable and editable versions of Wikipedia content.

**Simple Demo (简单演示):**

```python
# Assuming you have a WikipediaPage object named 'page'
# markdown_content = get_markdown_texts(page)
# You would then save this to a .md file.
```

**4. File Type Handling:**

```python
FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[WikipediaPage], Any]] = {
    "html": get_html_bytes,
    "pdf": get_pdf_bytes,
    "md": get_markdown_texts,
}


FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": dump_bytes_to_file,
    "pdf": dump_bytes_to_file,
    "md": dump_texts_to_file,
}
```

**Code Description (中文描述):**

*   **`FILE_TYPE_TO_GET_FUNCTION`:**  一个字典，将文件类型（"html", "pdf", "md"）映射到相应的函数，用于获取该文件类型的内容。
*   **`FILE_TYPE_TO_DUMP_FUNCTION`:**  一个字典，将文件类型映射到相应的函数，用于将该文件类型的内容保存到磁盘。 这些 `dump_bytes_to_file` and `dump_texts_to_file` 函数假定是从 `data_process.utils.io` 模块导入的。

**How it's used (如何使用):**

These dictionaries provide a centralized way to manage the different file types and the functions used to retrieve and save them.  This makes the code more modular and easier to extend.

**5. Synchronous Download Function:**

```python
def download_all_titles(titles: List[str], dump_path_by_type_list: List[Dict[str, str]]) -> Tuple[bool, Dict[str, bool]]:
    """Try to download all wikipedia pages with the given titles. Download all or nothing, no partial downloads.

    Args:
        titles (List[str]):
        dump_path_by_type_list (List[Dict[str, str]]):

    Returns:
        bool: True if all wikipedia pages can be found and downloaded successfully, else False.
        Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is whether the
            page exists or not.
    """
    pages: List[WikipediaPage] = []
    title_valid: Dict[str, bool] = {}
    for title in titles:
        page = WIKI_WIKI.page(title)
        if not page.exists():
            title_valid[title] = False
            return False, title_valid
        pages.append(page)
        title_valid[title] = True

    for page, dump_path_by_type in zip(pages, dump_path_by_type_list):
        for filetype, dump_path in dump_path_by_type.items():
            obj = FILE_TYPE_TO_GET_FUNCTION[filetype](page)
            FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)

    return True, title_valid
```

**Code Description (中文描述):**

*   **`download_all_titles(titles: List[str], dump_path_by_type_list: List[Dict[str, str]]) -> Tuple[bool, Dict[str, bool]]`:**  尝试下载所有给定标题的 Wikipedia 页面。 这是一个“全有或全无”的操作：如果任何页面不存在，则函数会立即返回 `False`。

    *   `titles`:  要下载的 Wikipedia 页面标题的列表。
    *   `dump_path_by_type_list`:  一个列表，其中每个元素都是一个字典。 该字典将文件类型（"html", "pdf", "md"）映射到相应的保存路径。
*   该函数首先检查所有页面是否存在。如果所有页面都存在，则它会迭代每个页面和相应的保存路径，并使用 `FILE_TYPE_TO_GET_FUNCTION` 和 `FILE_TYPE_TO_DUMP_FUNCTION` 下载和保存页面。
*   如果所有页面都成功下载，则函数返回 `True`；否则，返回 `False`。`title_valid` 记录每个标题是否有效。

**How it's used (如何使用):**

This function orchestrates the entire synchronous download process. It takes a list of titles and a list of dump path configurations, checks for page existence, retrieves the data in the specified formats, and saves it to disk.

**Simple Demo (简单演示):**

```python
# titles = ["Albert Einstein", "Marie Curie"]
# dump_paths = [
#     {"html": "einstein.html", "pdf": "einstein.pdf"},
#     {"html": "curie.html", "pdf": "curie.pdf"}
# ]
# success, title_valid = download_all_titles(titles, dump_paths)
# if success:
#     print("All pages downloaded successfully!")
# else:
#     print("One or more pages could not be downloaded.")
```

**6. Asynchronous Implementations:**

The code then provides asynchronous versions of the core functions using `asyncio` and `aiohttp`. This allows for concurrent downloads, potentially significantly improving performance.

```python
################################################################################
## Async Implementation Below
################################################################################

async def async_get_raw_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


async def async_get_html_bytes(page: WikipediaPage) -> bytes:
    url = page.fullurl
    return await async_get_raw_bytes(url)


async def async_get_pdf_bytes(page: WikipediaPage) -> bytes:
    parsed_title = urllib.parse.quote(page.title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{parsed_title}"
    return await async_get_raw_bytes(url)


ASYNC_FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[WikipediaPage], Any]] = {
    "html": async_get_html_bytes,
    "pdf": async_get_pdf_bytes,
}


ASYNC_FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": async_dump_bytes_to_file,
    "pdf": async_dump_bytes_to_file,
}


async def async_connect_and_save(page: WikipediaPage, filetype: str, dump_path: str):
    obj = await ASYNC_FILE_TYPE_TO_GET_FUNCTION[filetype](page)
    await ASYNC_FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)


async def async_download_all_titles(
    batch_titles: List[List[str]],
    batch_path_lists: List[List[Dict[str, str]]],
) -> List[Tuple[bool, Dict[str, bool]]]:
    """For each (titles, path_lists) pair in the given batch list, try to download all or nothing, no partial downloads.

    Args:
        batch_titles (List[List[str]]):
        batch_path_lists (List[List[Dict[str, str]]]):

    Returns:
        List[Tuple[bool, Dict[str, bool]]]: Each element is this list contains two items:
            - bool: True if all wikipedia pages can be found and downloaded successfully, else False.
            - Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is
                whether the page exists or not.
    """
    # I tried several methods here, async cannot be applied to check page exists, I only use it in downloading
    valid_batch_titles = []
    valid_batch_lists = []
    title_valid: Dict[str, bool] = {}

    # keep the batch that all pages exist
    for titles, batch_path in zip(batch_titles, batch_path_lists):
        list_all_exist = True
        valid_pages = []
        for title in titles:
            page = WIKI_WIKI.page(title)
            if not page.exists():
                list_all_exist = False
                title_valid[title] = False
                return False, title_valid
            valid_pages.append(page)
            title_valid[title] = True

        if list_all_exist:
            valid_batch_titles.extend(valid_pages)
            valid_batch_lists.extend(batch_path)

    # create download tasks
    download_tasks = []

    for page, dump_path_by_type in zip(valid_batch_titles, valid_batch_lists):
        for filetype, dump_path in dump_path_by_type.items():
            download_tasks.append(
                async_connect_and_save(page, filetype, dump_path)
            )

    await asyncio.gather(*download_tasks)

    return True, title_valid
```

**Code Description (中文描述):**

*   **`async_get_raw_bytes(url: str) -> bytes`:** 异步函数，使用 `aiohttp` 从给定的 URL 获取原始字节数据。`aiohttp.ClientSession` 用于管理连接池，提高效率。
*   **`async_get_html_bytes(page: WikipediaPage) -> bytes`:** 异步获取 HTML 内容。
*   **`async_get_pdf_bytes(page: WikipediaPage) -> bytes`:** 异步获取 PDF 内容。
*   **`ASYNC_FILE_TYPE_TO_GET_FUNCTION`:** 异步版本的文件类型到获取函数映射字典。
*   **`ASYNC_FILE_TYPE_TO_DUMP_FUNCTION`:** 异步版本的文件类型到保存函数映射字典。
*   **`async_connect_and_save(page: WikipediaPage, filetype: str, dump_path: str)`:** 异步函数，用于获取特定文件类型的内容并将其保存到磁盘。
*   **`async_download_all_titles(batch_titles: List[List[str]], batch_path_lists: List[List[Dict[str, str]]]) -> List[Tuple[bool, Dict[str, bool]]]`:** 异步版本的 `download_all_titles` 函数，用于批量下载 Wikipedia 页面。它接受一批标题列表和相应的保存路径列表。
    *   The key idea is still the same: check if all titles are valid using synchronous method. Only proceed to download a batch of titles if it is valid.

**How it's used (如何使用):**

The asynchronous implementation allows for concurrent downloads, making it much faster when downloading multiple pages.  It's crucial to use `await` when calling asynchronous functions.

**Simple Demo (简单演示):**

```python
# async def main():
#     batch_titles = [["Albert Einstein", "Marie Curie"], ["Isaac Newton"]]
#     batch_paths = [
#         [{"html": "einstein.html", "pdf": "einstein.pdf"}, {"html": "curie.html", "pdf": "curie.pdf"}],
#         [{"html": "newton.html", "pdf": "newton.pdf"}]
#     ]
#     results = await async_download_all_titles(batch_titles, batch_paths)
#     print(results)

# if __name__ == "__main__":
#     asyncio.run(main())
```

**Key Improvements and Explanations:**

*   **Asynchronous Operations:** The use of `asyncio` and `aiohttp` enables concurrent downloads, significantly speeding up the process.
*   **Error Handling:** The code includes basic error handling (checking for HTTP status code 200), but more robust error handling could be added (e.g., handling network errors, timeouts).
*   **Modularity:** The code is well-organized into functions, making it easier to understand and maintain.  The use of dictionaries to map file types to functions improves modularity.
*   **Type Hints:** Type hints (`typing` module) improve code readability and help catch errors early.
*   **"All or Nothing" Logic:** The "all or nothing" approach ensures data consistency.  If any page fails to download, the entire process is aborted.
*   **Batch Processing:** The asynchronous version is designed to handle batches of titles, making it more efficient for large-scale downloads.

This comprehensive explanation should give you a solid understanding of the code and how to use it. Remember to install the necessary libraries (`wikipediaapi`, `requests`, `aiohttp`) before running the code.
