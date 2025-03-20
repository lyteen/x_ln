Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\wikipedia.py`

好的，我们重新来。这次我会把重点放在代码的可读性和可维护性上，并添加更清晰的注释和错误处理。我会提供一些优化的建议，并给出示例代码。

**1. 优化的 `get_raw_bytes` 函数:**

```python
import requests
from typing import Optional

def get_raw_bytes(url: str) -> Optional[bytes]:
    """
    从给定的URL获取原始字节数据。

    Args:
        url (str): 要获取数据的URL。

    Returns:
        Optional[bytes]: 如果成功获取数据，则返回字节数据；如果发生错误，则返回None。
    """
    try:
        response = requests.get(url, timeout=10)  # 添加超时，防止无限等待
        response.raise_for_status()  # 检查HTTP状态码，如果不是200，则抛出异常
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}") # 打印详细错误信息
        return None
```

**描述:**

*   **错误处理:** 使用 `try...except` 块来捕获 `requests` 库可能抛出的异常 (例如，连接错误，超时等)。
*   **HTTP状态码检查:** `response.raise_for_status()`  会检查HTTP状态码。如果状态码不是200 (OK)，它会抛出一个异常，确保我们只处理成功的响应。
*   **超时:** 添加 `timeout=10` 参数到 `requests.get()`  调用，防止程序在请求无响应时无限期地等待。
*   **返回类型:**  使用 `Optional[bytes]`  作为返回类型，表示函数可能返回 `bytes`  或 `None`。
*   **详细错误信息:** 打印详细的错误信息，方便调试。

**示例用法:**

```python
url = "https://www.example.com/some_resource"
data = get_raw_bytes(url)
if data:
    print(f"Successfully fetched {len(data)} bytes from {url}")
else:
    print(f"Failed to fetch data from {url}")
```

---

**2. 优化的 `get_html_bytes` 和 `get_pdf_bytes` 函数:**

```python
from wikipediaapi import WikipediaPage
import urllib.parse

def get_html_bytes(page: WikipediaPage) -> Optional[bytes]:
    """
    从维基百科页面获取HTML内容。

    Args:
        page (WikipediaPage): 维基百科页面对象。

    Returns:
        Optional[bytes]: 如果成功获取HTML，则返回字节数据；如果发生错误，则返回None。
    """
    url = page.fullurl
    return get_raw_bytes(url)

def get_pdf_bytes(page: WikipediaPage) -> Optional[bytes]:
    """
    从维基百科页面获取PDF内容。

    Args:
        page (WikipediaPage): 维基百科页面对象。

    Returns:
        Optional[bytes]: 如果成功获取PDF，则返回字节数据；如果发生错误，则返回None。
    """
    parsed_title = urllib.parse.quote(page.title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{parsed_title}"
    return get_raw_bytes(url)
```

**描述:**

*   这两个函数现在都使用改进的 `get_raw_bytes` 函数来处理实际的HTTP请求和错误处理。
*   **类型提示:** 明确指定了参数和返回值的类型，提高代码可读性。
*   **Optional 返回类型:** 返回类型使用 `Optional[bytes]`。

**示例用法:**

```python
import wikipediaapi

WIKI_WIKI = wikipediaapi.Wikipedia('MyProjectName', 'en')
page = WIKI_WIKI.page("Albert Einstein")
if page.exists():
    html_data = get_html_bytes(page)
    if html_data:
        print(f"Successfully fetched {len(html_data)} bytes of HTML for '{page.title}'")
    else:
        print(f"Failed to fetch HTML for '{page.title}'")

    pdf_data = get_pdf_bytes(page)
    if pdf_data:
        print(f"Successfully fetched PDF for '{page.title}'")
    else:
        print(f"Failed to fetch PDF for '{page.title}'")
else:
    print(f"Page '{page.title}' does not exist.")
```

---

**3. 优化的 `_extract_markdown_texts` 和 `get_markdown_texts` 函数:**

```python
from typing import List
from wikipediaapi import WikipediaPageSection, WikipediaPage

def _extract_markdown_texts(sections: List[WikipediaPageSection], level: int) -> str:
    """
    递归地从维基百科页面章节中提取Markdown文本。

    Args:
        sections (List[WikipediaPageSection]): 维基百科页面章节列表。
        level (int): 标题的层级。

    Returns:
        str: 提取的Markdown文本。
    """
    texts = ""
    for section in sections:
        title_prefix = "#" * level
        texts += f"{title_prefix} **{section.title}**\n\n"
        texts += f"{section.text}\n\n"
        texts += _extract_markdown_texts(section.sections, level + 1)
    return texts


def get_markdown_texts(page: WikipediaPage) -> str:
    """
    从维基百科页面获取Markdown文本。

    Args:
        page (WikipediaPage): 维基百科页面对象。

    Returns:
        str: 提取的Markdown文本。
    """
    texts = f"# **{page.title}**\n\n"
    texts += f"{page.summary.strip()}\n\n"
    texts += _extract_markdown_texts(page.sections, level=2)
    return texts
```

**描述:**

*   **类型提示:** 使用类型提示，使代码更易于阅读和理解。
*   **代码清晰:**  代码结构保持清晰。

**示例用法:**

```python
import wikipediaapi

WIKI_WIKI = wikipediaapi.Wikipedia('MyProjectName', 'en')
page = WIKI_WIKI.page("Albert Einstein")

if page.exists():
    markdown_text = get_markdown_texts(page)
    print(markdown_text[:500])  # 打印前500个字符
else:
    print("Page not found")
```

---

**4. 优化的 `download_all_titles` 函数:**

```python
from typing import List, Dict, Tuple, Callable, Any, Optional
from wikipediaapi import WikipediaPage

from data_process.utils.io import dump_bytes_to_file, dump_texts_to_file

FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[WikipediaPage], Optional[Any]]] = {
    "html": get_html_bytes,
    "pdf": get_pdf_bytes,
    "md": get_markdown_texts,
}

FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": dump_bytes_to_file,
    "pdf": dump_bytes_to_file,
    "md": dump_texts_to_file,
}

def download_all_titles(titles: List[str], dump_path_by_type_list: List[Dict[str, str]]) -> Tuple[bool, Dict[str, bool]]:
    """
    尝试下载所有给定标题的维基百科页面。要么全部下载成功，要么全部不下载。

    Args:
        titles (List[str]): 维基百科页面标题列表。
        dump_path_by_type_list (List[Dict[str, str]]):  每个标题对应的文件类型和存储路径的字典列表。

    Returns:
        Tuple[bool, Dict[str, bool]]:
            - bool: 如果所有页面都存在且成功下载，则返回True，否则返回False。
            - Dict[str, bool]:  键是访问过的标题字符串，值是页面是否存在。
    """
    pages: List[WikipediaPage] = []
    title_valid: Dict[str, bool] = {}
    all_valid = True  # 跟踪所有标题是否有效

    for title in titles:
        page = WIKI_WIKI.page(title)
        title_valid[title] = page.exists()  # 记录页面是否存在
        if not page.exists():
            all_valid = False
            print(f"Page '{title}' does not exist.") # 打印不存在的页面
            break  # 如果任何一个页面不存在，则立即停止

        pages.append(page)

    if not all_valid:
        return False, title_valid  # 如果任何一个页面不存在，则返回False

    for page, dump_path_by_type in zip(pages, dump_path_by_type_list):
        for filetype, dump_path in dump_path_by_type.items():
            get_func = FILE_TYPE_TO_GET_FUNCTION.get(filetype)
            dump_func = FILE_TYPE_TO_DUMP_FUNCTION.get(filetype)

            if not get_func or not dump_func:
                print(f"Unsupported filetype: {filetype}")
                return False, title_valid #返回False, 因为有不支持的filetype

            obj = get_func(page)
            if obj: # only dump when data is not None
              dump_func(obj, dump_path)
            else:
              print(f"Failed to get data for '{page.title}' in {filetype} format.")
              return False, title_valid #返回False，因为获取数据失败

    return True, title_valid
```

**描述:**

*   **更早的验证:** 在下载任何内容之前，首先检查所有页面是否存在。如果任何一个页面不存在，则立即返回，避免不必要的下载尝试。
*   **记录页面存在状态:** 无论页面是否存在，都会记录到 `title_valid` 字典中。
*   **错误处理:** 捕获下载过程中的异常，如果下载失败，则返回 `False`。
*   **清晰的控制流:** 使用 `all_valid` 变量来跟踪所有标题是否有效，使代码更易于理解。
*   **文件类型检查:** 增加文件类型支持性检查，避免运行时错误。
*   **数据有效性检查:** 只有当`get_func`返回非None时才进行dump。

**示例用法:**

```python
titles = ["Albert Einstein", "Marie Curie", "Invalid Page Title"]
dump_paths = [
    {"html": "einstein.html", "md": "einstein.md"},
    {"html": "curie.html", "md": "curie.md"},
    {"html": "invalid.html", "md": "invalid.md"},
]

success, validity = download_all_titles(titles, dump_paths)

if success:
    print("All pages downloaded successfully.")
else:
    print("Failed to download all pages.")

print(f"Validity check: {validity}")
```

---

**5. 异步部分 – 重要提示和修改:**

原代码的异步部分存在一些问题。检查页面是否存在的操作是同步的，这抵消了异步的优势。我将**移除**原代码中试图在异步函数中做同步检查的部分，简化流程。你需要确保在调用异步函数之前，已经通过同步方式验证了页面是否存在。

```python
import asyncio
import aiohttp
from typing import Any, Callable, Dict, List, Tuple, Optional
import urllib.parse

from wikipediaapi import WikipediaPage

from data_process.utils.io import async_dump_bytes_to_file


ASYNC_FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[WikipediaPage], Any]] = {
    "html": async_get_html_bytes,
    "pdf": async_get_pdf_bytes,
}


ASYNC_FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": async_dump_bytes_to_file,
    "pdf": async_dump_bytes_to_file,
}


async def async_get_raw_bytes(url: str) -> bytes:
    """异步地从给定的URL获取原始字节数据."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.read()
    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {e}")
        return None


async def async_get_html_bytes(page: WikipediaPage) -> Optional[bytes]:
    """异步地从维基百科页面获取HTML内容."""
    url = page.fullurl
    return await async_get_raw_bytes(url)


async def async_get_pdf_bytes(page: WikipediaPage) -> Optional[bytes]:
    """异步地从维基百科页面获取PDF内容."""
    parsed_title = urllib.parse.quote(page.title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{parsed_title}"
    return await async_get_raw_bytes(url)



async def async_connect_and_save(page: WikipediaPage, filetype: str, dump_path: str):
    """异步连接到维基百科页面，获取指定类型的数据并保存到文件."""
    get_func = ASYNC_FILE_TYPE_TO_GET_FUNCTION.get(filetype)
    dump_func = ASYNC_FILE_TYPE_TO_DUMP_FUNCTION.get(filetype)

    if not get_func or not dump_func:
        print(f"Unsupported filetype: {filetype}")
        return False

    obj = await get_func(page)
    if obj:
        await dump_func(obj, dump_path)
        return True
    else:
        print(f"Failed to get data for '{page.title}' in {filetype} format.")
        return False


async def async_download_all_titles(
    pages: List[WikipediaPage],
    path_lists: List[Dict[str, str]],
) -> bool:
    """异步地下载多个维基百科页面，如果所有页面都成功下载，则返回True，否则返回False."""
    tasks = []
    for page, dump_path_by_type in zip(pages, path_lists):
        for filetype, dump_path in dump_path_by_type.items():
            tasks.append(async_connect_and_save(page, filetype, dump_path))

    results = await asyncio.gather(*tasks)
    return all(results) # Ensure all downloads were successful


# 示例用法 (需要放在 asyncio.run() 中):
async def main():
    import wikipediaapi
    WIKI_WIKI = wikipediaapi.Wikipedia('MyProjectName', 'en')
    titles = ["Albert Einstein", "Marie Curie"] # 确保页面存在
    pages = [WIKI_WIKI.page(title) for title in titles]
    dump_paths = [
        {"html": "async_einstein.html", "pdf": "async_einstein.pdf"},
        {"html": "async_curie.html", "pdf": "async_curie.pdf"},
    ]

    # **重要: 在调用异步函数之前，先验证页面是否存在!**
    valid_pages = []
    valid_paths = []
    for page, path in zip(pages, dump_paths):
        if page.exists():
            valid_pages.append(page)
            valid_paths.append(path)
        else:
            print(f"Page {page.title} does not exist. Skipping async download.")

    if valid_pages:
        success = await async_download_all_titles(valid_pages, valid_paths)
        if success:
            print("All pages downloaded successfully (async).")
        else:
            print("Failed to download all pages (async).")
    else:
        print("No valid pages to download asynchronously.")

if __name__ == "__main__":
    asyncio.run(main())
```

**重要的修改和解释:**

1.  **移除异步存在性检查:**  完全移除了在异步函数中检查页面存在性的尝试。 这是因为 `wikipediaapi` 库的 `page.exists()` 方法是同步的，如果在异步事件循环中调用它，会阻塞事件循环，抵消异步带来的好处。
2.  **预先同步验证:** 示例代码展示了如何在调用异步函数 `async_download_all_titles` 之前，使用同步方式验证页面是否存在。
3.  **简化 `async_download_all_titles`:** `async_download_all_titles` 现在只负责下载已经确认存在的页面。 它接收一个 `WikipediaPage` 对象列表和一个路径字典列表。
4.  **错误处理:**  `async_get_raw_bytes` 函数现在包含基本的错误处理，如果获取页面失败，则返回 `None`。
5.  **清晰的 `async_connect_and_save`:** `async_connect_and_save` 函数现在返回一个布尔值，表示页面是否成功下载。
6.  **`asyncio.gather` 返回值:** `async_download_all_titles` 函数现在使用 `asyncio.gather(*tasks)` 来并行运行所有下载任务，并检查 `asyncio.gather` 的返回值，以确保所有下载任务都成功完成。
7.  **示例用法:** 示例用法更清晰，并且在 `asyncio.run()`  中运行，这是运行异步代码的正确方式。
8.  **类型提示:**  增加了类型提示，使代码更易于阅读和理解。
9.  **超时:**  在 `async_get_raw_bytes` 中添加了超时，防止请求无限期挂起。

**总结:**

这些改进集中在以下几个方面：

*   **错误处理:**  更健壮的错误处理，可以捕获和报告各种可能的错误。
*   **代码可读性:** 更清晰的代码结构和更多的注释，使代码更易于阅读和理解。
*   **性能:** 避免在异步代码中执行同步操作，提高性能。
*   **可靠性:**  通过预先验证页面是否存在，避免不必要的下载尝试，提高程序的可靠性。
*   **类型提示:** 更多地使用类型提示，增强代码的可维护性。

请记住，异步编程很复杂，需要仔细考虑。务必理解这些修改背后的原因，并在你的实际应用中进行适当的调整。 如果`wikipediaapi`库提供了异步的页面存在性检查，那么可以考虑将存在性检查也放入异步流程中。
