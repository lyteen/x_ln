Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\wikidata.py`

Okay, let's break down the code piece by piece with explanations in Chinese and example usage.

**1. 导入必要的库 (Import Necessary Libraries):**

```python
import asyncio
import requests
from requests import Response
from typing import Any, Callable, Dict, List, Tuple

import aiohttp
from bs4 import BeautifulSoup

from data_process.utils.io import dump_bytes_to_file, dump_texts_to_file, async_dump_bytes_to_file
```

*   **描述:**  这段代码导入了 Python 中常用的库。`asyncio` 用于异步编程，`requests` 用于发送 HTTP 请求，`BeautifulSoup` 用于解析 HTML，`aiohttp` 用于异步 HTTP 请求。 `typing` 提供了类型提示。`data_process.utils.io` 包含了将数据写入文件的函数。

    *   `asyncio`:  异步编程，允许并发执行多个任务。
    *   `requests`:  发送同步 HTTP 请求，用于获取网页内容。
    *   `aiohttp`: 发送异步 HTTP 请求，更加高效，用于并发下载。
    *   `BeautifulSoup`: 解析 HTML 和 XML 文档，提取所需信息。
    *   `data_process.utils.io`:  自定义的 I/O 模块，包含文件写入函数。

**2.  `parse_contents` 函数 (Parse Contents Function):**

```python
def parse_contents(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')

    title: str = soup.find("span", {"class": "wikibase-title-label"}).get_text()
    description: List[str] = []
    heading_desc: str = soup.find('span', class_='wikibase-descriptionview-text').get_text()
    description.append(heading_desc)
    extra_descriptions = soup.find_all('li', class_='wikibase-entitytermsview-aliases-alias')
    for desc in extra_descriptions:
        description.append(desc.get_text())

    statements: dict[str: List[str]] = {}
    statement_groups = soup.find_all(class_='wikibase-statementgroupview')
    for group in statement_groups:
        property_label = group.find(class_='wikibase-statementgroupview-property-label')
        if property_label:
            property_text = property_label.get_text().strip()
            values = []
            value_elements = group.find_all(class_='wikibase-snakview-value')
            # filter references here, I think its helpless
            for value in value_elements:
                if not value.find_parent(class_='wikibase-statementview-references-container'):
                    values.append(value.get_text().strip())
            statements[property_text] = values

    return {
        'title': title,
        'description': description,
        'statements': statements
    }
```

*   **描述:** 这个函数接收 HTML 内容作为输入，使用 BeautifulSoup 解析 HTML，并提取页面的标题、描述和语句（Statements）。它将提取的信息组织成一个字典返回。

    *   `BeautifulSoup(html_content, 'html.parser')`:  使用 BeautifulSoup 解析 HTML 内容。
    *   `soup.find(...)`:  在 HTML 中查找特定标签和属性的元素，提取文本内容。
    *   提取标题 (`title`)、描述 (`description`) 和语句 (`statements`)，并存储在字典中。

*   **用法示例:**

```python
# 假设 html_content 是从网页获取的 HTML 字符串
html_content = """
<span class="wikibase-title-label">Example Title</span>
<span class="wikibase-descriptionview-text">Example Description</span>
<li class="wikibase-entitytermsview-aliases-alias">Alias 1</li>
<div class="wikibase-statementgroupview">
    <div class="wikibase-statementgroupview-property-label">Property 1</div>
    <div class="wikibase-snakview-value">Value 1</div>
</div>
"""
parsed_data = parse_contents(html_content)
print(parsed_data)
# 输出: {'title': 'Example Title', 'description': ['Example Description', 'Alias 1'], 'statements': {'Property 1': ['Value 1']}}
```

**3. `contents_to_markdown_string` 函数 (Contents to Markdown String Function):**

```python
def contents_to_markdown_string(contents: dict) -> str:
    markdown_content = f"# {contents['title']}\n\n"

    for desc_idx in range(len(contents['description'])):
        if desc_idx != len(contents['description']) - 1:
            markdown_content += f"{contents['description'][desc_idx]} | "
        else:
            markdown_content += f"{contents['description'][desc_idx]}\n\n"

    markdown_content += '## **Statements**\n\n'
    for key, values in contents['statements'].items():
        markdown_content += f'### {key}:\n'
        for value in values:
            markdown_content += f'- {value}\n'

    return markdown_content
```

*   **描述:** 这个函数接收 `parse_contents` 函数返回的字典，并将其转换为 Markdown 格式的字符串。

    *   使用 f-string 格式化字符串，将标题、描述和语句插入到 Markdown 文本中。
    *   `#` 表示一级标题，`##` 表示二级标题，`-` 表示列表项。

*   **用法示例:**

```python
contents = {'title': 'Example Title', 'description': ['Example Description', 'Alias 1'], 'statements': {'Property 1': ['Value 1']}}
markdown_text = contents_to_markdown_string(contents)
print(markdown_text)
# 输出:
# # Example Title
#
# Example Description | Alias 1
#
# ## **Statements**
#
# ### Property 1:
# - Value 1
```

**4. `get_html_bytes`, `get_pdf_bytes`, `get_markdown_texts` 函数 (Get Data Functions):**

```python
def get_html_bytes(response: Response) -> bytes:
    return response.content


def get_pdf_bytes(response: Response) -> bytes:
    # The `url` ends with ".json".
    qid = response.url.split("/")[-1].replace(".json", "")
    url = f"https://www.wikidata.org/api/rest_v1/page/pdf/{qid}"
    with requests.get(url) as response:
        assert response.status_code == 200, "Url must be accessible since the given qid is checked to be valid."
        ret = response.content
    return ret


def get_markdown_texts(response: Response) -> str:
    contents = parse_contents(response.text)
    texts = contents_to_markdown_string(contents)
    return texts
```

*   **描述:** 这些函数分别从 `requests.Response` 对象中提取不同类型的数据。 `get_html_bytes` 提取 HTML 内容的字节，`get_pdf_bytes` 从 Wikidata API 获取 PDF 内容的字节，`get_markdown_texts` 将 HTML 内容解析成 Markdown 文本。

    *   `response.content`:  获取 HTTP 响应的内容，以字节形式返回。
    *   `response.text`:  获取 HTTP 响应的内容，以字符串形式返回。
    *   `requests.get(url)`:  发送 HTTP GET 请求，获取指定 URL 的内容。

**5.  `FILE_TYPE_TO_GET_FUNCTION` 和 `FILE_TYPE_TO_DUMP_FUNCTION` 字典 (File Type Function Mappings):**

```python
FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[Response], Any]] = {
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

*   **描述:** 这些字典将文件类型（如 "html", "pdf", "md"）映射到相应的获取数据和写入文件的函数。 这样可以根据文件类型动态选择要执行的操作。

    *   `FILE_TYPE_TO_GET_FUNCTION`:  将文件类型映射到从 HTTP 响应中提取数据的函数。
    *   `FILE_TYPE_TO_DUMP_FUNCTION`:  将文件类型映射到将数据写入文件的函数。
    *   `Callable[[Response], Any]`:  表示一个接收 `Response` 对象作为参数并返回任意类型值的函数。
    *   `Callable[[Any, str], None]`:  表示一个接收任意类型值和字符串作为参数且不返回任何值的函数。

**6. `download_all_titles` 函数 (Download All Titles Function):**

```python
def download_all_titles(
    titles: List[str], dump_path_by_type_list: List[Dict[str, str]], title2qid: Dict[str, str],
) -> Tuple[bool, Dict[str, bool]]:
    """Try to download all wikidata pages with the given titles. Download all or nothing, no partial downloads.

    Args:
        titles (List[str]):
        dump_path_by_type_list (List[Dict[str, str]]):

    Returns:
        bool: True if all wikidata pages can be found and downloaded successfully, else False.
        Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is whether the
            page exists or not.
    """
    responses: List[Response] = []
    title_valid: Dict[str, bool] = {}
    for title in titles:
        qid = title2qid.get(title, None)
        if qid is None:
            title_valid[title] = False
            return False, title_valid

        url = f"https://www.wikidata.org/wiki/{qid}"
        response = requests.get(url)
        if response.status_code != 200:
            title_valid[title] = False
            return False, title_valid
        responses.append(response)
        title_valid[title] = True

    for response, dump_path_by_type in zip(responses, dump_path_by_type_list):
        for filetype, dump_path in dump_path_by_type.items():
            obj = FILE_TYPE_TO_GET_FUNCTION[filetype](response)
            FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)

    return True, title_valid
```

*   **描述:** 这个函数尝试下载给定标题列表中的所有 Wikidata 页面。 如果任何一个页面无法下载，则整个操作失败。

    *   首先，它检查所有标题对应的 QID 是否存在。如果不存在，则返回 `False`。
    *   然后，它发送 HTTP 请求获取每个页面的内容。 如果任何请求失败，则返回 `False`。
    *   最后，它根据文件类型将数据写入文件。

*   **参数:**
    *   `titles`:  要下载的 Wikidata 页面的标题列表。
    *   `dump_path_by_type_list`:  每个标题对应的文件类型和保存路径的字典列表。例如 `[{'html': 'path/to/file.html', 'md': 'path/to/file.md'}]`。
    *   `title2qid`:  一个将标题映射到 QID 的字典。

*   **返回值:**
    *   `bool`:  指示是否所有页面都成功下载。
    *   `Dict[str, bool]`:  一个字典，指示每个标题是否有效。

*   **用法示例:**

```python
titles = ['Q1', 'Q2']  # 假设 Q1 和 Q2 是 Wikidata 标题
dump_path_by_type_list = [
    {'html': 'q1.html', 'md': 'q1.md'},
    {'html': 'q2.html', 'md': 'q2.md'}
]
title2qid = {'Q1': 'Q1', 'Q2': 'Q2'}  # 假设 Q1 对应 Q1，Q2 对应 Q2

success, title_valid = download_all_titles(titles, dump_path_by_type_list, title2qid)

if success:
    print("所有页面下载成功!")
else:
    print("下载失败!")
    print(f"无效的标题: {title_valid}")
```

**7. 异步实现 (Async Implementation):**

代码的其余部分实现了异步版本，以提高下载效率。  异步允许并发执行多个 HTTP 请求，而无需等待每个请求完成。 主要区别在于使用了 `aiohttp` 库和 `async` / `await` 关键字。 `async_download_all_titles`  函数与 `download_all_titles` 函数的功能相同，但使用异步方式实现。 其内部函数和逻辑也类似，只是使用了 `aiohttp` 和 `asyncio` 库进行异步操作。

*   **`async_get_html_bytes` 和 `async_get_pdf_bytes`:** 异步获取 HTML 和 PDF 内容。
*   **`ASYNC_FILE_TYPE_TO_GET_FUNCTION` 和 `ASYNC_FILE_TYPE_TO_DUMP_FUNCTION`:** 异步函数映射。
*   **`async_fetch_response`:**  异步获取 HTTP 响应。
*   **`async_download_all_titles`:**  异步下载所有标题对应的页面。

*   **用法示例:**

```python
async def main():
    titles = ['Q1', 'Q2']
    dump_path_by_type_list = [
        {'html': 'q1_async.html', 'md': 'q1_async.md'},
        {'html': 'q2_async.html', 'md': 'q2_async.md'}
    ]
    title2qid = {'Q1': 'Q1', 'Q2': 'Q2'}

    success, title_valid = await async_download_all_titles(titles, dump_path_by_type_list, title2qid)

    if success:
        print("所有页面异步下载成功!")
    else:
        print("异步下载失败!")
        print(f"无效的标题: {title_valid}")

if __name__ == "__main__":
    asyncio.run(main())
```

**总结 (Summary):**

这段代码提供了一个从 Wikidata 下载页面并将其保存为不同文件类型的工具。 它既支持同步下载，也支持异步下载，异步下载速度更快，效率更高。 代码结构清晰，易于理解和使用。

希望这个详细的解释能够帮助你理解这段代码。
