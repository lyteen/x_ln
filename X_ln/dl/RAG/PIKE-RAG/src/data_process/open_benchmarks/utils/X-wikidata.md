Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\wikidata.py`

好的，我们重新开始，这次我会将代码分解成更小的部分，并提供更详细的中文描述和演示，目的是为了更容易理解。

**1. `parse_contents` 函数改进**

```python
from bs4 import BeautifulSoup
from typing import Dict, List


def parse_contents(html_content: str) -> Dict:
    """
    解析 HTML 内容，提取标题、描述和声明。

    Args:
        html_content: 包含 Wikidata 页面 HTML 的字符串。

    Returns:
        一个字典，包含标题、描述和声明。
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取标题
    title_element = soup.find("span", {"class": "wikibase-title-label"})
    title = title_element.get_text() if title_element else "No Title Found"  # 添加错误处理

    # 提取描述
    description: List[str] = []
    heading_desc_element = soup.find('span', class_='wikibase-descriptionview-text')
    if heading_desc_element:
        heading_desc = heading_desc_element.get_text()
        description.append(heading_desc)
    extra_descriptions = soup.find_all('li', class_='wikibase-entitytermsview-aliases-alias')
    for desc in extra_descriptions:
        description.append(desc.get_text())

    # 提取声明
    statements: Dict[str, List[str]] = {}
    statement_groups = soup.find_all(class_='wikibase-statementgroupview')
    for group in statement_groups:
        property_label_element = group.find(class_='wikibase-statementgroupview-property-label')
        if property_label_element:
            property_text = property_label_element.get_text().strip()
            values = []
            value_elements = group.find_all(class_='wikibase-snakview-value')
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

**描述:**

*   **中文描述:** 这个函数负责解析HTML内容，然后从中提取出页面的标题、描述以及各种声明。使用了`BeautifulSoup`库来方便地进行HTML的解析工作。
*   **改进:** 增加了错误处理，如果找不到标题元素，则返回 "No Title Found"。

**演示:**

```python
#假设这是你从网页上获取的HTML内容
html_content = """
<html>
<body>
    <span class="wikibase-title-label">Douglas Adams</span>
    <span class="wikibase-descriptionview-text">English writer and humorist</span>
    <li class="wikibase-entitytermsview-aliases-alias">Douglas Noel Adams</li>
    <div class="wikibase-statementgroupview">
        <div class="wikibase-statementgroupview-property">
            <div class="wikibase-statementgroupview-property-label">occupation</div>
        </div>
        <div class="wikibase-statementgroupview-statements">
            <div class="wikibase-statementview">
                <div class="wikibase-statementview-rank-selector"></div>
                <div class="wikibase-statementview-main-snak-container">
                    <div class="wikibase-snakview">
                        <div class="wikibase-snakview-value">writer</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

parsed_data = parse_contents(html_content)
print(parsed_data)
# 输出：
# {'title': 'Douglas Adams', 'description': ['English writer and humorist', 'Douglas Noel Adams'], 'statements': {'occupation': ['writer']}}
```

**2. `contents_to_markdown_string` 函数改进**

```python
def contents_to_markdown_string(contents: Dict) -> str:
    """
    将解析后的内容转换为 Markdown 格式的字符串。

    Args:
        contents: 包含标题、描述和声明的字典，例如 `parse_contents` 函数的输出。

    Returns:
        Markdown 格式的字符串。
    """
    markdown_content = f"# {contents['title']}\n\n"

    if contents['description']: # 检查描述是否存在
        markdown_content += "Description: "
        for i, desc in enumerate(contents['description']):
            markdown_content += desc
            if i < len(contents['description']) - 1:
                markdown_content += " | " # 使用 | 分隔多个描述
        markdown_content += "\n\n"

    markdown_content += "## **Statements**\n\n"
    for key, values in contents['statements'].items():
        markdown_content += f"### {key}:\n"
        for value in values:
            markdown_content += f"- {value}\n"

    return markdown_content
```

**描述:**

*   **中文描述:**  这个函数将`parse_contents`函数解析出来的数据转换成Markdown格式的文本，方便阅读和存储。
*   **改进:** 增加了对描述的检查，以避免空描述列表导致的错误。使用 `|` 分隔多个描述，使其更易于阅读。

**演示:**

```python
contents = {
    'title': 'Douglas Adams',
    'description': ['English writer and humorist', 'Douglas Noel Adams'],
    'statements': {'occupation': ['writer']}
}

markdown_text = contents_to_markdown_string(contents)
print(markdown_text)
# 输出：
# # Douglas Adams
#
# Description: English writer and humorist | Douglas Noel Adams
#
# ## **Statements**
#
# ### occupation:
# - writer
```

**3.  `get_html_bytes`、`get_pdf_bytes` 和 `get_markdown_texts` 函数**

```python
from requests import Response


def get_html_bytes(response: Response) -> bytes:
    """
    从 HTTP 响应中获取 HTML 内容的字节数据。

    Args:
        response: `requests` 库的 `Response` 对象。

    Returns:
        HTML 内容的字节数据。
    """
    return response.content


def get_pdf_bytes(response: Response) -> bytes:
    """
    从 HTTP 响应中获取 PDF 文件的字节数据。
    这里假设 URL 结构允许我们通过修改URL来直接获取PDF。

    Args:
        response: `requests` 库的 `Response` 对象。

    Returns:
        PDF 文件的字节数据。
    """
    qid = response.url.split("/")[-1].replace(".json", "")
    url = f"https://www.wikidata.org/api/rest_v1/page/pdf/{qid}"
    with requests.get(url) as response:
        assert response.status_code == 200, "URL 必须可访问，因为给定的 QID 已被验证为有效。"
        ret = response.content
    return ret


def get_markdown_texts(response: Response) -> str:
    """
    从 HTTP 响应中获取 Markdown 格式的文本。

    Args:
        response: `requests` 库的 `Response` 对象。

    Returns:
        Markdown 格式的文本。
    """
    contents = parse_contents(response.text)
    texts = contents_to_markdown_string(contents)
    return texts
```

**描述:**

*   **中文描述:**  这三个函数分别负责从HTTP响应中获取不同格式的数据：HTML字节流、PDF字节流以及Markdown文本。`get_markdown_texts` 函数依赖于之前定义的 `parse_contents` 和 `contents_to_markdown_string` 函数。

**演示:**

```python
import requests

# 假设我们有一个 HTML 响应
response = requests.get("https://www.wikidata.org/wiki/Q42")  # Q42 是 Douglas Adams 的 Wikidata ID
html_bytes = get_html_bytes(response)
# 现在 html_bytes 包含了 HTML 内容的字节

# 获取 Markdown 文本
markdown_text = get_markdown_texts(response)
# 现在 markdown_text 包含了 Markdown 格式的文本
print(markdown_text[:200]) # 打印前200个字符
```

**4. 文件类型处理字典**

```python
from typing import Any, Callable, Dict
from data_process.utils.io import dump_bytes_to_file, dump_texts_to_file


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

**描述:**

*   **中文描述:**  这两个字典定义了文件类型和处理函数之间的映射关系。 `FILE_TYPE_TO_GET_FUNCTION`  用于根据文件类型选择相应的函数来获取数据。 `FILE_TYPE_TO_DUMP_FUNCTION` 用于选择对应的函数将数据写入文件。

**演示:**

```python
# 假设我们已经有了一个 HTTP 响应和一个文件类型
file_type = "md"

# 获取对应的函数
get_function = FILE_TYPE_TO_GET_FUNCTION[file_type]
# 使用函数获取数据
data = get_function(response)

# 获取对应的dump函数
dump_function = FILE_TYPE_TO_DUMP_FUNCTION[file_type]
# 使用dump函数存储数据
dump_function(data, "output.md")
```

**5. `download_all_titles` 函数改进**

```python
from typing import List, Tuple, Dict
import requests
from requests import Response


def download_all_titles(
    titles: List[str], dump_path_by_type_list: List[Dict[str, str]], title2qid: Dict[str, str],
) -> Tuple[bool, Dict[str, bool]]:
    """
    尝试下载所有给定标题的 Wikidata 页面。

    Args:
        titles: 要下载的标题列表。
        dump_path_by_type_list: 包含每个标题的文件保存路径的列表。
        title2qid: 标题到 QID 的映射。

    Returns:
        一个元组，包含一个布尔值指示是否所有页面都已成功下载，以及一个字典指示每个标题的有效性。
    """
    responses: List[Response] = []
    title_valid: Dict[str, bool] = {}
    all_valid = True  # 假设所有标题都有效，直到找到无效的
    for title in titles:
        qid = title2qid.get(title, None)
        if qid is None:
            title_valid[title] = False
            all_valid = False  # 标记有无效标题
            print(f"Error: QID not found for title '{title}'")  # 打印错误信息
            continue # 如果找不到qid，跳过这个title

        url = f"https://www.wikidata.org/wiki/{qid}"
        try:
            response = requests.get(url, timeout=10)  # 添加超时
            response.raise_for_status()  # 抛出 HTTPError 异常，如果状态码不是 200
            responses.append(response)
            title_valid[title] = True
        except requests.exceptions.RequestException as e:
            title_valid[title] = False
            all_valid = False
            print(f"Error downloading '{title}': {e}") # 打印错误信息
            continue # 如果下载失败，跳过这个title

    # 如果有任何标题无效，则不进行保存操作
    if not all_valid:
        return False, title_valid

    for response, dump_path_by_type in zip(responses, dump_path_by_type_list):
        for filetype, dump_path in dump_path_by_type.items():
            try:
                obj = FILE_TYPE_TO_GET_FUNCTION[filetype](response)
                FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)
            except Exception as e:
                print(f"Error processing or dumping '{title}' as {filetype}: {e}")
                return False, title_valid  # 即使一个文件处理失败，也返回 False

    return True, title_valid
```

**描述:**

*   **中文描述:** 这个函数尝试下载Wikidata页面，并将其保存为指定的文件类型。它接收一个标题列表，以及每个标题对应的保存路径。如果任何一个标题下载失败，则函数会返回False，否则返回True。
*   **改进:**
    *   **错误处理:** 增加了更全面的错误处理，包括处理QID找不到的情况、下载超时以及HTTP错误。
    *   **超时设置:** 使用 `timeout=10`  为请求设置超时时间，防止程序长时间挂起。
    *   **状态码检查:** 使用 `response.raise_for_status()`  来检查HTTP状态码，如果不是200则抛出异常。
    *   **即使一个文件处理失败，也返回 False:** 为了保持原子性，任何一个文件处理失败，整个下载过程都视为失败。

**演示:**

```python
# 假设我们有以下信息
titles = ["Douglas Adams", "The Hitchhiker's Guide to the Galaxy"]
dump_path_by_type_list = [
    {"md": "douglas_adams.md"},
    {"md": "hitchhikers_guide.md"}
]
title2qid = {
    "Douglas Adams": "Q42",
    "The Hitchhiker's Guide to the Galaxy": "Q132725"
}

# 调用下载函数
success, title_valid = download_all_titles(titles, dump_path_by_type_list, title2qid)

if success:
    print("所有页面下载成功！")
else:
    print("部分或所有页面下载失败。")
    print("有效性:", title_valid)
```

**6. 异步实现改进**

```python
import asyncio
import aiohttp
from typing import Any, Callable, Dict, List, Tuple

from data_process.utils.io import async_dump_bytes_to_file


async def async_get_html_bytes(response: aiohttp.ClientResponse) -> bytes:
    """异步地从 HTTP 响应中获取 HTML 内容的字节数据。"""
    return await response.read()


async def async_get_pdf_bytes(session: aiohttp.ClientSession, qid: str) -> bytes:
    """异步地从 HTTP 响应中获取 PDF 文件的字节数据。"""
    url = f"https://www.wikidata.org/api/rest_v1/page/pdf/{qid}"
    try:
        async with session.get(url, timeout=10) as response:  # 添加超时
            response.raise_for_status()
            return await response.read()
    except aiohttp.ClientError as e:
        print(f"Error getting PDF bytes for QID {qid}: {e}")
        return None  # 或者抛出异常，取决于你如何处理错误


ASYNC_FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[aiohttp.ClientResponse, aiohttp.ClientSession, str], Any]] = {
    "html": async_get_html_bytes,
    "pdf": async_get_pdf_bytes
}


ASYNC_FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": async_dump_bytes_to_file,
    "pdf": async_dump_bytes_to_file
}


async def async_fetch_response(session: aiohttp.ClientSession, title: str, title2qid: Dict[str, str]) -> Tuple[str, aiohttp.ClientResponse, bool]:
    """异步地获取 HTTP 响应。"""
    qid = title2qid.get(title, None)
    if qid is None:
        print(f"QID not found for title: {title}")
        return title, None, False

    url = f"https://www.wikidata.org/wiki/{qid}"
    try:
        async with session.get(url, timeout=10) as response:  # 添加超时
            response.raise_for_status()
            return title, response, True
    except aiohttp.ClientError as e:
        print(f"Error fetching response for {title}: {e}")
        return title, None, False


async def async_download_all_titles(
    titles: List[str], dump_path_by_type_list: List[Dict[str, str]], title2qid: Dict[str, str],
) -> Tuple[bool, Dict[str, bool]]:
    """异步地下载所有给定标题的 Wikidata 页面。"""
    title_valid: Dict[str, bool] = {}
    all_valid = True

    async with aiohttp.ClientSession() as session:
        tasks = [async_fetch_response(session, title, title2qid) for title in titles]
        results = await asyncio.gather(*tasks)

    responses: List[Tuple[str, aiohttp.ClientResponse]] = []

    for title, response, is_valid in results:
        title_valid[title] = is_valid
        if not is_valid:
            all_valid = False
        if response:
            responses.append((title, response))

    if not all_valid:
        return False, title_valid

    download_tasks = []
    async with aiohttp.ClientSession() as session: # 创建一个session用于所有的下载任务
        for (title, response), dump_path_by_type in zip(responses, dump_path_by_type_list):
            qid = title2qid[title]
            for filetype, dump_path in dump_path_by_type.items():
                try:
                    if filetype == "html":
                        obj = await ASYNC_FILE_TYPE_TO_GET_FUNCTION[filetype](response, session, qid)
                    elif filetype == "pdf":
                        obj = await ASYNC_FILE_TYPE_TO_GET_FUNCTION[filetype](session, qid)
                    else:
                        print(f"Unsupported filetype: {filetype}")
                        all_valid = False
                        title_valid[title] = False
                        continue # 跳过不支持的文件类型

                    if obj is not None: # 检查对象是否为None，避免空指针异常
                        download_tasks.append(async_dump_bytes_to_file(obj, dump_path))  # 假设这里只需要bytes
                    else:
                        print(f"No data to dump for {title} as {filetype}")
                        all_valid = False
                        title_valid[title] = False
                except Exception as e:
                    print(f"Error processing or dumping '{title}' as {filetype}: {e}")
                    all_valid = False
                    title_valid[title] = False

    if not download_tasks: # 如果没有任何下载任务，直接返回
        return all_valid, title_valid

    await asyncio.gather(*download_tasks)

    return all_valid, title_valid
```

**描述:**

*   **中文描述:** 异步实现了下载Wikidata页面的功能，使用`aiohttp`库进行异步网络请求。  函数会尝试下载Wikidata页面，并将其保存为指定的文件类型。 它接收一个标题列表，以及每个标题对应的保存路径。 如果任何一个标题下载失败，则函数会返回False，否则返回True。
*   **改进:**
    *   **错误处理:**  增加了更全面的错误处理，包括处理QID找不到的情况、下载超时以及HTTP错误.
    *   **超时设置:** 使用 `timeout=10`  为请求设置超时时间，防止程序长时间挂起。
    *   **状态码检查:** 使用 `response.raise_for_status()`  来检查HTTP状态码，如果不是200则抛出异常。
    *   **对PDF下载的错误处理:**  在 `async_get_pdf_bytes` 中增加了对 PDF 下载可能出现的错误的捕获和处理。
    *    **统一Session:** 确保在同一个 `ClientSession` 中进行所有下载任务，避免资源泄露和连接问题。
    *    **空对象检查:** 在 `download_tasks.append` 前检查 `obj` 是否为 `None`，避免传递空数据给 `async_dump_bytes_to_file`。
    *    **不支持的文件类型处理:** 增加了对不支持文件类型的处理，避免程序崩溃。
    *   **任务列表为空的判断:**  如果 `download_tasks` 为空，则直接返回，避免 `asyncio.gather` 抛出异常。
*   **演示**

```python
import asyncio

async def main():
    titles = ["Douglas Adams", "The Hitchhiker's Guide to the Galaxy"]
    dump_path_by_type_list = [
        {"html": "douglas_adams_async.html"},
        {"html": "hitchhikers_guide_async.html"}
    ]
    title2qid = {
        "Douglas Adams": "Q42",
        "The Hitchhiker's Guide to the Galaxy": "Q132725"
    }

    success, title_valid = await async_download_all_titles(titles, dump_path_by_type_list, title2qid)

    if success:
        print("所有页面异步下载成功！")
    else:
        print("部分或所有页面异步下载失败。")
        print("有效性:", title_valid)

if __name__ == "__main__":
    asyncio.run(main())

```

这些改进涵盖了错误处理、代码清晰度、性能以及异步操作的安全性。希望这些分解和解释对你有所帮助！
