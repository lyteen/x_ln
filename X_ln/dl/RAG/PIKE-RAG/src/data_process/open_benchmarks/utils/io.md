Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\io.py`

好的，重新开始。

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
from typing import Any, List

import aiofiles
import jsonlines

# 1. dump_bytes_to_file
def dump_bytes_to_file(obj: bytes, filepath: str) -> None:
    """
    将字节数据写入文件。
    Writes bytes data to a file.
    """
    with open(filepath, "wb") as writer:
        writer.write(obj)
    return

# 示例/Example:
# data = b"This is some binary data."
# dump_bytes_to_file(data, "my_binary_file.bin") # 把字节数据写入my_binary_file.bin 文件

# 描述/Description:
# 此函数接收一个字节对象和一个文件路径，并将该字节对象以二进制模式写入指定的文件。
# This function takes a bytes object and a file path, and writes the bytes object to the specified file in binary mode.

# 2. async_dump_bytes_to_file
async def async_dump_bytes_to_file(data: bytes, filepath: str) -> None:
    """
    异步地将字节数据写入文件。
    Asynchronously writes bytes data to a file.
    """
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(data)
    return

# 示例/Example:
# import asyncio
# async def main():
#     data = b"This is some binary data to be written asynchronously."
#     await async_dump_bytes_to_file(data, "my_async_binary_file.bin")

# asyncio.run(main())  #异步写入数据到 my_async_binary_file.bin

# 描述/Description:
# 此函数与 `dump_bytes_to_file` 类似，但使用 `aiofiles` 库异步地执行文件写入操作。这对于在异步环境中避免阻塞很有用。
# This function is similar to `dump_bytes_to_file`, but uses the `aiofiles` library to perform the file write operation asynchronously. This is useful for avoiding blocking in asynchronous environments.

# 3. dump_texts_to_file
def dump_texts_to_file(texts: str, filepath: str) -> None:
    """
    将文本数据写入文件。
    Writes text data to a file.
    """
    with open(filepath, "w", encoding="utf-8") as writer:
        writer.write(texts)
    return

# 示例/Example:
# text = "This is some text data."
# dump_texts_to_file(text, "my_text_file.txt") # 把文本数据写入my_text_file.txt

# 描述/Description:
# 此函数接收一个字符串和一个文件路径，并将该字符串以 UTF-8 编码写入指定的文件。
# This function takes a string and a file path, and writes the string to the specified file using UTF-8 encoding.

# 4. load_from_json_file
def load_from_json_file(filepath: str) -> Any:
    """
    从JSON文件加载数据。
    Loads data from a JSON file.
    """
    object = None
    if os.path.exists(filepath):
        with open(filepath, "r") as fin:
            object = json.load(fin)
    return object

# 示例/Example:
# data = load_from_json_file("my_data.json") # 从 my_data.json加载数据
# if data:
#     print(data)

# 描述/Description:
# 此函数接收一个文件路径，并尝试从该文件加载 JSON 数据。 如果文件存在，则使用 `json.load` 解析文件内容并返回 Python 对象。如果文件不存在，则返回 `None`。
# This function takes a file path and attempts to load JSON data from that file. If the file exists, it uses `json.load` to parse the file contents and returns a Python object. If the file does not exist, it returns `None`.

# 5. dump_to_json_file
def dump_to_json_file(filepath: str, object: Any) -> None:
    """
    将数据以JSON格式写入文件。
    Writes data to a file in JSON format.
    """
    with open(filepath, "w") as fout:
        json.dump(object, fout)
    return

# 示例/Example:
# my_data = {"name": "John", "age": 30}
# dump_to_json_file("my_data.json", my_data) # 将my_data 以JSON格式写入 my_data.json

# 描述/Description:
# 此函数接收一个文件路径和一个 Python 对象，并将该对象以 JSON 格式写入指定的文件。
# This function takes a file path and a Python object, and writes the object to the specified file in JSON format.

# 6. load_from_jsonlines
def load_from_jsonlines(filepath: str) -> List:
    """
    从JSON Lines文件加载数据。
    Loads data from a JSON Lines file.
    """
    with jsonlines.open(filepath, "r") as reader:
        data = [d for d in reader]
    return data

# 示例/Example:
# data = load_from_jsonlines("my_data.jsonl") # 从 my_data.jsonl 文件加载数据
# print(data)

# 描述/Description:
# 此函数接收一个文件路径，并从该文件加载 JSON Lines 数据。JSON Lines 格式每行包含一个 JSON 对象。该函数返回一个包含所有 JSON 对象的列表。
# This function takes a file path and loads JSON Lines data from that file. The JSON Lines format contains one JSON object per line. The function returns a list containing all JSON objects.

# 7. dump_to_jsonlines
def dump_to_jsonlines(filepath: str, objs: Any) -> None:
    """
    将数据以JSON Lines格式写入文件。
    Writes data to a file in JSON Lines format.
    """
    with jsonlines.open(filepath, "w") as writer:
        writer.write_all(objs)
    return

# 示例/Example:
# my_data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
# dump_to_jsonlines("my_data.jsonl", my_data) # 将my_data 写入 my_data.jsonl 文件

# 描述/Description:
# 此函数接收一个文件路径和一个 Python 对象列表，并将这些对象以 JSON Lines 格式写入指定的文件。每个对象将写入文件的一行。
# This function takes a file path and a list of Python objects, and writes these objects to the specified file in JSON Lines format. Each object will be written to a line in the file.
```
Each function is explained with:
*   A brief English docstring.
*   A Chinese translation of the docstring.
*   A code example demonstrating usage.
*   A description detailing functionality and usage in both English and Chinese.
