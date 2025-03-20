Lan: `py` From`dl/RAG/PIKE-RAG/src\ragflow\deepdoc\parser\json_parser.py`

好的，这是改进后的代码，附带中文描述和示例：

```python
# -*- coding: utf-8 -*-
#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# 本代码主要参考了以下文档，并进行了适应性修改
# from https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/json.py

import json
from typing import Any, List, Dict

from rag.nlp import find_codec


class RAGFlowJsonParser:
    """
    RAGFlow JSON 解析器，用于将 JSON 数据分割成较小的块，以便于处理。

    主要功能：
    - 将 JSON 数据分割成多个块，每个块的大小在可配置的范围内。
    - 能够处理嵌套的 JSON 对象和列表。
    - 提供将列表转换为字典的预处理步骤。

    """

    def __init__(
        self, max_chunk_size: int = 2000, min_chunk_size: int | None = None
    ):
        """
        初始化 RAGFlowJsonParser。

        Args:
            max_chunk_size: 每个块的最大大小（字符数）。
            min_chunk_size: 每个块的最小大小（字符数）。
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size * 2  # *2 为了适应unicode 吧。
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(max_chunk_size - 200, 50)
        )

    def __call__(self, binary: bytes) -> List[str]:
        """
        将二进制数据解析为 JSON 块。

        Args:
            binary: 包含 JSON 数据的二进制数据。

        Returns:
            JSON 块的列表，每个块都是一个 JSON 字符串。
        """
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
        json_data = json.loads(txt)
        chunks = self.split_json(json_data, True)
        sections = [json.dumps(line, ensure_ascii=False) for line in chunks if line]
        return sections

    @staticmethod
    def _json_size(data: Dict[str, Any]) -> int:
        """
        计算 JSON 对象序列化后的字符串长度。

        Args:
            data: JSON 对象。

        Returns:
            JSON 对象序列化后的字符串长度。
        """
        return len(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def _set_nested_dict(d: Dict[str, Any], path: List[str], value: Any) -> None:
        """
        根据给定的路径在嵌套字典中设置值。

        Args:
            d: 要修改的字典。
            path: 键的路径（例如，["a", "b", "c"]）。
            value: 要设置的值。
        """
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        """
        将列表转换为字典的预处理步骤。

        这对于处理包含列表的 JSON 数据非常有用，因为它可以更容易地将其分割成块。

        Args:
            data: 要处理的数据。

        Returns:
            如果输入是字典，则返回处理后的字典。如果输入是列表，则返回转换为字典后的结果。否则，返回原始数据。
        """
        if isinstance(data, dict):
            # 处理字典中的每个键值对
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        elif isinstance(data, list):
            # 将列表转换为以索引为键的字典
            return {
                str(i): self._list_to_dict_preprocessing(item)
                for i, item in enumerate(data)
            }
        else:
            # 基本情况：项目既不是字典也不是列表，因此不作更改地返回它
            return data

    def _json_split(
        self,
        data: Any,
        current_path: List[str] | None,
        chunks: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]]:
        """
        将 JSON 数据分割成最大大小的字典，同时保持其结构。

        Args:
            data: 要分割的 JSON 数据。
            current_path: 当前路径（用于跟踪嵌套对象）。
            chunks: 已经创建的块的列表。

        Returns:
            JSON 块的列表。
        """
        current_path = current_path or []
        chunks = chunks or [{}]
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                    # 将项目添加到当前块
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                        # 块足够大，开始一个新的块
                        chunks.append({})

                    # 迭代
                    self._json_split(value, new_path, chunks)
        else:
            # 处理单个项目
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(
        self,
        json_data: Any,
        convert_lists: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        将 JSON 分割成 JSON 块的列表。

        Args:
            json_data: 要分割的 JSON 数据。
            convert_lists: 是否将列表转换为字典。

        Returns:
            JSON 块的列表。
        """

        if convert_lists:
            preprocessed_data = self._list_to_dict_preprocessing(json_data)
            chunks = self._json_split(preprocessed_data, None, None)
        else:
            chunks = self._json_split(json_data, None, None)

        # 删除最后一个空块
        if chunks and not chunks[-1]:
            chunks.pop()
        return chunks

    def split_text(
        self,
        json_data: Dict[str, Any],
        convert_lists: bool = False,
        ensure_ascii: bool = True,
    ) -> List[str]:
        """
        将 JSON 分割成 JSON 格式化字符串的列表。

        Args:
            json_data: 要分割的 JSON 数据。
            convert_lists: 是否将列表转换为字典。
            ensure_ascii: 是否确保输出为 ASCII。

        Returns:
            JSON 格式化字符串的列表。
        """

        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)

        # 转换为字符串
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]


# 示例用法
if __name__ == "__main__":
    # 创建一个 RAGFlowJsonParser 实例
    parser = RAGFlowJsonParser(max_chunk_size=500, min_chunk_size=200)

    # 模拟一些 JSON 数据
    data = {
        "name": "Example Document",
        "description": "This is a long description of the document.",
        "keywords": ["example", "document", "json", "splitting"],
        "authors": [
            {"name": "John Doe", "affiliation": "University A"},
            {"name": "Jane Smith", "affiliation": "Company B"},
        ],
        "content": "This is the main content of the document.  It is very long and needs to be split into smaller chunks." * 10,
    }

    # 将 JSON 数据分割成块
    json_strings = parser.split_text(data, convert_lists=True)

    # 打印每个块
    for i, chunk in enumerate(json_strings):
        print(f"Chunk {i + 1}:")
        print(chunk)
        print("---")

    binary_data = json.dumps(data).encode('utf-8')
    sections = parser(binary_data)

    # 打印每个块
    for i, chunk in enumerate(sections):
        print(f"Section {i + 1}:")
        print(chunk)
        print("---")

```

**代码解释:**

1.  **类定义 (Class Definition):**  `RAGFlowJsonParser` 类包含了所有 JSON 解析和分割的逻辑。
2.  **初始化 (\_\_init\_\_):**  构造函数允许设置 `max_chunk_size` 和 `min_chunk_size`，用于控制分割后的 JSON 块的大小。
3.  **`__call__` 方法:** 接受二进制数据，自动检测编码，解码为文本，加载 JSON，分割 JSON，并返回分割后的 JSON 字符串列表。
4.  **`_json_size` 方法:** 静态方法，用于计算 JSON 对象序列化后的字符串长度，用于判断是否超过了最大块大小。
5.  **`_set_nested_dict` 方法:** 静态方法，用于在嵌套的字典中设置值，根据给定的路径（键的列表）。
6.  **`_list_to_dict_preprocessing` 方法:**  将 JSON 数据中的列表转换为字典，键为列表的索引。 这对于处理包含列表的 JSON 数据，并保持数据的结构非常有用。
7.  **`_json_split` 方法:**  递归地分割 JSON 数据，根据 `max_chunk_size` 限制每个块的大小。