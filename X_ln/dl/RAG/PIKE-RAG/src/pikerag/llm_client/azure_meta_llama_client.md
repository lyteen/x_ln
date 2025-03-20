Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\llm_client\azure_meta_llama_client.py`

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import urllib.request
from typing import List

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger


class AzureMetaLlamaClient(BaseLLMClient):
    """
    AzureMetaLlamaClient 类，用于与 Azure 上部署的 Meta Llama 模型进行交互。
    """
    NAME = "AzureMetaLlamaClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger=None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        """
        初始化 AzureMetaLlamaClient。

        Args:
            location (str, optional): 位置信息. Defaults to None.
            auto_dump (bool, optional): 是否自动转储日志. Defaults to True.
            logger (Logger, optional): 日志记录器. Defaults to None.
            max_attempt (int, optional): 最大尝试次数. Defaults to 5.
            exponential_backoff_factor (int, optional): 指数退避因子. Defaults to None.
            unit_wait_time (int, optional): 单位等待时间（秒）. Defaults to 60.
            **kwargs: 其他参数。
        """
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        self._init_agent(**kwargs)

    def _init_agent(self, **kwargs) -> None:
        """
        初始化客户端代理，从环境变量中获取 endpoint 和 API key。

        Args:
            **kwargs: 其他参数，包括 llama_endpoint_name 和 llama_key_name。
        """
        llama_endpoint_name = kwargs.get("llama_endpoint_name", None)
        if llama_endpoint_name is None:
            llama_endpoint_name = "LLAMA_ENDPOINT"  # 默认环境变量名
        self._endpoint = os.getenv(llama_endpoint_name)
        assert self._endpoint, "LLAMA_ENDPOINT is not set!" # 断言环境变量已设置

        llama_key_name = kwargs.get("llama_key_name", None)
        if llama_key_name is None:
            llama_key_name = "LLAMA_API_KEY" # 默认环境变量名
        self._api_key = os.getenv(llama_key_name)
        assert self._api_key, "LLAMA_API_KEY is not set!" # 断言环境变量已设置

    def _wrap_header(self, **llm_config) -> dict:
        """
        包装请求头，添加 Content-Type, Authorization 和 azureml-model-deployment。

        Args:
            **llm_config: LLM 的配置信息，必须包含 'model' 字段。

        Returns:
            dict: 包装后的请求头。
        """
        assert "model" in llm_config, f"`model` must be provided in `llm_config` to call AzureMetaLlamaClient!"
        header = {
            'Content-Type':'application/json',
            'Authorization':('Bearer '+ self._api_key),
            'azureml-model-deployment': llm_config["model"],
        }
        return header

    def _wrap_body(self, messages: List[dict], **llm_config) -> bytes:
        """
        包装请求体，将 messages 和 llm_config 封装为 JSON 字符串。

        Args:
            messages (List[dict]): 对话消息列表。
            **llm_config: LLM 的配置信息。

        Returns:
            bytes: 包装后的请求体，编码为 bytes。
        """
        data = {
            "input_data": {
                "input_string": messages,
                "parameters": llm_config,
            }
        }
        body = str.encode(json.dumps(data))
        return body

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> bytes:
        """
        发送请求到 Azure Meta Llama 模型，获取响应。

        Args:
            messages (List[dict]): 对话消息列表。
            **llm_config: LLM 的配置信息。

        Returns:
            bytes: 模型的响应，编码为 bytes。
        """
        response: bytes = None
        num_attempt: int = 0
        while num_attempt < self._max_attempt:
            try:
                header = self._wrap_header(**llm_config)
                body = self._wrap_body(messages, **llm_config)
                req = urllib.request.Request(self._endpoint, body, header)
                response = urllib.request.urlopen(req).read()
                break

            except urllib.error.HTTPError as error:
                self.warning(f"  Failed due to Exception: {str(error.code)}")
                print(error.info())
                print(error.read().decode("utf8", 'ignore'))
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")

        return response

    def _get_content_from_response(self, response: bytes, messages: List[dict] = None) -> str:
        """
        从响应中提取内容。

        Args:
            response (bytes): 模型的响应。
            messages (List[dict], optional): 对话消息列表，用于调试. Defaults to None.

        Returns:
            str: 提取出的内容。如果返回内容为空，会记录警告信息。
        """
        try:
            content = json.loads(response.decode('utf-8'))["output"]
            if content is None:
                warning_message = f"Non-Content returned"

                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")

                content = ""
        except:
            content = ""

        return content

# Example Usage (假设环境变量 LLAMA_ENDPOINT 和 LLAMA_API_KEY 已设置)
if __name__ == '__main__':
    # 模拟设置环境变量
    os.environ["LLAMA_ENDPOINT"] = "https://your-llama-endpoint.com/api"
    os.environ["LLAMA_API_KEY"] = "your_llama_api_key"

    # 初始化客户端
    client = AzureMetaLlamaClient(logger=Logger("azure_llama_client"))

    # 构造消息和配置
    messages = [{"role": "user", "content": "你好！"}]
    llm_config = {"model": "llama2-70b", "temperature": 0.7}

    # 获取响应
    response = client._get_response_with_messages(messages, **llm_config)

    # 提取内容
    if response:
        content = client._get_content_from_response(response, messages)
        print(f"模型返回的内容: {content}")
    else:
        print("请求失败，请检查日志。")
```

**代码解释:**

1. **类定义:** `AzureMetaLlamaClient` 类继承自 `BaseLLMClient`，实现了与 Azure 上部署的 Meta Llama 模型交互的逻辑。

2. **初始化 (`__init__`)**:
   - 接收 `location`, `auto_dump`, `logger`, `max_attempt`, `exponential_backoff_factor`, `unit_wait_time` 等参数，这些参数传递给父类 `BaseLLMClient` 的构造函数。
   - 调用 `self._init_agent(**kwargs)` 初始化客户端代理。

3. **`_init_agent(self, **kwargs)`**:
   - 从环境变量中获取 Llama 模型的 endpoint 和 API key。
   - `llama_endpoint_name` 和 `llama_key_name` 允许通过 `kwargs` 指定环境变量名，默认分别是 "LLAMA_ENDPOINT" 和 "LLAMA_API_KEY"。
   - 使用 `os.getenv()` 获取环境变量的值，并使用 `assert` 确保环境变量已设置。

4. **`_wrap_header(self, **llm_config)`**:
   - 构造 HTTP 请求头，包含 `Content-Type`, `Authorization` (使用 API key) 和 `azureml-model-deployment` (指定要使用的模型)。
   - 从 `llm_config` 中获取模型名称，并使用 `assert` 确保 `llm_config` 包含 `model` 字段。

5. **`_wrap_body(self, messages: List[dict], **llm_config)`**:
   - 构造 HTTP 请求体，将 `messages` (对话消息列表) 和 `llm_config` (LLM 配置) 封装成 JSON 字符串。
   - 使用 `json.dumps()` 将 Python 字典转换为 JSON 字符串，并使用 `str.encode()` 将字符串编码为 bytes。

6. **`_get_response_with_messages(self, messages: List[dict], **llm_config)`**:
   - 发送 HTTP 请求到 Azure Meta Llama 模型，并获取响应。
   - 使用 `urllib.request.Request` 构造 HTTP 请求。
   - 使用 `urllib.request.urlopen` 发送请求并读取响应。
   - 实现重试机制，如果请求失败 (抛出 `urllib.error.HTTPError`)，则会重试 `self._max_attempt` 次。
   - 在每次重试之间，使用 `self._wait()` 函数等待一段时间。

7. **`_get_content_from_response(self, response: bytes, messages: List[dict] = None)`**:
   - 从 HTTP 响应中提取内容。
   - 使用 `json.loads()` 将 JSON 字符串解析为 Python 字典，并从字典中提取 `output` 字段作为内容。
   - 如果 `output` 字段为空，则会记录警告信息。

8. **Example Usage (`if __name__ == '__main__':`)**:
   - 提供了一个简单的示例，演示如何使用 `AzureMetaLlamaClient` 类。
   - 首先，模拟设置环境变量 `LLAMA_ENDPOINT` 和 `LLAMA_API_KEY`。
   - 然后，初始化 `AzureMetaLlamaClient` 类。
   - 构造消息和配置，并调用 `_get_response_with_messages()` 函数获取响应。
   - 最后，调用 `_get_content_from_response()` 函数从响应中提取内容，并打印出来。

**如何使用 (Demo):**

1. **设置环境变量:**  需要先设置 `LLAMA_ENDPOINT` 和 `LLAMA_API_KEY` 环境变量。  这些变量指定了 Llama 模型在 Azure 上的 endpoint 和用于身份验证的 API key。

   ```bash
   export LLAMA_ENDPOINT="你的 Llama 模型 endpoint"
   export LLAMA_API_KEY="你的 Llama API key"
   ```

2. **运行代码:**  直接运行 Python 脚本。

   ```bash
   python your_script_name.py
   ```

   将 `your_script_name.py` 替换为包含以上代码的文件名。

3. **查看输出:**  程序会打印从 Llama 模型返回的内容。  如果请求失败，程序会打印错误信息，并建议检查日志。

**关键点:**

- **环境变量:**  代码依赖于环境变量来配置 Llama 模型的 endpoint 和 API key。  确保这些环境变量已正确设置。
- **异常处理:**  代码包含异常处理机制，可以处理 HTTP 请求失败的情况。
- **重试机制:**  代码包含重试机制，可以在请求失败时自动重试。
- **日志记录:**  代码使用 `Logger` 类进行日志记录，可以帮助调试问题。
