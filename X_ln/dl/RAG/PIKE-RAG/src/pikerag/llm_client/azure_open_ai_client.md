Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\llm_client\azure_open_ai_client.py`

Okay, restarting the explanation and code breakdown.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import re
import time
from typing import Callable, List, Literal, Optional, Union

import openai
from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion import ChatCompletion
from pickledb import PickleDB

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger


def get_azure_active_directory_token_provider() -> Callable[[], str]:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    return token_provider


def parse_wait_time_from_error(error: openai.RateLimitError) -> Optional[int]:
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile('(?<!\\\\)\'').sub('\"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        wait_time = int(matches.group(1)) + 3  # NOTE: wait 3 more seconds here.
        return wait_time
    except Exception as e:
        return None


class AzureOpenAIClient(BaseLLMClient):
    NAME = "AzureOpenAIClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        """LLM Communication Client for Azure OpenAI endpoints.

        Args:
            location (str): the file location of the LLM client communication cache. No cache would be created if set to
                None. Defaults to None.
            auto_dump (bool): automatically save the Client's communication cache or not. Defaults to True.
            logger (Logger): client logger. Defaults to None.
            max_attempt (int): Maximum attempt time for LLM requesting. Request would be skipped if max_attempt reached.
                Defaults to 5.
            exponential_backoff_factor (int): Set to enable exponential backoff retry manner. Every time the wait time
                would be `exponential_backoff_factor ^ num_attempt`. Set to None to disable and use the `unit_wait_time`
                manner. Defaults to None.
            unit_wait_time (int): `unit_wait_time` would be used only if the exponential backoff mode is disabled. Every
                time the wait time would be `unit_wait_time * num_attempt`, with seconds (s) as the time unit. Defaults
                to 60.
        """
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("AZURE_OPENAI_API_KEY", None) is None:
            client_configs["azure_ad_token_provider"] = get_azure_active_directory_token_provider()

        self._client = AzureOpenAI(**client_configs)

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        response: ChatCompletion = None
        num_attempt: int = 0
        while num_attempt < self._max_attempt:
            try:
                # TODO: handling the kwargs not passed issue for other Clients
                response = self._client.chat.completions.create(messages=messages, **llm_config)
                break

            except openai.RateLimitError as e:
                self.warning("  Failed due to RateLimitError...")
                # NOTE: mask the line below to keep trying if failed due to RateLimitError.
                # num_attempt += 1
                wait_time = parse_wait_time_from_error(e)
                self._wait(num_attempt, wait_time=wait_time)
                self.warning(f"  Retrying...")

            except openai.BadRequestError as e:
                self.warning(f"  Failed due to Exception: {e}")
                self.warning(f"  Skip this request...")
                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")

        return response

    def _get_content_from_response(self, response: ChatCompletion, messages: List[dict] = None) -> str:
        try:
            content = response.choices[0].message.content
            if content is None:
                finish_reason = response.choices[0].finish_reason
                warning_message = f"Non-Content returned due to {finish_reason}"

                if "content_filter" in finish_reason:
                    for reason, res_dict in response.choices[0].content_filter_results.items():
                        if res_dict["filtered"] is True or res_dict["severity"] != "safe":
                            warning_message += f", '{reason}': {res_dict}"

                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")

                content = ""
        except Exception as e:
            self.warning(f"Try to get content from response but get exception:\n  {e}")
            self.debug(
                f"  Response: {response}\n"
                f"  Last message: {messages}"
            )
            content = ""

        return content

    def close(self):
        super().close()
        self._client.close()


class AzureOpenAIEmbedding(Embeddings):
    def __init__(self, **kwargs) -> None:
        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("AZURE_OPENAI_API_KEY", None) is None:
            client_configs["azure_ad_token_provider"] = get_azure_active_directory_token_provider()

        self._client = AzureOpenAI(**client_configs)

        self._model = kwargs.get("model", "text-embedding-ada-002")

        cache_config = kwargs.get("cache_config", {})
        cache_location = cache_config.get("location", None)
        auto_dump = cache_config.get("auto_dump", True)
        if cache_location is not None:
            self._cache: PickleDB = PickleDB(location=cache_location)
        else:
            self._cache = None

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        if self._cache is None:
            return

        self._cache.set(query, embedding)
        return

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        if self._cache is None:
            return False

        return self._cache.get(query)

    def _get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        while True:
            try:
                response = self._client.embeddings.create(input=texts, model=self._model)
                break

            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if e is not None:
                    print(f"Embedding failed due to RateLimitError, wait for {expected_wait} seconds")
                    time.sleep(expected_wait)
                else:
                    print(f"Embedding failed due to RateLimitError, but failed parsing expected waiting time, wait for 30 seconds")
                    time.sleep(30)

            except Exception as e:
                print(f"Embedding failed due to exception {e}")
                exit(0)

        return response

    def embed_documents(self, texts: List[str], batch_call: bool=False) -> List[List[float]]:
        # NOTE: call self._get_response(texts) would cause RateLimitError, it may due to large batch size.
        if batch_call is True:
            response = self._get_response(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding =  self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding
```

**1. `get_azure_active_directory_token_provider()`:**

```python
def get_azure_active_directory_token_provider() -> Callable[[], str]:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    return token_provider
```

**描述 (Chinese):** 这个函数用于获取 Azure Active Directory (Azure AD) 的 token 提供器。 它使用 `azure.identity` 库中的 `DefaultAzureCredential` 来自动获取凭据，然后使用这些凭据来获取 Azure Cognitive Services 的访问令牌。这个函数返回的是一个可调用对象，每次调用它都会返回一个新的 token。

**使用方法 (Chinese):**  如果你的 Azure OpenAI 服务配置为使用 Azure AD 身份验证，你可以使用这个函数来获取 token 提供器，并将其传递给 `AzureOpenAI` 客户端。这允许你的代码在不需要硬编码 API 密钥的情况下访问 Azure OpenAI 服务。

**示例 (Chinese):**

```python
token_provider = get_azure_active_directory_token_provider()
client = AzureOpenAI(azure_ad_token_provider=token_provider, azure_endpoint="YOUR_AZURE_OPENAI_ENDPOINT")
```

**2. `parse_wait_time_from_error()`:**

```python
def parse_wait_time_from_error(error: openai.RateLimitError) -> Optional[int]:
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile('(?<!\\\\)\'').sub('\"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        wait_time = int(matches.group(1)) + 3  # NOTE: wait 3 more seconds here.
        return wait_time
    except Exception as e:
        return None
```

**描述 (Chinese):** 这个函数用于从 `openai.RateLimitError` 异常中解析出需要等待的时间（秒）。 当你超出 Azure OpenAI 服务的速率限制时，会抛出 `RateLimitError`。 错误消息通常包含建议的等待时间。 这个函数使用正则表达式来提取这个时间，并返回一个整数。 如果解析失败，则返回 `None`。

**使用方法 (Chinese):**  在调用 Azure OpenAI 服务时，你应该捕获 `RateLimitError` 异常。 如果捕获到异常，可以使用此函数来获取等待时间，并在重试之前暂停一段时间。

**示例 (Chinese):**

```python
try:
    response = client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}], model="gpt-35-turbo")
except openai.RateLimitError as e:
    wait_time = parse_wait_time_from_error(e)
    if wait_time:
        print(f"超出速率限制，等待 {wait_time} 秒后重试。")
        time.sleep(wait_time)
    else:
        print("超出速率限制，但无法解析等待时间。")

```

**3. `AzureOpenAIClient` Class:**

```python
class AzureOpenAIClient(BaseLLMClient):
    NAME = "AzureOpenAIClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        """LLM Communication Client for Azure OpenAI endpoints."""
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("AZURE_OPENAI_API_KEY", None) is None:
            client_configs["azure_ad_token_provider"] = get_azure_active_directory_token_provider()

        self._client = AzureOpenAI(**client_configs)

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        response: ChatCompletion = None
        num_attempt: int = 0
        while num_attempt < self._max_attempt:
            try:
                response = self._client.chat.completions.create(messages=messages, **llm_config)
                break

            except openai.RateLimitError as e:
                self.warning("  Failed due to RateLimitError...")
                # NOTE: mask the line below to keep trying if failed due to RateLimitError.
                # num_attempt += 1
                wait_time = parse_wait_time_from_error(e)
                self._wait(num_attempt, wait_time=wait_time)
                self.warning(f"  Retrying...")

            except openai.BadRequestError as e:
                self.warning(f"  Failed due to Exception: {e}")
                self.warning(f"  Skip this request...")
                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")

        return response

    def _get_content_from_response(self, response: ChatCompletion, messages: List[dict] = None) -> str:
        try:
            content = response.choices[0].message.content
            if content is None:
                finish_reason = response.choices[0].finish_reason
                warning_message = f"Non-Content returned due to {finish_reason}"

                if "content_filter" in finish_reason:
                    for reason, res_dict in response.choices[0].content_filter_results.items():
                        if res_dict["filtered"] is True or res_dict["severity"] != "safe":
                            warning_message += f", '{reason}': {res_dict}"

                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response}")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")

                content = ""
        except Exception as e:
            self.warning(f"Try to get content from response but get exception:\n  {e}")
            self.debug(
                f"  Response: {response}\n"
                f"  Last message: {messages}"
            )
            content = ""

        return content

    def close(self):
        super().close()
        self._client.close()

```

**描述 (Chinese):** `AzureOpenAIClient` 类是用于与 Azure OpenAI 服务进行通信的客户端。它继承自 `BaseLLMClient`，并实现了向 Azure OpenAI 发送请求、处理响应和处理速率限制的逻辑。

**主要组件 (Chinese):**

*   **`__init__()`:** 构造函数，用于初始化客户端。它接受位置、自动转储、日志记录器、最大尝试次数、指数退避因子、单位等待时间等参数。 它还配置了 `AzureOpenAI` 客户端，如果未提供 API 密钥，则使用 Azure AD token 提供程序。
*   **`_get_response_with_messages()`:**  发送聊天补全请求到 Azure OpenAI 服务。它使用指数退避重试机制来处理速率限制错误。
*   **`_get_content_from_response()`:** 从响应中提取文本内容。 它还处理内容过滤，并记录任何警告消息。
*   **`close()`:**  关闭客户端连接。

**使用方法 (Chinese):**

```python
client = AzureOpenAIClient(
    location="my_cache.pkl",  # 可选：用于缓存 API 响应的文件路径
    auto_dump=True,           # 可选：是否自动保存缓存
    client_config={           # Azure OpenAI 客户端的配置
        "azure_endpoint": "YOUR_AZURE_OPENAI_ENDPOINT",
        "api_version": "YOUR_API_VERSION",
        "api_key": "YOUR_API_KEY"  # 如果使用 API 密钥进行身份验证
        # 或者，使用 Azure AD 身份验证：
        # "azure_ad_token_provider": get_azure_active_directory_token_provider()
    }
)

messages = [{"role": "user", "content": "你好，世界！"}]
response = client._get_response_with_messages(messages, model="gpt-35-turbo")
content = client._get_content_from_response(response, messages)
print(content)

client.close()
```

**4. `AzureOpenAIEmbedding` Class:**

```python
class AzureOpenAIEmbedding(Embeddings):
    def __init__(self, **kwargs) -> None:
        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("AZURE_OPENAI_API_KEY", None) is None:
            client_configs["azure_ad_token_provider"] = get_azure_active_directory_token_provider()

        self._client = AzureOpenAI(**client_configs)

        self._model = kwargs.get("model", "text-embedding-ada-002")

        cache_config = kwargs.get("cache_config", {})
        cache_location = cache_config.get("location", None)
        auto_dump = cache_config.get("auto_dump", True)
        if cache_location is not None:
            self._cache: PickleDB = PickleDB(location=cache_location)
        else:
            self._cache = None

    def _save_cache(self, query: str, embedding: List[float]) -> None:
        if self._cache is None:
            return

        self._cache.set(query, embedding)
        return

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        if self._cache is None:
            return False

        return self._cache.get(query)

    def _get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        while True:
            try:
                response = self._client.embeddings.create(input=texts, model=self._model)
                break

            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if e is not None:
                    print(f"Embedding failed due to RateLimitError, wait for {expected_wait} seconds")
                    time.sleep(expected_wait)
                else:
                    print(f"Embedding failed due to RateLimitError, but failed parsing expected waiting time, wait for 30 seconds")
                    time.sleep(30)

            except Exception as e:
                print(f"Embedding failed due to exception {e}")
                exit(0)

        return response

    def embed_documents(self, texts: List[str], batch_call: bool=False) -> List[List[float]]:
        # NOTE: call self._get_response(texts) would cause RateLimitError, it may due to large batch size.
        if batch_call is True:
            response = self._get_response(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding =  self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding

```

**描述 (Chinese):** `AzureOpenAIEmbedding` 类是用于生成文本嵌入的类。它继承自 `langchain_core.embeddings.Embeddings`，并使用 Azure OpenAI 嵌入模型来生成嵌入。它还支持缓存嵌入以提高性能。

**主要组件 (Chinese):**

*   **`__init__()`:** 构造函数，用于初始化嵌入客户端。 它配置了 `AzureOpenAI` 客户端，并设置了缓存。
*   **`_save_cache()`:** 将嵌入保存到缓存。
*   **`_get_cache()`:** 从缓存中检索嵌入。
*   **`_get_response()`:**  调用 Azure OpenAI 嵌入 API。 它处理速率限制错误。
*   **`embed_documents()`:**  生成多个文档的嵌入。
*   **`embed_query()`:**  生成单个查询的嵌入。

**使用方法 (Chinese):**

```python
embedding_client = AzureOpenAIEmbedding(
    client_config={
        "azure_endpoint": "YOUR_AZURE_OPENAI_ENDPOINT",
        "api_version": "YOUR_API_VERSION",
        "api_key": "YOUR_API_KEY"  # 如果使用 API 密钥进行身份验证
        # 或者，使用 Azure AD 身份验证：
        # "azure_ad_token_provider": get_azure_active_directory_token_provider()
    },
    model="text-embedding-ada-002",
    cache_config={
        "location": "embedding_cache.pkl",  # 可选：用于缓存嵌入的文件路径
        "auto_dump": True           # 可选：是否自动保存缓存
    }
)

text = "这是一个示例文本。"
embedding = embedding_client.embed_query(text)
print(embedding)

documents = ["文本1", "文本2", "文本3"]
embeddings = embedding_client.embed_documents(documents)
print(embeddings)
```

这个解释包含了代码中所有关键部分的描述，使用方式和简单的示例。  希望这次的解释更清晰！
