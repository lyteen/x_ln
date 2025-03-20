Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\llm_client\azure_open_ai_client.py`

Okay, let's re-approach this with a focus on providing more robust and maintainable code, along with detailed Chinese explanations. I'll break down the code into smaller, manageable chunks, with explanations for each. I will also give a simple demo

**1.  Enhanced `AzureOpenAIClient`:**

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import re
import time
from typing import Callable, List, Optional

import openai
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger


def get_azure_active_directory_token_provider() -> Callable[[], str]:
    """
    从 Azure Active Directory 获取访问令牌。
    Gets an access token from Azure Active Directory.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    return token_provider


def parse_wait_time_from_error(error: openai.RateLimitError) -> Optional[int]:
    """
    从 OpenAI 的速率限制错误中解析等待时间（秒）。
    Parses the wait time (in seconds) from an OpenAI rate limit error.
    """
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile('(?<!\\\\)\'').sub('\"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        if matches: # Ensure matches is not None
            wait_time = int(matches.group(1)) + 3  # NOTE: wait 3 more seconds here.
            return wait_time
        else:
            return None

    except Exception:
        return None


class AzureOpenAIClient(BaseLLMClient):
    """
    用于与 Azure OpenAI 端点通信的 LLM 客户端。
    LLM client for communicating with Azure OpenAI endpoints.
    """

    NAME = "AzureOpenAIClient"

    def __init__(
        self,
        location: str = None,
        auto_dump: bool = True,
        logger: Logger = None,
        max_attempt: int = 5,
        exponential_backoff_factor: int = None,
        unit_wait_time: int = 60,
        **kwargs,
    ) -> None:
        """
        初始化 AzureOpenAIClient。

        Args:
            location (str, optional):  LLM 客户端通信缓存的文件位置. Defaults to None.
            auto_dump (bool, optional):  自动保存缓存. Defaults to True.
            logger (Logger, optional):  日志记录器. Defaults to None.
            max_attempt (int, optional):  最大尝试次数. Defaults to 5.
            exponential_backoff_factor (int, optional): 指数退避因子. Defaults to None.
            unit_wait_time (int, optional):  单位等待时间（秒）. Defaults to 60.
            **kwargs:  传递给 AzureOpenAI 客户端的其他配置.
        """
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        client_configs = kwargs.get("client_config", {})
        if client_configs.get("api_key", None) is None and os.environ.get("AZURE_OPENAI_API_KEY", None) is None:
            client_configs["azure_ad_token_provider"] = get_azure_active_directory_token_provider()

        self._client = AzureOpenAI(**client_configs)

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> ChatCompletion:
        """
        使用给定的消息列表从 Azure OpenAI 获取 ChatCompletion。
        Gets a ChatCompletion from Azure OpenAI using the provided list of messages.
        """
        response: ChatCompletion = None
        num_attempt: int = 0
        while num_attempt < self._max_attempt:
            try:
                response = self._client.chat.completions.create(messages=messages, **llm_config)
                break

            except openai.RateLimitError as e:
                self.warning("  Failed due to RateLimitError...")
                wait_time = parse_wait_time_from_error(e)
                self._wait(num_attempt, wait_time=wait_time)
                self.warning(f"  Retrying...")

            except openai.BadRequestError as e:
                self.warning(f"  Failed due to BadRequestError: {e}")
                self.warning("  Skipping this request...")
                break

            except Exception as e:
                self.warning(f"  Failed due to Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")

        return response

    def _get_content_from_response(self, response: ChatCompletion, messages: List[dict] = None) -> str:
        """
        从 ChatCompletion 响应中提取内容。
        Extracts the content from a ChatCompletion response.
        """
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
            self.debug(f"  Response: {response}\n" f"  Last message: {messages}")
            content = ""

        return content

    def close(self):
        """
        关闭 Azure OpenAI 客户端。
        Closes the Azure OpenAI client.
        """
        super().close()
        self._client.close()
```

**Key Improvements and Explanations:**

*   **Error Handling:**  Includes more specific error handling (e.g., `BadRequestError`) and logs relevant information for debugging.  处理了更多特定的错误类型，并记录了相关信息以方便调试.
*   **Clarity and Comments:**  Added docstrings and comments to explain the purpose of each method and argument. 增加了文档字符串和注释来解释每个方法和参数的用途.
*   **Azure AD Token Provider:**  The code for obtaining the Azure AD token is now in its own function for better organization. 获取 Azure AD 令牌的代码现在位于其自己的函数中，以便更好地组织.
*   **`parse_wait_time_from_error` robust:** add a check to make sure `matches` is not `None` to avoid potential error

**How to use AzureOpenAIClient:**

```python
# Example Usage (示例用法)
if __name__ == "__main__":
    # Assuming you have set environment variables for AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc.
    # 假设您已设置 AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT 等环境变量。

    from pikerag.utils.logger import Logger
    logger = Logger("AzureOpenAIClient_Example")  # Replace with your desired logger setup

    client = AzureOpenAIClient(logger=logger,  max_attempt=3)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    llm_config = {"model": "gpt-35-turbo", "temperature": 0.7} # modify to your deployment name and llm config
    response = client._get_response_with_messages(messages, **llm_config)

    if response:
        content = client._get_content_from_response(response, messages)
        print(f"Response from Azure OpenAI: {content}")
    else:
        print("Failed to get a response from Azure OpenAI.")
    client.close()
```

**Explanation of the Example:**

1.  **Import necessary modules:** imports the needed modules
2.  **Create logger:** creates logger to help recording the activities
3.  **Initialize `AzureOpenAIClient`:**  Creates an instance of the `AzureOpenAIClient` with the necessary configuration, including retry settings and the logger. 初始化 AzureOpenAIClient，包括重试设置和日志记录器.
4.  **Define messages:** Defines the conversation history as a list of dictionaries, following the OpenAI message format. 定义对话历史记录，按照 OpenAI 消息格式使用字典列表表示.
5.  **Define llm_config:** Define llm related configuration, especially the model and temperature, you may need to modify it according to your model deployment
6.  **Call `_get_response_with_messages`:** Sends the messages to the Azure OpenAI endpoint and gets the `ChatCompletion` object.  将消息发送到 Azure OpenAI 端点，并获取 ChatCompletion 对象.
7.  **Extract and print content:** Extracts the text content from the response and prints it. 从响应中提取文本内容并打印.
8.  **Close the client:** Closes the client to release resources. 关闭客户端以释放资源。

**2.  Enhanced `AzureOpenAIEmbedding`:**

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from typing import List, Union, Literal, Optional

import openai
from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
from pickledb import PickleDB

def get_azure_active_directory_token_provider() -> Callable[[], str]:
    """
    从 Azure Active Directory 获取访问令牌。
    Gets an access token from Azure Active Directory.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    return token_provider


def parse_wait_time_from_error(error: openai.RateLimitError) -> Optional[int]:
    """
    从 OpenAI 的速率限制错误中解析等待时间（秒）。
    Parses the wait time (in seconds) from an OpenAI rate limit error.
    """
    try:
        info_str: str = error.args[0]
        info_dict_str: str = info_str[info_str.find("{"):]
        error_info: dict = json.loads(re.compile('(?<!\\\\)\'').sub('\"', info_dict_str))
        error_message = error_info["error"]["message"]
        matches = re.search(r"Try again in (\d+) seconds", error_message)
        if matches: # Ensure matches is not None
            wait_time = int(matches.group(1)) + 3  # NOTE: wait 3 more seconds here.
            return wait_time
        else:
            return None

    except Exception:
        return None

class AzureOpenAIEmbedding(Embeddings):
    """
    使用 Azure OpenAI 获取文本嵌入的类。
    Class for getting text embeddings using Azure OpenAI.
    """

    def __init__(self, **kwargs) -> None:
        """
        初始化 AzureOpenAIEmbedding。
        Initializes AzureOpenAIEmbedding.

        Args:
            **kwargs:  传递给 AzureOpenAI 客户端的其他配置.
        """
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
        """
        将嵌入保存到缓存。
        Saves the embedding to the cache.
        """
        if self._cache is None:
            return

        self._cache.set(query, embedding)

    def _get_cache(self, query: str) -> Union[List[float], Literal[False]]:
        """
        从缓存中获取嵌入。
        Gets the embedding from the cache.
        """
        if self._cache is None:
            return False

        return self._cache.get(query)

    def _get_response(self, texts: Union[str, List[str]]) -> CreateEmbeddingResponse:
        """
        从 Azure OpenAI 获取嵌入响应。
        Gets the embedding response from Azure OpenAI.
        """
        while True:
            try:
                response = self._client.embeddings.create(input=texts, model=self._model)
                break

            except openai.RateLimitError as e:
                expected_wait = parse_wait_time_from_error(e)
                if expected_wait is not None:
                    print(f"Embedding failed due to RateLimitError, wait for {expected_wait} seconds")
                    time.sleep(expected_wait)
                else:
                    print(f"Embedding failed due to RateLimitError, but failed parsing expected waiting time, wait for 30 seconds")
                    time.sleep(30)

            except Exception as e:
                print(f"Embedding failed due to exception {e}")
                raise e # re-raise the exception rather than exit(0), this way, the user could handle the exception

        return response

    def embed_documents(self, texts: List[str], batch_call: bool = False) -> List[List[float]]:
        """
        获取文档的嵌入。
        Gets the embeddings for a list of documents.
        """
        if batch_call:
            response = self._get_response(texts)
            embeddings = [res.embedding for res in response.data]
        else:
            embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        获取查询的嵌入。
        Gets the embedding for a query.
        """
        embedding = self._get_cache(text)
        if embedding is False:
            response = self._get_response(text)
            embedding = response.data[0].embedding
            self._save_cache(text, embedding)
        return embedding
```

**Key Improvements and Explanations:**

*   **Error Handling:** Re-raise the exception
*   **Clarity and Comments:** Added docstrings and comments to explain the purpose of each method and argument. 增加了文档字符串和注释来解释每个方法和参数的用途.
*   **Azure AD Token Provider:**  The code for obtaining the Azure AD token is now in its own function for better organization. 获取 Azure AD 令牌的代码现在位于其自己的函数中，以便更好地组织.
*   **`parse_wait_time_from_error` robust:** add a check to make sure `matches` is not `None` to avoid potential error

**How to use AzureOpenAIEmbedding:**

```python
# Example Usage (示例用法)
if __name__ == "__main__":
    # Assuming you have set environment variables for AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc.
    # 假设您已设置 AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT 等环境变量。

    embedding_client = AzureOpenAIEmbedding(model="text-embedding-ada-002") # modify to your deployment name
    text = "This is a sample sentence." # try "你好，这是一个例子。"

    embedding = embedding_client.embed_query(text)

    print(f"Embedding for '{text}': {embedding[:10]}...")  # Print only the first 10 elements
```

**Explanation of the Example:**

1.  **Import necessary modules:** imports the needed modules
2.  **Initialize `AzureOpenAIEmbedding`:**  Creates an instance of the `AzureOpenAIEmbedding` with the necessary configuration, including retry settings and the logger. 初始化 AzureOpenAIClient，包括重试设置和日志记录器.
3.  **Call `embed_query`:** Sends the texts to the Azure OpenAI endpoint and gets the embedding object.  将消息发送到 Azure OpenAI 端点，并获取 embedding 对象.
4.  **Print embedding:** print the embedding of the text

These improvements prioritize clarity, maintainability, and robustness. They provide a more solid foundation for building applications that interact with Azure OpenAI. The Chinese comments and explanations enhance accessibility for Chinese-speaking developers.
