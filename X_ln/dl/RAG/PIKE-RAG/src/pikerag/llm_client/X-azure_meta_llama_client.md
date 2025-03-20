Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\llm_client\azure_meta_llama_client.py`

Okay, I will provide an improved version of the `AzureMetaLlamaClient` class with explanations and improvements, followed by a simple demo. This version will focus on:

*   **Enhanced Error Handling:** More robust error handling for API calls.
*   **Retry Mechanism with Jitter:** Adding jitter to the retry mechanism to avoid synchronized retries.
*   **Asynchronous Request (Optional):** An option to use asynchronous requests for better performance (requires `aiohttp`).
*   **Improved Logging:** More informative logging.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import urllib.request
import urllib.error
import time
import random
from typing import List, Optional, Dict

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger


class AzureMetaLlamaClient(BaseLLMClient):
    NAME = "AzureMetaLlamaClient"

    def __init__(
        self,
        location: str = None,
        auto_dump: bool = True,
        logger: Logger = None,
        max_attempt: int = 5,
        exponential_backoff_factor: float = 2.0,  # Use float for backoff
        unit_wait_time: int = 60,
        use_async: bool = False, # Option to enable asynchronous requests
        **kwargs,
    ) -> None:
        super().__init__(
            location,
            auto_dump,
            logger,
            max_attempt,
            exponential_backoff_factor,
            unit_wait_time,
            **kwargs,
        )
        self._use_async = use_async
        if self._use_async:
            try:
                import aiohttp
                self._aiohttp = aiohttp  # Store the module for later use
                self._session = None # Initialize session later, create a session if use async
            except ImportError:
                self._use_async = False
                self.warning("aiohttp is not installed.  Falling back to synchronous requests.")
        else:
            self._aiohttp = None  # Prevent possible errors
            self._session = None # Not in use if not async

        self._init_agent(**kwargs)

    def _init_agent(self, **kwargs) -> None:
        llama_endpoint_name = kwargs.get("llama_endpoint_name", None)
        if llama_endpoint_name is None:
            llama_endpoint_name = "LLAMA_ENDPOINT"
        self._endpoint = os.getenv(llama_endpoint_name)
        assert self._endpoint, "LLAMA_ENDPOINT is not set!"

        llama_key_name = kwargs.get("llama_key_name", None)
        if llama_key_name is None:
            llama_key_name = "LLAMA_API_KEY"
        self._api_key = os.getenv(llama_key_name)
        assert self._api_key, "LLAMA_API_KEY is not set!"

    def _wrap_header(self, **llm_config) -> dict:
        assert "model" in llm_config, "`model` must be provided in `llm_config` to call AzureMetaLlamaClient!"
        header = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self._api_key,
            "azureml-model-deployment": llm_config["model"],
        }
        return header

    def _wrap_body(self, messages: List[dict], **llm_config) -> bytes:
        data = {
            "input_data": {
                "input_string": messages,
                "parameters": llm_config,
            }
        }
        body = json.dumps(data).encode("utf-8")  # Explicitly encode to utf-8
        return body

    async def _async_get_response(self, messages: List[dict], llm_config: Dict) -> bytes:
        """Asynchronous method to get response from the endpoint."""
        if self._session is None:
            # Create session if not exists
            self._session = self._aiohttp.ClientSession()
        header = self._wrap_header(**llm_config)
        body = self._wrap_body(messages, **llm_config)

        try:
            async with self._session.post(self._endpoint, headers=header, data=body) as response:
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                return await response.read() # Return bytes
        except self._aiohttp.ClientError as e:
            self.error(f"Async request failed: {e}")
            raise
        except Exception as e:
             self.error(f"Async request failed with general exception: {e}")
             raise

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> bytes:
        response: bytes = None
        num_attempt: int = 0
        while num_attempt < self._max_attempt:
            try:
                if self._use_async:
                    import asyncio
                    try:
                        response = asyncio.run(self._async_get_response(messages, llm_config))
                    except Exception as e:
                        self.warning(f"Async request failed: {e}")
                        response = None # Set to None to trigger retry
                else:
                    header = self._wrap_header(**llm_config)
                    body = self._wrap_body(messages, **llm_config)
                    req = urllib.request.Request(self._endpoint, body, header)
                    with urllib.request.urlopen(req) as res:
                        response = res.read()
                if response:
                    break  # Exit loop if successful

            except urllib.error.HTTPError as error:
                self.warning(f"  HTTPError: {error.code} - {error.reason}")
                try:
                    error_message = error.read().decode("utf8", 'ignore')
                    self.warning(f"  Error message: {error_message}")
                except Exception:
                    self.warning("  Could not decode error message.")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")
            except Exception as e:
                self.warning(f"  General Exception: {e}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning(f"  Retrying...")
        if not response:
            self.error(f"Failed to get response after {self._max_attempt} attempts.")
        return response

    def _get_content_from_response(self, response: bytes, messages: List[dict] = None) -> str:
        try:
            response_str = response.decode("utf-8")  # Decode here
            content = json.loads(response_str)["output"]
            if content is None:
                warning_message = "Non-Content returned"
                self.warning(warning_message)
                self.debug(f"  -- Complete response: {response_str}") # Log decoded response
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")
                content = ""
        except (json.JSONDecodeError, KeyError) as e:
            self.error(f"Error parsing response: {e}")
            self.debug(f"  -- Raw response: {response}")
            content = ""
        except Exception as e:
            self.error(f"Unexpected error during response processing: {e}")
            content = ""

        return content

    def _wait(self, attempt_number: int) -> None:
        """Waits before retrying the API call."""
        sleep_time = self.unit_wait_time * (self.exponential_backoff_factor ** (attempt_number - 1))
        # Add jitter to avoid synchronized retries
        sleep_time += random.uniform(0, self.unit_wait_time)
        sleep_time = min(sleep_time, 300)  # Limit max sleep time to 5 minutes
        self.info(f"  Waiting {sleep_time:.2f} seconds before retry...")
        time.sleep(sleep_time)

    def close(self) -> None:
        """Close the aiohttp session if it exists"""
        if self._session:
            import asyncio
            asyncio.run(self._session.close())
            self._session = None

# Demo Usage (需要设置环境变量 LLAMA_ENDPOINT 和 LLAMA_API_KEY)
if __name__ == "__main__":
    # 需要先配置好 Logger
    class SimpleLogger:
        def info(self, msg):
            print(f"INFO: {msg}")
        def warning(self, msg):
            print(f"WARNING: {msg}")
        def error(self, msg):
            print(f"ERROR: {msg}")
        def debug(self, msg):
            print(f"DEBUG: {msg}")

    logger = SimpleLogger()
    # 假设已经设置了环境变量 LLAMA_ENDPOINT 和 LLAMA_API_KEY
    client = AzureMetaLlamaClient(logger=logger, max_attempt=3, use_async=False)  # 初始化客户端，可以尝试开启异步
    try:
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        llm_config = {"model": "your_model_deployment_name"}  # 替换为你的模型部署名称
        response = client.get_content(messages, **llm_config)

        print(f"Response: {response}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close() # Close session when done
```

**Key improvements and explanations (关键改进和解释):**

*   **Asynchronous Requests (异步请求):** Added `use_async` flag. If set to `True` and `aiohttp` is installed, the client will use asynchronous requests.  This can significantly improve performance, especially when handling multiple requests concurrently. 如果设置为`True`且安装了`aiohttp`，客户端将使用异步请求。这可以显著提高性能，尤其是在同时处理多个请求时。
*   **Error Handling (错误处理):** Improved error handling for `urllib.request` and added handling for potential `json.JSONDecodeError` during response processing. 对于`urllib.request`改进了错误处理，并添加了对响应处理期间潜在的`json.JSONDecodeError`的处理。
*   **Retry with Jitter (带抖动的重试):**  The `_wait` method now includes jitter, which adds a random component to the sleep time before retries.  This helps to avoid synchronized retries when multiple clients are experiencing the same issue.  `_wait`方法现在包含抖动，它在重试之前的睡眠时间中添加一个随机分量。这有助于避免在多个客户端遇到相同问题时进行同步重试。
*   **Logging (日志记录):**  Added more detailed logging to provide better insights into the client's behavior and potential issues. 添加了更详细的日志记录，以更好地了解客户端的行为和潜在问题。
*   **Explicit Encoding/Decoding (显式编码/解码):** Explicitly encoding the request body to UTF-8 and decoding the response. 显式地将请求正文编码为 UTF-8 并解码响应。
*   **Type Hints (类型提示):** Added more type hints for better code readability and maintainability. 添加了更多类型提示，以提高代码可读性和可维护性。
*    **Closing session**: When using async option, `close()` function is needed to correctly close the `aiohttp.ClientSession()`.

**How to use (如何使用):**

1.  **Install `aiohttp` (安装 `aiohttp`):** If you want to use the asynchronous functionality, install `aiohttp`:
    ```bash
    pip install aiohttp
    ```

2.  **Set Environment Variables (设置环境变量):** Make sure you have set the `LLAMA_ENDPOINT` and `LLAMA_API_KEY` environment variables.  确保您已设置 `LLAMA_ENDPOINT` 和 `LLAMA_API_KEY` 环境变量。

3.  **Instantiate and Use (实例化和使用):**

    ```python
    from your_module import AzureMetaLlamaClient  # 替换 your_module

    # 假设已经设置了环境变量 LLAMA_ENDPOINT 和 LLAMA_API_KEY
    client = AzureMetaLlamaClient(logger=logger, max_attempt=3, use_async=True)  # 初始化客户端，开启异步
    try:
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        llm_config = {"model": "your_model_deployment_name"}  # 替换为你的模型部署名称
        response = client.get_content(messages, **llm_config)

        print(f"Response: {response}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close() # Close session when done
    ```

**Important Considerations (重要考虑事项):**

*   **Replace `your_model_deployment_name`:**  Make sure to replace `"your_model_deployment_name"` with the actual name of your Azure ML model deployment. 确保将 `"your_model_deployment_name"` 替换为 Azure ML 模型部署的实际名称。
*   **Error Handling:** The code includes basic error handling, but you might want to add more sophisticated error handling based on your specific requirements. 代码包含基本的错误处理，但您可能需要根据您的具体要求添加更复杂的错误处理。
*   **Logging:** The `SimpleLogger` is a placeholder. You should replace it with a proper logging implementation using the `logging` module. `SimpleLogger` 是一个占位符。您应该使用 `logging` 模块将其替换为正确的日志记录实现。
*    **Async session**: Please note the session should be closed after use, by calling `close()`

This improved version provides better error handling, retry mechanisms, asynchronous support, and logging, making it more robust and easier to debug.  这个改进的版本提供了更好的错误处理、重试机制、异步支持和日志记录，使其更健壮且更易于调试。
