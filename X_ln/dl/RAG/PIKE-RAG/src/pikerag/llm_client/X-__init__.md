Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\llm_client\__init__.py`

好的，明白了。现在重新开始。

为了优化LLM Client的代码结构和功能，我们可以考虑以下几个方面：

*   **抽象工厂模式:**  创建一个抽象工厂，用于生成不同类型的LLM Client实例，而无需客户端代码指定具体的类。
*   **重试机制:**  添加重试逻辑，以处理API调用中的瞬时错误。
*   **异步支持:**  提供异步方法，以提高并发性。
*   **指标收集:**  集成指标收集功能，以监控LLM Client的性能。

下面是代码示例：

**1. 抽象类 (Abstract Class) - `BaseLLMClient` (保持不变)**

   ```python
   # pikerag/llm_client/base.py
   from abc import ABC, abstractmethod
   from typing import Optional, Dict

   class BaseLLMClient(ABC):
       @abstractmethod
       def generate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                    temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Abstract method to generate text based on a prompt.

           Args:
               prompt (str): The input prompt for text generation.
               max_tokens (int): The maximum number of tokens to generate.  Defaults to 200.
               stop (Optional[list[str]]): A list of strings at which to stop generation. Defaults to None.
               temperature (float):  Sampling temperature, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.0
               top_p (float): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. Defaults to 1.0
               **kwargs: Additional keyword arguments.

           Returns:
               str: The generated text.
           """
           pass

       @abstractmethod
       async def agenerate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                     temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Asynchronous version of generate.
           """
           pass

       @abstractmethod
       def get_token_count(self, text: str) -> int:
           """
           Abstract method to get the token count of a text.

           Args:
               text (str): The input text.

           Returns:
               int: The token count.
           """
           pass
   ```

   **描述:** `BaseLLMClient` 是一个抽象基类，定义了LLM Client需要实现的通用接口。 其中包含了 `generate` 用于同步生成文本，`agenerate` 用于异步生成文本，和 `get_token_count` 用于获取token数量。

**2. 抽象工厂 (Abstract Factory) - `LLMClientFactory`**

   ```python
   # pikerag/llm_client/factory.py
   from abc import ABC, abstractmethod
   from typing import Dict
   from pikerag.llm_client.base import BaseLLMClient

   class LLMClientFactory(ABC):
       @abstractmethod
       def create_client(self, config: Dict) -> BaseLLMClient:
           """
           Abstract method to create an LLM client.

           Args:
               config (Dict): Configuration dictionary for the LLM client.

           Returns:
               BaseLLMClient: An instance of an LLM client.
           """
           pass
   ```

   **描述:** `LLMClientFactory` 是一个抽象工厂，用于创建不同类型的LLM Client。 它定义了一个 `create_client` 方法，该方法接受一个配置字典，并返回一个 `BaseLLMClient` 实例。

**3. 具体工厂 (Concrete Factory) - `AzureOpenAIFactory`**

   ```python
   # pikerag/llm_client/azure_openai_factory.py
   from typing import Dict
   from pikerag.llm_client.factory import LLMClientFactory
   from pikerag.llm_client.azure_open_ai_client import AzureOpenAIClient
   from pikerag.llm_client.base import BaseLLMClient


   class AzureOpenAIFactory(LLMClientFactory):
       def create_client(self, config: Dict) -> BaseLLMClient:
           """
           Creates an AzureOpenAIClient.

           Args:
               config (Dict): Configuration dictionary for the AzureOpenAIClient.
               Must include 'api_key', 'azure_endpoint', 'deployment_name', and 'api_version'.

           Returns:
               AzureOpenAIClient: An instance of AzureOpenAIClient.
           """
           return AzureOpenAIClient(**config)  # Unpack the config dictionary
   ```

   **描述:** `AzureOpenAIFactory` 是一个具体工厂，用于创建 `AzureOpenAIClient` 实例。 它实现了 `create_client` 方法，该方法使用配置字典创建一个 `AzureOpenAIClient` 实例。

**4. 具体 LLM Client - `AzureOpenAIClient` (改进)**

   ```python
   # pikerag/llm_client/azure_open_ai_client.py
   import openai
   import tiktoken
   import asyncio
   from typing import Optional, Dict
   from pikerag.llm_client.base import BaseLLMClient
   import tenacity
   from tenacity import retry, stop_after_attempt, wait_random_exponential

   class AzureOpenAIClient(BaseLLMClient):
       def __init__(self, api_key: str, azure_endpoint: str, deployment_name: str, api_version: str):
           openai.api_key = api_key
           openai.azure_endpoint = azure_endpoint
           openai.api_type = "azure"
           openai.api_version = api_version
           self.deployment_name = deployment_name
           self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

       @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
       def generate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                    temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Generates text using Azure OpenAI.  Includes retry logic.

           Args:
               prompt (str): The input prompt for text generation.
               max_tokens (int): The maximum number of tokens to generate.  Defaults to 200.
               stop (Optional[list[str]]): A list of strings at which to stop generation. Defaults to None.
               temperature (float):  Sampling temperature. Defaults to 0.0
               top_p (float):  Nucleus sampling probability. Defaults to 1.0
               **kwargs: Additional keyword arguments.

           Returns:
               str: The generated text.
           """
           try:
               response = openai.Completion.create(
                   engine=self.deployment_name,
                   prompt=prompt,
                   max_tokens=max_tokens,
                   stop=stop,
                   temperature=temperature,
                   top_p=top_p,
                   **kwargs
               )
               return response.choices[0].text.strip()
           except Exception as e:
               print(f"Error during text generation: {e}")
               raise

       @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
       async def agenerate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                            temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Asynchronous version of generate.  Includes retry logic.
           """
           try:
               response = await openai.Completion.acreate(
                   engine=self.deployment_name,
                   prompt=prompt,
                   max_tokens=max_tokens,
                   stop=stop,
                   temperature=temperature,
                   top_p=top_p,
                   **kwargs
               )
               return response.choices[0].text.strip()
           except Exception as e:
               print(f"Error during asynchronous text generation: {e}")
               raise


       def get_token_count(self, text: str) -> int:
           """
           Gets the token count of a text using tiktoken.

           Args:
               text (str): The input text.

           Returns:
               int: The token count.
           """
           return len(self.tokenizer.encode(text))

   ```

   **描述:**

    *   **重试机制 (Retry Mechanism):** 使用 `tenacity` 库添加了重试逻辑。 `generate` 和 `agenerate` 方法会在API调用失败时自动重试最多3次，采用指数退避策略。
    *   **异步支持 (Asynchronous Support):** 实现了 `agenerate` 方法，用于异步生成文本。 这可以提高并发性。

**5.  `HFMetaLlamaClient` 适配重试机制**

   ```python
   # pikerag/llm_client/hf_meta_llama_client.py

   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from typing import Optional
   from pikerag.llm_client.base import BaseLLMClient
   import tenacity
   from tenacity import retry, stop_after_attempt, wait_random_exponential
   import asyncio

   class HFMetaLlamaClient(BaseLLMClient):
       def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
           self.tokenizer = AutoTokenizer.from_pretrained(model_name)
           self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
           self.device = device

       @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
       def generate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                    temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Generates text using a Hugging Face Meta Llama model. Includes retry logic.

           Args:
               prompt (str): The input prompt for text generation.
               max_tokens (int): The maximum number of tokens to generate.  Defaults to 200.
               stop (Optional[list[str]]): A list of strings at which to stop generation. Defaults to None.
               temperature (float):  Sampling temperature. Defaults to 0.0
               top_p (float):  Nucleus sampling probability. Defaults to 1.0
               **kwargs: Additional keyword arguments.

           Returns:
               str: The generated text.
           """
           input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
           output = self.model.generate(input_ids, max_length=max_tokens + input_ids.shape[1],
                                        temperature=temperature, top_p=top_p,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **kwargs)
           generated_text = self.tokenizer.decode(output[:, input_ids.shape[1]:][0], skip_special_tokens=True)
           return generated_text

       @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
       async def agenerate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                            temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Asynchronous version of generate. Includes retry logic.
           """
           # Since transformers' generate method is not natively asynchronous,
           # we run it in a separate thread to avoid blocking the event loop.
           loop = asyncio.get_event_loop()
           return await loop.run_in_executor(None, self.generate, prompt, max_tokens, stop, temperature, top_p, **kwargs)


       def get_token_count(self, text: str) -> int:
           """
           Gets the token count of a text using the model's tokenizer.

           Args:
               text (str): The input text.

           Returns:
               int: The token count.
           """
           return len(self.tokenizer.encode(text))

   ```

   **描述:**

    *   **重试机制 (Retry Mechanism):**  使用 `tenacity` 库添加了重试逻辑。 `generate` 和 `agenerate` 方法会在API调用失败时自动重试最多3次，采用指数退避策略。
    *   **异步支持 (Asynchronous Support):** 实现了 `agenerate` 方法，但是因为 transformers 的generate方法是同步的，所以需要使用 `loop.run_in_executor` 将其放到线程池中运行，避免阻塞主线程。

**6.  `AzureMetaLlamaClient` 适配重试机制**

   ```python
   # pikerag/llm_client/azure_meta_llama_client.py
   import openai
   import tiktoken
   from typing import Optional
   from pikerag.llm_client.base import BaseLLMClient
   import tenacity
   from tenacity import retry, stop_after_attempt, wait_random_exponential
   import asyncio

   class AzureMetaLlamaClient(BaseLLMClient):
       def __init__(self, api_key: str, azure_endpoint: str, deployment_name: str, api_version: str):
           openai.api_key = api_key
           openai.azure_endpoint = azure_endpoint
           openai.api_type = "azure"
           openai.api_version = api_version
           self.deployment_name = deployment_name
           self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") # 替换为Meta Llama兼容的tokenizer

       @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
       def generate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                    temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Generates text using Azure OpenAI for Meta Llama models. Includes retry logic.

           Args:
               prompt (str): The input prompt for text generation.
               max_tokens (int): The maximum number of tokens to generate.  Defaults to 200.
               stop (Optional[list[str]]): A list of strings at which to stop generation. Defaults to None.
               temperature (float):  Sampling temperature. Defaults to 0.0
               top_p (float):  Nucleus sampling probability. Defaults to 1.0
               **kwargs: Additional keyword arguments.

           Returns:
               str: The generated text.
           """
           try:
               response = openai.Completion.create(
                   engine=self.deployment_name,
                   prompt=prompt,
                   max_tokens=max_tokens,
                   stop=stop,
                   temperature=temperature,
                   top_p=top_p,
                   **kwargs
               )
               return response.choices[0].text.strip()
           except Exception as e:
               print(f"Error during text generation: {e}")
               raise

       @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
       async def agenerate(self, prompt: str, max_tokens: int = 200, stop: Optional[list[str]] = None,
                            temperature: float = 0.0, top_p: float = 1.0, **kwargs) -> str:
           """
           Asynchronous version of generate. Includes retry logic.
           """
           try:
               response = await openai.Completion.acreate(
                   engine=self.deployment_name,
                   prompt=prompt,
                   max_tokens=max_tokens,
                   stop=stop,
                   temperature=temperature,
                   top_p=top_p,
                   **kwargs
               )
               return response.choices[0].text.strip()
           except Exception as e:
               print(f"Error during asynchronous text generation: {e}")
               raise

       def get_token_count(self, text: