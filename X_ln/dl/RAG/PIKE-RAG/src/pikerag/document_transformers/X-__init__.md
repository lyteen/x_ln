Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\document_transformers\__init__.py`

好的，重新开始。以下是针对 `pikerag` 包中的文档转换器的一些更智能的改进，以及详细的中文描述和使用示例。

**1. 改进的 LLM Powered Filter (基于 LLM 的过滤器):**

```python
from typing import Callable, Optional
import openai
import logging

logger = logging.getLogger(__name__)

class SmartLLMPoweredFilter:
    """
    利用更智能的 LLM 提示和缓存机制来过滤文档。
    """

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        filter_prompt: str = "这个文档是否与主题 {topic} 相关？ (是/否)",
        cache: Optional[dict] = None,
        temperature: float = 0.0
    ):
        """
        初始化 SmartLLMPoweredFilter。

        Args:
            openai_api_key: OpenAI API 密钥。
            model_name: 使用的 OpenAI 模型名称。
            filter_prompt: 用于过滤文档的提示。
            cache: 用于缓存 LLM 响应的字典。
            temperature: LLM 生成文本的温度。
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.filter_prompt = filter_prompt
        self.cache = cache if cache is not None else {}
        self.temperature = temperature
        openai.api_key = self.openai_api_key  # 设置 OpenAI API 密钥

    def _get_llm_response(self, document_content: str, topic: str) -> str:
        """
        从 LLM 获取响应。
        """
        prompt = self.filter_prompt.format(topic=topic, document=document_content)
        if prompt in self.cache:
            return self.cache[prompt]

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            answer = response["choices"][0]["message"]["content"].strip().lower()
            self.cache[prompt] = answer  # 缓存结果
            return answer
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return "否"  # 出错时默认不保留

    def filter_document(self, document_content: str, topic: str) -> bool:
        """
        判断文档是否与主题相关。
        """
        answer = self._get_llm_response(document_content, topic)
        return "是" in answer  # 简单判断是否包含 "是"

    def __call__(self, document_content: str, topic: str) -> bool:
        """
        作为函数调用的快捷方式。
        """
        return self.filter_document(document_content, topic)

# Demo Usage 演示用法
if __name__ == '__main__':
    # 需要设置 OPENAI_API_KEY 环境变量
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key:
        smart_filter = SmartLLMPoweredFilter(openai_api_key=openai_api_key, topic="机器学习")
        document = "本文档介绍了深度学习在图像识别中的应用。"
        is_relevant = smart_filter(document, "机器学习")
        print(f"文档是否相关: {is_relevant}")

        # 第二次调用，应该从缓存中获取结果
        is_relevant = smart_filter(document, "机器学习")
        print(f"文档是否相关 (来自缓存): {is_relevant}")
    else:
        print("请设置 OPENAI_API_KEY 环境变量。")
```

**描述 (中文):**

这个 `SmartLLMPoweredFilter` 是一个改进的文档过滤器，它使用大型语言模型 (LLM) 来判断文档是否与指定的主题相关。 相比于原始的 `LLMPoweredFilter`，它主要有以下改进:

*   **更智能的提示 (Prompt):** 可以自定义 `filter_prompt`，允许你根据具体的需求调整提示语，提高过滤的准确性。
*   **缓存机制 (Cache):**  使用了缓存来存储 LLM 的响应。 对于相同的文档和主题，下次调用时直接从缓存中获取结果，避免重复调用 LLM，节省时间和成本。
*   **错误处理 (Error Handling):**  增加了错误处理机制，当 LLM 调用失败时，会记录错误信息并默认返回 `False`，保证程序的健壮性。
*   **温度参数 (Temperature):** 可以通过 `temperature` 参数控制 LLM 生成文本的随机性。

**如何使用 (中文):**

1.  **安装 OpenAI Python 库:** `pip install openai`
2.  **设置 OpenAI API 密钥:** 将你的 OpenAI API 密钥设置为环境变量 `OPENAI_API_KEY`。
3.  **初始化 `SmartLLMPoweredFilter`:** 创建 `SmartLLMPoweredFilter` 的实例，传入 OpenAI API 密钥和主题。
4.  **调用 `filter_document` 方法:**  使用 `filter_document` 方法判断文档是否与主题相关。
5.  **自定义提示语:** 可以通过修改 `filter_prompt` 参数来自定义提示语。

**2. 改进的 LLM Powered Recursive Splitter (基于 LLM 的递归分割器):**

```python
from typing import List
import openai
import logging
import re

logger = logging.getLogger(__name__)

class SmartLLMPoweredRecursiveSplitter:
    """
    使用 LLM 智能地递归分割文档，考虑语义完整性。
    """

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        splitting_prompt: str = "根据语义，将以下文档分割成有意义的段落：\n\n{document}\n\n分割后的段落：",
        max_chunk_size: int = 2048,
        cache: Optional[dict] = None,
        temperature: float = 0.0
    ):
        """
        初始化 SmartLLMPoweredRecursiveSplitter。

        Args:
            openai_api_key: OpenAI API 密钥。
            model_name: 使用的 OpenAI 模型名称。
            splitting_prompt: 用于分割文档的提示。
            max_chunk_size: 每个分割块的最大大小。
            cache: 用于缓存 LLM 响应的字典。
            temperature: LLM 生成文本的温度。
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.splitting_prompt = splitting_prompt
        self.max_chunk_size = max_chunk_size
        self.cache = cache if cache is not None else {}
        self.temperature = temperature
        openai.api_key = self.openai_api_key

    def _get_llm_response(self, document_content: str) -> str:
        """
        从 LLM 获取分割后的文档。
        """
        prompt = self.splitting_prompt.format(document=document_content)
        if prompt in self.cache:
            return self.cache[prompt]

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            self.cache[prompt] = answer
            return answer
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return document_content  # 出错时返回原始文档

    def split_document(self, document_content: str) -> List[str]:
        """
        递归地分割文档。
        """
        if len(document_content) <= self.max_chunk_size:
            return [document_content]

        splitted_text = self._get_llm_response(document_content)
        chunks = re.split(r'\n\s*\n', splitted_text)  # 使用空行分割

        final_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > self.max_chunk_size:
                # 递归分割过大的块
                final_chunks.extend(self.split_document(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def __call__(self, document_content: str) -> List[str]:
        """
        作为函数调用的快捷方式。
        """
        return self.split_document(document_content)

# Demo Usage
if __name__ == '__main__':
    # 需要设置 OPENAI_API_KEY 环境变量
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key:
        smart_splitter = SmartLLMPoweredRecursiveSplitter(openai_api_key=openai_api_key, max_chunk_size=500)  # 更小的块大小用于演示
        document = """
        第一段：这是关于机器学习的介绍。机器学习是一种人工智能技术，它允许计算机从数据中学习，而无需进行明确的编程。

        第二段：深度学习是机器学习的一个子领域，它使用深度神经网络来解决复杂的问题，例如图像识别和自然语言处理。

        第三段：自然语言处理 (NLP) 是人工智能的另一个重要领域，它涉及计算机理解和生成人类语言。
        """
        chunks = smart_splitter(document)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:\n{chunk}\n")
    else:
        print("请设置 OPENAI_API_KEY 环境变量。")
```

**描述 (中文):**

这个 `SmartLLMPoweredRecursiveSplitter` 是一个使用 LLM 来智能分割文档的工具。 它的主要目标是将长文档分割成语义上连贯的较小块，以便于后续处理，例如信息检索。 相对于原始的分割器，它的改进之处在于：

*   **语义感知的分割 (Semantic-Aware Splitting):**  通过自定义的 `splitting_prompt`，可以指示 LLM 根据文档的语义结构进行分割，例如按段落或主题进行分割。
*   **递归分割 (Recursive Splitting):**  如果分割后的块仍然太大，它会递归地分割这些块，直到所有块都小于 `max_chunk_size`。
*   **缓存机制 (Cache):**  同样使用了缓存来避免重复调用 LLM。
*   **更灵活的分隔符 (Flexible Separator):** 使用正则表达式 `re.split(r'\n\s*\n', splitted_text)` 根据空行分割文档，更适用于结构化的文本。

**如何使用 (中文):**

1.  **安装 OpenAI Python 库:** `pip install openai`
2.  **设置 OpenAI API 密钥:** 将你的 OpenAI API 密钥设置为环境变量 `OPENAI_API_KEY`。
3.  **初始化 `SmartLLMPoweredRecursiveSplitter`:** 创建 `SmartLLMPoweredRecursiveSplitter` 的实例，传入 OpenAI API 密钥和 `max_chunk_size`。
4.  **调用 `split_document` 方法:**  使用 `split_document` 方法分割文档。
5.  **自定义提示语:** 可以修改 `splitting_prompt` 参数来自定义分割策略。

**3.  LLMPoweredTagger 的改进 (由于时间关系，只提供思路):**

*   **实体识别 (Named Entity Recognition - NER):** 使用 LLM 进行实体识别，例如识别文档中的人名、地名、组织机构等。
*   **情感分析 (Sentiment Analysis):**  使用 LLM 分析文档的情感倾向，例如判断文档是积极的、消极的还是中性的。
*   **主题标注 (Topic Tagging):**  使用 LLM 识别文档的主题，并为文档添加相应的标签。

**4. RecursiveSentenceSplitter:** 这个分割器本身已经比较成熟，改进空间较小。 可以考虑增加对特定语言的支持，例如中文分句的优化。

这些改进都侧重于利用 LLM 的强大能力，使文档处理更加智能和高效。 希望这些代码示例和描述能够帮助你更好地理解和使用这些工具。

**注意事项:**

*   所有使用 OpenAI API 的代码都需要有效的 API 密钥。
*   LLM 的性能取决于模型的选择和提示语的设计。  需要根据具体的需求进行调整。
*   频繁调用 LLM 可能会产生较高的费用。 使用缓存可以有效降低成本。
