Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\serve\inference.html`

好的，现在我重新开始，提供一个更智能的版本，重点在于代码改进和易于理解的中文解释。

**1. 优化停止条件 (Optimized Stopping Criteria):**

```python
import torch
from transformers import StoppingCriteria

class LengthBasedStoppingCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        """
        基于最大长度的停止条件。

        Args:
            max_length: 生成的最大 token 数量。
        """
        super().__init__()
        self.max_length = max_length
        self.current_length = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        检查是否达到停止条件。

        Args:
            input_ids: 当前生成的 token 序列。
            scores: 模型输出的 logits。

        Returns:
            如果达到停止条件，则返回 True，否则返回 False。
        """
        self.current_length += 1
        return self.current_length >= self.max_length


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_word_ids: List[List[int]], device: torch.device):
        """
        基于停止词的停止条件。

        Args:
            stop_word_ids: 停止词的 token ID 列表。每个停止词是一个 token ID 列表，允许停止多 token 词。
            device: 用于存储停止词 tensor 的设备 (CPU 或 CUDA)。
        """
        super().__init__()
        self.stop_words = [torch.tensor(word).to(device) for word in stop_word_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        检查是否生成了任何停止词。

        Args:
            input_ids: 当前生成的 token 序列。
            scores: 模型输出的 logits。

        Returns:
            如果生成了任何停止词，则返回 True，否则返回 False。
        """
        for stop_word in self.stop_words:
            if len(stop_word) > input_ids.shape[1]:  # 停止词比当前序列还长
                continue
            if torch.equal(input_ids[0, -len(stop_word):], stop_word): # 比较 input_ids 的最后几个 token 和 stop_word
                return True
        return False
```

**描述:** 这段代码定义了两种更精细的停止条件：基于最大长度和基于停止词。

*   **LengthBasedStoppingCriteria:** 简单地基于生成 token 的最大数量停止生成。
*   **StopWordsCriteria:**  检查是否生成了预定义的停止词。  关键改进是：
    *   支持多 token 停止词。
    *   更有效地在 CUDA 设备上执行停止词检查。

**如何使用:**  创建 `LengthBasedStoppingCriteria` 或 `StopWordsCriteria` 的实例，并将其添加到 `StoppingCriteriaList` 中。

---

**2. 优化生成函数 (Optimized Generation Function):**

```python
import torch
from transformers import (
    TextIteratorStreamer,
    StoppingCriteriaList,
)
from typing import List, Optional

@torch.inference_mode()
def generate_optimized(
    model: torch.nn.Module,
    tokenizer,
    inputs: dict,  # 使用字典来传递所有输入
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    stop_word_ids: Optional[List[List[int]]] = None,
):
    """
    使用优化的方式流式传输多模态模型的文本输出。

    Args:
        model:  DeepSeek 模型。
        tokenizer:  Tokenizer。
        inputs: 包含 'input_ids', 'attention_mask', 'images' 等的字典。
        max_new_tokens: 生成的最大 token 数量。
        temperature:  Temperature 采样参数。
        top_p: Top-p 采样参数。
        repetition_penalty: 重复惩罚参数。
        stop_word_ids: 停止词的 token ID 列表. None 表示没有停止词。

    Yields:
        生成的文本片段。
    """
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 重要：跳过特殊 token
    stopping_criteria_list = []

    # 添加长度停止条件
    stopping_criteria_list.append(LengthBasedStoppingCriteria(max_length=max_new_tokens))

    # 添加停止词停止条件
    if stop_word_ids:
        device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动获取设备
        stopping_criteria_list.append(StopWordsCriteria(stop_word_ids, device))

    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)

    generation_kwargs = dict(
        **inputs,  # 解包所有输入参数
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
        use_cache=True, # 启用缓存
    )

    # 检查是否支持 generate 函数的某些参数，避免错误
    if not hasattr(model.generate, '__wrapped__'): # 兼容 accelerate 模型
        for k in ["images", "images_seq_mask", "images_spatial_crop"]:
            if k in generation_kwargs:
                del generation_kwargs[k]  # 移除不支持的参数

    thread = Thread(target=model.generate, kwargs=generation_kwargs) # 使用线程来避免阻塞
    thread.start()

    for new_text in streamer:
        yield new_text
```

**描述:**  这个 `generate_optimized` 函数做了以下优化：

*   **统一的输入处理:** 使用字典 `inputs` 接收所有输入，简化了函数签名。
*   **动态设备获取:** 自动检测模型所在的设备，并用于停止词的张量。
*   **流式输出:** 使用 `TextIteratorStreamer` 流式传输生成的文本，提高了响应速度。
*   **跳过特殊 token:**  `skip_special_tokens=True` 避免生成特殊 token。
*   **兼容性处理:** 检查模型是否支持 `generate` 函数的特定参数，并进行调整，提高了代码的鲁棒性。使用`__wrapped__`判断是否是accelerate的模型
*    **启用缓存:** 默认启用 `use_cache=True` 以加速生成。

**如何使用:**  准备好包含 `input_ids`、`attention_mask` 和其他必要输入的字典。 调用 `generate_optimized` 函数，并迭代结果以获取生成的文本。

---

**3. 集成示例 (Integration Example):**

```python
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
import torch

# 假设 model_path 已经设置好
model_path = "your_model_path"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = DeepseekVLV2Processor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()

# 准备图像和文本
image = Image.open("your_image.jpg").convert("RGB")
prompt = "描述这张图片。"

# 构建对话
conversations = [
    {
        "role": "user",
        "content": prompt,
        "images": [image]
    }
]

# 准备输入
inputs = processor(conversations=conversations, images=[image], return_tensors="pt").to("cuda")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
pixel_values = inputs['pixel_values'] # for images

# 可选：停止词
stop_words = ["</s>"]  # 示例停止词
stop_word_ids = [tokenizer.encode(word) for word in stop_words]

# 调用优化后的生成函数
with torch.inference_mode():
    for text in generate_optimized(
        model=model,
        tokenizer=tokenizer,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values":pixel_values},  # 显式传递所有输入
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        stop_word_ids=stop_word_ids,
    ):
        print(text, end="", flush=True)
print()
```

**描述:** 这个示例演示了如何将优化后的 `generate_optimized` 函数集成到你的代码中。

1.  **加载模型和 tokenizer:**  使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 加载模型。

2.  **准备输入:**  使用 `DeepseekVLV2Processor` 处理图像和文本，并将其转换为模型所需的格式。

3.  **设置停止词 (可选):**  定义一个停止词列表，并将其转换为 token ID。

4.  **调用 `generate_optimized`:**  将模型、tokenizer 和处理后的输入传递给 `generate_optimized` 函数。

5.  **流式打印输出:** 迭代 `generate_optimized` 函数的输出，并打印生成的文本。

**总结:**

这个版本提供了更智能的代码，具有更好的可读性、可维护性和性能。 通过使用优化的停止条件和生成函数，你可以提高文本生成的质量和效率。 同时，详细的中文注释和集成示例使代码更易于理解和使用。
