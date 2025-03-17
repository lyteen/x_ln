Lan: `py` From`dl/flux/src\flux\modules\conditioner.py`

好的，让我们重新开始，并且这次我们更关注代码的可读性、效率以及添加一些更高级的特性。

**目标:**

1.  **提高效率:** 避免不必要的张量复制，特别是在处理大型文本批次时。
2.  **增强可读性:** 使用更清晰的变量名和注释。
3.  **添加灵活性:** 允许更细粒度的控制，例如指定设备。
4.  **添加对齐功能:** 可以输出对齐的文本表示，便于进行后续的图文匹配或其他任务。

下面是改进后的 `HFEmbedder` 类：

```python
import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from typing import List, Optional, Union

class HFEmbedder(nn.Module):
    def __init__(self,
                 version: str,
                 max_length: int,
                 device: Optional[Union[str, torch.device]] = None,
                 output_attentions: bool = False,
                 **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu" # 默认使用cuda，如果可用

        # 初始化 Tokenizer 和 模型
        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs).to(self.device)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs).to(self.device)

        self.hf_module = self.hf_module.eval().requires_grad_(False) # 设置为评估模式，禁用梯度

        self.output_attentions = output_attentions # 是否输出attention权重

    def forward(self, text: List[str], return_token_embeddings: bool = False) -> Union[Tensor, dict]:
        """
        前向传播函数，将文本转换为嵌入向量。

        Args:
            text: 文本列表.
            return_token_embeddings: 是否返回token级别的嵌入向量.

        Returns:
            如果 return_token_embeddings 为 True, 返回一个包含 'pooled_output' 和 'token_embeddings' 的字典.
            否则，仅返回 pooled_output (Tensor).
        """

        # 使用 Tokenizer 进行编码
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(self.device) # 将编码后的数据移动到正确的设备

        # 通过 Hugging Face 模型获得输出
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"],
            attention_mask=batch_encoding["attention_mask"],  # 使用 attention mask
            output_hidden_states=return_token_embeddings,  # 是否输出 hidden states
            output_attentions=self.output_attentions # 是否输出 attention weights
        )

        # 根据需求返回不同的输出
        if return_token_embeddings:
            return {
                "pooled_output": outputs[self.output_key],  # 池化后的输出
                "token_embeddings": outputs.last_hidden_state, # token 级别的嵌入向量
                "attentions": outputs.attentions if self.output_attentions else None # attention weights
            }
        else:
            return outputs[self.output_key]

# 示例用法
if __name__ == '__main__':
    # 初始化 HFEmbedder，指定模型版本和最大长度
    embedder = HFEmbedder(version="openai/clip-vit-base-patch32", max_length=77, device="cpu", output_attentions=True)  # 强制使用 CPU

    # 一些示例文本
    texts = ["A photo of a cat.", "A photo of a dog."]

    # 获取嵌入向量
    embeddings = embedder(texts)
    print("池化后的嵌入向量形状:", embeddings.shape)

    # 获取 token 级别的嵌入向量
    detailed_output = embedder(texts, return_token_embeddings=True)
    print("池化后的嵌入向量形状:", detailed_output["pooled_output"].shape)
    print("Token 级别的嵌入向量形状:", detailed_output["token_embeddings"].shape)
    print("Attention weights:", detailed_output["attentions"][0].shape) # 查看第一层的attention weights
```

**代码详解 (中文):**

*   **`__init__` (初始化):**
    *   `device`:  添加了 `device` 参数，允许你指定模型运行的设备（CPU 或 CUDA）。如果未指定，它会自动检测 CUDA 是否可用。
    *   `.to(self.device)`:  在加载模型后，立即将模型移动到指定的设备。
    *   `output_attentions`：指定是否输出attention权重
*   **`forward` (前向传播):**
    *   `return_token_embeddings`: 添加了一个 `return_token_embeddings` 参数。如果设置为 `True`，函数将返回一个字典，其中包含池化后的输出和每个 token 的嵌入向量。
    *   Attention Mask: 使用了 `attention_mask` 来屏蔽 padding 的 token，这对于处理可变长度的文本非常重要。
    *   更灵活的输出: 根据 `return_token_embeddings` 的值，返回不同类型的输出，允许更细粒度的控制。
*   **效率改进:**
    *   避免不必要的张量复制: 使用 `.to(self.device)` 将张量移动到正确的设备，而不是创建新的复制。
*   **示例用法:**
    *   展示了如何使用 `HFEmbedder` 类，并演示了如何获取池化后的嵌入向量和 token 级别的嵌入向量。

**关键改进总结:**

*   **设备指定:**  现在可以显式指定模型运行的设备。
*   **Token 级别嵌入:** 可以选择返回每个 token 的嵌入向量，这对于很多下游任务很有用，例如文本对齐。
*   **效率:**  避免了不必要的张量复制。
*   **可读性:**  代码更加清晰易懂，添加了详细的注释。
*   **Attention机制输出：**可以选择是否输出attention权重，方便后续分析

希望这个改进后的版本更符合你的需求！记住，这只是一个示例，你可以根据自己的具体应用场景进行进一步的修改和优化。
