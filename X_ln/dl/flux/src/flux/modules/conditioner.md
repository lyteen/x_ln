Lan: `py` From`dl/flux/src\flux\modules\conditioner.py`

好的，我们重新开始。

**1. HFEmbedder 类初始化 (\_\_init\_\_):**

```python
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")  # 判断是否使用CLIP模型
        self.max_length = max_length  # 设置最大序列长度
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state" # 根据模型类型设置输出键

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)  # 加载CLIP tokenizer
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)  # 加载CLIP模型
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)  # 加载T5 tokenizer
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)  # 加载T5模型

        self.hf_module = self.hf_module.eval().requires_grad_(False)  # 设置模型为评估模式，禁止梯度更新

# Example Usage 演示用法:
if __name__ == '__main__':
    # 初始化CLIPEmbedder
    clip_embedder = HFEmbedder(version="openai/clip-vit-base-patch32", max_length=77)

    # 初始化T5Embedder
    t5_embedder = HFEmbedder(version="t5-small", max_length=512) # T5-small 是一个例子

    print("HFEmbedder 初始化完成!")
```

**描述:** 这个 `__init__` 方法是类的构造函数。它接收模型版本 (例如 "openai/clip-vit-base-patch32" 或 "t5-small") 和最大序列长度作为参数。根据模型类型，它加载相应的 tokenizer 和模型 (CLIP 或 T5)。最后，它将模型设置为评估模式，并禁止梯度更新，以节省计算资源。

**如何使用:**  首先，需要使用 `HFEmbedder` 类创建一个对象。 在创建对象时，需要指定 `version` 和 `max_length`。  `version` 决定使用 CLIP 模型还是 T5 模型。  `max_length` 限制输入文本的最大长度。

**2. HFEmbedder 类的前向传播 (forward):**

```python
from torch import Tensor

# (继续之前代码)
    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )  # 使用 tokenizer 对文本进行编码

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )  # 将编码后的输入传递给模型

        return outputs[self.output_key]  # 返回模型的输出

# Example Usage 演示用法:
if __name__ == '__main__':
    # (续接之前的代码)

    # 使用CLIPEmbedder
    clip_text = ["这是一个示例文本", "这是另一个文本"]
    clip_embeddings = clip_embedder(clip_text)
    print(f"CLIP embeddings 形状: {clip_embeddings.shape}")

    # 使用T5Embedder
    t5_text = ["这是一个示例文本", "这是另一个文本"]
    t5_embeddings = t5_embedder(t5_text)
    print(f"T5 embeddings 形状: {t5_embeddings.shape}")
```

**描述:** 这个 `forward` 方法接收一个文本列表作为输入。 它首先使用 tokenizer 将文本编码为模型可以理解的格式。然后，它将编码后的输入传递给预训练的 transformer 模型 (CLIP 或 T5)。最后，它返回模型的输出，例如 CLIP 的 `pooler_output` 或 T5 的 `last_hidden_state`。

**如何使用:** 创建 `HFEmbedder` 对象后，可以使用 `forward` 方法将文本转换为嵌入向量。  `forward` 方法接收一个字符串列表作为输入，并返回一个 `Tensor`，它包含每个输入文本的嵌入向量。
