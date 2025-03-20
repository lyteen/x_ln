Lan: `py` From`dl/flux/src\flux\model.py`


**1. `FluxParams` 数据类 (Data Class):**

```python
from dataclasses import dataclass
from typing import List

@dataclass
class FluxParams:
    in_channels: int  # 输入图像的通道数 (例如：3 for RGB)
    out_channels: int # 输出图像的通道数
    vec_in_dim: int   # 输入向量的维度 (例如：时间步嵌入的维度)
    context_in_dim: int # 上下文嵌入维度 (例如：文本描述的嵌入维度)
    hidden_size: int  # 模型隐藏层的维度
    mlp_ratio: float  # MLP 层的扩展比率
    num_heads: int    # 注意力头的数量
    depth: int        # 双流 Transformer 块的数量
    depth_single_blocks: int  # 单流 Transformer 块的数量
    axes_dim: List[int] # 用于位置编码的轴维度
    theta: int         # 位置编码的 theta 参数
    qkv_bias: bool    # 是否在 QKV 注意力中使用偏置
    guidance_embed: bool # 是否嵌入 guidance 信息
```

**描述:** `FluxParams` 是一个数据类，用于存储 `Flux` 模型的所有配置参数。 使用数据类可以更清晰地组织和管理这些参数。

**如何使用:** 创建 `FluxParams` 实例，并将其传递给 `Flux` 类的构造函数。

```python
params = FluxParams(
    in_channels=3,
    out_channels=3,
    vec_in_dim=256,
    context_in_dim=768,
    hidden_size=512,
    mlp_ratio=4.0,
    num_heads=8,
    depth=6,
    depth_single_blocks=2,
    axes_dim=[16, 16, 16, 16],
    theta=10000,
    qkv_bias=True,
    guidance_embed=False,
)
```

**2. `Flux` 模型:**

```python
import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

class Flux(nn.Module):
    """
    用于序列流匹配的 Transformer 模型。
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} 必须能被 num_heads {params.num_heads} 整除"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"获得的 {params.axes_dim} 与预期的位置维度 {pe_dim} 不符")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("输入 img 和 txt 张量必须具有 3 个维度。")

        # 在序列 img 上运行
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("没有获得 guidance distilled 模型的 guidance 强度。")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
```

**描述:** `Flux` 模型是一个基于 Transformer 的架构，旨在用于序列流匹配。它接受图像、文本、时间步长和引导信号作为输入，并输出预测。

**关键组成部分:**

*   **`EmbedND`:** 用于生成位置嵌入。
*   **`MLPEmbedder`:** 用于嵌入时间步长、向量和引导信号。
*   **`DoubleStreamBlock`:** 用于处理图像和文本的双流 Transformer 块。
*   **`SingleStreamBlock`:** 用于处理连接后的图像和文本的单流 Transformer 块。
*   **`LastLayer`:** 用于将隐藏表示映射到输出。

**如何使用:**

1.  创建 `FluxParams` 实例。
2.  使用 `FluxParams` 实例创建 `Flux` 模型的实例。
3.  准备输入张量（`img`、`img_ids`、`txt`、`txt_ids`、`timesteps`、`y`、`guidance`）。
4.  将输入张量传递给 `forward` 方法。

**3. `FluxLoraWrapper`:**

```python
from flux.modules.lora import LinearLora, replace_linear_with_lora

class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
```

**描述:** `FluxLoraWrapper` 类是 `Flux` 模型的包装器，它添加了 LoRA (Low-Rank Adaptation) 功能。 LoRA 是一种参数高效的微调技术，它通过在原始模型中注入低秩矩阵来减少训练参数的数量。

**关键组成部分:**

*   **`replace_linear_with_lora`:** 此函数用 LoRA 线性层替换 `Flux` 模型中的所有线性层。
*   **`LinearLora`:** LoRA 线性层。
*   **`set_lora_scale`:** 用于设置 LoRA 缩放比例的函数。

**如何使用:**

1.  创建 `FluxParams` 实例。
2.  使用 `FluxParams` 实例创建 `FluxLoraWrapper` 模型的实例，指定 LoRA 秩和比例。
3.  像使用普通的 `Flux` 模型一样使用 `FluxLoraWrapper` 模型。
4.  可以使用 `set_lora_scale` 方法调整 LoRA 缩放比例。

**4. 模块 (`flux.modules.layers` 和 `flux.modules.lora`) 概述:**

虽然没有提供这些模块的具体代码，但我们可以推断它们的作用：

*   **`flux.modules.layers`:** 包含构建 `Flux` 模型所需的各种层，例如 `DoubleStreamBlock`、`EmbedND`、`LastLayer`、`MLPEmbedder`、`SingleStreamBlock` 和 `timestep_embedding`。这些层定义了模型的架构和功能。
*   **`flux.modules.lora`:** 包含 LoRA 相关的功能，例如 `LinearLora`（LoRA 线性层）和 `replace_linear_with_lora`（用 LoRA 层替换线性层的函数）。

**整体流程总结:**

1.  定义 `FluxParams` 以配置模型。
2.  使用 `FluxParams` 创建 `Flux` 模型或 `FluxLoraWrapper` 模型。
3.  准备输入数据 (图像、文本、时间步长等)。
4.  将输入数据传递给模型的 `forward` 方法。
5.  获得模型的输出。
6.  如果使用 `FluxLoraWrapper`，可以使用 `set_lora_scale` 方法调整 LoRA 缩放比例。

希望以上解释更清楚！
