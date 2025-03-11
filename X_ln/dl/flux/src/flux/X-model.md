Lan: `py` From`dl/flux/src\flux\model.py`

好的，这次我将以更注重改进原有代码的方式，并结合中文描述和演示来重新组织答案。

**目标：**  专注于对现有 `Flux` 和 `FluxLoraWrapper` 类的改进，提供更高效、更灵活的设计。

**1. 改进的 `Flux` 模型:**

```python
from dataclasses import dataclass

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
from flux.modules.lora import LinearLora, replace_linear_with_lora


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    use_checkpointing: bool = False # NEW: 添加检查点标志


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
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
        self.use_checkpointing = params.use_checkpointing # 存储检查点标志

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
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # 使用检查点包装器
        if self.use_checkpointing and self.training:
          img, txt = self._checkpointed_double_blocks(img, txt, vec, pe)
          img = self._checkpointed_single_blocks(torch.cat((txt, img), 1), vec, pe)[:, txt.shape[1] :, ...] # 裁剪txt部分
        else:
          for block in self.double_blocks:
              img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

          img = torch.cat((txt, img), 1)
          for block in self.single_blocks:
              img = block(img, vec=vec, pe=pe)
          img = img[:, txt.shape[1] :, ...] # 裁剪txt部分

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def _checkpointed_double_blocks(self, img, txt, vec, pe):
        """使用检查点包装器运行双流块。"""
        from torch.utils.checkpoint import checkpoint_sequence

        def run_block(block, img, txt, vec, pe):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            return img, txt

        segments = len(self.double_blocks)
        return checkpoint_sequence([lambda img, txt, vec, pe, block=block: run_block(block, img, txt, vec, pe) for block in self.double_blocks],
                                  segments=segments,
                                  img=img,
                                  txt=txt,
                                  vec=vec,
                                  pe=pe)
    def _checkpointed_single_blocks(self, img, vec, pe):
        """使用检查点包装器运行单流块。"""
        from torch.utils.checkpoint import checkpoint_sequence

        def run_block(block, img, vec, pe):
            img = block(img, vec=vec, pe=pe)
            return img

        segments = len(self.single_blocks)
        return checkpoint_sequence([lambda img, vec, pe, block=block: run_block(block, img, vec, pe) for block in self.single_blocks],
                                  segments=segments,
                                  img=img,
                                  vec=vec,
                                  pe=pe)

```

**改进说明:**

*   **Checkpointing (检查点):** 添加了`use_checkpointing`标志到`FluxParams`和`Flux`类中。当设置为`True`时，在训练过程中会使用`torch.utils.checkpoint.checkpoint_sequence`来减少内存占用。这对于非常深的模型特别有用。我创建了`_checkpointed_double_blocks`和`_checkpointed_single_blocks`方法来包装模型的前向传播过程，使其与检查点机制兼容。
*   **Txt裁剪**: 修正了在经过`single_blocks`后图像`img`的裁剪， 确保文本`txt`部分被正确去除。

**如何使用:**

1.  在 `FluxParams` 中设置 `use_checkpointing=True`。
2.  确保你的训练循环中包含 `model.train()` 和 `model.eval()`，以便启用或禁用检查点。

**代码解释 (中文):**

这段代码的主要改进是加入了检查点机制。检查点（Checkpointing）是一种用于减少深度学习模型训练时内存消耗的技术。它通过在计算图中的某些点（即检查点）保存激活值，并在需要时重新计算它们，而不是一直将所有激活值都保存在内存中。这样可以显著降低内存需求，但会略微增加计算时间。

**2. 改进的 `FluxLoraWrapper` 模型:**

```python
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
        self.lora_modules = replace_linear_with_lora( # 保存LoRA模块
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.lora_modules:  # 仅迭代LoRA模块
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)

    def disable_lora(self):  # 添加禁用LoRA的方法
        for module in self.lora_modules:
            if isinstance(module, LinearLora):
                module.disable()

    def enable_lora(self):  # 添加启用LoRA的方法
        for module in self.lora_modules:
            if isinstance(module, LinearLora):
                module.enable()

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
```

**改进说明:**

*   **LoRA Module Tracking (LoRA 模块跟踪):** 现在 `FluxLoraWrapper` 会跟踪所有被 LoRA 替换的模块。这使得更容易控制 LoRA 模块，例如，一次性设置所有 LoRA 模块的缩放比例。
*   **Enable/Disable LoRA (启用/禁用 LoRA):**  添加了 `disable_lora` 和 `enable_lora` 方法，允许在不移除 LoRA 模块的情况下禁用或启用它们。这对于实验或微调非常有用。
*   **Print Trainable Parameters (打印可训练参数):** 添加 `print_trainable_parameters`  方法来打印模型中可训练参数的数量，方便调试和了解 LoRA 的效果。

**代码解释 (中文):**

`FluxLoraWrapper` 的改进主要集中在更好地管理和控制 LoRA 模块。通过跟踪 LoRA 模块，我们可以方便地批量设置它们的缩放比例或启用/禁用它们。`disable_lora` 和 `enable_lora` 方法提供了一种在不删除 LoRA 模块的情况下，快速切换 LoRA 效果的方式。 `print_trainable_parameters` 方便查看lora之后的训练参数量

**3. 演示:**

```python
# 演示如何使用改进的类
if __name__ == '__main__':
    # 1. 创建 FluxParams 实例
    params = FluxParams(
        in_channels=3,
        out_channels=3,
        vec_in_dim=128,
        context_in_dim=512,
        hidden_size=256,
        mlp_ratio=4.0,
        num_heads=8,
        depth=2,
        depth_single_blocks=1,
        axes_dim=[16, 16],
        theta=1.0,
        qkv_bias=True,
        guidance_embed=False,
        use_checkpointing=False
    )

    # 2. 创建 FluxLoraWrapper 实例
    model = FluxLoraWrapper(params=params, lora_rank=32, lora_scale=0.8)

    # 3. 打印可训练参数
    model.print_trainable_parameters()

    # 4. 禁用 LoRA
    model.disable_lora()
    print("禁用 LoRA 后:")
    model.print_trainable_parameters()

    # 5. 启用 LoRA
    model.enable_lora()
    print("启用 LoRA 后:")
    model.print_trainable_parameters()

    # 6. 设置 LoRA 缩放比例
    model.set_lora_scale(0.5)

    # 7. 创建虚拟输入
    batch_size = 2
    seq_len = 16
    img = torch.randn(batch_size, seq_len, params.in_channels)
    img_ids = torch.randint(0, 10, (batch_size, seq_len))
    txt = torch.randn(batch_size, seq_len, params.context_in_dim)
    txt_ids = torch.randint(0, 10, (batch_size, seq_len))
    timesteps = torch.rand(batch_size)
    y = torch.randn(batch_size, params.vec_in_dim)
    guidance = None  # 或者 torch.rand(batch_size) if params.guidance_embed is True

    # 8. 前向传播
    output = model(img, img_ids, txt, txt_ids, timesteps, y, guidance)
    print("输出形状:", output.shape)
```

**演示说明:**

这段演示代码展示了如何创建 `FluxLoraWrapper` 实例，打印可训练参数，禁用/启用 LoRA，设置 LoRA 缩放比例，以及进行前向传播。  您可以根据需要修改演示代码来测试不同的功能。

**总结:**

这些改进旨在使 `Flux` 和 `FluxLoraWrapper` 模型更灵活、更易于使用，并提供更好的内存管理能力。 检查点和 LoRA 控制功能的结合，让你能够更好地调整模型，以适应不同的训练场景和资源限制。
