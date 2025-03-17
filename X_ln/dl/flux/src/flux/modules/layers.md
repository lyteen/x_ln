Lan: `py` From`dl/flux/src\flux\modules\layers.py`

好的，我们重新开始。

**1. 位置编码模块 (EmbedND):**

```python
import torch
import torch.nn as nn
from torch import Tensor

from flux.math import rope  # 假设 rope 函数已定义

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim  # 编码的维度
        self.theta = theta # rope中的一个参数
        self.axes_dim = axes_dim # 每个轴的尺寸

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1] # 轴的数量
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3, # 连接维度
        )

        return emb.unsqueeze(1) #增加一个维度

# 演示如何使用这个模块
if __name__ == '__main__':
    # 假设已经定义了 rope 函数
    class DummyRope(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, ids, axis_dim, theta):
            # 模拟 rope 函数的功能，返回一个随机张量
            return torch.randn(ids.shape[0], 1, ids.shape[1], 64)

    rope = DummyRope()

    embed_nd = EmbedND(dim=64, theta=10000, axes_dim=[32, 32])
    ids = torch.randint(0, 32, (2, 16, 2))  # 示例输入 ids，形状为 (B, L, n_axes)
    embedding = embed_nd(ids)
    print(f"位置编码后的形状: {embedding.shape}")
```

**描述:** 这个模块实现了 N 维位置编码。 它使用 `rope` 函数 (假设已定义) 来生成每个轴的位置编码，然后将它们连接起来。 `axes_dim` 参数指定每个轴的尺寸，`ids` 输入是位置索引。

**如何使用:** 首先，初始化 `EmbedND` 类，指定维度、`theta` 和轴尺寸。 然后，将位置索引张量传递给 `forward` 方法。 该方法返回位置编码。位置编码被用于transformer的Attention模块。

**2. 时间步嵌入 (timestep_embedding):**

```python
import math
import torch
from torch import Tensor


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    创建正弦时间步嵌入。
    :param t: 一个一维张量，包含 N 个索引，每个批次元素一个。
              这些索引可以是小数。
    :param dim: 输出的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个 (N, D) 张量的位置嵌入。
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

# 演示如何使用时间步嵌入函数
if __name__ == '__main__':
    t = torch.arange(0, 10)  # 示例时间步
    embedding_dim = 32
    embeddings = timestep_embedding(t, embedding_dim)
    print(f"时间步嵌入的形状: {embeddings.shape}")  # 输出: 时间步嵌入的形状: torch.Size([10, 32])
```

**描述:** 此函数创建正弦时间步嵌入，用于在扩散模型中编码时间信息。它将时间步 `t` 转换为维度为 `dim` 的嵌入向量。正余弦函数的频率由 `max_period` 控制。

**如何使用:** 将时间步张量 `t` 和所需的维度 `dim` 传递给函数。 该函数返回时间步嵌入张量。

**3. MLP 嵌入器 (MLPEmbedder):**

```python
import torch
import torch.nn as nn
from torch import Tensor

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True) # 输入层
        self.silu = nn.SiLU() # SiLU 激活函数
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True) # 输出层

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x))) # 前向传播

# 演示如何使用 MLP 嵌入器
if __name__ == '__main__':
    mlp_embedder = MLPEmbedder(in_dim=128, hidden_dim=256)
    dummy_input = torch.randn(1, 128) # 示例输入
    embedding = mlp_embedder(dummy_input)
    print(f"MLP嵌入后的形状: {embedding.shape}")
```

**描述:**  这个模块使用多层感知器 (MLP) 将输入嵌入到更高维度的空间。 它包含一个输入线性层、一个 SiLU 激活函数和一个输出线性层。

**如何使用:**  初始化 `MLPEmbedder` 类，指定输入维度和隐藏维度。 然后，将输入张量传递给 `forward` 方法。 该方法返回 MLP 嵌入。通常，这个模块用于 time embedding 的处理.

**4. RMS 归一化 (RMSNorm):**

```python
import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6) # 计算 RMS
        return (x * rrms).to(dtype=x_dtype) * self.scale # 归一化

# 演示如何使用 RMS 归一化
if __name__ == '__main__':
    rms_norm = RMSNorm(dim=512)
    dummy_input = torch.randn(1, 10, 512)  # 示例输入
    normalized_input = rms_norm(dummy_input)
    print(f"RMS 归一化后的形状: {normalized_input.shape}")
```

**描述:**  RMS 归一化是一种归一化技术，它通过均方根 (RMS) 缩放输入。 它与层归一化类似，但没有学习到的偏置项。

**如何使用:**  初始化 `RMSNorm` 类，指定维度。 然后，将输入张量传递给 `forward` 方法。 该方法返回 RMS 归一化后的输出。

**5. QK 归一化 (QKNorm):**

```python
import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6) # 计算 RMS
        return (x * rrms).to(dtype=x_dtype) * self.scale # 归一化

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim) # Query 归一化
        self.key_norm = RMSNorm(dim) # Key 归一化

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q) # 归一化 Query
        k = self.key_norm(k) # 归一化 Key
        return q.to(v), k.to(v) #返回归一化之后的q,k，并且将dtype转化成v的dtype

# 演示如何使用 QK 归一化
if __name__ == '__main__':
    qk_norm = QKNorm(dim=64)
    dummy_q = torch.randn(1, 8, 64) # 示例 Query
    dummy_k = torch.randn(1, 8, 64) # 示例 Key
    dummy_v = torch.randn(1, 8, 64) # 示例 Value
    normalized_q, normalized_k = qk_norm(dummy_q, dummy_k, dummy_v)
    print(f"归一化后的 Query 形状: {normalized_q.shape}")
    print(f"归一化后的 Key 形状: {normalized_k.shape}")
```

**描述:** 此模块对 Query 和 Key 张量应用 RMS 归一化。 它使用单独的 `RMSNorm` 层来归一化 Query 和 Key。

**如何使用:** 初始化 `QKNorm` 类，指定维度。 然后，将 Query、Key 和 Value 张量传递给 `forward` 方法。 该方法返回归一化后的 Query 和 Key 张量。 一般在attention计算之前使用。

**6. 自注意力 (SelfAttention):**

```python
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from flux.math import attention # 假设 attention 函数已定义

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) # 可学习的缩放参数

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6) # 计算 RMS
        return (x * rrms).to(dtype=x_dtype) * self.scale # 归一化

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim) # Query 归一化
        self.key_norm = RMSNorm(dim) # Key 归一化

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q) # 归一化 Query
        k = self.key_norm(k) # 归一化 Key
        return q.to(v), k.to(v) #返回归一化之后的q,k，并且将dtype转化成v的dtype

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads # 每个头的维度

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # QKV 线性层
        self.norm = QKNorm(head_dim) # QK 归一化
        self.proj = nn.Linear(dim, dim) # 输出投影层

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x) # 计算 QKV
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads) # 调整形状
        q, k = self.norm(q, k, v) # 归一化 QK
        x = attention(q, k, v, pe=pe) # 计算注意力
        x = self.proj(x) # 投影
        return x

# 演示如何使用自注意力模块
if __name__ == '__main__':

    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, q, k, v, pe):
            return torch.randn(q.shape[1], q.shape[2], v.shape[-2], v.shape[-1])

    attention = DummyAttention()
    self_attention = SelfAttention(dim=256, num_heads=8)
    dummy_input = torch.randn(1, 16, 256)  # 示例输入
    dummy_pe = torch.randn(1, 8, 32, 32)
    attention_output = self_attention(dummy_input, dummy_pe)
    print(f"自注意力后的形状: {attention_output.shape}")
```

**描述:** 这个模块实现了自注意力机制。 它首先使用线性层计算 Query、Key 和 Value 张量。 然后，它使用 `QKNorm` 归一化 Query 和 Key。 接下来，它使用 `attention` 函数 (假设已定义) 计算注意力权重，并使用注意力权重聚合 Value 张量。 最后，它使用线性层将聚合后的 Value 张量投影到输出空间。

**如何使用:** 初始化 `SelfAttention` 类，指定维度和注意力头的数量。 然后，将输入张量和位置编码传递给 `forward` 方法。 该方法返回自注意力输出。

希望这个重启后的更详细的解释对您有所帮助！
