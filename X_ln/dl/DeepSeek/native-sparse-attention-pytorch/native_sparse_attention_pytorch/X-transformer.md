Lan: `py` From`dl/DeepSeek/native-sparse-attention-pytorch\native_sparse_attention_pytorch\transformer.py`

好的，现在重新开始，我会提供一个更智能的版本，并附带小段代码描述和简单的中文演示。

**1. 改进的 Rotary Embedding (旋转位置编码):**

```python
import torch
from torch import nn
import math

class ImprovedRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 计算频率
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预先计算旋转矩阵
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, dim]
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            self.cos_cached = freqs.cos()[None, :, :]
            self.sin_cached = freqs.sin()[None, :, :]

        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]

    @staticmethod
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, seq_len=None):
        cos, sin = self.forward(q, seq_len=seq_len)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

# Demo 演示
if __name__ == '__main__':
    batch_size = 2
    seq_len = 128
    dim = 64
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    rotary_emb = ImprovedRotaryEmbedding(dim)
    q_embed, k_embed = rotary_emb.apply_rotary_pos_emb(q, k)

    print(f"原始 Query 形状: {q.shape}")
    print(f"旋转后 Query 形状: {q_embed.shape}")
    print(f"原始 Key 形状: {k.shape}")
    print(f"旋转后 Key 形状: {k_embed.shape}")
```

**描述:**  这段代码定义了一个改进的旋转位置编码模块 `ImprovedRotaryEmbedding`。

**主要改进:**

*   **预先计算旋转矩阵:**  为了加速计算，预先计算了旋转矩阵的 cosine 和 sine 值，并缓存起来。
*   **动态序列长度:**  可以处理不同长度的序列，并且只在需要时才更新缓存。
*   **清晰的旋转函数:** `rotate_half` 函数更清晰地实现了向量的旋转。

**如何使用:**  初始化 `ImprovedRotaryEmbedding` 类，指定维度。 然后，将 Query (Q) 和 Key (K) 传递给 `apply_rotary_pos_emb` 方法。

---

**2. 改进的 Attention (注意力机制):**

```python
import torch
from torch import nn
import torch.nn.functional as F

class ImprovedAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=True, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.norm = nn.LayerNorm(dim)  # Use LayerNorm instead of RMSNorm
        self.qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)  # Combine Q, K, V
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim_head * heads, dim)
        self.proj_dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        x = self.norm(x)
        b, n, _ = x.shape

        qkv = self.qkv(x).reshape(b, n, self.heads, self.dim_head, 3).permute(0, 2, 1, 3, 4) # [B, H, N, D, 3]
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:  # Add mask support
            attn = attn.masked_fill(mask == 0, float('-inf'))

        if self.causal:
            causal_mask = torch.triu(torch.ones((n, n), device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v).reshape(b, n, self.heads * self.dim_head)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out

# Demo 演示
if __name__ == '__main__':
    batch_size = 2
    seq_len = 128
    dim = 64

    x = torch.randn(batch_size, seq_len, dim)
    attention = ImprovedAttention(dim, dim_head=32, heads=2, causal=True, dropout=0.1)  # Example usage
    output = attention(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

**描述:** 这段代码定义了一个改进的注意力机制模块 `ImprovedAttention`。

**主要改进:**

*   **Layer Normalization:** 使用 `LayerNorm` 替换了 `RMSNorm`。`LayerNorm`在某些情况下可能表现更好。
*   **QKV 组合:**  使用一个线性层来生成 Q, K, V，而不是三个独立的层，这可以提高效率。
*   **Dropout:**  在 Attention 和 Projection 之后添加了 Dropout 层，以防止过拟合。
*   **Mask 支持:**  增加了对 Mask 的支持，允许在注意力计算中屏蔽某些 token。

**如何使用:**  初始化 `ImprovedAttention` 类，指定维度、头数等参数。 然后，将输入 `x` 传递给 `forward` 方法。

---

**3. 改进的 FeedForward (前馈网络):**

```python
import torch
from torch import nn

class ImprovedFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), # Use LayerNorm
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Demo 演示
if __name__ == '__main__':
    batch_size = 2
    seq_len = 128
    dim = 64

    x = torch.randn(batch_size, seq_len, dim)
    feedforward = ImprovedFeedForward(dim, expansion_factor=2.0, dropout=0.1)
    output = feedforward(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

**描述:** 这段代码定义了一个改进的前馈网络模块 `ImprovedFeedForward`。

**主要改进:**

*   **Layer Normalization:** 使用 `LayerNorm` 而不是 `RMSNorm`。
*   **Dropout:**  在两个线性层之后都添加了 Dropout 层，以防止过拟合。

**如何使用:**  初始化 `ImprovedFeedForward` 类，指定维度和扩展因子。 然后，将输入 `x` 传递给 `forward` 方法。

这些改进后的模块可以组合成一个更强大和灵活的 Transformer 模型。  在实际使用中，需要根据具体任务和数据集调整参数。
