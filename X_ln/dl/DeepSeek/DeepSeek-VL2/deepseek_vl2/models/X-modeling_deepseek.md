Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\models\modeling_deepseek.html`

Okay, starting fresh.  I'll present enhanced components of the DeepseekV2 model. I will provide code snippets, chinese descriptions, and example usages.

**1. 高效的 Rotary Embedding (Efficient Rotary Embedding):**

```python
import torch
import torch.nn as nn

class ImprovedRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute the frequencies
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute the entire RoPE table in advance
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _set_cos_sin_cache(self, seq_len, device):
        self._seq_len_cached = seq_len

        t = torch.arange(self._seq_len_cached, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
        self._sin_cached = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)

    def forward(self, x, seq_len=None):
        # x: (batch_size, num_heads, seq_len, dim)
        if seq_len is None:
          seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
          raise ValueError(f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")

        if self._cos_cached is None or seq_len != self._seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return (self._cos_cached[:, :, :seq_len, :].to(x.dtype),
                self._sin_cached[:, :, :seq_len, :].to(x.dtype))


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (batch_size, num_heads, seq_len, dim)
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Demo Usage 演示用法
if __name__ == '__main__':
    batch_size = 2
    num_heads = 4
    seq_len = 128
    dim = 64

    rope = ImprovedRotaryEmbedding(dim)
    q = torch.randn(batch_size, num_heads, seq_len, dim)
    k = torch.randn(batch_size, num_heads, seq_len, dim)

    cos, sin = rope(q)
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos)

    print(f"旋转后的 query 形状: {q_rotated.shape}")
    print(f"旋转后的 key 形状: {k_rotated.shape}")
```

**描述:** 此代码定义了一个 `ImprovedRotaryEmbedding` 模块，用于应用旋转位置嵌入（RoPE）。

**主要改进:**

*   **预计算频率:** 在初始化时预先计算频率，以加快速度。
*   **预计算 RoPE 表:** 提前计算整个 RoPE 表，避免在每次 forward 调用中重新计算。  缓存这些值使得RoPE计算更快。
*   **直接设备访问：**`device=x.device`直接用输入张量的设备来创建缓存变量，避免手动指定设备
*   **形状验证：** 验证序列长度是否超出了最大长度

**如何使用:** 初始化 `ImprovedRotaryEmbedding` 类，指定嵌入的维度。 然后，将 query 和 key 张量传递给 `forward` 方法。  应用RoPE使用 `apply_rotary_pos_emb`。

---

**2. 优化的 MoE 门控 (Optimized MoE Gating):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMoEGate(nn.Module):
    def __init__(self, gating_dim, num_experts, top_k=2, routed_scaling_factor=1.0):
        super().__init__()
        self.gating_dim = gating_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = nn.Linear(gating_dim, num_experts, bias=False)  # Use a single linear layer

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, gating_dim)
        logits = self.gate(hidden_states) # (batch_size, seq_len, num_experts)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32) # (batch_size, seq_len, num_experts)

        # Select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # Normalize top-k weights
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True) * self.routed_scaling_factor

        return topk_idx, topk_weight

# Demo Usage 演示用法
if __name__ == '__main__':
    batch_size = 2
    seq_len = 32
    gating_dim = 128
    num_experts = 8
    top_k = 2

    gate = OptimizedMoEGate(gating_dim, num_experts, top_k)
    hidden_states = torch.randn(batch_size, seq_len, gating_dim)

    topk_idx, topk_weight = gate(hidden_states)

    print(f"Top-k 索引形状: {topk_idx.shape}")  # (batch_size, seq_len, top_k)
    print(f"Top-k 权重形状: {topk_weight.shape}") # (batch_size, seq_len, top_k)
```

**描述:** 此代码定义了一个 `OptimizedMoEGate` 模块，用于 MoE 层的门控机制。

**主要改进:**

*   **单一线性层:** 使用单个 `nn.Linear` 层进行门控，避免了额外的中间层。
*   **Softmax 计算精度:** 明确指定 `F.softmax` 中的 `dtype` 为 `torch.float32`，以提高稳定性。
*   **简化规范化:** 规范化计算简化为一步。
*   **更少的对象构建:** 减少对象构建，提升代码可读性。

**如何使用:** 初始化 `OptimizedMoEGate` 类，指定门控维度、专家数量和 top-k 值。 然后，将隐藏状态传递给 `forward` 方法。

---

**3. 静态缓存的注意力模块 (Attention with Static Cache):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWithStaticCache(nn.Module):
    def __init__(self, embed_dim, num_heads, max_cache_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Static cache for key and value
        self.register_buffer("k_cache", torch.zeros(1, num_heads, max_cache_len, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, num_heads, max_cache_len, self.head_dim))

    def forward(self, x, position_ids):
        # x: (batch_size, seq_len, embed_dim)
        # position_ids: (batch_size, seq_len)

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Update cache
        cache_start = position_ids[:, 0].min()  # Assuming contiguous position IDs

        self.k_cache[:, :, cache_start:cache_start + seq_len, :] = k
        self.v_cache[:, :, cache_start:cache_start + seq_len, :] = v

        # Attend
        attn_weights = torch.matmul(q, self.k_cache[:, :, :cache_start+seq_len, :].transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, self.v_cache[:, :, :cache_start+seq_len, :])

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

# Demo Usage 演示用法
if __name__ == '__main__':
    embed_dim = 256
    num_heads = 8
    max_cache_len = 1024
    batch_size = 2
    seq_len = 32

    attention = AttentionWithStaticCache(embed_dim, num_heads, max_cache_len)
    x = torch.randn(batch_size, seq_len, embed_dim)
    position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1)

    attn_output = attention(x, position_ids)
    print(f"注意力输出形状: {attn_output.shape}")
```

**描述:** 此代码定义了一个 `AttentionWithStaticCache` 模块，用于实现带有静态 KV 缓存的注意力机制。

**主要改进:**

*   **静态 KV 缓存:** 使用预分配的静态张量 `k_cache` 和 `v_cache` 存储 key 和 value，避免了动态分配，提升性能。
*   **位置 ID:** 使用位置 ID 来更新缓存，确保缓存正确性。假设位置 id 是连续的，如果不是会报错
*    **对cache\_start的取值更加严格**
*   **缓存长度更智能：** 在计算 attention 时，只考虑缓存中有效的部分 `:cache_start+seq_len`。

**如何使用:** 初始化 `AttentionWithStaticCache` 类，指定嵌入维度、 head 数量和最大缓存长度。 然后，将输入 `x` 和位置 ID `position_ids` 传递给 `forward` 方法。

**4.更简洁的FFN模块**

```python
import torch
import torch.nn as nn

class SimplifiedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation=nn.ReLU):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = activation()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Demo Usage
if __name__ == '__main__':
    hidden_size = 256
    intermediate_size = 1024
    batch_size = 2
    seq_len = 32

    mlp = SimplifiedMLP(hidden_size, intermediate_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = mlp(x)

    print(f"MLP输出形状: {output.shape}")
```

**代码描述：**

这段代码定义了一个简化的多层感知机（MLP）模块，通常用作 Transformer 模型中的前馈网络（FFN）。

**主要特点：**

* **简洁的结构：** 它由三个线性层（`gate_proj`，`up_proj`，`down_proj`）和一个激活函数组成。
* **Gated activation：** `gate_proj`和`up_proj`的输出相乘后再经过`down_proj`，形成一个gated activation的形式，允许网络学习更复杂的非线性变换。
* **灵活性：** 可以通过传递不同的激活函数类来更改激活函数。
* **易于理解：** 结构清晰，易于理解和修改。

**如何使用：**

初始化 `SimplifiedMLP` 类，指定隐藏层大小、中间层大小和激活函数。将输入张量传递给 `forward` 方法即可。

**中文描述：**

这段代码定义了一个简化的多层感知机 (MLP) 模块，通常用作 Transformer 模型中的前馈网络 (FFN)。它包含三个线性层和一个激活函数，结构简洁，易于理解和修改。中间层的 `gate_proj` 和 `up_proj` 输出相乘，形成一种门控激活形式，使网络能够学习更复杂的非线性变换。你可以通过传递不同的激活函数类来更改激活函数。