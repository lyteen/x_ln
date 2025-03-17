Lan: `py` From`dl/DeepSeek/native-sparse-attention-pytorch\test_triton_nsa.py`

好的，我们重新开始。

**目标:** 这段代码旨在比较PyTorch原生稀疏注意力机制与Triton实现的稀疏注意力机制的输出和梯度，以验证Triton实现的正确性。使用了`native_sparse_attention_pytorch`库。

**整体流程:**

1.  **定义`regular_attend`函数:** 使用PyTorch操作实现稀疏注意力，作为基准。
2.  **创建模拟输入:** 生成用于测试的随机张量。
3.  **计算前向和反向传播:** 分别使用`regular_attend` (PyTorch) 和 `native_sparse_attend` (Triton) 计算输出和梯度。
4.  **断言:** 比较两个实现的输出和梯度，检查它们是否足够接近。

下面分解代码，并附上中文解释和简单的示例。

**1. 导入必要的库:**

```python
from math import ceil
import torch

from native_sparse_attention_pytorch.native_sparse_attention import (
    create_sliding_mask,
    flex_attention
)

from native_sparse_attention_pytorch.triton_native_sparse_attention import (
    native_sparse_attend,
    round_up_multiple,
    pad_to_multiple,
)

import einx
from einops import rearrange, einsum, repeat

assert torch.cuda.is_available()
```

*   `torch`: PyTorch库，用于张量操作和神经网络。
*   `native_sparse_attention_pytorch`: 包含原生稀疏注意力实现的库。
*   `einx`, `einops`: 用于简化张量操作的库（例如，重塑和求和）。
*   `assert torch.cuda.is_available()`: 确保CUDA可用，因为Triton实现需要在GPU上运行。

**2. 辅助函数:**

```python
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def abs_diff(x, y):
    return (x - y).abs().amax()

def divisible_by(num, den):
    return (num % den) == 0
```

这些是实用函数，用于检查变量是否存在、提供默认值和计算绝对差异。它们提高了代码的可读性。

**3. `regular_attend` 函数 (PyTorch 原生稀疏注意力):**

```python
def regular_attend(
    q, k, v,
    indices,
    mask,
    block_size,
    sliding_window_size = None,
    sel_scale = None,
    return_lse = False,
    return_sliding_window_out = False
):
    q_heads, seq_len, kv_heads, device = q.shape[1], q.shape[-2], k.shape[1], q.device
    assert divisible_by(q_heads, kv_heads)

    if return_sliding_window_out:
        kv_seq_len = k.shape[-2]
        assert seq_len == kv_seq_len

        sliding_window_size = default(sliding_window_size, block_size)
        sliding_mask = create_sliding_mask(kv_seq_len, sliding_window_size)
        sliding_out = flex_attention(q, k, v, block_mask = sliding_mask, enable_gqa = True)

    q, k, v = tuple(pad_to_multiple(t, block_size, dim = -2) for t in (q, k, v))

    if exists(sel_scale):
        sel_scale = pad_to_multiple(sel_scale, block_size, dim = -2)

    g = q_heads // kv_heads # `g` stands for `g`roups of query heads per kv head

    w = ceil(seq_len / block_size)

    q, k, v = tuple(rearrange(t, 'b h (w n) d -> b h w n d', n = block_size) for t in (q, k, v))

    scale = q.shape[-1] ** -0.5

    q = rearrange(q, 'b (h g) ... -> b h g ...', g = g)

    # block causal diagonal

    sim = einsum(q, k, 'b h g w i d, b h w j d -> b h g w i j')
    causal_mask = torch.ones((block_size, block_size), device = device, dtype = torch.bool).triu(1)
    sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # rest of the indices

    num_sel_kv_blocks = indices.shape[-1]
    has_sel_kv_blocks = num_sel_kv_blocks > 0

    if has_sel_kv_blocks:
        indices, mask = tuple(pad_to_multiple(t, block_size, dim = -2) for t in (indices, mask))

        bk, bv = k, v
        sel_bk = einx.get_at('b h [w] n d, b h i sel -> b h i (sel n) d', bk, indices)
        sel_bv = einx.get_at('b h [w] n d, b h i sel -> b h i (sel n) d', bv, indices)

        q = rearrange(q, 'b h g w n d -> b h g (w n) d')
        bsim = einsum(q, sel_bk, 'b h g i d, b h i j d -> b h g i j')

        bsim = rearrange(bsim, 'b h g (w i) (sel j) -> b h g w i sel j', sel = num_sel_kv_blocks, i = fine_block_size)

        if exists(sel_scale):
            sel_scale = rearrange(sel_scale, 'b h (w i) sel -> b h w i sel', i = fine_block_size)
            bsim = einx.multiply('b h g w i sel j, b h w i sel -> b h g w i sel j', bsim, sel_scale)

        mask = rearrange(mask, 'b h (w i) sel -> b h 1 w i sel', i = fine_block_size)
        bsim = torch.where(mask[..., None], bsim, -torch.finfo(bsim.dtype).max)

        sim = rearrange(sim, 'b h g w i j -> b h g w i 1 j')

        sim = torch.cat((sim, bsim), dim = -2)
        sim = rearrange(sim, 'b h g w i causal_and_sel j -> b h g w i (causal_and_sel j)')

        sel_bv = rearrange(sel_bv, 'b h (w i) j d -> b h w i j d', i = fine_block_size)

        v = repeat(v, 'b h w j d -> b h w i j d', i = fine_block_size)
        v = torch.cat((v, sel_bv), dim = -2)
        v = rearrange(v, 'b h w i j d -> b h w i j d')

    # attend

    sim = sim * scale
    attn = sim.softmax(dim = -1)

    if has_sel_kv_blocks:
        out = einsum(attn, v, 'b h g w i j, b h w i j d -> b h g w i d')
    else:
        out = einsum(attn, v, 'b h g w i j, b h w j d -> b h g w i d')

    out = rearrange(out, 'b h g w n d -> b (h g) (w n) d')

    out = out[..., :seq_len, :]

    if return_sliding_window_out:
        out = (out, sliding_out)

    if not return_lse:
        return out

    lse = sim.logsumexp(dim = -1)
    lse = rearrange(lse, 'b g h w n -> b (g h) (w n)')
    lse = lse[..., :seq_len]

    return out, lse
```

*   **输入:** Queries (q), Keys (k), Values (v), indices (选择的key/value索引), mask, block_size。
*   **功能:**
    *   执行分块的稀疏注意力。
    *   `indices` 和 `mask` 指定要关注的键/值块。
    *   可选地，可以计算滑动窗口注意力作为补充。
*   **输出:** 注意力输出。

**关键步骤解释:**

*   **分块:** 输入被分成大小为 `block_size` 的块。`w = ceil(seq_len / block_size)` 计算块的数量。
*   **稀疏选择:** `indices` 用于从键和值中选择相关的块。
*   **注意力计算:** 使用 `einsum` 计算查询和键之间的相似度，然后进行 softmax 操作得到注意力权重。
*   **加权求和:** 使用注意力权重对值进行加权求和，得到最终输出。

**4. 创建模拟输入:**

```python
# mock inputs

batch = 4
seq_len = 507
q_heads = 4
kv_heads = 2
fine_block_size = 32
num_sel = 6
dim_head = 64
fused_sliding_window = False
block_dk_dv_use_dot = False # need sufficient shared memory, A100 works

q = torch.randn(batch, q_heads, seq_len, dim_head).cuda()
k = torch.randn(batch, kv_heads, seq_len, dim_head).cuda()
v = torch.randn(batch, kv_heads, seq_len, dim_head).cuda()

indices = torch.randint(0, 2, (batch, kv_heads, seq_len, num_sel)).cuda()
mask = torch.randint(0, 2, (batch, kv_heads, seq_len, num_sel)).bool().cuda()
sel_scale = torch.ones((batch, kv_heads, seq_len, num_sel)).cuda()
```

这段代码创建了一批随机张量，作为注意力机制的输入。这些张量包括queries (q), keys (k), values (v),  稀疏索引 (indices) 和 mask。所有张量都被移动到CUDA设备上。 `sel_scale` 用于缩放选择的KV块。

**5.  前向和反向传播 (PyTorch):**

```python
# both regular and nsa pathways `r` and `n`

rq, rk, rv, rsel_scale = tuple(t.clone().requires_grad_() for t in (q, k, v, sel_scale))
nq, nk, nv, nsel_scale = tuple(t.clone().requires_grad_() for t in (q, k, v, sel_scale))

# regular forwards and backwards

out, rlse = regular_attend(rq, rk, rv, indices, mask, block_size = fine_block_size, sel_scale = rsel_scale, return_lse = True, return_sliding_window_out = fused_sliding_window)

if fused_sliding_window:
    loss = sum(out).sum()
else:
    loss = out.sum()

loss.backward()
```

*   `rq, rk, rv, rsel_scale`:  创建输入的副本，并设置 `requires_grad=True`，以便计算梯度。
*   `out, rlse = regular_attend(...)`:  调用 `regular_attend` 函数计算输出和 log-sum-exp (lse)。
*   `loss = out.sum()`:  计算输出的总和作为损失函数。
*   `loss.backward()`:  计算梯度。

**6. 前向和反向传播 (Triton):**

```python
# triton nsa forwards and backwards

nsa_out, nlse = native_sparse_attend(nq, nk, nv, fine_block_size, indices, mask, sel_scale = nsel_scale, return_lse = True, block_dk_dv_use_dot = block_dk_dv_use_dot, return_sliding_window_out = fused_sliding_window)

if fused_sliding_window:
    nsa_loss = sum(nsa_out).sum()
else:
    nsa_loss = nsa_out.sum()

nsa_loss.backward()
```

*   `nsa_out, nlse = native_sparse_attend(...)`: 使用Triton实现的 `native_sparse_attend` 函数计算输出和 log-sum-exp。
*   `nsa_loss = nsa_out.sum()`: 计算损失。
*   `nsa_loss.backward()`: 计算梯度。

**7. 断言 (比较结果):**

```python
# asserts

if fused_sliding_window:
    out, sliding_out = out
    nsa_out, sliding_nsa_out = nsa_out
    assert torch.allclose(sliding_out, sliding_nsa_out, atol = 1e-2)

assert torch.allclose(out, nsa_out, atol = 1e-2)
assert torch.allclose(rlse, nlse, atol = 1e-2)

assert torch.allclose(rsel_scale.grad, nsel_scale.grad, atol = 2e-2)
assert torch.allclose(nv.grad, rv.grad, atol = 1e-2)
assert torch.allclose(nq.grad, rq.grad, atol = 1e-2)
assert torch.allclose(nk.grad, rk.grad, atol = 2e-2)

print('✅ outputs and gradients are same between pytorch native sparse attn and triton native sparse attn')
```

*   `torch.allclose(a, b, atol=1e-2)`:  检查两个张量 `a` 和 `b` 是否在绝对容差 `atol` 内相等。这里 `atol` 设置为 `1e-2` 或 `2e-2`。
*   **目的:** 比较 PyTorch 和 Triton 实现的输出和梯度，确保它们的结果一致。
*   **如果断言都通过，则打印成功消息。**

**简单示例 (概念):**

假设 `q` 代表一个句子中每个单词的查询向量，`k` 和 `v` 代表另一个句子中每个单词的键和值向量。 `indices` 可能指定，对于查询句子中的每个单词，只需要关注键值句子中的几个特定单词（例如，与该查询单词相关的单词）。`mask`可以过滤掉一些不应该被关注的块。

**总结:**

此代码是一个综合性的测试，用于验证 Triton 实现的稀疏注意力机制的正确性。通过比较 PyTorch 原生实现和 Triton 实现的输出和梯度，确保了 Triton 实现的准确性和可靠性。
