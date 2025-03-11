Lan: `py` From`dl/DeepSeek/native-sparse-attention-pytorch\native_sparse_attention_pytorch\transformer.py`

好的，重新开始。我将对提供的代码进行解释，并生成带有中文注释和简短描述的代码片段，并附上使用示例。

**1. 辅助函数 (Helper Functions):**

```python
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1
```

**描述:** 这些是一些实用函数。 `exists` 检查变量是否为 `None`。 `default` 如果第一个参数为 `None`，则返回第二个参数，否则返回第一个参数。`at_most_one_of` 检查 bools 中最多只有一个为真。

**用法示例:**

```python
x = None
y = 10
z = exists(x)  # z 是 False
w = default(x, y) # w 是 10
a = True
b = False
c = at_most_one_of(a, b)  # c 是 True
```

**2. Gumbel 采样 (Gumbel Sampling):**

```python
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs
```

**描述:** 这些函数用于 Gumbel-softmax 技巧，这是一种从分类分布中采样的常用方法，尤其是在生成模型中。 `gumbel_noise` 生成 Gumbel 噪声。 `gumbel_sample` 从 logits 中采样。`top_k`函数用于过滤概率分布，仅保留概率最高的 k 个选项，其余设置为负无穷大。这有助于控制生成样本的多样性。

**用法示例:**

```python
import torch

logits = torch.randn(1, 10)  # 示例 logits
sample = gumbel_sample(logits, temperature=0.7) # 使用 0.7 的温度进行采样
filtered_logits = top_k(logits, thres = 0.9) # 仅保留概率最高的 10%
print(f"采样结果的索引: {sample}")
```

**3. 注意力机制 (Attention):**

```python
class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True,
        kv_heads = None
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.heads = heads
        self.kv_heads = default(kv_heads, heads)
        dim_inner = heads * dim_head
        dim_kv_inner = kv_heads * dim_head

        self.causal = causal

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_kv_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_kv_inner, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x
    ):

        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # relative positions

        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        # naive gqa

        k, v = tuple(repeat(t, 'b h ... -> b (g h) ...', g = self.heads // self.kv_heads) for t in (k, v))

        # attention branch

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = self.causal
        )

        out = self.merge_heads(out)

        return self.to_out(out)
```

**描述:** 这是标准的多头注意力机制的一个实现，使用了 RMSNorm 进行归一化，Rotary Embedding 用于位置编码，并支持分组查询注意力 (GQA)。

**用法示例:**

```python
import torch
from rotary_embedding_torch import RotaryEmbedding

# 假设 RMSNorm 已经在您的环境中定义
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim = -1, keepdim = True) * self.scale
        return x / (norm + self.eps) * self.g

# 假设 Rearrange 已经在您的环境中定义
from einops.layers.torch import Rearrange
from einops import repeat

# 使用示例
dim = 256
attention = Attention(dim=dim, dim_head=64, heads=8, causal=True)
dummy_input = torch.randn(1, 32, dim)  # 批次大小 1，序列长度 32，维度 256
output = attention(dummy_input)
print(f"注意力机制输出形状: {output.shape}")  # 预期输出：torch.Size([1, 32, 256])
```

**4. 前馈网络 (FeedForward):**

```python
def FeedForward(dim, expansion_factor = 4.):
    dim_hidden = int(dim * expansion_factor)

    return nn.Sequential(
        RMSNorm(dim),
        Linear(dim, dim_hidden),
        nn.GELU(),
        Linear(dim_hidden, dim)
    )
```

**描述:** 这是带有 GELU 激活函数的简单前馈网络。 它用于在每个注意力层之后引入非线性。

**用法示例:**

```python
import torch

# 假设 RMSNorm 已经在您的环境中定义
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim = -1, keepdim = True) * self.scale
        return x / (norm + self.eps) * self.g

dim = 256
feedforward = FeedForward(dim=dim, expansion_factor=4.)
dummy_input = torch.randn(1, 32, dim)  # 批次大小 1，序列长度 32，维度 256
output = feedforward(dummy_input)
print(f"前馈网络输出形状: {output.shape}")  # 预期输出：torch.Size([1, 32, 256])
```

**5. Transformer 模型 (Transformer):**

```python
class Transformer(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        ff_expansion_factor = 4.,
        use_sparse_attn = False,
        causal = True,
        use_flex_sliding_window = False,
        use_flex_fine_selection = False,
        use_triton_fine_selection = False,
        sparse_attn_kwargs: dict = dict(
            sliding_window_size = 32,
            compress_block_size = 4,
            selection_block_size = 4,
            num_selected_blocks = 4,
        )
    ):
        super().__init__()
        assert at_most_one_of(use_flex_fine_selection, use_triton_fine_selection), 'either using flex or custom triton kernel for fine attn, but not both'

        self.token_emb = nn.Embedding(num_tokens, dim)

        if use_flex_sliding_window or use_flex_fine_selection:
            assert exists(flex_attention), 'flex attention is not available on your current version of pytorch'

        self.causal = causal

        self.use_sparse_attn = use_sparse_attn
        self.use_flex_sliding_window = use_sparse_attn & use_flex_sliding_window
        self.use_flex_fine_selection = use_sparse_attn & use_flex_fine_selection

        layers = []
        for _ in range(depth):

            if use_sparse_attn:
                attn = SparseAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    kv_heads = kv_heads,
                    causal = causal,
                    use_triton_kernel = use_triton_fine_selection,
                    **sparse_attn_kwargs
                )
            else:
                attn = Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    causal = causal,
                    kv_heads = kv_heads
                )

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            layers.append(ModuleList([attn, ff]))

        self.attn_sliding_window_size = getattr(attn, 'sliding_window_size', None)
        self.attn_fine_block_size = getattr(attn, 'selection_block_size', None)

        self.layers = ModuleList(layers)

        self.norm = RMSNorm(dim)
        self.to_logits = Linear(dim, num_tokens, bias = False)

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
        use_cache_kv = False
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        cache = None

        for ind in tqdm(range(sample_num_times)):
            is_first = ind == 0

            logits, next_cache = self.forward(
                out,
                cache = cache,
                return_cache = True,
                disable_flex = True,
                disable_triton_kernel = not is_first
            )

            if use_cache_kv:
                cache = next_cache

            logits = logits[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    def forward_inference(
        self,
        ids,
        cache = None
    ):
        return ids

    def forward(
        self,
        ids,
        return_loss = False,
        disable_flex = False,
        disable_triton_kernel = False,
        cache = None,
        return_cache = False
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            disable_flex |= True
            disable_triton_kernel |= True

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        seq_len = ids.shape[-1]

        # token embedding

        tokens = self.token_emb(ids)

        # prepare maybe flex attention masks

        attn_kwargs = dict(
            disable_triton_kernel = disable_triton_kernel
        )

        if not disable_flex and self.use_flex_sliding_window:
            attn_kwargs.update(
                sliding_window_flex_mask = create_sliding_mask(seq_len, self.attn_sliding_window_size, causal = self.causal)
            )

        if not disable_flex and self.use_flex_fine_selection:
            attn_kwargs.update(
                fine_selection_flex_mask = create_fine_mask(seq_len, self.attn_fine_block_size, causal = self.causal)
            )

        # cache

        cache = default(cache, [])
        iter_cache = iter(cache)

        next_cache = []

        if is_inferencing:
            tokens = tokens[:, -1:]

        # layers

        for attn, ff in self.layers:

            attn_out, layer_cache = attn(
                tokens,
                cache = next(iter_cache, None),
                return_cache = True,
                **attn_kwargs
            )

            next_cache.append(layer_cache)

            tokens = attn_out + tokens
            tokens = ff(tokens) + tokens

        embed = self.norm(tokens)

        logits = self.to_logits(embed)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
```

**描述:** 这是 Transformer 模型的完整实现。 它包括 token 嵌入、多层注意力块、前馈网络和最终的 logits 映射。 它还支持使用稀疏注意力以提高效率。该类具有`sample`方法，该方法可以生成文本

**用法示例:**

```python
import torch

# 假设 RMSNorm 已经在您的环境中定义
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim = -1, keepdim = True) * self.scale
        return x / (norm + self.eps) * self.g

# 使用示例
num_tokens = 10000
dim = 256
depth = 6

transformer = Transformer(
    num_tokens = num_tokens,
    dim = dim,
    depth = depth,
    dim_head = 64,
    heads = 8
)

dummy_input = torch.randint(0, num_tokens, (1, 32))  # 批次大小 1，序列长度 32
logits = transformer(dummy_input)
print(f"Transformer 输出形状: {logits.shape}")  # 预期输出：torch.Size([1, 32, 10000])

# 文本生成示例
prompt = torch.randint(0, num_tokens, (1, 10)) # 初始prompt
generated_text = transformer.sample(prompt, seq_len = 50) # 生成50个token
print(f"生成的文本: {generated_text}")
```

**总结:**

这段代码实现了一个灵活的 Transformer 模型，具有标准的多头注意力、稀疏注意力支持以及 Gumbel 采样等功能。 通过调整参数，可以针对不同的任务和数据集优化模型。 希望这些详细的解释和示例可以帮助您理解和使用这段代码。
