Lan: `py` From`dl/DeepSeek/native-sparse-attention-pytorch\native_sparse_attention_pytorch\native_sparse_attention.py`


**1. 辅助函数 (Helper Functions):**

```python
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def divisible_by(num, den):
    return (num % den) == 0

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)
    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def interpolate_1d(x, length, mode = 'bilinear'):
    x, inverse_pack = pack_one_with_inverse(x, '* n')
    x = rearrange(x, 'b n -> b 1 n 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    x = rearrange(x, 'b 1 n 1 -> b n')
    return inverse_pack(x)

def straight_through(t, target):
    return t + (target - t).detach()
```

**描述:** 这段代码定义了一系列辅助函数，用于处理常见的任务，例如检查变量是否存在(`exists`)，提供默认值(`default`)，将数字四舍五入到最接近的倍数(`round_down_mult`, `round_up_mult`)，检查可除性(`divisible_by`)，找到最大负值(`max_neg_value`)，以及处理张量的填充和插值(`pad_at_dim`, `interpolate_1d`)。`pack_one_with_inverse`和`straight_through`用于更高级的张量操作和梯度处理。

**如何使用:** 这些函数被 `SparseAttention` 类内部广泛使用，以简化代码并提高可读性。 例如，`default` 用于为可选参数提供默认值，而 `pad_at_dim` 用于在进行 attention 计算之前填充张量。

**2. Attention 函数 (Attend Function):**

```python
def attend(
    q, k, v,
    mask = None,
    return_sim = False,
    scale = None
):
    scale = default(scale, q.shape[-1] ** -0.5)

    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads

    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh = num_grouped_queries)

    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale

    mask_value = max_neg_value(sim)

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value // 10)

    attn = sim.softmax(dim = -1)

    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')

    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')

    if not return_sim:
        return attn_out

    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')

    return attn_out, sim
```

**描述:** 这个函数实现了标准的 scaled dot-product attention。 它接收查询 (q)、键 (k) 和值 (v) 张量，以及一个可选的掩码 (mask)。 该函数首先计算 q 和 k 之间的相似度矩阵 (sim)，然后应用掩码（如果提供），然后对相似度矩阵执行 softmax 操作以获得 attention 权重 (attn)。 最后，它使用 attention 权重对 v 进行加权，并返回 attention 输出。`num_grouped_queries` 处理了 Grouped-Query Attention (GQA)。

**如何使用:**  `SparseAttention` 类中的多个 attention 机制会调用此函数。 它封装了 attention 计算的核心逻辑。

**3. SparseAttention 类:**

这是代码的核心，下面分解关键部分：

```python
class SparseAttention(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        sliding_window_size,
        compress_block_size,
        selection_block_size,
        num_selected_blocks,
        kv_heads = None,
        num_compressed_mem_kv = 1,
        causal = False,
        norm = True,
        use_diff_topk = False,
        use_triton_kernel = False,
        interpolated_importance_score = False,
        query_heads_share_selected_kv = True, # if set to True, importance score is averaged across query heads to select top-n buckets of kv per kv head - but can be set to False for each query head within a group to look at different sets of kv buckets. will be more memory and compute of course
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor = 1.,
        strategy_combine_mlp: Module | None = None
    ):
        super().__init__()

        # attention heads
        # handling gqa if `kv_heads` is set

        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)

        self.heads = heads
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads

        # scale

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # autoregressive or not - will extend this work for long context video / genomics use-cases

        self.causal = causal

        # rotary

        self.rotary_emb = RotaryEmbedding(dim_head)

        # qkv

        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)

        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)

        self.qkv_split = qkv_split

        # sliding window strategy

        self.sliding_window = LocalAttention(
            dim = dim_head,
            window_size = sliding_window_size,
            causal = causal,
            exact_windowsize = True,
            autopad = True,
            use_rotary_pos_emb = False
        )

        self.sliding_window_size = sliding_window_size

        # compress strategy

        self.compress_block_size = compress_block_size

        assert num_compressed_mem_kv > 0

        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n = compress_block_size)

        self.num_mem_compress_kv = num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))

        if not exists(compress_mlp):
            compress_dim = compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)

            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # selection related

        self.use_diff_topk = use_diff_topk

        self.interpolated_importance_score = interpolated_importance_score # in the case fine block size < compressed block size, will weigh space better when selecting

        self.query_heads_share_selected_kv = query_heads_share_selected_kv

        self.selection_block_size = selection_block_size

        assert num_selected_blocks >= 0

        if num_selected_blocks == 0:
            print(f'`num_selected_blocks` should be set greater than 0, unless if you are ablating it for experimental purposes')

        self.num_selected_blocks = num_selected_blocks

        self.use_triton_kernel = use_triton_kernel

        # they combine the three sparse branches through a learned combine with sigmoid activation

        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 3 * heads)

            # init to sliding windows first, as network tends to pick up on local patterns first before distant ones

            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * heads))

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # combining heads

        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)
```

**a. `__init__` (初始化):**

   - 初始化各种参数，包括维度(`dim`, `dim_head`)、attention heads (`heads`, `kv_heads`)、窗口大小(`sliding_window_size`)、压缩块大小(`compress_block_size`)、选择块大小(`selection_block_size`)和要选择的块数 (`num_selected_blocks`)。
   - 设置各种策略的开关，如 `use_diff_topk`, `use_triton_kernel`, `interpolated_importance_score`, `query_heads_share_selected_kv`。
   - 定义了三个关键的 attention 策略：`sliding_window` (局部注意力), `compress` (压缩注意力), 和 `select` (选择性注意力)。
   - 使用 `nn.Linear` 和 `Rearrange` 等模块构建各种线性层和张量重塑操作。
   - `RotaryEmbedding` 用于编码位置信息。
   - `strategy_combine_mlp` 用于学习如何组合来自不同 attention 策略的输出。

**描述:** 初始化函数定义了SparseAttention模块的结构和超参数。它设置了所有必要的组件，包括线性层，注意力机制和各种策略参数。
*   **参数含义:**
    *   `dim`: 输入特征的维度。
    *   `dim_head`: 每个注意头部的维度。
    *   `heads`: 注意头部的数量。
    *   `sliding_window_size`: 滑动窗口的大小。
    *   `compress_block_size`: 压缩块的大小。
    *   `selection_block_size`: 选择块的大小。
    *   `num_selected_blocks`: 要选择的块的数量。
    *   `kv_heads`: key/value 注意头部的数量，用于 GQA。
    *   `num_compressed_mem_kv`: 压缩的 memory key/value 的数量。
    *   `causal`: 是否使用因果注意力。
    *   `norm`: 是否使用 RMSNorm 归一化。
    *   `use_diff_topk`: 是否使用 differential topk。
    *   `use_triton_kernel`: 是否使用 Triton kernel。
    *   `interpolated_importance_score`: 是否使用插值来计算重要性得分。
    *   `query_heads_share_selected_kv`: 查询头部是否共享选择的 key/value。
    *   `compress_mlp`: 用于压缩 key/value 的 MLP。
    *   `compress_mlp_expand_factor`: 压缩 MLP 的扩展因子。
    *   `strategy_combine_mlp`: 用于组合不同注意力策略的 MLP。

**b. `forward` (前向传播):**

```python
    def forward(
        self,
        inp,
        cache = None,
        disable_triton_kernel = False,
        sliding_window_flex_mask = None,
        fine_selection_flex_mask = None,
        return_cache = False
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            assert inp.shape[1] == 1, 'input must be single tokens if inferencing with cache key values'
            return self.forward_inference(inp, cache, return_cache = return_cache)

        assert not (not self.causal and return_cache)

        batch, seq_len, scale, heads, device = *inp.shape[:2], self.scale, self.heads, inp.device

        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm

        inp = self.norm(inp)

        # queries, keys, values

        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        # compressed key / values - variables prepended with `c` stands for compressed

        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)

        k_compress_input = self.split_compress_window(k[..., :compress_divisible_seq_len, :] + k_pos)
        v_compress_input = self.split_compress_window(v[..., :compress_divisible_seq_len, :] + v_pos)

        run_k = k[..., compress_divisible_seq_len:, :]
        run_v = v[..., compress_divisible_seq_len:, :]

        cq = q
        ck = self.k_compress(k_compress_input)   # Equation (7) of the Native Sparse Attention paper
        cv = self.v_compress(v_compress_input)

        if return_cache:
            cache_compressed_kv = ((ck, cv), (run_k, run_v))

        # 1. coarse attention over compressed

        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = batch)

        num_mem_compress_kv = mem_ck.shape[-2]

        ck = cat((mem_ck, ck), dim = -2)
        cv = cat((mem_cv, cv), dim = -2)

        # compressed masking

        cmask = None

        if self.causal:
            cq_seq = arange(seq_len, device = device)
            ck_seq = ((arange(num_compress_blocks, device = device) + 1) * self.compress_block_size) - 1
            ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value = -1)

            cmask = einx.less('j, i -> i j', ck_seq, cq_seq)

        compressed_attn_out, csim = attend(cq, ck, cv, mask = cmask, return_sim = True)

        # for 2. and 3., will give them relative positions with rotary - compressed needs to be handled separately (even if they already have intra block absolute positions)

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # handle cache

        if return_cache:
            cache_kv = (k, v)

        # 2. fine attention over selected based on compressed attention logits - variables prepended with `f` stands for the fine attention pathway

        importance_scores = csim[..., num_mem_compress_kv:]

        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # maybe average the compressed attention across each grouped queries (per key / values)

        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)

            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1

        # handle if compress block size does not equal to the fine block size
        # cannot parse their equation, so will just improvise
        # first we expand all the compressed scores to the full sequence length, then average within each fine / selection block size - pad on the right to 0s, which should be fine as sliding window convers the local anyways

        if has_selected_kv_for_fine_attn:

            if self.compress_block_size != self.selection_block_size:

                compress_seq_len = num_compress_blocks * self.compress_block_size

                if self.interpolated_importance_score:
                    importance_scores = interpolate_1d(importance_scores, compress_seq_len)
                else:
                    importance_scores = repeat(importance_scores, '... j -> ... (j block_size)', block_size = self.compress_block_size)

                padding = fine_divisible_seq_len - compress_seq_len

                fine_query_seq_len = importance_scores.shape[-2]
                fine_query_padding = fine_divisible_seq_len - importance_scores.shape[-2]

                importance_scores = F.pad(importance_scores, (0, padding))

                # mask out the diagonal since block causal is included by default for fine attending

                block_causal_mask = torch.ones((num_fine_blocks,) * 2, device = device, dtype = torch.bool).tril(-1)
                block_causal_mask = repeat(block_causal_mask, 'i j -> (i n1) (j n2)', n1 = self.selection_block_size, n2 = self.selection_block_size)
                block_causal_mask = block_causal_mask[:fine_query_seq_len]

                importance_scores = importance_scores.masked_fill(~block_causal_mask, max_neg_value(csim))

                importance_scores = reduce(importance_scores, '... (j block_size) -> ... j', 'mean', block_size = self.selection_block_size)

            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]

        # handle if number of total blocks is less than number to select for fine attention

        fq = q
        fk = k
        fv = v

        if has_selected_kv_for_fine_attn:

            # get the top-n kv segments for fine attention

            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)

            gates = None

            if self.use_diff_topk:
                gates = straight_through(selected_importance_values, 1.)

            if self.use_triton_kernel and not disable_triton_kernel:

                from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend

                fmask = selected_importance_values > 1e-10

                fine_attn_out = native_sparse_attend(
                    fq, fk, fv,
                    self.selection_block_size,
                    selected_block_indices,
                    fmask,
                    sel_scale = gates,
                    include_block_causal = self.causal
                )

            elif exists(fine_selection_flex_mask):
                assert not self.use_diff_topk, 'differential topk is not available for flex attention'

                # flex attention for the selection for fine attention

                fine_block_mask = fine_selection_flex_mask(selected_block_indices, num_grouped_queries = fine_num_grouped_queries)

                fine_attn_out = flex_attention(fq, fk, fv, block_mask = fine_block_mask, enable_gqa = True)

            else:
                fmask = selected_importance_values > 1e-10

                if seq_len < fine_divisible_seq_len:
                    remainder = fine_divisible_seq_len - seq_len
                    fk = pad_at_dim(fk, (0, remainder), value = 0., dim = -2)
                    fv = pad_at_dim(fv, (0, remainder), value = 0., dim = -2)
                    fq = pad_at_dim(fq, (0, remainder), value = 0., dim = -2)

                    fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)

                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

                    if exists(gates):
                        gates = pad_at_dim(gates, (0, remainder), value = 0, dim = -2)

                if self.causal:
                    # handle block causal diagonal in the diagram, but run experiments without to see

                    fine_window_seq = arange(fine_divisible_seq_len, device = device) // self.selection_block_size
                    fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b = batch, h = selected_block_indices.shape[1])
                    selected_block_indices = cat((selected_block_indices, fine_window_seq), dim = -1) # for the block causal diagonal in fig2

                    fmask = repeat(fmask, 'b h i w -> b h i w j', j = self.selection_block_size)

                    causal_mask = torch.ones((self.selection_block_size,) * 2, device = device, dtype = torch.bool).tril()
                    causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w = num_fine_blocks, b = batch, h = fmask.shape[1])

                    fmask = cat((fmask, causal_mask), dim = -2)
                    fmask = rearrange(fmask, 'b h i w j -> b h 1 i (w j)')

                else:
                    fmask = repeat(fmask, 'b h i w -> b h 1 i (w j)', j = self.selection_block_size)

                # select out the spatial crops of keys / values for fine attention

                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

                # get_at("b h [w] j d, b h i selected -> b h i selected j d", fkv, selected_block_indices)

                if self.query_heads_share_selected_kv:
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                else:
                    fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                    fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])

                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)

                # differential topk gating

                if self.use_diff_topk:
                    if self.causal:
                        gates = F.pad(gates, (0, 1), value = 1.)

                    fk = einx.multiply('b h i sel, b h i sel j d -> b h i sel j d', gates, fk)

                # merge selected key values

                fk, fv = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))

                # fine attention

                fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh = fine_num_grouped_queries)

                fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * self.scale

                mask_value = max_neg_value(fsim)

                fsim = fsim.masked_fill(~fmask, mask_value)

                fattn = fsim.softmax(dim = -1)

                fine_attn_out = einsum(fattn, fv, 'b h qh i j, b h i j d -> b h qh i d')

                fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')

                fine_attn_out = fine_attn_out[..., :seq_len, :]

        else:
            # if only first block, just do a simple block causal

            seq_len = fk.shape[-2]
            fmask = None

            if self.causal:
                fmask = causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).tril()

            fine_attn_out = attend(fq, fk, fv, mask = fmask)

        # 3. overlapping sliding window, this is unsurprising and expected - `s` for sliding

        sq = q
        sk = k
        sv = v

        if exists(sliding_window_flex_mask):
            sliding_window_attn_out = flex_attention(sq, sk, sv, block_mask = sliding_window_flex_mask, enable_gqa = True)
        else:
            sk, sv = tuple(repeat(t, 'b h ... -> b (h num_grouped_queries) ...', num_grouped_queries = self.num_grouped_queries) for t in (sk, sv))

            sliding_window_attn_out = self.sliding_window(sq, sk, sv)

        # combine strategies

        strategy_weighted_combine = self.to_strategy_combine(inp)

        out = einsum(strategy_weighted_combine, stack([compressed_attn_out, fine_attn_out, sliding_window_attn_out]), 'b h n s, s b h n d -> b h n d')

        # merge heads and combine them

        out = self.merge_heads(out)

        out = self.combine_heads(out)

        if not return_cache:
            return out

        return out, (cache_kv, cache_compressed_kv)
```

**描述:**

1.  **输入处理:**
    *   首先判断是否处于推理模式 (`is_inferencing`)，如果是，则调用 `forward_inference` 函数。
    *   计算压缩和选择块的大小。
    *   进行输入归一化 (`self.norm`)。
    *   将输入映射到 query, key 和 value (`self.to_qkv`)，并分割 attention heads (`self.split_heads`).

2.  **压缩注意力 (Compressed Attention):**
    *   对 key 和 value 进行压缩，使用 `self.k_compress` 和 `self.v_compress`。
    *   计算压缩的 query, key 和 value 之间的 attention (`compressed_attn_out`)。

3.  **精细注意力 (Fine Attention):**
    *   基于压缩的 attention logits，选择最相关的 key/value 段 (`selected_block_indices`)。
    *   使用选择的 key/value 段计算精细 attention (`fine_attn_out`)。
    *    在 `has_selected_kv_for_fine_attn` 存在时，计算选定的 kv 段的精细注意力。计算包括处理不同块大小（压缩块和精细块），执行 top-k 选择，并应用差分 top-k gating（如果启用）。此外，代码还包含使用 Triton 内核进行加速的选项，或者使用 `flex_attention` 库（如果可用）。
4.  **滑动窗口注意力 (Sliding Window Attention):**
    *   使用滑动窗口计算局部 attention (`sliding_window_attn_out`)。
    *   使用 `flex_attention` 库进行计算的选项。

5.  **策略组合 (Strategy Combination):**
    *   使用学习到的权重组合来自压缩注意力、精细注意力和滑动窗口注意力的输出 (`strategy_weighted_combine`)。
    *   合并 attention heads (`self.merge_heads`)。
    *   通过线性层组合最终输出 (`self.combine_heads`)。

6.  **缓存处理:**
    *   如果 `return_cache` 为真，则返回缓存的 key/value。

**c. `forward_inference` (推理前向传播):**

推理过程会更加复杂，涉及到缓存机制，以便于自回归生成。 (代码太长，此处省略详细分析，主要思路是利用缓存加速推理)

**如何使用:**

1.  **初始化:** 创建 `SparseAttention` 类的实例，传入所需的参数。
2.  **前向传播:** 将输入张量传递给 `forward` 方法。
3.  **输出:** `forward` 方法返回 attention 输出张量。

```python
# 示例
import torch

# 假设的输入参数
dim = 512 # 输入维度
dim_head = 64 # 每个头的维度
heads = 8 # 注意力头的数量
sliding_window_size = 64 # 滑动窗口大小
compress_block_size = 32 # 压缩块大小
selection_block_size = 16 # 选择块大小
num_selected_blocks = 8 # 选择块的数量

# 创建 SparseAttention 实例
sparse_attn = SparseAttention(
    dim=dim,
    dim_head=dim_head,
    heads=heads,
    sliding_window_size=sliding_window_size,
    compress_block_size=compress_block_size,
    selection_block_size=selection_block_size,
    num_selected_blocks=num_selected_blocks,
    causal=True
)

# 创建随机输入张量 (B, N, D)
batch_size = 2
seq_len = 128
input_tensor = torch.randn(batch_size, seq_len, dim)

# 前向传播
output_tensor, cache_updates = sparse_attn(input_tensor, return_cache=True)  # 第一次调用没有cache，用于初始化

# 打印输出形状
print("输出形状:", output_tensor.shape) # torch.Size([2, 128, 512])
print("缓存形状:", len(cache_updates))

# 使用缓存进行后续的自回归预测 (假设每次只预测一个token)
# 创建新的输入张量，seq_len = 1
input_tensor_next = torch.randn(batch_size, 1, dim)
output_tensor_next, cache_updates_next = sparse_attn(input_tensor_next, cache=cache_updates, return_cache=True)

print("下一次输出形状:", output_tensor_next.shape)
print("下一次缓存形状:", len(cache_updates_next))
```

**代码总结**

这段代码实现了一种稀疏注意力机制（Sparse Attention），旨在提高长序列处理的效率。它通过结合压缩注意力、精细注意力和滑动窗口注意力等多种策略，从而选择性地关注最重要的输入部分，减少计算量。此外，代码还支持使用 Triton 内核和 `flex_attention` 库进行加速，并提供了缓存机制以优化自回归生成过程。整个实现非常复杂，但高度可配置，以适应不同的任务和硬件环境。