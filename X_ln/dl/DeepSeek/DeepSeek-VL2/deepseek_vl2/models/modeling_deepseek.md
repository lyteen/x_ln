Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\models\modeling_deepseek.html`

好的，现在开始重新解释代码。

**1.  获取未填充数据 (`_get_unpad_data`):**

```python
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
```

**描述:** 这个函数用于处理带有填充 (padding) 的注意力掩码 (attention mask)。它计算每个序列的长度，找到非填充 token 的索引，并计算累积序列长度，这对于高效的 FlashAttention 处理是必要的。

**如何使用:** 在 FlashAttention 中，为了提高效率，通常会移除填充 token。 这个函数将注意力掩码转换为 FlashAttention 可以理解的格式。

**示例:** 假设 `attention_mask` 是 `[[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]`，那么函数会返回非填充 token 的索引 `[0, 1, 2, 5, 6]`，累积序列长度 `[0, 3, 5]` 和最大序列长度 `3`。

**2. DeepseekV2 RMS Normalization (`DeepseekV2RMSNorm`):**

```python
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

**描述:** 这个类实现了 RMS (Root Mean Square) 归一化层。它对输入 hidden states 进行归一化，使其具有单位 RMS，并使用可学习的权重进行缩放。它类似于 T5LayerNorm。

**如何使用:** RMS 归一化是一种常见的归一化技术，可以提高模型的训练稳定性和性能。 通常在 Transformer 模型的每个子层 (例如，注意力层和前馈层) 之前或之后使用。

**示例:**
```python
# 创建一个 RMSNorm 实例
rms_norm = DeepseekV2RMSNorm(hidden_size=768)
# 输入 hidden states
hidden_states = torch.randn(1, 128, 768) # batch_size=1, seq_len=128, hidden_size=768
# 应用 RMS 归一化
normalized_hidden_states = rms_norm(hidden_states)
```

**3. Rotary Embedding (`DeepseekV2RotaryEmbedding`):**

```python
class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

**描述:** 此类实现了 Rotary Position Embedding (RoPE)。 RoPE 是一种将位置信息编码到 Transformer 模型中的方法，它使用旋转矩阵来编码不同位置之间的关系。与绝对位置编码相比，RoPE 具有更好的泛化能力，可以处理比训练时更长的序列。

**如何使用:** Rotary Embedding 通常应用于 Transformer 模型的 Query 和 Key 向量。 它将位置信息注入到注意力机制中，使模型能够感知输入序列中不同 token 的位置。

**示例:**
```python
# 创建一个 RotaryEmbedding 实例
rotary_emb = DeepseekV2RotaryEmbedding(dim=128, max_position_embeddings=2048)
# 输入 query 和 key 向量
query = torch.randn(1, 32, 128, 128) # batch_size=1, num_heads=32, seq_len=128, head_size=128
key = torch.randn(1, 32, 128, 128)
# 应用 Rotary Embedding
cos, sin = rotary_emb(query, seq_len=128)
# 这里需要使用 apply_rotary_pos_emb 函数来实际应用 rotary embedding
position_ids = torch.arange(128).unsqueeze(0) # 创建位置 IDs
query_embed, key_embed = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
```

**4.  apply_rotary_pos_emb 函数**
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
**描述**
这个函数实现的是将旋转位置编码(Rotary Position Embedding)应用到query和key的张量上。Rotary Embedding将位置信息编码到Transformer模型中。这个函数接收query (q), key (k) 张量，以及由`DeepseekV2RotaryEmbedding`产生的cos和sin值，然后根据`position_ids`来应用旋转。

**原理**
1.  根据`position_ids`从预计算的`cos`和`sin`中选择相应的值，并添加一个维度以便广播。
2.  将query和key张量reshape为可应用旋转的形式。具体来说，将最后一维分成两半，然后交换维度，最后再reshape回去。
3.  应用旋转：计算`q_embed = (q * cos) + (rotate_half(q) * sin)`和`k_embed = (k * cos) + (rotate_half(k) * sin)`。这里`rotate_half`函数的作用是将向量的后半部分旋转到前面。
4.  返回嵌入后的query和key张量。

**用例**

```python
# 假设已经有了query, key, cos, sin, position_ids
# 示例形状
batch_size = 2
num_heads = 8
seq_len = 16
head_dim = 64

query = torch.randn(batch_size, num_heads, seq_len, head_dim)
key = torch.randn(batch_size, num_heads, seq_len, head_dim)
cos = torch.randn(seq_len, head_dim)  # 预计算的cos值
sin = torch.randn(seq_len, head_dim)  # 预计算的sin值
position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # 位置ID

# 应用旋转位置编码
query_embed, key_embed = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

# 打印结果形状
print("Query嵌入后的形状:", query_embed.shape)  # torch.Size([2, 8, 16, 64])
print("Key嵌入后的形状:", key_embed.shape)    # torch.Size([2, 8, 16, 64])
```

**5. Mix of Experts (MoE) 相关模块 (MoEGate, AddAuxiliaryLoss, DeepseekV2MoE):**

这部分代码实现了 Mixture of Experts (MoE) 层。MoE 是一种提高模型容量和性能的技术，它使用多个 "专家" (MLP) 并根据输入动态地选择激活哪些专家。

*   **`MoEGate`:**  门控机制，用于根据输入选择激活哪些专家。
*   **`AddAuxiliaryLoss`:**  在训练过程中添加辅助损失，以提高 MoE 层的训练稳定性和性能。
*   **`DeepseekV2MoE`:**  MoE 层的核心实现，包含多个专家和一个门控机制。

由于 MoE 的代码量较大，这里只给出大致描述和如何使用，详细的解释可以参考相关的 MoE 论文和资料。

**如何使用 MoE:**

1.  创建一个 `DeepseekV2MoE` 实例，指定专家数量和门控机制的参数。
2.  将输入 hidden states 传递给 `forward` 方法。 该方法返回 MoE 层的输出。

**示例:**
```python
# 创建一个 DeepseekV2MoE 实例
moe_layer = DeepseekV2MoE(config) # 需要一个 DeepseekV2Config 实例
# 输入 hidden states
hidden_states = torch.randn(1, 128, 768) # batch_size=1, seq_len=128, hidden_size=768
# 应用 MoE 层
moe_output = moe_layer(hidden_states)
```

**6. 注意力机制 (DeepseekV2Attention, DeepseekV2FlashAttention2):**

这部分代码实现了 DeepSeekV2 模型的注意力机制。 包含了普通注意力机制和 Flash Attention 2 加速版本。

*   **`DeepseekV2Attention`:**  标准的 Multi-Head Attention 实现。
*   **`DeepseekV2FlashAttention2`:** 使用 Flash Attention 2 算法加速的注意力机制。

**如何使用:**

1.  创建一个 `DeepseekV2Attention` 或 `DeepseekV2FlashAttention2` 实例。
2.  将输入 hidden states, 注意力掩码等传递给 `forward` 方法。

**示例:**
```python
# 创建一个 DeepseekV2Attention 实例
attention = DeepseekV2Attention(config) # 需要一个 DeepseekV2Config 实例
# 输入 hidden states 和注意力掩码
hidden_states = torch.randn(1, 128, 768) # batch_size=1, seq_len=128, hidden_size=768
attention_mask = torch.ones(1, 128) # 示例注意力掩码
# 应用注意力机制
attention_output, attention_weights, _ = attention(hidden_states, attention_mask=attention_mask)
```

**7. 解码器层 (DeepseekV2DecoderLayer):**

```python
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_mla:
            attn_implementation = "mla_" + config._attn_implementation
        else:
            attn_implementation = "mha_" + config._attn_implementation

        self.self_attn = ATTENTION_CLASSES[attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
```

**描述:** 这是一个 Transformer 解码器层，包含自注意力机制、MLP (前馈神经网络) 以及 Layer Normalization。

**如何使用:** 这是构建 Transformer 模型的基本模块。 它接收输入 hidden states, 注意力掩码, 以及其他参数，并返回处理后的 hidden states。

**示例:**
```python
# 创建一个 DeepseekV2DecoderLayer 实例
decoder_layer = DeepseekV2DecoderLayer(config, layer_idx=0) # 需要 DeepseekV2Config 实例
# 输入 hidden states 和注意力掩码
hidden_states = torch.randn(1, 128, 768)
attention_mask = torch.ones(1, 1, 128, 128) # 注意力掩码的形状取决于是否使用 FlashAttention
# 应用解码器层
layer_output = decoder_layer(hidden_states, attention_mask=attention_mask)
```

**8. DeepseekV2 模型 (DeepseekV2Model):**

```python
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
```

**描述:** 这是 DeepseekV2 模型的核心组件，它由多个 `DeepseekV2DecoderLayer` 组成。 它接收输入 token 序列 (input\_ids) 并返回 hidden states。

**如何使用:** 创建一个 `DeepseekV2Model` 实例，并传入 `input_ids` 和 `attention_mask` 等参数。

**示例:**
```python
# 创建 DeepseekV2Model 实例
model = DeepseekV2Model(config) # 需要 DeepseekV2Config 实例
# 输入 input_ids 和 attention_mask
input_ids = torch.randint(0, config.vocab_size, (1, 128))
attention_mask = torch.ones(1, 128)
# 应用模型
output = model(input_ids, attention_mask=attention_mask)
last_hidden_state = output.last_hidden_state # 获取最后的 hidden state
```

**9. DeepseekV2ForCausalLM 模型 (DeepseekV2ForCausalLM):**

```python
class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,

            attention_mask=None,
            cache_position=None,

            pixel_values=None,
            image_sizes=None,
            num_logits_to_keep=None,
            **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = self.language.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        cache_position = model_inputs["cache_position"]
        if cache_position[0] == 0:
            model_inputs["images"] = images
            model_inputs["images_seq_mask"] = images_seq_mask
            model_inputs["images_spatial_crop"] = images_spatial_crop

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


AutoConfig.register("vision", VisionEncoderConfig)
AutoConfig.register("mlp_projector", MlpProjectorConfig)
AutoConfig.register("deepseek_vl_v2", DeepseekVLV2Config)
AutoModelForCausalLM.register(DeepseekVLV2Config, DeepseekVLV2ForCausalLM)
```
**描述:**
`incremental_prefill` 函数:
这个函数用于增量预填充 (`incremental prefilling`) 过程，即将输入序列分块处理，并计算每一块的 `past_key_values`。


将输入序列 `inputs_embeds` 分成大小为 `chunk_size` 的块，并逐个块地传递给模型。
在处理每个块时，它会计算 position_ids 和 past_key_values，并将它们传递给模型。
它使用 `torch.no_grad()`上下文管理器来禁用梯度计算，因为在预填充过程中不需要梯度。
它将每个块的 `past_key_values` 存储在 prefilling_key_values 列表中，并在最后返回。

此函数用于高效地处理长输入序列，避免一次性将整个序列加载到 GPU 内存中。


`_clear_cuda_cache` 函数:
此函数用于清除 CUDA 内存缓存。
它调用 `gc.collect()` 来强制执行垃圾回收，并使用 `torch.cuda.empty_cache()` 和 `torch.cuda.synchronize()` 来清除 CUDA 缓存。
在处理大型模型和长序列时，清除 CUDA 缓存非常重要，以避免内存不足错误。


`_move_past_key_values_to_cpu` 和 `_move_past_key_values_to_gpu` 函数:
这些函数用于在 CPU 和 GPU 之间移动 past_key_values。
当需要将 past_key_values 存储在 CPU 内存中或将其传递给另一个 GPU 设备时，可以使用这些函数。


`_reorder_cache` 函数:
此函数用于在 beam search 过程中重新排序 past_key_values。
它根据 beam_idx 重新排列 past_key_values，以确保每个 beam 都有正确的上下文。