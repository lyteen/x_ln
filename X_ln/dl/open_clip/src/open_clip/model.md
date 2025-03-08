Lan: `py` From`X_ln\dl/open_clip/src\open_clip\model.html`

好的，这是对代码的详细解释，附带中文注释和使用示例：

**1. `CLIPVisionCfg` 数据类:**

```python
@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12  # 视觉Transformer的层数，可以是整数或整数元组 (ResNet). 默认12层
    width: int = 768  # 视觉Transformer的宽度（embedding维度）. 默认768
    head_width: int = 64 # 每个attention head的宽度. 默认64
    mlp_ratio: float = 4.0 # MLP层的扩展比例. 默认4.0
    patch_size: int = 16 # 输入图像patch的大小. 默认16
    image_size: Union[Tuple[int, int], int] = 224 # 输入图像的大小. 默认224

    ls_init_value: Optional[float] = None  # Layer Scale 初始化值
    patch_dropout: float = 0.  # Patch Dropout的比例
    attentional_pool: bool = False  # 是否使用attentional pooling
    attn_pooler_queries: int = 256  # Attentional Pooler 查询数量
    attn_pooler_heads: int = 8  # Attentional Pooler Head 数量
    no_ln_pre: bool = False  # 是否禁用 Transformer 前的 LayerNorm
    pos_embed_type: str = 'learnable' # 位置编码的类型
    final_ln_after_pool: bool = False  # 在池化后是否应用最终的LayerNorm
    pool_type: str = 'tok' # 池化方式, tok: class token, gap: global average pooling
    output_tokens: bool = False # 是否输出所有token

    act_kwargs: Optional[dict] = None  # 激活函数的参数
    norm_kwargs: Optional[dict] = None  # Layer Normalization的参数

    timm_model_name: Optional[str] = None  # 使用timm模型的名字，优先级高于layers, width, patch_size
    timm_model_pretrained: bool = False  # 是否使用timm模型的预训练权重
    timm_pool: str = 'avg'  # timm模型的特征池化方式
    timm_proj: str = 'linear'  # timm模型的线性投影方式
    timm_proj_bias: bool = False  # 是否启用timm模型最终投影的偏置
    timm_drop: float = 0.  # timm模型的dropout比例
    timm_drop_path: Optional[float] = None  # timm模型的stochastic depth

# 使用示例:
vision_cfg = CLIPVisionCfg(layers=12, width=768, image_size=224)
print(vision_cfg.width) # 输出: 768
```

**描述:** `CLIPVisionCfg` 是一个数据类，用于存储视觉编码器的配置。 它定义了视觉Transformer或ResNet的各种超参数，例如层数、宽度、patch大小和图像大小。 它还包括关于attention pooling、LayerNorm、位置嵌入和激活函数配置的设置。 如果指定了 `timm_model_name`，则使用来自 `timm` 库的预训练模型，并忽略其他参数（如 `layers`、`width` 等）。

**用途:**  这个类用于在创建CLIP模型时配置视觉编码器。 它提供了一种方便的方式来组织和传递视觉编码器的参数。

**2. `CLIPTextCfg` 数据类:**

```python
@dataclass
class CLIPTextCfg:
    context_length: int = 77  # 文本的最大长度. 默认77
    vocab_size: int = 49408  # 词汇表的大小. 默认49408
    hf_tokenizer_name: Optional[str] = None # HuggingFace tokenizer 名称
    tokenizer_kwargs: Optional[dict] = None # HuggingFace tokenizer 参数

    width: int = 512  # 文本Transformer的宽度（embedding维度）. 默认512
    heads: int = 8  # 注意力头的数量. 默认8
    layers: int = 12  # 文本Transformer的层数. 默认12
    mlp_ratio: float = 4.0  # MLP层的扩展比例. 默认4.0
    ls_init_value: Optional[float] = None  # Layer Scale 初始化值
    embed_cls: bool = False # 是否嵌入class token
    pad_id: int = 0 # padding token ID
    no_causal_mask: bool = False  # 是否禁用因果masking
    final_ln_after_pool: bool = False  # 在池化后是否应用最终的LayerNorm
    pool_type: str = 'argmax'  # 池化方式. argmax, mean, etc.
    proj_bias: bool = False # 是否在projection layer中启用bias
    proj_type: str = 'linear'  # text projection 类型，可以是 linear, mlp, none.
    output_tokens: bool = False # 是否输出所有token

    act_kwargs: dict = None # 激活函数的参数
    norm_kwargs: dict = None # Layer Normalization的参数

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None  # 使用HuggingFace模型的名字
    hf_model_pretrained: bool = True  # 是否使用HuggingFace模型的预训练权重
    hf_proj_type: str = 'mlp' # HuggingFace 模型 projection 类型，可以是 linear, mlp, none
    hf_pooler_type: str = 'mean_pooler'  # HuggingFace模型的池化方式，例如 attentional pooling

# 使用示例:
text_cfg = CLIPTextCfg(context_length=77, vocab_size=49408, width=512)
print(text_cfg.width) # 输出: 512
```

**描述:** `CLIPTextCfg` 是一个数据类，用于存储文本编码器的配置。 它定义了文本Transformer的各种超参数，例如上下文长度、词汇表大小、宽度、头数和层数。 它还包括关于因果掩码、LayerNorm、池化和文本投影的设置。  可以选择使用 Hugging Face 的预训练模型。

**用途:**  这个类用于在创建CLIP模型时配置文本编码器。 它提供了一种组织和传递文本编码器参数的方法。

**3. `get_cast_dtype` 函数:**

```python
def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

# 使用示例:
dtype = get_cast_dtype("fp16")
print(dtype) # 输出: torch.float16
```

**描述:** 这个函数根据给定的精度字符串返回相应的 `torch.dtype`。 它可以返回 `torch.bfloat16` (bf16) 或 `torch.float16` (fp16)。

**用途:**  用于根据指定的精度配置（例如，fp16 或 bf16）设置模型中参数的数据类型，从而提高训练效率和减少内存使用。

**4. `get_input_dtype` 函数:**

```python
def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype

# 使用示例:
dtype = get_input_dtype("bf16")
print(dtype) # 输出: torch.bfloat16
```

**描述:**  这个函数类似于 `get_cast_dtype`，但它确定输入数据应该使用的数据类型。

**用途:** 确保输入数据具有与模型期望的精度相匹配的数据类型。

**5. `_build_vision_tower` 函数:**

```python
def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg) # 如果 vision_cfg 是字典，则转换成 CLIPVisionCfg 对象

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU # 选择激活函数 QuickGELU 或者 nn.GELU

    if vision_cfg.timm_model_name:
        # 使用 timm 模型
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        # 使用 ModifiedResNet
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        # 使用 VisionTransformer
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual

# 使用示例：
embed_dim = 512
vision_cfg = CLIPVisionCfg(layers=12, width=768, image_size=224)
vision_tower = _build_vision_tower(embed_dim, vision_cfg)
print(type(vision_tower)) # 输出： <class 'models.transformer.VisionTransformer'> (取决于配置)
```

**描述:** 这个函数根据 `vision_cfg` 构建视觉编码器。 它可以构建 `TimmModel`（使用 `timm` 库），`ModifiedResNet` 或 `VisionTransformer`。激活函数也会根据 `quick_gelu` 参数进行选择。`cast_dtype` 用于确定是否使用 `LayerNormFp32`。

**用途:**  用于创建 CLIP 模型的视觉编码器组件。 它根据配置选择合适的视觉编码器架构。

**6. `_build_text_tower` 函数:**

```python
def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg) # 如果 text_cfg 是字典，则转换成 CLIPTextCfg 对象

    if text_cfg.hf_model_name:
        # 使用 Hugging Face 模型
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        # 使用 TextTransformer
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

# 使用示例：
embed_dim = 512
text_cfg = CLIPTextCfg(context_length=77, vocab_size=49408, width=512)
text_tower = _build_text_tower(embed_dim, text_cfg)
print(type(text_tower)) # 输出： <class 'models.hf_model.HFTextEncoder'> 或 <class 'models.transformer.TextTransformer'> (取决于配置)
```

**描述:**  这个函数根据 `text_cfg` 构建文本编码器。  它可以构建 `HFTextEncoder`（使用 Hugging Face 模型）或 `TextTransformer`。激活函数也会根据 `quick_gelu` 参数进行选择。 `cast_dtype` 用于确定是否使用 `LayerNormFp32`。

**用途:** 用于创建 CLIP 模型的文本编码器组件。它根据配置选择合适的文本编码器架构。

**7. `CLIP` 类:**

```python
class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict # 是否输出字典格式

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype) # 构建视觉编码器

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype) # 构建文本编码器
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.text_pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale) # 初始化logit scale
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias) # 初始化logit bias
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats) # 锁定视觉编码器的参数

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable) # 设置视觉编码器的梯度检查点
        self.transformer.grad_checkpointing = enable # 设置文本Transformer的梯度检查点

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'} # 设置不需要weight decay的参数
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image) # 通过视觉编码器提取图像特征
        return F.normalize(features, dim=-1) if normalize else features # 是否对特征进行归一化

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x # 是否对特征进行归一化

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True) # 提取图像特征
        text_features = self.encode_text(text, normalize=True) # 提取文本特征
        image_logits = self.logit_scale.exp() * image_features @ text_features.T # 计算图像-文本相似度
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits # 返回图像到文本和文本到图像的logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None # 提取图像特征
        text_features = self.encode_text(text, normalize=True) if text is not None else None # 提取文本特征

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict # 返回字典格式的输出

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp() # 返回图像特征、文本特征和logit scale

# 使用示例:
embed_dim = 512
vision_cfg = CLIPVisionCfg(layers=12, width=768, image_size=224)
text_cfg = CLIPTextCfg(context_length=77, vocab_size=49408, width=512)
clip_model = CLIP(embed_dim, vision_cfg, text_cfg)

dummy_image = torch.randn(1, 3, 224, 224)
dummy_text = torch.randint(0, 49408, (1, 77))

image_features, text_features, logit_scale = clip_model(image=dummy_image, text=dummy_text)

print(f"图像特征形状: {image_features.shape}") # 输出： torch.Size([1, 512])
print(f"文本特征形状: {text_features.shape}")   # 输出： torch.Size([1, 512])
print(f"Logit Scale: {logit_scale}")       # 输出： tensor(14.2813, grad_fn=<ExpBackward0>)
```

**描述:** `CLIP` 类是 CLIP 模型的核心。 它包含视觉编码器和文本编码器。它接收图像和文本作为输入，并将它们编码成特征向量。它还计算图像和文本特征之间的相似度，并返回 logits。

**用途:**  这是使用图像和文本输入来运行 CLIP 模型的主要类。 它提供了一个 `forward` 方法，用于编码图像和文本，以及一个 `get_logits` 方法，用于计算相似度分数。

**8. `CustomTextCLIP` 类:**

```python
class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = set()
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        if hasattr(self.text, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('text.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()

# 使用示例:
embed_dim = 512
vision_cfg = CLIPVisionCfg(layers=12, width=768, image_size=224)
text_cfg = CLIPTextCfg(context_length=77, vocab_size=49408, width=512)
custom_clip_model = CustomTextCLIP(embed_dim, vision_cfg, text_cfg)

dummy_image = torch.randn(1, 3, 224, 224)
dummy_text = torch.randint(0, 49408, (1, 77))

image_features, text_features, logit_scale = custom_clip_model(image=dummy_image, text=dummy_text)

print(f"图像特征形状: {image_features.shape}")
print(f"文本特征形状: {text_features.shape}")
print(f"Logit Scale: {logit_scale}")
```

**描述:** `CustomTextCLIP` 类是 `CLIP` 类的变体。 主要的区别是, `CLIP`类的文本塔是由 `TextTransformer` 直接实现的,而`CustomTextCLIP`的文本塔是由`_build_text_tower`构建, 可以选择 `HFTextEncoder` （Hugging Face 模型）或 `TextTransformer`。
同时, `CustomTextCLIP` 提供了 `lock_text_tower` 方法用于锁定文本塔的参数。

**用途:** 与 `CLIP` 类相似，但允许更灵活地选择文本编码器并锁定其参数。

**9. `convert_weights_to_lp` 函数:**

```python
def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)

# 使用示例:
embed_dim = 512
vision_cfg = CLIPVisionCfg(layers=12, width=768, image_size=224)
text_cfg = CLIPTextCfg(context_length=77, vocab_size=49408, width=512)
clip_model = CLIP(embed_dim, vision_cfg, text_cfg)

convert_weights_to_lp(clip_model, dtype=torch.float16)

for name, param in clip_model.named_parameters():
    if param.dtype == torch.float16:
        print(f"参数 {name} 的数据类型已转换为 float16")
        break
```

**描述:** 此函数将模型中适用参数的数据类型转换为低精度类型（bf16 或 fp16）。 它会遍历模型的所有层，并将卷积层、线性层、注意力层的权重和偏置转换为指定的 `dtype`。

**用途:**  用于降低模型的内存占用，并可能提高在支持低精度计算的硬件上的训练速度。

**10. `convert_weights_to_fp16`:**

```python
convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat
```

向后兼容. 实际调用的是 `convert_weights_to_lp`.

**11. `convert_to_custom_text_state_dict` 函数:**

```python
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict

# 使用示例:
old_state_dict = {'text_projection': torch.randn(512, 512),
                    'positional_embedding': torch.randn(77, 512),
                    'visual.conv1.weight': torch.randn(64, 3, 7, 7)}
new_state_dict = convert_to_custom_text_state_dict(old_state_dict)
print(new_state_dict.keys()) # 输出: dict_keys(['text.text_projection', 'text.positional_embedding', 'visual.conv1.weight'])
```

**描述:**  此函数用于将旧格式的 state_dict 转换为与 `CustomTextCLIP` 模型兼容的新格式。 在旧格式中，文本编码器的参数直接位于 state_dict 的顶层。 在新格式中，它们位于 `text.` 前缀下。

**用途:**  用于加载使用旧版本的代码训练的模型的权重。

**12. `build_model_from_openai_state