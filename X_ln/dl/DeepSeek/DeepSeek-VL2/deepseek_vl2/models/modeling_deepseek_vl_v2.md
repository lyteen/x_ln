Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\models\modeling_deepseek_vl_v2.html`

好的，我们重新开始，并对代码的关键部分进行解释，生成每一小段代码并附上简短的中文描述，并加入代码的使用方法和简单示例。

**1. `MlpProjector` 模块**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpProjector(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh ** 0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                         padding=0)  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)
```

**描述:**
`MlpProjector` 是一个用于将视觉特征投影到语言模型嵌入空间的模块。 它支持多种投影类型，包括 `identity`（恒等映射）、`linear`（线性层）、`mlp_gelu`（多层感知机与GELU激活）和 `downsample_mlp_gelu`（先下采样再使用多层感知机）。  `token_pooling` 的作用是将图像patch进行合并，减少token的数量.
**使用方法:**

1.  创建一个 `MlpProjectorConfig` 对象来指定投影配置。
2.  初始化 `MlpProjector`，传入配置对象。
3.  将视觉特征输入到 `forward` 方法中。

**示例:**
```python
from attrdict import AttrDict
# 假设我们有配置对象
class Config:
    def __init__(self):
        self.projector_type = "linear" # 或者 "mlp_gelu", "downsample_mlp_gelu", "identity"
        self.input_dim = 1152
        self.n_embed = 2048
        self.depth = 2
        self.mlp_ratio = 1
        self.downsample_ratio = 2
        self.token_pooling = False

cfg = Config()


# 初始化投影模块
mlp_projector = MlpProjector(cfg)

# 假设 vision_features 的形状是 [batch_size, seq_len, input_dim]
batch_size = 2
seq_len = 196
input_dim = 1152
vision_features = torch.randn(batch_size, seq_len, input_dim)

# 进行投影
projected_features = mlp_projector(vision_features)

# projected_features 的形状是 [batch_size, seq_len, n_embed]
print(f"投影后的特征形状: {projected_features.shape}") # torch.Size([2, 196, 2048])
```

**2. `VisionEncoderConfig` 和 `MlpProjectorConfig` 配置类**

```python
from transformers.configuration_utils import PretrainedConfig

class VisionEncoderConfig(PretrainedConfig):
    model_type: str = "vision"

    model_name: str = "siglip_large_patch16_384"
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = "map"
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = "skip"
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(
            self,
            model_name: str = "siglip_large_patch16_384",
            image_size: int = 384,
            patch_size: int = 16,
            width: int = 1024,
            layers: int = 24,
            heads: int = 16,
            mlp_ratio: int = 4,
            global_pool: str = "map",
            ignore_head: bool = True,
            class_token: bool = False,
            num_classes: int = 0,
            use_checkpoint: bool = False,
            **kwargs
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
            self,
            projector_type: str = "downsample_mlp_gelu",
            input_dim: int = 1152,
            n_embed: int = 2048,
            depth: int = 2,
            mlp_ratio: int = 1,
            downsample_ratio: int = 2,
            **kwargs
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)
```

**描述:**
这两个类 `VisionEncoderConfig` 和 `MlpProjectorConfig` 都是用于配置视觉编码器和 MLP 投影器的配置类。  它们继承自 `PretrainedConfig`，并定义了模型的各种超参数。

**使用方法:**
1.  创建一个 `VisionEncoderConfig` 或 `MlpProjectorConfig` 对象，并设置所需的参数。
2.  将配置对象传递给相应的模型。

**示例:**
```python
# 创建一个 VisionEncoderConfig 对象
vision_config = VisionEncoderConfig(
    image_size=384,
    patch_size=16,
    width=1024,
    layers=24,
    heads=16,
    mlp_ratio=4
)

# 创建一个 MlpProjectorConfig 对象
projector_config = MlpProjectorConfig(
    projector_type="linear",
    input_dim=1024,
    n_embed=2048
)

# 打印配置信息
print(f"视觉编码器配置: {vision_config}")
print(f"投影器配置: {projector_config}")
```

**3. `DeepSeekVLV2CausalLMOutputWithPast` 数据类**

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple
from transformers.modeling_outputs import ModelOutput
import torch

@dataclass
class DeepSeekVLV2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for DeepSeek-VL2 causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
```

**描述:**
`DeepSeekVLV2CausalLMOutputWithPast` 是一个数据类，用于存储 DeepSeek-VL2 因果语言模型的输出。 它包含损失、logits、过去的键值对、隐藏状态、注意力权重和 rope_deltas。  dataclass 可以方便的存储和传递模型的输出信息.

**使用方法:**
模型的前向传播函数返回此类的实例。

**示例:**
```python
# 假设我们有模型的输出
loss = torch.tensor(0.5)
logits = torch.randn(2, 10, 50257)  # [batch_size, sequence_length, vocab_size]
past_key_values = None # 通常是一个包含多层信息的tuple
hidden_states = None
attentions = None
rope_deltas = None

# 创建一个 DeepSeekVLV2CausalLMOutputWithPast 对象
output = DeepSeekVLV2CausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=past_key_values,
    hidden_states=hidden_states,
    attentions=attentions,
    rope_deltas=rope_deltas
)

# 打印输出信息
print(f"损失: {output.loss}")
print(f"logits 形状: {output.logits.shape}")
```

**4. `DeepseekVLV2Config` 配置类**

```python
from transformers.configuration_utils import PretrainedConfig
from .configuration_deepseek import DeepseekV2Config # 假设这个config文件存在

class VisionEncoderConfig(PretrainedConfig): # 示例配置类，需要定义好
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MlpProjectorConfig(PretrainedConfig): # 示例配置类，需要定义好
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig
    language_config: DeepseekV2Config

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),)

    def __init__(
            self,
            tile_tag: str = "tile_tag",
            global_view_pos: str = "head",
            candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),),
            **kwargs
    ):
        super().__init__(**kwargs)

        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get("projector_config", {})
        self.projector_config = MlpProjectorConfig(**projector_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, DeepseekV2Config):
            self.language_config = language_config
        else:
            self.language_config = DeepseekV2Config(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
```

**描述:**
`DeepseekVLV2Config` 是用于配置 DeepSeek-VL2 模型的配置类。 它包含视觉编码器配置、MLP 投影器配置和语言模型配置。 `tile_tag`, `global_view_pos` 和 `candidate_resolutions` 定义了如何处理图像tokens.

**使用方法:**
1.  创建 `VisionEncoderConfig`， `MlpProjectorConfig` 和 `DeepseekV2Config` 对象，并设置所需的参数。
2.  创建一个 `DeepseekVLV2Config` 对象，并将上面创建的配置对象传递给它。
3.  将配置对象传递给 `DeepseekVLV2ForCausalLM` 模型。

**示例:**
```python
# 假设我们已经有了 VisionEncoderConfig, MlpProjectorConfig 和 DeepseekV2Config 对象
# vision_config, projector_config, language_config = ...

# 创建一个 DeepseekVLV2Config 对象
deepseek_vl_config = DeepseekVLV2Config(
    vision_config=vision_config,
    projector_config=projector_config,
    language_config=language_config,
    tile_tag="2D",
    global_view_pos="head",
    candidate_resolutions=((384, 384),)
)

# 打印配置信息
print(f"DeepSeek-VL2 配置: {deepseek_vl_config}")
```

**5. `DeepseekVLV2PreTrainedModel` 基类**

```python
from transformers import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

class DeepseekVLV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekVLV2Config
    base_model_prefix = "deepseek_vl_v2"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
```

**描述:**
`DeepseekVLV2PreTrainedModel` 是 DeepSeek-VL2 模型的基类。 它继承自 `PreTrainedModel`，并定义了模型的配置类和基础模型前缀。 这个类主要用于transformers库的集成，提供一些通用的方法和属性。

**使用方法:**
`DeepseekVLV2ForCausalLM` 继承自此类.  通常不需要直接使用此类。

**示例:**
不需要示例，因为这个类是作为基类使用的。

**6. `DeepseekVLV2ForCausalLM` 模型类**

```python
import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, List, Tuple

from transformers import PreTrainedModel
from .siglip_vit import VisionTransformer # 假设这个文件存在
from .modeling_deepseek import DeepseekV2ForCausalLM # 假设这个文件存在

class DeepseekVLV2ForCausalLM(DeepseekVLV2PreTrainedModel):

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)

        # image token format 形式
        # FIXME 目前tile tag & global_view_pos的默认取值都是之前的实验策略；后续应当去掉默认取值，改为没有取值就raise error
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # 用于format image token sequence的特殊token
        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}")
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed)) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)

    def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            **ignore_kwargs
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (1 + num_width_tiles * num_height_tiles)

            total_tiles.append(images[idx, :batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw ** 0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1: tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.tile_tag == "2D":

                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w
                    )

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(
                        self.image_newline,
                        "d -> (th h) 1 d",
                        th=num_height_tiles,
                        h=h
                    )

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :], local_features], dim=0)
                    else:
                        global_local_features = torch.cat(
                            [local_features, self.view_seperator[None, :], global_features], dim=0)

                else:
                    # abandoned，实际上不会走这个逻辑
                    global_features = torch.cat(
                        [self.tile_indicators[0:1], global_features], dim=0
                    )
                    local_features = torch.cat(
                        [self.tile_indicators[1:num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1
                    )
                    local_features = rearrange(local_features, 'crop_num hw d -> (crop_num hw) d')

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), images_in_this_batch)

        return input_embeds

    @torch.no_grad()
    def incremental_prefilling(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            chunk_size: int = 1024
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            del images
            del images_seq_mask
            del images_spatial_crop

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            self._clear_cuda_cache()

        bzs, seq_len, _ = inputs_embeds.shape
        past_key_values = None

        # remain the last token for the next forward
        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start: chunk_end]
            chunk_attention_mask = attention_mask[:, 0: chunk_end]
            # print(f"start = {chunk_start}, end = {chunk_end}, prefilling_len = {prefilling_len}, seq_len = {seq_len}")

            # compute position_ids
            if past_key_values is not None:
                position_ids = torch.arange(
                    chunk_start,
                    chunk_end,
                    dtype=torch.long,
                    device=inputs_embeds.device
                ).unsqueeze(0)
                past_key_values = self._move_past_key_values_to_gpu(past_key_values, inputs_embeds.device)
            else:
                position_ids = None

            # chunk-forward
            with torch.no_grad():
                outputs = self.forward(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
                # update past_key_values
                past_key_values = outputs.past_key_values
                past_key_values = self._move_past_key_values_to_cpu(past_key_values)

                del outputs, position_ids
                self._clear_cuda_cache()

        prefilling_key_values = []
        for layer_past in past_key_values:
            prefilling_key_values.append(
                (
                    layer_past[0][:, :, 0: prefilling_len, ...].to(inputs_embeds.device),
                    layer_past[1][:, :, 0: prefilling_len, ...].to(inputs_embeds.device),
                )
            )

        return inputs_embeds, prefilling_key_values

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,

            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] =