Lan: `py` From`dl/genie2-pytorch\genie2_pytorch\genie2.py`


**0. 前置依赖和类型定义:**

```python
from __future__ import annotations  # 允许类型提示中使用类自身

from math import ceil, sqrt
from random import random
from functools import partial  # 用于创建偏函数

import torch
from torch import nn, tensor
import torch.nn.functional as F  # 常用函数，如激活函数，损失函数

from torchvision.utils import save_image  # 保存图像

import einx  # 张量操作库
from einops import rearrange, reduce, repeat, pack, unpack  # 张量操作库

from vector_quantize_pytorch import (
    VectorQuantize,
    ResidualVQ
)  # 向量量化库

from x_transformers.x_transformers import (
    RotaryEmbedding
)  # 旋转位置编码

from x_transformers import (
    Decoder,
    AutoregressiveWrapper
)  # Transformer解码器

from imagen_pytorch import Imagen  # Imagen扩散模型库

# tensor typing

import jaxtyping  # 用于更精确的类型提示
from beartype import beartype  # 使用beartype进行类型检查

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# einstein notation

# b - batch
# c - channels
# t - time
# h - height
# w - width
# n - sequence (flattened latent time * height * width)
# s - space sequence
# l - logits
# a - number of actions (multiple keys pressed)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

def lens_to_mask(lens, total_len):
    seq = torch.arange(total_len, device = lens.device)
    return einx.less('n, b -> b n', seq, lens)

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, ps, inv_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one(x, 'b *')
    y, _ = pack_one(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = l2norm(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# input action related helprs

def valid_action_input(inp):
    inp = inp.split(',')
    return all(i.strip().isdigit() for i in inp)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)
```

**描述:**
这段代码导入了必要的库，定义了张量类型，以及一些辅助函数，方便后续代码使用. 常见的函数包括 `exists`, `default`, `l2norm`, `pack_one`等，用于判断变量是否存在，设置默认值，计算L2范数，打包张量等操作. `gumbel_sample` 用于Gumbel重参数化采样.

**1. MetaTokenWrapper (MetaTokenWrapper):**

```python
# wrapper for adding meta tokens

class MetaTokenWrapper(Module):
    def __init__(
        self,
        fn: Decoder,
        num_meta_tokens
    ):
        super().__init__()
        self.fn = fn
        self.meta_tokens = nn.Parameter(torch.zeros(num_meta_tokens, fn.dim))

    def forward(self, x, *args, **kwargs):

        meta_tokens = repeat(self.meta_tokens, '... -> b ...', b = x.shape[0])

        x, packed_shape = pack([meta_tokens, x], 'b * d')

        out = self.fn(x, *args, **kwargs)

        _, out = unpack(out, packed_shape, 'b * d')

        return out
```

**描述:**
`MetaTokenWrapper` 是一个用于在 Transformer 解码器之前添加 meta tokens 的包装器。Meta tokens 可以用来表示一些全局信息或提示，类似于 prompt learning。
`num_meta_tokens` 是 meta token 的数量.
在 `forward` 方法中，meta tokens 首先被复制到与输入 `x` 相同的批次大小。 然后，meta tokens 和输入 `x` 被打包在一起，输入到 Transformer 解码器 `self.fn` 中。 最后，解码器的输出被解包，只返回对应于输入 `x` 的部分。
**用途:** 这个wrapper的目的是为transformer提供一些额外的，可以学习的全局信息.
**演示:** 无演示，这是一个wrapper，需要在Genie2中被调用。

**2. CausalConv3d (CausalConv3d):**

```python
# causal convolution for letting (r)vq be temporally aware

class CausalConv3d(Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        bias = False
    ):
        super().__init__()
        self.padding = (0, 0, 0, 0, kernel_size - 1, 0)
        self.conv = nn.Conv3d(dim, dim_out, (kernel_size, 1, 1), bias = bias)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)
```

**描述:**
`CausalConv3d` 是一个 3D 因果卷积层，用于处理时序数据。因果卷积保证了当前时刻的输出只依赖于过去时刻的输入，而不会泄露未来的信息。
`kernel_size` 是卷积核的大小.
在 `__init__` 方法中，根据 `kernel_size` 计算了 padding 的大小，以保证因果性。 在 `forward` 方法中，首先对输入 `x` 进行 padding，然后进行 3D 卷积。
**用途:** 用于在VQ之前或之后，为每个latent code提供时序信息。
**演示:** 无演示，需要在Genie2中被调用。

**3. Genie2 (Genie2):**

```python
# main class

class Genie2(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_latent,
        num_actions: int | None = None,
        depth = 12,
        attn_dim_head = 64,
        heads = 8,
        latent_channel_first = False,
        cfg_train_action_dropout = 0.5,
        transformer_kwargs: dict = dict(
            add_value_residual = True,
            learned_value_residual_mix = True,
            ff_glu = True,
            use_rmsnorm = True,
            num_residual_streams = 4
        ),
        action_transformer_kwargs: dict = dict(
            add_value_residual = True,
            learned_value_residual_mix = True,
            ff_glu = True,
            use_rmsnorm = True,
            depth = 2,
            heads = 4,
            attn_dim_head = 64
        ),
        num_meta_tokens = 16, # meta tokens used in Hymba https://www.arxiv.org/abs/2411.13676
        vq_codebook_size = 4096,
        vq_kwargs: dict = dict(),
        encoder: Module = nn.Identity(),
        decoder: Module = nn.Identity(),
        vq_commit_loss_weight = 1.,
        allow_multiple_actions = False,
        max_num_actions = 10,
        action_autoregressive_loss_weight = 0.1,
        add_temporal_convs = False,
        temporal_conv_kernel_size = 5,
        temporal_autoencoder_recon_loss_weight = 0.2,
        is_video_enc_dec = False # by default will assume image encoder / decoder, but in the future, video diffusion models with temporal compression will likely perform even better, imo
    ):
        super().__init__()

        self.num_actions = num_actions
        self.action_embed = nn.Embedding(num_actions, dim) if exists(num_actions) else None

        self.encoder = encoder
        self.decoder = decoder

        self.is_video_enc_dec = is_video_enc_dec

        self.dim_latent = dim_latent
        self.latent_channel_first = latent_channel_first

        self.latent_to_model = nn.Linear(dim_latent, dim)
        self.model_to_latent = nn.Linear(dim, dim_latent)

        self.time_rotary = RotaryEmbedding(
            dim = attn_dim_head // 2
        )

        # quantize latents

        # if working with image encoder / decoder, can do some temporal encoding with a 3d convolution before quantization

        self.pre_vq_transform = nn.Identity()
        self.post_vq_transform = nn.Identity()

        self.need_recon_loss = False

        if add_temporal_convs:
            assert not is_video_enc_dec, 'if using a video encoder / decoder, adding temporal convolutions is not necessary'
            self.pre_vq_transform = CausalConv3d(dim_latent, dim_latent, temporal_conv_kernel_size)
            self.post_vq_transform = CausalConv3d(dim_latent, dim_latent, temporal_conv_kernel_size)
            self.need_recon_loss = True

        self.vq = VectorQuantize(
            dim = dim_latent,
            codebook_size = vq_codebook_size,
            rotation_trick = True,
            **vq_kwargs
        )

        self.vq_commit_loss_weight = vq_commit_loss_weight
        self.vq_recon_loss_weight = temporal_autoencoder_recon_loss_weight

        # wrapper for adding meta tokens

        self.num_meta_tokens = num_meta_tokens
        meta_token_wrapper = partial(MetaTokenWrapper, num_meta_tokens = num_meta_tokens) if num_meta_tokens > 0. else identity

        # main "world model" dynamics model transformer

        self.transformer = meta_token_wrapper(Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = attn_dim_head,
            **transformer_kwargs
        ))

        # action related

        self.allow_multiple_actions = allow_multiple_actions
        self.max_num_actions = max_num_actions # in the case multiple actions are allowed, maximum number of actions allowed

        has_action_loss = action_autoregressive_loss_weight > 0.
        self.has_action_loss = has_action_loss

        self.to_action_pred = None

        if has_action_loss:
            if allow_multiple_actions:
                dim_action_transformer = dim // 2

                self.action_eos_id = num_actions
                self.action_pos_embed = nn.Parameter(torch.zeros(max_num_actions, dim))

                self.to_action_pred = nn.Sequential(
                    nn.Linear(dim, dim_action_transformer, bias = False),
                    meta_token_wrapper(dim, Decoder(
                        dim = dim_action_transformer,
                        **action_transformer_kwargs
                    )),
                    nn.Linear(dim_action_transformer, num_actions + 1, bias = False)
                )
            else:
                self.to_action_pred = nn.Linear(dim, num_actions, bias = False)

        self.action_autoregressive_loss_weight = action_autoregressive_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # needed for classifier free guidance

        self.cfg_train_action_dropout = cfg_train_action_dropout
```

**描述:**
`Genie2` 是整个模型的核心类。它集成了编码器、向量量化器、Transformer 解码器和解码器，用于学习环境的动态模型。

**参数解释:**

*   `dim`: Transformer模型的维度.
*   `dim_latent`: 潜在空间的维度.
*   `num_actions`: 动作的数量.
*   `depth`: Transformer 模型的层数.
*   `attn_dim_head`: 注意力头的维度.
*   `heads`: 注意力头的数量.
*   `latent_channel_first`: 指示潜在空间的通道是否在第一个维度.
*   `cfg_train_action_dropout`: 在训练期间 action dropout 的概率。用于classifier free guidance.
*   `transformer_kwargs`: Transformer解码器的参数.
*   `action_transformer_kwargs`: 用于预测动作的Transformer解码器的参数.
*   `num_meta_tokens`: Meta tokens 的数量.
*   `vq_codebook_size`: 向量量化码本的大小.
*   `vq_kwargs`: 向量量化器的参数.
*   `encoder`: 编码器模块.
*   `decoder`: 解码器模块.
*   `vq_commit_loss_weight`: 向量量化 commit loss 的权重.
*    `allow_multiple_actions` 是否允许多个action被输入
*    `max_num_actions` 最大的action数量
*   `action_autoregressive_loss_weight`: action autoregressive loss 的权重.
*   `add_temporal_convs`: 是否添加时间卷积.
*   `temporal_conv_kernel_size`: 时间卷积核的大小.
*   `temporal_autoencoder_recon_loss_weight`: temporal autoencoder 重构损失的权重.
*    `is_video_enc_dec` 是否encoder和decoder都是用于video的

**模块解释:**

*   `action_embed`: 动作嵌入层.
*   `encoder`: 编码器.
*   `decoder`: 解码器.
*   `latent_to_model`: 将潜在空间映射到模型空间的线性层.
*   `model_to_latent`: 将模型空间映射到潜在空间的线性层.
*   `time_rotary`: 旋转位置编码.
*   `pre_vq_transform`: 向量量化之前的变换.
*   `post_vq_transform`: 向量量化之后的变换.
*   `vq`: 向量量化器.
*   `transformer`: Transformer 解码器.
*   `to_action_pred`: 预测动作的线性层或 Transformer 解码器。

**用途:**  `Genie2` 用于学习环境的动态模型，可以根据当前状态和动作预测未来的状态。它包含了图像（视频）编码、离散化、世界模型学习等多个模块。
**演示:**  由于 `Genie2` 是一个复杂的模型，完整的演示需要大量的代码和数据。 在后续部分中，我将展示如何使用 `Genie2` 进行训练和生成。

接下来，我会继续分解 `Genie2` 类的其他方法，并提供详细的解释和演示.
