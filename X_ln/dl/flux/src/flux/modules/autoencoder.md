Lan: `py` From`dl/flux/src\flux\modules\autoencoder.py`

好的，现在我将重新开始，并对提供的代码进行详细解释，并附上中文描述和简单的示例。

**1. AutoEncoderParams 数据类**

```python
from dataclasses import dataclass

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
```

**描述:** 这是一个使用 `dataclass` 定义的数据类，用于存储 AutoEncoder 的配置参数。`dataclass` 可以自动生成 `__init__`、`__repr__` 等方法，方便使用。

*   `resolution`: 输入图像的分辨率.
*   `in_channels`: 输入图像的通道数.
*   `ch`: 模型的基础通道数.
*   `out_ch`: 输出图像的通道数.
*   `ch_mult`: 通道数的倍增因子列表，用于控制不同层级的通道数.
*   `num_res_blocks`: 每个层级的残差块数量.
*   `z_channels`: 潜在空间的通道数.
*   `scale_factor`: 缩放因子，用于缩放潜在向量.
*   `shift_factor`: 平移因子，用于平移潜在向量.

**如何使用:**  创建 `AutoEncoderParams` 实例，传入相应的参数，用于初始化 AutoEncoder。

**示例:**

```python
params = AutoEncoderParams(
    resolution=64,
    in_channels=3,
    ch=64,
    out_ch=3,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    z_channels=32,
    scale_factor=0.18215,
    shift_factor=0.
)
```

**2. swish 激活函数**

```python
import torch
from torch import Tensor

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)
```

**描述:**  `swish` 函数是一个激活函数，定义为 `x * sigmoid(x)`。它通常比 ReLU 激活函数表现更好。

**如何使用:**  在神经网络层之后使用 `swish` 函数，以引入非线性。

**示例:**

```python
x = torch.randn(1, 10)
output = swish(x)
print(output.shape) # 输出: torch.Size([1, 10])
```

**3. AttnBlock 注意力模块**

```python
from einops import rearrange
import torch
from torch import nn

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))
```

**描述:**  `AttnBlock` 是一个注意力模块，使用 GroupNorm 进行归一化，并使用 scaled dot-product attention。它接收输入 `x`，计算 query, key, value，应用注意力机制，最后将注意力输出投影回原始空间。`einops` 库的 `rearrange` 函数用于改变张量的形状以适应注意力计算。

**如何使用:**  在 AutoEncoder 的中间层使用 `AttnBlock`，以捕捉图像中的长程依赖关系。

**示例:**

```python
attn_block = AttnBlock(in_channels=64)
x = torch.randn(1, 64, 32, 32)
output = attn_block(x)
print(output.shape) # 输出: torch.Size([1, 64, 32, 32])
```

**4. ResnetBlock 残差块**

```python
import torch
from torch import nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h
```

**描述:** `ResnetBlock` 是一个残差块，包含两个卷积层和 GroupNorm 归一化层，以及 Swish 激活函数。 如果输入通道数和输出通道数不同，则使用 1x1 卷积层进行 shortcut 连接。

**如何使用:**  在 AutoEncoder 的编码器和解码器中使用 `ResnetBlock`，以构建深层网络并避免梯度消失问题。

**示例:**

```python
resnet_block = ResnetBlock(in_channels=64, out_channels=128)
x = torch.randn(1, 64, 32, 32)
output = resnet_block(x)
print(output.shape) # 输出: torch.Size([1, 128, 32, 32])
```

**5. Downsample 下采样模块**

```python
import torch
from torch import nn

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = nn.functional.pad(x, pad, mode="constant", value=0) # add this line
        x = nn.functional.pad(x, pad, mode="constant", value=0) # add this line

        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x
```

**描述:** `Downsample` 模块使用卷积层进行下采样，步长为 2。为了避免不对称填充，在卷积之前手动进行填充。

**如何使用:** 在编码器中使用 `Downsample` 模块，以减小图像尺寸并提取高级特征。

**示例:**

```python
downsample = Downsample(in_channels=64)
x = torch.randn(1, 64, 32, 32)
output = downsample(x)
print(output.shape) # 输出: torch.Size([1, 64, 15, 15])
```

**6. Upsample 上采样模块**

```python
import torch
from torch import nn

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x
```

**描述:** `Upsample` 模块使用最近邻插值进行上采样，然后使用卷积层进行平滑。

**如何使用:** 在解码器中使用 `Upsample` 模块，以增大图像尺寸并重建图像。

**示例:**

```python
upsample = Upsample(in_channels=64)
x = torch.randn(1, 64, 16, 16)
output = upsample(x)
print(output.shape) # 输出: torch.Size([1, 64, 32, 32])
```

**7. Encoder 编码器**

```python
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
```

**描述:** `Encoder` 模块将输入图像编码为潜在向量。 它由一系列下采样层，残差块和注意力块组成。 `conv_in` 用于将输入通道转换为基本通道 `ch`。  `down` 是一个 ModuleList，包含多个下采样层。 每个下采样层包含若干个 `ResnetBlock` 和一个 `Downsample` 模块（除了最后一层）。 中间层 `mid` 包含残差块和注意力块。 最后，`norm_out` 和 `conv_out` 用于将特征映射到潜在空间。

**如何使用:**  创建 `Encoder` 实例，传入相应的参数，然后将输入图像传递给 `forward` 方法。

**示例:**

```python
encoder = Encoder(
    resolution=64,
    in_channels=3,
    ch=64,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    z_channels=32,
)
x = torch.randn(1, 3, 64, 64)
output = encoder(x)
print(output.shape) # 输出: torch.Size([1, 64, 8, 8]) 注意，输出的通道数是2 * z_channels (均值和方差)
```

**8. Decoder 解码器**

```python
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
```

**描述:** `Decoder` 模块将潜在向量解码为图像。 它由一系列上采样层，残差块和注意力块组成。 `conv_in` 用于将潜在向量转换为基本通道 `block_in`。 `up` 是一个 ModuleList，包含多个上采样层。 每个上采样层包含若干个 `ResnetBlock` 和一个 `Upsample` 模块（除了第一层）。 中间层 `mid` 包含残差块和注意力块。 最后，`norm_out` 和 `conv_out` 用于将特征映射到输出空间。

**如何使用:**  创建 `Decoder` 实例，传入相应的参数，然后将潜在向量传递给 `forward` 方法。

**示例:**

```python
decoder = Decoder(
    resolution=64,
    in_channels=3,
    ch=64,
    out_ch=3,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    z_channels=32,
)
z = torch.randn(1, 32, 8, 8)
output = decoder(z)
print(output.shape) # 输出: torch.Size([1, 3, 64, 64])
```

**9. DiagonalGaussian 对角高斯分布**

```python
import torch
from torch import nn

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean
```

**描述:** `DiagonalGaussian` 模块用于从对角高斯分布中采样。它将输入张量 `z` 分成均值 `mean` 和对数方差 `logvar`。 如果 `sample` 为 `True`，则从高斯分布中采样；否则，返回均值。

**如何使用:** 在 AutoEncoder 的编码器之后使用 `DiagonalGaussian` 模块，以生成潜在向量。

**示例:**

```python
diagonal_gaussian = DiagonalGaussian(sample=True)
z = torch.randn(1, 64, 8, 8)
output = diagonal_gaussian(z)
print(output.shape) # 输出: torch.Size([1, 32, 8, 8]) 如果sample为True，否则输出为 torch.Size([1, 32, 8, 8])
```

**10. AutoEncoder 自编码器**

```python
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, 1, h * w, c)
        k = k.reshape(b, 1, h * w, c)
        v = v.reshape(b, 1, h * w, c)
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        return h_.reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int =