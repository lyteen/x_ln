Lan: `py` From`dl/genie2-pytorch\train_vq_e2e.py`

**1. 导入必要的库 (Import Libraries):**

```python
from shutil import rmtree # 用于删除目录及其内容
from pathlib import Path # 用于处理文件路径

import torch # PyTorch 深度学习框架
from torch import tensor, nn # PyTorch 张量和神经网络模块
from torch.nn import Module # PyTorch 模块基类
import torch.nn.functional as F # PyTorch 常用函数
from torch.utils.data import Dataset, DataLoader # PyTorch 数据集和数据加载器
from torch.optim import Adam # Adam 优化器

from einops import rearrange, repeat, pack, unpack # einops 张量操作库
from einops.layers.torch import Rearrange # einops 层

import torchvision # PyTorch 视觉库
import torchvision.transforms as T # PyTorch 图像变换
from torchvision.utils import save_image # 保存图像

# 清理结果目录，并创建新的目录
rmtree('./results', ignore_errors = True) # 如果存在 results 目录，则删除它
results_folder = Path('./results') # 创建 results 目录的 Path 对象
results_folder.mkdir(exist_ok = True, parents = True) # 创建 results 目录，如果存在则不报错，并创建父目录
```

**描述:** 这部分代码导入了所有需要的库。 `shutil` 和 `pathlib` 用于文件系统操作。 `torch` 及其子模块是 PyTorch 框架的核心。 `einops` 用于方便地进行张量形状变换。`torchvision` 用于加载和处理图像数据。最后清理了结果目录`results`，并创建了一个新的。

**2. 辅助函数 (Helper Function):**

```python
# functions

def divisible_by(num, den):
    return (num % den) == 0

#判断num是否能被den整除
```

**描述:** `divisible_by` 函数检查一个数是否能被另一个数整除。

**3. VQImageAutoregressiveAutoencoder 类 (VQImageAutoregressiveAutoencoder Class):**

```python
from x_transformers import Decoder # 导入 x_transformers 的解码器
from vector_quantize_pytorch import VectorQuantize as VQ # 导入 vector_quantize_pytorch 的向量量化器
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to # 导入旋转函数

from genie2_pytorch.genie2 import ( # 导入 genie2 的函数
    gumbel_sample, # Gumbel 采样
    min_p_filter # 最小 p 滤波
)

class Lambda(Module): # Lambda 层
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class VQImageAutoregressiveAutoencoder(Module): # VQ 图像自回归自编码器
    def __init__(
        self,
        image_size, # 图像大小
        patch_size, # patch 大小
        dim, # 嵌入维度
        codebook_size, # 码本大小
        decay = 0.9, # EMA 衰减率
        depth = 3, # 解码器深度
        dim_head = 16, # 注意力头的维度
        heads = 4, # 注意力头的数量
        recon_from_pred_codes_weight = 1., # 从预测代码重建的权重
        recon_loss_weight = 1., # 重建损失权重
        vq_commit_loss_weight = 1., # VQ 提交损失权重
        ar_commit_loss_weight = 1. # AR 提交损失权重
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size) # 确保图像大小可以被 patch 大小整除

        self.seq_length = (image_size // patch_size) ** 2 # 序列长度，即 patch 的数量

        self.encode = nn.Sequential( # 编码器
            Lambda(lambda x: x * 2 - 1), # 归一化到 [-1, 1]
            Rearrange('... 1 (h p1) (w p2) -> ...  (h w) (p1 p2)', p1 = patch_size, p2 = patch_size), # 将图像分割成 patch
            nn.Linear(patch_size ** 2, dim), # 线性层，将 patch 嵌入到 dim 维度
        )

        self.vq = VQ( # 向量量化器
            dim = dim, # 嵌入维度
            codebook_size = codebook_size, # 码本大小
            rotation_trick = True, # 旋转技巧
            decay = decay # EMA 衰减率
        )

        self.start_token = nn.Parameter(torch.zeros(dim)) # 起始 token

        self.decoder = nn.Sequential( # 解码器
            Decoder( # x_transformers 解码器
                dim = dim, # 嵌入维度
                heads = heads, # 注意力头的数量
                depth = depth, # 解码器深度
                attn_dim_head = dim_head, # 注意力头的维度
                rotary_pos_emb = True # 旋转位置嵌入
            ),
            nn.Linear(dim, dim) # 线性层
        )

        self.decode = nn.Sequential( # 解码器
            nn.Linear(dim, patch_size ** 2), # 线性层，将 dim 维度映射到 patch 大小
            Rearrange('... (h w) (p1 p2) -> ... 1 (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = image_size // patch_size), # 将 patch 重新组合成图像
            Lambda(lambda x: (x + 1) * 0.5), # 归一化到 [0, 1]
        )

        self.recon_from_pred_codes_weight = recon_from_pred_codes_weight # 从预测代码重建的权重
        self.recon_loss_weight = recon_loss_weight # 重建损失权重

        self.vq_commit_loss_weight = vq_commit_loss_weight # VQ 提交损失权重
        self.ar_commit_loss_weight = ar_commit_loss_weight # AR 提交损失权重

    @property
    def device(self):
        return next(self.parameters()).device # 获取设备

    @torch.no_grad()
    def sample(
        self,
        num_samples = 64, # 采样数量
        min_p = 0.25, # 最小概率
        temperature = 1.5 # 温度
    ):
        self.eval() # 设置为评估模式

        out = torch.empty((num_samples, 0), dtype = torch.long, device = self.device) # 初始化输出

        codebook = self.vq.codebook # 获取码本
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = num_samples) # 重复起始 token

        for _ in range(self.seq_length): # 循环生成序列
            codes = self.vq.get_codes_from_indices(out) # 获取代码

            inp = torch.cat((start_tokens, codes), dim = -2) # 将起始 token 和代码连接

            embed = self.decoder(inp) # 解码

            logits = -torch.cdist(embed, codebook) # 计算 logits

            logits = logits[:, -1] # 获取最后一个 token 的 logits
            logits = min_p_filter(logits, min_p) # 最小 p 滤波
            sampled = gumbel_sample(logits, temperature = temperature) # Gumbel 采样

            out = torch.cat((out, sampled), dim = -1) # 将采样结果连接到输出

        sampled_codes = self.vq.get_codes_from_indices(out) # 获取采样代码
        images = self.decode(sampled_codes) # 解码

        return images.clamp(0., 1.) # 将图像裁剪到 [0, 1]

    def forward(
        self,
        image # 输入图像
    ):
        self.train() # 设置为训练模式

        encoded = self.encode(image) # 编码

        quantized, codes, commit_loss = self.vq(encoded) # 向量量化

        # setup autoregressive, patches as tokens scanned from each row left to right

        start_tokens = repeat(self.start_token, '... -> b 1 ...', b = encoded.shape[0]) # 重复起始 token

        tokens = torch.cat((start_tokens, quantized[:, :-1]), dim = -2) # 将起始 token 和量化后的代码连接

        pred_codes = self.decoder(tokens) # 预测代码

        logits = -torch.cdist(pred_codes, self.vq.codebook) # 计算 logits

        ce_loss = F.cross_entropy( # 交叉熵损失
            rearrange(logits, 'b n l -> b l n'), # 调整 logits 的形状
            codes # 目标代码
        )

        # recon loss, learning autoencoder end to end

        recon_image_from_pred_codes = 0. # 从预测代码重建的图像
        recon_image_from_vq = 0. # 从 VQ 重建的图像

        if self.recon_from_pred_codes_weight > 0.: # 如果从预测代码重建的权重大于 0
            rotated_pred_codes = rotate_to(pred_codes, self.vq.get_codes_from_indices(codes)) # 旋转预测代码
            recon_image_from_pred_codes = self.decode(rotated_pred_codes) # 解码

        if self.recon_from_pred_codes_weight < 1.: # 如果从预测代码重建的权重小于 1
            recon_image_from_vq = self.decode(quantized) # 解码

        # weighted combine

        recon_image = ( # 加权组合重建图像
            recon_image_from_pred_codes * self.recon_from_pred_codes_weight +
            recon_image_from_vq * (1. - self.recon_from_pred_codes_weight)
        )

        # mse loss

        recon_loss = F.mse_loss( # MSE 损失
            recon_image, # 重建图像
            image # 原始图像
        )

        # ar commit loss

        ar_commit_loss = F.mse_loss(pred_codes, quantized) # AR 提交损失

        # total loss and breakdown

        total_loss = ( # 总损失
            ce_loss + # 交叉熵损失
            recon_loss * self.recon_loss_weight + # 重建损失
            commit_loss * self.vq_commit_loss_weight + # VQ 提交损失
            ar_commit_loss * self.ar_commit_loss_weight # AR 提交损失
        )

        return total_loss, (image, recon_image), (ce_loss, recon_loss, commit_loss, ar_commit_loss) # 返回总损失，图像，重建图像，以及各个损失
```

**描述:** 这是代码的核心部分。 `VQImageAutoregressiveAutoencoder` 类定义了一个 VQ-VAE 模型，该模型使用自回归解码器来生成图像。

*   **编码器 (Encoder):** 将输入图像编码为潜在向量。
*   **向量量化器 (Vector Quantizer):** 将潜在向量量化为离散的代码。
*   **解码器 (Decoder):** 使用自回归模型根据之前的代码预测下一个代码，然后将代码解码为图像。
*   **`sample` 方法:** 从模型中采样图像。 它使用 `gumbel_sample` 和 `min_p_filter` 来提高采样质量。
*   **`forward` 方法:** 执行前向传播，计算损失。 它包括交叉熵损失 (ce_loss)、重建损失 (recon_loss)、VQ 提交损失 (commit_loss) 和自回归提交损失 (ar_commit_loss)。

**4. 数据集和数据加载器 (Dataset and DataLoader):**

```python
# data related + optimizer

class MnistDataset(Dataset): # MNIST 数据集
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST( # 加载 MNIST 数据集
            './data', # 数据集路径
            download = True # 如果不存在则下载
        )

    def __len__(self):
        return len(self.mnist) # 返回数据集大小

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx] # 获取图像和标签
        digit_tensor = T.PILToTensor()(pil) # 将 PIL 图像转换为张量
        return (digit_tensor / 255).float() # 归一化到 [0, 1]

def cycle(iter_dl): # 循环数据加载器
    while True:
        for batch in iter_dl:
            yield batch

dataset = MnistDataset() # 创建 MNIST 数据集

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True) # 创建数据加载器
iter_dl = cycle(dataloader) # 创建循环数据加载器
```

**描述:** 这段代码定义了 `MnistDataset` 类，用于加载 MNIST 数据集。 `cycle` 函数创建一个无限循环的数据加载器。

**5. 模型、优化器和训练循环 (Model, Optimizer, and Training Loop):**

```python
# model

model = VQImageAutoregressiveAutoencoder( # 创建 VQ 图像自回归自编码器
    dim = 256, # 嵌入维度
    depth = 4, # 解码器深度
    codebook_size = 64, # 码本大小
    decay = 0.95, # EMA 衰减率
    image_size = 28, # 图像大小
    patch_size = 4, # patch 大小
    recon_from_pred_codes_weight = 0.5 # 从预测代码重建的权重
)

# data related + optimizer

optimizer = Adam(model.parameters(), lr = 3e-4) # 创建 Adam 优化器

# train loop

for step in range(1, 100_000 + 1): # 训练循环

    loss, (image, recon_image), (ce_loss, recon_loss, vq_commit_loss, ar_commit_loss) = model(next(iter_dl)) # 前向传播，计算损失
    loss.backward() # 反向传播

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 梯度裁剪

    optimizer.step() # 更新参数
    optimizer.zero_grad() # 清零梯度

    loss_str = "\t".join([f"{loss_name}: {loss.item():.3f}" # 格式化损失字符串
        for loss_name, loss in (
            ('recon', recon_loss), # 重建损失
            ('ce', ce_loss), # 交叉熵损失
            ('vq commit', vq_commit_loss), # VQ 提交损失
            ('ar commit', ar_commit_loss) # AR 提交损失
        )
    ])

    print(f'{step}: {loss_str}') # 打印损失

    if divisible_by(step, 500): # 每 500 步保存一次图像
        save_image( # 保存图像
            rearrange([image, recon_image], 'ir b 1 h w -> 1 (b h) (ir w)'), # 调整图像形状
            str(results_folder / f'{step}.train.recon.png') # 保存路径
        )

        image = model.sample(num_samples = 64) # 采样图像

        save_image( # 保存图像
            rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(), # 调整图像形状
            str(results_folder / f'{step}.png') # 保存路径
        )
```

**描述:**  这部分代码初始化模型和 Adam 优化器。 训练循环迭代 100,000 步。在每一步中，计算损失，执行反向传播，裁剪梯度，并更新模型参数。每 500 步，保存重建的图像和采样的图像。

总的来说，该代码实现了一个 VQ-VAE 模型，该模型使用自回归解码器来生成图像。 该模型在 MNIST 数据集上进行训练。 训练循环包括前向传播、损失计算、反向传播、梯度裁剪和参数更新。定期保存重建的图像和采样的图像，以便监控训练进度。
