Lan: `py` From`dl/flux/src\flux\modules\image_embedders.py`

好的，重新开始。下面是对您提供的代码的解释和示例，以及中文描述。

**1. `DepthImageEncoder` (深度图像编码器)**

```python
import torch
from torch import nn
from transformers import AutoModelForDepthEstimation, AutoProcessor
from einops import repeat


class DepthImageEncoder:
    depth_model_name = "LiheYoung/depth-anything-large-hf"

    def __init__(self, device):
        self.device = device
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(self.depth_model_name)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        hw = img.shape[-2:]

        img = torch.clamp(img, -1.0, 1.0)
        img_byte = ((img + 1.0) * 127.5).byte()

        img = self.processor(img_byte, return_tensors="pt")["pixel_values"]
        depth = self.depth_model(img.to(self.device)).predicted_depth
        depth = repeat(depth, "b h w -> b 3 h w")
        depth = torch.nn.functional.interpolate(depth, hw, mode="bicubic", antialias=True)

        depth = depth / 127.5 - 1.0
        return depth
```

**描述:**  `DepthImageEncoder` 类使用 `depth-anything-large-hf` 模型将 RGB 图像转换为深度图。

*   **`__init__(self, device)`:**  初始化函数，加载预训练的深度估计模型和相应的处理器。  `device` 参数指定模型运行的设备（例如，'cuda' 或 'cpu'）。
*   **`__call__(self, img: torch.Tensor) -> torch.Tensor`:**  前向传播函数，接受一个 RGB 图像张量作为输入，并返回一个深度图张量。
    *   `img = torch.clamp(img, -1.0, 1.0)`: 将输入图像的像素值裁剪到 -1.0 到 1.0 的范围内。
    *   `img_byte = ((img + 1.0) * 127.5).byte()`: 将像素值从 [-1, 1] 映射到 [0, 255] 并转换为字节类型，因为深度估计模型通常在 0-255 范围内的图像上进行训练。
    *   `img = self.processor(img_byte, return_tensors="pt")["pixel_values"]`: 使用处理器将图像转换为模型所需的格式。
    *   `depth = self.depth_model(img.to(self.device)).predicted_depth`: 使用深度估计模型预测深度图。
    *   `depth = repeat(depth, "b h w -> b 3 h w")`: 将深度图从单通道复制到三个通道。
    *   `depth = torch.nn.functional.interpolate(depth, hw, mode="bicubic", antialias=True)`: 将深度图调整到与输入图像相同的大小。
    *   `depth = depth / 127.5 - 1.0`:  将像素值恢复到 [-1, 1] 的范围。

**使用方法:**

```python
# 示例用法
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_encoder = DepthImageEncoder(device)
    # 创建一个随机图像张量
    dummy_image = torch.randn(1, 3, 256, 256).to(device)  # B, C, H, W
    # 使用深度图像编码器生成深度图
    depth_map = depth_encoder(dummy_image)
    print(f"深度图形状: {depth_map.shape}")  # 输出: 深度图形状: torch.Size([1, 3, 256, 256])
    print(f"深度图数据类型: {depth_map.dtype}") # 输出: 深度图数据类型: torch.float32
```

**中文描述:**

`DepthImageEncoder` 类是一个深度图像编码器，它利用预训练的深度估计模型 `depth-anything-large-hf` 将输入的彩色图像转换为深度图像。 该类首先加载预训练模型及其对应的图像处理器，然后通过 `__call__` 方法接收输入图像。在 `__call__` 方法中，图像经过预处理、深度预测、通道复制和大小调整等步骤，最终输出与输入图像尺寸相同的深度图像。  这个深度图像可以作为下游任务的输入，比如条件生成模型。

**2. `CannyImageEncoder` (Canny 边缘检测编码器)**

```python
import cv2
import numpy as np
import torch
from einops import rearrange, repeat


class CannyImageEncoder:
    def __init__(
        self,
        device,
        min_t: int = 50,
        max_t: int = 200,
    ):
        self.device = device
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert img.shape[0] == 1, "Only batch size 1 is supported"

        img = rearrange(img[0], "c h w -> h w c")
        img = torch.clamp(img, -1.0, 1.0)
        img_np = ((img + 1.0) * 127.5).numpy().astype(np.uint8)

        # Apply Canny edge detection
        canny = cv2.Canny(img_np, self.min_t, self.max_t)

        # Convert back to torch tensor and reshape
        canny = torch.from_numpy(canny).float() / 127.5 - 1.0
        canny = rearrange(canny, "h w -> 1 1 h w")
        canny = repeat(canny, "b 1 ... -> b 3 ...")
        return canny.to(self.device)
```

**描述:** `CannyImageEncoder` 类使用 Canny 边缘检测算法将 RGB 图像转换为边缘图。

*   **`__init__(self, device, min_t: int = 50, max_t: int = 200)`:** 初始化函数，设置 Canny 边缘检测的最小和最大阈值 (`min_t` 和 `max_t`)，以及设备 (`device`)。
*   **`__call__(self, img: torch.Tensor) -> torch.Tensor`:** 前向传播函数，接受一个 RGB 图像张量作为输入，并返回一个边缘图张量。
    *   `assert img.shape[0] == 1, "Only batch size 1 is supported"`: 断言批处理大小为 1。
    *   `img = rearrange(img[0], "c h w -> h w c")`: 将图像张量的维度从 (C, H, W) 重新排列为 (H, W, C)。
    *   `img = torch.clamp(img, -1.0, 1.0)`: 裁剪图像的像素值。
    *   `img_np = ((img + 1.0) * 127.5).numpy().astype(np.uint8)`: 将像素值映射到 [0, 255] 并转换为 NumPy 数组。
    *   `canny = cv2.Canny(img_np, self.min_t, self.max_t)`: 使用 OpenCV 的 `cv2.Canny` 函数进行边缘检测。
    *   `canny = torch.from_numpy(canny).float() / 127.5 - 1.0`: 将 NumPy 数组转换为 PyTorch 张量，并将其像素值缩放到 [-1, 1] 的范围。
    *   `canny = rearrange(canny, "h w -> 1 1 h w")`: 将边缘图张量的维度从 (H, W) 重新排列为 (1, 1, H, W)。
    *   `canny = repeat(canny, "b 1 ... -> b 3 ...")`: 将边缘图从单通道复制到三个通道。

**使用方法:**

```python
# 示例用法
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    canny_encoder = CannyImageEncoder(device, min_t=50, max_t=150)
    # 创建一个随机图像张量
    dummy_image = torch.randn(1, 3, 256, 256).to(device)  # B, C, H, W
    # 使用 Canny 边缘检测编码器生成边缘图
    edge_map = canny_encoder(dummy_image)
    print(f"边缘图形状: {edge_map.shape}")  # 输出: 边缘图形状: torch.Size([1, 3, 256, 256])
    print(f"边缘图数据类型: {edge_map.dtype}") # 输出: 边缘图数据类型: torch.float32
```

**中文描述:**

`CannyImageEncoder` 类是一个 Canny 边缘检测编码器，它将输入的彩色图像转换为边缘图像。该类使用 OpenCV 库中的 `cv2.Canny` 函数来实现边缘检测。  该类首先初始化 Canny 边缘检测的阈值参数和设备，然后通过 `__call__` 方法接收输入图像。在 `__call__` 方法中，图像经过预处理、Canny 边缘检测、张量转换和通道复制等步骤，最终输出与输入图像尺寸相同的边缘图像。 这个边缘图可以作为 Stable Diffusion 等生成模型的条件输入。

**3. `ReduxImageEncoder` (Redux 图像编码器)**

```python
import os

import torch
from PIL import Image
from safetensors.torch import load_file as load_sft
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel

from flux.util import print_load_warning


class ReduxImageEncoder(nn.Module):
    siglip_model_name = "google/siglip-so400m-patch14-384"

    def __init__(
        self,
        device,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        redux_path: str | None = os.getenv("FLUX_REDUX"),
        dtype=torch.bfloat16,
    ) -> None:
        assert redux_path is not None, "Redux path must be provided"

        super().__init__()

        self.redux_dim = redux_dim
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype

        with self.device:
            self.redux_up = nn.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
            self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

            sd = load_sft(redux_path, device=str(device))
            missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
            print_load_warning(missing, unexpected)

            self.siglip = SiglipVisionModel.from_pretrained(self.siglip_model_name).to(dtype=dtype)
        self.normalize = SiglipImageProcessor.from_pretrained(self.siglip_model_name)

    def __call__(self, x: Image.Image) -> torch.Tensor:
        imgs = self.normalize.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)

        _encoded_x = self.siglip(**imgs.to(device=self.device, dtype=self.dtype)).last_hidden_state

        projected_x = self.redux_down(nn.functional.silu(self.redux_up(_encoded_x)))

        return projected_x
```

**描述:** `ReduxImageEncoder` 类使用 SigLIP 模型和 Redux 层对图像进行编码。

*   **`__init__(self, device, redux_dim: int = 1152, txt_in_features: int = 4096, redux_path: str | None = os.getenv("FLUX_REDUX"), dtype=torch.bfloat16)`:**  初始化函数，加载 SigLIP 模型、Redux 层，并设置设备和数据类型。
    *   `assert redux_path is not None, "Redux path must be provided"`: 确保提供了 Redux 层的路径。
    *   `self.redux_up = nn.Linear(redux_dim, txt_in_features * 3, dtype=dtype)`: 定义一个线性层，用于将 SigLIP 特征投影到更高维度。
    *   `self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)`: 定义一个线性层，用于将高维特征投影回原始 SigLIP 特征维度。
    *   `sd = load_sft(redux_path, device=str(device))`: 从文件中加载 Redux 层的权重。
    *   `self.siglip = SiglipVisionModel.from_pretrained(self.siglip_model_name).to(dtype=dtype)`: 加载 SigLIP 模型。
    *   `self.normalize = SiglipImageProcessor.from_pretrained(self.siglip_model_name)`: 加载 SigLIP 图像处理器。
*   **`__call__(self, x: Image.Image) -> torch.Tensor`:** 前向传播函数，接受一个 PIL 图像作为输入，并返回编码后的特征张量。
    *   `imgs = self.normalize.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)`: 使用 SigLIP 图像处理器预处理图像。
    *   `_encoded_x = self.siglip(**imgs.to(device=self.device, dtype=self.dtype)).last_hidden_state`: 使用 SigLIP 模型提取图像特征。
    *   `projected_x = self.redux_down(nn.functional.silu(self.redux_up(_encoded_x)))`: 使用 Redux 层对特征进行投影。

**使用方法:**

```python
# 示例用法
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 需要设置 FLUX_REDUX 环境变量指向 redux 权重文件路径
    os.environ["FLUX_REDUX"] = "path/to/your/redux_weights.safetensors"  # 替换成你的实际路径
    if os.environ.get("FLUX_REDUX") is None:
        print("请设置 FLUX_REDUX 环境变量指向 redux 权重文件路径")
        exit()

    try:
        redux_encoder = ReduxImageEncoder(device)
    except FileNotFoundError:
        print("找不到 redux 权重文件，请检查 FLUX_REDUX 环境变量是否设置正确")
        exit()

    # 创建一个 PIL 图像
    dummy_image = Image.new("RGB", (256, 256), color="red")
    # 使用 Redux 图像编码器生成特征
    feature = redux_encoder(dummy_image)
    print(f"特征形状: {feature.shape}")  # 输出: 特征形状: torch.Size([1, 197, 4096])  (通常SigLIP Patch大小是14，所以 256/14 大概是 197 个 patch)
    print(f"特征数据类型: {feature.dtype}") # 输出: 特征数据类型: torch.bfloat16 (或者你设置的dtype)
```

**注意:**  使用 `ReduxImageEncoder` 需要设置 `FLUX_REDUX` 环境变量，并确保该环境变量指向有效的 Redux 权重文件。 你需要下载对应的 `redux_weights.safetensors` 文件并替换`path/to/your/redux_weights.safetensors`为你的实际路径.  你可能需要安装 `safetensors` 库：`pip install safetensors`.

**中文描述:**

`ReduxImageEncoder` 类是一个图像编码器，它利用 SigLIP 模型和 Redux 层来提取图像特征。  SigLIP 模型用于提取图像的视觉特征，Redux 层用于对这些特征进行进一步的投影和处理。 该类首先加载 SigLIP 模型、Redux 层和图像处理器，然后通过 `__call__` 方法接收输入图像。在 `__call__` 方法中，图像经过预处理、SigLIP 特征提取和 Redux 层投影等步骤，最终输出编码后的特征张量。 该特征张量可以用于图像分类、图像检索等任务。

希望这些解释和示例能帮助你理解这些代码。