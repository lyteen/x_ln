Lan: `py` From`dl/flux/src\flux\sampling.py`

Okay, let's break down this code snippet piece by piece, explaining the key parts with Chinese comments and usage examples.

**1. `get_noise` Function:**

```python
import math
import torch

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing 为了方便打包，这里做了一定的padding
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
```

**描述 (Description):**
 This function generates Gaussian noise (随机高斯噪声) with a specified shape and data type. This noise serves as the starting point for the diffusion process (扩散过程的起点). The height and width are padded slightly to ensure they are multiples of 16 * 2 = 32, which is likely related to the architecture of the model used later (可能与后续模型架构有关).

**用法 (Usage):**

```python
# Example usage
noise = get_noise(
    num_samples=1,  # 生成一个样本 Generate one sample
    height=512,  # 图像高度 Image height
    width=512,  # 图像宽度 Image width
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 设备 Device
    dtype=torch.bfloat16,  # 数据类型 Data type
    seed=42,  # 随机种子 Random seed
)
print(f"Noise shape: {noise.shape}")  # 输出噪声张量的形状 Output noise tensor shape
```

**2. `prepare` Function:**

```python
from einops import rearrange, repeat
import torch

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: torch.Tensor, prompt: str | list[str]) -> dict[str, torch.Tensor]:
    bs, c, h, w = img.shape # b 是batchsize，c 是通道数，h是图像高，w是图像宽
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt) # 当batchsize=1，且 prompt是一个列表，那么以列表的长度作为batchsize

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) # 将图像分块，ph=2,pw=2 说明每个小patch是2x2
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs) # 如果原图batchsize=1，但实际bs>1，那么就复制

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None] # 赋值行索引
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :] # 赋值列索引
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs) # img_ids 存储每个patch的行号和列号，方便模型定位

    if isinstance(prompt, str):
        prompt = [prompt] # 如果prompt 是字符串，转成列表
    txt = t5(prompt) # 使用t5模型编码prompt
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs) # 如果原txt的batchsize=1，但实际bs>1，那么就复制
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # 文本的位置编码

    vec = clip(prompt) # 使用clip模型编码prompt
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs) # # 如果原vec的batchsize=1，但实际bs>1，那么就复制

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }
```

**描述 (Description):**
 This function preprocesses input images and text prompts for the diffusion model. It splits the image into patches (将图像分割成小块), encodes the text prompt using HFEmbedder (T5 and CLIP embeddings), and creates image and text ID tensors. The image IDs likely represent the spatial location of each patch (图像ID可能代表每个图像块的空间位置). The text embeddings and IDs are used to condition the diffusion process (文本嵌入和ID用于调节扩散过程).

**用法 (Usage):**

```python
# Assuming t5_embedder and clip_embedder are initialized instances of HFEmbedder
# 假设 t5_embedder 和 clip_embedder 是 HFEmbedder 的初始化实例

# dummy implementations for HFEmbedder
class HFEmbedder():
  def __init__(self):
    pass
  def __call__(self, prompts):
    if isinstance(prompts, str):
      prompts = [prompts]
    return torch.randn(len(prompts), 128)

t5_embedder = HFEmbedder()
clip_embedder = HFEmbedder()


# Example usage
dummy_img = torch.randn(1, 3, 64, 64) # 假设输入图像是 (B, C, H, W) 格式
prompt = "A cat sitting on a mat." # 提示词
prepared_data = prepare(t5_embedder, clip_embedder, dummy_img, prompt)
print(f"Prepared image shape: {prepared_data['img'].shape}") # 输出准备好的图像张量的形状
print(f"Prepared text shape: {prepared_data['txt'].shape}")   # 输出准备好的文本张量的形状
```

**3. `prepare_control` Function:**

```python
from einops import rearrange, repeat
from PIL import Image
import numpy as np
import torch

def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: torch.Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, torch.Tensor]:
    # load and encode the conditioning image 加载并编码 condition 图
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)  # 使用 Canny or Depth image encoder
        img_cond = ae.encode(img_cond) # 使用 AutoEncoder 进行编码

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt) # 复用 prepare 函数
    return_dict["img_cond"] = img_cond
    return return_dict
```

**描述 (Description):**
 This function prepares conditioning information based on an image for controlled image generation (为可控图像生成准备基于图像的条件信息). It loads an image from `img_cond_path`, resizes it, preprocesses it, encodes it using either a DepthImageEncoder or a CannyImageEncoder followed by an AutoEncoder.  The resulting encoded `img_cond` tensor is added to the dictionary returned by the `prepare` function.

**用法 (Usage):**

```python
# Assuming t5_embedder, clip_embedder, autoencoder, depth_encoder are initialized
# 假设 t5_embedder, clip_embedder, autoencoder, depth_encoder 已经初始化

# dummy implementations for HFEmbedder, AutoEncoder, DepthImageEncoder
#  简易HFEmbedder, AutoEncoder, DepthImageEncoder实现
class HFEmbedder():
  def __init__(self):
    pass
  def __call__(self, prompts):
    if isinstance(prompts, str):
      prompts = [prompts]
    return torch.randn(len(prompts), 128)

class AutoEncoder():
  def __init__(self):
    pass
  def encode(self, x):
    return torch.randn(x.shape[0], 16, x.shape[2]//4, x.shape[3]//4)  # Example encoded output

class DepthImageEncoder():
  def __init__(self):
    pass
  def __call__(self, x):
    return torch.randn(x.shape)

t5_embedder = HFEmbedder()
clip_embedder = HFEmbedder()
autoencoder = AutoEncoder()
depth_encoder = DepthImageEncoder()

# Example usage

dummy_img = torch.randn(1, 3, 64, 64)
prompt = "A cat sitting on a mat."
img_cond_path = "test.png"  # replace it by your local image
import numpy as np
from PIL import Image
img = Image.fromarray((np.random.rand(100,100,3)*255).astype(np.uint8))
img.save("test.png")

prepared_data = prepare_control(t5_embedder, clip_embedder, dummy_img, prompt, autoencoder, depth_encoder, img_cond_path)
print(f"Prepared image shape: {prepared_data['img'].shape}")
print(f"Prepared conditioning image shape: {prepared_data['img_cond'].shape}")
```

**4. `prepare_fill` Function:**

```python
from einops import rearrange, repeat
from PIL import Image
import numpy as np
import torch

def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: torch.Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, torch.Tensor]:
    # load and encode the conditioning image and the mask 加载并编码 condition image 和 mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)  # 将condition image 遮盖住masked 的区域
        img_cond = ae.encode(img_cond)    # 使用 AutoEncoder 编码 condition image
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",  # 分块处理 mask
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) # 再次分块

        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img_cond = torch.cat((img_cond, mask), dim=-1) # 将处理过的 mask 和 condition image 拼接在一起

    return_dict = prepare(t5, clip, img, prompt)   # 复用 prepare 函数
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict
```

**描述 (Description):**
 This function prepares inputs for inpainting (准备用于图像修复的输入). It loads a conditioning image and a mask. It applies the mask to the conditioning image, encodes the masked image using an AutoEncoder, and concatenates the processed mask with the encoded conditioning image (将mask处理后与编码的condition图像连接). This concatenated tensor serves as the `img_cond` and is combined with other data from the `prepare` function. The mask indicates which regions of the image need to be filled (mask 指示图像的哪些区域需要填充).

**用法 (Usage):**

```python
# Assuming t5_embedder, clip_embedder, autoencoder are initialized
# 假设 t5_embedder, clip_embedder, autoencoder 已经初始化

# dummy implementations for HFEmbedder, AutoEncoder
#  简易HFEmbedder, AutoEncoder实现
class HFEmbedder():
  def __init__(self):
    pass
  def __call__(self, prompts):
    if isinstance(prompts, str):
      prompts = [prompts]
    return torch.randn(len(prompts), 128)

class AutoEncoder():
  def __init__(self):
    pass
  def encode(self, x):
    return torch.randn(x.shape[0], 16, x.shape[2]//4, x.shape[3]//4)  # Example encoded output

t5_embedder = HFEmbedder()
clip_embedder = HFEmbedder()
autoencoder = AutoEncoder()

# Example usage
dummy_img = torch.randn(1, 3, 64, 64)
prompt = "A cat sitting on a mat."
img_cond_path = "test_cond.png"  # replace it by your local image
mask_path = "test_mask.png"      # replace it by your local image

# Dummy image and mask
import numpy as np
from PIL import Image
img_cond = Image.fromarray((np.random.rand(64,64,3)*255).astype(np.uint8))
img_cond.save("test_cond.png")

mask = Image.fromarray((np.random.rand(64,64)*255).astype(np.uint8))
mask.save("test_mask.png")


prepared_data = prepare_fill(t5_embedder, clip_embedder, dummy_img, prompt, autoencoder, img_cond_path, mask_path)
print(f"Prepared image shape: {prepared_data['img'].shape}")
print(f"Prepared conditioning image shape: {prepared_data['img_cond'].shape}")
```

**5. `prepare_redux` Function:**

```python
from einops import rearrange, repeat
from PIL import Image
import torch

def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: torch.Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, torch.Tensor]:
    bs, _, h, w = img.shape # b 是batchsize，c 是通道数，h是图像高，w是图像宽
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt) # 当batchsize=1，且 prompt是一个列表，那么以列表的长度作为batchsize

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)  # 使用 ReduxImageEncoder 编码 condition 图

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs) # 如果condition batchsize=1，但实际bs>1，那么就复制

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) # 将图像分块，ph=2,pw=2 说明每个小patch是2x2
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs) # 如果原图batchsize=1，但实际bs>1，那么就复制

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None] # 赋值行索引
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :] # 赋值列索引
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs) # img_ids 存储每个patch的行号和列号，方便模型定位

    if isinstance(prompt, str):
        prompt = [prompt] # 如果prompt 是字符串，转成列表
    txt = t5(prompt) # 使用t5模型编码prompt
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2) # 将文本编码和图像编码连接起来
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs) # 如果原txt的batchsize=1，但实际bs>1，那么就复制
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # 文本的位置编码

    vec = clip(prompt) # 使用clip模型编码prompt
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs) # # 如果原vec的batchsize=1，但实际bs>1，那么就复制

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }
```

**描述 (Description):**
This function is similar to `prepare_control` but utilizes a `ReduxImageEncoder` for encoding the conditioning image (类似于 prepare_control, 但使用ReduxImageEncoder编码condition图像).  Instead of returning `img_cond` separately, it concatenates the encoded image conditioning information with the text embedding (`txt`).

**用法 (Usage):**

```python
# Assuming t5_embedder, clip_embedder, redux_encoder are initialized
# 假设 t5_embedder, clip_embedder, redux_encoder 已经初始化

# dummy implementations for HFEmbedder, ReduxImageEncoder
# 简易HFEmbedder, ReduxImageEncoder实现
class HFEmbedder():
  def __init__(self):
    pass
  def __call__(self, prompts):
    if isinstance(prompts, str):
      prompts = [prompts]
    return torch.randn(len(prompts), 128)

class ReduxImageEncoder():
  def __init__(self):
    pass
  def __call__(self, x):
    return torch.randn(1, 64) # Example encoded output, batch size of 1!


t5_embedder = HFEmbedder()
clip_embedder = HFEmbedder()
redux_encoder = ReduxImageEncoder()

# Example usage
dummy_img = torch.randn(1, 3, 64, 64)
prompt = "A cat sitting on a mat."
img_cond_path = "test_redux.png"

# Dummy image
import numpy as np
from PIL import Image
img_cond = Image.fromarray((np.random.rand(64,64,3)*255).astype(np.uint8))
img_cond.save("test_redux.png")

prepared_data = prepare_redux(t5_embedder, clip_embedder, dummy_img, prompt, redux_encoder, img_cond_path)
print(f"Prepared image shape: {prepared_data['img'].shape}")
print(f"Prepared text shape: {prepared_data['txt'].shape}") # Notice that txt now contains the image conditioning.
```

**6. `time_shift`, `get_lin_function`, and `get_schedule` Functions:**

```python
import math
import torch
from typing import Callable

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points 基于两个点进行线性估计，预估mu
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps) # 通过 time_shift 函数调整 timesteps

    return timesteps.tolist()
```

**描述 (Description):**
These functions define the diffusion timesteps schedule (定义扩散的时间步长表). `get_schedule` creates a schedule of timesteps that are used in the diffusion process. The `time_shift` function applies a shift to the timesteps. The shift is intended to favor higher timesteps for higher-signal images, which might improve the generation process (目的是在高信号图像中偏向更高的时间步长，这可能会改善生成过程). `get_lin_function` is a helper function to define linear relationship between the shift parameter and image size.

**用法 (Usage):**

```python
# Example usage
num_steps = 100 # 步数 Number of steps
image_seq_len = 1024 # 图像序列长度 Image sequence length
schedule = get_schedule(num_steps, image_seq_len, shift=True)
print(f"Schedule length: {len(schedule)}")
print(f"First 10 timesteps: {schedule[:10]}") # 打印前10个timestep Print the first 10 timesteps
```

**7. `denoise` Function:**

```python
import torch

def denoise(
    model: Flux,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens
    img_cond: torch.Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):  # 遍历 timestep
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)  # 当前 timestep 的张量
        pred = model(   # 模型推理
            img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img, # 如果 img_cond 不为空，将 img 和 img_cond 连接在一起作为输入
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred   # 更新图像

    return img
```

**描述 (Description):**
 This function implements the denoising loop in the diffusion process (实现扩散过程中的去噪循环). It iterates through the timesteps, and in each step, it uses the `Flux` model to predict the noise. It then updates the image by subtracting the predicted noise, effectively denoising the image.  Guidance is applied (指导的应用).  The `img_cond` (if present) is concatenated with the image before being passed to the model.

**用法 (Usage):**

```python
# Assuming flux_model is an initialized instance of Flux
# 假设 flux_model 是 Flux 的初始化实例

# dummy implementation for Flux model
class Flux():
  def __init__(self):
    pass
  def __call__(self, img, img_ids, txt, txt_ids, y, timesteps, guidance):
    return torch.randn(img.shape)


# Initialize a dummy Flux model
flux_model = Flux()

# Example usage
num_samples = 1
height = 64
width = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
seed = 42
noise = get_noise(num_samples, height, width, device, dtype, seed)

# dummy image, text and embedding data
img_ids = torch.randn(num_samples, 32*32, 3).to(device)
txt = torch.randn(num_samples, 128).to(device)
txt_ids = torch.randn(num_samples, 128, 3).to(device)
vec = torch.randn(num_samples, 512).to(device)
timesteps = get_schedule(num_steps=10, image_seq_len=64)
guidance = 4.0
img_cond = torch.randn(num_samples, 32*32, 16).to(device) # 假装img_cond存在

denoised_image = denoise(flux_model, noise, img_ids, txt, txt_ids, vec, timesteps, guidance, img_cond)

print(f"Denoised image shape: {denoised_image.shape}")
```

**8. `unpack` Function:**

```python
import math
import torch
from einops import rearrange

def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
```

**描述 (Description):**
 This function performs the inverse operation of the patching process used in the `prepare` function (执行与 `prepare` 函数中使用的patch过程相反的操作). It rearranges the input tensor `x` from a patch-based representation back to a standard image representation (将输入张量 x 从基于patch的表示重新排列为标准图像表示). It effectively combines the patches to reconstruct the image (有效地组合patch以重建图像).

**用法 (Usage):**

```python
# Example usage
num_samples = 1
height = 64
width = 64
c = 16 # 图像通道数
ph = 2
pw = 2
h = math.ceil(height / 16)
w = math.ceil(width / 16)

# Generate a dummy patch-based tensor
dummy_patch_tensor = torch.randn(num_samples, h * w, c * ph * pw) # 创造虚拟patch tensor

# Unpack the tensor
unpacked_image = unpack(dummy_patch_tensor, height, width)

print(f"Unpacked image shape: {unpacked_image.shape}")
```

In summary, this code provides the core building blocks for a diffusion-based image generation pipeline, including noise generation, data preparation, timestep scheduling, denoising, and image reconstruction (总而言之，此代码为基于扩散的图像生成管道提供了核心构建块，包括噪声生成、数据准备、时间步长调度、去噪和图像重建).  The `prepare` functions handle conditioning information for various types of image generation tasks, such as text-to-image, controlled generation, and inpainting (prepare函数处理各种图像生成任务的条件信息，例如文本到图像、受控生成和图像修复). Remember to replace the dummy implementations with your actual models and encoders.
