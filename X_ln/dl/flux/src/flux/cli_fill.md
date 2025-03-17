Lan: `py` From`dl/flux/src\flux\cli_fill.py`

好的，重新开始。

**1. 导入库 (Import Libraries):**

```python
import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from PIL import Image
from transformers import pipeline
from flux.sampling import denoise, get_noise, get_schedule, prepare_fill, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image
```

**描述:** 这部分代码导入了所有必要的库。
*   `os, re, time, dataclasses, glob`: 用于文件操作、正则表达式、时间测量、数据类和文件路径匹配。
*   `torch`: PyTorch 深度学习框架。
*   `fire`: 用于从命令行参数自动生成 CLI。
*   `PIL (Pillow)`: 用于图像处理。
*   `transformers`: Hugging Face 的 transformers 库，用于加载预训练模型。
*   `flux.sampling`: 包含用于采样的函数，例如去噪、获取噪声、获取时间表、准备填充和解包。
*   `flux.util`: 包含实用函数，例如加载配置、自动编码器 (AE)、CLIP 模型、flow模型、T5 模型和保存图像。

**如何使用:** 这些库在程序的各个部分中使用，例如加载模型、处理图像、采样和保存结果。 没有可直接运行的演示，因为这只是导入语句。

**2. 采样选项数据类 (SamplingOptions Dataclass):**

```python
@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str
    img_mask_path: str
```

**描述:**  定义了一个名为 `SamplingOptions` 的数据类。 数据类是用于存储数据的类，可以方便地定义属性和默认值。  此数据类存储了采样过程的各种选项，例如提示、图像尺寸、采样步数、引导值、随机种子以及条件图像和掩码的路径。

**如何使用:**  `SamplingOptions` 类的实例用于将采样配置传递给采样函数。例如：

```python
options = SamplingOptions(
    prompt="a cat",
    width=512,
    height=512,
    num_steps=50,
    guidance=7.5,
    seed=42,
    img_cond_path="cat.png",
    img_mask_path="mask.png",
)
```

**3. 解析提示的函数 (parse_prompt Function):**

```python
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options
```

**描述:** `parse_prompt` 函数用于交互式地更新采样选项。 它会提示用户输入新的提示或命令。  它支持使用以斜杠开头的命令来设置种子 (`/s`)、引导值 (`/g`)、采样步数 (`/n`) 和退出 (`/q`)。如果用户直接输入提示，则更新 `options` 对象的 `prompt` 属性。

**如何使用:**  此函数在 `main` 函数的循环中使用，以允许用户在每次迭代时修改采样参数。例如：

```python
options = SamplingOptions(...) # 初始化 options
options = parse_prompt(options)  # 提示用户输入新的 prompt
```

**4. 解析图像条件路径的函数 (parse_img_cond_path Function):**

```python
def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next conditioning image (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the conditioning image or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while True:
        img_cond_path = input(user_question)

        if img_cond_path.startswith("/"):
            if img_cond_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_cond_path.startswith("/h"):
                    print(f"Got invalid command '{img_cond_path}'\n{usage}")
                print(usage)
            continue

        if img_cond_path == "":
            break

        if not os.path.isfile(img_cond_path) or not img_cond_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_cond_path}' does not exist or is not a valid image file")
            continue
        else:
            with Image.open(img_cond_path) as img:
                width, height = img.size

            if width % 32 != 0 or height % 32 != 0:
                print(f"Image dimensions must be divisible by 32, got {width}x{height}")
                continue

        options.img_cond_path = img_cond_path
        break

    return options
```

**描述:** `parse_img_cond_path` 函数允许用户交互式地指定条件图像的路径。它验证文件是否存在、是否是支持的图像格式 (jpg, jpeg, png, webp) 以及尺寸是否可以被 32 整除。

**如何使用:** 此函数在 `main` 函数的循环中使用，以允许用户在每次迭代时更改条件图像。例如：

```python
options = SamplingOptions(...) # 初始化 options
options = parse_img_cond_path(options)  # 提示用户输入新的条件图像路径
```

**5. 解析图像掩码路径的函数 (parse_img_mask_path Function):**

```python
def parse_img_mask_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next conditioning mask (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the conditioning mask or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while True:
        img_mask_path = input(user_question)

        if img_mask_path.startswith("/"):
            if img_mask_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_mask_path.startswith("/h"):
                    print(f"Got invalid command '{img_mask_path}'\n{usage}")
                print(usage)
            continue

        if img_mask_path == "":
            break

        if not os.path.isfile(img_mask_path) or not img_mask_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_mask_path}' does not exist or is not a valid image file")
            continue
        else:
            with Image.open(img_mask_path) as img:
                width, height = img.size

            if width % 32 != 0 or height % 32 != 0:
                print(f"Image dimensions must be divisible by 32, got {width}x{height}")
                continue
            else:
                with Image.open(options.img_cond_path) as img_cond:
                    img_cond_width, img_cond_height = img_cond.size

                if width != img_cond_width or height != img_cond_height:
                    print(
                        f"Mask dimensions must match conditioning image, got {width}x{height} and {img_cond_width}x{img_cond_height}"
                    )
                    continue

        options.img_mask_path = img_mask_path
        break

    return options
```

**描述:** `parse_img_mask_path` 函数允许用户交互式地指定图像掩码的路径。它验证文件是否存在、是否是支持的图像格式、尺寸是否可以被 32 整除，以及尺寸是否与条件图像匹配。

**如何使用:** 此函数在 `main` 函数的循环中使用，以允许用户在每次迭代时更改图像掩码。例如：

```python
options = SamplingOptions(...) # 初始化 options
options = parse_img_mask_path(options)  # 提示用户输入新的掩码图像路径
```

**6. 主函数 (main Function):**

```python
@torch.inference_mode()
def main(
    seed: int | None = None,
    prompt: str = "a white paper cup",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float = 30.0,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/cup.png",
    img_mask_path: str = "assets/cup_mask.png",
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image. This demo assumes that the conditioning image and mask have
    the same shape and that height and width are divisible by 32.

    Args:
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        img_cond_path: path to conditioning image (jpeg/png/webp)
        img_mask_path: path to conditioning mask (jpeg/png/webp
    """
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    name = "flux-dev-fill"
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(torch_device, max_length=128)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    with Image.open(img_cond_path) as img:
        width, height = img.size
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
        img_mask_path=img_mask_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)

        with Image.open(opts.img_cond_path) as img:
            width, height = img.size
        opts.height = height
        opts.width = width

        opts = parse_img_mask_path(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None
        if offload:
            t5, clip, ae = t5.to(torch_device), clip.to(torch.device), ae.to(torch_device)
        inp = prepare_fill(
            t5,
            clip,
            x,
            prompt=opts.prompt,
            ae=ae,
            img_cond_path=opts.img_cond_path,
            mask_path=opts.img_mask_path,
        )

        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs and AE to CPU, load model to gpu
        if offload:
            t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        idx = save_image(nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt)

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
            opts = parse_img_cond_path(opts)

            with Image.open(opts.img_cond_path) as img:
                width, height = img.size
            opts.height = height
            opts.width = width

            opts = parse_img_mask_path(opts)
        else:
            opts = None
```

**描述:** `main` 函数是程序的核心。 它执行以下步骤：

1.  **初始化:**
    *   设置设备 (CPU 或 CUDA)。
    *   加载必要的模型 (T5, CLIP, flow 模型, AE)。
    *   创建一个 `SamplingOptions` 对象来存储采样参数。
    *   初始化 NSFW 分类器。
2.  **交互式循环 (如果 `loop` 为 True):**
    *   调用 `parse_prompt`、`parse_img_cond_path` 和 `parse_img_mask_path` 函数来允许用户修改采样参数。
3.  **采样循环:**
    *   生成随机噪声。
    *   准备输入 (`prepare_fill`)，包括文本嵌入和图像条件。
    *   生成时间步序列 (`get_schedule`).
    *   使用 flow 模型去噪 (`denoise`)。
    *   将潜在表示解码为像素空间 (`ae.decode`)。
    *   保存生成的图像 (`save_image`)。
4.  **清理:**
    *   如果在 `offload` 模式下运行，则将模型移动到 CPU 以释放 GPU 内存。

**如何使用:** 使用 `fire` 库从命令行调用此函数。 例如：

```bash
python your_script.py --prompt "a futuristic car" --num_steps 100 --seed 123
```

**7. 应用程序入口点 (app Function and `if __name__ == "__main__":`)**

```python
def app():
    Fire(main)


if __name__ == "__main__":
    app()
```

**描述:**
*   `app()` 函数使用 `fire.Fire(main)` 将 `main` 函数转换为命令行界面 (CLI)。
*   `if __name__ == "__main__":` 确保 `app()` 函数仅在脚本作为主程序运行时才被调用，而不是作为模块导入时。

**如何使用:**
运行脚本后，可以使用命令行参数控制 `main` 函数的行为。 例如：

```bash
python your_script.py --prompt "a cat playing piano" --output_dir "output_images"
```

此命令将运行 `main` 函数，将提示设置为 "a cat playing piano"，并将生成的图像保存在名为 "output\_images" 的目录中。

总而言之，此代码实现了一个基于流模型的图像生成 pipeline，并提供了一个交互式界面，用于使用命令行参数控制采样过程。 该代码加载了必要的模型、准备输入、对噪声进行去噪、解码潜在表示并保存生成的图像。 `fire` 库允许轻松创建命令行界面。
