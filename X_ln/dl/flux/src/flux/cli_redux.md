Lan: `py` From`dl/flux/src\flux\cli_redux.py`

好的，我们重新开始。我将对提供的代码进行分解，提供关键部分的解释，并用中文描述每个代码片段，同时附上使用方法和简单示例。

**1. 导入必要的库 (Import necessary libraries)**

```python
import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.modules.image_embedders import ReduxImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_redux, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image
```

**描述:**  这段代码导入了程序运行所需的各种Python库。包括用于文件操作的 `os`，正则表达式的 `re`，时间管理的 `time`，数据类定义的 `dataclasses`，文件查找的 `glob`，深度学习框架 `torch`，命令行参数解析工具 `fire`，以及 `transformers` 库中的 `pipeline` 用于加载预训练模型。还导入了 `flux` 库中自定义的模块和函数，这些模块和函数主要用于图像嵌入、采样和模型加载。

**使用方法:**  这些库在程序的后续部分被调用，以实现图像生成、模型加载、数据处理等功能。 需要确保你的环境中已经安装了这些库 (`pip install torch fire transformers`)。

**2. 采样选项数据类 (Sampling Options Data Class)**

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
```

**描述:**  使用 `@dataclass` 装饰器定义了一个名为 `SamplingOptions` 的数据类。  数据类可以自动生成 `__init__`， `__repr__` 等方法，简化了类的定义。  这个类用于存储采样过程中的各种参数，例如提示词（`prompt`）、图像宽度（`width`）、图像高度（`height`）、采样步数（`num_steps`）、引导强度（`guidance`）、随机种子（`seed`）和条件图像路径（`img_cond_path`）。

**使用方法:**  可以通过创建 `SamplingOptions` 类的实例来存储和传递采样参数。例如：`options = SamplingOptions(prompt="a cat", width=512, height=512, num_steps=50, guidance=7.5, seed=42, img_cond_path="cat.jpg")`

**3. 解析提示词函数 (Parse Prompt Function)**

```python
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Leave this field empty to do nothing "
        "or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
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
    return options
```

**描述:**  `parse_prompt` 函数负责解析用户输入的命令，并更新 `SamplingOptions` 对象。它会提示用户输入命令，如果命令以 "/" 开头，则根据命令类型更新 `width`、`height`、`guidance`、`seed` 或 `num_steps` 等参数。如果用户输入 "/q"，则退出程序。否则，返回更新后的 `SamplingOptions` 对象。

**使用方法:**  在交互式循环中调用 `parse_prompt` 函数，允许用户动态调整采样参数。例如：`options = parse_prompt(options)`。 这通常在 `loop` 模式下使用。

**4. 解析条件图像路径函数 (Parse Image Conditioning Path Function)**

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

        options.img_cond_path = img_cond_path
        break

    return options
```

**描述:**  `parse_img_cond_path` 函数负责解析用户输入的条件图像路径。 它会提示用户输入图像路径，如果路径存在且是有效的图像文件，则更新 `SamplingOptions` 对象的 `img_cond_path` 属性。如果用户输入 "/q"，则退出程序。如果用户直接回车，则保持当前图像路径不变。

**使用方法:** 在交互式循环中调用 `parse_img_cond_path` 函数，允许用户动态调整条件图像。例如：`options = parse_img_cond_path(options)`。这通常在 `loop` 模式下使用。

**5. 主要采样函数 (Main Sampling Function)**

```python
@torch.inference_mode()
def main(
    name: str = "flux-dev",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/robot.webp",
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
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
    """
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

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
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    img_embedder = ReduxImageEncoder(torch_device)

    rng = torch.Generator(device="cpu")
    prompt = ""
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)

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
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare_redux(
            t5,
            clip,
            x,
            prompt=opts.prompt,
            encoder=img_embedder,
            img_cond_path=opts.img_cond_path,
        )
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
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
        else:
            opts = None
```

**描述:** `main` 函数是程序的核心，负责加载模型、准备输入数据、执行采样过程、解码图像和保存结果。

*   **模型加载:** 使用 `load_t5`, `load_clip`, `load_flow_model`, `load_ae`, `ReduxImageEncoder` 等函数加载各种模型组件。
*   **输入准备:**  使用 `get_noise` 函数生成初始噪声，并使用 `prepare_redux` 函数将文本提示词、条件图像和噪声组合成模型的输入。
*   **采样:** 使用 `denoise` 函数执行扩散模型的去噪过程，生成潜在表示。
*   **解码:** 使用 `ae.decode` 函数将潜在表示解码为图像。
*   **保存:** 使用 `save_image` 函数保存生成的图像，并添加元数据。
*   **交互式循环:** 如果 `loop` 参数为 `True`，则进入交互式循环，允许用户动态调整参数并重新生成图像。
*   **NSFW classifier:** 使用 `pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)`来加载一个NSFW检测模型，用于检测生成图像是否为NSFW内容。

**使用方法:**  通过命令行运行 `main` 函数，并传入各种参数，例如模型名称、图像尺寸、随机种子、采样步数等。例如： `python your_script_name.py --name flux-dev --width 512 --height 512 --seed 42 --loop`。

**6. 应用入口 (Application Entry)**

```python
def app():
    Fire(main)


if __name__ == "__main__":
    app()
```

**描述:**  这段代码定义了一个 `app` 函数，使用 `fire.Fire` 将 `main` 函数转换为一个命令行接口。 当脚本直接运行时 (`__name__ == "__main__"`), 调用 `app` 函数，从而启动命令行界面。

**使用方法:**  在命令行中运行 Python 脚本，`fire.Fire` 会自动解析命令行参数，并将它们传递给 `main` 函数。

**总结:**

这段代码实现了一个基于扩散模型的图像生成程序。 它支持文本提示词和条件图像作为输入，并允许用户通过命令行参数和交互式命令来调整生成过程。 该程序还包括了对生成图像进行 NSFW 检查的功能。
