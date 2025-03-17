Lan: `py` From`dl/flux/src\flux\cli_control.py`

Okay, let's break down the code piece by piece, with explanations in Chinese and simple demos.

**1. Imports and Data Class:**

```python
import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.modules.image_embedders import CannyImageEncoder, DepthImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_control, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str
    lora_scale: float | None
```

**描述 (描述):**

*   This part imports necessary libraries like `torch` for PyTorch operations, `transformers` for using pre-trained models, `fire` for command-line argument parsing, and others for file handling and data structures.  The `SamplingOptions` dataclass is defined to hold the parameters for the sampling process (prompt, image size, number of steps, etc.).

*   **中文解释:**  这段代码导入了各种需要的库，比如 `torch` 用于 PyTorch 操作, `transformers` 用于使用预训练模型, `fire` 用于命令行参数解析，还有其他一些用于文件操作和数据结构的库。 `SamplingOptions` 数据类定义用于保存采样过程的各种参数 (提示词, 图片大小, 采样步数 等等)。

**2. `parse_prompt` Function:**

```python
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
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
    if prompt != "":
        options.prompt = prompt
    return options
```

**描述 (描述):**

*   This function handles interactive prompt input. It allows the user to change parameters like width, height, seed, guidance, and number of steps by entering commands starting with `/`. If the user enters a new prompt, it updates the `options.prompt`.  If the user enters `/q`, it quits the interactive session.

*   **中文解释:** 这个函数处理交互式的提示词输入。它允许用户通过输入以 `/` 开头的命令来更改参数，例如宽度、高度、种子、guidance和采样步数。 如果用户输入一个新的提示词，它会更新 `options.prompt`。 如果用户输入 `/q`，它会退出交互会话。

**Demo Usage (演示用法):**

To use this, you'd call it within a loop. Let's assume `options` is an instance of `SamplingOptions`:

```python
# Example usage (模拟使用)
options = SamplingOptions(prompt="a cat", width=512, height=512, num_steps=50, guidance=7.5, seed=None, img_cond_path="", lora_scale=None)

options = parse_prompt(options) # Get the first prompt or modify settings.
if options:
    print(f"Prompt is now: {options.prompt}, width: {options.width}")
```

If you run this and type `/w 640` then press enter, the output will show the width updated to 640. If you type `a dog` then press enter, the prompt will change to `a dog`.

**3. `parse_img_cond_path` Function:**

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

**描述 (描述):**

*   This function handles interactive input for the conditioning image path. It prompts the user for an image path.  It checks if the file exists and is a valid image format. If the user enters `/q`, it quits.  If the user enters an empty string, it repeats the previous image path.

*   **中文解释:** 这个函数处理条件图像路径的交互式输入。 它提示用户输入一个图像路径。 它检查文件是否存在并且是有效的图像格式。 如果用户输入 `/q`，则退出。 如果用户输入一个空字符串，它会重复之前的图像路径。

**Demo Usage (演示用法):**

```python
# Example usage (模拟使用)
options = SamplingOptions(prompt="a cat", width=512, height=512, num_steps=50, guidance=7.5, seed=None, img_cond_path="old_image.png", lora_scale=None)

options = parse_img_cond_path(options)
if options:
    print(f"Conditioning image path is now: {options.img_cond_path}")
```

If you run this and type `new_image.jpg` (assuming `new_image.jpg` exists), the conditioning image path will be updated. If `new_image.jpg` doesn't exist or isn't a valid image, an error message will be printed.

**4. `parse_lora_scale` Function:**

```python
def parse_lora_scale(options: SamplingOptions | None) -> tuple[SamplingOptions | None, bool]:
    changed = False

    if options is None:
        return None, changed

    user_question = "Next lora scale (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the lora scale or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/q"):
            print("Quitting")
            return None, changed
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.lora_scale = float(prompt)
        changed = True
    return options, changed
```

**描述 (描述):**

*   This function handles interactive input for the LoRA scale. It prompts the user to enter a new LoRA scale.  It updates the `options.lora_scale` if a valid float is entered.  The `changed` flag indicates whether the LoRA scale was actually modified.

*   **中文解释:** 这个函数处理 LoRA scale 的交互式输入。 它提示用户输入一个新的 LoRA scale。 如果输入了一个有效的浮点数，它会更新 `options.lora_scale`。 `changed` 标志指示 LoRA scale 是否真正被修改。

**Demo Usage (演示用法):**

```python
# Example usage (模拟使用)
options = SamplingOptions(prompt="a cat", width=512, height=512, num_steps=50, guidance=7.5, seed=None, img_cond_path="image.png", lora_scale=0.5)

options, changed = parse_lora_scale(options)
if options and changed:
    print(f"LoRA scale is now: {options.lora_scale}")
```

If you run this and type `0.75` then press enter, the LoRA scale will be updated to 0.75, and `changed` will be `True`.

**5. `main` Function:**

```python
@torch.inference_mode()
def main(
    name: str,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    prompt: str = "a robot made out of gold",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float | None = None,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/robot.webp",
    lora_scale: float | None = 0.85,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
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

    assert name in [
        "flux-dev-canny",
        "flux-dev-depth",
        "flux-dev-canny-lora",
        "flux-dev-depth-lora",
    ], f"Got unknown model name: {name}"
    if guidance is None:
        if name in ["flux-dev-canny", "flux-dev-canny-lora"]:
            guidance = 30.0
        elif name in ["flux-dev-depth", "flux-dev-depth-lora"]:
            guidance = 10.0
        else:
            raise NotImplementedError()

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
    t5 = load_t5(torch_device, max_length=512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # set lora scale
    if "lora" in name and lora_scale is not None:
        for _, module in model.named_modules():
            if hasattr(module, "set_scale"):
                module.set_scale(lora_scale)

    if name in ["flux-dev-depth", "flux-dev-depth-lora"]:
        img_embedder = DepthImageEncoder(torch_device)
    elif name in ["flux-dev-canny", "flux-dev-canny-lora"]:
        img_embedder = CannyImageEncoder(torch_device)
    else:
        raise NotImplementedError()

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
        lora_scale=lora_scale,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)
        if "lora" in name:
            opts, changed = parse_lora_scale(opts)
            if changed:
                # update the lora scale:
                for _, module in model.named_modules():
                    if hasattr(module, "set_scale"):
                        module.set_scale(opts.lora_scale)

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
            t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)
        inp = prepare_control(
            t5,
            clip,
            x,
            prompt=opts.prompt,
            ae=ae,
            encoder=img_embedder,
            img_cond_path=opts.img_cond_path,
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
            if "lora" in name:
                opts, changed = parse_lora_scale(opts)
                if changed:
                    # update the lora scale:
                    for _, module in model.named_modules():
                        if hasattr(module, "set_scale"):
                            module.set_scale(opts.lora_scale)
        else:
            opts = None
```

**描述 (描述):**

*   This is the main function that orchestrates the image generation process. It loads the necessary models (T5, CLIP, flow model, autoencoder), prepares the input, denoises the noise, decodes the latent space to pixels, and saves the image. It handles both single image generation and interactive looping. The `offload` option is used to move models to CPU to save GPU memory.

*   **中文解释:** 这是主函数，它协调图像生成过程。 它加载必要的模型（T5、CLIP、flow model、自动编码器），准备输入，对噪声进行去噪，将潜在空间解码为像素，并保存图像。 它处理单张图像生成和交互式循环。 `offload` 选项用于将模型移动到 CPU 以节省 GPU 内存。

**Key Steps Explained (关键步骤解释):**

1.  **Model Loading (模型加载):** Loads pre-trained models (T5, CLIP, flow model, autoencoder).
2.  **Input Preparation (输入准备):**
    *   Generates initial noise (`get_noise`).
    *   Prepares conditioning information (prompt, image conditioning) using `prepare_control`.  This function utilizes T5, CLIP, the autoencoder, and the image embedder to create the necessary input tensors for the flow model.
3.  **Denoising (去噪):**  The core sampling loop uses the flow model to iteratively denoise the initial noise (`denoise`).  The `timesteps` are a schedule that determines the noise levels at each step.  The `guidance` value controls the strength of the prompt.
4.  **Decoding (解码):** Decodes the denoised latent space back into pixel space using the autoencoder (`ae.decode`).
5.  **Saving (保存):** Saves the generated image (`save_image`). It also uses an NSFW classifier to filter inappropriate content and adds metadata to the image.

**6. `app` and `if __name__ == "__main__":`:**

```python
def app():
    Fire(main)


if __name__ == "__main__":
    app()
```

**描述 (描述):**

*   This part uses `fire` to make the `main` function accessible from the command line.  When you run the script, `Fire(main)` will automatically generate command-line arguments based on the `main` function's parameters.  The `if __name__ == "__main__":` block ensures that the `app()` function is only called when the script is run directly, not when it's imported as a module.

*   **中文解释:**  这部分使用 `fire` 使 `main` 函数可以从命令行访问。 当你运行脚本时，`Fire(main)` 会自动根据 `main` 函数的参数生成命令行参数。 `if __name__ == "__main__":` 块确保 `app()` 函数仅在直接运行脚本时才会被调用，而不是当它作为模块导入时。

**How to Run (如何运行):**

1.  **Install dependencies (安装依赖):**

    ```bash
    pip install torch transformers fire
    ```

2.  **Run the script (运行脚本):**

    ```bash
    python your_script_name.py --name flux-dev-canny --prompt "a futuristic city" --width 512 --height 512
    ```

    Replace `your_script_name.py` with the actual name of your Python file. You can adjust the other arguments as needed.  If you want to use the interactive loop, add the `--loop` argument:

    ```bash
    python your_script_name.py --name flux-dev-canny --loop
    ```

    This will start the script in interactive mode, prompting you for prompts, image paths, and other parameters.

**Important Notes (重要提示):**

*   **`flux` package:** This code relies on the `flux` package, which is not a standard PyPI package.  You'll need to obtain and install this package separately. The code assumes it's available in your Python environment.

*   **Model weights:** You'll also need to download the model weights for the specific `name` you choose (e.g., `flux-dev-canny`).  The code assumes that the `load_flow_model`, `load_ae`, `load_clip`, and `load_t5` functions handle the loading of these weights.

*   **Hardware Requirements:** Image generation can be resource-intensive.  A GPU is highly recommended for faster performance. The `offload` option can help reduce GPU memory usage, but it will slow down the process.
