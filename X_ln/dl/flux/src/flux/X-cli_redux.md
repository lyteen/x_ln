Lan: `py` From`dl/flux/src\flux\cli_redux.py`

Okay, let's refine this code for better clarity and efficiency. I'll break it down into smaller, well-documented parts.

**1. Improved Configuration Handling:**

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "flux-dev"
    width: int = 1360
    height: int = 768
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps: Optional[int] = None
    guidance: float = 2.5
    offload: bool = False
    output_dir: str = "output"
    add_sampling_metadata: bool = True
    img_cond_path: str = "assets/robot.webp"

    def __post_init__(self):
        if self.width % 16 != 0:
            self.width = 16 * (self.width // 16)
            print(f"Width adjusted to {self.width} for divisibility by 16.")
        if self.height % 16 != 0:
            self.height = 16 * (self.height // 16)
            print(f"Height adjusted to {self.height} for divisibility by 16.")
        os.makedirs(self.output_dir, exist_ok=True)

# Example Usage
if __name__ == '__main__':
    config = ModelConfig(width=1367, height=770, output_dir="my_output")
    print(f"Model Name: {config.name}")
    print(f"Output Directory: {config.output_dir}")
```

**描述 (描述):**

*   **`ModelConfig` dataclass:**  使用 `dataclass` 来集中管理配置参数，使代码更易读和维护。
*   **Type Hints (类型提示):**  添加了类型提示 (例如 `Optional[int]`)，以提高代码的可读性和可维护性，并且IDE可以进行校验.
*   **Input Validation (输入验证):**  `__post_init__` 方法在对象创建后自动运行，用于验证和调整输入，确保宽度和高度是 16 的倍数。
*   **Directory Creation (目录创建):**  自动创建输出目录，避免手动创建。
*  **示例用法 (示例用法):** 提供了一个简单的例子，展示了如何使用 `ModelConfig` 类。

**优点 (优点):**

*   结构化的配置管理.
*   自动输入验证.
*   代码更清晰，更易于理解.

---

**2. Refactored Image Saving:**

```python
import os
import re
from glob import iglob
from PIL import Image, ImageDraw, ImageFont
from typing import List

def find_next_index(output_dir: str, output_name: str) -> int:
    """Find the next available index for saving images."""
    pattern = re.compile(r"img_(\d+)\.jpg$")
    existing_indices = []
    for filename in iglob(os.path.join(output_dir, output_name.replace("{idx}", "*"))):
        match = pattern.search(filename)
        if match:
            existing_indices.append(int(match.group(1)))
    return (max(existing_indices) + 1) if existing_indices else 0


def save_image_with_metadata(
    image: torch.Tensor,
    output_path: str,
    prompt: str,
    add_sampling_metadata: bool = True,
):
    """Saves a PyTorch tensor as an image with optional metadata."""
    img = Image.fromarray((image.cpu().numpy().clip(0, 1) * 255).astype("uint8").squeeze()) # Convert tensor to PIL Image

    if add_sampling_metadata and prompt:
        try:
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", size=14) # You might need to adjust the path
            draw.text((10, 10), prompt, (0, 0, 0), font=font)
        except Exception as e:
            print(f"Warning: Could not add metadata due to {e}")

    img.save(output_path, "JPEG")  # Save as JPEG

# Example Usage
if __name__ == '__main__':
    dummy_image = torch.rand((1, 3, 256, 256))
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True) # create output if not exists
    output_name = "img_{idx}.jpg"
    idx = find_next_index(output_dir, output_name)
    output_path = os.path.join(output_dir, output_name.format(idx=idx))
    save_image_with_metadata(dummy_image, output_path, "A beautiful landscape")
    print(f"Image saved to {output_path}")
```

**描述 (描述):**

*   **`find_next_index` function:**  提取出来计算下一个可用索引的代码，使其更易于测试和重用。
*   **`save_image_with_metadata` function:** 将图像保存功能封装在一个函数中，并添加了可选的元数据。
*   **Clear Image Conversion (清晰的图像转换):** 使用更明确的代码将张量转换为 PIL 图像。
*   **Error Handling (错误处理):**  在添加元数据时添加了 try-except 块，以防止错误导致程序崩溃。
*   **字体设置 (字体设置)** 增加了对字体不存在的异常处理
*   **示例用法 (示例用法):** 展示如何使用这些功能保存带元数据的图像。

**优点 (优点):**

*   代码模块化.
*   更强大的错误处理.
*   更清晰的图像转换步骤.

---

**3. Simplified Prompt Parsing:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingOptions:
    prompt: str = ""
    width: int = 1360
    height: int = 768
    num_steps: int = 50
    guidance: float = 2.5
    seed: Optional[int] = None
    img_cond_path: str = "assets/robot.webp"


def parse_prompt(options: SamplingOptions) -> Optional[SamplingOptions]:
    """Parses user input to update sampling options."""
    while True:
        user_question = "Write /h for help, /q to quit and leave empty to repeat):\n"
        prompt = input(user_question)

        if prompt.startswith("/"):
            command = prompt.split()[0]  # Extract command
            try:
                if command == "/w":
                    options.width = 16 * (int(prompt.split()[1]) // 16)
                    print(f"Setting width to {options.width}")
                elif command == "/h":
                    options.height = 16 * (int(prompt.split()[1]) // 16)
                    print(f"Setting height to {options.height}")
                elif command == "/s":
                    options.seed = int(prompt.split()[1])
                    print(f"Setting seed to {options.seed}")
                elif command == "/g":
                    options.guidance = float(prompt.split()[1])
                    print(f"Setting guidance to {options.guidance}")
                elif command == "/n":
                    options.num_steps = int(prompt.split()[1])
                    print(f"Setting steps to {options.num_steps}")
                elif command == "/q":
                    print("Quitting")
                    return None
                elif command == "/h":
                    print(get_usage()) # 调用usage函数，避免代码冗余
                else:
                    print(f"Invalid command: {command}")
                    print(get_usage())
            except (IndexError, ValueError):
                print("Invalid command format.")
                print(get_usage())
        else:
            options.prompt = prompt  # Update the prompt
            break  # Exit loop
    return options

def get_usage():
    return (
        "Usage: Leave this field empty to do nothing "
        "or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

# Example Usage (测试)
if __name__ == '__main__':
    options = SamplingOptions()
    updated_options = parse_prompt(options)

    if updated_options:
        print(f"Updated Prompt: {updated_options.prompt}")
        print(f"Updated Width: {updated_options.width}")
        print(f"Updated Height: {updated_options.height}")
        print(f"Updated Seed: {updated_options.seed}")
```

**描述 (描述):**

*   **Clearer Command Parsing (更清晰的命令解析):**  使用 `prompt.split()` 和 try-except 块来更简洁地解析命令。
*   **Error Handling (错误处理):**  添加了处理 `IndexError` 和 `ValueError` 的错误处理程序，以避免因格式错误的命令而导致的崩溃。
*   **Usage Message Function (用法消息函数):** 创建一个单独的函数来存储用法消息，避免冗余。
*   **返回类型(返回类型):** 明确返回 `Optional[SamplingOptions]` 以表明函数可能返回 `None`.
*   **示例用法 (示例用法):**  展示如何使用更新后的 `parse_prompt` 函数。

**优点 (优点):**

*   更简洁，可读性更强的代码.
*   处理输入错误能力更强.

---

**4. Main Function Integration (主函数集成):**

```python
import time
import torch
from fire import Fire
from transformers import pipeline
from flux.modules.image_embedders import ReduxImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_redux, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5

from typing import Optional

# Import other functions from above
from improved_vqvae import SimpleVQVAE  # 假设改进的VQVAE保存在 improved_vqvae.py
# Import ModelConfig, SamplingOptions, parse_prompt, find_next_index, save_image_with_metadata, get_usage from above
# ...

@torch.inference_mode()
def main(config: ModelConfig = ModelConfig(), loop: bool = False):
    """Main function to run the sampling process."""

    nsfw_classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection", device=config.device
    )

    if config.name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {config.name}. Choose from {available}")

    torch_device = torch.device(config.device)
    num_steps = config.num_steps if config.num_steps else (4 if config.name == "flux-schnell" else 50)

    output_name = "img_{idx}.jpg"
    idx = find_next_index(config.output_dir, output_name)

    # Initialize components
    t5 = load_t5(torch_device, max_length=256 if config.name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(config.name, device="cpu" if config.offload else torch_device)
    ae = load_ae(config.name, device="cpu" if config.offload else torch_device)
    img_embedder = ReduxImageEncoder(torch_device)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=config.prompt,
        width=config.width,
        height=config.height,
        num_steps=num_steps,
        guidance=config.guidance,
        seed=config.seed,
        img_cond_path=config.img_cond_path,
    )

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # Prepare input
        x = get_noise(
            1, opts.height, opts.width, device=torch_device, dtype=torch.bfloat16, seed=opts.seed
        )
        opts.seed = None
        if config.offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare_redux(
            t5, clip, x, prompt=opts.prompt, encoder=img_embedder, img_cond_path=opts.img_cond_path
        )
        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(config.name != "flux-schnell"))

        # Offload TEs to CPU, load model to GPU
        if config.offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # Denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # Offload model, load autoencoder to GPU
        if config.offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # Decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        output_path = os.path.join(config.output_dir, output_name.format(idx=idx))
        save_image_with_metadata(x, output_path, opts.prompt, config.add_sampling_metadata)
        idx += 1 # Increment outside the function
        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        else:
            opts = None

def app():
    Fire(main)

if __name__ == "__main__":
    app()
```

**描述 (描述):**

*   **Configuration Object (配置对象):**  使用 `ModelConfig` 对象将所有配置参数传递给 `main` 函数。
*   **Centralized Index Management (集中索引管理):**  在 `main` 函数中管理 `idx` 变量，避免全局状态。
*   **Clearer Structure (更清晰的结构):**  将代码组织成更小的，更易于理解的块。
*   **示例用法 (示例用法):**   展示如何使用更新后的 `main` 函数和 `Fire` 库。

**优点 (优点):**

*   所有配置都集中在一个地方.
*   代码更易于测试.
*   结构更清晰.

**如何运行 (如何运行):**

1.  **Save the Code (保存代码):** 将这些代码片段保存到单独的 Python 文件中 (例如, `config.py`, `image_utils.py`, `prompt_parser.py`, `main.py`).
2.  **Install Dependencies (安装依赖项):** 使用 `pip install torch fire transformers Pillow` 安装必要的库.
3.  **Run the Main Script (运行主脚本):** 运行 `python main.py --loop` 启动交互式会话，或 `python main.py` 运行单次采样.  您可以使用命令行参数覆盖默认配置 (例如, `python main.py --width=512 --height=512`).

This revised structure provides better organization, error handling, and readability. It leverages dataclasses for configuration, separates concerns into functions, and includes error handling and clear examples.  The comments are in both English and Chinese to aid understanding.  Remember to install all the required libraries.
