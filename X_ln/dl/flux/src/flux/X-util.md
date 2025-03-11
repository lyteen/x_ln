Lan: `py` From`dl/flux/src\flux\util.py`

好的，我们重新开始。我将提供一些代码片段的改进，并附带中文描述和简单的演示。

**1. 改进的 `ModelSpec` 和 `configs` 数据结构:**

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class ModelSpec:
    """模型配置规范，包含模型参数、路径等信息。"""
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: Optional[str] = None
    lora_path: Optional[str] = None
    ae_path: Optional[str] = None
    repo_id: Optional[str] = None
    repo_flow: Optional[str] = None
    repo_ae: Optional[str] = None
    description: Optional[str] = None  # Add a description

# 示例配置 - 更清晰，更易于维护.  Example Configurations - Clearer and easier to maintain.
configs: Dict[str, ModelSpec] = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=None,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
        description="开发版本 - 用于一般用途。 Development version - for general purposes."
    ),
    # ... 更多配置 ... more configurations
}


# 演示用法 Demo Usage:
if __name__ == "__main__":
    model_config = configs["flux-dev"]
    print(f"模型名称: {model_config.__class__.__name__}")
    print(f"模型描述: {model_config.description}")
    print(f"CKPT路径: {model_config.ckpt_path}")

```

**描述:**

这段代码使用 `dataclass` 重新定义了 `ModelSpec`，并使用了类型提示 (`Optional[str]`)。`Optional` 表明一个字段可以是字符串或 `None`。此外，增加了一个 `description` 字段，用于添加模型的简短描述。`configs` 字典现在使用了类型提示 `Dict[str, ModelSpec]`，使其更加清晰。

**主要改进:**

*   **类型提示:**  使用类型提示可以提高代码的可读性和可维护性。
*   **更清晰的数据结构:**  `dataclass` 提供了一种简洁的方式来定义数据类。
*   **描述字段:**  添加了 `description` 字段，可以更好地组织和理解模型配置。

**如何使用:**

创建一个 `ModelSpec` 实例，并将其添加到 `configs` 字典中。  你可以通过键来访问配置，例如 `configs["flux-dev"]`。

---

**2. 改进的 `load_flow_model` 函数:**

```python
def load_flow_model(
    name: str, device: str | torch.device = "cuda", hf_download: bool = True, verbose: bool = False
) -> Flux:
    """加载Flux模型，支持从本地或 Hugging Face Hub 下载。Loads a Flux model, supporting download from local or Hugging Face Hub."""

    config = configs.get(name)
    if not config:
        raise ValueError(f"未找到名为 '{name}' 的模型配置. Model configuration not found for '{name}'.")

    print(f"正在初始化模型: {name}  Initializing model: {name}")
    ckpt_path = config.ckpt_path
    lora_path = config.lora_path

    # 从 Hugging Face Hub 下载模型 Download model from Hugging Face Hub if necessary
    if (
        ckpt_path is None
        and config.repo_id is not None
        and config.repo_flow is not None
        and hf_download
    ):
        try:
            ckpt_path = hf_hub_download(config.repo_id, config.repo_flow)
            print(f"从 Hugging Face Hub 下载模型成功: {ckpt_path} Downloaded model successfully from Hugging Face Hub: {ckpt_path}")
        except Exception as e:
            print(f"从 Hugging Face Hub 下载模型失败: {e} Failed to download model from Hugging Face Hub: {e}")
            raise

    with torch.device("meta" if ckpt_path is not None else device):
        if lora_path is not None:
            model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
        else:
            model = Flux(config.params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("正在加载检查点 Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)

    if config.lora_path is not None:
        print("正在加载 LoRA Loading LoRA")
        lora_sd = load_sft(config.lora_path, device=str(device))
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
    return model
```

**描述:**

这段代码改进了 `load_flow_model` 函数，使其更加健壮，并提供了更好的错误处理。

**主要改进:**

*   **配置验证:**  在函数开始时，验证是否找到了给定的 `name` 的模型配置。如果未找到，则会引发 `ValueError`。
*   **Hugging Face Hub 下载错误处理:**  使用 `try...except` 块来捕获从 Hugging Face Hub 下载模型时可能发生的任何异常。如果下载失败，则会打印一条错误消息，并重新引发该异常。
*   **日志记录:**  添加了更多日志记录语句，以提供有关模型加载过程的更多信息。

**如何使用:**

调用 `load_flow_model` 函数，并传入模型名称、设备和其他可选参数。  该函数将加载模型并返回 `Flux` 实例。

---

**3. 改进的 `save_image` 函数:**

```python
from typing import Callable

def save_image(
    nsfw_classifier: Callable, # Explicitly type nsfw_classifier as a Callable
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    nsfw_threshold: float = 0.85,
) -> int:
    """保存图像，如果 NSFW 分数低于阈值。Saves an image if the NSFW score is below the threshold."""
    fn = output_name.format(idx=idx)
    print(f"正在保存 {fn}  Saving {fn}")
    try:
        # 确保图像数据在正确的范围内 Ensure image data is in the correct range
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        # 使用 NSFW 分类器 Use NSFW classifier
        nsfw_results = nsfw_classifier(img)
        nsfw_score = next((result["score"] for result in nsfw_results if result["label"] == "nsfw"), 0.0) # Default to 0.0 if 'nsfw' label not found

        if nsfw_score < nsfw_threshold:
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = name
            if add_sampling_metadata:
                exif_data[ExifTags.Base.ImageDescription] = prompt
            img.save(fn, exif=exif_data, quality=95, subsampling=0)
            idx += 1
            print(f"图像已保存: {fn}  Image saved: {fn}")
        else:
            print("生成的图像可能包含 NSFW 内容。  Your generated image may contain NSFW content.")

    except Exception as e:
        print(f"保存图像时出错: {e}  Error saving image: {e}")

    return idx
```

**描述:**

这段代码改进了 `save_image` 函数，使其更加健壮，并提供了更好的错误处理和类型提示。

**主要改进:**

*   **类型提示:** 显式地将 `nsfw_classifier` 参数类型设置为 `Callable`。
*   **NSFW 结果处理:**  使用 `next` 函数来安全地从 NSFW 分类器结果中提取 NSFW 分数。 如果找不到“nsfw”标签，则默认为 0.0。
*   **Try...Except 块:**  使用 `try...except` 块来捕获保存图像时可能发生的任何异常。 这可以防止脚本因意外错误而崩溃。
*   **详细的日志记录:**  添加了详细的日志记录，以指示图像何时成功保存，以及何时由于 NSFW 内容而被跳过。
*   **代码格式化:** 代码更加清晰，可读性更强。

**如何使用:**

调用 `save_image` 函数，传入 NSFW 分类器、模型名称、输出名称、图像张量和其他可选参数。 该函数将保存图像（如果 NSFW 分数低于阈值）并返回更新后的索引。

这些改进旨在提高代码的可靠性、可维护性和可读性。  每个改进都附带了中文描述，以便于理解。希望这些对你有所帮助！
