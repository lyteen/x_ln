Lan: `py` From`dl/flux/src\flux\util.py`

Okay, restarting with explanations and Chinese descriptions.

```python
import os
from dataclasses import dataclass

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from imwatermark import WatermarkEncoder
from PIL import ExifTags, Image
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxLoraWrapper, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder


def save_image(
    nsfw_classifier,
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    nsfw_threshold: float = 0.85,
) -> int:
    fn = output_name.format(idx=idx)
    print(f"Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

    if nsfw_score < nsfw_threshold:
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        idx += 1
    else:
        print("Your generated image may contain NSFW content.")

    return idx


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    lora_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
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
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
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
            guidance_embed=False,
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
    ),
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_CANNY"),
        lora_path=None,
        params=FluxParams(
            in_channels=128,
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
    ),
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=os.getenv("FLUX_DEV_CANNY_LORA"),
        params=FluxParams(
            in_channels=128,
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
    ),
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_DEPTH"),
        lora_path=None,
        params=FluxParams(
            in_channels=128,
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
    ),
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=os.getenv("FLUX_DEV_DEPTH_LORA"),
        params=FluxParams(
            in_channels=128,
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
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FILL"),
        lora_path=None,
        params=FluxParams(
            in_channels=384,
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
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    # 打印加载警告。如果checkpoint缺少或者有多余的key，就打印出来。
    # Prints loading warnings. If the checkpoint has missing or unexpected keys, print them.
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(
    name: str, device: str | torch.device = "cuda", hf_download: bool = True, verbose: bool = False
) -> Flux:
    # 加载 Flux 模型
    # Loading Flux Model
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    lora_path = configs[name].lora_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        # 如果没有本地checkpoint路径，并且配置了Hugging Face Hub，则从HF Hub下载。
        # If there's no local checkpoint path and HF Hub is configured, download from HF Hub.
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    #  使用 "meta" 设备初始化模型，如果 ckpt_path 为 None，则使用指定的设备
    # Initialize the model using the "meta" device if ckpt_path is None, otherwise the specified device.
    with torch.device("meta" if ckpt_path is not None else device):
        if lora_path is not None:
            # 如果存在LoRA路径，则使用LoRA包装器。
            # Use the LoRA wrapper if a LoRA path exists.
            model = FluxLoraWrapper(params=configs[name].params).to(torch.bfloat16)
        else:
            model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # 加载 checkpoint
        # load checkpoint
        # load_sft 不支持 torch.device，所以先转换成字符串
        # load_sft doesn't support torch.device, so convert to string first
        sd = load_sft(ckpt_path, device=str(device))
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)

    if configs[name].lora_path is not None:
        print("Loading LoRA")
        # 加载 LoRA
        # load LoRA
        lora_sd = load_sft(configs[name].lora_path, device=str(device))
        # 加载 LoRA 参数 + 覆盖 norms 中的 scale 值
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # 加载 T5 模型
    # Load T5 Model
    # max length 64, 128, 256 和 512 应该都可以用（如果你的序列足够短）
    # max length 64, 128, 256, and 512 should all work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    # 加载 CLIP 模型
    # Load CLIP Model
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    # 加载 AutoEncoder 模型
    # Load AutoEncoder Model
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        # 如果没有本地checkpoint路径，并且配置了Hugging Face Hub，则从HF Hub下载。
        # If there's no local checkpoint path and HF Hub is configured, download from HF Hub.
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # 加载 AutoEncoder
    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    可选地扩展 state_dict 以匹配模型的参数形状。
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
                )
                # expand with zeros: 用零扩展
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict


class WatermarkEmbedder:
    # 水印嵌入器
    # Watermark Embedder
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image
        向输入图像添加预定义的水印

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
            与输入相同，但带有水印
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        # 水印库期望输入为 cv2 BGR 格式
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# A fixed 48-bit message that was chosen at random
# 随机选择的固定 48 位消息
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
# bin(x)[2:] 将 x 的位作为 str 给出，使用 int 将它们转换为 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
```

**Explanation of Key Parts (关键部分解释):**

1.  **`save_image` Function (保存图像函数):**

    *   负责将生成的图像保存到文件。
    *   Applies clamping, watermark embedding, and re-arranges the tensor dimensions to be compatible with `PIL.Image`.
    *   Adds EXIF metadata to the image, indicating it was AI-generated, the model used, and optionally the prompt.
    *   使用PIL.Image转换为图像格式并保存文件.
    *   Also includes NSFW filtering before saving.
        保存之前会检查图片是否为nsfw.

    ```python
    def save_image(
        nsfw_classifier,
        name: str,
        output_name: str,
        idx: int,
        x: torch.Tensor,
        add_sampling_metadata: bool,
        prompt: str,
        nsfw_threshold: float = 0.85,
    ) -> int:
        fn = output_name.format(idx=idx) #生成文件名
        print(f"Saving {fn}")
        # bring into PIL format and save 转换为PIL格式并保存
        x = x.clamp(-1, 1) #将像素值限制在-1到1之间
        x = embed_watermark(x.float()) #嵌入水印
        x = rearrange(x[0], "c h w -> h w c") #调整tensor的维度

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy()) #转换为PIL图像
        nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0] #检查是否为nsfw

        if nsfw_score < nsfw_threshold:
            exif_data = Image.Exif() #添加exif信息
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = name
            if add_sampling_metadata:
                exif_data[ExifTags.Base.ImageDescription] = prompt
            img.save(fn, exif=exif_data, quality=95, subsampling=0) #保存图像
            idx += 1
        else:
            print("Your generated image may contain NSFW content.")

        return idx

    # Demo Usage 演示用法
    # Assuming you have a dummy nsfw_classifier and a tensor 'x'
    # 假设你有一个虚拟的nsfw_classifier和一个tensor 'x'
    # dummy_nsfw_classifier = lambda img: [{"label": "nsfw", "score": 0.2}] # Mock classifier
    # dummy_image_tensor = torch.randn(1, 3, 256, 256) # Example image tensor
    # save_image(dummy_nsfw_classifier, "test_model", "output_{idx}.png", 0, dummy_image_tensor, True, "test prompt")
    ```

2.  **`ModelSpec` Dataclass (模型规格数据类):**

    *   定义了模型配置的结构，包括模型参数、自动编码器参数、checkpoint路径和LoRA路径。
    *   Organizes the configurations for different models. Includes Flux parameters, AutoEncoder parameters, checkpoint paths, and LoRA paths.

    ```python
    @dataclass
    class ModelSpec:
        params: FluxParams #Flux模型参数
        ae_params: AutoEncoderParams #AutoEncoder模型参数
        ckpt_path: str | None #checkpoint路径
        lora_path: str | None #LoRA路径
        ae_path: str | None #AutoEncoder路径
        repo_id: str | None #Hugging Face Hub仓库ID
        repo_flow: str | None #Flux模型文件名
        repo_ae: str | None #AutoEncoder模型文件名

    # Example Usage 示例用法
    # model_config = ModelSpec(
    #     params=FluxParams(...), #省略了FluxParams的初始化
    #     ae_params=AutoEncoderParams(...), #省略了AutoEncoderParams的初始化
    #     ckpt_path="path/to/checkpoint.safetensors",
    #     lora_path="path/to/lora.safetensors",
    #     ae_path="path/to/ae.safetensors",
    #     repo_id="my_repo",
    #     repo_flow="flow.safetensors",
    #     repo_ae="ae.safetensors"
    # )
    ```

3.  **`configs` Dictionary (配置字典):**

    *   A dictionary that maps model names to their corresponding `ModelSpec` instances.
        将模型名称映射到对应的`ModelSpec`实例的字典。
    *   Contains configurations for various Flux models, including their parameters, paths, and Hugging Face repository details.

    ```python
    configs = {
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
        ),
    ...
    }
    #Access the configuration for "flux-dev"
    #访问"flux-dev"的配置
    #flux_dev_config = configs["flux-dev"]
    #print(flux_dev_config.repo_id) #Output: black-forest-labs/FLUX.1-dev
    ```

4.  **`print_load_warning` Function (打印加载警告函数):**

    *   在加载模型时，如果发现checkpoint文件缺少键或者有多余的键，就打印警告。
    *   Prints warnings if the model loading process encounters missing or unexpected keys in the checkpoint.

    ```python
    def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
        # 打印加载警告。如果checkpoint缺少或者有多余的key，就打印出来。
        # Prints loading warnings. If the checkpoint has missing or unexpected keys, print them.
        if len(missing) > 0 and len(unexpected) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
            print("\n" + "-" * 79 + "\n")
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
        elif len(missing) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        elif len(unexpected) > 0:
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

    #Demo Usage 演示用法
    #print_load_warning(["layer1.weight", "layer2.bias"], ["extra_layer.weight"]) #Prints the missing and unexpected keys
    ```

5.  **`load_flow_model` Function (加载 Flux 模型函数):**

    *   负责加载 Flux 模型，包括从本地路径或 Hugging Face Hub 下载。
    *   Handles loading the Flux model, including downloading from the Hugging Face Hub if necessary and loading LoRA weights.

    ```python
    def load_flow_model(
        name: str, device: str | torch.device = "cuda", hf_download: bool = True, verbose: bool = False
    ) -> Flux:
        # 加载 Flux 模型
        # Loading Flux Model
        print("Init model")
        ckpt_path = configs[name].ckpt_path #从configs中获取checkpoint路径
        lora_path = configs[name].lora_path #从configs中获取LoRA路径
        if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_flow is not None
            and hf_download
        ):
            # 如果没有本地checkpoint路径，并且配置了Hugging Face Hub，则从HF Hub下载。
            # If there's no local checkpoint path and HF Hub is configured, download from HF Hub.
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow) #从HF Hub下载checkpoint

        #  使用 "meta" 设备初始化模型，如果 ckpt_path 为 None，则使用指定的设备
        # Initialize the model using the "meta" device if ckpt_path is None, otherwise the specified device.
        with torch.device("meta" if ckpt_path is not None else device): #使用meta device初始化模型
            if lora_path is not None:
                # 如果存在LoRA路径，则使用LoRA包装器。
                # Use the LoRA wrapper if a LoRA path exists.
                model = FluxLoraWrapper(params=configs[name].params).to(torch.bfloat16) #使用LoRA wrapper
            else:
                model = Flux(configs[name].params).to(torch.bfloat16) #创建Flux模型

        if ckpt_path is not None:
            print("Loading checkpoint")
            # 加载 checkpoint
            # load checkpoint
            # load_sft 不支持 torch.device，所以先转换成字符串
            # load_sft doesn't support torch.device, so convert to string first
            sd = load_sft(ckpt_path, device=str(device)) #使用safetensors加载checkpoint
            sd = optionally_expand_state_dict(model, sd) #可选地扩展state_dict
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True) #加载模型参数
            if verbose:
                print_load_warning(missing, unexpected) #打印加载警告

        if configs[name].lora_path is not None:
            print("Loading LoRA")
            # 加载 LoRA
            # load LoRA
            lora_sd = load_sft(configs[name].lora_path, device=str(device)) #加载LoRA参数
            # 加载 LoRA 参数 + 覆盖 norms 中的 scale 值
            # loading the lora params + overwriting scale values in the norms
            missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True) #加载LoRA参数
            if verbose:
                print_load_warning(missing, unexpected) #打印加载警告
