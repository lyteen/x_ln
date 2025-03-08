Lan: `py` From`X_ln\dl/open_clip/src\open_clip\factory.html`

Okay, I will restart and provide explanations and code snippets for the given `open_clip` library code.

**1. Model Configuration and Discovery**

```python
import json
from pathlib import Path
import re

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs()
def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())

def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    """ Fetch model config from builtin (local library) configs.
    """
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

# Demo Usage 演示用法
if __name__ == '__main__':
    # List available models 列出可用模型
    print(f"Available models: {list_models()}")

    # Example usage of get_model_config 获取模型配置的例子
    model_name = "ViT-B-32" # 示例模型名 (Example model name) - NOTE: This must exist in your model_configs
    model_config = get_model_config(model_name)

    if model_config:
        print(f"Configuration for {model_name}: {model_config.keys()}") # print keys for the configuration
    else:
        print(f"No configuration found for {model_name}")

    # Example: add model config:  (For Demo purpose only) - 只是为了演示
    # create a dummy config file in a tmp directory
    import tempfile
    import os
    dummy_config = {"embed_dim": 512, "vision_cfg": {"image_size": 224}, "text_cfg": {"context_length": 77}}
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "dummy_model.json"
        with open(config_path, "w") as f:
            json.dump(dummy_config, f)

        add_model_config(config_path)
        print(f"Available models after adding config: {list_models()}")
```

**描述:**
- **`_MODEL_CONFIG_PATHS`**: 存储模型配置文件的路径列表.
- **`_MODEL_CONFIGS`**: 一个字典，用于缓存已加载的模型配置.  键是模型名称，值是配置数据.
- **`_rescan_model_configs()`**: 扫描指定路径下的所有JSON文件，加载模型配置，并将它们存储在`_MODEL_CONFIGS`字典中. 它会检查JSON文件是否包含`embed_dim`, `vision_cfg`, `text_cfg`这三个键.  使用`_natural_key`进行排序，保证模型名称按照自然顺序排列（例如，ViT-B-16在ViT-B-32之前）.
- **`list_models()`**: 返回可用模型架构的列表.
- **`add_model_config(path)`**: 添加一个新的模型配置路径或文件，并更新模型配置注册表.
- **`get_model_config(model_name)`**: 从内置配置中获取指定模型的配置. 如果找到配置，返回其深拷贝，否则返回 `None`.
**使用方式:**  `_rescan_model_configs`初始化模型配置，`list_models()`可以获取可用模型名称，`get_model_config(model_name)` 获取指定模型的详细配置，该配置用于创建相应的模型实例.

**2. Hugging Face Hub Configuration Retrieval**

```python
from typing import Optional
from .pretrained import download_pretrained_from_hf  # Assuming this import works within the file context

def _get_hf_config(
        model_id: str,
        cache_dir: Optional[str] = None,
):
    """ Fetch model config from HuggingFace Hub.
    """
    config_path = download_pretrained_from_hf(
        model_id,
        filename='open_clip_config.json',
        cache_dir=cache_dir,
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config
```

**描述:**
- **`_get_hf_config(model_id, cache_dir)`**:  从Hugging Face Hub下载模型配置文件 (名为 `open_clip_config.json`)，并将其解析为 Python 字典.  `model_id` 指定 Hugging Face 仓库的名称 (例如 "openai/clip-vit-base-patch32"). `cache_dir` 指定下载文件的缓存目录. `download_pretrained_from_hf`是一个假设存在的函数，作用是从HF下载指定文件。

**使用方式:**  当模型名称以 `hf-hub:` 开头时，此函数用于从 Hugging Face Hub 加载模型配置.

**3. Tokenizer Loading**

```python
from typing import Optional, Dict, Any
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH # Assuming these imports exist

def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
):
    """Get the text tokenizer for the model."""
    HF_HUB_PREFIX = 'hf-hub:' # Moved here for scope
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX):]
        try:
            config = _get_hf_config(model_name, cache_dir=cache_dir)['model_cfg']
        except Exception:
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                cache_dir=cache_dir,
                **kwargs,
            )
            return tokenizer
    else:
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        tokenizer = HFTokenizer(
            text_config['hf_tokenizer_name'],
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer

# Demo Usage 演示用法
if __name__ == '__main__':
    # Example: Get tokenizer for a specific model
    model_name = "ViT-B-32" # 示例模型名称，需要模型配置文件存在
    tokenizer = get_tokenizer(model_name)

    print(f"Tokenizer for {model_name}: {type(tokenizer)}")
```

**描述:**
- **`get_tokenizer(model_name, context_length, cache_dir, **kwargs)`**: 加载指定模型的文本分词器.
    - 如果 `model_name` 以 `hf-hub:` 开头，则从 Hugging Face Hub 加载模型配置并使用 `HFTokenizer`.  如果从HF下载config失败，则直接使用`HFTokenizer`进行加载
    - 否则，从本地配置加载模型配置.
    - 如果模型配置中指定了 `hf_tokenizer_name`，则使用 `HFTokenizer` (Hugging Face Tokenizer).
    - 否则，使用 `SimpleTokenizer`.
    -  `context_length` 是文本序列的最大长度.

**使用方式:**  此函数用于获取与特定 CLIP 模型关联的文本分词器.  分词器用于将文本转换为模型可以理解的数字 token.

**4. Checkpoint Loading**

```python
import torch
from pathlib import Path
from typing import Dict, Union
from safetensors.torch import load_file

def load_state_dict(
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    """Load state dict from checkpoint path."""
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(
        model: Union[CLIP, CustomTextCLIP], # Assuming CLIP and CustomTextCLIP are defined
        checkpoint_path: str,
        strict: bool = True,
        weights_only: bool = True,
        device='cpu',
):
    """Load checkpoint into model."""
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        # Separate path loading numpy big_vision (SigLIP) weights
        from open_clip.convert import load_big_vision_weights # Assuming this import exists
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # Detect & convert 3rd party state_dicts -> open_clip
    state_dict = convert_state_dict(model, state_dict) # Assuming this import exists

    # Detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)  # Assuming this import exists

    # correct if logit_scale differs in being scaler vs 1d param
    if 'logit_scale' in state_dict and model.logit_scale.ndim != state_dict['logit_scale'].ndim:
        state_dict['logit_scale'] = state_dict['logit_scale'].reshape(model.logit_scale.shape)

    # correct if logit_bias differs in being scaler vs 1d param
    if 'logit_bias' in state_dict and model.logit_bias.ndim != state_dict['logit_bias'].ndim:
        state_dict['logit_bias'] = state_dict['logit_bias'].reshape(model.logit_bias.shape)

    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if 'logit_bias' not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model) # Assuming this import exists
    resize_text_pos_embed(state_dict, model) # Assuming this import exists

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

# Demo Usage 演示用法
if __name__ == '__main__':
    # Create a dummy model
    class DummyCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = torch.nn.Parameter(torch.ones([])) # scalar
            self.logit_bias = None # Example to allow loading for SigLIP training purpose

        def load_state_dict(self, state_dict, strict=True):
            super().load_state_dict(state_dict, strict)

    dummy_model = DummyCLIP()

    # Create a dummy state dict
    dummy_state_dict = {"logit_scale": torch.tensor(2.6592)}

    # Save the dummy state dict to a file
    torch.save({"state_dict": dummy_state_dict}, "dummy_checkpoint.pth")

    # Load the dummy checkpoint
    incompatible_keys = load_checkpoint(dummy_model, "dummy_checkpoint.pth")
    print(f"Incompatible keys: {incompatible_keys}")

    # Clean up the dummy checkpoint file
    import os
    os.remove("dummy_checkpoint.pth")
```

**描述:**
- **`load_state_dict(checkpoint_path, device, weights_only)`**: 从给定的路径加载模型状态字典. 支持 `.pth` (PyTorch checkpoints) 和 `.safetensors` 格式.  根据文件后缀，选择合适的加载方式.
- **`load_checkpoint(model, checkpoint_path, strict, weights_only, device)`**: 将给定的检查点加载到模型中.
    - 如果检查点文件是 `.npz` 或 `.npy` 格式，则调用 `load_big_vision_weights` 函数（假定存在）来加载 SigLIP 权重.
    - 调用 `load_state_dict` 函数加载状态字典.
    - 调用 `convert_state_dict` 函数（假定存在）将第三方状态字典转换为 `open_clip` 格式.
    - 调整 `logit_scale` 的形状，以确保其与模型的 `logit_scale` 参数的形状匹配.
    - 调用 `resize_pos_embed` 和 `resize_text_pos_embed` 函数（假定存在）调整位置嵌入的大小.
    - 最后，使用 `model.load_state_dict` 函数将状态字典加载到模型中.  `strict` 参数控制是否需要模型的全部参数都在状态字典中.

**使用方式:**  `load_checkpoint` 是加载预训练权重的关键函数. 它处理各种检查点格式、状态字典转换和形状调整，以确保权重可以正确加载到模型中.

**5. Model Creation**

```python
import torch
import logging
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import asdict
from .model import CLIP, CustomTextCLIP # Assuming these imports exist
from .coca_model import CoCa # Assuming this import exists
from .pretrained import get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model # Assuming these imports exist
from .transform import PreprocessCfg, merge_preprocess_dict
from .model import get_cast_dtype, set_model_preprocess_cfg  # Assuming these imports exist

def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        load_weights_only: bool = True,
        **model_kwargs,
):
    """Creates and configures a contrastive vision-language model."""

    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    HF_HUB_PREFIX = 'hf-hub:' # Moved here for scope
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir) # Assuming this import works
        config = _get_hf_config(model_id, cache_dir=cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if force_patch_dropout is not None:
        # override the default patch dropout value
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
    if pretrained_image:
        if is_timm_model:
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert False, 'pretrained image towers currently only supported for timm models'

    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = get_cast_dtype(precision)
    is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
    if is_hf_model:
        # load pretrained weights for HF text model IFF no CLIP weights being loaded
        model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
    custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

    model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
    if custom_text:
        if "multimodal_cfg" in model_cfg:
            model = CoCa(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
    else:
        model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)
            from .transformer import LayerNormFp32 # Assuming this import works

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)
            model.apply(_convert_ln)
        else:
            model.to(device=device)
            from .model import convert_weights_to_lp # Assuming this import works
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    pretrained_loaded = False
    if pretrained:
        checkpoint_path = ''
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg:
            checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
            preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            #... [rest of pretrained config handling] ...
        elif os.path.exists(pretrained):
            checkpoint_path = pretrained

        if checkpoint_path:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path, weights_only=load_weights_only)
        # ... [rest of pretrained loading handling] ...
        pretrained_loaded = True
    elif has_hf_hub_prefix:
        logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
        load_checkpoint(model, checkpoint_path, weights_only=load_weights_only)
        pretrained_loaded = True

    if require_pretrained and not pretrained_loaded:
        raise RuntimeError(
            f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model

# Demo Usage 演示用法
if __name__ == '__main__':
    # Example: Create a CLIP model
    model_name = "ViT-B-32"  # 确保你的 'model_configs' 目录下有这个模型
    model = create_model(model_name, pretrained=None, device='cpu') # 不加载预训练权重

    print(f"Created model: {type(model)}")
```

**描述:**
- **`create_model(model_name, pretrained, precision, device, jit, ...)`**:  创建并配置一个对比视觉语言模型 (CLIP).
    -  `model_name` 指定要创建的模型架构.
    -  `pretrained` 指定预训练权重的标签或路径.
    -  `precision` 指定模型精度 (例如, "fp32", "fp16").
    -  `device` 指定设备 (例如, "cpu", "cuda").
    -  `jit` 如果为 True, 则使用 TorchScript JIT 编译模型.
    -  `force_quick_gelu`, `force_custom_text`, `force_patch_dropout`, `force_image_size` 允许覆盖模型配置中的默认值.
    - 如果 `model_name` 以 `hf-hub:` 开头，则从 Hugging Face Hub 加载模型配置和权重.
    - 根据 `model_cfg` 中的 `custom_text` 标志选择创建 `CLIP`，`CustomTextCLIP` 或 `CoCa` 模型.
    - 如果 `pretrained` 不为 None，则加载预训练权重.
    - 如果 `jit` 为 True，则使用 `torch.jit.script` 编译模型。
    - `set_model_preprocess_cfg`函数设定了图像预处理的参数。

**使用方式:**  `create_model` 是创建 CLIP 模型的入口点.  它负责加载模型配置、创建模型实例、加载预训练权重，以及配置模型精度和设备.

**6. Loss Function Creation**

```python
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss # Assuming these imports exist

def create_loss(args):
    """Create loss function based on arguments."""
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, "Horovod not currently supported for SigLip"
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
            dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
        )

    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )

# Demo Usage 演示用法
if __name__ == '__main__':
    # Dummy args object for testing 测试用的虚拟参数对象
    class Args:
        def __init__(self):
            self.distill = False
            self.model = "CLIP"
            self.local_loss = True
            self.gather_with_grad = False
            self.rank = 0
            self.world_size = 1
            self.horovod = False
            self.siglip = False
            self.coca_caption_loss_weight = 0.8
            self.coca_contrastive_loss_weight = 0.2
            self.loss_dist_impl = "torch"  # Default for SigLip

    args = Args()
    loss_fn = create_loss(args)

    print(f"Created loss function: {type(loss_fn)}")
```

**描述:**
- **`create_loss(args)`**:  根据给定的参数创建损失函数.
    - 如果 `args.distill` 为 True，则创建 `DistillClipLoss` (用于知识蒸馏).
    - 如果 `args.model` 包含 "coca"，则创建 `CoCaLoss`.
    - 如果 `args.siglip` 为 True，则创建 `SigLipLoss`.
    - 否则，创建 `ClipLoss`.
    - 该函数根据不同的训练目标选择不同的损失函数。

**使用方式:**  `create_loss` 用于实例化训练过程中使用的损失函数。

**7. Model and Transforms Creation**

```python
from typing import Optional, Tuple, Union, Dict, Any
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs  # Assuming this import exists
def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        load_weights_only: bool = True,
        **model_kwargs,
):
    """Creates model and image transforms."""
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        load_weights_only=load_weights_only,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val

# Demo Usage 演示用法
if __name__ == '__main__':
    # Example: Create a CLIP model and its transforms 创建 CLIP 模型和它的 transforms
    model_name = "ViT-B-32" # 确保你的 'model_configs' 目录下有这个模型
    model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=None, device='cpu')

    print(f"Created model: {type(model)}")
    print(f"Created train transform: {preprocess_train}")
    print(f"Created val transform: {preprocess_val}")
```

**描述:**
- **`create_model_and_transforms(model_name, pretrained, precision, device, jit, ...)`**:  创建一个 CLIP 模型和图像预处理 transforms.
    -  首先，调用 `create_model` 函数创建 CLIP 模型.
    -  然后，基于模型的 `preprocess_cfg` (预处理配置)  和给定的参数，创建训练和验证 transforms.
    -  `image_transform_v2` 函数 (假定存在) 用于创建 transforms.

**使用方式:**  `create_model_and_transforms` 简化了 CLIP 模型和 transforms 的创建过程.  返回的 `preprocess_train` 和 `preprocess_val` transforms 可以直接用于训练和验证数据加载器.

**8. Creation from Pretrained**

```python
from typing import Optional, Tuple, Union, Dict, Any
from .transform import image_transform_v2, PreprocessCfg, merge_preprocess_kwargs # Assuming these imports exist

def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_