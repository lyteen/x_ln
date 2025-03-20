Lan: `py` From`dl/DeepSeek-V3\inference\convert.py`

**目标:** 优化并详细解释提供的代码，使其更易于理解和维护。

**1. 更加模块化的代码结构:**

```python
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file

# 常量定义，集中管理
MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def rename_key(name: str) -> str:
    """
    重命名键，使其与目标格式匹配。
    Rename the key to match the target format.
    """
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    key = name.split(".")[-2]
    assert key in MAPPING, f"Key {key} not found in mapping: {key}"
    new_key, _ = MAPPING[key]
    name = name.replace(key, new_key)
    return name


def shard_tensor(tensor: torch.Tensor, mp_rank: int, mp_size: int, dim: int) -> torch.Tensor:
    """
    根据模型并行度对张量进行分片。
    Shard the tensor according to the model parallelism size.
    """
    assert tensor.size(dim) % mp_size == 0, f"Dimension {dim} must be divisible by {mp_size}"
    shard_size = tensor.size(dim) // mp_size
    return tensor.narrow(dim, mp_rank * shard_size, shard_size).contiguous()


def process_checkpoint(
    hf_ckpt_path: str, save_path: str, n_experts: int, mp: int
):
    """
    转换并保存模型检查点文件。
    Convert and save model checkpoint files.
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue

                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model.") :]
                name = rename_key(name)

                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue  # Skip experts not belonging to this rank
                    else:
                        key = name.split(".")[-2]
                        _, dim = MAPPING[key] # Access dim from the mapping
                        if dim is not None:
                            new_param = shard_tensor(param, i, mp, dim)
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(
            state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
        )

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    process_checkpoint(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)


if __name__ == "__main__":
    main()
```

**代码改进说明:**

1.  **常量集中管理 (Centralized Constant Management):**  将 `mapping` 字典移动到文件顶部，并重命名为 `MAPPING`，使其更易于查找和修改。这提高了代码的可读性和可维护性。

    ```python
    MAPPING = {
        "embed_tokens": ("embed", 0),
        ...
    }
    ```

2.  **函数拆分 (Function Decomposition):**  将主要逻辑拆分为更小的函数，每个函数负责一个明确的任务：
    *   `rename_key(name: str) -> str`: 负责键的重命名。
    *   `shard_tensor(tensor: torch.Tensor, mp_rank: int, mp_size: int, dim: int) -> torch.Tensor`: 负责张量的分片。
    *   `process_checkpoint(...)`:  包含主循环和文件处理逻辑。
    *   `main()`: 负责解析命令行参数并调用 `process_checkpoint`。

    这种拆分提高了代码的可读性、可测试性和可重用性。

3.  **类型提示 (Type Hints):** 添加了类型提示，提高了代码的可读性和可维护性，并允许静态类型检查工具（如 MyPy）检测潜在的错误。

4.  **详细注释 (Detailed Comments):**  添加了更详细的注释，解释了每个函数和关键代码段的目的和功能。 包括中文注释方便理解.

5.  **专家分片逻辑优化 (Expert Sharding Logic Optimization):** 更加明确地跳过不属于当前 rank 的专家，避免不必要的处理。

6.  **从 Mapping 获取 Dim (Get Dim From Mapping):** 从 `MAPPING` 字典中获取分片维度 `dim`，使代码更具可读性和可维护性。

**Demo (演示):**

假设你有以下目录结构：

```
hf_ckpt/
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── tokenizer.model
└── special_tokens_map.json
```

你需要将这些检查点转换为模型并行度为 2，总共有 8 个专家的格式：

```bash
python your_script.py --hf-ckpt-path hf_ckpt --save-path converted_ckpt --n-experts 8 --model-parallel 2
```

该脚本将会在 `converted_ckpt` 目录中生成以下文件：

```
converted_ckpt/
├── model0-mp2.safetensors
├── model1-mp2.safetensors
├── tokenizer.model
└── special_tokens_map.json
```

`model0-mp2.safetensors` 和 `model1-mp2.safetensors` 分别包含模型并行度为 2 的模型的不同分片。

**总结 (Summary):**

这段代码通过更清晰的结构、更详细的注释和类型提示，提高了可读性和可维护性。通过拆分函数，使得代码更易于测试和重用。 针对专家和张量的分片逻辑也进行了优化。中文注释帮助理解代码的意图。
