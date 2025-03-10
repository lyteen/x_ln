Lan: `py` From`dl/DeepSeek-V3\inference\convert.py`


**1. 导入库 (Import Libraries):**

```python
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file
```

**描述:** 这段代码导入了必要的 Python 库.
*   `os`: 用于处理文件和目录操作。
*   `shutil`: 用于高级文件操作，例如复制文件。
*   `argparse`: 用于解析命令行参数。
*   `glob`: 用于根据模式匹配文件路径。
*   `tqdm`: 用于显示进度条。
*   `torch`: PyTorch 深度学习框架。
*   `safetensors.torch`: 用于安全地加载和保存张量。

**如何使用:** 这些库在使用这段代码之前需要安装.  例如, 可以使用`pip install torch safetensors tqdm`安装 `torch`、`safetensors` 和 `tqdm`.

**2. 映射字典 (Mapping Dictionary):**

```python
mapping = {
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
```

**描述:** `mapping` 字典定义了模型参数名称的映射关系。 这用于将 Hugging Face 模型中的参数名称转换为目标格式的名称。  元组中的第二个元素指示分片维度（如果适用）。 `None` 表示不分片。

**如何使用:** 在 `main` 函数中，此字典用于重命名和分片模型参数。  例如，`"q_proj"` 参数将被重命名为 `"wq"`，并且沿维度 0 进行分片。

**3. 主函数 (Main Function):**

```python
def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
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
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)
```

**描述:** `main` 函数执行以下操作：

*   加载 Hugging Face 格式的检查点文件 (`.safetensors`)。
*   根据 `mapping` 字典重命名参数。
*   根据模型并行度 (`mp`) 对参数进行分片。
*   将分片的参数保存到单独的文件中。
*   复制 tokenizer 文件。

**代码解释：**

*   `torch.set_num_threads(8)`: 设置 PyTorch 使用的线程数。
*   `n_local_experts = n_experts // mp`: 计算每个模型并行进程上的专家数量。
*   `state_dicts = [{} for _ in range(mp)]`:  创建一个列表，其中包含用于存储每个模型并行进程的状态字典的空字典。
*   `for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors")))`: 循环遍历 Hugging Face 检查点路径中的所有 `.safetensors` 文件。`tqdm` 用于显示进度条。
*   `with safe_open(file_path, framework="pt", device="cpu") as f:`: 使用 `safe_open` 函数以安全的方式打开检查点文件。`framework="pt"` 指定文件包含 PyTorch 张量。`device="cpu"` 指定张量加载到 CPU 上。
*   `for name in f.keys():`: 循环遍历检查点文件中的所有参数名称。
*   `if "model.layers.61" in name: continue`: 跳过名称中包含 `"model.layers.61"` 的参数。
*   `param: torch.Tensor = f.get_tensor(name)`:  从检查点文件获取指定名称的张量。
*   `if name.startswith("model."): name = name[len("model."):]`: 如果参数名称以 `"model."` 开头，则删除此前缀。
*   `name = name.replace("self_attn", "attn") ... name = name.replace("e_score_correction_bias", "bias")`:  替换参数名称中的某些字符串，例如将 `"self_attn"` 替换为 `"attn"`。
*   `key = name.split(".")[-2]`: 从参数名称中提取键。
*   `assert key in mapping, f"Key {key} not found in mapping"`: 确保键存在于 `mapping` 字典中。
*   `new_key, dim = mapping[key]`: 从 `mapping` 字典中获取新的键和维度。
*   `name = name.replace(key, new_key)`: 将参数名称中的键替换为新的键。
*   `for i in range(mp):`:  循环遍历模型并行进程。
*   `new_param = param`: 创建参数的副本。
*   `if "experts" in name and "shared_experts" not in name:`: 检查参数是否与专家相关。
*   `idx = int(name.split(".")[-3])`: 从参数名称中提取专家的索引。
*   `if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts: continue`:  如果专家不在当前模型并行进程上，则跳过该专家。
*   `elif dim is not None:`: 检查维度是否不为 `None`。
*   `assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"`: 确保指定维度可以被模型并行度整除。
*   `shard_size = param.size(dim) // mp`: 计算分片大小。
*   `new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()`:  沿指定维度对参数进行分片。`.contiguous()` 创建张量的连续副本。
*   `state_dicts[i][name] = new_param`: 将分片的参数添加到当前模型并行进程的状态字典中。
*   `os.makedirs(save_path, exist_ok=True)`: 创建保存路径目录，如果目录已存在，则不引发错误。
*   `for i in trange(mp):`:  循环遍历模型并行进程。`trange` 用于显示进度条。
*   `save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))`: 将当前模型并行进程的状态字典保存到 `.safetensors` 文件。
*   `for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):`: 循环遍历 Hugging Face 检查点路径中所有名称中包含 `"token"` 的文件。
*   `new_file_path = os.path.join(save_path, os.path.basename(file_path))`:  创建新文件路径。
*   `shutil.copyfile(file_path, new_file_path)`: 将文件复制到新文件路径。

**如何使用:**  调用 `main` 函数，传入 Hugging Face 检查点路径、保存路径、专家数量和模型并行度。

**4. 主程序 (Main Execution Block):**

```python
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
```

**描述:**  这段代码定义了主程序入口点。它使用 `argparse` 来解析命令行参数，并调用 `main` 函数来执行检查点转换和分片。 它还执行一个断言，以确保专家数量可以被模型并行度整除。

**如何使用:**  运行脚本，传入以下命令行参数：

*   `--hf-ckpt-path`:  Hugging Face 检查点文件的路径。
*   `--save-path`:  保存转换后的检查点文件的路径。
*   `--n-experts`:  模型中的专家数量。
*   `--model-parallel`:  模型并行度。

**示例:**

```bash
python your_script_name.py --hf-ckpt-path /path/to/hf/checkpoint --save-path /path/to/save/checkpoint --n-experts 16 --model-parallel 2
```

这段命令将从 `/path/to/hf/checkpoint` 读取 Hugging Face 检查点，将模型参数分成 2 份（`--model-parallel 2`），假设模型有 16 个专家，并将转换后的分片检查点保存到 `/path/to/save/checkpoint`。

**总结:**

这段代码用于将 Hugging Face 格式的模型检查点转换为可以与特定模型并行策略一起使用的格式。  它通过重命名参数、分片参数并将分片保存到单独的文件来实现这一点。 它是一个相当定制化的转换脚本，依赖于 `mapping` 字典来正确转换参数名称。 适用于转换 Mixture-of-Experts 模型权重.
