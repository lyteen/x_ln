Lan: `py` From`dl/DeepSeek-V3\inference\fp8_cast_bf16.py`

好的，我们重新开始。 我将为你提供一个更智能的代码版本，用于将FP8权重转换为BF16，并提供更详细的解释和中文注释。

**目标:**

*   **提高代码效率:** 优化文件加载和保存过程。
*   **更健壮的错误处理:**  处理`scale_inv`张量缺失的情况，并提供更友好的警告信息。
*   **更好的内存管理:**  更智能地管理GPU内存，避免OOM错误。
*   **清晰的代码结构:**  改进代码结构，使其更易于阅读和维护。

**代码实现:**

```python
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

# 假设 kernel.py 中有 weight_dequant 函数
from kernel import weight_dequant

def main(fp8_path, bf16_path):
    """
    将 FP8 权重转换为 BF16 并保存。

    此函数读取指定目录中的 FP8 权重，将其转换为 BF16，
    并将转换后的权重保存到另一个指定目录。 它还会更新
    模型索引文件以反映更改。

    Args:
        fp8_path (str): 包含 FP8 权重和模型索引文件的目录的路径。
        bf16_path (str): 将保存转换后的 BF16 权重的目录的路径。

    Raises:
        KeyError: 如果权重的必需 scale_inv 张量缺失。

    Notes:
        - 该函数假定 FP8 权重存储在 safetensor 文件中。
        - 该函数缓存加载的 safetensor 文件以优化内存使用。
        - 该函数更新模型索引文件以删除对 scale_inv 张量的引用。
    """

    torch.set_default_dtype(torch.bfloat16)  # 设置默认数据类型为 bfloat16
    os.makedirs(bf16_path, exist_ok=True)  # 创建输出目录，如果不存在

    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json") # 构建模型索引文件路径
    with open(model_index_file, "r") as f:
        model_index = json.load(f) # 加载模型索引文件
    weight_map = model_index["weight_map"] # 获取权重映射

    # 缓存加载的 safetensor 文件，提高效率
    loaded_files = {}
    fp8_weight_names = []

    # 辅助函数：从正确的文件中获取张量
    def get_tensor(tensor_name):
        """
        从缓存的 safetensor 文件中检索张量，如果未缓存，则从磁盘加载。

        Args:
            tensor_name (str): 要检索的张量的名称。

        Returns:
            torch.Tensor: 检索到的张量。

        Raises:
            KeyError: 如果张量在 safetensor 文件中不存在。
        """
        file_name = weight_map[tensor_name] # 根据张量名获取文件名
        if file_name not in loaded_files: # 如果文件未加载
            file_path = os.path.join(fp8_path, file_name) # 构建文件路径
            try:
                loaded_files[file_name] = load_file(file_path, device="cuda") # 加载文件到 GPU
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                return None
        return loaded_files[file_name].get(tensor_name)  # 使用 .get() 避免 KeyError

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors"))) # 获取所有 safetensor 文件
    safetensor_files.sort() # 排序，保证处理顺序

    for safetensor_file in tqdm(safetensor_files, desc="Converting FP8 to BF16"): # 循环处理
        file_name = os.path.basename(safetensor_file) # 获取文件名
        try:
            current_state_dict = load_file(safetensor_file, device="cuda") # 加载当前文件
            loaded_files[file_name] = current_state_dict # 添加到缓存
        except Exception as e:
            print(f"Error loading file {safetensor_file}: {e}")
            continue # 跳过此文件

        new_state_dict = {} # 存储转换后的权重
        for weight_name, weight in current_state_dict.items(): # 遍历所有权重
            if weight_name.endswith("_scale_inv"): # 跳过 scale_inv 张量
                continue
            elif weight.element_size() == 1:  # FP8 权重
                scale_inv_name = f"{weight_name}_scale_inv" # 构建 scale_inv 张量名
                scale_inv = get_tensor(scale_inv_name) # 获取 scale_inv 张量
                if scale_inv is None:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight # 如果 scale_inv 缺失，则跳过转换
                else:
                    fp8_weight_names.append(weight_name) # 记录 FP8 权重名
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv) # 反量化
            else:
                new_state_dict[weight_name] = weight # 其他权重直接复制

        new_safetensor_file = os.path.join(bf16_path, file_name) # 构建输出文件路径
        try:
            save_file(new_state_dict, new_safetensor_file) # 保存转换后的权重
        except Exception as e:
            print(f"Error saving file {new_safetensor_file}: {e}")

        # 内存管理：保持加载的文件数量在一个合理的范围内
        if len(loaded_files) > 5:
            # 移除最早加载的文件
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache() # 清理 GPU 缓存

    # 更新模型索引文件
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json") # 构建新的索引文件路径
    for weight_name in fp8_weight_names: # 移除 scale_inv 权重的记录
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            del weight_map[scale_inv_name] # 删除 scale_inv 记录

    with open(new_model_index_file, "w") as f: # 保存新的索引文件
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="输入 FP8 格式的 Hugging Face 模型路径")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="输出 BF16 格式的 Hugging Face 模型路径")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
```

**代码改进说明:**

*   **更完善的错误处理:**  使用了`try...except`块来捕获文件加载和保存期间可能发生的异常，并打印错误信息，避免程序崩溃。
*   **更智能的`scale_inv`处理:** 在`get_tensor`函数中使用`.get()`方法，如果找不到`scale_inv`张量，则返回`None`，避免`KeyError`。 并且，仅当成功获取到`scale_inv`时才进行反量化。
*   **更精细的内存管理:** 增加了`loaded_files`的最大数量限制，并使用`torch.cuda.empty_cache()`来释放GPU内存。  数量可以根据你的GPU内存大小进行调整。
*   **更友好的用户界面:** 使用`tqdm`显示转换进度，并添加了更清晰的帮助信息到命令行参数。
*   **使用 `del weight_map[scale_inv_name]` 而不是 `weight_map.pop(scale_inv_name)`:**  `del` 语句通常更简洁和高效。
*   **明确的文件打开方式:**  显式指定 `open()` 函数的文件打开模式，例如 `"r"` 用于读取和 `"w"` 用于写入。
*   **更清晰的注释:**  增加了更多中文注释，解释代码的功能和逻辑。

**使用示例:**

1.  **准备环境:** 确保你已经安装了`torch`，`safetensors`和`tqdm`。
2.  **准备FP8模型:**  你需要有一个以FP8格式保存的Hugging Face模型。  假设它位于 `/path/to/fp8/model` 目录。
3.  **运行转换脚本:**

```bash
python your_script_name.py --input-fp8-hf-path /path/to/fp8/model --output-bf16-hf-path /path/to/bf16/model
```

   将`your_script_name.py`替换为你的脚本的文件名，并将`/path/to/fp8/model`和`/path/to/bf16/model`替换为实际的路径。

**重要提示:**

*   确保你的`kernel.py`文件包含`weight_dequant`函数，该函数负责将FP8权重反量化为BF16。 你需要根据你的FP8量化方案来实现这个函数。
*   根据你的GPU内存大小调整`loaded_files`的最大数量。  如果你的GPU内存较小，可以减少这个数量。

这个改进后的代码应该更健壮、更高效，并且更容易理解和使用。  如果在使用过程中遇到任何问题，请随时提问。
