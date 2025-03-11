Lan: `py` From`dl/DeepSeek-V3\inference\fp8_cast_bf16.py`

Readed
---

该代码作用：
* 读取权重精度( Accuracy )为 fp8 的 model 转化为 fp16，并对相关的 .json 文件 model list 进行处理

---

```python
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant # 假设kernel.py文件存在，并包含weight_dequant函数
```

**1. 引入必要的库 (Import Libraries)**

```python
# 引入所需的 Python 库。
# - os: 用于与操作系统交互，例如创建目录。
# - json: 用于处理 JSON 数据，例如读取和写入模型索引文件。
# - argparse: 用于解析命令行参数。
# - glob: 用于查找与特定模式匹配的文件。
# - tqdm: 用于显示进度条。
# - torch: PyTorch 深度学习框架。
# - safetensors.torch: 用于安全地加载和保存 PyTorch 张量。
# - kernel.weight_dequant: (假设) 包含 FP8 反量化函数的自定义模块。
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant # 假设kernel.py文件存在，并包含weight_dequant函数

```

**描述:** 这段代码导入了必要的 Python 库，用于文件操作、数据处理、命令行参数解析、深度学习和张量安全保存等任务。`kernel.weight_dequant` 假设存在一个自定义模块，用于执行 FP8 权重反量化。

**2. `main` 函数定义 (Main Function Definition)**

```python
def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16) # 设置全局默认数据类型为bfloat16
    os.makedirs(bf16_path, exist_ok=True) # 创建保存BF16权重的目录，如果目录存在则不报错
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json") # 构建模型索引文件的路径
    with open(model_index_file, "r") as f: # 打开模型索引文件
        model_index = json.load(f) # 从JSON文件中加载模型索引数据
    weight_map = model_index["weight_map"] # 获取权重映射关系，用于查找权重对应的文件名

    # Cache for loaded safetensor files
    loaded_files = {} # 用于缓存已经加载的 safetensor 文件，减少重复加载
    fp8_weight_names = [] # 存储 FP8 权重的名称，用于更新模型索引

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name] # 根据权重名称从权重映射中获取对应的文件名
        if file_name not in loaded_files: # 如果文件没有被加载到缓存中
            file_path = os.path.join(fp8_path, file_name) # 构建完整的文件路径
            loaded_files[file_name] = load_file(file_path, device="cuda") # 加载 safetensor 文件到 CUDA 设备并缓存
        return loaded_files[file_name][tensor_name] # 从加载的文件中获取对应的张量
```

**描述:** `main` 函数是整个转换过程的核心。 它设置了全局默认数据类型为 `bfloat16`，创建输出目录，加载模型索引文件，并定义了一个辅助函数 `get_tensor` 用于从 safetensor 文件中获取张量。  `loaded_files` 用于缓存加载的文件，避免重复加载。`fp8_weight_names`用于记录fp8的权重名，方便后续更新`model.safetensors.index.json`文件。

**3. 权重文件迭代和转换 (Weight File Iteration and Conversion)**

```python
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors"))) # 获取 FP8 权重目录下的所有 safetensor 文件
    safetensor_files.sort() # 对文件列表进行排序
    for safetensor_file in tqdm(safetensor_files): # 遍历所有 safetensor 文件，并显示进度条
        file_name = os.path.basename(safetensor_file) # 获取文件名
        current_state_dict = load_file(safetensor_file, device="cuda") # 加载 safetensor 文件到 CUDA 设备
        loaded_files[file_name] = current_state_dict # 将加载的文件放入缓存

        new_state_dict = {} # 用于存储转换后的权重
        for weight_name, weight in current_state_dict.items(): # 遍历当前文件中的所有权重
            if weight_name.endswith("_scale_inv"): # 如果是 scale_inv 权重，则跳过
                continue
            elif weight.element_size() == 1:  # FP8 weight # 如果权重的数据类型是 fp8 (element_size 为 1)
                scale_inv_name = f"{weight_name}_scale_inv" # 构建对应的 scale_inv 权重名称
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name) # 从缓存或文件中获取 scale_inv 张量
                    fp8_weight_names.append(weight_name) # 将 FP8 权重的名称添加到列表中
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv) # 对 FP8 权重进行反量化
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion") # 如果找不到对应的 scale_inv 张量，则打印警告信息并跳过
                    new_state_dict[weight_name] = weight # 保留原始 FP8 权重
            else:
                new_state_dict[weight_name] = weight # 如果不是 FP8 权重，则直接复制

        new_safetensor_file = os.path.join(bf16_path, file_name) # 构建新的 safetensor 文件路径
        save_file(new_state_dict, new_safetensor_file) # 将转换后的权重保存到新的 safetensor 文件

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2: # 如果缓存的文件数量超过 2 个
            oldest_file = next(iter(loaded_files)) # 获取最旧的文件名
            del loaded_files[oldest_file] # 从缓存中删除最旧的文件
            torch.cuda.empty_cache() # 清空 CUDA 缓存
```

**描述:** 此部分代码遍历 FP8 权重目录中的所有 safetensor 文件，并将 FP8 权重转换为 BF16。  它使用 `weight_dequant` 函数执行反量化，并处理缺少 `scale_inv` 张量的情况。  为了节省内存，代码只缓存最近使用的两个文件，并定期清理 CUDA 缓存。

**4. 更新模型索引 (Update Model Index)**

```python
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json") # 构建新的模型索引文件路径
    for weight_name in fp8_weight_names: # 遍历所有 FP8 权重的名称
        scale_inv_name = f"{weight_name}_scale_inv" # 构建对应的 scale_inv 权重名称
        if scale_inv_name in weight_map: # 如果 scale_inv 权重存在于权重映射中
            weight_map.pop(scale_inv_name) # 从权重映射中删除 scale_inv 权重
    with open(new_model_index_file, "w") as f: # 打开新的模型索引文件
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2) # 将更新后的权重映射保存到新的模型索引文件
```

**描述:** 此部分代码更新模型索引文件，删除对 `scale_inv` 张量的引用。 这是因为在转换后，这些 `scale_inv` 张量不再需要，并且不应包含在 BF16 模型中。

**5. 命令行参数解析 (Command-Line Argument Parsing)**

```python
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Path to the FP8 weights directory.") # 添加输入 FP8 权重目录的命令行参数
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="Path to the BF16 weights directory.") # 添加输出 BF16 权重目录的命令行参数
    args = parser.parse_args() # 解析命令行参数
    main(args.input_fp8_hf_path, args.output_bf16_hf_path) # 调用 main 函数执行转换
```

**描述:** 此部分代码定义了命令行参数，允许用户指定输入 FP8 权重目录和输出 BF16 权重目录。 它使用 `argparse` 库解析这些参数，并将它们传递给 `main` 函数。

**如何使用:**

1.  **安装必要的库:**

    ```bash
    pip install torch safetensors tqdm
    ```

2.  **准备 FP8 权重:**  你需要有一个包含 FP8 权重和 `model.safetensors.index.json` 文件的目录。 这些权重应该以 safetensor 格式存储。
3.  **运行脚本:**

    ```bash
    python your_script_name.py --input-fp8-hf-path /path/to/fp8/weights --output-bf16-hf-path /path/to/bf16/weights
    ```

    将 `/path/to/fp8/weights` 替换为你的 FP8 权重目录的实际路径，并将 `/path/to/bf16/weights` 替换为你想要保存 BF16 权重的目录的路径。

**Simple Demo (简单演示):**

1.  **创建虚拟的 FP8 权重目录和文件:**

    (为了演示，我们将创建一些虚拟文件和数据。在实际使用中，你需要替换成你自己的FP8权重)

    ```python
    import os
    import json
    import torch
    from safetensors.torch import save_file

    # 创建目录
    fp8_path = "dummy_fp8_weights"
    os.makedirs(fp8_path, exist_ok=True)

    # 创建虚拟权重数据
    weight_1 = torch.randn(10, 10, dtype=torch.uint8) # 模拟 FP8 权重 (注意类型是uint8)
    scale_inv_1 = torch.randn(1, dtype=torch.bfloat16)
    weight_2 = torch.randn(5, 5, dtype=torch.bfloat16)

    # 保存到 safetensors 文件
    save_file({"weight_1": weight_1, "weight_1_scale_inv":scale_inv_1, "weight_2": weight_2}, os.path.join(fp8_path, "model.safetensors"))

    # 创建 model.safetensors.index.json 文件
    index_data = {
        "metadata": {},
        "weight_map": {
            "weight_1": "model.safetensors",
            "weight_1_scale_inv": "model.safetensors",
            "weight_2": "model.safetensors",
        },
    }
    with open(os.path.join(fp8_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)
    ```

2.  **运行转换脚本:** (假设你的脚本名为 `convert_fp8_to_bf16.py`)

    ```bash
    python convert_fp8_to_bf16.py --input-fp8-hf-path dummy_fp8_weights --output-bf16-hf-path dummy_bf16_weights
    ```

3.  **检查输出目录:**  你会看到一个名为 `dummy_bf16_weights` 的目录，其中包含转换后的 BF16 权重和更新后的 `model.safetensors.index.json` 文件。

**注意:**

*   `kernel.weight_dequant`  需要根据你的实际 FP8 量化方案来实现。 上面的代码只是提供了一个框架。
*   这个demo创建的是假的FP8权重，因为真正的FP8权重需要特定的硬件支持才能生成。所以，实际运行的时候，`weight_dequant`函数可能会报错。

这个详细的解释和演示应该能帮助你理解代码的各个部分以及如何使用它。希望对你有帮助！
