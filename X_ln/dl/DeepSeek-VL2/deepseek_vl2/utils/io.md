Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\utils\io.html`

好的，我们重新开始，这次我会更详细地解释代码的各个部分，并提供更清晰的用例说明，以及使用中文进行解释。

**1. `load_pretrained_model` 函数:**

```python
import torch
from transformers import AutoModelForCausalLM


def load_pretrained_model(model_path: str):
    """
    加载预训练的 DeepSeek-VL-V2 模型。

    Args:
        model_path (str): 预训练模型的路径。

    Returns:
        tuple: 包含 tokenizer, vl_chat_processor, vl_gpt 的元组。
               - tokenizer: 用于文本编码和解码的分词器。
               - vl_chat_processor: DeepSeekVLV2Processor 对象，用于处理图像和文本输入。
               - vl_gpt: DeepSeekVLV2ForCausalLM 对象，即加载的预训练模型。
    """

    from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

    # 从预训练模型路径加载 DeepseekVLV2Processor
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer  # 获取分词器

    # 从预训练模型路径加载 DeepseekVLV2ForCausalLM 模型
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True  # 允许加载远程代码
    )

    # 将模型移动到 CUDA 设备，并转换为 bfloat16 数据类型，设置为评估模式
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return tokenizer, vl_chat_processor, vl_gpt


# 示例用法
if __name__ == '__main__':
    # 假设你的模型路径是 'path/to/your/model'
    model_path = 'path/to/your/model'  # 替换为你的实际模型路径
    try:
        tokenizer, vl_chat_processor, vl_gpt = load_pretrained_model(model_path)
        print("模型加载成功！")
        print(f"Tokenizer: {type(tokenizer)}")
        print(f"VL Chat Processor: {type(vl_chat_processor)}")
        print(f"VL GPT Model: {type(vl_gpt)}")

        # 在这里可以使用加载的模型进行推理
        # 例如：使用 tokenizer 对文本进行编码，然后将编码后的文本和图像输入到 vl_gpt 模型中
    except Exception as e:
        print(f"模型加载失败：{e}")
```

**描述:**

*   这个函数的主要作用是从指定的 `model_path` 加载 DeepSeek-VL-V2 预训练模型。
*   它使用 `DeepseekVLV2Processor` 来处理图像和文本输入，并从中获取 `tokenizer` 用于文本分词。
*   使用 `AutoModelForCausalLM.from_pretrained` 加载模型，并将其移动到 GPU (如果可用) 并设置为评估模式。 `trust_remote_code=True` 允许加载模型中可能包含的自定义代码。
*   返回 `tokenizer`，`vl_chat_processor`和加载的 `vl_gpt` 模型。

**使用方法和简单演示:**

1.  **替换模型路径:** 将 `model_path = 'path/to/your/model'` 替换为你实际的模型路径。
2.  **运行代码:** 运行此代码段将尝试加载模型，并在加载成功后打印加载的组件类型。
3.  **推理:** 在 `# 在这里可以使用加载的模型进行推理` 注释下方，你可以添加代码来使用加载的模型进行推理。例如，你可以使用 `tokenizer` 对文本进行编码，然后将编码后的文本和图像输入到 `vl_gpt` 模型中。

**2. `load_pil_images` 函数:**

```python
import PIL.Image
from typing import Dict, List


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """
    从对话列表中加载 PIL 图像。

    Args:
        conversations (List[Dict[str, str]]): 包含消息列表的对话，消息中可能包含图像路径。
            示例:
            [
                {
                    "role": "User",
                    "content": "<image>\n提取此图像中的所有信息并将其转换为 markdown 格式。",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        List[PIL.Image.Image]: PIL 图像列表。
    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue  # 如果消息中没有图像，则跳过

        for image_path in message["images"]:
            try:
                pil_img = PIL.Image.open(image_path)  # 打开图像
                pil_img = pil_img.convert("RGB")  # 转换为 RGB 格式
                pil_images.append(pil_img)  # 添加到图像列表
            except FileNotFoundError:
                print(f"警告：找不到图像文件 {image_path}")
            except Exception as e:
                print(f"加载图像 {image_path} 时出错：{e}")

    return pil_images


# 示例用法
if __name__ == '__main__':
    conversations = [
        {
            "role": "User",
            "content": "<image>\n提取此图像中的所有信息并将其转换为 markdown 格式。",
            "images": ["./examples/table_datasets.png", "./examples/another_image.jpg"]  # 替换为实际图像路径
        },
        {"role": "Assistant", "content": ""},
    ]

    try:
        pil_images = load_pil_images(conversations)
        print(f"成功加载 {len(pil_images)} 张图像。")
        for img in pil_images:
            print(f"图像格式：{img.format}, 图像大小：{img.size}")
            # 在这里可以使用加载的图像
            # 例如：显示图像，或者将其传递给模型进行处理
    except Exception as e:
        print(f"加载图像时出错：{e}")
```

**描述:**

*   此函数接受一个对话列表，其中每个对话条目可能包含一个 `images` 键，其值为图像路径的列表。
*   它遍历对话列表，打开每个图像路径，将其转换为 RGB 格式，并将其添加到 `pil_images` 列表中。
*   如果找不到图像文件，则打印警告消息。
*   返回包含所有成功加载的 PIL 图像的列表。

**使用方法和简单演示:**

1.  **替换图像路径:** 将示例对话中的图像路径（例如 `"./examples/table_datasets.png"`）替换为你实际的图像路径。
2.  **运行代码:** 运行此代码段将尝试加载图像，并在加载成功后打印加载的图像数量和格式。
3.  **使用图像:** 在 `# 在这里可以使用加载的图像` 注释下方，你可以添加代码来使用加载的图像。 例如，你可以使用 `img.show()` 显示图像，或者将其传递给模型进行处理。

**3. `load_json` 函数:**

```python
import json

def load_json(filepath):
    """
    从文件中加载 JSON 数据。

    Args:
        filepath (str): JSON 文件的路径。

    Returns:
        dict: 从文件中加载的 JSON 数据。
    """
    try:
        with open(filepath, "r", encoding='utf-8') as f:  # 显式指定编码为 UTF-8
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解码失败 {filepath}：{e}")
        return None
    except Exception as e:
        print(f"加载 JSON 文件 {filepath} 时出错：{e}")
        return None


# 示例用法
if __name__ == '__main__':
    filepath = "data.json"  # 替换为你的 JSON 文件路径

    # 创建一个虚拟的 JSON 文件
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump({"name": "示例数据", "value": 123}, f, ensure_ascii=False, indent=4)  # 确保写入非 ASCII 字符

    try:
        data = load_json(filepath)
        if data:
            print("JSON 数据加载成功！")
            print(f"数据类型：{type(data)}")
            print(f"数据内容：{data}")
            # 在这里可以使用加载的 JSON 数据
            # 例如：访问数据中的特定字段
            name = data.get("name")
            value = data.get("value")
            print(f"Name: {name}, Value: {value}")
    except Exception as e:
        print(f"加载 JSON 文件时出错：{e}")
```

**描述:**

*   这个函数接受一个文件路径作为输入，并尝试从该文件中加载 JSON 数据。
*   它使用 `json.load()` 函数将 JSON 数据从文件加载到 Python 字典中。
*   如果文件不存在，或者 JSON 解码失败，则打印错误消息并返回 `None`。

**使用方法和简单演示:**

1.  **替换文件路径:** 将 `filepath = "data.json"` 替换为你实际的 JSON 文件路径。
2.  **创建虚拟 JSON 文件 (可选):**  示例代码包含创建虚拟 `data.json` 文件的部分，以便你可以直接运行代码进行测试。  `ensure_ascii=False` 用于确保可以正确写入非 ASCII 字符， `indent=4` 使得json文件更易读。
3.  **运行代码:** 运行此代码段将尝试加载 JSON 数据，并在加载成功后打印数据的类型和内容。
4.  **使用数据:** 在 `# 在这里可以使用加载的 JSON 数据` 注释下方，你可以添加代码来使用加载的 JSON 数据。 例如，你可以访问数据中的特定字段，或者将其传递给其他函数进行处理。

希望这些详细的解释和示例能够帮助你理解这些代码的作用和使用方法。 如果你有任何其他问题，请随时提出。
