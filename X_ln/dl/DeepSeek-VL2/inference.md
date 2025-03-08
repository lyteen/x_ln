Lan: `py` From`X_ln\dl/DeepSeek-VL2\inference.html`

```python
# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox

# 加载PIL图像
def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """
    从对话列表中加载PIL图像。

    Args:
        conversations (List[Dict[str, str]]): 包含消息列表的对话。 示例:
            [
                {
                    "role": "User",
                    "content": "<image>\n提取此图像中的所有信息并将其转换为markdown格式。",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): PIL图像列表。

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images

# Demo Usage 演示用法
if __name__ == '__main__':
    # 示例对话
    example_conversation = [
        {
            "role": "User",
            "content": "<image>\n请描述这张图片。",
            "images": ["images/incontext_visual_grounding_1.jpeg"]  # 替换为实际图像路径
        },
        {"role": "Assistant", "content": ""},
    ]
    images = load_pil_images(example_conversation)
    print(f"已加载 {len(images)} 张图像.")

```

**解释:**
*   **`load_pil_images(conversations)` 函数:**  这个函数接收一个对话列表作为输入。对话列表中，每个元素是字典，字典中包括角色(`role`)，内容(`content`)和图片(`images`)。该函数从对话的图片路径中加载图像。使用 `PIL.Image.open()` 打开图像，并使用 `convert("RGB")` 将图像转换为 RGB 格式。 最后，返回一个包含所有加载的PIL图像的列表。
*   **用处:**  这个函数用于预处理输入数据，将图像文件加载到内存中，以便后续模型可以使用。

```python
# 主函数
def main(args):

    dtype = torch.bfloat16  # 设置数据类型为bfloat16

    # 指定模型路径
    model_path = args.model_path
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path) # 加载processor
    tokenizer = vl_chat_processor.tokenizer # 获取tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype  # 使用指定的dtype
    )
    vl_gpt = vl_gpt.cuda().eval() # 将模型加载到CUDA设备并设置为评估模式

    # 单个图像对话示例
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<image>\n<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image.",
            "images": [
                "images/incontext_visual_grounding_1.jpeg",  # 替换为实际图像路径
                "images/icl_vg_2.jpeg" # 替换为实际图像路径
            ],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 加载图像并准备输入
    pil_images = load_pil_images(conversation)
    print(f"len(pil_images) = {len(pil_images)}")

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True, # 强制批处理
        system_prompt="" # 设置系统提示为空
    ).to(vl_gpt.device, dtype=dtype) # 将输入数据移动到模型所在的设备，并转换为指定的dtype

    with torch.no_grad(): # 禁用梯度计算

        if args.chunk_size == -1:
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs) # 准备输入嵌入
            past_key_values = None # 初始化past_key_values
        else:
            # 使用40G GPU进行 vl2-small 的增量预填充
            inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                input_ids=prepare_inputs.input_ids, # 输入ID
                images=prepare_inputs.images, # 图像
                images_seq_mask=prepare_inputs.images_seq_mask, # 图像序列掩码
                images_spatial_crop=prepare_inputs.images_spatial_crop, # 图像空间裁剪
                attention_mask=prepare_inputs.attention_mask, # 注意力掩码
                chunk_size=args.chunk_size # 块大小
            )

        # 运行模型以获得响应
        outputs = vl_gpt.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id, # 填充token ID
            bos_token_id=tokenizer.bos_token_id, # 开始token ID
            eos_token_id=tokenizer.eos_token_id, # 结束token ID
            max_new_tokens=512, # 最大新token数

            do_sample=True, # 启用采样
            temperature=0.4, # 温度
            top_p=0.9, # top_p
            repetition_penalty=1.1, # 重复惩罚

            use_cache=True, # 使用缓存
        )

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False) # 解码答案
        print(f"{prepare_inputs['sft_format'][0]}", answer)

        vg_image = parse_ref_bbox(answer, image=pil_images[-1]) # 解析参考边界框
        if vg_image is not None:
            vg_image.save("./vg.jpg", format="JPEG", quality=85) # 保存图像

# Demo Usage 演示用法
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        default="deepseek-ai/deepseek-vl2",
                        help="model name or local path to the model")
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()
    main(args)
```

**解释:**
*   **`main(args)` 函数:** 这是程序的主要执行函数。
    *   它首先设置数据类型为 `torch.bfloat16`。
    *   然后，它使用 `DeepseekVLV2Processor.from_pretrained()` 加载预训练的 processor，并获取 tokenizer。  Processor 负责将原始输入（文本和图像）转换为模型可以理解的格式。
    *   使用 `AutoModelForCausalLM.from_pretrained()` 加载 `DeepseekVLV2ForCausalLM` 模型。  `trust_remote_code=True` 允许加载来自远程仓库的自定义代码。  `.cuda()` 将模型加载到 GPU 上，`.eval()` 将模型设置为评估模式，禁用 dropout 等训练时使用的层。
    *   定义一个对话列表 `conversation`，其中包含用户的问题和图像路径。
    *   调用 `load_pil_images()` 函数加载图像。
    *   使用 `vl_chat_processor()` 预处理对话和图像，准备模型的输入。
    *   使用 `torch.no_grad()` 上下文管理器禁用梯度计算，以减少内存使用并加速推理。
    *   根据 `args.chunk_size` 的值，选择使用 `vl_gpt.prepare_inputs_embeds()` 或 `vl_gpt.incremental_prefilling()` 来准备输入嵌入。`incremental_prefilling()` 可以在GPU内存有限的情况下，分块处理输入，减少内存占用。
    *   调用 `vl_gpt.generate()` 函数生成模型的输出。  这个函数使用模型进行推理，生成文本响应。
    *   使用 `tokenizer.decode()` 解码模型的输出，将 token ID 转换为可读的文本。
    *   调用 `parse_ref_bbox()` 函数解析模型输出中的边界框信息，并在图像上绘制边界框。
    *   如果检测到边界框，则将带有边界框的图像保存到文件中。

*   **用处:**  这个函数实现了 DeepSeek VL-2 模型的推理过程，包括加载模型、预处理输入、生成输出和后处理输出。

**如何使用:**
1.  确保你已经安装了所有必要的依赖项，例如 `torch`、`transformers` 和 `PIL`。
2.  下载 DeepSeek VL-2 模型的权重文件并将其保存在本地目录中。
3.  修改 `conversation` 变量中的图像路径，指向你自己的图像文件。
4.  运行脚本，指定模型路径和块大小：

```bash
python your_script_name.py --model_path /path/to/your/model --chunk_size -1
```

如果你的GPU内存不够，可以尝试调整`chunk_size`的值为一个较小的整数。

这段代码展示了如何使用 DeepSeek VL-2 模型进行图像相关的任务，例如视觉定位。

