Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\serve\inference.html`

好的，现在我重新开始，将对代码进行分解，并附上中文解释和使用示例。

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

from threading import Thread
from typing import List

import torch
import transformers
from joblib.externals.cloudpickle import instance
from transformers import (
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.models.conversation import Conversation

```

**描述:** 这部分代码是导入必要的Python库和模块。 这些库包括 `threading` 用于多线程，`typing` 用于类型提示，`torch` 用于深度学习，`transformers` 用于使用预训练模型，`joblib` 用于序列化对象，以及从`deepseek_vl2` 导入特定的模型类和对话类。

**用途:** 这些导入语句允许代码使用这些库和模块中定义的函数和类。 例如，`torch` 提供了张量操作和神经网络构建块，而 `transformers` 提供了访问和使用预训练语言模型的工具。

```python
def load_model(model_path, dtype=torch.bfloat16):
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    )
    vl_gpt = vl_gpt.cuda().eval()
    return tokenizer, vl_gpt, vl_chat_processor
```

**描述:** `load_model` 函数负责加载预训练的 DeepSeek VL-V2 模型。 它使用 `DeepseekVLV2Processor` 加载处理器，然后使用 `AutoModelForCausalLM` 加载模型本身。  该模型随后被移动到 CUDA 设备（如果可用），并设置为评估模式。

**用途:** 此函数是初始化模型的关键步骤。 加载的模型将用于后续的生成和推理。

**演示:**

```python
# 假设 model_path 指向你的模型文件夹
# model_path = "/path/to/your/deepseek-vl2-model"
# tokenizer, vl_gpt, vl_chat_processor = load_model(model_path)
# print("模型加载完成！")
```

这个例子演示了如何使用 `load_model` 函数。需要提供模型路径。 加载后，它将打印一条消息。 请注意，你需要将 `/path/to/your/deepseek-vl2-model` 替换为模型实际所在的路径。

```python
def convert_conversation_to_prompts(conversation: Conversation):
    conv_prompts = []

    last_image = None

    messages = conversation.messages
    for i in range(0, len(messages), 2):

        if isinstance(messages[i][1], tuple):
            text, images = messages[i][1]
            last_image = images[-1]
        else:
            text, images = messages[i][1], []

        prompt = {
            "role": messages[i][0],
            "content": text,
            "images": images
        }
        response = {"role": messages[i + 1][0], "content": messages[i + 1][1]}
        conv_prompts.extend([prompt, response])

    return conv_prompts, last_image
```

**描述:** `convert_conversation_to_prompts` 函数将 `Conversation` 对象转换为一系列提示，模型可以用来生成响应。它遍历对话的消息，提取角色、内容和图像，并将它们格式化为模型期望的格式。

**用途:** 该函数是准备输入以进行模型推理的关键步骤。 它确保对话以模型能够理解的方式呈现。

**演示:**

```python
# 假设你有一个 Conversation 对象叫做 my_conversation
# from deepseek_vl2.models.conversation import Conversation  # 确保已导入
# my_conversation = Conversation(messages=[("user", "你好", []), ("assistant", "你好！", [])]) # 创建一个示例对话
# conv_prompts, last_image = convert_conversation_to_prompts(my_conversation)
# print("转换后的提示:", conv_prompts)
# print("最后一张图片:", last_image)
```

这个例子演示了如何使用 `convert_conversation_to_prompts` 函数。  它创建了一个虚拟 `Conversation` 对象，然后使用该函数将其转换为提示。 输出将显示转换后的提示和最后一张图片（如果存在）。 需要注意，这只是一个示例，实际的 `Conversation` 对象可能包含更复杂的消息和图像数据。

```python
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        for stop in self.stops:
            if input_ids.shape[-1] < len(stop):
                continue
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False
```

**描述:** `StoppingCriteriaSub` 类继承自 `StoppingCriteria`，用于定义生成过程的停止条件。 特别是，当生成的文本包含指定的停止词时，它会停止生成。  该类将停止词移动到 CUDA 设备（如果可用）以加快比较速度。

**用途:** 此类控制生成过程，防止模型生成无限文本或不相关的输出。

**演示:**

```python
# 假设你想要在生成文本中遇到 "EOS" 时停止
# stop_words = ["EOS"]
# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[torch.tensor(tokenizer.encode(stop_word)) for stop_word in stop_words])])
# print("停止条件已创建！")
```

这个例子演示了如何创建一个 `StoppingCriteriaSub` 对象。它定义了一个停止词 "EOS"，并使用它来创建 `StoppingCriteriaList`。  此停止条件将用于后续的文本生成过程中。 请注意，你需要一个 tokenizer 对象来编码停止词。

```python
@torch.inference_mode()
def deepseek_generate(
    conversations: list,
    vl_gpt: torch.nn.Module,
    vl_chat_processor: DeepseekVLV2Processor,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    chunk_size: int = -1
):
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        pil_images.extend(message["images"])

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversations,
        images=pil_images,
        inference_mode=True,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    return generate(
        vl_gpt,
        tokenizer,
        prepare_inputs,
        max_gen_len=max_length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        stop_words=stop_words,
        chunk_size=chunk_size
    )
```

**描述:** `deepseek_generate` 函数是生成文本的高级函数。 它接受一系列对话，一个模型，一个处理器，一个 tokenizer，停止词以及生成参数。 它首先提取对话中包含的所有图像，然后使用处理器准备模型输入。 最后，它调用 `generate` 函数来实际生成文本。 `@torch.inference_mode()` 装饰器禁用了梯度计算，从而加快了推理速度。

**用途:** 这是与模型交互的主要函数。 它封装了准备输入和生成文本所需的所有步骤。

**演示:**

```python
# 假设你已经加载了模型 (tokenizer, vl_gpt, vl_chat_processor) 并且有 conversations
# conversations = [{"role": "user", "content": "描述一下这张图片。", "images": [your_image]}] # 创建一个包含图片的对话
# stop_words = ["EOS"]
# generated_text = deepseek_generate(conversations, vl_gpt, vl_chat_processor, tokenizer, stop_words)
# print("生成的文本:", generated_text)

# 注意: your_image 必须是 PIL 图像对象
```

这个例子演示了如何使用 `deepseek_generate` 函数。它创建一个包含图像的对话，定义停止词，然后调用该函数来生成文本。  输出将是模型生成的文本描述。 请注意，你需要将 `your_image` 替换为实际的 PIL 图像对象。

```python
@torch.inference_mode()
def generate(
    vl_gpt,
    tokenizer,
    prepare_inputs,
    max_gen_len: int = 256,
    temperature: float = 0,
    repetition_penalty=1.1,
    top_p: float = 0.95,
    stop_words: List[str] = [],
    chunk_size: int = -1
):
    """Stream the text output from the multimodality model with prompt and image inputs."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    stop_words_ids = [
        torch.tensor(tokenizer.encode(stop_word)) for stop_word in stop_words
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    if chunk_size != -1:
        inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=chunk_size
        )
    else:
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        past_key_values = None

    generation_config = dict(
        inputs_embeds=inputs_embeds,
        input_ids=prepare_inputs.input_ids,
        images=prepare_inputs.images,
        images_seq_mask=prepare_inputs.images_seq_mask,
        images_spatial_crop=prepare_inputs.images_spatial_crop,
        attention_mask=prepare_inputs.attention_mask,
        past_key_values=past_key_values,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_gen_len,
        do_sample=True,
        use_cache=True,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )

    if temperature > 0:
        generation_config.update(
            {
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
        )
    else:
        generation_config["do_sample"] = False

    thread = Thread(target=vl_gpt.generate, kwargs=generation_config)
    thread.start()

    yield from streamer
```

**描述:** `generate` 函数是文本生成的核心。它使用 `TextIteratorStreamer` 以流式方式生成文本，允许在生成完成之前显示结果。 它还处理停止条件、温度采样和其他生成参数。 此函数创建了一个线程来运行实际的生成过程，以便主线程可以处理流式输出。 `@torch.inference_mode()` 装饰器禁用了梯度计算，从而加快了推理速度。

**用途:** 这是实际生成文本的地方。 它处理模型的前向传递和解码过程。

**演示:**

由于 `generate` 函数是一个生成器，所以它的使用方式略有不同。

```python
# 假设你已经加载了模型 (tokenizer, vl_gpt, vl_chat_processor) 并且有 prepare_inputs
# # prepare_inputs 应该由 vl_chat_processor 生成 (参见 deepseek_generate 函数)
# stop_words = ["EOS"]
# for chunk in generate(vl_gpt, tokenizer, prepare_inputs, stop_words=stop_words):
#     print(chunk, end="") # 逐块打印生成的文本
```

这个例子演示了如何使用 `generate` 函数。它假设你已经有了 `prepare_inputs` (通过 `vl_chat_processor` 创建)。  该函数返回一个生成器，你可以遍历它来逐块打印生成的文本。  `end=""` 确保文本块连接在一起，而不是打印在单独的行上。
