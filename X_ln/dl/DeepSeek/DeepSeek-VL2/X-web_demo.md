Lan: `py` From`X_ln\dl/DeepSeek-VL2\web_demo.html`

好的，现在我将重新启动并提供一个改进的版本，重点在于模块化、可读性和可维护性。 我将把原始代码分解成更小的、更易于管理的函数和类，并添加注释以解释每个部分的作用。

**1. 配置文件管理 (Configuration Management):**

```python
import argparse

class AppConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, required=True,
                                 choices=[
                                     "DeepSeek-VL2-tiny", "DeepSeek-VL2-small", "DeepSeek-VL2",
                                     "deepseek-ai/deepseek-vl2-tiny", "deepseek-ai/deepseek-vl2-small",
                                     "deepseek-ai/deepseek-vl2"
                                 ], help="模型名称")
        self.parser.add_argument("--local_path", type=str, default="", help="本地模型路径 (可选)")
        self.parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP 地址")
        self.parser.add_argument("--port", type=int, default=37913, help="端口号")
        self.parser.add_argument("--root_path", type=str, default="", help="根路径")
        self.parser.add_argument("--lazy_load", action='store_true', help="是否延迟加载模型")
        self.parser.add_argument("--chunk_size", type=int, default=-1,
                                 help="分块大小，用于模型预填充 (适用于显存较小的 GPU)")

    def parse_args(self):
        return self.parser.parse_args()

# 示例使用：
if __name__ == '__main__':
    config = AppConfig()
    args = config.parse_args()
    print(f"模型名称: {args.model_name}")
    print(f"本地路径: {args.local_path}")
```

**描述：** `AppConfig` 类用于管理命令行参数。 这使得配置应用程序更容易，并允许你从命令行更改设置。 `parse_args()` 方法返回一个包含所有参数的对象。

**2. 模型加载器 (Model Loader):**

```python
import torch
from deepseek_vl2.serve.inference import load_model

class ModelLoader:
    def __init__(self):
        self.deployed_models = dict()

    def fetch_model(self, model_name: str, local_path: str = "", dtype=torch.bfloat16):
        """加载模型，如果已经加载则直接返回"""
        if model_name in self.deployed_models:
            print(f"{model_name} 已经加载.")
            return self.deployed_models[model_name]

        model_path = local_path if local_path else model_name
        print(f"{model_name} 正在加载...")
        self.deployed_models[model_name] = load_model(model_path, dtype=dtype)
        print(f"成功加载 {model_name}.")
        return self.deployed_models[model_name]

# 示例使用：
if __name__ == '__main__':
    model_loader = ModelLoader()
    # 假设已经有一个名为 "DeepSeek-VL2-small" 的模型
    model_info = model_loader.fetch_model("DeepSeek-VL2-small")
    print(f"模型信息: {model_info}")
```

**描述：** `ModelLoader` 类负责加载和管理模型。 它使用字典来存储已加载的模型，避免重复加载。 `fetch_model` 方法首先检查模型是否已加载，如果不是，则加载它并将其存储在字典中。

**3. Prompt 生成器 (Prompt Generator):**

```python
import torch
import gradio as gr
from deepseek_vl2.models.conversation import SeparatorStyle

IMAGE_TOKEN = "<image>"

def generate_prompt_with_history(text, images, history, vl_chat_processor, tokenizer, max_length=2048):
    """生成带有历史记录的 prompt"""

    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    conversation = vl_chat_processor.new_chat_template()

    if history:
        conversation.messages = history

    if images:
        num_image_tags = text.count(IMAGE_TOKEN)
        num_images = len(images)

        if num_images > num_image_tags:
            pad_image_tags = num_images - num_image_tags
            image_tokens = "\n".join([IMAGE_TOKEN] * pad_image_tags)
            text = image_tokens + "\n" + text
        elif num_images < num_image_tags:
            remove_image_tags = num_image_tags - num_images
            text = text.replace(IMAGE_TOKEN, "", remove_image_tags)

        text = (text, images)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")
    conversation_copy = conversation.copy()

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = current_prompt.replace("</s>", "") if sft_format == "deepseek" else current_prompt

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            gr.Error("用户和助手之间的消息未配对.")
            return None

        try:
            for _ in range(2):
                conversation.messages.pop(0)
        except IndexError:
            gr.Error("输入文本处理失败，无法在本轮中响应.")
            return None

    gr.Error("Prompt 未能在 max_length 限制内生成.")
    return None

def get_prompt(conv) -> str:
    """获取用于生成的 prompt"""
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        ret = system_prompt + seps[0] if system_prompt else ""
        for i, (role, message) in enumerate(conv.messages):
            if message:
                message = message[0] if type(message) is tuple else message  # 提取消息内容
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt()

# 示例使用：
# 需要导入 vl_chat_processor, tokenizer, 并且设置 images, history
# 示例代码只展示了函数调用，实际使用需要根据你的项目设置
# prompt = generate_prompt_with_history(text, images, history, vl_chat_processor, tokenizer, max_length=2048)
```

**描述：** 这些函数负责根据用户输入、图像和历史记录生成 prompt。 `generate_prompt_with_history` 函数处理图像标记和历史记录，而 `get_prompt` 函数根据对话风格生成最终的 prompt。

**4. Gradio 接口构建 (Gradio Interface Builder):**

```python
import gradio as gr
from deepseek_vl2.serve.app_modules.gradio_utils import (
    cancel_outputing,
    delete_last_conversation,
    reset_state,
    reset_textbox,
    wrap_gen_fn,
)
from deepseek_vl2.serve.app_modules.overwrites import reload_javascript
from deepseek_vl2.serve.app_modules.presets import (
    CONCURRENT_COUNT,
    MAX_EVENTS,
    description,
    description_top,
    title
)
from deepseek_vl2.serve.app_modules.utils import (
    configure_logger,
    is_variable_assigned,
    strip_stop_words,
    parse_ref_bbox,
    pil_to_base64,
    display_example
)

def build_gradio_interface(args):
    """构建 Gradio 界面"""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        history = gr.State([])
        input_text = gr.State()
        input_images = gr.State()

        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="deepseek_chatbot",
                        show_share_button=True,
                        bubble_full_width=False,
                        height=600,
                    )
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(
                            show_label=False, placeholder="Enter text", container=False
                        )
                    with gr.Column(
                        min_width=70,
                    ):
                        submitBtn = gr.Button("Send")
                    with gr.Column(
                        min_width=70,
                    ):
                        cancelBtn = gr.Button("Stop")
                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn")

            with gr.Column():
                upload_images = gr.Files(file_types=["image"], show_label=True)
                gallery = gr.Gallery(columns=[3], height="200px", show_label=True)

                upload_images.change(preview_images, inputs=upload_images, outputs=gallery)

                with gr.Tab(label="Parameter Setting") as parameter_row:
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=8192,
                        value=4096,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
                    model_select_dropdown = gr.Dropdown(
                        label="Select Models",
                        choices=[args.model_name],
                        multiselect=False,
                        value=args.model_name,
                        interactive=True,
                    )

                    show_images = gr.HTML(visible=False)

        def format_examples(examples_list):
            examples = []
            for images, texts in examples_list:
                examples.append([images, display_example(images), texts])

            return examples

        examples_list = [  # 示例数据
            [["images/visual_grounding_1.jpeg"], "<|ref|>The giraffe at the back.<|/ref|>"],
            [["images/visual_grounding_2.jpg"], "找到<|ref|>淡定姐<|/ref|>"],
            [["images/visual_grounding_3.png"], "Find all the <|ref|>Watermelon slices<|/ref|>"],
            [["images/grounding_conversation_1.jpeg"], "<|grounding|>I want to throw out the trash now, what should I do?"],
            [["images/incontext_visual_grounding_1.jpeg", "images/icl_vg_2.jpeg"], "<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image."],
            [["images/vqa_1.jpg"], "Describe each stage of this image in detail"],
            [["images/multi_image_1.jpeg", "images/mi_2.jpeg", "images/mi_3.jpeg"], "能帮我用这几个食材做一道菜吗?"],
        ]

        gr.Examples(
            examples=format_examples(examples_list),
            inputs=[upload_images, show_images, text_box],
        )

        gr.Markdown(description)  # 描述信息

        input_widgets = [
            input_text,
            input_images,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
            model_select_dropdown,
        ]
        output_widgets = [chatbot, history, status_display]

        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[text_box, upload_images],
            outputs=[input_text, input_images, text_box, upload_images, submitBtn],
            show_progress=True,
        )

        predict_args = dict(
            fn=predict,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        retry_args = dict(
            fn=retry,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        reset_args = dict(
            fn=reset_textbox, inputs=[], outputs=[text_box, status_display]
        )

        predict_events = [
            text_box.submit(**transfer_input_args).then(**predict_args),
            submitBtn.click(**transfer_input_args).then(**predict_args),
        ]

        emptyBtn.click(reset_state, outputs=output_widgets, show_progress=True)
        emptyBtn.click(**reset_args)
        retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            output_widgets,
            show_progress=True,
        )

        cancelBtn.click(cancel_outputing, [], [status_display], cancels=predict_events)

    return demo

# 示例使用：
# 需要传入命令行参数 args
# demo = build_gradio_interface(args)
```

**描述：**  `build_gradio_interface` 函数负责构建 Gradio 界面。它创建了 chatbot、文本框、按钮、滑块和下拉菜单等组件。 它还定义了 Gradio 事件，例如提交文本、单击按钮和取消生成。

**5. 主要函数 (Main Function):**

```python
import gradio as gr
from deepseek_vl2.serve.app_modules.overwrites import reload_javascript
from deepseek_vl2.serve.app_modules.presets import (
    CONCURRENT_COUNT,
    MAX_EVENTS,
)

def main():
    """主函数"""
    config = AppConfig()
    args = config.parse_args()

    model_loader = ModelLoader()
    if not args.lazy_load:
        model_loader.fetch_model(args.model_name, args.local_path)

    demo = build_gradio_interface(args)
    demo.title = "DeepSeek-VL2 Chatbot"

    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT, max_size=MAX_EVENTS).launch(
        share=True,
        favicon_path="deepseek_vl2/serve/assets/favicon.ico",
        inbrowser=False,
        server_name=args.ip,
        server_port=args.port,
        root_path=args.root_path
    )

if __name__ == "__main__":
    main()
```

**描述：** `main` 函数是应用程序的入口点。 它解析命令行参数，加载模型，构建 Gradio 界面，并启动应用程序。

**改进总结：**

*   **模块化：** 代码被分解成更小的、更易于管理的函数和类。
*   **可读性：** 添加了注释来解释每个部分的作用。
*   **可维护性：**  通过使用类和函数，更容易更改和扩展代码。
*   **配置管理：** 使用 `argparse` 来管理命令行参数。
*   **模型加载：** 使用 `ModelLoader` 来加载和管理模型。
*   **Prompt 生成：** 使用 `generate_prompt_with_history` 和 `get_prompt` 来生成 prompt。
*   **Gradio 界面：** 使用 `build_gradio_interface` 来构建 Gradio 界面。

这个改进的版本更易于理解、修改和维护。  例如，如果要添加新的模型，只需要修改 `AppConfig` 类中的 `model_name` 参数，并在 `ModelLoader` 类中添加加载新模型的代码。 如果要修改 Gradio 界面，只需要修改 `build_gradio_interface` 函数。

希望这个改进的版本能够满足您的需求！
