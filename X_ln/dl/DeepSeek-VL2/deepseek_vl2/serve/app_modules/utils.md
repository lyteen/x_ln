Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\serve\app_modules\utils.html`

okay, restarting and providing a detailed explanation of the provided code.

**1. License and Imports 许可和导入**

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

# -*- coding:utf-8 -*-
from __future__ import annotations

import html
import logging
import io
import os
import re
import base64
import time
from PIL import Image, ImageDraw, ImageFont

import mdtex2html
from markdown import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import ClassNotFound, get_lexer_by_name, guess_lexer
```

**描述:** 这部分代码包含版权声明和必要的模块导入.

-   `__future__ import annotations`: 允许在类定义中使用类本身作为类型提示，而无需提前声明.
-   `html`: 用于 HTML 编码和解码.
-   `logging`: 用于记录日志信息.
-   `io`: 用于处理输入/输出流.
-   `os`: 提供与操作系统交互的功能，例如创建目录.
-   `re`: 用于正则表达式操作.
-   `base64`: 用于 Base64 编码和解码，常用于图像数据的处理.
-   `time`: 用于处理时间相关操作.
-   `PIL (Image, ImageDraw, ImageFont)`: Python Imaging Library，用于图像处理.
-   `mdtex2html`: 用于将 Markdown 和 LaTeX 混合文本转换为 HTML.
-   `markdown`: 用于将 Markdown 文本转换为 HTML.
-   `pygments`: 用于代码语法高亮.

**2. Presets 预设参数**

```python
from deepseek_vl2.serve.app_modules.presets import (
    ALREADY_CONVERTED_MARK,
    BOX2COLOR,
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE
)
```

**描述:** 这部分代码导入了预设的常量和配置，例如已经转换的标记, 框的颜色映射，最大和最小的图像尺寸。这些参数可能用于控制 Gradio 应用的行为。

-   `ALREADY_CONVERTED_MARK`: 一个字符串标记，指示文本是否已经被转换为 HTML.
-   `BOX2COLOR`: 一个字典，将框的索引映射到颜色，用于图像中的对象检测框.
-   `MAX_IMAGE_SIZE`, `MIN_IMAGE_SIZE`: 图像的最大和最小尺寸，用于缩放图像.

**3. Logger Configuration 日志配置**

```python
logger = logging.getLogger("gradio_logger")


def configure_logger():
    logger = logging.getLogger("gradio_logger")
    logger.setLevel(logging.DEBUG)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("deepseek_vl2/serve/logs", exist_ok=True)
    file_handler = logging.FileHandler(
        f"deepseek_vl2/serve/logs/{timestr}_gradio_log.log"
    )
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
```

**描述:** 这段代码配置了日志记录器. 它创建了一个名为 `gradio_logger` 的日志记录器，并设置了日志级别和格式。 日志将输出到控制台和文件中。

-   `configure_logger()`:  配置日志记录器，设置日志级别为 DEBUG，创建文件处理器和控制台处理器，并设置日志格式。  日志文件以时间戳命名，保存在 `deepseek_vl2/serve/logs` 目录下。

**如何使用:** 在应用的启动阶段调用 `configure_logger()` 来初始化日志记录器。 然后，可以使用 `logger.info()`, `logger.debug()`, `logger.warning()`, `logger.error()` 等方法来记录日志信息。

**4. Text Processing Functions 文本处理函数**

```python
def strip_stop_words(x, stop_words):
    for w in stop_words:
        if w in x:
            return x[: x.index(w)].strip()
    return x.strip()


def format_output(history, text, x):
    updated_history = history + [[text, x]]
    a = [[y[0], convert_to_markdown(y[1])] for y in updated_history]
    return a, updated_history
```

**描述:** 这部分代码包含用于文本处理的函数.

-   `strip_stop_words(x, stop_words)`: 从字符串 `x` 中移除停止词。如果字符串包含任何停止词，则返回停止词之前的字符串。
-   `format_output(history, text, x)`:  格式化输出历史记录，将新文本添加到历史记录中，并将文本转换为 Markdown 格式.

**5. Markdown Conversion Functions Markdown转换函数**

```python
def markdown_to_html_with_syntax_highlight(md_str):  # deprecated
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)

        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("text", stripall=True)

        formatter = HtmlFormatter()
        highlighted_code = highlight(code, lexer, formatter)

        return f'<pre><code class="{lang}">{highlighted_code}</code></pre>'

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:  # deprecated
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(
                r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()
            ):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    result += ALREADY_CONVERTED_MARK
    return result
```

**描述:** 这部分代码包含将 Markdown 文本转换为 HTML 的函数，并提供代码语法高亮功能.

-   `markdown_to_html_with_syntax_highlight(md_str)` (deprecated): 将 Markdown 文本转换为 HTML，并使用 Pygments 进行代码语法高亮. 该函数通过正则表达式查找代码块，并使用 Pygments 相应地突出显示代码.
-   `normalize_markdown(md_text)` (deprecated): 标准化 Markdown 文本，处理列表格式.
-   `convert_mdtext(md_text)`: 将 Markdown 文本转换为 HTML，并处理代码块和内联代码。它使用 `mdtex2html` 来转换 Markdown 文本，并使用 `markdown_to_html_with_syntax_highlight` 来突出显示代码块。最后，它添加 `ALREADY_CONVERTED_MARK` 标记.

**6. Other Text Conversion Functions 其他文本转换函数**

```python
def convert_asis(userinput):
    return f'<p style="white-space:pre-wrap;">{html.escape(userinput)}</p>{ALREADY_CONVERTED_MARK}'


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    return any(s.endswith(stop_word) for stop_word in stop_words)


def detect_converted_mark(userinput):
    return bool(userinput.endswith(ALREADY_CONVERTED_MARK))


def detect_language(code):
    first_line = "" if code.startswith("\n") else code.strip().split("\n", 1)[0]
    language = first_line.lower() if first_line else ""
    code_without_language = code[len(first_line) :].lstrip() if first_line else code
    return language, code_without_language


def convert_to_markdown(text):
    text = text.replace("$", "&#36;")
    text = text.replace("\r\n", "\n")

    def replace_leading_tabs_and_spaces(line):
        new_line = []

        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line) :]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += f"{line}\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += f"{line}\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text


def add_language_tag(text):
    def detect_language(code_block):
        try:
            lexer = guess_lexer(code_block)
            return lexer.name.lower()
        except ClassNotFound:
            return ""

    code_block_pattern = re.compile(r"(```)(\w*\n[^`]+```)", re.MULTILINE)

    def replacement(match):
        code_block = match.group(2)
        if match.group(2).startswith("\n"):
            language = detect_language(code_block)
            return (
                f"```{language}{code_block}```" if language else f"```\n{code_block}```"
            )
        else:
            return match.group(1) + code_block + "```"

    text2 = code_block_pattern.sub(replacement, text)
    return text2
```

**描述:** 这部分代码包含其他文本转换和处理函数.

-   `convert_asis(userinput)`:  将用户输入转换为 HTML，保留空格和换行符，并进行 HTML 转义.  同时添加 `ALREADY_CONVERTED_MARK`.
-   `is_stop_word_or_prefix(s, stop_words)`:  检查字符串 `s` 是否以任何停止词结尾.
-   `detect_converted_mark(userinput)`:  检查用户输入是否包含 `ALREADY_CONVERTED_MARK` 标记.
-   `detect_language(code)`: 检测代码块的编程语言.
-   `convert_to_markdown(text)`: 将文本转换为 Markdown 格式，处理特殊字符和代码块.
-   `add_language_tag(text)`:  尝试检测代码块的语言，并在代码块的开头添加语言标签.

**7. Image Processing Functions 图像处理函数**

```python
def is_variable_assigned(var_name: str) -> bool:
    return var_name in locals()


def pil_to_base64(
    image: Image.Image,
    alt: str = "user upload image",
    resize: bool = True,
    max_size: int = MAX_IMAGE_SIZE,
    min_size: int = MIN_IMAGE_SIZE,
    format: str = "JPEG",
    quality: int = 95
) -> str:

    if resize:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_size / aspect_ratio, min_size, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))

    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{alt}" />'

    return img_str


def parse_ref_bbox(response, image: Image.Image):
    try:
        image = image.copy()
        image_h, image_w = image.size
        draw = ImageDraw.Draw(image)

        ref = re.findall(r'<\|ref\|>.*?<\|/ref\|>', response)
        bbox = re.findall(r'<\|det\|>.*?<\|/det\|>', response)
        assert len(ref) == len(bbox)

        if len(ref) == 0:
            return None

        boxes, labels = [], []
        for box, label in zip(bbox, ref):
            box = box.replace('<|det|>', '').replace('<|/det|>', '')
            label = label.replace('<|ref|>', '').replace('<|/ref|>', '')
            box = box[1:-1]
            for onebox in re.findall(r'\[.*?\]', box):
                boxes.append(eval(onebox))
                labels.append(label)

        for indice, (box, label) in enumerate(zip(boxes, labels)):
            box = (
                int(box[0] / 999 * image_h),
                int(box[1] / 999 * image_w),
                int(box[2] / 999 * image_h),
                int(box[3] / 999 * image_w),
            )

            box_color = BOX2COLOR[indice % len(BOX2COLOR.keys())]
            box_width = 3
            draw.rectangle(box, outline=box_color, width=box_width)

            text_x = box[0]
            text_y = box[1] - 20
            text_color = box_color
            font = ImageFont.truetype("deepseek_vl2/serve/assets/simsun.ttc", size=20)
            draw.text((text_x, text_y), label, font=font, fill=text_color)

        # print(f"boxes = {boxes}, labels = {labels}, re-render = {image}")
        return image
    except:
        return None
```

**描述:** 这部分代码包含用于图像处理的函数.

-   `is_variable_assigned(var_name)`: 检查变量是否已赋值.
-   `pil_to_base64(image, alt, resize, max_size, min_size, format, quality)`:  将 PIL 图像转换为 Base64 编码的字符串，以便在 HTML 中显示。可以选择调整图像大小并设置图像格式和质量.
-   `parse_ref_bbox(response, image)`:  解析包含对象检测信息的文本响应，并在图像上绘制边界框。它使用正则表达式来查找边界框坐标和标签，并使用 `ImageDraw` 在图像上绘制矩形和文本。它依赖于预设的 `BOX2COLOR` 来确定框的颜色.

**8. Example Display Function 示例展示函数**

```python
def display_example(image_list):
    images_html = ""
    for i, img_path in enumerate(image_list):
        image = Image.open(img_path)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=100)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{img_path}" style="height:80px; margin-right: 10px;" />'
        images_html += img_str

    result_html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <div style="flex: 1; margin-right: 10px;">{images_html}</div>
    </div>
    """

    return result_html
```

**描述:** `display_example(image_list)`:  生成 HTML 代码，用于显示图像列表. 它将每个图像转换为 Base64 编码的字符串，并将其嵌入到 `<img>` 标签中。

**总体说明:**

这段代码主要用于处理文本和图像，并将其转换为 HTML 格式，以便在 Gradio 应用中显示。它包含了许多实用函数，例如 Markdown 转换、代码语法高亮、图像大小调整和 Base64 编码。`parse_ref_bbox` 函数用于解析对象检测结果，并在图像上绘制边界框。 这些功能对于构建交互式多模态应用非常有用.
