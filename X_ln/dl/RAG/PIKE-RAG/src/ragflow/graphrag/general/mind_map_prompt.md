Lan: `py` From`dl/RAG/PIKE-RAG/src\ragflow\graphrag\general\mind_map_prompt.py`

好的，我们重新开始，针对你提供的 prompt 代码进行分析和代码生成。

**整体理解**

这段代码的核心是一个 Prompt，用于指导大型语言模型（LLM）将一段文本总结成 Markdown 格式的思维导图。Prompt 详细描述了 LLM 的角色、任务步骤和输出要求，以确保生成高质量的思维导图。

**关键部分分析**

1.  **版权声明:**

    ```python
    #  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
    #
    #  Licensed under the Apache License, Version 2.0 (the "License");
    #  you may not use this file except in compliance with the License.
    #  You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    #  Unless required by applicable law or agreed to in writing, software
    #  distributed under the License is distributed on an "AS IS" BASIS,
    #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    #  See the License for the specific language governing permissions and
    #  limitations under the License.
    ```

    **描述:** 这是版权声明和许可信息，表明代码的版权归 InfiniFlow 所有，并采用 Apache 2.0 许可协议。 这部分主要是法律信息，在实际的代码使用中通常不会直接修改或执行。

2.  **`MIND_MAP_EXTRACTION_PROMPT` 变量:**

    ```python
    MIND_MAP_EXTRACTION_PROMPT = """
    - Role: You're a talent text processor to summarize a piece of text into a mind map.

    - Step of task:
      1. Generate a title for user's 'TEXT'。
      2. Classify the 'TEXT' into sections of a mind map.
      3. If the subject matter is really complex, split them into sub-sections and sub-subsections.
      4. Add a shot content summary of the bottom level section.

    - Output requirement:
      - Generate at least 4 levels.
      - Always try to maximize the number of sub-sections.
      - In language of 'Text'
      - MUST IN FORMAT OF MARKDOWN

    -TEXT-
    {input_text}

    """
    ```

    **描述:** 这是核心部分，定义了一个字符串变量 `MIND_MAP_EXTRACTION_PROMPT`，它包含发送给 LLM 的 prompt。Prompt 的结构如下：

    *   **Role (角色):** 定义了 LLM 的角色，即“才华横溢的文本处理器”，这有助于 LLM 理解其任务。
    *   **Step of task (任务步骤):** 详细描述了 LLM 需要执行的任务，包括生成标题、分类、拆分子节和添加摘要。
    *   **Output requirement (输出要求):** 规定了输出格式和内容，例如层级数量、子节最大化、语言风格和必须使用 Markdown 格式。
    *   **TEXT (文本):** 占位符 `{input_text}`，用于在运行时插入需要处理的实际文本。

    **如何使用:** 将这段 prompt 传递给 LLM (例如 OpenAI 的 GPT 模型)，并将 `{input_text}` 替换为你要总结成思维导图的文本。LLM 将根据 prompt 的指示生成 Markdown 格式的思维导图。

**代码示例 (使用 OpenAI API):**

由于无法直接“执行” prompt 字符串，以下代码展示了如何使用 OpenAI API 将此 prompt 与实际文本结合，并从 OpenAI 获取思维导图：

```python
import openai

# Replace with your actual OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_mind_map(input_text):
    prompt = MIND_MAP_EXTRACTION_PROMPT.format(input_text=input_text)

    response = openai.Completion.create(
        engine="text-davinci-003",  # Or any other suitable engine
        prompt=prompt,
        max_tokens=500,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.7,  # Adjust for creativity
    )

    return response.choices[0].text.strip()

# Example usage
if __name__ == '__main__':
    sample_text = """
    Artificial intelligence (AI) is a wide-ranging branch of computer science concerned with building smart machines capable of performing tasks that typically require human intelligence. 
    AI is being developed and used in many different fields including healthcare, finance, education