Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\qa\multiple_choice.py`

Okay, I will restart and provide a more robust and adaptable version of the code. This version will address the following improvements:

*   **Multiple Choice Support:**  The parser and templates will be modified to handle scenarios where multiple options can be selected as correct answers.
*   **Improved Error Handling:** Enhanced error handling during the parsing stage.
*   **More Structured Output:** Standardize the output format for better downstream processing.
*   **Cleaner Code:** Improve code readability and maintainability.

Here's the refactored code with explanations:

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import traceback
from typing import Dict, List, Tuple, Union

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.lxml_parser import get_soup_from_content


multiple_choice_qa_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option(s) from the given options. One or more options may be correct. The chosen option(s) should be correct and the most suitable to answer the given question. If you don't have sufficient data to determine, randomly choose one or more option(s) from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask(s), separated by commas if multiple are chosen.</mask>
    <options>The option details corresponding to the chosen option masks, each separated by a newline if multiple are chosen.</options>
  </answer>
</result>

# Question
{content}

# Options
{options_str}

# Thinking and Answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str"],
)


multiple_choice_qa_with_reference_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option(s) from the given options. One or more options may be correct. You can refer to the references provided when thinking and answering. Please note that the references may or may not be relevant to the question. If you don't have sufficient information to determine, randomly choose one or more option(s) from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask(s), separated by commas if multiple are chosen.</mask>
    <options>The option details corresponding to the chosen option masks, each separated by a newline if multiple are chosen.</options>
  </answer>
</result>

# Question
{content}

# Options
{options_str}

# References
{references_str}

# Thinking and Answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str", "references_str"],
)


multiple_choice_qa_with_reference_and_review_template = MessageTemplate(
    template=[
        ("system", "You are an helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option(s) from the given options. One or more options may be correct. You can refer to the references provided when thinking and answering. Please note that the references may or may not be relevant to the question. If you don't have sufficient information to determine, randomly choose one or more option(s) from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask(s), separated by commas if multiple are chosen.</mask>
    <options>The option details corresponding to the chosen option masks, each separated by a newline if multiple are chosen.</options>
  </answer>
</result>

# Question
{content}

# Options
{options_str}

# References
{references_str}

# Review
Let's now review the question, options and output format again:

# Question
{content}

# Options
{options_str}

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask(s), separated by commas if multiple are chosen.</mask>
    <options>The option details corresponding to the chosen option masks, each separated by a newline if multiple are chosen.</options>
  </answer>
</result>

# Thinking and Answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str", "references_str"],
)


class MultipleChoiceQaParser(BaseContentParser):
    def __init__(self) -> None:
        self.option_masks: List[str] = []
        self.options: Dict[str, str] = {}

    def encode(self, content: str, options: Dict[str, str], answer_mask_labels: List[str], **kwargs) -> Tuple[str, dict]:
        """
        Encodes the input content and options into a format suitable for the language model.

        Args:
            content (str): The question or content to be presented to the model.
            options (Dict[str, str]): A dictionary of options, where keys are option masks (e.g., 'A', 'B') and values are the option text.
            answer_mask_labels (List[str]): A list of the correct answer mask labels.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[str, dict]: The encoded content and a dictionary of supplementary information to be used in the prompt.
        """
        self.option_masks = sorted(list(options.keys()))
        self.options = options.copy()

        options_str = "\n".join([f"{key}: {self.options[key]}" for key in self.option_masks])

        for mask_label in answer_mask_labels:
            if mask_label not in self.option_masks:
                raise ValueError(
                    f"Given answer mask label {mask_label}, but no corresponding option provided: {self.option_masks}"
                )

        return content, {"options_str": options_str}

    def decode(self, content: str, options: Dict[str, str], **kwargs) -> Dict[str, Union[str, List[str]]]:
        """
        Decodes the output from the language model.

        Args:
            content (str): The raw text output from the language model.
            options (Dict[str, str]): A dictionary of options, where keys are option masks and values are the option text.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Union[str, List[str]]]: A dictionary containing the extracted thinking, answer masks, and chosen options.
        """
        if not content:
            return {}

        try:
            result_soup: BeautifulSoup = get_soup_from_content(content, tag="result")

            if result_soup is None:
                thinking_soup = get_soup_from_content(content, tag="thinking")
                answer_soup = get_soup_from_content(content, "answer")
            else:
                thinking_soup = result_soup.find("thinking")
                answer_soup = result_soup.find("answer")

            thinking = thinking_soup.text if thinking_soup else ""

            if answer_soup:
                mask_soup = answer_soup.find("mask")
                masks_text = mask_soup.text.strip() if mask_soup else ""
                masks = [m.strip() for m in masks_text.split(",") if m.strip()]  # Handle multiple masks

                option_soup = answer_soup.find("options")
                options_text = option_soup.text.strip() if option_soup else ""
                chosen_options = [o.strip() for o in options_text.split("\n") if o.strip()]  # Handle multiple options

                # Validation: Ensure that the number of masks matches the number of chosen options
                if len(masks) != len(chosen_options):
                    print(f"Warning: Number of masks ({len(masks)}) does not match the number of chosen options ({len(chosen_options)}).")
                    # Consider logging or raising an exception if this is critical.

                # Validate the masks
                for mask in masks:
                    if mask not in self.option_masks:
                        print(f"Warning: Invalid mask '{mask}' found in the answer.")
                        # Consider logging or raising an exception if this is critical.
            else:
                masks = []
                chosen_options = []

        except Exception as e:
            print("Error during decoding:")
            print("Content:", content)
            print("Exception:", e)
            traceback.print_exc()
            return {}  # or raise the exception if you want to stop execution

        return {
            "thinking": thinking,
            "answer": masks,
            "chosen_options": chosen_options,
        }


class MultipleChoiceQaWithReferenceParser(MultipleChoiceQaParser):
    def encode(self, content: str, options: Dict[str, str], answer_mask_labels: List[str], **kwargs) -> Tuple[str, Dict]:
        content, supplementary = super().encode(content, options, answer_mask_labels, **kwargs)

        references = kwargs.get("references", [])
        supplementary["references_str"] = "\n".join([reference.strip() for reference in references])

        return content, supplementary


multiple_choice_qa_protocol = CommunicationProtocol(
    template=multiple_choice_qa_template,
    parser=MultipleChoiceQaParser(),
)


multiple_choice_qa_with_reference_protocol = CommunicationProtocol(
    template=multiple_choice_qa_with_reference_template,
    parser=MultipleChoiceQaWithReferenceParser(),
)


multiple_choice_qa_with_reference_and_review_protocol = CommunicationProtocol(
    template=multiple_choice_qa_with_reference_and_review_template,
    parser=MultipleChoiceQaWithReferenceParser(),
)

```

**Key Improvements and Explanations:**

1.  **Multiple Choice Handling:**

    *   **Templates:** The prompt templates (`multiple_choice_qa_template`, `multiple_choice_qa_with_reference_template`, `multiple_choice_qa_with_reference_and_review_template`) are updated to reflect that multiple options can be selected.  The instructions now explicitly state this.  The output format within the prompt is also updated to specify comma-separated masks and newline-separated options.
    *   **Parser (decode):**  The `decode` method in `MultipleChoiceQaParser` now correctly parses comma-separated masks and newline-separated options.  It splits the `mask` and `options` text based on the delimiters.
    *   **Output Structure:** The `decode` method returns a dictionary where `"answer"` is now a `List[str]` (list of masks) and `"chosen_options"` is also a `List[str]` (list of chosen option text).

2.  **Error Handling:**

    *   **`encode` Validation:**  The `encode` method now explicitly checks if the `answer_mask_labels` are valid option masks. It raises a `ValueError` if an invalid mask is provided, providing more informative error messages.
    *   **`decode` Validation:**  The `decode` method includes checks to ensure the number of masks and chosen options are consistent.  It also validates that the extracted masks are valid option masks.  Warnings are printed to the console if inconsistencies are found.
    *   **Exception Handling:** A `try...except` block is used in the `decode` method to catch potential errors during parsing (e.g., if the LLM output is not in the expected format).  A traceback is printed to help with debugging, and an empty dictionary is returned to prevent the program from crashing.

3.  **Structured Output:**

    *   The `decode` method returns a dictionary with the following structure:

        ```python
        {
            "thinking": str,        # The LLM's reasoning
            "answer": List[str],    # A list of the chosen option masks (e.g., ["A", "C"])
            "chosen_options": List[str] # A list of the chosen option details (e.g., ["Option A text", "Option C text"])
        }
        ```

    This structure makes it easier to access and process the LLM's output in downstream tasks.

4.  **Code Clarity:**

    *   Comments have been added to explain the purpose of each section of the code.
    *   More descriptive variable names are used.
    *   The code is formatted consistently for better readability.

**Example Usage:**

```python
# Example Usage

options = {
    "A": "Option A: The capital of France is Paris.",
    "B": "Option B: The capital of France is London.",
    "C": "Option C: The Earth is flat.",
    "D": "Option D: The Earth is a sphere."
}

# The LLM might return this string (simulating multiple correct answers):
llm_output = """
<result>
  <thinking>The capital of France is Paris, and the Earth is a sphere.</thinking>
  <answer>
    <mask>A,D</mask>
    <options>Option A: The capital of France is Paris.
Option D: The Earth is a sphere.</options>
  </answer>
</result>
"""

parser = MultipleChoiceQaParser()
decoded_output = parser.decode(llm_output, options)

print(decoded_output)
# Expected Output:
# {
#     'thinking': 'The capital of France is Paris, and the Earth is a sphere.',
#     'answer': ['A', 'D'],
#     'chosen_options': ['Option A: The capital of France is Paris.', 'Option D: The Earth is a sphere.']
# }
```

**说明 (Chinese Explanation):**

这段代码对之前的版本进行了全面的改进，以支持多项选择题，并提供更清晰的输出和更好的错误处理。

1.  **支持多项选择：** 提示模板现在明确指出可以选择多个答案。解析器 (parser) 也经过修改，可以正确解析以逗号分隔的选项掩码 (mask) 和以换行符分隔的选项内容。 输出结果的结构也更加清晰，`answer` 字段现在是一个包含所有选择的选项掩码的列表，`chosen_options` 字段是包含所有选择的选项内容的列表。

2.  **错误处理：** `encode` 方法会验证给定的答案掩码是否有效。 `decode` 方法会检查选项掩码和选项内容是否一致，并在发现问题时发出警告。 此外，`decode` 方法还包含 `try...except` 块，以处理解析过程中可能出现的异常，并防止程序崩溃。

3.  **结构化输出：** `decode` 方法返回一个字典，其中包含思考过程 (`thinking`)、答案掩码列表 (`answer`) 和选择的选项内容列表 (`chosen_options`)。 这种结构使得下游任务更容易访问和处理 LLM 的输出。

4.  **代码清晰度：** 添加了注释来解释代码的目的，并使用更具描述性的变量名。 代码格式也经过一致性调整，提高了可读性。

This revised version provides a more robust and flexible solution for handling multiple-choice questions with language models.
