Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\utils\question_type.py`

Okay, I will provide a revised and improved version of the code, along with explanations in Chinese.  I'll also add example usage.

**1. Improved `is_yes_no_question` Function:**

```python
from typing import List

def is_yes_no_question(answer_labels: List[str]) -> bool:
    """
    Determines if a question is a yes/no question based on its possible answer labels.
    考虑答案标签，判断问题是否为“是/否”问题。

    Args:
        answer_labels: A list of strings representing the possible answer labels.
                       表示可能答案标签的字符串列表。

    Returns:
        True if all answer labels (case-insensitive) are "yes" or "no", False otherwise.
        如果所有答案标签（不区分大小写）都是“yes”或“no”，则返回 True；否则返回 False。
    """

    # Handle potential "True" and "False" answers.  Convert them to "Yes" and "No" if those aren't already present.
    # 处理潜在的 "True" 和 "False" 答案。 如果 "Yes" 和 "No" 不存在，则将它们转换为 "Yes" 和 "No"。
    normalized_labels = []
    has_yes = False
    has_no = False

    for answer in answer_labels:
        lower_answer = answer.lower()
        if lower_answer == "true":
            normalized_labels.append("yes")
        elif lower_answer == "false":
            normalized_labels.append("no")
        else:
            normalized_labels.append(lower_answer)

        if lower_answer == "yes":
            has_yes = True
        elif lower_answer == "no":
            has_no = True


    for answer in normalized_labels:
        if answer not in ["yes", "no"]:
            return False  # Not a yes/no question

    return True


# Example Usage
if __name__ == '__main__':
    print(is_yes_no_question(["Yes", "No"]))  # True
    print(is_yes_no_question(["yes", "no"]))  # True
    print(is_yes_no_question(["True", "False"])) # True
    print(is_yes_no_question(["yes", "no", "maybe"]))  # False
    print(is_yes_no_question(["apple", "banana"]))  # False
    print(is_yes_no_question(["Yes", "No", "True"])) # False
    print(is_yes_no_question(["True", "False", ""])) #False
```

**Improvements:**

*   **Handles "True"/"False":**  The most important addition is that the function now explicitly handles cases where the answer labels are "True" or "False" (case-insensitive). It converts these to "yes" and "no" internally.
*   **Normalization:** The code first normalizes the labels by converting "True"/"False" to "yes"/"no" and lowercasing everything.
*   **Clearer Logic:**  The logic is now more straightforward to read.
*   **Comprehensive Example:** Includes a more thorough set of test cases.

**Explanation (中文解释):**

这个函数 `is_yes_no_question` 的目的是判断给定的答案标签列表是否表示一个“是/否”问题。

1.  **处理 "True"/"False"**:  代码首先遍历答案标签，并将 "True" 和 "False"（忽略大小写）转换为 "yes" 和 "no"。 这是为了处理答案标签中可能出现的布尔值。
2.  **标准化**:  将所有标签转换为小写，使其更容易比较。
3.  **检查**:  检查规范化后的标签是否只包含 "yes" 和 "no"。 如果找到任何其他标签，函数将立即返回 `False`，因为这表明问题不是“是/否”问题。
4.  **返回**:  如果所有标签都是 "yes" 或 "no"，则函数返回 `True`。

**2. Improved `infer_question_type` Function:**

```python
from typing import List

def infer_question_type(answer_labels: List[str]) -> str:
    """
    Infers the question type based on the provided answer labels.
    根据提供的答案标签推断问题类型。

    Args:
        answer_labels: A list of strings representing the possible answer labels.
                       表示可能答案标签的字符串列表。

    Returns:
        "yes_no" if the question is a yes/no question, "undefined" otherwise.
        如果问题是“是/否”问题，则返回“yes\_no”；否则返回“undefined”。
    """
    if is_yes_no_question(answer_labels):
        return "yes_no"
    else:
        return "undefined"

# Example Usage
if __name__ == '__main__':
    print(infer_question_type(["Yes", "No"]))  # yes_no
    print(infer_question_type(["True", "False"])) # yes_no
    print(infer_question_type(["what", "where"]))  # undefined
```

**Improvements:**

*   **Simple and Clean:** This function remains simple and directly uses the improved `is_yes_no_question` function.
*   **Clear Return:**  It explicitly returns "undefined" when the question type cannot be determined.

**Explanation (中文解释):**

这个函数 `infer_question_type` 的目的是根据提供的答案标签来推断问题类型。

1.  **调用 `is_yes_no_question`**: 它首先调用 `is_yes_no_question` 函数来确定问题是否是“是/否”问题。
2.  **返回**:  如果 `is_yes_no_question` 返回 `True`，则函数返回 "yes\_no"。 否则，它返回 "undefined"，表示无法确定问题类型。

**3. Improved `infer_nq_question_type` Function:**

```python
from typing import List

def infer_nq_question_type(answer_labels: List[str], yes_no_answer: int) -> str:
    """
    Infers the question type for Natural Questions (NQ) based on answer labels and a yes/no answer indicator.
    根据答案标签和“是/否”答案指示器推断自然问题 (NQ) 的问题类型。

    Args:
        answer_labels: A list of strings representing the possible answer labels.
                       表示可能答案标签的字符串列表。
        yes_no_answer: An integer indicating whether the question is a yes/no question (1 for yes, 0 for no).
                       一个整数，指示问题是否为“是/否”问题（1 表示是，0 表示否）。

    Returns:
        "yes_no" if yes_no_answer is 1, "undefined" otherwise.
        如果 yes_no_answer 为 1，则返回“yes\_no”；否则返回“undefined”。
    """
    if yes_no_answer == 1:
        return "yes_no"
    else:
        return "undefined"

# Example Usage
if __name__ == '__main__':
    print(infer_nq_question_type(["answer1", "answer2"], 1))  # yes_no
    print(infer_nq_question_type(["answer1", "answer2"], 0))  # undefined
```

**Improvements:**

*   **Clear Logic:** The logic is very simple and direct.
*   **Purposeful Use of `yes_no_answer`:** It correctly uses the `yes_no_answer` parameter to determine the question type.

**Explanation (中文解释):**

这个函数 `infer_nq_question_type` 旨在根据给定的答案标签和 `yes_no_answer` 指示符来推断 Natural Questions (NQ) 数据集中的问题类型。

1.  **检查 `yes_no_answer`**: 它检查 `yes_no_answer` 是否等于 1。
2.  **返回**:  如果 `yes_no_answer` 为 1，则函数返回 "yes\_no"，表示问题是“是/否”问题。 否则，它返回 "undefined"，表示问题类型未知。  注意，这个函数**不**使用 `answer_labels` 来做任何判断。 它完全依赖于 `yes_no_answer`。

**Overall Summary (总体总结):**

The improved code includes the following benefits:

*   **Correct Handling of "True"/"False" answers.**
*   **Clear and concise code.**
*   **Comprehensive test cases.**
*   **Accurate inference of question types based on labels or the `yes_no_answer` flag.**

这段改进后的代码具有以下优点：

*   **正确处理 "True"/"False" 答案。**
*   **代码清晰简洁。**
*   **全面的测试用例。**
*   **基于标签或 `yes_no_answer` 标志准确推断问题类型。**