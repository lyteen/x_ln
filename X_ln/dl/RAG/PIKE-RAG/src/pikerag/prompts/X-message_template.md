Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\message_template.py`

Okay, restarting with a focus on providing even more refined and helpful responses.  I will now aim to break down the code into smaller, more digestible chunks with improved explanations and a more practical demo.

**1. Streamlined MessageTemplate Class:**

```python
from __future__ import annotations
from string import Formatter
from typing import Dict, List, Tuple, Union, Callable

from pydantic import BaseModel, model_validator


formatter = Formatter()


class MessageTemplate(BaseModel):
    """A message template for a language model."""
    template: List[Tuple[str, str]]  # (role, content) pairs
    input_variables: List[str] = None # Variables to fill
    partial_variables: Dict[str, Union[str, Callable[[], str]]] = {} # Pre-filled values

    @model_validator(mode="after")
    def validate_input_variables(self) -> MessageTemplate:
        """Validates input variables against the template."""
        input_variables_in_template = sorted(
            {
                field_name
                for _, content_template in self.template
                for _, field_name, _, _ in formatter.parse(content_template)
                if field_name is not None
            }
        )

        if self.input_variables is None:
            self.input_variables = list(input_variables_in_template)

        else:
            input_variable_set = set(self.input_variables)
            partial_variable_set = set(self.partial_variables.keys())
            parsed_variable_set = set(input_variables_in_template)

            # Check if all variables in template are declared
            for variable in parsed_variable_set:
                assert variable in input_variable_set or variable in partial_variable_set, (
                    f"{variable} in template but not shown in input variables list!"
                )

            # Check if all declared variables are actually used in the template
            for variable in input_variable_set:
                assert variable in parsed_variable_set, (
                    f"{variable} in input variable list but cannot found in template!"
                )

        return self

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> MessageTemplate:
        """Partially fill the template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = [v for v in self.input_variables if v not in kwargs] # Keep only unfilled variables
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Union[str, Callable[[], str]]) -> Dict[str, str]:
        """Merge partial and user-provided variables."""
        partial_kwargs = {
            k: v() if callable(v) else v  # Evaluate callable partial values
            for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    def format(self, **kwargs) -> List[Dict[str, str]]:
        """Format the template into a list of messages."""
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result: List[Dict[str, str]] = [
            {
                "role": role,
                "content": formatter.format(content, **kwargs),
            }
            for role, content in self.template
        ]
        return result

```

**代码描述 (Chinese):**

这段代码定义了一个 `MessageTemplate` 类，用于管理语言模型的消息模板。模板由一系列包含角色和内容的元组组成。这个类允许你预先填充 (partial) 一些变量，然后在运行时使用 format 方法填充剩下的变量，最终生成一个可以用于语言模型的包含 role 和 content 的消息列表。

*   `template`:  消息模板本身，是一个 role 和 content 的 tuple 的列表。例如 `[("system", "你好"), ("user", "我的名字是 {name}")]`。
*   `input_variables`:  format 方法需要填充的变量名列表。 建议手动指定，而不是依赖自动推断。
*   `partial_variables`:  预先填充的变量，可以在初始化时设置。

`validate_input_variables` 方法会检查所有在模板中使用的变量都已声明，并且所有声明的变量都在模板中使用，以避免错误。

`partial` 方法允许你预先填充一些变量，返回一个新的 `MessageTemplate` 实例，这个实例已经填充了部分变量。

`format` 方法将所有变量填充到模板中，生成最终的消息列表。 如果 `partial_variables` 包含 callable 的值，则会在 format 时调用。

**Demo (Chinese):**

```python
# 使用示例 (中文)
template = MessageTemplate(
    template=[
        ("system", "你是一个有用的助手。"),
        ("user", "我的问题是：{question}，我需要 {language} 版本的答案。"),
    ],
    input_variables=["question", "language"],
)

# 预先填充 language 变量
template = template.partial(language="中文")

# 最终填充 question 变量并生成消息
messages = template.format(question="如何使用 MessageTemplate?")
print(messages)
```

**解释 (Chinese):**

在这个例子中，我们首先定义了一个包含 `question` 和 `language` 两个变量的模板。 然后，我们使用 `partial` 方法预先填充了 `language` 变量为 "中文"。  最后，我们使用 `format` 方法填充 `question` 变量并生成最终的消息列表。 输出的 `messages` 将是一个包含 role 和 content 的 Python 字典的 List，可以直接用于语言模型。

---

**Key Improvements and Explanations:**

1.  **Clearer Validation:**  The `validate_input_variables` method now explicitly checks for both missing and unused variables, raising helpful assertion errors.
2.  **Callable Partial Values:** The `_merge_partial_and_user_variables` method now correctly handles callable values in `partial_variables`, evaluating them when the `format` method is called.  This allows for dynamic values to be used.
3.  **More Concise `partial` Method:**  The `partial` method now uses a list comprehension for improved readability.
4.  **Improved Documentation:** Added more comprehensive docstrings and comments.
5.  **Chinese Demo:** Provided a complete demo example with Chinese comments to demonstrate the usage of the class in a practical scenario.

This refined response offers more robust variable validation, supports dynamic partial values, and provides a clear and usable example, making the `MessageTemplate` class more reliable and easier to integrate into your projects.
