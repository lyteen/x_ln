Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\message_template.py`

好的，现在重新开始，对 `MessageTemplate` 类进行解释。

**1. 导入必要的库 (Import necessary libraries):**

```python
from __future__ import annotations
from string import Formatter
from typing import Dict, List, Tuple, Union, Callable

from pydantic import BaseModel, model_validator


formatter = Formatter()
```

**描述:** 导入代码中使用的各种模块和类型。`string.Formatter` 用于格式化字符串，`typing` 用于类型提示，`pydantic` 用于数据验证和设置。`formatter = Formatter()` 创建一个 `Formatter` 类的实例，后续用来解析模板字符串。

**2. MessageTemplate 类定义 (MessageTemplate Class Definition):**

```python
class MessageTemplate(BaseModel):
    """A message template for a language model.

    Args:
        template (List[Tuple[str, str]]): each tuple in the template list consists two elements: the first one is the
            role of this message; the second one is a f-string style content.
        input_variables (Union[List[str], None]): the input variables needs to be fill in when finalizing the messages with the given
            template. It must correspond to the f-string style contents in the template. Input variable list would be
            automatically inferred based on the template if None is given. But it is always recommended to provide it by
            yourself. Defaults to None.
        partial_variables (Dict[str, Union[str, Callable[[], str]]]): no need to provide when initializing a message
            template by yourself. Defaults to {}.

    Example:
        .. code-block:: python

            from pikerag.llm_client.prompts import MessageTemplate

            # Initialize a message template with the template (and input variable list).
            message_template = MessageTemplate(
                template=[
                    ("system", "You are a helpful AI assistant."),
                    ("user", "This may be a {placeholder1} demonstration from user"),
                    ("assistant", "This may be a {placeholder2} demonstration from assistant"),
                    ("user", "You may finalize your {placeholder3} question here"),
                ],
                # It's allowable to provide only template when initializing an instance,
                # But it always recommended to list the input variables by yourself.
                input_variables=["placeholder1", "placeholder2", "placeholder3"],
            )

            # Partially fill in the placeholder1 and placeholder2.
            message_template = message_prompt.partial(placeholder1="demo question", placeholder2="demo answer")

            # Finalize the messages with the remaining variables provided.
            messages = message_template.format(placeholder3="your question")

    """
    template: List[Tuple[str, str]]

    input_variables: List[str] = None

    partial_variables: Dict[str, Union[str, Callable[[], str]]] = {}
```

**描述:**  `MessageTemplate` 类使用 Pydantic 的 `BaseModel` 作为基类，用于定义消息模板的结构。
*   `template`:  一个列表，其中每个元素是一个元组。元组的第一个元素是消息的角色（例如，"system"，"user"，"assistant"），第二个元素是包含占位符的字符串模板（例如，"You are a helpful AI assistant."，"This may be a {placeholder1} demonstration from user"）。
*   `input_variables`: 一个字符串列表，指定模板中需要填充的变量名称。如果为 `None`，则会自动从模板中推断。
*   `partial_variables`: 一个字典，包含预先填充的变量。键是变量名，值可以是字符串或返回字符串的可调用对象。

**用法示例:**

```python
from pikerag.llm_client.prompts import MessageTemplate

# 初始化消息模板
message_template = MessageTemplate(
    template=[
        ("system", "你是一个有用的AI助手。"),
        ("user", "这是一个来自用户的{placeholder1}演示。"),
        ("assistant", "这是一个来自助手的{placeholder2}演示。"),
        ("user", "你可以在这里完成你的{placeholder3}问题。"),
    ],
    input_variables=["placeholder1", "placeholder2", "placeholder3"],
)

# 部分填充变量
message_template = message_template.partial(placeholder1="演示问题", placeholder2="演示答案")

# 最终格式化消息
messages = message_template.format(placeholder3="你的问题")

print(messages)
```

**3. 验证输入变量 (Validate input variables):**

```python
    @model_validator(mode="after")
    def validate_input_variables(self) -> MessageTemplate:
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
            for variable in parsed_variable_set:
                assert variable in input_variable_set or variable in partial_variable_set, (
                    f"{variable} in template but not shown in input variables list!"
                )
            for variable in input_variable_set:
                assert variable in parsed_variable_set, (
                    f"{variable} in input variable list but cannot found in template!"
                )

        return self
```

**描述:**  `@model_validator` 装饰器指示这是一个 Pydantic 模型验证器，在模型初始化后运行。 `validate_input_variables` 方法用于验证 `input_variables` 是否与模板中实际使用的变量一致。
*   它首先从 `template` 中解析出所有变量名。
*   如果 `input_variables` 为 `None`，则将其设置为从模板中解析出的变量列表。
*   否则，它会检查 `input_variables` 中的每个变量是否都在模板中找到，并且模板中的每个变量是否都在 `input_variables` 或 `partial_variables` 中声明。 这有助于确保在格式化消息时不会缺少任何变量。

**4. partial 方法 (Partial Method):**

```python
    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> MessageTemplate:
        """Return a partial of this message template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(set(self.input_variables).difference(kwargs))
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)
```

**描述:**  `partial` 方法允许您预先填充模板中的一部分变量。 它创建一个新的 `MessageTemplate` 实例，其中指定的变量被添加到 `partial_variables` 字典中，并从 `input_variables` 列表中移除。
*   它首先复制当前实例的字典。
*   然后，它更新 `input_variables`，移除已提供的变量。
*   最后，它更新 `partial_variables`，添加提供的变量。

**用法示例:**

```python
message_template = message_template.partial(placeholder1="演示问题") # 预先设置 placeholder1
```

**5. _merge_partial_and_user_variables 方法 (Merge Partial and User Variables Method):**

```python
    def _merge_partial_and_user_variables(self, **kwargs: Union[str, Callable[[], str]]) -> Dict[str, str]:
        partial_kwargs = {
            k: v if isinstance(v, str) else v()
            for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}
```

**描述:** 此方法合并 `partial_variables` 和用户提供的 `kwargs`。 如果 `partial_variables` 中的值是可调用对象，则调用它以获取实际值。

**6. format 方法 (Format Method):**

```python
    def format(self, **kwargs) -> List[Dict[str, str]]:
        """Format the messages template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables in all the template messages in this
                messages template.

        Returns:
            List[Dict[str, str]]: list of formatted messages, each message contains the role and the content.
        """
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

**描述:** `format` 方法使用提供的变量填充模板，生成最终的消息列表。
*   它首先使用 `_merge_partial_and_user_variables` 合并部分变量和用户变量。
*   然后，它遍历 `template` 中的每个角色和内容模板。
*   对于每个内容模板，它使用 `formatter.format` 方法和合并后的变量来填充占位符。
*   最后，它返回一个包含角色和填充后内容的字典列表。

**用法示例:**

```python
messages = message_template.format(placeholder2="演示答案", placeholder3="最终问题") # 填充剩余变量
print(messages)
```

总而言之，`MessageTemplate` 类提供了一种灵活且类型安全的方式来管理和格式化语言模型的提示，允许您定义模板、预先填充变量并验证输入，从而简化了与语言模型的交互。
