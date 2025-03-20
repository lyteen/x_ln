Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\self_ask\self_ask.py`

好的，我们重新开始，并用中文详细解释这段代码，并附上代码片段和使用示例。

**整体介绍**

这段代码实现了一种名为 "Self-Ask" 的问答策略，它允许 AI 助手通过提出一系列的子问题来逐步解决复杂的问题。它包含以下几个主要部分：

1.  **`self_ask_template`**: 定义了 AI 助手进行 Self-Ask 对话的模板。
2.  **`SelfAskParser`**: 用于解析 AI 助手生成的文本，提取后续问题或最终答案。
3.  **`self_ask_protocol`**: 将模板和解析器组合在一起，形成一个完整的通信协议。

**1. `self_ask_template` (消息模板)**

```python
from pikerag.prompts import MessageTemplate

self_ask_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at question-answering."),
        ("user", """
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Are follow up questions needed here: Yes.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
Are follow up questions needed here: No.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Are follow up questions needed here: Yes.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
Are follow up questions needed here: No.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Are follow up questions needed here: Yes.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
Are follow up questions needed here: No.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Are follow up questions needed here: Yes.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Are follow up questions needed here: Yes.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Are follow up questions needed here: Yes.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
Are follow up questions needed here: No.
So the final answer is: No

Question: {content}
{followup_context}
{asking_prefix}
""".strip()),
    ],
    input_variables=["content", "followup_context", "asking_prefix"],
)
```

**描述:**

*   **`MessageTemplate`**:  这是一个用于创建消息模板的类。它允许定义一系列的系统和用户消息，并将它们组合成一个提示，用于指导 AI 助手的行为。
*   **`template`**:  这是一个列表，包含一系列的元组。每个元组的第一个元素是 "system" 或 "user"，表示消息的发送者；第二个元素是消息的内容。
    *   系统消息 `"You are a helpful AI assistant good at question-answering."` 指示 AI 助手扮演的角色。
    *   用户消息包含几个 Self-Ask 问答的例子，这些例子展示了 AI 助手如何提出子问题，获取中间答案，并最终给出最终答案。
    *   最后一个用户消息包含三个占位符：`{content}` (原始问题), `{followup_context}` (之前的问答历史), 和 `{asking_prefix}` (提示 AI 助手是否需要提出更多问题或给出最终答案)。
*   **`input_variables`**:  这是一个列表，包含 `template` 中占位符的名称。

**如何使用:**

这个模板被用来构建发送给 LLM 的提示。通过提供示例，模型可以学习如何进行 self-ask 问答。

**2. `SelfAskParser` (解析器)**

```python
import re
from typing import Dict, List, Optional, Tuple
from pikerag.prompts import BaseContentParser

class SelfAskParser(BaseContentParser):
    def __init__(self) -> None:
        self._final_answer_prefix = "Are follow up questions needed here: No.\nSo the final answer is: "
        self._final_answer_pattern = re.compile(r"So the final answer is:(.*)", re.DOTALL)
        self._follow_up_prefix = "Are follow up questions needed here: "
        self._follow_up_pattern = re.compile(r"Follow up:(.*)", re.DOTALL)

        self._ask_final: bool = False

    def encode(
        self, content: str, followup_pairs: List[Tuple[str, str]], ask_followup: bool, ask_final: bool, **kwargs,
    ) -> Tuple[str, Dict]:
        followup_context: str = "\n".join(
            [
                f"Are follow up questions needed here: Yes.\nFollow up: {q}\nIntermediate Answer: {a}"
                for q, a in followup_pairs
            ]
        )

        assert ask_followup != ask_final, f"There should be and should only be one `True` for `ask_followup` and `ask_final`"

        if len(followup_pairs) >= 5 and ask_followup is True:
            ask_followup = False
            ask_final = True

        self._ask_final = ask_final
        if ask_followup is True:
            asking_prefix = self._follow_up_prefix
        else:
            asking_prefix = self._final_answer_prefix

        return content, {
            "followup_context": followup_context,
            "asking_prefix": asking_prefix,
        }

    def decode(self, content: str,  **kwargs) -> Tuple[Optional[str], Optional[str]]:
        if isinstance(content, str):
            content = content.strip()

            if self._ask_final is False:
                follow_up_match = re.search(self._follow_up_pattern, content)
                if follow_up_match is not None:
                    follow_up = follow_up_match.group(1)
                    return None, follow_up

                final_answer_match = re.search(self._final_answer_pattern, content)
                if final_answer_match is not None:
                    final_answer = final_answer_match.group(1)
                    return final_answer, None

            else:
                return content, None

        return None, None
```

**描述:**

*   **`BaseContentParser`**:  这是一个基类，用于定义内容解析器的接口。`SelfAskParser` 继承自这个基类。
*   **`__init__`**:  构造函数，初始化了解析器。
    *   `_final_answer_prefix`:  最终答案的前缀，用于在 AI 助手的输出中识别最终答案。
    *   `_final_answer_pattern`:  用于匹配最终答案的正则表达式。
    *   `_follow_up_prefix`:  后续问题的前缀，用于在 AI 助手的输出中识别后续问题。
    *   `_follow_up_pattern`:  用于匹配后续问题的正则表达式。
    *   `_ask_final`:  一个标志，指示 AI 助手是否应该给出最终答案。
*   **`encode`**:  将输入内容编码成适合 LLM 的格式。
    *   构建 `followup_context`，包含之前的问答历史。
    *   根据 `ask_followup` 和 `ask_final` 的值，设置 `asking_prefix`。
    *   返回编码后的内容和一个包含 `followup_context` 和 `asking_prefix` 的字典。
*   **`decode`**:  解析 LLM 的输出，提取后续问题或最终答案。
    *   使用正则表达式匹配 `_follow_up_pattern` 或 `_final_answer_pattern`。
    *   如果匹配到后续问题，返回 `(None, follow_up)`。
    *   如果匹配到最终答案，返回 `(final_answer, None)`。
    *   如果没有匹配到任何内容，返回 `(None, None)`。

**如何使用:**

*   **`encode`**:  在将问题发送给 LLM 之前，使用 `encode` 方法将问题、问答历史和提示信息组合在一起，生成最终的提示文本。
*   **`decode`**:  在收到 LLM 的输出后，使用 `decode` 方法解析输出，提取后续问题或最终答案。

**3. `self_ask_protocol` (通信协议)**

```python
from pikerag.prompts import CommunicationProtocol

self_ask_protocol = CommunicationProtocol(
    template=self_ask_template,
    parser=SelfAskParser(),
)
```

**描述:**

*   **`CommunicationProtocol`**:  这是一个用于定义通信协议的类。它将消息模板和解析器组合在一起，形成一个完整的协议。
*   **`template`**:  用于生成提示的 `MessageTemplate` 对象。
*   **`parser`**:  用于解析 LLM 输出的 `SelfAskParser` 对象。

**如何使用:**

`self_ask_protocol` 对象可以被用于与 LLM 进行 Self-Ask 对话。它提供了一种方便的方式来编码问题、发送给 LLM、解析 LLM 的输出，并提取后续问题或最终答案。

**整体流程示例**

1.  **提出问题:**  用户提出一个复杂的问题。
2.  **编码问题:**  使用 `self_ask_protocol.parser.encode` 方法将问题、问答历史 (如果存在) 和提示信息组合在一起，生成最终的提示文本。
3.  **发送给 LLM:**  将提示文本发送给 LLM。
4.  **接收输出:**  接收 LLM 的输出。
5.  **解析输出:**  使用 `self_ask_protocol.parser.decode` 方法解析输出，提取后续问题或最终答案。
6.  **重复步骤 2-5:**  如果提取到后续问题，则将其作为新的问题，重复步骤 2-5。
7.  **给出答案:**  如果提取到最终答案，则将其呈现给用户。

**总结**

这段代码实现了一种 Self-Ask 问答策略，它允许 AI 助手通过提出一系列的子问题来逐步解决复杂的问题。它包含消息模板、解析器和通信协议三个主要部分，这些部分共同协作，使得 AI 助手能够有效地进行 Self-Ask 对话。  这种方法特别适用于需要分解成多个步骤才能解决的复杂问题。
