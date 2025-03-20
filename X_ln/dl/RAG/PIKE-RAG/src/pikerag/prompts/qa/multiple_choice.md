Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\qa\multiple_choice.py`

好的，重新开始，我会详细解释并提供中文描述和演示。

**1. MultipleChoiceQaParser:**

```python
import traceback
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser
from pikerag.utils.lxml_parser import get_soup_from_content

class MultipleChoiceQaParser(BaseContentParser):
    def __init__(self) -> None:
        self.option_masks: List[str] = []
        self.options: Dict[str, str] = {}

    def encode(self, content: str, options: Dict[str, str], answer_mask_labels: List[str], **kwargs) -> Tuple[str, dict]:
        self.option_masks = sorted(list(options.keys()))
        self.options = options.copy()

        # NOTE: could enable re-ordering method in the future, do remember to check the answer mask as well.
        options_str = "\n".join([f"{key}: {self.options[key]}" for key in self.option_masks])

        for mask_label in answer_mask_labels:
            assert mask_label in self.option_masks, (
                f"Given answer mask label {mask_label}, but no corresponding option provided: {self.option_masks}"
            )

        return content, {"options_str": options_str}

    # TODO: update the decode interface to be Tuple[answer, dict]
    def decode(self, content: str, options: Dict[str, str], **kwargs) -> dict:
        if content is None or content == "":
            return {}

        try:
            result_soup: BeautifulSoup = get_soup_from_content(content, tag="result")
            if result_soup is not None:
                thinking_soup = result_soup.find("thinking")
                answer_soup = result_soup.find("answer")
            else:
                thinking_soup = get_soup_from_content(content, tag="thinking")
                answer_soup = get_soup_from_content(content, "answer")

            if thinking_soup is not None:
                thinking = thinking_soup.text
            else:
                thinking = ""

            if answer_soup is not None:
                mask_soup = answer_soup.find("mask")
                mask = mask_soup.text.strip() if mask_soup is not None else ""
                option_soup = answer_soup.find("option")
                option = option_soup.text.strip() if option_soup is not None else ""
            else:
                mask = ""
                option = ""

            if len(mask) == 1:
                assert mask in self.option_masks, f"choose {mask} from {self.option_masks}\n{content}"
                if option != self.options[mask]:
                    print()
                    print(f"Answer option: [{option}]")
                    print(f"But the Given: [{self.options[mask]}]")
            elif len(mask) == 0:
                print("No mask extracted")
            else:
                print(f"Multiple options chosen: {mask}")

        except Exception as e:
            print("Content:")
            print(content)
            print("Exception")
            print(e)
            traceback.print_exc()
            exit(0)

        return {
            "thinking": thinking,
            "answer": mask,
            "chosen_option": option,
        }

```

**描述:**
这个类 `MultipleChoiceQaParser` 继承自 `BaseContentParser`，用于处理多项选择题的编码和解码。它有两个主要方法：`encode` 和 `decode`。

*   **`__init__`**: 初始化方法，设置了 `option_masks` (选项的标识符列表) 和 `options` (选项标识符到选项内容的字典)。
*   **`encode`**: 编码方法，接收问题内容 `content`、选项字典 `options` 和答案标识符 `answer_mask_labels`。它将选项格式化成字符串 `options_str`，并确保答案标识符在选项中存在。最后，返回问题内容和包含 `options_str` 的字典。
*   **`decode`**: 解码方法，接收模型生成的文本 `content` 和选项字典 `options`。它使用 `BeautifulSoup` 和 `lxml_parser` 来解析文本，提取模型的思考过程 (`thinking`)、选择的答案标识符 (`mask`) 和选项内容 (`option`)。然后，返回包含这些信息的字典。如果在解析过程中发生任何异常，会打印错误信息并退出。

**如何使用:**

```python
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.lxml_parser import get_soup_from_content

# 假设你已经定义了 MultipleChoiceQaParser
# 示例：
options = {"A": "选项A的内容", "B": "选项B的内容", "C": "选项C的内容"}
content = "问题的内容"
answer_mask_labels = ["A"]  # 正确答案的标识符

parser = MultipleChoiceQaParser()
encoded_content, supplementary = parser.encode(content, options, answer_mask_labels)

print("编码后的内容:", encoded_content)
print("补充信息:", supplementary)

# 假设模型生成了以下文本
model_output = """
<result>
  <thinking>经过思考，我认为选项A最合适。</thinking>
  <answer>
    <mask>A</mask>
    <option>选项A的内容</option>
  </answer>
</result>
"""

decoded_output = parser.decode(model_output, options)
print("解码后的输出:", decoded_output)
```

这个例子展示了如何使用 `MultipleChoiceQaParser` 来编码问题和选项，以及如何解码模型生成的文本，提取答案和思考过程。

**2. MultipleChoiceQaWithReferenceParser:**

```python
class MultipleChoiceQaWithReferenceParser(MultipleChoiceQaParser):
    def encode(self, content: str, options: Dict[str, str], answer_mask_labels: List[str], **kwargs) -> Tuple[str, Dict]:
        content, supplementary = super().encode(content, options, answer_mask_labels[0], **kwargs)

        references = kwargs.get("references", [])
        supplementary["references_str"] = "\n".join([reference.strip() for reference in references])

        return content, supplementary
```

**描述:**
`MultipleChoiceQaWithReferenceParser` 继承自 `MultipleChoiceQaParser`，并在编码过程中添加了对参考信息的处理。

*   **`encode`**:  重写了父类的 `encode` 方法。首先调用父类的 `encode` 方法，获取编码后的内容和补充信息。然后，从 `kwargs` 中获取参考信息列表 `references`，将其格式化成字符串 `references_str`，并添加到补充信息字典中。

**如何使用:**

```python
# 假设你已经定义了 MultipleChoiceQaWithReferenceParser

options = {"A": "选项A的内容", "B": "选项B的内容", "C": "选项C的内容"}
content = "问题的内容"
answer_mask_labels = ["A"]
references = ["参考信息1", "参考信息2"]

parser = MultipleChoiceQaWithReferenceParser()
encoded_content, supplementary = parser.encode(content, options, answer_mask_labels, references=references)

print("编码后的内容:", encoded_content)
print("补充信息:", supplementary)
```

这个例子展示了如何在编码过程中添加参考信息。这些参考信息可以帮助模型更好地回答问题。

**3. CommunicationProtocol和MessageTemplate:**

```python
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate

multiple_choice_qa_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option from the given options, the chosen option should be correct and the most suitable one to answer the given question. If you don't have sufficient data to determine, randomly choose one option from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask. Please note that only one single mask is allowable.</mask>
    <option>The option detail corresponds to the chosen option mask.</option>
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
        ("system", "You are an helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option from the given options, the chosen option should be correct and the most suitable one to answer the given question. You can refer to the references provided when thinking and answering. Please note that the references may or may not be relevant to the question. If you don't have sufficient information to determine, randomly choose one option from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask. Please note that only one single mask is allowable.</mask>
    <option>The option detail corresponds to the chosen option mask.</option>
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

**描述:**

*   **`MessageTemplate`**: `MessageTemplate` 用于定义 LLM 的 prompt 模板。它包含一个 `template` 列表，其中每个元素是一个包含角色和消息内容的元组。`input_variables` 列表指定了模板中需要填充的变量。

*   **`CommunicationProtocol`**: `CommunicationProtocol` 将 `MessageTemplate` 和 `BaseContentParser` 结合起来，定义了与 LLM 进行交互的协议。它指定了用于生成 prompt 的模板和用于解析 LLM 输出的解析器。

**如何使用:**

```python
# 示例：
from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate

# 假设你已经定义了 multiple_choice_qa_template 和 MultipleChoiceQaParser

# 使用 CommunicationProtocol
knowledge_domain = "历史"
content = "二战爆发的时间？"
options = {"A": "1937年", "B": "1939年", "C": "1941年"}

protocol = multiple_choice_qa_protocol # or multiple_choice_qa_with_reference_protocol

prompt = protocol.template.render(knowledge_domain=knowledge_domain, content=content, options_str="\n".join([f"{k}: {v}" for k, v in options.items()]))

print("生成的 Prompt:\n", prompt)

# 假设模型生成了以下文本
model_output = """
<result>
  <thinking>二战爆发于1939年。</thinking>
  <answer>
    <mask>B</mask>
    <option>1939年</option>
  </answer>
</result>
"""

decoded_output = protocol.parser.decode(model_output, options)
print("解码后的输出:", decoded_output)
```

这个例子展示了如何使用 `CommunicationProtocol` 和 `MessageTemplate` 来生成 prompt 并解析模型的输出。通过定义模板和解析器，可以方便地与 LLM 进行交互，并提取所需的信息。

希望这些更详细的解释和示例能帮助你更好地理解这些代码。