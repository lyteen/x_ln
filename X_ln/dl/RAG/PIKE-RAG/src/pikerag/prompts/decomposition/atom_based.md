Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\decomposition\atom_based.py`

好的，我们重新开始，这次我会更详细地解释每一部分代码，并附上中文描述和使用示例。

**1. `atom_infos_to_context_string` 函数**

```python
from typing import List
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo

def atom_infos_to_context_string(chosen_atom_infos: List[AtomRetrievalInfo], limit: int=80000) -> str:
    """
    将 AtomRetrievalInfo 列表转换为上下文字符串。

    Args:
        chosen_atom_infos: AtomRetrievalInfo 对象的列表，包含检索到的知识片段信息。
        limit: 上下文字符串的最大长度限制，默认为 80000。

    Returns:
        一个字符串，包含所有检索到的知识片段，格式化为 "Title: {title}. Content: {content}" 或 "{content}"。
    """
    context: str = ""
    chunk_id_set = set()  # 用于跟踪已经添加的块ID，避免重复

    for info in chosen_atom_infos:
        if info.source_chunk_id in chunk_id_set:
            continue  # 如果当前块ID已经存在，则跳过
        chunk_id_set.add(info.source_chunk_id)

        if info.source_chunk_title is not None:
            context += f"\nTitle: {info.source_chunk_title}. Content: {info.source_chunk}\n"  # 如果有标题，则添加标题和内容
        else:
            context += f"\n{info.source_chunk}\n"  # 否则只添加内容

        if len(context) >= limit:
            break  # 如果上下文长度超过限制，则停止添加

    context = context.strip()  # 移除首尾空白字符
    return context
```

**描述:**  `atom_infos_to_context_string` 函数接收一个 `AtomRetrievalInfo` 对象列表，并将它们转换成一个可供语言模型使用的上下文字符串。这个函数的主要目的是将检索到的知识片段格式化为易于理解和使用的形式。它会去重，并根据标题是否存在进行不同的格式化。

**如何使用:** 这个函数通常在问答系统中使用，用于将检索到的知识片段作为上下文提供给语言模型，帮助模型更好地回答问题。

**示例:**

```python
# 假设你已经有了一个 AtomRetrievalInfo 对象列表
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
atom_infos = [
    AtomRetrievalInfo(source_chunk="This is the first chunk of information.", source_chunk_id="chunk1"),
    AtomRetrievalInfo(source_chunk="This is the second chunk of information.", source_chunk_title="Second Chunk", source_chunk_id="chunk2"),
    AtomRetrievalInfo(source_chunk="This is a duplicate chunk.", source_chunk_id="chunk1"), # Duplicate Chunk
]

context_string = atom_infos_to_context_string(atom_infos)
print(context_string)
# Expected Output:
# This is the first chunk of information.
# Title: Second Chunk. Content: This is the second chunk of information.
```

**2. `question_decomposition_template` 和 `QuestionDecompositionParser`**

```python
from typing import Dict, List, Tuple
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.utils.json_parser import parse_json


question_decomposition_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to analyse the providing context then raise atomic sub-questions for the knowledge that can help you answer the question better. Think in different ways and raise as many diverse questions as possible.

# Output Format
Please output in following JSON format:
{{
    "thinking": <A string. Your thinking for this task, including analysis to the question and the given context.>,
    "sub_questions": <A list of string. The sub-questions indicating what you need.>
}}

# Context
The context we already have:
{chosen_context}

# Question
{content}

# Your Output:
""".strip()),
    ],
    input_variables=["content", "chosen_context"],
    partial_variables={
        "system_prompt": "You are a helpful AI assistant good at question decomposition.",
    },
)


class QuestionDecompositionParser(BaseContentParser):
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, dict]:
        """
        编码输入，将问题和已选择的上下文转化为模板可以使用的格式。

        Args:
            content: 原始问题内容。
            chosen_atom_infos:  已选择的 AtomRetrievalInfo 列表。
            **kwargs: 其他可选参数。

        Returns:
            包含问题内容和编码后数据的元组。
        """
        context = atom_infos_to_context_string(chosen_atom_infos) #context是一个字符串
        return content, {"chosen_context": context}

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, List[str]]:
        """
        解码模型的输出，提取思维过程和子问题列表。

        Args:
            content:  模型输出的字符串。
            **kwargs: 其他可选参数。

        Returns:
            一个元组，包含解码是否成功、思维过程和子问题列表。
        """
        try:
            output = parse_json(content)

            thinking: str = output["thinking"]
            sub_questions = output["sub_questions"]
            return len(sub_questions) > 0, thinking, sub_questions #如果成功则返回True
        except Exception as e:
            print(f"[QuestionDecompositionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", []


question_decompose_protocol = CommunicationProtocol(
    template=question_decomposition_template,
    parser=QuestionDecompositionParser(),
)
```

**描述:**  这部分代码定义了一个问题分解的流程。`question_decomposition_template` 是一个提示模板，用于指示语言模型将一个复杂的问题分解为更小的、更原子化的子问题。 `QuestionDecompositionParser` 负责将原始问题和检索到的上下文编码成模板可以使用的格式，以及将模型的输出（包含思维过程和子问题列表）解码出来。

**如何使用:**  这个流程通常用于需要多步推理的问答场景中。首先，使用 `QuestionDecompositionParser` 将原始问题和上下文编码成模板可以使用的格式。然后，将编码后的数据传递给语言模型，生成子问题列表。最后，使用 `QuestionDecompositionParser` 将模型的输出解码出来，得到子问题列表。

**示例:**

```python
# 假设你有一个问题和一些 AtomRetrievalInfo 对象
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo

question = "What is the capital of France and what is its population?"
atom_infos = [
    AtomRetrievalInfo(source_chunk="France is a country in Europe.", source_chunk_id="chunk1"),
]

# 使用 QuestionDecompositionParser 编码输入
parser = QuestionDecompositionParser()
encoded_content, supplementary = parser.encode(question, atom_infos)

# 模拟语言模型的输出
model_output = """
{
    "thinking": "The question requires knowing the capital and population of France. I need to find information about both.",
    "sub_questions": ["What is the capital of France?", "What is the population of Paris?"]
}
"""

# 使用 QuestionDecompositionParser 解码输出
success, thinking, sub_questions = parser.decode(model_output)

if success:
    print(f"Thinking: {thinking}")
    print(f"Sub-questions: {sub_questions}")
else:
    print("Decoding failed.")

# Expected output:
# Thinking: The question requires knowing the capital and population of France. I need to find information about both.
# Sub-questions: ['What is the capital of France?', 'What is the population of Paris?']
```

**3. `atom_question_selection_template` 和 `AtomQuestionSelectionParser`**

```python
from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.utils.json_parser import parse_json


atom_question_selection_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to analyse the providing context then decide which sub-questions may be useful to be answered before you can answer the given question. Select a most relevant sub-question from the given question list, avoid selecting sub-question that can already be answered with the given context or with your own knowledge.

# Output Format
Please output in following JSON format:
{{
    "thinking": <A string. Your thinking for this selection task.>,
    "question_idx": <An integer, indicating a sub-question index from 1 to {num_atoms}.>
}}

# Context
The context we already have:
{chosen_context}

# Sub-Questions You Can Choose From
{atom_list_str}

# Question
{content}

# Your output:
""".strip()),
    ],
    input_variables=["content", "num_atoms", "chosen_context", "atom_list_str"],
    partial_variables={
        "system_prompt": "You are a helpful AI assistant on question answering.",
    },
)


class AtomQuestionSelectionParser(BaseContentParser):
    def __init__(self) -> None:
        super().__init__()
        self._atom_info_candidates: List[AtomRetrievalInfo] = []

    def encode(
        self, content: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo], **kwargs,
    ) -> Tuple[str, dict]:
        """
        编码输入，将问题、候选子问题列表和已选择的上下文转化为模板可以使用的格式。

        Args:
            content: 原始问题内容。
            atom_info_candidates: 候选子问题列表 (AtomRetrievalInfo 对象列表)。
            chosen_atom_infos: 已选择的 AtomRetrievalInfo 列表。
            **kwargs: 其他可选参数。

        Returns:
            包含问题内容和编码后数据的元组。
        """
        context = atom_infos_to_context_string(chosen_atom_infos)

        atom_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            atom_list_str += f"Question {i + 1}: {info.atom}\n"

        self._atom_info_candidates = atom_info_candidates # 保存候选问题列表

        return content, {
            "num_atoms": len(atom_info_candidates),
            "chosen_context": context,
            "atom_list_str": atom_list_str,
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, AtomRetrievalInfo]:
        """
        解码模型的输出，提取思维过程和选择的子问题。

        Args:
            content: 模型输出的字符串。
            **kwargs: 其他可选参数。

        Returns:
            一个元组，包含解码是否成功、思维过程和选择的子问题 (AtomRetrievalInfo 对象)。
        """
        try:
            output = parse_json(content)
            thinking: str = output["thinking"]
            question_idx = output["question_idx"] #拿到选择的下标
            if question_idx is not None and question_idx > 0 and question_idx <= len(self._atom_info_candidates):
                chosen_info = self._atom_info_candidates[question_idx - 1]
                return True, thinking, chosen_info #返回被选中的AtomRetrievalInfo
            else:
                return False, thinking, None
        except Exception as e:
            print(f"[AtomQuestionSelectionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", None


atom_question_selection_protocol = CommunicationProtocol(
    template=atom_question_selection_template,
    parser=AtomQuestionSelectionParser(),
)
```

**描述:** 这部分代码定义了一个选择子问题的流程。`atom_question_selection_template` 是一个提示模板，用于指示语言模型从一个候选子问题列表中选择最相关的子问题。`AtomQuestionSelectionParser` 负责将原始问题、候选子问题列表和检索到的上下文编码成模板可以使用的格式，以及将模型的输出（包含思维过程和选择的子问题）解码出来。

**如何使用:** 这个流程通常用于需要从多个子问题中选择一个进行回答的场景中。首先，使用 `AtomQuestionSelectionParser` 将原始问题、候选子问题列表和上下文编码成模板可以使用的格式。然后，将编码后的数据传递给语言模型，选择最相关的子问题。最后，使用 `AtomQuestionSelectionParser` 将模型的输出解码出来，得到选择的子问题。

**示例:**

```python
# 假设你有一个问题、一个候选子问题列表和一些 AtomRetrievalInfo 对象
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo

question = "What is the capital of France and what is its population?"
atom_info_candidates = [
    AtomRetrievalInfo(atom="What is the capital of France?", source_chunk_id="q1"),
    AtomRetrievalInfo(atom="What is the population of Paris?", source_chunk_id="q2"),
    AtomRetrievalInfo(atom="What is the area of France?", source_chunk_id="q3"),
]
atom_infos = [
    AtomRetrievalInfo(source_chunk="France is a country in Europe.", source_chunk_id="chunk1"),
]

# 使用 AtomQuestionSelectionParser 编码输入
parser = AtomQuestionSelectionParser()
encoded_content, supplementary = parser.encode(question, atom_info_candidates, atom_infos)

# 模拟语言模型的输出
model_output = """
{
    "thinking": "The capital of France is a more direct question to start with.",
    "question_idx": 1
}
"""

# 使用 AtomQuestionSelectionParser 解码输出
success, thinking, chosen_info = parser.decode(model_output)

if success:
    print(f"Thinking: {thinking}")
    print(f"Chosen sub-question: {chosen_info.atom}")
else:
    print("Decoding failed.")

# Expected output:
# Thinking: The capital of France is a more direct question to start with.
# Chosen sub-question: What is the capital of France?
```

**4. `chunk_selection_template` 和 `ChunkSelectionParser`**

```python
from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.utils.json_parser import parse_json


chunk_selection_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to analyse the providing context then decide which paragraph in the list may be useful for you to answer the given question. Select a most relevant paragraph from the given paragraph list.

# Output Format
Please output in following JSON format:
{{
    "thinking": <A string. Your thinking for this selection task.>,
    "paragraph_idx": <An integer. A paragraph index from 1 to {num_chunks}.>
}}

# Context
The context we already have:
{chosen_context}

# Paragraph List You Can Choose From
{chunk_list_str}

# Question
{content}

# Your output:
""".strip()),
    ],
    input_variables=["content", "chosen_context", "num_chunks", "chunk_list_str"],
    partial_variables={
        "system_prompt": "You are a helpful AI assistant on question answering.",
    },
)


class ChunkSelectionParser(BaseContentParser):
    def __init__(self) -> None:
        super().__init__()
        self._atom_info_candidates: List[AtomRetrievalInfo] = []

    def encode(
        self, content: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo], **kwargs,
    ) -> Tuple[str, dict]:
        """
        编码输入，将问题、候选知识片段列表和已选择的上下文转化为模板可以使用的格式。

        Args:
            content: 原始问题内容。
            atom_info_candidates: 候选知识片段列表 (AtomRetrievalInfo 对象列表)。
            chosen_atom_infos: 已选择的 AtomRetrievalInfo 列表。
            **kwargs: 其他可选参数。

        Returns:
            包含问题内容和编码后数据的元组。
        """
        context = atom_infos_to_context_string(chosen_atom_infos)

        chunk_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            if info.source_chunk_title is not None:
                chunk_list_str += f"Paragraph {i + 1}: Title: {info.source_chunk_title}. Content: {info.source_chunk}\n"
            else:
                chunk_list_str += f"Paragraph {i + 1}: {info.source_chunk}\n"

        self._atom_info_candidates = atom_info_candidates # 保存候选知识片段列表

        return content, {
            "num_chunks": len(atom_info_candidates),
            "chosen_context": context,
            "chunk_list_str": chunk_list_str,
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, AtomRetrievalInfo]:
        """
        解码模型的输出，提取思维过程和选择的知识片段。

        Args:
            content: 模型输出的字符串。
            **kwargs: 其他可选参数。

        Returns:
            一个元组，包含解码是否成功、思维过程和选择的知识片段 (AtomRetrievalInfo 对象)。
        """
        try:
            output = parse_json(content)
            thinking: str = output["thinking"]
            paragraph_idx = output["paragraph_idx"]
            if paragraph_idx is not None and paragraph_idx > 0 and paragraph_idx <= len(self._atom_info_candidates):
                chosen_info = self._atom_info_candidates[paragraph_idx - 1]
                return True, thinking, chosen_info
            else:
                return False, thinking, None
        except Exception as e:
            print(f"[ChunkSelectionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", None


chunk_selection_protocol = CommunicationProtocol(
    template=chunk_selection_template,
    parser=ChunkSelectionParser(),
)
```

**描述:** 这部分代码定义了一个选择知识片段的流程。`chunk_selection_template` 是一个提示模板，用于指示语言模型从一个候选知识片段列表中选择最相关的片段。`ChunkSelectionParser` 负责将原始问题、候选知识片段列表和检索到的上下文编码成模板可以使用的格式，以及将模型的输出（包含思维过程和选择的知识片段）解码出来。

**如何使用:** 这个流程通常用于需要从多个知识片段中选择一个进行回答的场景中。首先，使用 `ChunkSelectionParser` 将原始问题、候选知识片段列表和上下文编码成模板可以使用的格式。然后，将编码后的数据传递给语言模型，选择最相关的知识片段。最后，使用 `ChunkSelectionParser` 将模型的输出解码出来，得到选择的知识片段。

**示例:**

```python
# 假设你有一个问题、一个候选知识片段列表和一些 AtomRetrievalInfo 对象
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo

question = "What is the capital of France?"
atom_info_candidates = [
    AtomRetrievalInfo(source_chunk="Paris is the capital of France.", source_chunk_id="chunk1"),
    AtomRetrievalInfo(source_chunk="France is a country in Europe.", source_chunk_id="chunk2"),
    AtomRetrievalInfo(source_chunk="The Eiffel Tower is in Paris.", source_chunk_id="chunk3"),
]
atom_infos = [
    AtomRetrievalInfo(source_chunk="France is a country in Europe.", source_chunk_id="chunk4"),
]

# 使用 ChunkSelectionParser 编码输入
parser = ChunkSelectionParser()
encoded_content, supplementary = parser.encode(question, atom_info_candidates, atom_infos)

# 模拟语言模型的输出
model_output = """
{
    "thinking": "The first chunk directly answers the question.",
    "paragraph_idx": 1
}
"""

# 使用 ChunkSelectionParser 解码输出
success, thinking, chosen_info = parser.decode(model_output)

if success:
    print(f"Thinking: {thinking}")
    print(f"Chosen chunk: {chosen_info.source_chunk}")
else:
    print("Decoding failed.")

# Expected output:
# Thinking: The first chunk directly answers the question.
# Chosen chunk: Paris is the capital of France.
```

**5. `ContextQaParser` 和 `final_qa_protocol`**

```python
from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.prompts.qa.generation import generation_qa_with_reference_template, GenerationQaParser
from pikerag.utils.json_parser import parse_json

class ContextQaParser(GenerationQaParser):
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, Dict]:
        """
        编码输入，将问题和已选择的知识片段转化为模板可以使用的格式。

        Args:
            content: 原始问题内容。
            chosen_atom_infos: 已选择的 AtomRetrievalInfo 列表。
            **kwargs: 其他可选参数。

        Returns:
            包含问题内容和编码后数据的元组。
        """
        _, supplementary =  super().encode(content, **kwargs)

        context_if_any = ""
        if len(chosen_atom_infos) > 0:
            context_if_any = atom_infos_to_context_string(chosen_atom_infos)
        supplementary["context_if_any"] = context_if_any #context_if_any是一个字符串

        return content, supplementary


final_qa_protocol = CommunicationProtocol(
    template=generation_qa_with_reference_template,
    parser=ContextQaParser(),
)
```

**描述:** 这部分代码定义了一个最终问答的流程。`ContextQaParser` 继承自 `GenerationQaParser`，并扩展了其功能，可以处理包含上下文信息的问答。它负责将原始问题和已选择的知识片段编码成模板可以使用的格式。  `final_qa_protocol` 定义了整个问答流程，包括模板和解析器。

**如何使用:**  这个流程通常用于在已经选择了相关的知识片段后，生成最终答案的场景中。首先，使用 `ContextQaParser` 将原始问题和已选择的知识片段编码成模板可以使用的格式。然后，将编码后的数据传递给语言模型，生成最终答案。

**示例:**

```python
# 假设你有一个问题和一些 AtomRetrievalInfo 对象
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo

question = "What is the capital of France?"
atom_infos = [
    AtomRetrievalInfo(source_chunk="Paris is the capital of France.", source_chunk_id="chunk1"),
]

# 使用 ContextQaParser 编码输入
parser = ContextQaParser()
encoded_content, supplementary = parser.encode(question, atom_infos)

# 模拟语言模型的输出 (注意，这个例子需要你配置好 `generation_qa_with_reference_template` 和相应的 GenerationQaParser)
model_output = "Paris is the capital of France."

# 由于这里没有实际的解码过程，我们只是模拟最终的答案
final_answer = model_output

print(f"Final Answer: {final_answer}")

# Expected output:
# Final Answer: Paris is the capital of France.
```

**总结:**

这段代码定义了一系列用于知识检索增强问答的流程，包括问题分解、子问题选择、知识片段选择和最终问答。每个流程都包含一个提示模板和一个解析器，用于将原始数据编码成模板可以使用的格式，以及将模型的输出解码出来。 这些流程可以组合使用，构建一个完整的知识检索增强问答系统。

希望这次的解释更详细，更易于理解!
