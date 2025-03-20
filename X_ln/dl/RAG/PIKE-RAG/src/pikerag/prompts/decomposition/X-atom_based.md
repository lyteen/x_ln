Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\decomposition\atom_based.py`

Alright, restarting from the beginning! I will aim to provide even smarter and more modular code, with clear explanations in both English and Chinese.

**1. Enhanced Context Handling Utility Function**

```python
from typing import List
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo

def atom_infos_to_context_string(chosen_atom_infos: List[AtomRetrievalInfo], limit: int = 80000) -> str:
    """
    Converts a list of AtomRetrievalInfo objects into a context string.

    Args:
        chosen_atom_infos: A list of AtomRetrievalInfo objects.
        limit: The maximum length of the context string.

    Returns:
        A string containing the concatenated context.
    """

    context: str = ""
    chunk_id_set = set()

    for info in chosen_atom_infos:
        if info.source_chunk_id in chunk_id_set:
            continue  # Avoid duplicate chunks
        chunk_id_set.add(info.source_chunk_id)

        if info.source_chunk_title is not None:
            context += f"\nTitle: {info.source_chunk_title}. Content: {info.source_chunk}\n"
        else:
            context += f"\n{info.source_chunk}\n"

        if len(context) >= limit:
            break  # Stop when the limit is reached

    context = context.strip()
    return context


# Demo Usage 演示用法
if __name__ == '__main__':
    class MockAtomRetrievalInfo:  # Create a mock class for demonstration
        def __init__(self, source_chunk_id, source_chunk_title, source_chunk):
            self.source_chunk_id = source_chunk_id
            self.source_chunk_title = source_chunk_title
            self.source_chunk = source_chunk

    # Sample Data
    infos = [
        MockAtomRetrievalInfo(1, "Title 1", "Content 1"),
        MockAtomRetrievalInfo(2, None, "Content 2"),
        MockAtomRetrievalInfo(1, "Title 1", "Content 1"), # Duplicate ID
    ]

    context_str = atom_infos_to_context_string(infos)
    print(context_str)
```

**Description:**

This function takes a list of `AtomRetrievalInfo` objects and creates a context string. It prevents duplicate chunks based on `source_chunk_id` and truncates the context if it exceeds the specified `limit`.

**中文描述:**

这个函数接收一个 `AtomRetrievalInfo` 对象的列表，并创建一个上下文字符串。它会根据 `source_chunk_id` 避免重复的块，并且如果上下文超过指定的 `limit`，则会截断上下文。

**Improvements:**

*   **Clearer comments and docstrings.**
*   **Duplicate chunk prevention using `chunk_id_set`.**
*   **Includes a demo with a mock class.**

---

**2. Refactored Question Decomposition Prompt & Parser**

```python
from typing import List, Tuple, Dict
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.utils.json_parser import parse_json
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol

class QuestionDecompositionParser(BaseContentParser):
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, dict]:
        context = atom_infos_to_context_string(chosen_atom_infos)
        return content, {"chosen_context": context}

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, List[str]]:
        try:
            output = parse_json(content)
            thinking: str = output["thinking"]
            sub_questions = output["sub_questions"]
            return len(sub_questions) > 0, thinking, sub_questions
        except Exception as e:
            print(f"[QuestionDecompositionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", []


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


question_decompose_protocol = CommunicationProtocol(
    template=question_decomposition_template,
    parser=QuestionDecompositionParser(),
)
# Demo Usage 演示用法
if __name__ == '__main__':
    # Mock AtomRetrievalInfo
    class MockAtomRetrievalInfo:  # Create a mock class for demonstration
        def __init__(self, source_chunk_id, source_chunk_title, source_chunk):
            self.source_chunk_id = source_chunk_id
            self.source_chunk_title = source_chunk_title
            self.source_chunk = source_chunk
    # Sample Data
    infos = [
        MockAtomRetrievalInfo(1, "Title 1", "Content 1"),
        MockAtomRetrievalInfo(2, None, "Content 2"),
        MockAtomRetrievalInfo(1, "Title 1", "Content 1"), # Duplicate ID
    ]
    question = "What is the capital of France?"
    parser = QuestionDecompositionParser()
    content, supplementary = parser.encode(question, infos)
    # Simulate LLM Output
    llm_output = """
    {
        "thinking": "The question asks about the capital of France.  I need to ensure I have up-to-date geographic knowledge.",
        "sub_questions": ["What is the current political structure of France?", "What is the largest city in France?", "Where is France located?"]
    }
    """
    success, thinking, sub_questions = parser.decode(llm_output)

    print(f"Success: {success}")
    print(f"Thinking: {thinking}")
    print(f"Sub Questions: {sub_questions}")
```

**Description:**

This code defines the prompt template, a parser class, and a communication protocol for question decomposition. It encodes the user question and context into a prompt and decodes the LLM's response (thinking process and sub-questions) from JSON format.

**中文描述:**

这段代码定义了问题分解的提示模板、解析器类和通信协议。 它将用户问题和上下文编码成提示，并从 JSON 格式解码 LLM 的响应（思维过程和子问题）。

**Improvements:**

*   **Clearer code structure**: Separating the message template and parser into distinct sections.
*   **Complete and Runnable Demo:** A comprehensive demonstration of how to use the `QuestionDecompositionParser` within a realistic workflow.
*   **Type hints are included for increased clarity.**

---

**3. Refactored Atom Question Selection Prompt & Parser**

```python
from typing import List, Tuple
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.utils.json_parser import parse_json

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant on question answering."


class AtomQuestionSelectionParser(BaseContentParser):
    def __init__(self) -> None:
        super().__init__()
        self._atom_info_candidates: List[AtomRetrievalInfo] = []

    def encode(
        self, content: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo], **kwargs,
    ) -> Tuple[str, dict]:
        context = atom_infos_to_context_string(chosen_atom_infos)

        atom_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            atom_list_str += f"Question {i + 1}: {info.atom}\n"

        self._atom_info_candidates = atom_info_candidates

        return content, {
            "num_atoms": len(atom_info_candidates),
            "chosen_context": context,
            "atom_list_str": atom_list_str,
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, AtomRetrievalInfo]:
        try:
            output = parse_json(content)
            thinking: str = output["thinking"]
            question_idx = output["question_idx"]
            if question_idx is not None and question_idx > 0 and question_idx <= len(self._atom_info_candidates):
                chosen_info = self._atom_info_candidates[question_idx - 1]
                return True, thinking, chosen_info
            else:
                return False, thinking, None
        except Exception as e:
            print(f"[AtomQuestionSelectionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", None


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
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)

atom_question_selection_protocol = CommunicationProtocol(
    template=atom_question_selection_template,
    parser=AtomQuestionSelectionParser(),
)
# Demo Usage 演示用法
if __name__ == '__main__':
    # Mock AtomRetrievalInfo
    class MockAtomRetrievalInfo:
        def __init__(self, atom, source_chunk_id, source_chunk_title, source_chunk):
            self.atom = atom
            self.source_chunk_id = source_chunk_id
            self.source_chunk_title = source_chunk_title
            self.source_chunk = source_chunk

    # Sample Data
    question = "What is the relationship between the French Revolution and the Enlightenment?"
    atom_info_candidates = [
        MockAtomRetrievalInfo("What were the key ideas of the Enlightenment?", 1, None, "Enlightenment Content"),
        MockAtomRetrievalInfo("What were the main causes of the French Revolution?", 2, "Causes", "Revolution Content"),
        MockAtomRetrievalInfo("Who were the major figures of the French Revolution?", 3, None, "Figures Content"),
    ]
    chosen_atom_infos = []  # Initially no context

    parser = AtomQuestionSelectionParser()
    content, supplementary = parser.encode(question, atom_info_candidates, chosen_atom_infos)

    # Simulate LLM Output
    llm_output = """
    {
        "thinking": "Understanding the Enlightenment's ideas is crucial before connecting it to the Revolution.  Option 1 is the most relevant starting point.",
        "question_idx": 1
    }
    """

    success, thinking, chosen_info = parser.decode(llm_output)

    print(f"Success: {success}")
    print(f"Thinking: {thinking}")
    if success:
        print(f"Chosen Atom: {chosen_info.atom}")
```

**Description:**

This code defines the prompt template, a parser class, and a communication protocol for selecting relevant atomic questions. It encodes the user question, available atomic questions, and existing context into a prompt. The parser decodes the LLM's response, selecting the appropriate atomic question.

**中文描述:**

这段代码定义了提示模板、解析器类和通信协议，用于选择相关的原子问题。 它将用户问题、可用的原子问题和现有上下文编码成提示。 解析器解码 LLM 的响应，选择合适的原子问题。

**Improvements:**

*   **More comprehensive demo**: The demo now includes a more elaborate scenario with atomic question candidates and simulates a complete LLM interaction.
*   **Clearer separation of concerns**: The `encode` and `decode` methods are well-defined, focusing on their specific responsibilities.

---

**4. Chunk Selection Prompt & Parser**

```python
from typing import List, Tuple
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.utils.json_parser import parse_json

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant on question answering."


class ChunkSelectionParser(BaseContentParser):
    def __init__(self) -> None:
        super().__init__()
        self._atom_info_candidates: List[AtomRetrievalInfo] = []

    def encode(
        self, content: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo], **kwargs,
    ) -> Tuple[str, dict]:
        context = atom_infos_to_context_string(chosen_atom_infos)

        chunk_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            if info.source_chunk_title is not None:
                chunk_list_str += f"Paragraph {i + 1}: Title: {info.source_chunk_title}. Content: {info.source_chunk}\n"
            else:
                chunk_list_str += f"Paragraph {i + 1}: {info.source_chunk}\n"

        self._atom_info_candidates = atom_info_candidates

        return content, {
            "num_chunks": len(atom_info_candidates),
            "chosen_context": context,
            "chunk_list_str": chunk_list_str,
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, AtomRetrievalInfo]:
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
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)

chunk_selection_protocol = CommunicationProtocol(
    template=chunk_selection_template,
    parser=ChunkSelectionParser(),
)

# Demo Usage 演示用法
if __name__ == '__main__':
    # Mock AtomRetrievalInfo
    class MockAtomRetrievalInfo:
        def __init__(self, source_chunk, source_chunk_title, source_chunk_id):
            self.source_chunk = source_chunk
            self.source_chunk_title = source_chunk_title
            self.source_chunk_id = source_chunk_id

    # Sample Data
    question = "What is the role of mitochondria in a cell?"
    atom_info_candidates = [
        MockAtomRetrievalInfo("Mitochondria are the powerhouses of the cell.", "Function", 1),
        MockAtomRetrievalInfo("The cell membrane protects the cell.", "Cell Membrane", 2),
        MockAtomRetrievalInfo("Ribosomes are responsible for protein synthesis.", "Ribosomes", 3),
    ]
    chosen_atom_infos = []  # Initially no context

    parser = ChunkSelectionParser()
    content, supplementary = parser.encode(question, atom_info_candidates, chosen_atom_infos)

    # Simulate LLM Output
    llm_output = """
    {
        "thinking": "The question is about mitochondria's role. Paragraph 1 directly addresses this.",
        "paragraph_idx": 1
    }
    """

    success, thinking, chosen_info = parser.decode(llm_output)

    print(f"Success: {success}")
    print(f"Thinking: {thinking}")
    if success:
        print(f"Chosen Chunk: {chosen_info.source_chunk}")
```

**Description:**

This code defines a prompt template, parser class, and communication protocol for selecting the most relevant chunk of information.  It is very similar in structure to the Atom Question Selection code, but focuses on selecting chunks.

**中文描述:**

这段代码定义了一个提示模板、解析器类和通信协议，用于选择最相关的信息块。它的结构与原子问题选择代码非常相似，但侧重于选择信息块。

**Improvements:**

*   **Consistent structure:** Maintains a consistent code structure for easier understanding and maintainability.
*   **Realistic demo scenario:** The demonstration utilizes a relevant question and candidate chunks.

---

**5. Context-Aware Question Answering Prompt & Parser**

```python
from typing import Dict, List, Tuple
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.prompts.qa.generation import generation_qa_with_reference_template, GenerationQaParser

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant on question answering."


class ContextQaParser(GenerationQaParser):
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, Dict]:
        _, supplementary =  super().encode(content, **kwargs)

        context_if_any = ""
        if len(chosen_atom_infos) > 0:
            context_if_any = atom_infos_to_context_string(chosen_atom_infos)
        supplementary["context_if_any"] = context_if_any

        return content, supplementary

    # Dummy decode method, replace with actual implementation
    def decode(self, content: str, **kwargs) -> Tuple[bool, str]:
        # Replace this with your actual answer extraction logic from LLM output
        return True, content

final_qa_protocol = CommunicationProtocol(
    template=generation_qa_with_reference_template,
    parser=ContextQaParser(),
)

# Demo Usage 演示用法
if __name__ == '__main__':
    # Mock AtomRetrievalInfo
    class MockAtomRetrievalInfo:
        def __init__(self, source_chunk, source_chunk_title, source_chunk_id):
            self.source_chunk = source_chunk
            self.source_chunk_title = source_chunk_title
            self.source_chunk_id = source_chunk_id

    # Sample Data
    question = "Explain the role of mitochondria in cellular respiration."
    chosen_atom_infos = [
        MockAtomRetrievalInfo("Mitochondria are the primary site of cellular respiration.", "Function", 1),
        MockAtomRetrievalInfo("Cellular respiration produces ATP.", "ATP", 2),
    ]

    parser = ContextQaParser()
    content, supplementary = parser.encode(question, chosen_atom_infos)

    # Simulate LLM Output
    llm_output = "Mitochondria are the powerhouses of the cell and the primary site of cellular respiration, producing ATP."

    success, answer = parser.decode(llm_output)

    print(f"Success: {success}")
    print(f"Answer: {answer}")
```

**Description:**

This code defines the final QA component.  It encodes the question and chosen context (from the previous steps) into a final prompt. It leverages existing `generation_qa_with_reference_template` and `GenerationQaParser`  but overrides the `encode` method to add the context. The `decode` method is a placeholder and requires actual implementation of answer extraction from the LLM output.

**中文描述:**

这段代码定义了最终的 QA 组件。 它将问题和选择的上下文（来自前面的步骤）编码到最终提示中。 它利用现有的 `generation_qa_with_reference_template` 和 `GenerationQaParser`，但覆盖 `encode` 方法以添加上下文。 `decode` 方法是一个占位符，需要从 LLM 输出中实际提取答案。

**Improvements:**

*   **Leverages Existing Code:** Reuses the existing `generation_qa_with_reference_template` and `GenerationQaParser` to minimize code duplication and maintain consistency.
*   **Clear Separation of Concerns:**  The `encode` method is responsible for adding context, while the (placeholder) `decode` method should handle answer extraction.
*   **Demonstration:** The example demonstrates how to use the ContextQaParser.

**Key Considerations and Next Steps:**

*   **Error Handling:** Implement robust error handling in all parser classes, including more specific exception handling and logging.  Consider using `try...except` blocks more strategically.
*   **Input Validation:** Add input validation to the `encode` methods to ensure that the input data is in the expected format.
*   **LLM Integration:**  Replace the placeholder `decode` methods with actual implementations that extract answers from LLM outputs, handle different LLM response formats, and potentially include confidence scores.
*   **Modularity and Reusability:**  Continue to strive for greater modularity and reusability by creating helper functions and classes that can be shared across different components.
*   **Asynchronous Operations:**  For improved performance, consider using asynchronous operations for LLM calls and other potentially time-consuming tasks.
*   **Testing:** Implement comprehensive unit tests to ensure that all components are working correctly.

This revised version provides more modular, maintainable, and robust code with detailed explanations and demonstrations. Remember to replace the placeholder `decode` method in the `ContextQaParser` with your actual answer extraction logic.
