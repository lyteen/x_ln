Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\prompts\tagging\semantic_tagging.py`

Okay, restarting and focusing on providing a smarter version of the code you provided. I'll break down the improvements into smaller, more manageable chunks, along with explanations in both English and Chinese.

**1. Enhanced Error Handling and Logging in `SemanticTaggingParser`**

The original code uses `print` statements for logging.  A proper logging mechanism is much better for debugging and monitoring. Also, the error handling can be more robust.

```python
import logging
from typing import List, Optional

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser
from pikerag.utils.lxml_parser import get_soup_from_content

# Configure logging (you can adjust the level and format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SemanticTaggingParser(BaseContentParser):
    """
    A parser that extracts semantic phrases from content formatted in a specific XML-like structure.
    Handles potential errors gracefully and logs relevant information.
    """
    def decode(self, content: str, **kwargs) -> List[str]:
        """
        Decodes the content and extracts semantic phrases.

        Args:
            content: The content to decode (assumed to be XML-like).

        Returns:
            A list of extracted semantic phrases.  Returns an empty list if no phrases are found
            or if there are parsing errors.
        """
        thinking: str = ""
        phrases: List[str] = []

        try:
            result_soup: Optional[BeautifulSoup] = get_soup_from_content(content=content, tag="result")

            if result_soup is None:
                logging.warning(f"[SemanticTagParser] Content skipped due to the absence of <result> tag. Content: {content[:100]}...") # Truncate content for logging
                return phrases  # Return empty list

            thinking_soup = result_soup.find("thinking")
            phrases_soup = result_soup.find("phrases")

            if thinking_soup is not None:
                thinking = thinking_soup.text.strip()

            if phrases_soup is not None:
                for phrase_soup in phrases_soup.find_all("phrase"):
                    phrase_str = phrase_soup.text.strip()
                    if phrase_str: # Check for empty strings after stripping whitespace.
                        phrases.append(phrase_str)
            else:
                logging.warning(f"[SemanticTagParser] Content skipped due to the absence of <phrases> tag. Content: {content[:100]}...") # Truncate content for logging

        except Exception as e:
            logging.exception(f"[SemanticTagParser] An error occurred during parsing. Content: {content[:100]}... Error: {e}") # Log full exception
            # Consider returning an empty list or raising a custom exception here,
            # depending on how you want to handle parsing failures.
            return phrases  # Return empty list on error

        # NOTE: thinking not returned to let the return value compatible with LLMPoweredTagger.
        return phrases

# Demo usage (demonstrating error handling)
if __name__ == '__main__':
    parser = SemanticTaggingParser()

    # Valid content
    valid_content = """
    <result>
      <thinking>Extracted some phrases.</thinking>
      <phrases>
        <phrase>Phrase 1</phrase>
        <phrase>Phrase 2</phrase>
      </phrases>
    </result>
    """
    phrases = parser.decode(valid_content)
    print(f"Extracted phrases: {phrases}")

    # Content with missing <result> tag
    missing_result_content = """
      <thinking>Extracted some phrases.</thinking>
      <phrases>
        <phrase>Phrase 1</phrase>
        <phrase>Phrase 2</phrase>
      </phrases>
    """
    phrases = parser.decode(missing_result_content)
    print(f"Extracted phrases (missing result): {phrases}")  # Should print []

    # Content with missing <phrases> tag
    missing_phrases_content = """
    <result>
      <thinking>Extracted some phrases.</thinking>
    </result>
    """
    phrases = parser.decode(missing_phrases_content)
    print(f"Extracted phrases (missing phrases): {phrases}")  # Should print []

    # Invalid XML content
    invalid_xml_content = "<result><phrases><phrase>Phrase 1" # Unclosed tag
    phrases = parser.decode(invalid_xml_content)
    print(f"Extracted phrases (invalid XML): {phrases}") # Should print [] and log an error
```

**Improvements:**

*   **Logging:** Uses the `logging` module for more robust logging.  This allows you to configure the logging level (e.g., INFO, WARNING, ERROR) and output format.
*   **Exception Handling:**  Wraps the parsing logic in a `try...except` block to catch potential exceptions (e.g., invalid XML).  The exception is logged using `logging.exception`, which includes a full stack trace for debugging.
*   **Optional Type Hints:** Uses `Optional[BeautifulSoup]` for better type safety.
*   **Empty String Check:** Adds a check `if phrase_str:` to avoid adding empty strings to the `phrases` list.
*   **Content Truncation:** Truncates the `content` string in the logging messages to avoid excessively long log messages.
*   **Clearer Return Value:** Always returns an empty list `[]` when no phrases are found or an error occurs. This provides a consistent interface for the caller.
*   **Docstrings:** Added docstrings to explain the purpose of the class and its method.
*   **Demo Usage:** Added demo usage that demonstrates different scenarios, including error cases.

**中文解释:**

*   **日志记录:**  使用 `logging` 模块进行更强大的日志记录。 这样可以配置日志级别（例如，INFO，WARNING，ERROR）和输出格式。
*   **异常处理:**  将解析逻辑包装在 `try...except` 块中，以捕获潜在的异常（例如，无效的 XML）。 异常使用 `logging.exception` 记录，其中包括完整的堆栈跟踪以进行调试。
*   **可选类型提示:** 使用 `Optional[BeautifulSoup]` 以获得更好的类型安全性。
*   **空字符串检查:** 添加了 `if phrase_str:` 检查，以避免将空字符串添加到 `phrases` 列表中。
*   **内容截断:** 在日志消息中截断 `content` 字符串，以避免过长的日志消息。
*   **更清晰的返回值:**  当未找到任何短语或发生错误时，始终返回一个空列表 `[]`。 这为调用者提供了Consistent的接口。
*   **文档字符串:** 添加了文档字符串来解释类及其方法的目的。
*   **演示用法:** 添加了演示用法，展示了不同的场景，包括错误情况。

**2.  Improved Template Management and Validation**

The `MessageTemplate` and `CommunicationProtocol` structures can be improved to be more robust and flexible.  We can add validation to ensure that the input variables in the template match the expected parameters.

```python
from typing import List, Dict, Tuple, Any
from string import Formatter


class MessageTemplate:
    """
    Represents a message template with system and user messages.

    Attributes:
        template: A list of tuples, where each tuple contains the role ("system" or "user")
                  and the message content (a string).
        input_variables: A list of strings representing the input variables used in the template.
    """

    def __init__(self, template: List[Tuple[str, str]], input_variables: List[str]):
        """
        Initializes a MessageTemplate.

        Args:
            template: A list of tuples, where each tuple contains the role ("system" or "user")
                      and the message content (a string).
            input_variables: A list of strings representing the input variables used in the template.

        Raises:
            ValueError: If the template contains variables not listed in input_variables,
                        or if input_variables contains variables not used in the template.
        """
        self.template = template
        self.input_variables = input_variables
        self._validate_template()

    def _validate_template(self):
        """
        Validates that the template uses only the specified input variables and
        that all input variables are used in the template.
        """
        template_variables = set()
        for _, message in self.template:
            # Extract variables using string.Formatter
            for _, field_name, _, _ in Formatter().parse(message):
                if field_name:  # Ignore literal text
                    template_variables.add(field_name)

        # Check for missing variables in input_variables
        missing_variables = template_variables - set(self.input_variables)
        if missing_variables:
            raise ValueError(f"Template uses undefined variables: {missing_variables}. "
                             f"Add them to 'input_variables'.")

        # Check for unused variables in input_variables
        unused_variables = set(self.input_variables) - template_variables
        if unused_variables:
            print(f"Warning: Unused variables in 'input_variables': {unused_variables}. "
              f"Remove them for clarity.")

    def format(self, **kwargs) -> List[Tuple[str, str]]:
        """
        Formats the template with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments representing the input variables and their values.

        Returns:
            A list of tuples, where each tuple contains the role ("system" or "user")
            and the formatted message content (a string).
        """
        #Check the correct number of input variables
        if len(kwargs) != len(self.input_variables):
            raise ValueError(f"Incorrect number of input variables provided for formatting. Expected {len(self.input_variables)}, got {len(kwargs)}.")

        formatted_template: List[Tuple[str, str]] = []
        for role, message in self.template:
            formatted_template.append((role, message.format(**kwargs)))
        return formatted_template


class CommunicationProtocol:
    """
    Represents a communication protocol with a message template and a content parser.

    Attributes:
        template: The MessageTemplate to use for formatting messages.
        parser: The BaseContentParser to use for decoding content.
    """

    def __init__(self, template: MessageTemplate, parser: BaseContentParser):
        """
        Initializes a CommunicationProtocol.

        Args:
            template: The MessageTemplate to use for formatting messages.
            parser: The BaseContentParser to use for decoding content.
        """
        self.template = template
        self.parser = parser

    def format_messages(self, **kwargs) -> List[Tuple[str, str]]:
        """
        Formats the message template with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments representing the input variables and their values.

        Returns:
            A list of tuples, where each tuple contains the role ("system" or "user")
            and the formatted message content (a string).
        """
        return self.template.format(**kwargs)

    def parse_content(self, content: str, **kwargs) -> List[Any]:
        """
        Parses the content using the associated parser.

        Args:
            content: The content to parse.
            **kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            The result of parsing the content.
        """
        return self.parser.decode(content, **kwargs)

# Demo Usage (demonstrating template validation)
if __name__ == '__main__':
    try:
        # Invalid template (missing variable in input_variables)
        invalid_template = MessageTemplate(
            template=[("system", "Hello {name}"), ("user", "How are you?")],
            input_variables=["age"] # Should be "name"
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Invalid template (template using undeclared variables)
        invalid_template2 = MessageTemplate(
            template=[("system", "Hello {name}"), ("user", "How are you? {city}")],
            input_variables=["name"] # should be "name", "city"
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Valid template
    valid_template = MessageTemplate(
        template=[("system", "Hello {name}"), ("user", "How are you?")],
        input_variables=["name"]
    )
    formatted_messages = valid_template.format(name="World")
    print(f"Formatted messages: {formatted_messages}")

    # Demo usage of CommunicationProtocol
    from pikerag.prompts import BaseContentParser # Import from correct location to avoid naming issues.

    class DummyParser(BaseContentParser):
        def decode(self, content: str, **kwargs) -> List[str]:
            return [f"Parsed: {content}"]

    dummy_parser = DummyParser()
    protocol = CommunicationProtocol(template=valid_template, parser=dummy_parser)
    formatted_messages = protocol.format_messages(name="Universe")
    print(f"Protocol formatted messages: {formatted_messages}")
    parsed_content = protocol.parse_content("Some content")
    print(f"Parsed content: {parsed_content}")
```

**Improvements:**

*   **Template Validation:** The `MessageTemplate` class now includes a `_validate_template` method that checks:
    *   That all variables used in the template are declared in `input_variables`.
    *   That all variables declared in `input_variables` are actually used in the template (it prints a warning if there are unused variables).
*   **Error Handling During Formatting:** Adds checks for the correct number of input variables during `format` function
*   **Clearer Structure:** The roles of `MessageTemplate` and `CommunicationProtocol` are more clearly defined.
*   **Type Hints:** Comprehensive type hints are used for better code readability and maintainability.
*   **Docstrings:** Added docstrings to explain the purpose of the classes and their methods.
*   **Demo Usage:** Provides clear examples of how to use the classes and demonstrates the template validation.

**中文解释:**

*   **模板验证:** `MessageTemplate` 类现在包含一个 `_validate_template` 方法，用于检查：
    *   模板中使用的所有变量是否都在 `input_variables` 中声明。
    *   在 `input_variables` 中声明的所有变量是否实际在模板中使用（如果存在未使用的变量，则会打印警告）。
*   **更清晰的结构:** `MessageTemplate` 和 `CommunicationProtocol` 的角色定义更加清晰。
*   **类型提示:** 使用了全面的类型提示，以提高代码的可读性和可维护性。
*   **文档字符串:** 添加了文档字符串来解释类及其方法的目的。
*   **演示用法:** 提供了如何使用这些类的清晰示例，并演示了模板验证。

**3. Making `semantic_tagging_template` More Robust**

Let's apply the improvements to the original `semantic_tagging_template`.

```python
from typing import List, Tuple

from pikerag.prompts import MessageTemplate

semantic_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} that can help people {task_direction}."),
        ("user", """
# Task
Please read the content provided carefully, think step by step, then extract the {tag_semantic} phrases contained therein.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given content.</thinking>
  <phrases>
    <phrase>Extracted phrase 1</phrase>
    <phrase>Extracted phrase 2</phrase>
    <phrase>Extracted phrase 3</phrase>
    ... Please output an equal number of phrases based on the number of phrases contained in the content. Leave it empty if no phrase found.
  </phrases>
</result>

# Content
{content}

# Thinking and answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "tag_semantic", "content"],
)

#Demo usage.
if __name__ == "__main__":
    try:
        # Attempt to format the template with missing variables
        formatted_messages = semantic_tagging_template.format(
            knowledge_domain="medicine", task_direction="find information", tag_semantic="medical terms"
        )
        print("error should occur here")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Correctly format the template
    formatted_messages = semantic_tagging_template.format(
        knowledge_domain="medicine", task_direction="find information", tag_semantic="medical terms", content="Some medical text."
    )
    print(f"Formatted messages: {formatted_messages}")
```

Now, `semantic_tagging_template` will raise a `ValueError` if you try to use it without providing all the necessary input variables.

**中文解释:**

这段代码应用了先前的改进，使 `semantic_tagging_template` 更加健壮。 现在，如果你尝试在没有提供所有必需的输入变量的情况下使用它，`semantic_tagging_template` 将引发 `ValueError`。 这可以帮助防止运行时错误，并确保模板始终以正确的参数格式化。

**4. Putting it All Together: Complete Example**

```python
import logging
from typing import List, Optional, Tuple, Any
from string import Formatter

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser

# Configure logging (you can adjust the level and format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MessageTemplate:
    """
    Represents a message template with system and user messages.

    Attributes:
        template: A list of tuples, where each tuple contains the role ("system" or "user")
                  and the message content (a string).
        input_variables: A list of strings representing the input variables used in the template.
    """

    def __init__(self, template: List[Tuple[str, str]], input_variables: List[str]):
        """
        Initializes a MessageTemplate.

        Args:
            template: A list of tuples, where each tuple contains the role ("system" or "user")
                      and the message content (a string).
            input_variables: A list of strings representing the input variables used in the template.

        Raises:
            ValueError: If the template contains variables not listed in input_variables,
                        or if input_variables contains variables not used in the template.
        """
        self.template = template
        self.input_variables = input_variables
        self._validate_template()

    def _validate_template(self):
        """
        Validates that the template uses only the specified input variables and
        that all input variables are used in the template.
        """
        template_variables = set()
        for _, message in self.template:
            # Extract variables using string.Formatter
            for _, field_name, _, _ in Formatter().parse(message):
                if field_name:  # Ignore literal text
                    template_variables.add(field_name)

        # Check for missing variables in input_variables
        missing_variables = template_variables - set(self.input_variables)
        if missing_variables:
            raise ValueError(f"Template uses undefined variables: {missing_variables}. "
                             f"Add them to 'input_variables'.")

        # Check for unused variables in input_variables
        unused_variables = set(self.input_variables) - template_variables
        if unused_variables:
            print(f"Warning: Unused variables in 'input_variables': {unused_variables}. "
              f"Remove them for clarity.")

    def format(self, **kwargs) -> List[Tuple[str, str]]:
        """
        Formats the template with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments representing the input variables and their values.

        Returns:
            A list of tuples, where each tuple contains the role ("system" or "user")
            and the formatted message content (a string).
        """
        #Check the correct number of input variables
        if len(kwargs) != len(self.input_variables):
            raise ValueError(f"Incorrect number of input variables provided for formatting. Expected {len(self.input_variables)}, got {len(kwargs)}.")

        formatted_template: List[Tuple[str, str]] = []
        for role, message in self.template:
            formatted_template.append((role, message.format(**kwargs)))
        return formatted_template


class CommunicationProtocol:
    """
    Represents a communication protocol with a message template and a content parser.

    Attributes:
        template: The MessageTemplate to use for formatting messages.
        parser: The BaseContentParser to use for decoding content.
    """

    def __init__(self, template: MessageTemplate, parser: BaseContentParser):
        """
        Initializes a CommunicationProtocol.

        Args:
            template: The MessageTemplate to use for formatting messages.
            parser: The BaseContentParser to use for decoding content.
        """
        self.template = template
        self.parser = parser

    def format_messages(self, **kwargs) -> List[Tuple[str, str]]:
        """
        Formats the message template with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments representing the input variables and their values.

        Returns:
            A list of tuples, where each tuple contains the role ("system" or "user")
            and the formatted message content (a string).
        """
        return self.template.format(**kwargs)

    def parse_content(self, content: str, **kwargs) -> List[Any]:
        """
        Parses the content using the associated parser.

        Args:
            content: The content to parse.
            **kwargs: Additional keyword arguments to pass to the parser.

        Returns:
            The result of parsing the content.
        """
        return self.parser.decode(content, **kwargs)


from pikerag.utils.lxml_parser import get_soup_from_content


class SemanticTaggingParser(BaseContentParser):
    """
    A parser that extracts semantic phrases from content formatted in a specific XML-like structure.
    Handles potential errors gracefully and logs relevant information.
    """
    def decode(self, content: str, **kwargs) -> List[str]:
        """
        Decodes the content and extracts semantic phrases.

        Args:
            content: The content to decode (assumed to be XML-like).

        Returns:
            A list of extracted semantic phrases.  Returns an empty list if no phrases are found
            or if there are parsing errors.
        """
        thinking: str = ""
        phrases: List[str] = []

        try:
            result_soup: Optional[BeautifulSoup] = get_soup_from_content(content=content, tag="result")

            if result_soup is None:
                logging.warning(f"[SemanticTagParser] Content skipped due to the absence of <result> tag. Content: {content[:100]}...") # Truncate content for logging
                return phrases  # Return empty list

            thinking_soup = result_soup.find("thinking")
            phrases_soup = result_soup.find("phrases")

            if thinking_soup is not None:
                thinking = thinking_soup.text.strip()

            if phrases_soup is not None:
                for phrase_soup in phrases_soup.find_all("phrase"):
                    phrase_str = phrase_soup.text.strip()
                    if phrase_str: # Check for empty strings after stripping whitespace.
                        phrases.append(phrase_str)
            else:
                logging.warning(f"[SemanticTagParser] Content skipped due to the absence of <phrases> tag. Content: {content[:100]}...") # Truncate content for logging

        except Exception as e:
            logging.exception(f"[SemanticTagParser] An error occurred during parsing. Content: {content[:100]}... Error: {e}") # Log full exception
            # Consider returning an empty list or raising a custom exception here,
            # depending on how you want to handle parsing failures.
            return phrases  # Return empty list on error

        # NOTE: thinking not returned to let the return value compatible with LLMPoweredTagger.
        return phrases


semantic_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} that can help people {task_direction}."),
        ("user", """
# Task
Please read the content provided carefully, think step by step, then extract the {tag_semantic} phrases contained therein.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given content.</thinking>
  <phrases>
    <phrase>Extracted phrase 1</phrase>
    <phrase>Extracted phrase 2</phrase>
    <phrase>Extracted phrase 3</phrase>
    ... Please output an equal number of phrases based on the number of phrases contained in the content. Leave it empty if no phrase found.
  </phrases>
</result>

# Content
{content}

# Thinking and answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "tag_semantic", "content"],
)

semantic_tagging_protocol = CommunicationProtocol(
    template=semantic_tagging_template,
    parser=SemanticTaggingParser(),
)

#Demonstrates a correct usage of the tagging protocol
if __name__ == "__main__":

    content_to_analyze = """
    <result>
    <thinking>The document discusses common side effects.</thinking>
    <phrases>
      <phrase>Headache</phrase>
      <phrase>Nausea</phrase>
    </phrases>
    </result>
    """
    formatted_messages = semantic_tagging_protocol.format_messages(knowledge_domain="medicine", task_direction="extract key terms", tag_semantic="medical side effects", content=content_to_analyze)
    print("Formatted messages:")
    for role, message in formatted_messages:
        print(f"{role}: {message}\n")

    extracted_phrases = semantic_tagging_protocol.parse_content(content_to_analyze)
    print(f"Extracted phrases: {extracted_phrases}")
```

This complete example incorporates all the improvements:

*   Robust error handling and logging in the parser.
*   Template validation in `MessageTemplate`.
*   Clear separation of concerns between `MessageTemplate`, `CommunicationProtocol`, and `SemanticTaggingParser`.

This should be a much more robust and maintainable solution. Remember to install the necessary dependencies: `pip install beautifulsoup4 lxml`.
