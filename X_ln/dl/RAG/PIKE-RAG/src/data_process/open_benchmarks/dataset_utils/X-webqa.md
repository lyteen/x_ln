Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\webqa.py`

Alright, let's start over with a focus on making the code even better and more robust.  I'll generate each piece of code with explanations in both English and Chinese to make it easier to understand. I'll also include a demonstration to show how to use each part.  Let's aim for code that's clean, efficient, and easy to maintain.

**1. Improved `load_raw_data` Function:**

```python
from datasets import Dataset, load_dataset
from typing import Union

def load_raw_data(dataset_name: str = "Stanford/web_questions", split: str = "train") -> Dataset:
    """
    Loads the raw data from the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load.  Defaults to "Stanford/web_questions".
        split (str): The split of the dataset to load (e.g., "train", "validation", "test"). Defaults to "train".

    Returns:
        Dataset: The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset is not found.
    """
    try:
        dataset: Dataset = load_dataset(dataset_name, split=split)
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found. Please ensure it is available.")


# Demo Usage 演示用法
if __name__ == '__main__':
    try:
        dataset = load_raw_data(split="train")  # Load the training split
        print(f"Dataset loaded successfully with {len(dataset)} examples.")
        print(f"First example: {dataset[0]}") # Print first example
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

```

**Description (English):**

This improved `load_raw_data` function now takes the dataset name as an argument, making it more flexible. It also includes error handling for the case where the specified dataset is not found. This makes the function more robust and prevents unexpected crashes.  It also now displays the first example for quick inspection.

**Description (中文):**

这个改进的 `load_raw_data` 函数现在接受数据集名称作为参数，使其更具灵活性。 它还包括错误处理，以防止找不到指定数据集的情况。 这使函数更加健壮，并防止意外崩溃。 现在还显示第一个例子以进行快速检查。

**Key Improvements:**

*   **Flexibility:**  Accepts `dataset_name` as an argument.
*   **Error Handling:**  Includes a `try-except` block to catch `FileNotFoundError`.
*   **Clarity:** Improved docstrings for better understanding.
*   **Testability:** Simple `if __name__ == '__main__'` block for demonstration and testing.
*   **Information:** Shows length and first example after loading dataset successfully.

**2. Enhanced `format_raw_data` Function:**

```python
import uuid
from typing import Optional, Dict, Any, List
from data_process.utils.question_type import infer_question_type  # Assuming this exists

def format_raw_data(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Re-formats the raw data to fit the desired dataset protocol.

    Args:
        raw (Dict[str, Any]): The raw data dictionary.

    Returns:
        Optional[Dict[str, Any]]: The formatted data dictionary, or None if formatting fails.
    """
    try:
        # Extract the answers and infer the question type.
        answers: List[str] = raw.get("answers", [])
        if not answers:
            print(f"Warning: No answers found for question: {raw.get('question', 'N/A')}. Skipping.")
            return None # Skip if there are no answers

        qtype: str = infer_question_type(answers)

        # Extract the URL and create the supporting fact.
        url: str = raw.get("url", "")
        title: str = url.split("/")[-1].replace("_", " ") if url else "N/A"

        # Create the formatted data dictionary.
        formatted_data: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "question": raw.get("question", "N/A"),
            "answer_labels": answers,
            "question_type": qtype,
            "metadata": {
                "supporting_facts": [
                    {
                        "type": "wikipedia",
                        "title": title,
                    }
                ]
            },
        }
        return formatted_data
    except Exception as e:
        print(f"Error formatting data: {e}.  Raw data: {raw}")
        return None # Return None on any error

# Demo Usage 演示用法
if __name__ == '__main__':
    # Create a sample raw data dictionary.
    raw_data = {
        "question": "What is the capital of France?",
        "answers": ["Paris"],
        "url": "https://en.wikipedia.org/wiki/Paris",
    }

    # Format the raw data.
    formatted_data = format_raw_data(raw_data)

    # Print the formatted data.
    if formatted_data:
        print("Formatted Data:")
        print(formatted_data)
    else:
        print("Formatting failed.")
```

**Description (English):**

This improved `format_raw_data` function includes error handling and better data extraction. It now uses `.get()` to safely access values from the `raw` dictionary, providing default values if the keys are missing.  It also checks for empty answer lists and skips those, preventing errors.  A broad `try...except` block catches any formatting errors, logs them, and returns `None`.

**Description (中文):**

这个改进的 `format_raw_data` 函数包括错误处理和更好的数据提取。 它现在使用 `.get()` 安全地从 `raw` 字典访问值，如果缺少键则提供默认值。它还检查空的答案列表并跳过它们，防止错误。 一个宽泛的 `try...except` 块捕获任何格式化错误，记录它们，并返回 `None`。

**Key Improvements:**

*   **Safeguarding:** Uses `.get()` with default values to prevent `KeyError` exceptions.
*   **Empty Answer Handling:** Checks for empty `answers` lists and skips the entry.
*   **Comprehensive Error Handling:** Includes a `try...except` block to catch any formatting errors.
*   **Logging:** Prints the error message and raw data to help with debugging.
*   **Clear Return Value:** Returns `None` if formatting fails.
*   **Type Hints:** Uses type hints for better code readability and maintainability.

**3. Question Type Inference (Example - you'll need your actual `infer_question_type`):**

Since I don't have your `data_process.utils.question_type.infer_question_type`, I'll provide a VERY simple placeholder.  **You MUST replace this with your actual implementation.**

```python
# This is a PLACEHOLDER.  REPLACE WITH YOUR ACTUAL IMPLEMENTATION.
def infer_question_type(answers: List[str]) -> str:
    """
    Infers the question type based on the answers.  THIS IS A PLACEHOLDER.

    Args:
        answers (List[str]): The list of answers.

    Returns:
        str: The inferred question type.
    """
    if len(answers) > 0:
        return "entity"  # Just a placeholder
    else:
        return "unknown"

# Demo Usage - Only if you can run it with data
if __name__ == '__main__':
    print(infer_question_type(["Paris"])) # -> entity
    print(infer_question_type([])) # -> unknown
```

**IMPORTANT:**  The `infer_question_type` function is crucial.  Replace the placeholder with your *actual* implementation. The quality of this function *directly* impacts the quality of your dataset.

**4. Putting it all together (Example Usage):**

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict, Any, List

import uuid
from datasets import Dataset, load_dataset

# Assuming question_type.py is in the same directory as this file
# or the parent directory and is importable as such
try:
    from data_process.utils.question_type import infer_question_type
except ImportError:
    print("Warning: Could not import infer_question_type.  Please ensure it is in the correct location.")

def load_raw_data(dataset_name: str = "Stanford/web_questions", split: str = "train") -> Dataset:
    """Loads the raw data from the specified dataset."""
    try:
        dataset: Dataset = load_dataset(dataset_name, split=split)
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found. Please ensure it is available.")

def format_raw_data(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Re-formats the raw data to fit the desired dataset protocol."""
    try:
        answers: List[str] = raw.get("answers", [])
        if not answers:
            print(f"Warning: No answers found for question: {raw.get('question', 'N/A')}. Skipping.")
            return None

        qtype: str = infer_question_type(answers)

        url: str = raw.get("url", "")
        title: str = url.split("/")[-1].replace("_", " ") if url else "N/A"

        formatted_data: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "question": raw.get("question", "N/A"),
            "answer_labels": answers,
            "question_type": qtype,
            "metadata": {
                "supporting_facts": [
                    {
                        "type": "wikipedia",
                        "title": title,
                    }
                ]
            },
        }
        return formatted_data
    except Exception as e:
        print(f"Error formatting data: {e}.  Raw data: {raw}")
        return None

def process_dataset(dataset_name: str = "Stanford/web_questions", split: str = "train") -> List[Dict[str, Any]]:
    """Loads, formats, and returns the dataset."""
    try:
        raw_dataset: Dataset = load_raw_data(dataset_name, split=split)
        formatted_data: List[Dict[str, Any]] = []
        for raw_example in raw_dataset:
            formatted_example = format_raw_data(raw_example)
            if formatted_example:  # Only append if formatting was successful
                formatted_data.append(formatted_example)

        return formatted_data
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return []

# Demo Usage
if __name__ == '__main__':
    try:
        formatted_dataset = process_dataset()
        if formatted_dataset:
            print(f"Successfully processed {len(formatted_dataset)} examples.")
            print(f"First example: {formatted_dataset[0]}")
        else:
            print("No data processed.")
    except Exception as e:
        print(f"Error: {e}")
```

**Description (English):**

This complete example demonstrates how to load, format, and process a dataset using the functions defined earlier.  It includes error handling and skips examples that cannot be formatted correctly.

**Description (中文):**

这个完整的例子演示了如何使用前面定义的函数加载、格式化和处理数据集。它包括错误处理并跳过无法正确格式化的示例。

**Key improvements**

*   **`process_dataset` Function:** Centralizes the data loading and formatting workflow.
*   **Error Handling:** Robust error handling throughout the process.
*   **Data Integrity:** Skips malformed examples to maintain data integrity.
*   **Modularity:** The code is well-structured into functions, promoting reusability.

This revised approach focuses on creating robust, well-documented, and testable code that is less prone to errors. Make sure you replace the placeholder for `infer_question_type` with your actual implementation!  Also, ensure that the `data_process` module is accessible from where you run this script.  This is *significantly* better than the original.
