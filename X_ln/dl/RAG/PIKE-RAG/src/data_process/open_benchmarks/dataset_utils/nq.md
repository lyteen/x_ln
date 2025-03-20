Lan: `py` From`dl/RAG/PIKE-RAG/src\data_process\open_benchmarks\dataset_utils\nq.py`

好的，重新开始。我将为你解释这段Python代码，并提供每个关键部分的中文解释和示例。

**1. `clean_text(text: str) -> str`:**

```python
import unicodedata

def clean_text(text: str) -> str:
    normalized_text = unicodedata.normalize('NFKD', text)
    ascii_text = normalized_text.encode('ascii', 'ignore')
    cleaned_text = ascii_text.decode('utf-8')

    return cleaned_text
```

**描述:**  这个函数用于清洗文本数据。它首先将文本标准化为NFKD格式，然后尝试将其转换为ASCII编码，忽略所有无法转换的字符，最后将其解码回UTF-8。这有效地移除了文本中的非ASCII字符，并确保文本格式统一。

**如何使用:**  当你需要处理包含特殊字符或非英语字符的文本数据时，可以使用此函数。例如，从网页抓取的数据通常需要清洗。

**示例:**

```python
text = "Héllo, wørld! 你好，世界！"
cleaned_text = clean_text(text)
print(cleaned_text)  # 输出: Hello, world!
```

**2. `get_answer_labels(html_bytes: bytes, short_answers: List[dict]) -> List[str]`:**

```python
from typing import List
from bs4 import BeautifulSoup

def get_answer_labels(html_bytes: bytes, short_answers: List[dict]) -> List[str]:
    answer_labels: List[str] = []

    for answer in short_answers:
        if len(answer["start_byte"]) != 0 and len(answer["end_byte"]) != 0:
            start, end = int(answer["start_byte"][0]), int(answer["end_byte"][0])
            if start > 0 and end > 0 and start < end:
                evidence: str = html_bytes[start:end].decode()
                soup = BeautifulSoup(evidence, "html.parser")
                evidence = clean_text(soup.get_text())
                answer_labels.append(evidence)

    return answer_labels
```

**描述:**  这个函数从HTML字节数据中提取答案标签。它遍历`short_answers`列表，每个`short_answers`包含了答案在HTML文档中的起始和结束字节位置。然后，它使用`BeautifulSoup`解析HTML片段，并使用`clean_text`函数清洗提取的文本。

**如何使用:**  当你有包含答案位置信息的HTML数据时，可以使用此函数提取并清洗答案文本。

**示例:**

```python
html_content = b"<html><body><p>The answer is <b>42</b>.</p></body></html>"
short_answers = [{"start_byte": [26], "end_byte": [30]}]  # 假设答案 "42" 的位置

answer_labels = get_answer_labels(html_content, short_answers)
print(answer_labels)  # 输出: ['42']
```

**3. `get_evidence_contents(html_bytes: bytes, long_answer: List[dict]) -> str`:**

```python
from bs4 import BeautifulSoup

def get_evidence_contents(html_bytes: bytes, long_answer: List[dict]) -> str:
    contents = ""

    start, end = int(long_answer[0]["start_byte"]), int(long_answer[0]["end_byte"])
    if start > 0 and end > 0 and start < end:
        evidence: str = html_bytes[start:end].decode()
        soup = BeautifulSoup(evidence, "html.parser")
        evidence = clean_text(soup.get_text())
        contents = evidence

    return contents
```

**描述:** 这个函数与`get_answer_labels`类似，但它提取的是更长的文本证据，通常是包含答案的整个段落或文章。它从`long_answer`列表中的第一个元素获取起始和结束字节位置，提取HTML片段，并使用`BeautifulSoup`解析和清洗文本。

**如何使用:**  当你需要提取包含答案的上下文信息时，可以使用此函数。

**示例:**

```python
html_content = b"<html><body><p>The answer is <b>42</b>. This is the evidence.</p></body></html>"
long_answer = [{"start_byte": [0], "end_byte": [60]}]  # 假设整个段落是证据

evidence_contents = get_evidence_contents(html_content, long_answer)
print(evidence_contents)  # 输出: The answer is 42. This is the evidence.
```

**4. `load_raw_data(dataset_dir: str, split: str) -> Dataset`:**

```python
from datasets import Dataset, load_dataset

def load_raw_data(dataset_dir: str, split: str) -> Dataset:
    dataset: Dataset = load_dataset("google-research-datasets/natural_questions", "default", split=split)
    return dataset
```

**描述:** 这个函数使用`datasets`库加载原始数据集。 在这里，它专门加载Google Natural Questions数据集。`split`参数指定要加载的数据集部分（例如，"train"、"validation"）。

**如何使用:**  这是加载数据集的入口点。 你需要指定数据集的名称和要加载的部分。

**示例:**

```python
# 假设你有一个名为 my_dataset 的目录
dataset = load_raw_data("./my_dataset", "train")
print(dataset)
```

**5. `format_raw_data(raw: dict) -> Optional[dict]`:**

```python
import uuid
from typing import List, Optional

from data_process.utils.question_type import infer_nq_question_type #假设存在

def format_raw_data(raw: dict) -> Optional[dict]:
    # Step 1: parse answer labels and supporting facts, validate this record.
    html_source: str = raw["document"]["html"]
    html_bytes: bytes = html_source.encode()

    answer_labels: List[str] = get_answer_labels(html_bytes, raw["annotations"]["short_answers"])
    if len(answer_labels) == 0:
        return None

    evidence_contents: str = get_evidence_contents(html_bytes, raw["annotations"]["long_answer"])
    if len(evidence_contents) == 0:
        return None

    # Step 2: re-format to fit dataset protocol.
    qtype: str = infer_nq_question_type(answer_labels, raw["annotations"]["yes_no_answer"])

    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"]["text"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw["id"],
            "supporting_facts": [
                {
                    "type": "wikipedia",
                    "title": raw["document"]["title"],
                    "contents": evidence_contents,
                }
            ],
            "original_type": qtype,
        },
    }

    return formatted_data
```

**描述:** 这个函数将原始数据集中的每个样本转换为格式化的数据。它首先使用`get_answer_labels`和`get_evidence_contents`提取答案标签和证据内容。如果提取失败（例如，没有找到答案或证据），则返回`None`。然后，它使用`infer_nq_question_type`函数推断问题类型。最后，它创建一个包含问题、答案标签、问题类型和元数据的字典，并返回该字典。

**如何使用:** 在加载原始数据后，可以使用此函数将每个原始样本转换为更易于使用的格式。

**示例:**

```python
raw_data = {
    "id": "example_id",
    "question": {"text": "What is the answer?"},
    "document": {"title": "Example Document", "html": "<html><body><p>The answer is <b>42</b>.</p></body></html>"},
    "annotations": {
        "short_answers": [{"start_byte": [26], "end_byte": [30]}],
        "long_answer": [{"start_byte": [0], "end_byte": [60]}],
        "yes_no_answer": "NONE"
    }
}

formatted_data = format_raw_data(raw_data)
print(formatted_data)
```

总的来说，这段代码用于从Google Natural Questions数据集中提取和格式化数据，使其更易于用于问答模型训练。关键步骤包括清洗文本，提取答案和证据，推断问题类型，以及将数据组织成一个结构化的格式。

希望这次的解释更清楚！ 让我知道你是否还有其他问题。
