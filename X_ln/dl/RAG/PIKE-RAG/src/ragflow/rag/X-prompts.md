Lan: `py` From`dl/RAG/PIKE-RAG/src\ragflow\rag\prompts.py`

Okay, restarting with improved versions of the functions. I'll focus on clarity, efficiency, and robustness.

**1. Improved `chunks_format`:**

```python
def chunks_format(reference):
    """
    Formats chunk data extracted from a reference dictionary.

    Args:
        reference (dict): A dictionary containing a 'chunks' key with a list of chunk dictionaries.

    Returns:
        list: A list of dictionaries, each representing a formatted chunk. Returns an empty list if 'chunks' is missing or empty.
    """

    chunks = reference.get("chunks", [])
    formatted_chunks = []

    for chunk in chunks:
        formatted_chunk = {
            "id": chunk.get("chunk_id") or chunk.get("id"),  # Prioritize "chunk_id"
            "content": chunk.get("content") or chunk.get("content_with_weight"), # Prior