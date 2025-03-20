Lan: `py` From`dl/RAG/PIKE-RAG/src\ragflow\graphrag\light\graph_prompt.py`

好的，现在我将重新开始，并详细解释代码中的各个部分，附上中文描述，并提供使用示例。

**总览**

这段代码定义了一系列用于提示大型语言模型 (LLM) 的字符串。这些提示旨在用于诸如命名实体识别 (NER)、关系提取和问答等任务。该代码的目标是定义清晰和结构化的提示，以指导LLM的行为并确保一致的输出。

**1. 基础配置:**

```python
PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]
```

*   `PROMPTS = {}`:  创建一个空的字典 `PROMPTS`，用于存储所有的提示信息。
*   `PROMPTS["DEFAULT_LANGUAGE"] = "English"`:  设置默认语言为英语，用于指示LLM的输出语言。
*   `PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"`:  定义元组分隔符，用于分隔实体或关系的不同属性，如名称、类型、描述等。
*   `PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"`:  定义记录分隔符，用于分隔多个实体或关系记录。
*   `PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"`:  定义完成分隔符，用于指示LLM输出的结束。
*   `PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]`: 定义一个列表，包含一些用于指示处理进度的特殊字符。这在长时间运行的任务中可以用于显示动画效果。
*   `PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]`:  定义默认的实体类型列表，用于指导LLM识别和提取的实体类型，包括组织、人物、地理位置、事件和类别。

**如何使用:** 这些基础配置定义了后续提示的格式和行为。例如，在使用 `entity_extraction` 提示时，会用到 `DEFAULT_ENTITY_TYPES` 中的实体类型。

**2. 实体提取提示 (entity_extraction):**

```python
PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
"""
```

*   **描述:** 这个提示旨在指导LLM从给定的文本中提取实体和关系。 它详细说明了LLM应遵循的步骤，包括识别实体、提取关系以及格式化输出。
*   **关键部分:**
    *   `-Goal-`: 明确说明提示的目标，即从文本中识别实体和关系。
    *   `-Steps-`:  详细描述了LLM需要执行的步骤，包括实体识别、关系提取和关键词提取。
    *   `{language}`:  一个占位符，允许指定输出语言。
    *   `{entity_types}`:  一个占位符，将被替换为预定义的实体类型列表。
    *   `{tuple_delimiter}`: 一个占位符，指定用于分隔元组中元素的字符。
    *   `{record_delimiter}`: 一个占位符，指定用于分隔不同记录（例如，不同的实体或关系）的字符。
    *   `{completion_delimiter}`: 一个占位符，用于指示LLM生成的完成。
    *   `{examples}`:  一个占位符，将被替换为一些示例，以帮助LLM理解期望的输出格式。
    *   `{input_text}`:  一个占位符，将被替换为要从中提取实体和关系的实际文本。
*   **如何使用:**  使用这个提示需要将 `{entity_types}`、`{examples}` 和 `{input_text}` 替换为适当的值。 例如，你可以使用 `DEFAULT_ENTITY_TYPES` 中的实体类型，并提供一些示例来展示期望的输出格式。
*   **示例代码:**

    ```python
    entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]
    examples = PROMPTS["entity_extraction_examples"] # 从下面的例子中获取
    input_text = "Apple is planning to open a new store in New York."
    prompt = PROMPTS["entity_extraction"].format(
        language=PROMPTS["DEFAULT_LANGUAGE"],
        entity_types=", ".join(entity_types),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples="\n".join(examples),
        input_text=input_text
    )
    print(prompt)
    #  然后将 prompt 传递给 LLM
    ```

**3. 实体提取示例 (entity_extraction_examples):**

```python
PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################""",
]
```

*   **描述:**  `entity_extraction_examples` 包含了几个示例，展示了期望的输入文本和相应的输出格式。 这些示例对于指导LLM理解任务至关重要。
*   **如何使用:**  这些示例直接用于格式化 `entity_extraction` 提示。  在调用LLM之前，将这些示例插入到提示字符串中。

**4. 循环提取提示 (entiti\_continue\_extraction, entiti\_if\_loop\_extraction):**

```python
PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""
```

*   **描述:** 这两个提示用于迭代式实体提取。  `entiti_continue_extraction` 指示LLM在上一次提取中遗漏了许多实体，并要求LLM以相同的格式添加它们。  `entiti_if_loop_extraction` 询问LLM是否仍然有需要添加的实体，用于判断是否需要继续循环提取。
*   **如何使用:** 这些提示用于创建一个循环，直到所有实体都被提取出来。 首先，使用 `entity_extraction` 提示进行初始提取。 如果 `entiti_if_loop_extraction` 提示返回 "YES"，则使用 `entiti_continue_extraction` 提示再次提取。  重复此过程，直到 `entiti_if_loop_extraction` 返回 "NO"。

**5. RAG提示 (rag\_response, naive\_rag\_response, mix_rag_response):**

```python
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown."""

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown."""

PROMPTS["mix_rag_response"] = """---Role---

You are a professional assistant responsible for answering questions based on knowledge graph and textual information. Please respond in the same language as the user's question.

---Goal---

Generate a concise response that summarizes relevant points from the provided information. If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Data Sources---

1. Knowledge Graph Data:
{kg_context}

2. Vector Data:
{vector_context}

---Response Requirements---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Aim to keep content around 3 paragraphs for conciseness
- Each paragraph should be under a relevant section heading
- Each section should focus on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References", clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (VD)
  Format: [KG/VD] Source content

Add sections and commentary to the response as appropriate for the length and format. If the provided information is insufficient to answer the question, clearly state that you don't know or cannot provide an answer in the same language as the user's question."""
```

*   **描述:** 这些提示旨在用于检索增强生成 (RAG)。 `rag_response` 用于从结构化数据表回答问题，`naive_rag_response` 用于从非结构化文档回答问题, `mix_rag_response` 用于从知识图谱和文本信息中回答问题。  它们都指示LLM根据提供的数据生成答案，避免编造信息，并考虑时间戳来处理冲突的信息。
*   **关键部分:**
    *   `---Role---`: 定义LLM的角色，例如“有帮助的助手”。
    *   `---Goal---`: 描述LLM的目标，例如“根据提供的数据生成答案”。
    *   `{response_type}`:  一个占位符，将被替换为期望的响应类型和长度。
    *   `{context_data}`:  一个占位符，将被替换为用于生成答案的上下文数据。
    *   `{kg_context}`:  知识图谱数据占位符
    *   `{vector_context}`:  向量数据占位符
*   **如何使用:** 使用这些提示需要将 `{response_type}` 和 `{context_data}` 替换为适当的值。 上下文数据应该是与用户问题相关的信息。`mix_rag_response`还需要替换`{kg_context}`和`{vector_context}`
*   **示例代码:**

    ```python
    context_data = """
    Table:
    | Name | Age | City |
    |---|---|---|
    | John | 30 | New York |
    | Jane | 25 | London |
    """
    response_type = "A short paragraph summarizing the information."
    prompt = PROMPTS["rag_response"].format(response_type=response_type, context_data=context_data)
    print(prompt)
    # 然后将 prompt 传递给 LLM
    ```

**6. 相似度检查提示 (similarity\_check):**

```python
PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""
```

*   **描述:**  此提示用于确定两个问题之间的相似度。  它可以用于缓存答案并避免为相似的问题重复生成答案。
*   **关键部分:**
    *   `{original_prompt}`:  一个占位符，将被替换为原始问题。
    *   `{cached_prompt}`:  一个占位符，将被替换为缓存的问题。
*   **如何使用:** 使用此提示需要将 `{original_prompt}` 和 `{cached_prompt}` 替换为适当的值。 然后将提示传递给LLM，LLM应该返回一个介于0和1之间的相似度分数。
*   **示例代码:**

    ```python
    original_prompt = "What is the capital of France?"
    cached_prompt = "Tell me the capital city of France."
    prompt = PROMPTS["similarity_check"].format(original_prompt=original_prompt, cached_prompt=cached_prompt)
    print(prompt)
    # 然后将 prompt 传递给 LLM
    ```

总而言之，这段代码定义了一套精心设计的提示，旨在指导大型语言模型执行各种自然语言处理任务。通过明确定义目标、步骤和输出格式，这些提示有助于确保LLM生成一致、准确和有用的结果。
