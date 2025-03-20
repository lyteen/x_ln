Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\knowledge_retrievers\__init__.py`

好的，重新开始。我会给出更智能的代码示例，并附带更详细的中文描述。

**1. 基于距离加权的混合检索器 (DistanceWeightedHybridRetriever):**

   ```python
   from typing import List, Tuple
   from pikerag.knowledge_retrievers.base_qa_retriever import BaseQaRetriever

   class DistanceWeightedHybridRetriever(BaseQaRetriever):
       """
       结合多个检索器，并根据距离（或相似度）对结果进行加权。
       距离越近（相似度越高），权重越大。
       """

       def __init__(self, retrievers: List[BaseQaRetriever], weights: List[float] = None, distance_scaling: float = 1.0):
           """
           Args:
               retrievers: 要组合的检索器列表.
               weights: 每个检索器的权重列表。如果为None，则平均分配权重.
               distance_scaling: 距离缩放因子，用于调整距离对权重的影响.  越大，距离影响越大。
           """
           self.retrievers = retrievers
           if weights is None:
               self.weights = [1.0 / len(retrievers)] * len(retrievers)
           else:
               assert len(weights) == len(retrievers), "权重数量必须与检索器数量一致。"
               self.weights = weights
           self.distance_scaling = distance_scaling

       def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
           """
           执行检索。

           Args:
               query: 检索查询.
               top_k: 返回结果的数量.

           Returns:
               一个列表，包含检索到的文本块及其相应的得分.
           """
           all_results = []
           for retriever, weight in zip(self.retrievers, self.weights):
               results = retriever.retrieve(query, top_k=top_k)
               all_results.extend([(doc, score * weight) for doc, score in results])

           # 根据得分进行排序
           all_results.sort(key=lambda x: x[1], reverse=True)

           # 应用距离加权：将得分转换为概率，然后根据距离（1 - score）调整概率
           # 为了避免除以零，我们添加一个小的平滑项
           smoothed_scores = [max(0.001, score) for doc, score in all_results] # 保证得分大于0
           probabilities = [score / sum(smoothed_scores) for score in smoothed_scores]
           distances = [1.0 - score for doc, score in all_results]  # 距离 = 1 - 相似度
           weighted_probabilities = [p * (1.0 / (1.0 + d * self.distance_scaling)) for p, d in zip(probabilities, distances)]

           # 重新归一化加权概率
           total_weighted_probability = sum(weighted_probabilities)
           if total_weighted_probability > 0:
               normalized_weighted_probabilities = [p / total_weighted_probability for p in weighted_probabilities]
           else:
               normalized_weighted_probabilities = [0.0] * len(weighted_probabilities) # 避免全部权重为0的情况

           # 将加权概率应用回得分
           weighted_results = [(doc, p) for (doc, score), p in zip(all_results, normalized_weighted_probabilities)]

           # 重新排序并截断
           weighted_results.sort(key=lambda x: x[1], reverse=True)
           return weighted_results[:top_k]
   ```

   **描述:**  `DistanceWeightedHybridRetriever` 是一个用于组合多个检索器的类。它首先使用每个检索器获取结果，然后根据检索器自身的权重和文档的距离（1 - 相似度得分）对结果进行加权。 距离越近的文档，其权重越大。 这允许将不同类型的检索器（例如，基于关键词的 BM25 和基于语义的向量检索）结合起来，同时强调更相关的结果。

   **主要特点:**

   *   **混合检索:** 结合多个 `BaseQaRetriever` 实例。
   *   **加权组合:**  允许为每个检索器分配不同的权重。
   *   **距离加权:** 根据文档的相似度得分对结果进行加权，以优先考虑更相关的文档。`distance_scaling`参数控制距离的影响程度。
   *   **可配置性:**  `distance_scaling` 参数允许调整距离对权重的贡献。
   *   **容错性:**  考虑了所有得分都为0的情况，避免了除零错误。

   **示例:**

   ```python
   from pikerag.knowledge_retrievers.bm25_retriever import BM25QaChunkRetriever
   from pikerag.knowledge_retrievers.chroma_qa_retriever import QaChunkRetriever
   import chromadb

   # 创建一些模拟的知识库和数据
   documents = ["This is document 1 about apples.", "This is document 2 about bananas.", "This is document 3 about oranges."]
   chroma_client = chromadb.EphemeralClient() # 使用内存数据库
   collection = chroma_client.create_collection("my_collection")
   collection.add(documents=documents, ids=[f"doc{i}" for i in range(len(documents))])


   # 初始化检索器
   bm25_retriever = BM25QaChunkRetriever(documents=documents)
   chroma_retriever = QaChunkRetriever(collection=collection)

   # 创建混合检索器
   hybrid_retriever = DistanceWeightedHybridRetriever(
       retrievers=[bm25_retriever, chroma_retriever],
       weights=[0.6, 0.4],  # BM25权重更高
       distance_scaling=2.0
   )

   # 检索查询
   query = "fruit"
   results = hybrid_retriever.retrieve(query, top_k=3)

   # 打印结果
   for doc, score in results:
       print(f"文档: {doc}, 得分: {score}")
   ```

   在这个例子中，`DistanceWeightedHybridRetriever` 将 `BM25QaChunkRetriever` 和 `QaChunkRetriever` 组合在一起，并根据它们的权重和文档的距离对结果进行加权。  `distance_scaling`设置为2.0，表明距离对最终得分有较大的影响。 这允许检索器利用 BM25 的速度和关键词匹配能力，以及向量检索器的语义理解能力。

---

**2. 自适应Top-K选择器 (AdaptiveTopKSelector):**

   ```python
   from typing import List, Tuple
   from pikerag.knowledge_retrievers.base_qa_retriever import BaseQaRetriever

   class AdaptiveTopKSelector(BaseQaRetriever):
       """
       动态调整每个检索器的 top_k 值，以平衡检索性能和质量。
       更可靠的检索器应该被赋予更大的 top_k 值。
       """

       def __init__(self, retrievers: List[BaseQaRetriever], reliability_scores: List[float], base_top_k: int = 10):
           """
           Args:
               retrievers: 要组合的检索器列表。
               reliability_scores: 每个检索器的可靠性得分列表。得分越高，表示检索器越可靠。
               base_top_k: 基础的 top_k 值，所有检索器的 top_k 值都基于此值进行调整。
           """
           self.retrievers = retrievers
           self.reliability_scores = reliability_scores
           self.base_top_k = base_top_k

           # 计算每个检索器的 top_k 值
           total_reliability = sum(reliability_scores)
           self.top_k_values = [int(base_top_k * (score / total_reliability)) for score in reliability_scores]
           # 确保至少为1
           self.top_k_values = [max(1, k) for k in self.top_k_values]


       def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
           """
           执行检索。

           Args:
               query: 检索查询。
               top_k: 返回结果的数量。

           Returns:
               一个列表，包含检索到的文本块及其相应的得分。
           """
           all_results = []
           for retriever, k in zip(self.retrievers, self.top_k_values):
               results = retriever.retrieve(query, top_k=k)
               all_results.extend(results)

           # 根据得分进行排序
           all_results.sort(key=lambda x: x[1], reverse=True)
           return all_results[:top_k]

   ```

   **描述:**  `AdaptiveTopKSelector`  根据每个检索器的可靠性动态调整其 `top_k` 值。 更可靠的检索器将获得更大的 `top_k` 值，这意味着它们将在最终结果中贡献更多的文档。 这允许系统动态地优先考虑来自更值得信赖的来源的信息。

   **主要特点:**

   *   **自适应 Top-K:**  根据检索器的可靠性动态调整 `top_k` 值。
   *   **可靠性得分:** 使用 `reliability_scores` 表示每个检索器的可靠性。
   *   **基础 Top-K:**  `base_top_k` 参数允许调整所有检索器的整体 `top_k` 值。
   *   **平衡检索:**  在检索性能和质量之间取得平衡。

   **示例:**

   ```python
   from pikerag.knowledge_retrievers.bm25_retriever import BM25QaChunkRetriever
   from pikerag.knowledge_retrievers.chroma_qa_retriever import QaChunkRetriever
   import chromadb

   # 创建一些模拟的知识库和数据
   documents = ["This is document 1 about apples.", "This is document 2 about bananas.", "This is document 3 about oranges."]
   chroma_client = chromadb.EphemeralClient() # 使用内存数据库
   collection = chroma_client.create_collection("my_collection")
   collection.add(documents=documents, ids=[f"doc{i}" for i in range(len(documents))])


   # 初始化检索器
   bm25_retriever = BM25QaChunkRetriever(documents=documents)
   chroma_retriever = QaChunkRetriever(collection=collection)

   # 创建自适应 Top-K 选择器
   adaptive_selector = AdaptiveTopKSelector(
       retrievers=[bm25_retriever, chroma_retriever],
       reliability_scores=[0.8, 0.9],  # Chroma 检索器更可靠
       base_top_k=15
   )

   # 检索查询
   query = "fruit"
   results = adaptive_selector.retrieve(query, top_k=3)

   # 打印结果
   for doc, score in results:
       print(f"文档: {doc}, 得分: {score}")
   ```

   在这个例子中，`AdaptiveTopKSelector`  将 `BM25QaChunkRetriever` 和 `QaChunkRetriever` 组合在一起，并根据它们的可靠性调整它们的 `top_k` 值。  由于 Chroma 检索器更可靠，因此它将获得比 BM25 检索器更大的 `top_k` 值. 这意味着 Chroma 检索器将在最终结果中贡献更多的文档。

希望这些例子更智能，并且中文描述更清晰易懂！