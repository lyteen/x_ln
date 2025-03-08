## Abstract

**Keywords:** Large Language Models, Transformers, Attention Mechanism, Long Context, Compressive Memory, Streaming Inference

<details>
    <summary>关键词</summary>
    *   大型语言模型
    *   变换器
    *   注意力机制
    *   长语境
    *   压缩记忆
    *   流式推理
</details>

**Abstract:**
This work introduces an efficient method, Infini-attention, to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. Infini-attention incorporates a compressive memory into the vanilla attention mechanism and builds in both masked local attention and long-term linear attention mechanisms in a single Transformer block. The approach is demonstrated on long-context language modeling benchmarks, 1M sequence length passkey context block retrieval, and 500K length book summarization tasks with 1B and 8B LLMs. The approach introduces minimal bounded memory parameters and enables fast streaming inference for LLMs.

<details>
    <summary>摘要</summary>
    本研究介绍了一种名为Infini-attention的有效方法，用于将基于Transformer的大型语言模型（LLM）扩展到无限长的输入，同时保持有限的内存和计算。Infini-attention将压缩记忆整合到标准的注意力机制中，并在单个Transformer模块中构建了掩蔽局部注意力和长期线性注意力机制。该方法在长文本语言建模基准、百万序列长度的密码上下文块检索以及使用1B和8B LLM进行的500K长度书籍摘要任务中得到了验证。该方法引入了极小的有界内存参数，并支持LLM的快速流式推理。
</details>

**Main Methods:**

*   **Infini-attention:** A novel attention mechanism that incorporates compressive memory into the standard attention mechanism.
*   **Compressive Memory:** Reuses query, key, and value states from the dot-product attention computation to store bindings of key and value states in a compressive memory for long-term information.
*   **Linear Attention:** Employs linear attention for memory retrieval to improve computational efficiency.
*   **Causal Attention:** Implements masked local attention for capturing short-range contextual dependencies.
*   **Segment-level Streaming:** Processes extremely long inputs in a streaming fashion with segment chunking and BPTT.

<details>
    <summary>主要方法</summary>
    <ul>
      <li>Infini-attention: 一种将压缩记忆整合到标准注意力机制中的新型注意力机制.</li>
      <li>压缩记忆：复用点积注意力计算中的查询、键和值状态，将键值状态的绑定存储在压缩记忆中，以进行长期信息存储。</li>
      <li>线性注意力：采用线性注意力进行记忆检索，以提高计算效率。</li>
      <li>因果注意力：实现掩蔽的局部注意力，以捕获短程上下文依赖关系。</li>
      <li>分段流式处理：采用分段和BPTT以流式方式处理极长的输入。</li>
    </ul>
</details>

**Main Contributions:**

*   **Introduction of Infini-attention:** A practical attention mechanism with long-term compressive memory and local causal attention for efficiently modeling both long and short-range contextual dependencies.
*   **Plug-and-Play Adaptation:** Infini-attention requires minimal changes to standard scaled dot-product attention and supports plug-and-play continual pre-training and long-context adaptation.
*   **Unbounded Context with Bounded Resources:** The approach allows Transformer LLMs to scale to infinitely long context with a bounded memory and compute resource by processing extremely long inputs in a streaming fashion.
*   **State-of-the-Art Results:** Achieves new SOTA results on the 500K length book summarization task and solves 1M length passkey retrieval task.
*   **Minimal Memory Footprint:** Offers a significant reduction in memory footprint compared to previous methods.

<details>
    <summary>主要贡献</summary>
    <ul>
      <li>Infini-attention的引入：一种实用的注意力机制，具有长期压缩记忆和局部因果注意力，可以有效地建模长程和短程上下文依赖关系。</li>
      <li>即插即用适应性：Infini-attention仅需对标准的缩放点积注意力进行极小的更改，并支持即插即用的持续预训练和长上下文适应。</li>
      <li>有界资源下的无界上下文：该方法允许Transformer LLM扩展到无限长的上下文，同时通过以流式方式处理极长的输入来保持有限的内存和计算资源。</li>
      <li>最先进的结果：在50万长度的书籍摘要任务上取得了新的SOTA结果，并解决了百万长度的密码检索任务。</li>
      <li>最小的内存占用：与以前的方法相比，显着减少了内存占用。</li>
    </ul>
</details>
