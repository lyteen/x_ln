## Abstract

**Keywords:** Large Language Model, Mixture-of-Experts, Multi-head Latent Attention, Load Balancing, Multi-Token Prediction, FP8 Training, Knowledge Distillation.

<details>
    <summary>关键词</summary>
    <ul>
        大型语言模型，混合专家模型，多头潜在注意力机制，负载均衡，多令牌预测，FP8训练，知识蒸馏
    <ul>
</details>

**Abstract:**
We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters and 37B activated for each token.  It achieves efficient inference using Multi-head Latent Attention (MLA) and cost-effective training through the DeepSeekMoE architecture, validated in DeepSeek-V2. DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and a multi-token prediction training objective for enhanced performance. Pre-trained on 14.8 trillion diverse tokens, with Supervised Fine-Tuning and Reinforcement Learning, it outperforms open-source models and is comparable to leading closed-source models. It requires 2.788M H800 GPU hours for full training and exhibits remarkable stability. The model checkpoints are available.

<details>
    <summary>摘要</summary>
    <ul>
        我们提出了DeepSeek-V3，这是一个强大的混合专家(MoE)语言模型，总参数为671B，每个令牌激活37B。它通过使用多头潜在注意力(MLA)实现高效推理，并通过DeepSeekMoE架构实现经济高效的训练，这在DeepSeek-V2中得到了验证。DeepSeek-V3开创了一种用于负载均衡的辅助无损策略，并采用了一种多令牌预测训练目标，以增强性能。它在14.8万亿个多样化的令牌上进行了预训练，并通过监督微调和强化学习，其性能优于开源模型，并与领先的闭源模型相媲美。它需要278.8万H800 GPU小时才能完成完整训练，并表现出卓越的稳定性。模型检查点是可用的。
    <ul>
</details>

**Main Methods:**

1.  **Multi-head Latent Attention (MLA):**  This is used to achieve efficient inference by compressing the key-value (KV) cache, as validated in DeepSeek-V2. It facilitates the low-rank joint compression for attention keys and values to reduce KV cache size, resulting in significant memory savings.
2.  **DeepSeekMoE Architecture:** An Mixture-of-Experts architecture is used to achieve efficient and economical training. Includes finer-grained experts and some experts are isolates as shared ones.
3.  **Auxiliary-Loss-Free Load Balancing:** A novel strategy that seeks to balance the load across experts in the MoE architecture without relying on auxiliary losses, minimizing the adverse impact on model performance.
4.  **Multi-Token Prediction (MTP):** An objective that extends the prediction scope to multiple future tokens at each position. Implemented by sequentially predicting additional tokens, keeping the complete causal chain at each prediction depth.
5.  **FP8 Mixed Precision Training:** A mixed precision training framework utilizes the FP8 data format for training.
6.  **Knowledge Distillation:** Distilling the reasoning capability from the DeepSeek-R1 series of models.
7.  **DualPipe Algorithm**: An algorithm that has fewer pipeline bubbles, to overlap the computation and communication phases across forward and backward processes.
8.  **Node-Limited Routing**: Each token will be sent to at most M nodes, which are selected according to the sum of the highest affinity scores of the experts distributed on each node.

<details>
    <summary>主要方法</summary>
        <ul>
          <li>多头潜在注意力 (MLA): 用于通过压缩键值 (KV) 缓存实现高效推理，这已在 DeepSeek-V2 中得到验证。它有助于低秩联合压缩注意力键和值，从而减小 KV 缓存大小，从而显著节省内存。</li>
          <li>DeepSeekMoE 架构：一种混合专家 (MoE) 架构，用于实现高效经济的训练。包括更细粒度的专家，并且一些专家被隔离作为共享专家。</li>
          <li>辅助无损负载平衡：一种新颖的策略，旨在平衡 MoE 架构中各个专家的负载，而不依赖于辅助损失，从而最大限度地减少对模型性能的不利影响。</li>
	      <li>多令牌预测 (MTP)：一种将预测范围扩展到每个位置的多个未来令牌的目标。通过顺序预测其他令牌来实现，并在每个预测深度保持完整的因果链。</li>
          <li>FP8 混合精度训练：一种利用 FP8 数据格式进行训练的混合精度训练框架。</li>
          <li>知识蒸馏：从 DeepSeek-R1 系列模型中提取推理能力。</li>
	      <li>双管算法：一种管道气泡更少的算法，可重叠正向和反向过程中的计算和通信阶段。</li>
          <li>节点限制路由：每个令牌最多发送到 M 个节点，这些节点根据分布在每个节点上的专家的最高亲和力得分之和来选择。</li>
        </ul>
</details>

**Main Contributions:**

1.  **A Strong and Cost-Effective Language Model:** DeepSeek-V3 achieves state-of-the-art performance while maintaining cost efficiency.
2.  **Auxiliary-Loss-Free Load Balancing:** Pioneers a novel load balancing strategy that minimizes performance degradation associated with encouraging load balance in MoE models.
3.  **Multi-Token Prediction for Enhanced Performance:** Utilizes an MTP objective to enhance overall performance on evaluation benchmarks. This can be also reused for speculative decoding acceleration.
4.  **FP8 Mixed Precision Training Framework:** Designs and validates the feasibility and effectiveness of FP8 training on an extremely large-scale model.
5.  **Efficient Cross-Node MoE Training:** Overcomes communication bottlenecks in cross-node MoE training to achieve near-full computation-communication overlap and further reduce training costs.
6.  **Economical Training Costs:** Achieves pre-training of DeepSeek-V3 on 14.8T tokens at an economical cost, producing the currently strongest open-source base model.
7.  **Knowledge Distillation Methodology:** Introduces a novel methodology to distill reasoning capabilities from long-Chain-of-Thought (CoT) models into standard LLMs.

<details>
    <summary>主要贡献</summary>
        <ul>
          <li>强大且经济高效的语言模型：DeepSeek-V3 在保持成本效益的同时实现了最先进的性能。</li>
          <li>辅助无损负载平衡：开创了一种新颖的负载平衡策略，该策略最大限度地减少了与鼓励 MoE 模型中负载平衡相关的性能下降。</li>
          <li>用于增强性能的多令牌预测：利用 MTP 目标来增强评估基准的整体性能。这也可以重复用于推测解码加速。</li>
	      <li>FP8 混合精度训练框架：设计并验证了 FP8 训练在超大规模模型上的可行性和有效性。</li>
          <li>高效的跨节点 MoE 训练：克服了跨节点 MoE 训练中的通信瓶颈，实现了接近完全的计算通信重叠，并进一步降低了训练成本。</li>
          <li>经济的训练成本：以经济的成本完成了 DeepSeek-V3 在 14.8T 令牌上的预训练，产生了目前最强大的开源基础模型。</li>
	       <li>知识提炼方法： 介绍了一种新颖的方法，可将长链思维 (CoT) 模型中的推理能力提炼成标准 LLM。</li>
        </ul>
</details>
