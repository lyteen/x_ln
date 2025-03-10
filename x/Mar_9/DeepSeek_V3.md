## Abstract

**Keywords:** Large Language Models (LLMs), Mixture-of-Experts (MoE), Multi-head Latent Attention (MLA), FP8 Training, Load Balancing, Multi-Token Prediction.

<details>
    <summary>关键词 (Chinese)</summary>
    大规模语言模型 (LLMs), 混合专家模型 (MoE), 多头潜在注意力 (MLA), FP8 训练, 负载均衡, 多 Token 预测。
</details>

**Abstract:** 
We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters and 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, thoroughly validated in DeepSeek-V2. DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. Pre-trained on 14.8 trillion diverse, high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning, DeepSeek-V3 outperforms open-source models and rivals leading closed-source models. It requires only 2.788M H800 GPU hours for full training, with a remarkably stable process and publicly available checkpoints.

<details>
    <summary>摘要 (Chinese)</summary>
    我们提出了 DeepSeek-V3，一个强大的混合专家 (MoE) 语言模型，总参数达 671B，每个 token 激活 37B。为了实现高效推理和经济高效的训练，DeepSeek-V3 采用了多头潜在注意力 (MLA) 和 DeepSeekMoE 架构，这些架构已在 DeepSeek-V2 中得到充分验证。DeepSeek-V3 首创了一种无辅助损失的负载均衡策略，并设置了多 token 预测训练目标，以获得更强的性能。DeepSeek-V3 在 14.8 万亿个多样化的高质量 token 上进行了预训练，然后进行了监督微调和强化学习，其性能优于开源模型，并且可以媲美领先的闭源模型。完整的训练仅需 278.8 万 H800 GPU 小时，训练过程非常稳定，并且公开了可用的检查点。
</details>

**Main Methods:**
1.  **Multi-head Latent Attention (MLA):**  Adopts MLA for efficient inference and reduced Key-Value (KV) cache during generation.
2.  **DeepSeekMoE Architecture:** Uses finer-grained experts with shared experts for cost-effective training.
3.  **Auxiliary-Loss-Free Load Balancing:**  Pioneers an auxiliary-loss-free strategy to minimize the adverse impact on model performance from load balancing efforts.
4.  **Multi-Token Prediction (MTP):**  Employs a multi-token prediction training objective to enhance overall performance.
5.  **FP8 Mixed Precision Training:**  Utilizes FP8 data format for accelerated training and reduced GPU memory usage.
6.  **DualPipe Algorithm:**  Designs a DualPipe algorithm for efficient pipeline parallelism, overlapping computation and communication.
7.  **Cross-Node All-to-All Communication Kernels:** Develops efficient communication kernels to fully utilize InfiniBand and NVLink bandwidths.

<details>
    <summary>主要方法 (Chinese)</summary>
   <ol>
    <li><strong>多头潜在注意力 (MLA):</strong> 采用 MLA 实现高效推理，并减少生成过程中 Key-Value (KV) 缓存。</li>
    <li><strong>DeepSeekMoE 架构:</strong> 使用更细粒度的专家，并隔离一些专家作为共享的专家，实现经济高效的训练。</li>
    <li><strong>无辅助损失负载均衡:</strong> 首创一种无辅助损失策略，最大限度地减少负载均衡工作对模型性能的负面影响。</li>
    <li><strong>多 token 预测 (MTP):</strong> 采用多 token 预测训练目标，以提高整体性能。</li>
    <li><strong>FP8 混合精度训练:</strong> 利用 FP8 数据格式来加速训练并减少 GPU 内存使用量。</li>
    <li><strong>DualPipe 算法:</strong> 设计 DualPipe 算法，用于高效的流水线并行，实现计算和通信的重叠。</li>
    <li><strong>跨节点 All-to-All 通信内核:</strong> 开发高效的通信内核，以充分利用 InfiniBand 和 NVLink 带宽。</li>
   </ol>
</details>

**Main Contributions:**

1.  **Auxiliary-Loss-Free Load Balancing:**  Pioneers an auxiliary-loss-free strategy for load balancing that minimizes performance degradation.
2.  **Multi-Token Prediction (MTP) Objective:**  Investigates and proves the benefit of a Multi-Token Prediction (MTP) objective for model performance, which can also be used for speculative decoding.
3.  **FP8 Mixed Precision Training Framework:**  Designs and validates the feasibility and effectiveness of an FP8 mixed precision training framework on an extremely large-scale model.
4.  **Computation-Communication Overlap Optimization:**  Overcomes the communication bottleneck in cross-node MoE training through algorithm, framework, and hardware co-design, achieving near-full computation-communication overlap.
5.  **Economical Training:**  Completes pre-training DeepSeek-V3 on 14.8T tokens with only 2.664M H800 GPU hours, producing the strongest open-source base model.
6.  **Knowledge Distillation Methodology:** Introduces an innovative methodology to distill reasoning capabilities from long-Chain-of-Thought (CoT) models into standard LLMs.

<details>
    <summary>主要贡献 (Chinese)</summary>
    <ol>
        <li><strong>无辅助损失负载均衡:</strong> 首创了一种无辅助损失负载均衡策略，最大限度地减少了性能下降。</li>
        <li><strong>多 Token 预测 (MTP) 目标:</strong> 验证了多 token 预测目标对模型性能的益处，并且可用于推测性解码。</li>
        <li><strong>FP8 混合精度训练框架:</strong> 设计并验证了 FP8 混合精度训练框架在超大规模模型上的可行性和有效性。</li>
        <li><strong>计算-通信重叠优化:</strong> 通过算法、框架和硬件的协同设计，克服了跨节点 MoE 训练中的通信瓶颈，实现了接近完全的计算-通信重叠。</li>
        <li><strong>经济高效的训练:</strong> 仅使用 266.4 万 H800 GPU 小时，就完成了 DeepSeek-V3 在 14.8T token 上的预训练，生成了最强大的开源基础模型。</li>
	    <li><strong>知识蒸馏方法:</strong> 引入了一种创新的方法，将长链思维 (CoT) 模型的推理能力提炼到标准 LLM 中。</li>
   </ol>
</details>
