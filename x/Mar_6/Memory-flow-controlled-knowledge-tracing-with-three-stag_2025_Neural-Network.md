## Abstract

**Keywords:** Knowledge Tracing, Time Series Prediction, Educational Data Mining, Memory Processing, Cognitive Process

<details>
    <summary>关键词</summary>
    <ul>
        知识追踪, 时间序列预测, 教育数据挖掘, 记忆处理, 认知过程
    </ul>
</details>

**Main Methods:**

1.  **Sensory Memory Registration (SMR):** Employs a self-attention encoder, contrastive pre-training to maximize the similarity between positive augmentation views of learning sequence representations.
2.  **Short-term Memory Fusion (SMF):** Fuses relational and temporal properties of sensory memory through a dual-channel structure composed of attention and recurrent neural networks (RNNs).
3.  **Long-term Memory Retrieval (LMR):** Uses a monotonic gating mechanism to compute weights of hidden memory states and performs read-write operations on a memory matrix.
4.  **Knowledge State Retrieval:** Combines long-term and short-term memory vectors to retrieve latent knowledge states for future performance prediction.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><strong>感觉记忆注册 (SMR):</strong> 采用自注意力编码器，通过对比预训练最大化学习序列表征的正增强视图之间的相似性。</li>
        <li><strong>短时记忆融合 (SMF):</strong> 通过由注意力和循环神经网络 (RNN) 组成的双通道结构，融合感觉记忆的关系和时间属性。</li>
        <li><strong>长期记忆检索 (LMR):</strong> 使用单调门控机制来计算隐藏记忆状态的权重，并在记忆矩阵上执行读写操作。</li>
        <li><strong>知识状态检索:</strong> 结合长期和短期记忆向量，检索潜在的知识状态，用于未来的性能预测。</li>
    </ul>
</details>

**Main Contributions:**

1.  **MFCKT Framework:** Proposed a Memory Flow-Controlled Knowledge Tracing (MFCKT) framework with three stages to address memory inconsistency issues in existing methods.
2.  **Memory Flow Reconstruction:** Reconstructed the learning process into three flow stages (sensory registration, short-term fusion, and long-term retrieval) to model multiple types of memory and simulate the conversion mechanisms between them.
3.  **Dual-Channel Structure:** Designed a dual-channel structure with attention mechanisms and RNNs for effectively mining relational and temporal properties of memory.
4.  **Monotonic Gating Mechanism:** Developed a monotonic gating mechanism for obtaining long-term memory through read-write operations on a memory matrix.
5.  **Experimental Validation:** Conducted extensive experiments on five real-world datasets, which verified the effectiveness, superiority, and interpretability of MFCKT.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><strong>MFCKT 框架:</strong> 提出了一种具有三个阶段的记忆流控制知识追踪 (MFCKT) 框架，以解决现有方法中存在的记忆不一致问题。</li>
        <li><strong>记忆流重建:</strong> 将学习过程重构为三个流动阶段（感觉注册、短期融合和长期检索），以建模多种类型的记忆并模拟它们之间的转换机制。</li>
        <li><strong>双通道结构:</strong> 设计了一种具有注意力机制和 RNN 的双通道结构，以有效地挖掘记忆的关系和时间属性。</li>
        <li><strong>单调门控机制:</strong> 开发了一种单调门控机制，通过对记忆矩阵进行读写操作来获得长期记忆。</li>
        <li><strong>实验验证:</strong> 在五个真实世界的数据集上进行了广泛的实验，验证了 MFCKT 的有效性、优越性和可解释性。</li>
    </ul>
</details>

**Abstract:**

Knowledge Tracing (KT), as a pivotal technology in intelligent education systems, analyzes students' learning data to infer their knowledge acquisition and predict their future performance. Recent advancements in KT recognize the importance of memory laws on knowledge acquisition but neglect modeling the inherent structure of memory, which leads to the inconsistency between explicit student learning and implicit memory transformation. Therefore, to enhance the consistency, we propose a novel memory flow-controlled knowledge tracing with three stages (MFCKT). Extensive experimental results on five real-world datasets verify the superiority and interpretability of MFCKT.

<details>
    <summary>摘要</summary>
        知识追踪 (KT) 作为智能教育系统中的一项关键技术，分析学生的学习数据，以推断他们的知识获取并预测他们未来的表现。KT 的最新进展认识到记忆规律对知识获取的重要性，但忽略了对记忆内在结构的建模，这导致了显性学生学习和隐性记忆转换之间的不一致。因此，为了增强一致性，我们提出了一种新颖的记忆流控制三阶段知识追踪 (MFCKT)。在五个真实世界数据集上的大量实验结果验证了 MFCKT 的优越性和可解释性。
</details>