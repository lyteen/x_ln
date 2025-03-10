## Abstract

Readed
---
*  Transformer-XL 出自与 2019 GoogleBrain 的 paper （可能是 Infinite Context Transformer -- Infinite-attention 的早些版本）通过存储上一段隐藏层的 output （embedding）(Infinsite-attention 是进行压缩？) 输入来进行 model 长文本记忆 （Long-text）

        critical point：Segment-level Recurrence (片段级递归) 及实现隐藏层存储的关键技术
---

**Keywords:** protein language models, self-supervised learning, transfer learning, protein secondary structure prediction, subcellular localization, high-performance computing

<details>
    <summary>关键词</summary>
    <ul>
        蛋白质语言模型, 自监督学习, 迁移学习, 蛋白质二级结构预测, 亚细胞定位, 高性能计算
    </ul>
</details>

**Abstract:**
This paper introduces ProtTrans, a suite of protein language models (pLMs) trained using self-supervised learning on massive protein sequence datasets (up to 393 billion amino acids). The models, including auto-regressive (Transformer-XL, XLNet) and auto-encoder (BERT, Albert, Electra, T5) architectures, were trained on the Summit supercomputer using thousands of GPUs and TPUs. Dimensionality reduction reveals that pLM embeddings capture biophysical features.  The embeddings are validated as input for secondary structure prediction (Q3=81%-87%), and subcellular location prediction (Q10=81%, Q2=91%).  Notably, the ProtT5 model outperforms state-of-the-art methods for secondary structure prediction without using multiple sequence alignments (MSAs), offering a computationally efficient alternative.  The results suggest pLMs learn the grammar of the language of life. Models are publicly available.

<details>
    <summary>摘要</summary>
    <p>本文介绍了ProtTrans，一套蛋白质语言模型（pLMs），利用自监督学习方法在海量的蛋白质序列数据集（高达3930亿个氨基酸）上进行训练。 这些模型，包括自回归模型（Transformer-XL, XLNet）和自编码器模型（BERT, Albert, Electra, T5）架构，均在Summit超级计算机上使用数千个GPU和TPU进行训练。 降维结果表明，pLM嵌入能够捕捉生物物理特征。 嵌入被验证为二级结构预测（Q3=81%-87%）和亚细胞定位预测（Q10=81%, Q2=91%）的有效输入。 值得注意的是，ProtT5模型在不使用多序列比对(MSAs)的情况下，优于最先进的二级结构预测方法，从而提供了一种计算效率高的替代方案。 结果表明，pLMs学习到了生命语言的语法。 模型已公开提供。</p>
</details>

**Main Methods:**

1.  **Self-Supervised Pre-training:**  Utilized a variety of language model architectures (Transformer-XL, XLNet, BERT, Albert, Electra, T5) for self-supervised pre-training on protein sequences.  The models were trained to predict masked amino acids within a sequence.
2.  **Large-Scale Training:** Exploited High-Performance Computing (HPC) resources (Summit supercomputer, TPU Pods) to train models on datasets containing up to 393 billion amino acids.
3.  **Embedding Extraction:** Extracted embeddings (vector representations) from the last hidden layer of the pre-trained pLMs.
4.  **Dimensionality Reduction:**  Used t-distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensionality of the embeddings for visualization and analysis.
5.  **Supervised Fine-tuning (Transfer Learning):** Used the extracted pLM embeddings as input features for downstream supervised tasks, including protein secondary structure prediction and subcellular localization prediction.  Simple Convolutional Neural Networks (CNNs) and Feed Forward Networks (FNNs) were used as supervised models.
6. **Analysis of Attention Mechanisms:** Visualized attention weights of ProtAlbert model to identify conserved motifs, specifically the zinc-finger binding domain.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>自监督预训练: 使用多种语言模型架构（Transformer-XL, XLNet, BERT, Albert, Electra, T5）在蛋白质序列上进行自监督预训练。模型被训练来预测序列中被屏蔽的氨基酸。</li>
        <li>大规模训练: 利用高性能计算（HPC）资源（Summit超级计算机，TPU Pods）在包含多达3930亿个氨基酸的数据集上训练模型。</li>
        <li>嵌入提取: 从预训练pLM的最后一层提取嵌入（向量表示）。</li>
        <li>降维: 使用t分布随机邻域嵌入（t-SNE）来降低嵌入的维度，以进行可视化和分析。</li>
        <li>监督微调 (迁移学习): 将提取的pLM嵌入作为下游监督任务的输入特征，包括蛋白质二级结构预测和亚细胞定位预测。 简单的卷积神经网络 (CNN) 和前馈网络 (FNN) 被用作监督模型。</li>
	  <li>分析注意力机制：对ProtAlbert模型的注意力权重进行可视化，以识别保守基序，特别是锌指结合域。</li>
    </ul>
</details>

**Main Contributions:**

1.  **Development of Large Protein Language Models:** Trained several large-scale protein language models (pLMs) using self-supervised learning on unprecedentedly large protein sequence datasets.
2.  **Comprehensive Evaluation:**  Evaluated the effectiveness of different pLM architectures (auto-regressive vs. auto-encoder) and training strategies for capturing relevant information about protein structure and function.
3.  **Demonstrated Competitive Performance Without MSAs:** Showed that pLM embeddings can achieve competitive, and in the case of ProtT5 even surpass, state-of-the-art performance on protein secondary structure prediction without the need for computationally expensive multiple sequence alignments (MSAs).
4. **Analysis of Learned Protein Representation:**  Demonstrated the models' ability to learn and represent relevant information from the data, including structural classifications and biophysical properties.
5.  **Open Availability of Models:** Released the trained pLMs and code publicly to facilitate further research in protein bioinformatics.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>开发大型蛋白质语言模型: 利用自监督学习方法在空前庞大的蛋白质序列数据集上训练了多个大规模蛋白质语言模型（pLMs）。</li>
        <li>综合评估: 评估了不同pLM架构（自回归与自编码器）和训练策略在捕获蛋白质结构和功能相关信息方面的有效性。</li>
        <li>证明了在不使用多序列比对的情况下，具有竞争力的性能: 证明了pLM嵌入可以实现具有竞争力的，并且在ProtT5的情况下甚至超越了最先进的蛋白质二级结构预测性能，而无需进行计算成本高昂的多序列比对（MSA）。</li>
	   <li>分析学习到的蛋白质表征：证明了模型从数据中学习和表示相关信息的能力，包括结构分类和生物物理性质。</li>
       <li>模型开源: 公开发布了训练好的pLM和代码，以促进蛋白质生物信息学领域的进一步研究。</li>
    </ul>
</details>
