## Abstract

**Keywords:** Sparse Attention, Long-Context Modeling, Hardware-Aware Training, Transformer, Large Language Models, Efficiency.
<details>
    <summary>关键词</summary>
    <ul>
        稀疏注意力, 长文本建模, 硬件感知训练, Transformer, 大型语言模型, 效率
    </ul>
</details>

**Abstract:**
This paper introduces Natively trainable Sparse Attention (NSA), a hardware-aligned and natively trainable sparse attention mechanism designed to address the computational challenges of long-context modeling in next-generation language models. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection. The approach achieves substantial speedups through arithmetic intensity-balanced algorithm design and implementation optimizations for modern hardware and enables end-to-end training. Pretrained NSA models demonstrate comparable or superior performance to full attention models across general benchmarks, long-context tasks, and instruction-based reasoning, while achieving significant speedups on 64k-length sequences during decoding, forward, and backward propagation.

<details>
    <summary>摘要</summary>
    <ul>
        本文介绍了一种原生可训练的稀疏注意力（NSA），这是一种硬件对齐且原生可训练的稀疏注意力机制，旨在解决下一代语言模型中长文本建模的计算挑战。NSA 采用动态分层稀疏策略，将粗粒度token压缩与细粒度token选择相结合。该方法通过算术强度平衡的算法设计和针对现代硬件的实现优化实现了显著的加速，并实现了端到端训练。预训练的 NSA 模型在通用基准、长文本任务和基于指令的推理方面表现出与全注意力模型相当或更优越的性能，同时在解码、前向传播和反向传播期间在 64k 长度的序列上实现了显著的加速。
    </ul>
</details>

**Main Methods:**

1.  **Dynamic Hierarchical Sparse Strategy:** Combines coarse-grained token compression with fine-grained token selection to balance global context awareness and local precision.
2.  **Arithmetic Intensity-Balanced Algorithm Design:** Optimizes algorithm design to match the arithmetic intensity of modern hardware, maximizing computational throughput.
3.  **Hardware-Aligned Implementation:** Implements optimizations tailored to modern hardware architectures, including Tensor Core utilization and memory access patterns.
4.  **End-to-End Training:** Enables stable end-to-end training with trainable operators, reducing pretraining computation without sacrificing model performance.
5.  **Specialized Kernel Implementations:** Develops specialized Triton kernels to maximize practical efficiency, focusing on blockwise sparse attention for Tensor Core utilization and memory access.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><b>动态分层稀疏策略：</b> 结合粗粒度的token压缩和细粒度的token选择，以平衡全局上下文感知和局部精度。</li>
        <li><b>算术强度平衡算法设计：</b> 优化算法设计以匹配现代硬件的算术强度，从而最大限度地提高计算吞吐量。</li>
        <li><b>硬件对齐的实现：</b> 实施针对现代硬件架构量身定制的优化，包括 Tensor Core 的利用和内存访问模式。</li>
        <li><b>端到端训练：</b> 通过可训练的算子实现稳定的端到端训练，从而在不牺牲模型性能的情况下减少预训练计算。</li>
        <li><b>专用内核实现：</b> 开发专用的 Triton 内核以最大限度地提高实际效率，重点关注 Tensor Core 利用和内存访问的分块稀疏注意力。</li>
    </ul>
</details>

**Main Contributions:**

1.  **Novel Sparse Attention Mechanism (NSA):** Introduces a natively trainable sparse attention mechanism designed for efficient long-context modeling.
2.  **Hardware-Aligned Design:** Optimizes algorithm design and implementation for modern hardware, achieving substantial speedups in training and inference.
3.  **End-to-End Trainability:** Enables stable end-to-end training of sparse attention models, reducing pretraining computation costs without sacrificing model performance.
4.  **Performance Superiority:** Demonstrates comparable or superior performance to full attention models across a range of benchmarks and tasks.
5.  **Significant Speedups:** Achieves substantial speedups over full attention on 64k-length sequences during decoding, forward propagation, and backward propagation.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><b>新型稀疏注意力机制 (NSA)：</b> 引入了一种为高效长上下文建模而设计的原生可训练稀疏注意力机制。</li>
        <li><b>硬件对齐设计：</b> 优化了算法设计和实现以适应现代硬件，从而在训练和推理中实现了显著的加速。</li>
        <li><b>端到端可训练性：</b> 实现了稀疏注意力模型稳定的端到端训练，减少了预训练计算成本，而不会牺牲模型性能。</li>
        <li><b>卓越的性能：</b> 证明了在各种基准和任务中与全注意力模型相当或更优越的性能。</li>
        <li><b>显著的加速：</b> 在解码、前向传播和反向传播期间，在 64k 长度的序列上实现了比全注意力更显著的加速。</li>
    </ul>
</details>
