## Abstract

**Keywords:** Large Multimodal Models (LMMs), Visual Instruction Tuning, CLIP, VQA, Benchmarking, Data Efficiency, Compositional Capabilities, Model Hallucination.

<details>
    <summary>关键词</summary>
    <ul>
        大型多模态模型 (LMMs), 视觉指令微调, CLIP, VQA, 基准测试, 数据效率, 组合能力, 模型幻觉.
    <ul>
</details>

**Abstract:** This paper systematically investigates design choices in Large Multimodal Models (LMMs) within the LLaVA framework, demonstrating the surprising power and data efficiency of the fully-connected vision-language connector. With simple modifications such as using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with response formatting prompts, the authors establish stronger baselines achieving state-of-the-art results across 11 benchmarks using only 1.2M publicly available data and training in ~1 day on a single 8-A100 node. The paper also explores open problems in LMMs, including scaling to higher resolution inputs, compositional capabilities, and model hallucination.

<details>
    <summary>摘要</summary>
    <ul>
        本文系统地研究了在LLaVA框架下大型多模态模型（LMM）的设计选择，证明了全连接视觉-语言连接器的惊人能力和数据效率。通过简单的修改，例如使用CLIP-ViT-L-336px与MLP投影以及添加带有响应格式提示的面向学术任务的VQA数据，作者建立了更强的基线，仅使用1.2M公开可用数据并在单个8-A100节点上训练约1天，就在11个基准测试中实现了最先进的结果。该论文还探讨了LMM中的开放问题，包括扩展到更高分辨率的输入、组合能力和模型幻觉。
    <ul>
</details>

**Main Methods:**

1.  **LLaVA Framework Modification:** Simple yet effective modifications to the existing LLaVA architecture, such as replacing the linear projection layer with an MLP and incorporating formatted VQA datasets.
2.  **Data Efficiency Focus:** Leveraging publicly available data effectively, achieving state-of-the-art performance with a relatively small dataset of 1.2M image-text pairs.
3.  **Image Resolution Scaling:** Scaling input image resolution by dividing images into grids and encoding them independently.
4.  **Careful Data Scaling:** Providing empirical evidence for the scaling of data granularity in conjunction with the model's capability, crucial for improved capability.

<details>
    <summary>主要方法</summary>
        <ul>
          <li><strong>LLaVA 框架修改：<strong>对现有 LLaVA 架构进行简单而有效的修改，例如用 MLP 替换线性投影层，并结合格式化的 VQA 数据集。</li>
          <li><strong>数据效率重点：<strong>有效地利用公开可用的数据，并使用相对较小的 1.2M 图像-文本对数据集实现最先进的性能。</li>
          <li><strong>图像分辨率缩放：<strong>通过将图像划分为网格并独立编码来缩放输入图像分辨率。</li>
          <li><strong>谨慎的数据缩放：<strong>为数据粒度的缩放与模型的性能相结合提供经验证据，这对于提高性能至关重要。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Stronger Baselines:** Establishes stronger baselines for visual instruction tuning that achieve state-of-the-art performance on a broad range of benchmarks.
2.  **Data Efficiency Insights:** Demonstrates that fully-connected vision-language connectors in LMMs are surprisingly powerful and data-efficient.
3.  **Open Problem Exploration:** Provides early explorations into open problems in LMMs, such as scaling to higher resolution inputs, compositional capabilities, and model hallucination.
4.  **Reproducible Baselines:** Offers improved and easily-reproducible baselines to provide a reference for future research in open-source LMMs.
5.  **Resource Efficiency:** Achieves impressive results while using only publicly available data and training on a single 8-A100 node in approximately one day.

<details>
    <summary>主要贡献</summary>
        <ul>
          <li><strong>更强的基线：<strong>为视觉指令微调建立了更强的基线，在广泛的基准测试中实现了最先进的性能。</li>
          <li><strong>数据效率洞察：<strong>证明了 LMM 中全连接视觉-语言连接器出奇地强大且数据效率高。</li>
          <li><strong>开放问题探索：<strong>对 LMM 中的开放问题进行了早期探索，例如缩放到更高分辨率的输入、组合能力和模型幻觉。</li>
	      <li><strong>可重现的基线：<strong>提供改进且易于重现的基线，为开源 LMM 中的未来研究提供参考。</li>
          <li><strong>资源效率：<strong>仅使用公开可用的数据并在单个 8-A100 节点上训练约一天即可实现令人印象深刻的结果。</li>
        </ul>
</details>
