## Abstract

**Keywords:** Autoregressive Modeling, Image Generation, Scaling Laws, Zero-Shot Generalization, Transformers, Visual Representation Learning

<details>
    <summary>关键词</summary>
    <ul>
        自回归建模, 图像生成, 缩放律, 零样本泛化, Transformer, 视觉表征学习
    <ul>
</details>

**Abstract:**
This paper introduces Visual AutoRegressive (VAR) modeling, a novel image generation paradigm based on coarse-to-fine "next-scale prediction". VAR departs from traditional raster-scan approaches by autoregressively predicting multi-scale token maps, offering a more efficient and intuitive way for transformers to learn visual distributions.  The study shows that VAR achieves state-of-the-art results in image generation, surpassing diffusion transformers on the ImageNet 256x256 benchmark with a Fréchet Inception Distance (FID) of 1.73 and an inception score (IS) of 350.2, while achieving a 20x speedup in inference.  The paper further demonstrates VAR's superior data efficiency, scalability, and ability to emulate the scaling laws and zero-shot generalization observed in large language models (LLMs).  The authors release their models and code to facilitate further research in visual generation and unified learning.

<details>
    <summary>摘要</summary>
    <ul>
       本文介绍了一种名为 Visual AutoRegressive (VAR) 建模的新颖图像生成范式，该范式基于由粗到精的“下一尺度预测”。VAR 通过自回归预测多尺度 token 图来脱离传统的栅格扫描方法，为 transformers 提供了一种更高效和直观的方式来学习视觉分布。研究表明，VAR 在图像生成方面取得了最先进的结果，在 ImageNet 256x256 基准测试中超越了扩散 transformers，Fréchet Inception Distance (FID) 为 1.73，inception score (IS) 为 350.2，同时实现了 20 倍的推理加速。本文进一步证明了 VAR 卓越的数据效率、可扩展性以及模仿大型语言模型 (LLM) 中观察到的缩放律和零样本泛化的能力。作者发布了他们的模型和代码，以促进视觉生成和统一学习方面的进一步研究。
    <ul>
</details>

**Main Methods:**

1. **Next-Scale Prediction:** This is the core idea, where the autoregressive process predicts the next higher resolution token map conditioned on all previous, lower-resolution maps.
2. **Multi-Scale VQVAE:** A custom multi-scale vector quantization variational autoencoder (VQVAE) encodes images into a set of hierarchical token maps at different resolutions, providing the input representation for the VAR transformer.
3. **GPT-Style Transformer:** A standard decoder-only transformer architecture, similar to GPT-2, is used as the autoregressive model. Modifications are made, such as adaptive layer normalization (AdaLN), to improve performance in the visual domain.
4. **Block-wise Causal Masking:** During training, a block-wise causal attention mask ensures that each token map attends only to its "prefix" (lower-resolution maps), maintaining the autoregressive property.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>下一尺度预测: 这是核心思想，自回归过程预测下一个更高分辨率的 token 图，其条件是所有先前的、较低分辨率的图。</li>
        <li>多尺度 VQVAE: 一种定制的多尺度向量量化变分自动编码器 (VQVAE) 将图像编码成一组分层 token 图，这些 token 图具有不同的分辨率，从而为 VAR transformer 提供输入表示。</li>
        <li>GPT 风格 Transformer: 采用标准的仅解码器 Transformer 架构，类似于 GPT-2，作为自回归模型。进行了修改，例如自适应层归一化 (AdaLN)，以提高视觉领域的性能。</li>
        <li>块式因果掩码: 在训练期间，块式因果注意力掩码确保每个 token 图仅关注其“前缀”（较低分辨率的图），从而保持自回归特性。</li>
    <ul>
</details>

**Main Contributions:**

1. **Novel Autoregressive Paradigm:** Introduces the "next-scale prediction" approach for visual autoregressive modeling, deviating from traditional "next-token prediction" and offering improved efficiency and performance.
2. **State-of-the-Art Results:** Achieves significantly improved image generation quality compared to previous autoregressive models, surpassing diffusion transformers on ImageNet.
3. **Scaling Law Emulation:** Empirically demonstrates that VAR models exhibit power-law scaling laws similar to those observed in LLMs, suggesting potential for further performance gains with increased model size and training data.
4. **Zero-Shot Generalization:** Showcases the zero-shot generalization capabilities of VAR models in downstream tasks like image in-painting, out-painting, and editing, highlighting their versatility and adaptability.
5. **Open-Source Implementation:** Provides a comprehensive open-source code suite, including the multi-scale VQVAE tokenizer and VAR transformer training pipelines, promoting further research and development in the field.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>新颖的自回归范式: 引入了视觉自回归建模的“下一尺度预测”方法，偏离了传统的“下一 token 预测”，并提供了更高的效率和性能。</li>
        <li>最先进的结果: 与之前的自回归模型相比，实现了显着提高的图像生成质量，在 ImageNet 上超越了扩散 transformers。</li>
        <li>缩放律仿真: 经验证表明，VAR 模型表现出类似于 LLM 中观察到的幂律缩放律，这表明通过增加模型大小和训练数据可以进一步提高性能的潜力。</li>
        <li>零样本泛化: 展示了 VAR 模型在图像修复、外绘和编辑等下游任务中的零样本泛化能力，突出了它们的多功能性和适应性。</li>
        <li>开源实现: 提供了一个全面的开源代码套件，包括多尺度 VQVAE token 器和 VAR transformer 训练管道，从而促进了该领域的进一步研究和开发。</li>
    <ul>
</details>
