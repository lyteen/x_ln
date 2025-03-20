## Abstract

**Keywords:** Generative Models, Fractals, Autoregressive Models, Image Generation, Modularization, Self-Similarity

<details>
    <summary>关键词</summary>
    <ul>
        生成模型，分形，自回归模型，图像生成，模块化，自相似性
    <ul>
</details>

**Abstract:**
Modularization is a core principle in computer science, simplifying complex functions by breaking them into atomic blocks. This paper introduces a novel level of modularization, abstracting generative models into atomic generative modules. Analogous to fractals in mathematics, this method creates a new type of generative model through the recursive invocation of these atomic modules, leading to self-similar fractal architectures termed "fractal generative models." A demonstration is provided using autoregressive models as atomic modules, examined on the challenging task of pixel-by-pixel image generation, exhibiting strong likelihood estimation and generation quality. This work aims to introduce a new generative modeling paradigm, fostering a basis for further exploration.

<details>
    <summary>摘要</summary>
    <ul>
        模块化是计算机科学的核心原则，它通过将复杂函数分解为原子块来简化它们。本文介绍了一种新的模块化级别，将生成模型抽象为原子生成模块。类似于数学中的分形，这种方法通过递归调用这些原子模块创建了一种新型的生成模型，从而产生自相似的分形架构，称为“分形生成模型”。通过使用自回归模型作为原子模块提供了一个演示，并在逐像素图像生成的挑战性任务上进行了检验，表现出强大的似然估计和生成质量。这项工作旨在介绍一种新的生成建模范例，为进一步探索奠定基础。
    <ul>
</details>

**Main Methods:**

1.  **Recursive Generative Model Construction:** Builds a generative model by recursively invoking atomic generative modules (e.g., autoregressive models) within itself.
2.  **Fractal Architecture:** Creates a self-similar architecture across different levels of generative modules, inspired by the concept of fractals in mathematics.
3.  **Divide-and-Conquer Strategy:** Employs a divide-and-conquer approach to handle high-dimensional data by recursively partitioning the joint distribution and modeling each subset with atomic generative modules.
4.  **Autoregressive Image Generation:** Uses autoregressive models as the atomic generative modules to perform pixel-by-pixel image generation.
5.  **Transformer Networks:** Uses transformer networks in each level of the fractal architecture to model the dependencies between the generative modules.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><strong>递归生成模型构建：</strong>通过递归调用自身内部的原子生成模块（例如，自回归模型）来构建生成模型。</li>
        <li><strong>分形架构：</strong>创建一种在不同级别的生成模块中具有自相似性的架构，灵感来自数学中分形的概念。</li>
        <li><strong>分而治之策略：</strong>采用分而治之的方法，通过递归地划分联合分布并使用原子生成模块对每个子集进行建模来处理高维数据。</li>
        <li><strong>自回归图像生成：</strong>使用自回归模型作为原子生成模块来执行逐像素图像生成。</li>
        <li><strong>Transformer 网络：</strong>在分形架构的每一层中使用 Transformer 网络来建模生成模块之间的依赖关系。</li>
    </ul>
</details>

**Main Contributions:**

1.  **Novel Generative Modeling Paradigm:** Introduces a new paradigm for generative modeling based on the recursive invocation of atomic generative modules, resulting in fractal-like architectures.
2.  **Fractal Generative Model Framework:** Proposes a general framework for constructing fractal generative models and provides an instantiation using autoregressive models.
3.  **Strong Performance on Pixel-by-Pixel Image Generation:** Demonstrates that fractal generative models achieve strong performance on the challenging task of pixel-by-pixel image generation, both in terms of likelihood estimation and generation quality.
4.  **Computational Efficiency:** Shows that the proposed fractal architecture is computationally efficient due to its divide-and-conquer strategy, enabling the modeling of high-resolution images pixel-by-pixel.
5.  **Potential for Modeling Non-Sequential Data:** Suggests that fractal generative models are well-suited for modeling high-dimensional non-sequential data with intrinsic structures, extending beyond traditional sequence modeling approaches.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><strong>新的生成建模范例：</strong> 介绍了一种基于原子生成模块的递归调用的新的生成建模范例，从而产生类似分形的架构。</li>
        <li><strong>分形生成模型框架：</strong> 提出了一个用于构建分形生成模型的通用框架，并提供了一个使用自回归模型的实例。</li>
        <li><strong>在逐像素图像生成方面表现出色：</strong> 证明了分形生成模型在逐像素图像生成的挑战性任务中取得了出色的性能，无论是在似然估计还是生成质量方面。</li>
        <li><strong>计算效率：</strong> 表明所提出的分形架构由于其分而治之的策略而具有计算效率，从而可以逐像素地对高分辨率图像进行建模。</li>
        <li><strong>建模非序列数据的潜力：</strong> 建议分形生成模型非常适合建模具有内在结构的高维非序列数据，从而超越了传统的序列建模方法。</li>
    </ul>
</details>
