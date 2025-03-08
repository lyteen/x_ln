## Abstract

**Keywords:** Image debanding, Invertible networks, Banded deformable convolution

<details>
    <summary>关键词</summary>
    <ul>
        图像去条带，可逆网络，带状可变形卷积
    </ul>
</details>

**Abstract:**
Banding artifacts in images, stemming from color bit depth limitations, image compression, or over-editing, significantly degrade image quality, particularly in regions with smooth gradients. Image debanding aims to eliminate these artifacts while preserving image details. This paper introduces a novel image debanding approach using a cross-scale invertible neural network (INN). The proposed INN is information-lossless and enhanced with a more effective cross-scale scheme. Furthermore, we present banded deformable convolution, effectively leveraging the anisotropic nature of banding. This technique is compact, efficient, and more generalizable than existing deformable convolution methods. Our proposed INN exhibits superior quantitative and qualitative performance, demonstrated by experimental results.

<details>
    <summary>摘要</summary>
    <ul>
        图像中的条带伪影源于颜色位深度限制、图像压缩或过度编辑，显著降低图像质量，尤其是在具有平滑梯度的区域。图像去条带旨在消除这些伪影，同时保留图像细节。本文介绍了一种新的图像去条带方法，使用跨尺度可逆神经网络（INN）。所提出的INN是信息无损的，并使用更有效的跨尺度方案进行了增强。此外，我们提出了带状可变形卷积，有效利用了条带的各向异性。这种技术紧凑、高效，并且比现有的可变形卷积方法更具泛化性。我们的实验结果表明，所提出的INN在定量指标和视觉质量方面都表现出卓越的性能。
    </ul>
</details>

**Main Methods:**

*   **Cross-Scale Invertible Neural Network (INN):** Employs an INN with a cross-scale architecture for information-lossless image decomposition, separating banding patterns from image details.
*   **Banded Deformable Convolution:** Introduces a novel deformable convolution technique tailored for detecting and processing the anisotropic patterns of banding artifacts.
*   **Image Decomposition:** Frames image debanding as an image decomposition problem, separating the input image into a banding pattern layer and a latent image layer.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>**跨尺度可逆神经网络 (INN):** 采用具有跨尺度架构的 INN 进行信息无损图像分解，将条带图案与图像细节分离。</li>
        <li>**带状可变形卷积:** 引入一种新的可变形卷积技术，专门用于检测和处理条带伪影的各向异性模式。</li>
        <li>**图像分解:** 将图像去条带处理视为图像分解问题，将输入图像分解为条带图案层和潜在图像层。</li>
    </ul>
</details>

**Main Contributions:**

*   **Decomposition-Based INN:** Proposes a novel decomposition-based deep invertible model with a cross-scale coupling structure for image debanding, effectively distinguishing between image details and banding patterns across multiple scales.
*   **Banded Deformable Convolution:** Develops a banded deformable convolution technique for efficient detection and processing of anisotropic banding patterns, improving computational efficiency and maintaining a compact model size.
*   **State-of-the-Art Performance:** Demonstrates superior performance in quantitative metrics and visual quality compared to existing debanding filters.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>**基于分解的 INN:** 提出了一种新的基于分解的深度可逆模型，该模型具有跨尺度耦合结构，用于图像去条带，有效区分了图像细节和跨多个尺度的条带图案。</li>
        <li>**带状可变形卷积:** 开发了一种带状可变形卷积技术，用于有效检测和处理各向异性条带图案，提高计算效率并保持紧凑的模型尺寸。</li>
        <li>**最先进的性能:** 与现有的去条带滤波器相比，在定量指标和视觉质量方面都表现出卓越的性能。</li>
    </ul>
</details>
