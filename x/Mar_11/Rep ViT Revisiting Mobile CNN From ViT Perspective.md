Okay, here's the abstract generated from the provided OCR text:

## Abstract

**Keywords:** Lightweight CNN, Vision Transformer, Mobile Devices, Re-parameterization, ImageNet, Edge Deployment.

<details>
  <summary>关键词</summary>
  轻量级 CNN, 视觉 Transformer, 移动设备, 重参数化, ImageNet, 边缘部署.
</details>

**Abstract:**
Recently, lightweight Vision Transformers (ViTs) have shown superior performance and lower latency than lightweight Convolutional Neural Networks (CNNs) on mobile devices. This paper revisits the design of efficient lightweight CNNs from a ViT perspective. By incrementally enhancing MobileNetV3 with efficient architectural designs from lightweight ViTs, the authors propose a new family of pure lightweight CNNs called RepViT. Extensive experiments demonstrate that RepViT outperforms state-of-the-art lightweight ViTs, achieving over 80% top-1 accuracy on ImageNet with 1.0 ms latency on an iPhone 12. RepViT also demonstrates a nearly 10x faster inference speed compared to the advanced MobileSAM when combined with SAM. The authors hope RepViT serves as a strong baseline for lightweight models on edge devices.

<details>
  <summary>摘要</summary>
  最近，轻量级视觉 Transformer (ViT) 在移动设备上展现出比轻量级卷积神经网络 (CNN) 更优越的性能和更低的延迟。本文从 ViT 的角度重新审视了高效轻量级 CNN 的设计。通过整合轻量级 ViT 的高效架构设计来逐步增强 MobileNetV3，作者提出了一种新的纯轻量级 CNN 系列，名为 RepViT。大量实验表明，RepViT 优于最先进的轻量级 ViT，在 iPhone 12 上以 1.0 毫秒的延迟在 ImageNet 上实现了超过 80% 的 top-1 准确率。RepViT 与 SAM 结合使用时，与先进的 MobileSAM 相比，推理速度也提高了近 10 倍。作者希望 RepViT 可以作为边缘设备上轻量级模型的强大基线。
</details>

**Main Methods:**

1.  **Revisiting Mobile CNN Design:** The paper re-examines the design principles of lightweight CNNs, specifically focusing on architectural differences between lightweight CNNs and Vision Transformers.
2.  **Incremental Enhancement:** The authors enhance a standard lightweight CNN (MobileNetV3) by integrating efficient architectural designs from lightweight ViTs.
3.  **MetaFormer Structure Adoption:** RepViT leverages a ViT-like MetaFormer structure using re-parameterization convolutions to improve performance while maintaining efficiency.
4.  **Macro and Micro Design Optimizations:** Various macro (stem, downsampling layers, classifier, overall stage ratio) and micro (kernel size, SE layer placement) architectural elements are carefully optimized for mobile devices.
5.  **Structural Re-parameterization:** The approach utilizes structural re-parameterization in RepViT blocks, enabling enhanced model learning during training while eliminating computational costs during inference.

<details>
  <summary>主要方法</summary>
  <ol>
    <li><strong>重新审视移动 CNN 设计：</strong> 本文重新审视了轻量级 CNN 的设计原则，特别关注轻量级 CNN 和视觉 Transformer 之间的架构差异。</li>
    <li><strong>增量增强：</strong> 作者通过整合轻量级 ViT 的高效架构设计来增强标准的轻量级 CNN (MobileNetV3)。</li>
    <li><strong>MetaFormer 结构采用：</strong> RepViT 利用类似于 ViT 的 MetaFormer 结构，使用重参数化卷积来提高性能，同时保持效率。</li>
    <li><strong>宏观和微观设计优化：</strong> 对各种宏观（stem、下采样层、分类器、整体 stage 比例）和微观（核大小、SE 层位置）架构元素进行了精心优化，以适应移动设备。</li>
    <li><strong>结构重参数化：</strong> 该方法在 RepViT 块中利用结构重参数化，从而在训练期间实现增强的模型学习，同时消除推理期间的计算成本。</li>
  </ol>
</details>

**Main Contributions:**

1.  **RepViT: A New CNN Architecture:** A new family of pure lightweight CNNs (RepViT) is introduced, specifically designed for resource-constrained mobile devices.
2.  **Superior Performance and Efficiency:** RepViT demonstrates superior performance and efficiency compared to existing state-of-the-art lightweight ViTs on image classification and other computer vision tasks.
3.  **First Lightweight Model Achieving 80% Top-1 Accuracy at 1ms on iPhone:** RepViT is the first lightweight model to achieve over 80% top-1 accuracy on ImageNet with only 1.0 ms latency on an iPhone 12.
4.  **Fast Inference with SAM:** Combining RepViT with SAM achieves a nearly 10x faster inference speed than MobileSAM.
5.  **Strong Baseline for Edge Deployment:**  RepViT is proposed as a strong baseline model for lightweight models deployed on edge devices, encouraging further research in this area.

<details>
  <summary>主要贡献</summary>
  <ol>
    <li><strong>RepViT：一种新的 CNN 架构：</strong> 引入了一种新的纯轻量级 CNN 系列 (RepViT)，专为资源受限的移动设备设计。</li>
    <li><strong>卓越的性能和效率：</strong> RepViT 在图像分类和其他计算机视觉任务上表现出比现有的最先进的轻量级 ViT 更卓越的性能和效率。</li>
    <li><strong>首个在 iPhone 上以 1 毫秒延迟实现 80% Top-1 准确率的轻量级模型：</strong> RepViT 是首个在 iPhone 12 上仅以 1.0 毫秒的延迟在 ImageNet 上实现超过 80% top-1 准确率的轻量级模型。</li>
    <li><strong>使用 SAM 快速推理：</strong> 将 RepViT 与 SAM 相结合，与 MobileSAM 相比，实现了近 10 倍的推理速度。</li>
    <li><strong>边缘部署的强大基线：</strong> 提出 RepViT 作为部署在边缘设备上的轻量级模型的强大基线模型，鼓励该领域的进一步研究。</li>
  </ol>
</details>
