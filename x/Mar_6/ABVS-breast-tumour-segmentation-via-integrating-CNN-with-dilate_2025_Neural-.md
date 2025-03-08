## Abstract
**Keywords:** Breast tumour segmentation, Automated breast volume scanner, 3D Transformer-CNN segmentation network, Mask image modelling

<details>
    <summary>关键词</summary>
    <ul>
        乳腺肿瘤分割, 自动化乳腺体积扫描仪, 三维Transformer-CNN分割网络, 掩码图像建模
    <ul>
</details>

**Main Methods:**

This paper introduces a novel 3D segmentation network (DST-C) for Automated Breast Volume Scanner (ABVS) images.  Key methods include:

1.  **Dual-Branch Architecture:** Combines a CNN branch for detailed local feature extraction and a dilated sampling self-attention Transformer (DST) branch for global feature capture.
2.  **Dilated Sampling Self-Attention:** Reformulates Swin Transformer (ST) with dilated sampling to enhance the receptive field and reduce computational cost.
3.  **Spatial-Channel Attention (SCA) Interactive Bridge:**  A novel module connects the CNN and Transformer branches, fusing local and global features effectively, with spatial features guiding global representations.
4.  **Dual-Path Mask Image Modelling:** Self-supervised learning based on masking and reconstruction, pretraining both the CNN and Transformer encoders.
5.  **Adaptive Postprocessing:**  A unique postprocessing method reduces false positives and improves sensitivity by leveraging local range region growth with an adaptive threshold.

<details>
    <summary>主要方法</summary>
        <ul>
            <li><b>双分支架构：</b>结合CNN分支以提取详细的局部特征，以及使用扩张采样自注意力机制的Transformer (DST)分支以捕获全局特征。</li>
            <li><b>扩张采样自注意力：</b>通过扩张采样重新构建Swin Transformer (ST)，以增强感受野并降低计算成本。</li>
            <li><b>空间-通道注意力(SCA)交互桥：</b>一个连接CNN和Transformer分支的新模块，有效地融合局部和全局特征，通过空间特征引导全局表示。</li>
            <li><b>双路径掩码图像建模：</b>基于掩码和重建的自监督学习，预训练CNN和Transformer编码器。</li>
            <li><b>自适应后处理：</b>一种独特的后处理方法，通过利用具有自适应阈值的局部范围区域增长来减少假阳性并提高灵敏度。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Dilated Sampling Transformer:** Introduces dilated sampling to Swin Transformer for improved receptive field and efficiency in 3D medical volumes.
2.  **Parallel Interactive CNN-Transformer Structure:** Proposes a novel parallel architecture with SCA to effectively fuse local and global features from CNN and Transformer branches.
3.  **Dual-Path Mask Image Modelling Pretraining:** Develops a self-supervised pretraining strategy to enhance feature extraction for 3D segmentation, mitigating annotation scarcity.
4.  **State-of-the-Art Performance:** Achieves promising 3D segmentation and detection performance on an in-house ABVS dataset, demonstrating competitive results on KiTS19 and TDSC-ABUS 2023 datasets.
5.  **Postprocessing innovation:** The postprocessing strategy significantly enhances the sensitivity and specificity of tumor detection.


<details>
    <summary>主要贡献</summary>
        <ul>
            <li><b>扩张采样Transformer：</b>将扩张采样引入Swin Transformer，以提高三维医学体积的感受野和效率。</li>
            <li><b>并行交互式 CNN-Transformer 结构：</b>提出了一种新的并行架构，具有 SCA，可以有效地融合来自 CNN 和 Transformer 分支的局部和全局特征。</li>
            <li><b>双路径掩码图像建模预训练：</b>开发了一种自监督预训练策略，以增强三维分割的特征提取，从而减轻注释稀疏性。</li>
            <li><b>最先进的性能：</b>在内部 ABVS 数据集上实现了有希望的 3D 分割和检测性能，证明了 KiTS19 和 TDSC-ABUS 2023 数据集上的竞争结果。</li>
            <li><b>后处理创新：</b>后处理策略显着提高了肿瘤检测的灵敏度和特异性。</li>
        </ul>
</details>

```English
## Abstract

**Keywords:** Breast tumour segmentation, Automated breast volume scanner, 3D Transformer-CNN segmentation network, Mask image modelling

**Main Methods:**

This paper introduces a novel 3D segmentation network (DST-C) for Automated Breast Volume Scanner (ABVS) images.  Key methods include:

1.  **Dual-Branch Architecture:** Combines a CNN branch for detailed local feature extraction and a dilated sampling self-attention Transformer (DST) branch for global feature capture.
2.  **Dilated Sampling Self-Attention:** Reformulates Swin Transformer (ST) with dilated sampling to enhance the receptive field and reduce computational cost.
3.  **Spatial-Channel Attention (SCA) Interactive Bridge:**  A novel module connects the CNN and Transformer branches, fusing local and global features effectively, with spatial features guiding global representations.
4.  **Dual-Path Mask Image Modelling:** Self-supervised learning based on masking and reconstruction, pretraining both the CNN and Transformer encoders.
5.  **Adaptive Postprocessing:**  A unique postprocessing method reduces false positives and improves sensitivity by leveraging local range region growth with an adaptive threshold.

**Main Contributions:**

1.  **Dilated Sampling Transformer:** Introduces dilated sampling to Swin Transformer for improved receptive field and efficiency in 3D medical volumes.
2.  **Parallel Interactive CNN-Transformer Structure:** Proposes a novel parallel architecture with SCA to effectively fuse local and global features from CNN and Transformer branches.
3.  **Dual-Path Mask Image Modelling Pretraining:** Develops a self-supervised pretraining strategy to enhance feature extraction for 3D segmentation, mitigating annotation scarcity.
4.  **State-of-the-Art Performance:** Achieves promising 3D segmentation and detection performance on an in-house ABVS dataset, demonstrating competitive results on KiTS19 and TDSC-ABUS 2023 datasets.
5.  **Postprocessing innovation:** The postprocessing strategy significantly enhances the sensitivity and specificity of tumor detection.
```

<details>
    <summary>中文摘要</summary>
    给出乳腺癌发病率的快速增长，开发自动化乳腺体积扫描仪 (ABVS) 是为了高效准确地筛查乳腺肿瘤。 然而，由于乳腺肿瘤的大小和形状差异很大，因此审查 ABVS 图像是一项具有挑战性的任务。 我们提出了一种新颖的 3D 分割网络（即 DST-C），该网络将卷积神经网络 (CNN) 与扩张采样自注意力 Transformer (DST) 相结合。 在我们的网络中，从 DST 分支提取的全局特征由 CNN 分支提供的详细局部信息引导，该分支适应肿瘤大小和形态的多样性。 对于医学图像，尤其是 ABVS 图像，注释的稀缺导致模型训练困难。 因此，引入了一种基于双路径方法的用于掩码图像建模的自监督学习方法，以生成有价值的图像表示。 此外，提出了一种独特的后处理方法，以降低假阳性率并同时提高灵敏度。 实验结果表明，我们的模型在使用内部数据集时实现了有希望的 3D 分割和检测性能。 我们的代码可在以下网址获得：https://github.com/magnetliu/dstc-net。
</details>
