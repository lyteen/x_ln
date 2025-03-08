## Abstract

**Keywords:** Multi-label image classification, Transformer, Pseudo-region, Feature erasure, Contrastive learning

<details>
    <summary>关键词</summary>
    <ul>
        多标签图像分类, Transformer, 伪区域, 特征擦除, 对比学习
    <ul>
</details>

**Abstract:**

This paper addresses challenges in Multi-Label Image Classification (MLIC) stemming from the loss of spatial information in Transformer-based methods and the neglect of potential useful features beyond salient regions. A Dual Relation Transformer Network (DRTN) is proposed, trained end-to-end, and utilizes a grid aggregation scheme to generate pseudo-region features. A Dual Relation Enhancement (DRE) module captures correlations using grid and pseudo-region features. A Feature Enhancement and Erasure (FEE) module learns discriminative features and mines for additional potential features via attention-based region-level erasure. A Contrastive Learning (CL) module encourages foreground feature similarity while separating foregrounds from backgrounds. Experimental results on MS-COCO 2014, PASCAL VOC 2007, and NUS-WIDE datasets demonstrate DRTN's superior performance.

<details>
    <summary>摘要</summary>
    <ul>
        本文旨在解决多标签图像分类（MLIC）中，基于Transformer的方法中空间信息丢失以及忽略显著区域之外潜在有用特征的问题。提出了一个双重关系Transformer网络（DRTN），该网络以端到端的方式训练，并利用网格聚合方案生成伪区域特征。双重关系增强（DRE）模块使用网格和伪区域特征捕获相关性。特征增强和擦除（FEE）模块学习区分性特征，并通过基于注意力的区域级擦除挖掘额外的潜在特征。对比学习（CL）模块鼓励前景特征相似性，同时将前景与背景分离。在MS-COCO 2014、PASCAL VOC 2007和NUS-WIDE数据集上的实验结果表明，DRTN具有优越的性能。
    <ul>
</details>

**Main Methods:**

*   **Grid Aggregation for Pseudo-Region Features:** Compensates for spatial information loss by generating pseudo-region features using a grid aggregation scheme, without requiring expensive object detector annotations.
*   **Dual Relation Enhancement (DRE) Module:** Captures correlations between objects using both grid and pseudo-region features, leveraging a dual visual feature approach.
*   **Feature Enhancement and Erasure (FEE) Module:** Learns discriminative features by enhancing salient regions and mines additional potential features by erasing the most salient features using an attention mechanism and region-level erasure.
*   **Contrastive Learning (CL) Module:** Encourages salient and potential foreground features to be closer to each other, while pushing them away from background features, to learn more comprehensively.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>**用于伪区域特征的网格聚合：**通过使用网格聚合方案生成伪区域特征，补偿空间信息丢失，而无需昂贵的目标检测器标注。</li>
        <li>**双重关系增强（DRE）模块：**利用网格和伪区域特征，通过双重视觉特征方法捕获对象之间的相关性。</li>
        <li>**特征增强和擦除（FEE）模块：**通过增强显著区域学习区分性特征，并使用注意力机制和区域级擦除，通过擦除最显著的特征挖掘额外的潜在特征。</li>
        <li>**对比学习（CL）模块：**鼓励显著和潜在的前景特征更接近彼此，同时将它们推离背景特征，以更全面地学习。</li>
    <ul>
</details>

**Main Contributions:**

*   **Dual Relation Transformer Network (DRTN):** A novel network architecture for MLIC that effectively integrates grid and pseudo-region features.
*   **Dual Relation Enhancement (DRE) Module:** Proposes a module to capture object correlations by combining grid and pseudo-region features.
*   **Feature Enhancement and Erasure (FEE) Module:** Introduces a mechanism to learn discriminative features and mine additional potential valuable features by strategically erasing salient feature regions.
*   **Contrastive Learning (CL) Module:** Devises a loss function to improve feature representation by bringing salient and potential foreground features closer while pushing them away from background features.
*   **State-of-the-art Performance:** Achieves superior performance on multiple benchmark MLIC datasets (MS-COCO 2014, PASCAL VOC 2007, NUS-WIDE).

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>**双重关系 Transformer 网络（DRTN）：**一种用于 MLIC 的新型网络架构，有效整合了网格和伪区域特征。</li>
        <li>**双重关系增强（DRE）模块：**提出了一个通过结合网格和伪区域特征来捕获对象相关性的模块。</li>
        <li>**特征增强和擦除（FEE）模块：**引入了一种机制，通过策略性地擦除显著特征区域，来学习区分性特征并挖掘额外的潜在价值特征。</li>
        <li>**对比学习（CL）模块：**设计了一种损失函数，通过拉近显著和潜在前景特征的同时将其推离背景特征，从而改善特征表示。</li>
        <li>**最先进的性能：**在多个基准 MLIC 数据集（MS-COCO 2014、PASCAL VOC 2007、NUS-WIDE）上实现了卓越的性能。</li>
    <ul>
</details>
