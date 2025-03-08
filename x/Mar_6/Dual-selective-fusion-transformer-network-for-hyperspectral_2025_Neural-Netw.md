## Abstract

**Keywords:** Hyperspectral image classification, Transformer, Receptive field, Self-attention, Spatial-spectral joint

<details>
    <summary>关键词</summary>
    <ul>
        高光谱图像分类, Transformer, 感受野, 自注意力, 空间-光谱联合
    </ul>
</details>

**Abstract:** Transformer has achieved satisfactory results in the field of hyperspectral image (HSI) classification. However, existing Transformer models face two key challenges when dealing with HSI scenes characterized by diverse land cover types and rich spectral information: (1) A fixed receptive field overlooks the effective contextual scales required by various HSI objects; (2) invalid self-attention features in context fusion affect model performance. To address these limitations, we propose a novel Dual Selective Fusion Transformer Network (DSFormer) for HSI classification. DSFormer achieves joint spatial and spectral contextual modeling by flexibly selecting and fusing features across different receptive fields, effectively reducing unnecessary information interference by focusing on the most relevant spatial-spectral tokens. Specifically, we design a Kernel Selective Fusion Transformer Block (KSFTB) to learn an optimal receptive field by adaptively fusing spatial and spectral features across different scales, enhancing the model's ability to accurately identify diverse HSI objects. Additionally, we introduce a Token Selective Fusion Transformer Block (TSFTB), which strategically selects and combines essential tokens during the spatial-spectral self-attention fusion process to capture the most crucial contexts. Extensive experiments conducted on four benchmark HSI datasets demonstrate that the proposed DSFormer significantly improves land cover classification accuracy, outperforming existing state-of-the-art methods. Specifically, DSFormer achieves overall accuracies of 96.59%, 97.66%, 95.17%, and 94.59% in the Pavia University, Houston, Indian Pines, and Whu-HongHu datasets, respectively, reflecting improvements of 3.19%, 1.14%, 0.91%, and 2.80% over the previous model. The code will be available online at https://github.com/YichuXu/DSFormer.

<details>
    <summary>摘要</summary>
    <ul>
        Transformer在高光谱图像（HSI）分类领域取得了令人满意的结果。然而，现有的Transformer模型在处理以多样化的地物类型和丰富的光谱信息为特征的HSI场景时，面临着两个关键挑战：（1）固定的感受野忽略了各种HSI对象所需的有效上下文尺度；（2）上下文中无效的自注意力特征会影响模型性能。为了解决这些限制，我们提出了一种用于HSI分类的新型双重选择性融合Transformer网络（DSFormer）。DSFormer通过灵活地选择和融合不同感受野中的特征来实现联合空间和光谱上下文建模，有效地减少了不必要的信息干扰，从而专注于最相关的空间光谱标记。具体来说，我们设计了一个内核选择性融合Transformer块（KSFTB），通过自适应融合不同尺度的空间和光谱特征来学习最佳感受野，从而增强模型准确识别不同HSI对象的能力。此外，我们引入了一个令牌选择性融合Transformer块（TSFTB），该块在空间光谱自注意力融合过程中策略性地选择和组合基本令牌，以捕获最关键的上下文。在四个基准HSI数据集上进行的大量实验表明，所提出的DSFormer显著提高了地物分类精度，优于现有的最先进方法。具体来说，DSFormer在帕维亚大学、休斯顿、印第安松树和吴洪湖数据集上分别实现了96.59%、97.66%、95.17%和94.59%的总体精度，分别比之前的模型提高了3.19%、1.14%、0.91%和2.80%。代码将在https://github.com/YichuXu/DSFormer上在线提供。
    </ul>
</details>

**Main Methods:**

1.  **Kernel Selective Fusion Transformer Block (KSFTB):** This block learns an optimal receptive field by adaptively fusing spatial and spectral features across different scales, enhancing the model's ability to accurately identify diverse HSI objects.  It employs dilated depthwise convolutions to construct a large receptive field and a spatial-spectral selection mechanism to adaptively fuse appropriate receptive field sizes.
2.  **Token Selective Fusion Transformer Block (TSFTB):**  This block strategically selects and combines essential tokens during the spatial-spectral self-attention fusion process to capture the most crucial contexts. It uses a grouping strategy to retain HSI data characteristics and 3D convolutions to extract spatial and spectral tokens, selectively focusing on relevant tokens in self-attention.
3.  **Dual Selective Fusion Transformer Network (DSFormer) Architecture:** The overall architecture consists of convolutional layers, dual selective fusion transformer groups (DSFTGs) containing KSFTB and TSFTB, and a classification head. PCA is used for dimensionality reduction before processing the HSI data.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>**内核选择性融合Transformer块 (KSFTB):** 该模块通过自适应融合不同尺度的空间和光谱特征来学习最佳感受野，从而增强模型准确识别不同HSI对象的能力。它采用扩张深度卷积来构建大感受野，并采用空间光谱选择机制来自适应地融合合适的感受野大小。</li>
        <li>**令牌选择性融合Transformer块 (TSFTB):** 该模块在空间光谱自注意力融合过程中策略性地选择和组合基本令牌，以捕获最关键的上下文。它使用分组策略来保留 HSI 数据特征，并使用 3D 卷积来提取空间和光谱令牌，在自注意力中选择性地关注相关令牌。</li>
        <li>**双重选择性融合Transformer网络 (DSFormer) 架构:** 总体架构包括卷积层、包含 KSFTB 和 TSFTB 的双重选择性融合 Transformer 组 (DSFTG) 和分类头。在处理 HSI 数据之前，PCA 用于降维。</li>
    </ul>
</details>

**Main Contributions:**

1.  **Novel DSFormer Architecture:** A novel Dual Selective Fusion Transformer Network (DSFormer) is proposed for HSI classification, which adaptively selects and fuses features from diverse receptive fields.
2.  **KSFTB Design:**  The Kernel Selective Fusion Transformer Block (KSFTB) is designed to adaptively select and integrate appropriate contextual information, ensuring features are acquired within an optimal receptive field.
3.  **TSFTB Development:** The Token Selective Fusion Transformer Block (TSFTB) is developed to strategically select the most relevant tokens in the context fusion process, effectively integrating essential information.
4.  **State-of-the-Art Performance:** Extensive experiments on benchmark datasets demonstrate that DSFormer improves land cover classification accuracy compared to other methods.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>**新型 DSFormer 架构:** 提出了一种用于 HSI 分类的新型双重选择性融合 Transformer 网络 (DSFormer)，该网络自适应地选择和融合来自不同感受野的特征。</li>
        <li>**KSFTB 设计:** 设计了内核选择性融合 Transformer 块 (KSFTB) 来自适应地选择和整合适当的上下文信息，确保在最佳感受野内获取特征。</li>
        <li>**TSFTB 开发:** 开发了令牌选择性融合 Transformer 块 (TSFTB) 来策略性地选择上下文融合过程中最相关的令牌，从而有效地整合基本信息。</li>
        <li>**最先进的性能:** 在基准数据集上进行的大量实验表明，与其他方法相比，DSFormer 提高了地物分类精度。</li>
    </ul>
</details>
