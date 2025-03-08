## Abstract

**Keywords:** Self-supervised learning (SSL), Anomaly detection, Deep prior, Low-rank representation (LRR), Hyperspectral image (HSI)

<details>
    <summary>关键词</summary>
    <ul>
        自监督学习 (SSL), 异常检测, 深度先验, 低秩表示 (LRR), 高光谱图像 (HSI)
    <ul>
</details>

**Abstract:**
Hyperspectral anomaly detection (HAD) can identify and locate the targets without any known information and is widely applied in Earth observation and military fields. The majority of existing HAD methods use the low-rank representation (LRR) model to separate the background and anomaly through mathematical optimization, in which the anomaly is optimized with a handcrafted sparse prior (e.g., 12,1-norm). However, this may not be ideal since they overlook the spatial structure present in anomalies and make the detection result largely dependent on manually set sparsity. To tackle these problems, we redefine the optimization criterion for the anomaly in the LRR model with a self-supervised network called self-supervised anomaly prior (SAP). This prior is obtained by the pretext task of self-supervised learning, which is customized to learn the characteristics of hyperspectral anomalies. Specifically, this pretext task is a classification task to distinguish the original hyperspectral image (HSI) and the pseudo-anomaly HSI, where the pseudo-anomaly is generated from the original HSI and designed as a prism with arbitrary polygon bases and arbitrary spectral bands. In addition, a dual-purified strategy is proposed to provide a more refined background representation with an enriched background dictionary, facilitating the separation of anomalies from complex backgrounds. Extensive experiments on various hyperspectral datasets demonstrate that the proposed SAP offers a more accurate and interpretable solution than other advanced HAD methods.

<details>
    <summary>摘要</summary>
    <ul>
        高光谱异常检测 (HAD) 可以在没有任何已知信息的情况下识别和定位目标，并广泛应用于地球观测和军事领域。 大多数现有的 HAD 方法使用低秩表示 (LRR) 模型，通过数学优化来分离背景和异常，其中异常使用手工制作的稀疏先验（例如，12,1 范数）进行优化。 然而，这可能并不理想，因为它们忽略了异常中存在的空间结构，并使检测结果很大程度上依赖于手动设置的稀疏性。 为了解决这些问题，我们使用一个名为自监督异常先验 (SAP) 的自监督网络来重新定义 LRR 模型中异常的优化标准。 这种先验是通过自监督学习的预训练任务获得的，该任务经过定制以学习高光谱异常的特征。 具体来说，这个预训练任务是一个分类任务，用于区分原始高光谱图像 (HSI) 和伪异常 HSI，其中伪异常是从原始 HSI 生成的，并设计为具有任意多边形底面和任意光谱带的棱镜。 此外，还提出了一种双重提纯策略，以提供更精细的背景表示，并使用丰富的背景字典，从而促进从复杂背景中分离异常。 对各种高光谱数据集进行的大量实验表明，与其它先进的 HAD 方法相比，所提出的 SAP 提供了一种更准确且更易于解释的解决方案。
    <ul>
</details>

**Main Methods:**

1.  **Self-Supervised Anomaly Prior (SAP):** A self-supervised deep neural network (DNN) module is designed as the anomaly prior in the LRR model.  The pretext task for self-supervised learning is an image-level classification task to distinguish between the original HSI and HSI with pseudo-anomalies.
2.  **Pseudo-Anomaly Generation:**  Pseudo-anomalies are generated from the original HSI, customized to comprehensively consider both the sparsity and spatial characteristics of hyperspectral anomalies. They are designed as prisms with arbitrary polygon bases and arbitrary spectral bands, cropped and pasted at arbitrary spatial and spectral positions.
3.  **Low-Rank Representation (LRR) Model:**  The LRR model is used to separate the background and anomaly by solving a mathematical optimization problem. The optimization criterion for the anomaly is redefined with the SAP network.
4.  **Dual-Purified Strategy:** A dual-purified strategy is proposed for background dictionary construction.  This involves removing the class with the fewest spectral vectors and spectral vectors with low probabilities from the LREN classification results to create an enriched background dictionary. The LREN classifies pixels through a Gaussian Mixture Model.
5. **Adaptive Denoising Network: The low rank constraint is enforced via using a denoising network FFDNet, with adaptive noise estimation.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><b>自监督异常先验 (SAP):</b>  设计了一个自监督深度神经网络 (DNN) 模块作为 LRR 模型中的异常先验。自监督学习的预训练任务是一个图像级分类任务，用于区分原始 HSI 和带有伪异常的 HSI。</li>
        <li><b>伪异常生成:</b> 伪异常从原始 HSI 生成，经过定制，全面考虑了高光谱异常的稀疏性和空间特征。它们被设计成具有任意多边形底面和任意光谱带的棱镜，并在任意空间和光谱位置进行裁剪和粘贴。</li>
        <li><b>低秩表示 (LRR) 模型:</b>  LRR 模型通过解决数学优化问题来分离背景和异常。利用 SAP 网络重新定义异常的优化准则。</li>
        <li><b>双重提纯策略:</b> 提出了一种双重提纯策略用于背景字典构建。这涉及到从 LREN 分类结果中移除具有最少光谱向量的类和具有低概率的光谱向量，以创建一个丰富的背景字典。LREN通过高斯混合模型对像素进行分类。</li>
    	<li><b>自适应降噪网络：</b> 通过使用具有自适应噪声估计的降噪网络 FFDNet 来强制执行低秩约束。</li>
    <ul>
</details>

**Main Contributions:**

1.  **First Application of Deep Learning for Anomaly Prior in HAD:** SAP represents the first instance of leveraging deep learning to obtain an anomaly prior within the HAD field. This considers both spatial and spectral characteristics of anomalies and operates independently of manually set parameters.
2.  **Customized Pretext Task for Generalizable Anomaly Prior:** A novel pretext task is designed for self-supervised learning to distinguish between original and pseudo-anomaly HSIs, enabling the learned prior to generalize to diverse anomalies without requiring labels or extensive data.
3.  **Dual-Purified Background Dictionary Construction:** The dual-purified strategy provides a refined background representation, mitigating anomaly contamination in the background dictionary.
4. **State-of-the-Art Results:**  Extensive experiments on real hyperspectral datasets from different sensors demonstrate that the proposed SAP offers state-of-the-art performance.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><b>深度学习在 HAD 中首次用于异常先验:</b> SAP 代表了在 HAD 领域中利用深度学习来获得异常先验的第一个实例。这既考虑了异常的空间和光谱特征，又独立于手动设置的参数运行。</li>
        <li><b>用于可泛化异常先验的定制预训练任务：</b> 设计了一种新颖的自监督学习预训练任务，用于区分原始 HSI 和伪异常 HSI，使学习到的先验能够泛化到不同的异常，而无需标签或大量数据。</li>
        <li><b>双重提纯背景字典构建：</b> 双重提纯策略提供了一种精细的背景表示，减轻了背景字典中的异常污染。</li>
        <li><b>最先进的结果：</b> 对来自不同传感器的真实高光谱数据集进行的大量实验表明，所提出的 SAP 提供了最先进的性能。</li>
    <ul>
</details>
