## Abstract

**Keywords:** Graph-level anomaly detection, Graph transformer, Graph neural networks, Unsupervised graph representation learning

<details>
    <summary>关键词</summary>
    <ul>
        图级别异常检测、图变换器、图神经网络、无监督图表示学习
    <ul>
</details>

**Abstract:** 
Graph-Level Anomaly Detection (GLAD) aims to identify anomalous graphs deviating from the normal data distribution. Existing GLAD methods often rely on Graph Neural Networks (GNNs) for graph-level representations, but GNNs' limited receptive field may miss crucial anomaly information. Also, inadequate modeling of cross-graph relationships restricts the discovery of inter-graph anomaly patterns. This paper proposes a Dual-View Graph-of-Graph Representation Learning Network for unsupervised GLAD, considering both intra-graph and inter-graph perspectives. A Graph Transformer enhances GNNs' receptive field by using both attribute and structural information for better intra-graph mining. A Graph-of-Graph-based dual-view representation learning network captures cross-graph dependencies, and attribute/structure-based graph-of-graph representations facilitate a broader understanding of graph relationships. Anomaly scores from multiple perspectives quantify the degree of anomalies. Experiments on benchmark datasets demonstrate the method's effectiveness in anomaly detection.

<details>
    <summary>摘要</summary>
    <ul>
        图级别异常检测 (GLAD) 旨在识别偏离正常数据分布的异常图。现有的 GLAD 方法通常依赖于图神经网络 (GNN) 来进行图级别的表示，但 GNN 有限的感受野可能会遗漏关键的异常信息。此外，对跨图关系的不充分建模限制了对图间异常模式的发现。本文提出了一种用于无监督 GLAD 的双视图图的图表示学习网络，同时考虑了图内和图间的视角。图变换器通过使用属性和结构信息来增强 GNN 的感受野，以实现更好的图内挖掘。基于图的图的双视图表示学习网络捕获了跨图依赖关系，基于属性/结构的图的图表示有助于更广泛地理解图关系。来自多个角度的异常分数量化了异常程度。对基准数据集的实验证明了该方法在异常检测方面的有效性。
    <ul>
</details>

**Main Methods:**
*   **Transformer-Enhanced Graph Encoder:** Combines GNNs and Graph Transformers with structural positional encoding to expand the receptive field and sensitivity to structural anomalies.
*   **Dual-View Graph-of-Graph Representation Learning:** Explicitly models inter-graph relationships using structural and attribute information to construct cross-graph relationships.
*   **Intra-Graph Reconstruction Loss:** Refines the model by focusing on the assumption that anomalous nodes are difficult to reconstruct.
*   **Inter-Graph Reconstruction Loss:** Identifies anomalies spanning multiple graphs.
*   **Multi-Perspective Anomaly Scoring:** Combines anomaly scores from intra-graph and inter-graph perspectives for a more comprehensive anomaly assessment.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><b>Transformer增强图编码器：</b> 结合 GNN 和图 Transformer 以及结构位置编码，以扩大感受野和对结构异常的敏感性。</li>
        <li><b>双视图图的图表示学习：</b> 使用结构和属性信息显式建模图间关系，以构建跨图关系。</li>
        <li><b>图内重建损失：</b> 通过关注异常节点难以重建的假设来完善模型。</li>
        <li><b>图间重建损失：</b> 识别跨多个图的异常。</li>
        <li><b>多视角异常评分：</b> 结合来自图内和图间视角的异常分数，以实现更全面的异常评估。</li>
    </ul>
</details>

**Main Contributions:**

*   **Transformer-Enhanced Graph Encoder:** Integrates Transformer architecture with structural positional encoding to augment the representational power of traditional GNNs.
*   **Dual-View Graph-of-Graph Representation Learning:** Represents an initial attempt to explicitly model inter-graph information, utilizing both structural and attribute information from graphs.
*   **Improved GLAD Performance:** Demonstrated greater performance in detecting anomalies by considering both intra-graph and inter-graph perspectives, validated on multiple benchmark datasets.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><b>Transformer 增强图编码器：</b> 将 Transformer 架构与结构位置编码相结合，以增强传统 GNN 的表示能力。</li>
        <li><b>双视图图的图表示学习：</b> 代表了显式建模图间信息的初步尝试，利用来自图的结构和属性信息。</li>
        <li><b>改进的 GLAD 性能：</b> 通过考虑图内和图间视角来证明在检测异常方面的更高性能，并在多个基准数据集上进行了验证。</li>
    </ul>
</details>
