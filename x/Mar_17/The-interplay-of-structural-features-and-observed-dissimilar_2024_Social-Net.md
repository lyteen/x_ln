## Abstract

**Keywords:** Centrality, Correlation, Network topology, Threshold graphs

<details>
    <summary>关键词</summary>
    <ul>
        中心性, 相关性, 网络拓扑, 阈值图
    <ul>
</details>

**Abstract:** An abundance of centrality indices has been proposed which capture the importance of nodes in a network based on different structural features. While there remains a persistent belief that similarities in outcomes of indices is contingent on their technical definitions, a growing body of research shows that structural features affect observed similarities more than technicalities. We conduct a series of experiments on artificial networks to trace the influence of specific structural features on the similarity of indices which confirm previous results in the literature. Our analysis on 1163 real-world networks, however, shows that little of the observations on synthetic networks convincingly carry over to empirical settings. Our findings suggest that although it seems clear that (dis)similarities among centralities depend on structural properties of the network, using correlation-type analyses do not seem to be a promising approach to uncover such connections.

<details>
    <summary>摘要</summary>
    <ul>
        人们提出了大量的中心性指标，这些指标基于不同的结构特征来捕捉网络中节点的重要性。虽然仍然存在一种持久的信念，认为指标结果的相似性取决于其技术定义，但越来越多的研究表明，结构特征比技术细节更能影响观察到的相似性。我们进行了一系列人工网络实验，以追踪特定结构特征对指标相似性的影响，从而证实了文献中先前的结果。然而，我们对 1163 个真实世界网络的分析表明，在合成网络中观察到的现象很少能令人信服地转移到经验环境中。我们的研究结果表明，虽然中心性之间的（不）相似性似乎取决于网络的结构属性，但使用相关类型的分析似乎并不是一种很有希望的方法来揭示这些联系。
    <ul>
</details>

**Main Methods:**

*   **Experiments on Artificial Networks:**  The study employs experiments on artificial networks to investigate the influence of specific structural features on the similarity of centrality indices.
*   **Large-Scale Analysis on Real-World Networks:** A large-scale analysis is conducted on 1163 real-world networks from diverse backgrounds to analyze connections between indices and structural features.
*   **Rank Dissimilarity:** A new measure of rank dissimilarity (fraction of discordant pairs) is used instead of correlation indices to assess the (dis)similarity among indices.
*   **Statistical Entropy Analysis:** Statistical entropy analysis is used to find the best structural predictors for high (or low) rank dissimilarities among centrality indices.
*   **Edge Rewiring Experiment:** This experiment is used to connect the distance of a network to its closest threshold graph and rank dissimilarities to demonstrate that density can be completely independent of rank dissimilarities.
*   **Simulated Annealing for High Discordance:** Utilized an adapted simulated annealing algorithm to construct graphs that maximize rank dissimilarity between chosen indices.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><b>人工网络实验：</b> 该研究采用人工网络实验，以研究特定结构特征对中心性指标相似性的影响。</li>
        <li><b>大规模真实世界网络分析：</b> 对来自不同背景的 1163 个真实世界网络进行了大规模分析，以分析指标和结构特征之间的联系。</li>
        <li><b>等级差异性：</b> 使用等级差异性（不一致对的分数）这一新指标来代替相关性指标，以评估指标间的（不）相似性。</li>
        <li><b>统计熵分析：</b> 使用统计熵分析来寻找中心性指标之间高（或低）等级差异性的最佳结构预测因子。</li>
        <li><b>边缘重连实验：</b> 该实验用于连接网络与其最接近的阈值图的距离，并连接等级差异性，以证明密度可以完全独立于等级差异性。</li>
        <li><b>模拟退火算法用于高不一致性：</b> 利用改进的模拟退火算法来构建图，从而最大化所选指标之间的等级差异性。</li>
    <ul>
</details>

**Main Contributions:**

*   **Re-examination of Existing Results:**  The study re-examines existing results connecting structural features with the correlation among centrality indices using a different set of analytical tools.
*   **Novel Approach to Rank Dissimilarity:** A new measure of rank dissimilarity is introduced and used to assess relationships among centrality indices.
*   **Focus on Discordance:** The work shifts the focus from correlation to discordance (rank dissimilarity) to analyze relationships, offering a different perspective.
*   **Limitations of Correlation-Type Analyses:**  The study argues that correlation-type analyses may not be a promising approach to uncover connections between network structure and dissimilarities of indices.
*   **Importance of Betweenness Distinction:** The study highlights the importance of the distinction between betweenness and other centrality measures.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><b>对现有结果的重新审视：</b> 该研究使用不同的分析工具重新检查了连接结构特征与中心性指标之间相关性的现有结果。</li>
        <li><b>等级差异性新方法：</b> 引入了一种新的等级差异性度量方法，用于评估中心性指标之间的关系。</li>
        <li><b>关注不一致性：</b> 该研究将重点从相关性转移到不一致性（等级差异性），以分析关系，提供了不同的视角。</li>
        <li><b>相关类型分析的局限性：</b> 该研究认为，相关类型分析可能并不是一种很有希望的方法来揭示网络结构与指标差异性之间的联系。</li>
        <li><b>介数区分的重要性：</b> 该研究强调了区分介数与其他中心性指标的重要性。</li>
    <ul>
</details>
