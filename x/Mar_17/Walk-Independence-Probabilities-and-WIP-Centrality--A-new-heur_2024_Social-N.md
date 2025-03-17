## Abstract

**Keywords:** Network, diffusion, centrality, probability, information transmission, inclusion-exclusion principle, heuristic

<details>
    <summary>关键词</summary>
    <ul>
        网络, 扩散, 中心性, 概率, 信息传输, 容斥原理, 启发式算法
    </ul>
</details>

**Abstract:**
Calculating the true probability of signal transmission between any two nodes in a network is computationally hard. Diffusion centrality is often used as a heuristic but can lead to distorted results due to its failure to account for the inclusion-exclusion principle. This paper presents a new formula for node-to-node diffusion probabilities using De Morgan's laws to account for this principle. Like diffusion centrality, this formula assumes independence of signal travel probabilities along each walk and is called Walk-Independence Probabilities (WIP). These probabilities yield two new centrality measures: WIP centrality and blocking centrality, where the latter is calculated when some nodes block signals.

<details>
    <summary>摘要</summary>
    <ul>
        计算网络中任意两个节点之间信号传输的真实概率在计算上是困难的。扩散中心性经常被用作一种启发式方法，但由于未能考虑到容斥原理，可能导致结果失真。本文提出了一种新的节点到节点扩散概率公式，使用德摩根定律来解释这一原理。与扩散中心性一样，该公式假设信号沿网络中每条路径传播的概率是独立的，因此被称为路径独立概率（WIP）。这些概率产生了两种新的中心性度量：WIP 中心性和阻塞中心性，后者是在某些节点阻塞信号时计算的。
    </ul>
</details>

**Main Methods:**

1.  **Walk-Independence Probabilities (WIP):** Derives a new formula for approximating node-to-node diffusion probabilities in networks using De Morgan's laws to address the inclusion-exclusion principle.
2.  **WIP Centrality:** Constructs a new centrality measure based on the calculated Walk-Independence Probabilities.
3.  **Blocking Centrality:** Defines and calculates an induced centrality measure that quantifies the impact of certain nodes blocking signal transmission within the network.
4.  **Network Analysis:** Employs mathematical formulations and network examples to compare and contrast the behavior and results of WIP centrality with diffusion centrality under various network conditions.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><strong>路径独立概率（WIP）：</strong>推导出一种新的公式，用于近似计算网络中节点到节点的扩散概率，使用德摩根定律来解决容斥原理问题。</li>
        <li><strong>WIP 中心性：</strong>构建一种新的中心性度量，基于计算出的路径独立概率。</li>
        <li><strong>阻塞中心性：</strong>定义和计算一种诱导中心性度量，用于量化网络中特定节点阻塞信号传输的影响。</li>
        <li><strong>网络分析：</strong>采用数学公式和网络实例，在各种网络条件下比较和对比 WIP 中心性和扩散中心性的行为和结果。</li>
    </ul>
</details>

**Main Contributions:**

1.  **A New Heuristic for Diffusion Probabilities:** Provides a simple and computationally efficient heuristic for estimating node-to-node diffusion probabilities that accounts for the inclusion-exclusion principle, unlike traditional diffusion centrality.
2.  **WIP Centrality Measure:** Introduces a new centrality measure, WIP centrality, that utilizes the Walk-Independence Probabilities, offering a different perspective on node importance in network diffusion.
3.  **Blocking Centrality Measure:** Defines a blocking centrality measure to assess the impact of nodes blocking signal transmission, offering insights into negative influence in networks.
4.  **Comparison with Diffusion Centrality:** Demonstrates through examples and network analysis that diffusion centrality can overestimate inequality in diffusion capabilities, and shows how WIP centrality provides a more reliable and adaptable estimate.
5.  **Analysis of Blocking Effects:** Explores the effects of node blocking on network communication, linking visibility and blocking to network structure properties such as 2-connectedness and cycles.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><strong>扩散概率的新启发式方法：</strong>提供了一种简单且计算高效的启发式方法，用于估计节点到节点的扩散概率，该方法考虑了容斥原理，这与传统的扩散中心性不同。</li>
        <li><strong>WIP 中心性度量：</strong>引入了一种新的中心性度量，即 WIP 中心性，它利用路径独立概率，为网络扩散中节点的重要性提供了不同的视角。</li>
        <li><strong>阻塞中心性度量：</strong>定义了一种阻塞中心性度量，用于评估节点阻塞信号传输的影响，从而深入了解网络中的负面影响。</li>
        <li><strong>与扩散中心性的比较：</strong>通过示例和网络分析表明，扩散中心性可能会高估扩散能力的差异，并展示了 WIP 中心性如何提供更可靠和适应性更强的估计。</li>
        <li><strong>阻塞效应分析：</strong>探讨了节点阻塞对网络通信的影响，将可见性和阻塞与网络结构属性（如 2-连通性和环路）联系起来。</li>
    </ul>
</details>