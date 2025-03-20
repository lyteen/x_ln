## Abstract

**Keywords:** Personal networks, structural typology, dimensionality reduction, clustering, social networks analysis

<details>
   <summary>关键词</summary>
    <ul>
        个人网络，结构类型学，降维，聚类，社会网络分析
    </ul>
</details>

**Abstract:**
This study presents a method to uncover the fundamental dimensions of structural variability in Personal Networks (PNs) and develop a classification solely based on these structural properties. We address the limitations of previous literature and lay the foundation for a rigorous methodology to construct a Structural Typology of PNs. We test our method with a dataset of nearly 8,000 PNs belonging to high school students. We find that the structural variability of these PNs can be described in terms of six basic dimensions encompassing community and cohesive subgroup structure, as well as levels of cohesion, hierarchy, and centralization. Our method allows us to categorize these PNs into eight types and to interpret them structurally. We assess the robustness and generality of our methodology by comparing with previous results on structural typologies. To encourage its adoption, its improvement by others, and to support future research, we provide a publicly available Python class, enabling researchers to utilize our method and test the universality of our results.

<details>
    <summary>摘要</summary>
    <ul>
        本研究提出了一种方法，旨在揭示个人网络（PNs）结构变异性的基本维度，并开发一种完全基于这些结构属性的分类方法。我们解决了先前文献的局限性，并为构建 PN 的严格结构类型学奠定了基础。我们用一个包含近 8,000 个属于高中学生的 PN 的数据集来测试我们的方法。我们发现，这些 PN 的结构变异性可以用六个基本维度来描述，这些维度包括社群和凝聚子群结构，以及凝聚力、层级结构和中心化程度。我们的方法允许我们将这些 PN 分为八种类型，并从结构上解释它们。我们通过与先前关于结构类型学的研究结果进行比较，来评估我们方法的稳健性和通用性。为了鼓励采用和改进我们的方法，并支持未来的研究，我们提供了一个公开可用的 Python 类，使研究人员能够利用我们的方法并测试我们结果的普遍性。
    </ul>
</details>

**Main Methods:**

1.  **Data Collection and Preparation:** Collected and curated a large dataset of personal networks (PNs) from high school students, including friendship ties. Addressed missing data and removed outliers using established statistical methods.
2.  **Structural Metric Calculation:** Computed a comprehensive set of 41 structural metrics for each PN, covering various aspects of network structure such as connectivity, centrality, community structure, and structural holes.
3.  **Dimensionality Reduction:** Employed Exploratory Factor Analysis (EFA) with varimax rotation to reduce the dimensionality of the dataset and identify the fundamental, uncorrelated dimensions of structural variability in PNs. Used Parallel Analysis to determine the optimal number of factors to retain.
4.  **Clustering Analysis:** Applied k-means clustering on the reduced-dimensional data to categorize PNs into distinct structural types. Evaluated different numbers of clusters based on redundancy and consistency criteria, using Normalized Mutual Information (NMI) to assess the robustness of the clustering.
5.  **Typology Interpretation:** Interpreted the resulting typology by analyzing the structural characteristics of each cluster and visualizing representative networks. Compared the typology with previous findings in the literature.

<details>
    <summary>主要方法</summary>
    <ul>
        数据收集与准备：从高中学生那里收集和整理了一个大型的个人网络（PNs）数据集，包括友谊关系。使用已建立的统计方法处理缺失数据并删除异常值。
        结构度量计算：为每个 PN 计算了一组全面的 41 个结构度量，涵盖了网络结构的各个方面，如连通性、中心性、社群结构和结构洞。
        降维：采用探索性因素分析（EFA）与方差最大旋转法来降低数据集的维度，并识别 PN 中结构变异性的基本、不相关的维度。使用平行分析法来确定要保留的最佳因子数。
        聚类分析：将 k-means 聚类应用于降维后的数据，以将 PN 分为不同的结构类型。使用归一化互信息（NMI）评估了基于冗余性和一致性标准的不同聚类数量，以评估聚类的稳健性。
        类型学解释：通过分析每个集群的结构特征和可视化代表性网络来解释生成的类型学。将该类型学与文献中先前的研究结果进行了比较。
    </ul>
</details>

**Main Contributions:**

1.  **A General and Rigorous Methodology:** Developed a systematic and transparent methodology for constructing structural typologies of personal networks.
2.  **Identification of Fundamental Dimensions:** Identified six fundamental, uncorrelated dimensions that capture the key aspects of structural variability in PNs, including cohesion, hierarchy, community structure, and centralization.
3.  **A Data-Driven Structural Typology:** Classified personal networks into eight distinct structural types based on empirical data and quantitative analysis, avoiding subjective pre-classifications.
4.  **Reproducible and Extensible Tool:** Provided a publicly available Python class that implements the methodology, allowing researchers to replicate and extend the analysis to other datasets and contexts.
5.  **Integration with Previous Literature:** Demonstrated how the methodology can integrate and reconcile previous, seemingly disparate findings on personal network typologies.

<details>
    <summary>主要贡献</summary>
    <ul>
        通用且严谨的方法：开发了一种系统且透明的方法，用于构建个人网络的结构类型学。
        基本维度识别：识别了六个基本、不相关的维度，这些维度捕捉了 PN 中结构变异性的关键方面，包括凝聚力、层级结构、社群结构和中心化程度。
        数据驱动的结构类型学：基于经验数据和定量分析，将个人网络分类为八种不同的结构类型，避免了主观的预分类。
        可重现且可扩展的工具：提供了一个公开可用的 Python 类，该类实现了该方法，允许研究人员将分析复制和扩展到其他数据集和背景。
        与先前文献的整合：证明了该方法如何整合和调和先前关于个人网络类型学的研究结果，这些发现表面上看似截然不同。
    </ul>
</details>