## Abstract

**Keywords:** Bayesian hypothesis testing, Bayes factors, Exponential random graph models, g-priors
<details>
    <summary>关键词</summary>
    <ul>
        贝叶斯假设检验, 贝叶斯因子, 指数随机图模型, g先验
    <ul>
</details>

**Abstract:**
The exponential random graph (ERGM) model is a commonly used statistical framework for studying the determinants of tie formations from social network data. To test scientific theories under ERGMs, statistical inferential techniques are generally used based on traditional significance testing using p-values. This methodology has certain limitations, however, such as its inconsistent behavior when the null hypothesis is true, its inability to quantify evidence in favor of a null hypothesis, and its inability to test multiple hypotheses with competing equality and/or order constraints on the parameters of interest in a direct manner. To tackle these shortcomings, this paper presents Bayes factors and posterior probabilities for testing scientific expectations under a Bayesian framework. The methodology is implemented in the R package BFpack. The applicability of the methodology is illustrated using empirical collaboration networks and policy networks.

<details>
    <summary>摘要</summary>
    <ul>
        指数随机图 (ERGM) 模型是一种常用的统计框架，用于研究来自社交网络数据的联系形成的决定因素。 为了在 ERGM 下检验科学理论，通常使用基于使用 p 值的传统显着性检验的统计推断技术。 然而，这种方法具有一定的局限性，例如当原假设为真时其行为不一致，无法量化支持原假设的证据，并且无法测试多个具有竞争平等和/或顺序约束的假设 直接对感兴趣的参数进行处理。 为了解决这些缺点，本文提出了贝叶斯因子和后验概率，用于在贝叶斯框架下检验科学预期。 该方法在 R 软件包 BFpack 中实现。 该方法的适用性通过经验协作网络和策略网络来说明。
    <ul>
</details>

**Main Methods:**

1.  **Bayesian Hypothesis Testing:** Uses Bayes factors and posterior probabilities to test scientific expectations about tie formation in networks.
2.  **Exponential Random Graph Models (ERGMs):** Employs ERGMs to model the determinants of tie formation.
3.  **g-priors:** Introduces a weakly informative 'unit-information prior' based on Zellner's g-prior approach, facilitating automatic Bayesian hypothesis testing without requiring external prior information.
4.  **Gaussian Approximation:** Leverages Gaussian approximations of the posterior for fast Bayes factor computation.
5.  **BFpack Implementation:** Implements the methodology in the R package BFpack for easy usability.

<details>
    <summary>主要方法</summary>
        <ul>
          <li><strong>贝叶斯假设检验</strong>： 使用贝叶斯因子和后验概率来检验关于网络中联系形成的科学预期。</li>
          <li><strong>指数随机图模型 (ERGM)</strong>： 采用 ERGM 来模拟联系形成的决定因素。</li>
          <li><strong>g先验</strong>： 引入基于 Zellner 的 g 先验方法的弱信息“单位信息先验”，有助于自动贝叶斯假设检验，而无需外部先验信息。</li>
	      <li><strong>高斯近似</strong>： 利用后验的高斯近似来快速计算贝叶斯因子。</li>
          <li><strong>BFpack 实现</strong>： 在 R 包 BFpack 中实现该方法，以方便使用。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Novel Prior Specification:** Proposes a novel prior specification that can be used in an automatic fashion for Bayesian ERGM analysis.
2.  **Fast Bayes Factor Computation:** Introduces a fast method for computing Bayes factors between many hypotheses.
3.  **Broad Hypothesis Testing:** Enables a broad class of multiple hypothesis tests to be executed using equality and/or order constraints on ERGM coefficients.
4.  **Addresses Limitations of P-values:** Tackles the limitations of traditional p-value-based significance testing by providing a framework for quantifying evidence in favor of a null hypothesis and testing multiple constrained hypotheses.
5.  **Software Implementation:** Implements the proposed methodology in the R package BFpack, making it readily accessible to researchers.

<details>
    <summary>主要贡献</summary>
        <ul>
          <li><strong>新颖的先验规范</strong>： 提出了一种新颖的先验规范，可以以自动方式用于贝叶斯 ERGM 分析。</li>
          <li><strong>快速贝叶斯因子计算</strong>： 引入了一种快速计算多个假设之间贝叶斯因子的方法。</li>
          <li><strong>广泛的假设检验</strong>： 能够使用 ERGM 系数上的相等和/或顺序约束来执行各种多重假设检验。</li>
	      <li><strong>解决了 p 值的局限性</strong>： 通过提供一个量化支持零假设的证据和检验多个约束假设的框架，解决了传统的基于 p 值的显着性检验的局限性。</li>
          <li><strong>软件实现</strong>： 在 R 包 BFpack 中实现了所提出的方法，使其易于研究人员访问。</li>
        </ul>
</details>
