## Abstract

**Keywords:** Non-symmetric kernels, cross-domain learning, neural networks, kernel methods, approximation theory, ReLU networks, Sobolev spaces
<details>
    <summary>关键词</summary>
    <ul>
        非对称核, 跨域学习, 神经网络, 核方法, 逼近论, ReLU 网络, 索博列夫空间
    </ul>
</details>

**Abstract:**
This paper explores the approximation capabilities of kernel-based networks using non-symmetric kernels, motivated by applications like invariant learning, transfer learning, and synthetic aperture radar imaging. It introduces a general approach for studying these networks, including generalized translation networks and rotated zonal function kernels, without requiring positive definiteness of the kernels. The study derives estimates on the accuracy of uniform approximation of functions in a Sobolev class by ReLU networks, even when the smoothness parameter is not necessarily an integer.  The findings are applicable to the approximation of functions with limited smoothness relative to the dimension of the input space.

<details>
    <summary>摘要</summary>
    <ul>
        本文探讨了使用非对称核的核函数网络的逼近能力，其动机来自于不变学习、迁移学习和合成孔径雷达成像等应用。本文引入了一种通用方法来研究这些网络，包括广义平移网络和旋转区域函数核，而不需要核的正定性。该研究推导出了 ReLU 网络对索博列夫空间中的函数进行一致逼近的精度估计，即使在光滑度参数不一定是整数的情况下。这些发现适用于输入空间的维度相比，平滑度有限的函数的逼近。
    </ul>
</details>

**Main Methods:**

1.  **General Framework for Non-Symmetric Kernels:** Develops a general approach to analyze kernel-based networks with non-symmetric kernels.
2.  **Generalized Translation Networks:** Utilizes and extends the concept of generalized translation networks, which encompass neural networks and translation-invariant kernels.
3.  **Rotated Zonal Function Kernels:** Employs rotated zonal function kernels to handle invariance and related properties.
4.  **Approximation Theory & Probability Theory Combination:** Combines approximation-theoretic techniques with probabilistic tools to derive error estimates.
5.  **Recipe Theorem:** Establishes a "recipe theorem" that provides a unified means of obtaining approximation results for various networks.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>非对称核的通用框架：开发了一种分析具有非对称核的核函数网络的通用方法。</li>
        <li>广义平移网络：使用和扩展了广义平移网络的概念，该概念包括神经网络和平移不变核。</li>
        <li>旋转区域函数核：采用旋转区域函数核来处理不变性和相关属性。</li>
        <li>逼近论与概率论的结合：将逼近论技术与概率工具相结合，以推导出误差估计。</li>
        <li>配方定理：建立一个“配方定理”，为获得各种网络的逼近结果提供了一种统一手段。</li>
    </ul>
</details>

**Main Contributions:**

1.  **Analysis of Non-Symmetric Kernel Networks:** Offers a general framework for analyzing kernel-based networks with non-symmetric kernels, a relatively unexplored area.
2.  **Sobolev Class Approximation by ReLU Networks:** Provides estimates on the accuracy of uniform approximation of functions in Sobolev classes by ReLU networks, even with non-integer smoothness parameters.
3.  **Application to "Rough" Functions:**  Addresses the approximation of "rough" functions, where the smoothness is small compared to the input dimension.
4.  **Generalization of Bach (2017) Results:** Extends results from Bach (2017) concerning ReLU networks to the more general setting of the paper.
5.  **Novel Results for Zonal Function Networks:** Delivers new results for zonal function networks with specific activation functions, addressing cases where kernels may not be positive definite and Sobolev spaces are not well-characterized.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>非对称核网络分析：为分析具有非对称核的核函数网络提供了一个通用框架，这是一个相对未经探索的领域。</li>
        <li>ReLU 网络对索博列夫类的逼近：提供了 ReLU 网络对索博列夫类中函数进行一致逼近的精度估计，即使在光滑度参数为非整数的情况下。</li>
        <li>应用于“粗糙”函数：解决了“粗糙”函数的逼近问题，其中光滑度与输入维度相比很小。</li>
        <li>Bach (2017) 结果的推广：将关于 ReLU 网络的 Bach (2017) 的结果扩展到本文的更一般设置中。</li>
        <li>区域函数网络的新结果：为具有特定激活函数的区域函数网络提供新结果，解决了核可能不是正定且索博列夫空间没有明确特征的情况。</li>
    </ul>
</details>
