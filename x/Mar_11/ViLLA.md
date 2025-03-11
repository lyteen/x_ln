## Abstract

**Keywords:** Vision-Language Models (VLMs), Fine-Grained Representation Learning, Region-Attribute Relationships, Contrastive Learning, Self-Supervised Learning, Zero-Shot Object Detection, Retrieval Tasks, DOCMNIST Dataset.

<details>
    <summary>关键词</summary>
    <ul>
        视觉-语言模型（VLMs），细粒度表示学习，区域-属性关系，对比学习，自监督学习，零样本对象检测，检索任务，DOCMNIST数据集。
    </ul>
</details>

**Abstract:**
This paper investigates the ability of Vision-Language Models (VLMs) to capture fine-grained relationships between image regions and textual attributes, particularly when trained on complex, real-world datasets.  It demonstrates that standard VLMs struggle as the pairwise complexity of training data increases, leading to performance degradation on region-attribute relationship tasks.  To address this, the paper introduces ViLLA, a novel approach that uses a self-supervised mapping model to decompose image-text samples into region-attribute pairs, followed by a contrastive VLM trained on these generated pairs.  Experiments across synthetic, product, medical, and natural image domains demonstrate that ViLLA outperforms comparable VLMs on fine-grained reasoning tasks, including zero-shot object detection and retrieval.

<details>
    <summary>摘要</summary>
    <ul>
        本文研究了视觉-语言模型（VLMs）捕获图像区域和文本属性之间细粒度关系的能力，尤其是在复杂真实世界数据集上训练时。结果表明，随着训练数据配对复杂性的增加，标准VLMs会遇到困难，导致区域-属性关系任务的性能下降。为了解决这个问题，本文介绍了一种名为ViLLA的新方法，该方法使用自监督映射模型将图像-文本样本分解为区域-属性对，然后使用对比VLM对这些生成的对进行训练。在合成图像、产品图像、医学图像和自然图像领域进行的实验表明，ViLLA在细粒度推理任务（包括零样本对象检测和检索）方面优于同类VLMs。
    </ul>
</details>

**Main Methods:**

1.  **Pairwise Complexity Analysis:** Introduces a "pairwise complexity score" to quantify the number of region-attribute pairings within an image-text sample.
2.  **DOCMNIST Dataset:** Creates a synthetic dataset, DOCMNIST, to systematically control and evaluate the impact of pairwise complexity on VLM performance.
3.  **ViLLA Framework:**
    *   **Self-Supervised Mapping Model:** Decomposes image-text samples into region-attribute pairs using a lightweight, self-supervised model trained with a contrastive loss. This model learns to associate textual attributes with candidate image regions.
    *   **Contrastive VLM Training:** Trains a standard one-to-one VLM on the generated region-attribute pairs using a bidirectional contrastive loss, enabling it to learn fine-grained relationships.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><strong>配对复杂度分析</strong>：引入“配对复杂度评分”来量化图像-文本样本中区域-属性配对的数量。</li>
        <li><strong>DOCMNIST数据集</strong>：创建一个合成数据集DOCMNIST，以系统地控制和评估配对复杂度对VLM性能的影响。</li>
        <li><strong>ViLLA框架</strong>：
        <ul>
            <li><strong>自监督映射模型</strong>：使用轻量级自监督模型和对比损失将图像-文本样本分解为区域-属性对。该模型学习将文本属性与候选图像区域相关联。</li>
            <li><strong>对比VLM训练</strong>：使用双向对比损失在生成的区域-属性对上训练标准的一对一VLM，使其能够学习细粒度关系。</li>
        </ul>
        </li>
    </ul>
</details>

**Main Contributions:**

1.  **Systematic Evaluation of Complexity:** Demonstrates the negative impact of increasing training dataset complexity on the performance of standard VLMs in capturing fine-grained region-attribute relationships.
2.  **Novel ViLLA Framework:** Introduces a novel self-supervised approach, ViLLA, capable of effectively learning fine-grained relationships from complex multimodal datasets.
3.  **State-of-the-Art Performance:** Achieves state-of-the-art performance on zero-shot object detection (COCO and LVIS) and retrieval (CheXpert) tasks.
4.  **Accurate Region-Attribute Mappings:**  ViLLA generates more accurate region-attribute mappings compared to prior approaches, contributing to improved performance in downstream tasks.
5.  **DOCMNIST Dataset:** Introduces DOCMNIST as a synthetic and customizable dataset for controlled experiments.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><strong>复杂性的系统评估</strong>：证明了增加训练数据集复杂性对标准VLMs在捕获细粒度区域-属性关系方面的性能的负面影响。</li>
        <li><strong>新型ViLLA框架</strong>：介绍了一种新型自监督方法ViLLA，能够有效地从复杂的多模态数据集学习细粒度关系。</li>
        <li><strong>最先进的性能</strong>：在零样本对象检测（COCO和LVIS）和检索（CheXpert）任务上取得了最先进的性能。</li>
        <li><strong>准确的区域-属性映射</strong>：与之前的方法相比，ViLLA生成了更准确的区域-属性映射，从而有助于提高下游任务的性能。</li>
        <li><strong>DOCMNIST数据集</strong>：引入DOCMNIST作为合成的可定制数据集，用于受控实验。</li>
    </ul>
</details>
