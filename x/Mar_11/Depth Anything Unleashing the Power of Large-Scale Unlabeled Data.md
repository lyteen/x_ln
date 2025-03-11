
## Abstract

**Keywords:** Monocular Depth Estimation, Unlabeled Data, Data Augmentation, Semantic Priors, Zero-Shot Learning, Foundation Model

<details>
    <summary>关键词</summary>
    <ul>
        单目深度估计，无标签数据，数据增强，语义先验，零样本学习，基础模型
    <ul>
</details>

**Abstract:** This work introduces Depth Anything, a practical solution for robust monocular depth estimation. The key idea is to leverage large-scale unlabeled data (~62M images) to build a powerful foundation model, significantly improving generalization ability.  Two simple yet effective strategies are employed for data scaling-up:  creating a more challenging optimization target with data augmentation and enforcing the model to inherit semantic priors from pre-trained encoders via auxiliary supervision.  Extensive zero-shot evaluation demonstrates the model's generalization capability on unseen scenes.  Fine-tuning with metric depth data from NYUv2 and KITTI sets new state-of-the-art results. The resulting model also enhances depth-conditioned ControlNet.

<details>
    <summary>摘要</summary>
    <ul>
        本文介绍了一种名为 Depth Anything 的实用方法，用于实现稳健的单目深度估计。核心思想是利用大规模无标签数据（约 6200 万张图像）构建一个强大的基础模型，从而显著提高泛化能力。采用了两种简单而有效的策略来进行数据扩展：通过数据增强创建更具挑战性的优化目标，以及通过辅助监督强制模型从预训练编码器继承语义先验知识。广泛的零样本评估证明了该模型在未见场景中的泛化能力。通过使用来自 NYUv2 和 KITTI 的度量深度数据进行微调，实现了新的最先进水平。生成的模型还增强了深度条件 ControlNet。
    <ul>
</details>

**Main Methods:**

1.  **Large-Scale Unlabeled Data:**  A data engine is designed to collect and automatically annotate ~62M unlabeled images from diverse public datasets.
2.  **Challenging Optimization Target:** Data augmentation tools (color jittering, Gaussian blurring, and CutMix) are leveraged to create a more challenging optimization target, compelling the model to seek extra visual knowledge and robust representations.
3.  **Semantic Prior Inheritance:** An auxiliary supervision is developed to enforce the model to inherit rich semantic priors from pre-trained DINOv2 encoders using a feature alignment loss. Specifically, the model is encouraged to produce features that are similar to the frozen DINOv2 features but with a tolerance margin to allow for part-level variations within objects.
4.  **Self-Training:** Uses the teacher model to generate pseudo-labels for the unlabeled data, and then jointly trains a student model on both labeled and pseudo-labeled data.

<details>
    <summary>主要方法</summary>
     <ul>
        <li><strong>大规模无标签数据：</strong>设计了一个数据引擎，用于从各种公共数据集中收集并自动标注约 6200 万张无标签图像。</li>
        <li><strong>具有挑战性的优化目标：</strong>利用数据增强工具（色彩抖动、高斯模糊和 CutMix）来创建更具挑战性的优化目标，促使模型寻求额外的视觉知识和鲁棒的表示。</li>
        <li><strong>语义先验继承：</strong>开发了一种辅助监督，通过特征对齐损失，强制模型从预训练的 DINOv2 编码器继承丰富的语义先验。具体来说，鼓励模型生成与冻结的 DINOv2 特征相似的特征，但具有一定的容差范围，以允许对象内部的分段变化。</li>
        <li><strong>自训练：</strong>使用教师模型为无标签数据生成伪标签，然后共同训练一个学生模型，使其同时适应有标签数据和伪标签数据。</li>
     <ul>
</details>

**Main Contributions:**

1.  **Highlights the Value of Large-Scale Unlabeled Data:**  Demonstrates the significant benefit of scaling up MDE training with massive, cheap, and diverse unlabeled images.
2.  **Key Practice for Joint Training:**  Identifies and implements a key practice in jointly training on labeled and unlabeled data, challenging the model with a harder optimization target.
3.  **Semantic Prior Integration:** Proposes inheriting rich semantic priors from pre-trained encoders for better scene understanding, rather than relying on auxiliary semantic segmentation tasks, resulting in a more efficient and effective approach.
4.  **Superior Zero-Shot Performance:** Achieves state-of-the-art zero-shot MDE performance, outperforming existing methods like MiDaS-BEITL. Also demonstrates strong results after fine-tuning on metric depth datasets, even surpassing ZoeDepth.
5.  **Enhanced Depth-Conditioned ControlNet:** Demonstrates that the resulting depth estimation model leads to better depth-conditioned ControlNet results, improving image synthesis and video editing applications.

<details>
    <summary>主要贡献</summary>
       <ul>
        <li><strong>强调大规模无标签数据的价值：</strong>证明了使用大量、廉价且多样化的无标签图像来扩展 MDE 训练的显著优势。</li>
        <li><strong>联合训练的关键实践：</strong>识别并实施了在有标签和无标签数据上进行联合训练的关键实践，通过更困难的优化目标来挑战模型。</li>
        <li><strong>语义先验集成：</strong>提出通过从预训练的编码器继承丰富的语义先验，来更好地理解场景，而不是依赖于辅助的语义分割任务，从而实现了更高效和有效的方法。</li>
        <li><strong>卓越的零样本性能：</strong>实现了最先进的零样本 MDE 性能，优于现有的 MiDaS-BEITL 等方法。同时，在度量深度数据集上进行微调后也表现出强大的结果，甚至超过了 ZoeDepth。</li>
        <li><strong>增强的深度条件 ControlNet：</strong>表明由此产生的深度估计模型可以带来更好的深度条件 ControlNet 结果，从而改善图像合成和视频编辑应用。</li>
    <ul>
</details>