## Abstract

**Keywords:** Open-Vocabulary Object Detection, Vision-Language Modeling, YOLO, Zero-Shot Learning, Real-Time Detection

<details>
    <summary>关键词</summary>
    <ul>
        开放词汇目标检测, 视觉-语言建模, YOLO, 零样本学习, 实时检测
    <ul>
</details>

**Abstract:**
This paper introduces YOLO-World, a novel approach to enhance the You Only Look Once (YOLO) series with open-vocabulary object detection capabilities. It overcomes the limitations of traditional YOLO detectors by enabling the detection of objects beyond predefined and trained categories. This is achieved through vision-language modeling and pre-training on large-scale datasets, utilizing a new Re-parameterizable Vision-Language Path Aggregation Network (RepVL-PAN) and region-text contrastive loss to effectively integrate visual and linguistic information. YOLO-World demonstrates high efficiency and excels in detecting a wide range of objects in a zero-shot manner. On the LVIS dataset, YOLO-World achieves 35.4 AP with 52.0 FPS on V100.

<details>
    <summary>摘要</summary>
    <ul>
        本文介绍了一种新颖的 YOLO-World 方法，旨在增强 You Only Look Once (YOLO) 系列的开放词汇目标检测能力。它通过实现对超出预定义和训练类别之外的目标的检测，克服了传统 YOLO 检测器的局限性。这是通过视觉-语言建模和在大规模数据集上进行预训练来实现的，利用一种新的可重参数化视觉-语言路径聚合网络 (RepVL-PAN) 和区域-文本对比损失来有效地整合视觉和语言信息。YOLO-World 展示了高效率，并在零样本方式下检测各种目标方面表现出色。在 LVIS 数据集上，YOLO-World 在 V100 上实现了 35.4 AP 和 52.0 FPS。
    <ul>
</details>

**Main Methods:**

1.  **Re-parameterizable Vision-Language Path Aggregation Network (RepVL-PAN):**  A novel network designed to facilitate the interaction between visual and linguistic information. During inference, the text encoder can be removed, and the text embeddings can be re-parameterized into weights of RepVL-PAN for efficient deployment.  It consists of Text-guided CSPLayers and Image-Pooling Attention to enhance the multi-modal interaction.
2.  **Region-Text Contrastive Loss:** A loss function used to pre-train the model on large-scale datasets by unifying detection, grounding, and image-text data into region-text pairs.
3.  **Prompt-then-Detect Paradigm:** An efficient inference strategy where user-defined prompts are pre-encoded into an offline vocabulary, which is then seamlessly integrated into the detector without re-encoding prompts on the fly.
4.  **Pre-training on Large-Scale Datasets:** Leveraging diverse datasets including Objects365, GQA, and CC3M-Lite to improve zero-shot performance and generalization ability.
5.  **Zero-Shot Learning:** Training the model to detect objects from categories not seen during the training phase.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><strong>可重参数化视觉-语言路径聚合网络 (RepVL-PAN)</strong>：一种新型网络，旨在促进视觉和语言信息之间的交互。在推理过程中，可以移除文本编码器，并将文本嵌入重新参数化为 RepVL-PAN 的权重，以实现高效部署。它由文本引导的 CSPLayer 和图像池化注意力组成，以增强多模态交互。</li>
        <li><strong>区域-文本对比损失</strong>：一种损失函数，用于通过将检测、定位和图像-文本数据统一到区域-文本对中，在大规模数据集上预训练模型。</li>
        <li><strong>提示-然后-检测范式</strong>：一种高效的推理策略，其中用户定义的提示被预编码到离线词汇表中，然后无缝集成到检测器中，而无需即时重新编码提示。</li>
        <li><strong>大规模数据集上的预训练</strong>：利用包括 Objects365、GQA 和 CC3M-Lite 在内的多样化数据集，以提高零样本性能和泛化能力。</li>
        <li><strong>零样本学习</strong>：训练模型以检测在训练阶段未见过的类别的目标。</li>
    <ul>
</details>

**Main Contributions:**

1.  **YOLO-World:** Introduces YOLO-World, a high-efficiency open-vocabulary object detector for real-world applications.
2.  **RepVL-PAN:** Proposes a new RepVL-PAN architecture to connect vision and language features, along with an open-vocabulary region-text contrastive pre-training scheme.
3.  **Strong Zero-Shot Performance:** Demonstrates strong zero-shot performance on the LVIS dataset (35.4 AP with 52.0 FPS) and shows adaptability to downstream tasks.
4.  **Prompt-then-Detect:** Introduce a prompt-then-detect paradigm to improve efficiency of open-vocabulary object detection in real-world scenarios.
5.  **Open-Sourced Weights and Code:** The pre-trained weights and codes of YOLO-World will be open-sourced to facilitate more practical applications.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><strong>YOLO-World</strong>：介绍了 YOLO-World，一种用于现实世界应用的高效开放词汇目标检测器。</li>
        <li><strong>RepVL-PAN</strong>：提出了一种新的 RepVL-PAN 架构来连接视觉和语言特征，以及一种开放词汇区域-文本对比预训练方案。</li>
        <li><strong>强大的零样本性能</strong>：在 LVIS 数据集上展示了强大的零样本性能（35.4 AP，52.0 FPS），并展示了对下游任务的适应性。</li>
        <li><strong>提示-然后-检测</strong>：介绍了一种提示-然后-检测范式，以提高现实场景中开放词汇目标检测的效率。</li>
        <li><strong>开源权重和代码</strong>：YOLO-World 的预训练权重和代码将开源，以促进更实际的应用。</li>
    <ul>
</details>
