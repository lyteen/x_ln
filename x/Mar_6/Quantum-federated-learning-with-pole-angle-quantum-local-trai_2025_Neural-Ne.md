## Abstract

**Keywords:** Federated learning, Quantum neural networks, Quantum machine learning, Quantum federated learning, Slimmable neural networks

<details>
    <summary>关键词</summary>
    <ul>
        联邦学习，量子神经网络，量子机器学习，量子联邦学习，可伸缩神经网络
    <ul>
</details>

**Abstract:**
Quantum federated learning (QFL) has garnered significant interest as a novel paradigm, leveraging quantum neural networks (QNNs) for their quantum supremacy. To enhance classical QFL frameworks' flexibility and reliability, this paper introduces a new slimmable QFL (SlimQFL) approach using QNN-based slimmable neural network (QSNN) architectures. This design accounts for time-varying wireless channels and constrained computing resources, boosting efficiency by reducing parameters without sacrificing performance.  Moreover, the proposed QNN is novel due to its implementation of trainable measurement within QFL. QSNN is based on the characteristics of separated training and the joint angle and pole parameters' dynamic exploitation.  Evaluation results show that SlimQFL with both parameters achieves higher classification accuracy than standard QFL and provides transmission stability, especially in low-quality channels.

<details>
    <summary>摘要</summary>
    <ul>
        量子联邦学习 (QFL) 作为一种利用量子神经网络 (QNN) 实现量子霸权的新型范式，受到了广泛关注。为了提高传统 QFL 框架的灵活性和可靠性，本文提出了一种新的可伸缩 QFL (SlimQFL) 方法，该方法结合了基于 QNN 的可伸缩神经网络 (QSNN) 架构。这种创新设计考虑了时变无线信道和计算资源约束，通过减少参数数量来提高效率，同时不牺牲性能。此外，由于在 QFL 中实现了可训练的测量，所提出的 QNN 是新颖的。我们的 QSNN 的基本概念是基于分离训练的关键特征以及联合角度和极点参数的动态利用。性能评估结果表明，使用这两个参数，我们提出的基于 QSNN 的 SlimQFL 比标准 QFL 实现了更高的分类精度，并确保了传输稳定性，尤其是在信道质量较差的情况下。
    <ul>
</details>

**Main Methods:**

1.  **Slimmable Quantum Federated Learning (SlimQFL):** A novel QFL framework incorporating QNN-grounded slimmable neural network (QSNN) architectures. This enables the system to adapt to time-varying wireless communication channels and computing resource constraints.
2.  **Quantum Slimmable Neural Network (QSNN):** A QNN architecture with trainable measurement capabilities, allowing for independent training of both angle and pole parameters.
3.  **Pole-Angle Training:** A training strategy where the pole parameters of QSNN are trained first, followed by the training of angle parameters.
4.  **Dynamic Global Model Aggregation:** In each communication round, devices upload either only the pole parameters or both pole and angle parameters, depending on the current communication channel conditions. The server then aggregates the received parameters to construct a global QSNN.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>可伸缩量子联邦学习 (SlimQFL)：一种结合了基于 QNN 的可伸缩神经网络 (QSNN) 架构的新型 QFL 框架。这使得系统能够适应时变的无线通信信道和计算资源约束。</li>
        <li>量子可伸缩神经网络 (QSNN)：一种具有可训练测量功能的 QNN 架构，允许独立训练角度和极点参数。</li>
        <li>极角训练：一种训练策略，首先训练 QSNN 的极点参数，然后训练角度参数。</li>
        <li>动态全局模型聚合：在每个通信回合中，设备根据当前的通信信道条件，仅上传极点参数或极点和角度参数。然后，服务器聚合接收到的参数以构建全局 QSNN。</li>
    <ul>
</details>

**Main Contributions:**

1.  **SlimQFL Framework:** Proposes a new slimmable QFL framework, named SlimQFL, to adapt QFL to dynamic environments.
2.  **Trainable Measurement in QNN:** Integrates trainability into the measurement component of the QNN architecture, creating QSNN.
3.  **Pole-Angle Training Strategy:** Introduces a novel training strategy where pole and angle parameters of QSNN are trained separately and sequentially.
4.  **Dynamic Parameter Transmission:** Implements dynamic parameter transmission based on channel conditions, ensuring at least one parameter is transmitted to maintain communication stability.
5.  **Performance Evaluation:** Provides experimental results demonstrating that the proposed SlimQFL achieves higher classification accuracy and ensures transmission stability in poor channel conditions compared to standard QFL.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>SlimQFL 框架：提出了一种新的可伸缩 QFL 框架，名为 SlimQFL，以使 QFL 适应动态环境。</li>
        <li>QNN 中的可训练测量：将可训练性集成到 QNN 架构的测量组件中，创建 QSNN。</li>
        <li>极角训练策略：引入了一种新的训练策略，其中 QSNN 的极点和角度参数分别按顺序训练。</li>
        <li>动态参数传输：基于信道条件实现动态参数传输，确保至少传输一个参数以维持通信稳定性。</li>
        <li>性能评估：提供了实验结果，表明与标准 QFL 相比，所提出的 SlimQFL 实现了更高的分类精度，并确保了在较差信道条件下的传输稳定性。</li>
    <ul>
</details>