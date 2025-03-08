## Abstract

**Keywords:** Image Segmentation, Deep Convolutional Neural Networks, Encoder-Decoder, Pooling, Upsampling, Semantic Pixel-wise Segmentation, Road Scenes, Indoor Scenes.

<details>
    <summary>关键词</summary>
    <ul>
        图像分割, 深度卷积神经网络, 编码器-解码器, 池化, 上采样, 语义像素级分割, 道路场景, 室内场景
    <ul>
</details>

**Abstract:**

This paper introduces SegNet, a novel and practical deep fully convolutional neural network architecture designed for semantic pixel-wise image segmentation. SegNet comprises an encoder network, a corresponding decoder network, and a pixel-wise classification layer.  The encoder mirrors the VGG16 network's convolutional layers, while the decoder maps low-resolution encoder features back to full input resolution.  SegNet's innovation lies in its decoder's upsampling method, leveraging pooling indices from the encoder's max-pooling step for non-linear upsampling, eliminating the need for learning to upsample.  These sparse upsampled maps are then convolved to produce dense feature maps.  A comparative analysis against FCN, DeepLab-LargeFOV, and DeconvNet highlights the memory-accuracy trade-offs.  SegNet, optimized for scene understanding, exhibits efficiency in both memory and computational time.  Its smaller size and end-to-end trainability via SGD are advantageous.  A controlled benchmark on road and indoor scenes demonstrates SegNet's good performance with competitive inference time and efficient memory usage.  A Caffe implementation and web demo are provided.

<details>
    <summary>摘要</summary>
    <ul>
        本文介绍了一种新颖实用的深度全卷积神经网络架构 SegNet，专为语义像素级图像分割而设计。SegNet 包含一个编码器网络、一个相应的解码器网络和一个像素级分类层。编码器镜像了 VGG16 网络的卷积层，而解码器将低分辨率编码器特征映射回完整输入分辨率。SegNet 的创新之处在于其解码器的上采样方法，利用编码器的最大池化步骤中的池化索引进行非线性上采样，从而无需学习上采样。然后对这些稀疏上采样的映射进行卷积以生成密集的特征映射。与 FCN、DeepLab-LargeFOV 和 DeconvNet 相比，突出显示了内存-精度权衡。SegNet 针对场景理解进行了优化，在内存和计算时间方面都表现出效率。其较小的尺寸和通过 SGD 进行的端到端可训练性具有优势。道路和室内场景上的受控基准测试表明，SegNet 具有良好的性能，同时具有竞争性的推理时间和高效的内存使用。提供了 Caffe 实现和网络演示。
    <ul>
</details>

**Main Methods:**

*   **Encoder-Decoder Architecture:** Employs a deep, fully convolutional architecture with an encoder network topologically identical to the convolutional layers of VGG16 and a corresponding decoder network.
*   **Max-Pooling Indices for Upsampling:**  The decoder upsamples low-resolution feature maps using pooling indices computed during the max-pooling stage of the encoder, enabling non-linear upsampling without learning.
*   **Convolution with Trainable Filters:** Convolves the upsampled, sparse maps with trainable filters to produce dense feature maps.
*   **Pixel-wise Classification:** Utilizes a pixel-wise classification layer at the end of the decoder network for semantic segmentation.
*   **Stochastic Gradient Descent (SGD):** Trains the network end-to-end using stochastic gradient descent.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>编码器-解码器架构：采用深度全卷积架构，其中编码器网络在拓扑结构上与 VGG16 的卷积层相同，并具有相应的解码器网络。</li>
        <li>最大池化索引进行上采样：解码器使用在编码器的最大池化阶段计算的池化索引来对低分辨率特征图进行上采样，从而实现无需学习的非线性上采样。</li>
        <li>具有可训练滤波器的卷积：使用可训练的滤波器对上采样后的稀疏图进行卷积，以产生密集的特征图。</li>
        <li>逐像素分类：在解码器网络末端使用逐像素分类层进行语义分割。</li>
        <li>随机梯度下降 (SGD)：使用随机梯度下降端到端地训练网络。</li>
    </ul>
</details>

**Main Contributions:**

1.  **Novel and Practical Architecture (SegNet):** Presents a new deep fully convolutional neural network architecture (SegNet) for semantic pixel-wise image segmentation.
2.  **Efficient Upsampling Using Pooling Indices:** Introduces a novel upsampling method in the decoder network that leverages max-pooling indices from the encoder, eliminating the need for learning to upsample and enhancing boundary delineation.
3.  **Memory and Computationally Efficient:** Designed to be efficient in terms of both memory and computational time during inference, making it suitable for scene understanding applications.
4.  **End-to-End Trainable with SGD:** Can be trained end-to-end using stochastic gradient descent, simplifying the training process.
5.  **Controlled Benchmarking:** Performs a controlled benchmark against other architectures (FCN, DeepLab-LargeFOV, DeconvNet) on road scenes and SUN RGB-D indoor scenes, demonstrating its competitive performance.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>新颖实用的架构（SegNet）：提出了一种新的深度全卷积神经网络架构 (SegNet)，用于语义像素级图像分割。</li>
        <li>使用池化索引的高效上采样：在解码器网络中引入了一种新颖的上采样方法，该方法利用了来自编码器的最大池化索引，从而无需学习上采样并增强了边界划分。</li>
        <li>内存和计算效率：设计为在推理期间在内存和计算时间方面都具有效率，使其适用于场景理解应用。</li>
        <li>使用 SGD 进行端到端训练：可以使用随机梯度下降进行端到端训练，从而简化了训练过程。</li>
        <li>受控基准测试：对道路场景和 SUN RGB-D 室内场景上的其他架构（FCN、DeepLab-LargeFOV、DeconvNet）执行受控基准测试，证明了其具有竞争力的性能。</li>
    </ul>
</details>