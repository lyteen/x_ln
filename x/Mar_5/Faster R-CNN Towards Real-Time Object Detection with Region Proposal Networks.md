## Abstract

**Keywords:** Object detection, region proposal, convolutional neural network

<details>
    <summary>关键词</summary>
    <ul>
        目标检测，区域提议，卷积神经网络
    <ul>
</details>

**Main Methods:**
This paper introduces a novel Region Proposal Network (RPN) for generating region proposals for object detection. Key aspects of the method include:

1.  **Region Proposal Network (RPN):** A fully convolutional network (FCN) that shares full-image convolutional features with the detection network. RPN simultaneously predicts object bounds and objectness scores at each location.
2.  **Anchor Boxes:** "Anchor" boxes, which serve as references at multiple scales and aspect ratios.
3.  **Alternating Training Scheme:** A training scheme is proposed that alternates between fine-tuning the RPN for the region proposal task and then fine-tuning the Fast R-CNN for object detection, while keeping the proposals fixed.
4.  **Multi-Task Loss:** Loss Function follows the multi-task loss in Fast R-CNN which combine classification and bounding box regression.

<details>
    <summary>主要方法</summary>
    <ul>
       <li> 区域提议网络（RPN）：一种完全卷积网络（FCN），与检测网络共享全图像卷积特征。RPN 同时预测每个位置的目标边界和对象性得分。
       <li> Anchor Box： “Anchor”框，用作多个尺度和宽高比的参考。
       <li> 交替训练方案：提出了一种训练方案，该方案在微调 RPN 以进行区域提议任务，然后微调 Fast R-CNN 以进行目标检测之间交替进行，同时保持提议不变。
       <li> 多任务损失：损失函数遵循 Fast R-CNN 中的多任务损失，结合分类和边界框回归。
    <ul>
</details>

**Main Contributions:**

1.  **Nearly Cost-Free Region Proposals:** Enables nearly cost-free region proposals by sharing full-image convolutional features between the RPN and the detection network.
2.  **End-to-End Training:** The RPN is trained end-to-end to generate high-quality region proposals, which are then used by Fast R-CNN for detection.
3.  **Unified Network:** Merges RPN and Fast R-CNN into a single network by sharing their convolutional features, with the RPN component guiding the network to focus on relevant regions.
4.  **State-of-the-Art Accuracy and Speed:** Achieves state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with a frame rate of 5 fps on a GPU.
5.  **Top-Ranked Performance in Competitions:** Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks of the ILSVRC and COCO 2015 competitions.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>近乎零成本的区域提议：通过在 RPN 和检测网络之间共享全图像卷积特征，实现近乎零成本的区域提议。
        <li>端到端训练：RPN 经过端到端训练以生成高质量的区域提议，然后由 Fast R-CNN 用于检测。
        <li>统一网络：通过共享卷积特征将 RPN 和 Fast R-CNN 合并到单个网络中，RPN 组件引导网络专注于相关区域。
        <li>最先进的准确性和速度：在 PASCAL VOC 2007、2012 和 MS COCO 数据集上实现了最先进的目标检测准确性，并且在 GPU 上的帧速率为 5 fps。
        <li>比赛中的顶级性能：Faster R-CNN 和 RPN 是 ILSVRC 和 COCO 2015 年比赛中多个赛道的第一名获奖作品的基础。
    <ul>
</details>
