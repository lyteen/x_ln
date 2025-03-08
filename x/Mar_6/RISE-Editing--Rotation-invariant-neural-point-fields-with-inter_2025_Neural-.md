## Abstract

**Keywords:** Neural radiance fields, 3D scene editing, Point-based representation, Interactive editing

<details>
    <summary>关键词</summary>
    <ul>
        神经辐射场, 3D场景编辑, 基于点的表示, 交互式编辑
    </ul>
</details>

**Abstract:**
Neural Radiance Fields (NeRF) have shown great potential for synthesizing novel views. This paper presents rotation-invariant neural point fields with interactive segmentation for fine-grained and efficient editing. By leveraging implicit NeRF-based and explicit point-based representations, it introduces a novel rotation-invariant neural point field representation, enabling learning of local contents using Cartesian coordinates, leading to significant improvements in scene rendering quality after fine-grained editing.  A Rotation-Invariant Neural Inverse Distance Weighting Interpolation (RNIDWI) module is designed to aggregate the neural points. The traditional NeRF representation is disentangled into a scene-agnostic rendering module and scene-specific neural point fields. A multi-view ensemble learning strategy lifts the 2D inconsistent zero-shot segmentation results to 3D neural points field in real-time without post retraining. With simple click-based prompts on 2D images, users can efficiently segment the 3D neural point field and manipulate the corresponding neural points, enabling fine-grained editing of the implicit fields. The method offers enhanced editing capabilities, a simplified editing process, delivers photorealistic rendering quality, and surpasses related methods in terms of space-time efficiency and the types of editing functions.

<details>
    <summary>摘要</summary>
    <ul>
        神经辐射场（NeRF）在合成新视角方面展现出了巨大的潜力。本文提出了一种具有交互式分割功能的旋转不变神经点场，用于细粒度和高效的编辑。通过利用隐式NeRF和显式基于点的表示，引入了一种新的旋转不变神经点场表示，能够使用笛卡尔坐标学习局部内容，从而在细粒度编辑后显著提高场景渲染质量。设计了一个旋转不变神经逆距离加权插值（RNIDWI）模块来聚合神经点。将传统的NeRF表示解耦为与场景无关的渲染模块和特定于场景的神经点场。一种多视角集成学习策略将2D不一致的零样本分割结果实时提升到3D神经点场，无需后重训练。通过在2D图像上进行简单的点击提示，用户可以高效地分割3D神经点场并操纵相应的神经点，从而实现对隐式场的细粒度编辑。该方法提供了增强的编辑能力、简化的编辑流程，实现了逼真的渲染质量，并在时空效率和编辑功能类型方面超越了相关方法。
    </ul>
</details>

**Main Methods:**

1.  **Rotation-Invariant Neural Point Field Representation:** Learns local contents using Cartesian coordinates, improving rendering quality after editing. Employs a Radiance Rotation Invariant (RRI) constraint and Rotation-invariant Neural Inverse Distance Weighting Interpolation (RNIDWI) module to achieve rotation invariance.

2.  **Disentangled NeRF Representation:** Decomposes the traditional NeRF into a scene-agnostic rendering module and scene-specific neural point fields for efficient cross-scene compositing.

3.  **Multi-view Ensemble Learning:** Lifts 2D inconsistent zero-shot segmentation results to 3D neural point field in real-time without post-retraining, enabling interactive implicit field segmentation and editing.

4. **Interactive Editing Framework:**  Enables users to segment and manipulate the implicit point field with click-based prompts on 2D images, allowing for efficient and fine-grained editing.

<details>
    <summary>主要方法</summary>
        <ul>
          <li>旋转不变神经点场表示：使用笛卡尔坐标系学习局部内容，提高编辑后的渲染质量。 采用辐射旋转不变（RRI）约束和旋转不变神经逆距离加权插值（RNIDWI）模块来实现旋转不变性。</li>
          <li>解耦的NeRF表示：将传统的NeRF分解为与场景无关的渲染模块和特定于场景的神经点场，以实现高效的跨场景合成。</li>
          <li>多视角集成学习：将2D不一致的零样本分割结果实时提升到3D神经点场，无需后重训练，从而实现交互式隐式场分割和编辑。</li>
	  <li>交互式编辑框架：使用户能够通过基于点击的提示在2D图像上分割和操作隐式点场，从而实现高效和细粒度的编辑。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Novel Rotation-Invariant Representation:** Introduces a rotation-invariant neural point field representation to enhance rendering quality after editing.

2.  **Plug-and-Play Multi-View Ensemble Learning:** Presents an efficient strategy to lift 2D segmentation results to 3D without retraining.

3.  **Efficient and Fine-Grained Editing Framework:** Offers an interactive and efficient editing framework with simple user prompts.

4.  **Enhanced Editing Capabilities:** Provides enhanced editing capabilities, simplified editing, and photorealistic rendering.

5.  **Superior Performance:** Outperforms related methods in space-time efficiency and editing function types.

<details>
    <summary>主要贡献</summary>
        <ul>
          <li>新颖的旋转不变表示：引入了一种旋转不变神经点场表示，以提高编辑后的渲染质量。</li>
          <li>即插即用的多视角集成学习：提出了一种无需重训练即可将2D分割结果提升到3D的高效策略。</li>
          <li>高效且细粒度的编辑框架：提供了一个交互式且高效的编辑框架，用户只需简单的提示即可进行操作。</li>
	  <li>增强的编辑能力：提供了增强的编辑能力、简化的编辑流程和逼真的渲染。</li>
          <li>卓越的性能：在时空效率和编辑功能类型方面优于相关方法。</li>
        </ul>
</details>
