## Abstract

**Keywords:** SLAM, 3D Gaussian Splatting, Differentiable Rendering, Volumetric Representation, RGB-D Camera, Scene Reconstruction, Camera Tracking

<details>
    <summary>关键词</summary>
    <ul>
        SLAM, 3D高斯溅射, 可微渲染, 体积表示, RGB-D相机, 场景重建, 相机追踪
    </ul>
</details>

**Abstract:**
Dense Simultaneous Localization and Mapping (SLAM) is critical for robotics and augmented reality applications.  This work introduces SplaTAM, an approach that leverages explicit volumetric representations, i.e., 3D Gaussians, to enable high-fidelity reconstruction from a single unposed RGB-D camera. SplaTAM employs a simple online tracking and mapping system tailored to the underlying Gaussian representation and utilizes a silhouette mask to elegantly capture the presence of scene density.  Extensive experiments show that SplaTAM achieves superior performance in camera pose estimation, map construction, and novel-view synthesis over existing methods.

<details>
    <summary>摘要</summary>
    <ul>
       稠密同步定位与建图 (SLAM) 对于机器人和增强现实应用至关重要。本文介绍了一种名为 SplaTAM 的方法，该方法利用显式体积表示，即 3D 高斯分布，来实现从单个未定位的 RGB-D 相机进行高保真重建。 SplaTAM 采用一个简单的在线跟踪和建图系统，该系统专为底层高斯表示而定制，并利用轮廓掩模来优雅地捕获场景密度的存在。大量实验表明，SplaTAM 在相机姿态估计、地图构建和新视角合成方面优于现有方法。
    </ul>
</details>

**Main Methods:**

1. **3D Gaussian Splatting:** The scene is represented by a collection of 3D Gaussians, simplifying the representation by using only view-independent color and isotropic Gaussians.
2. **Differentiable Rendering:**  High-fidelity color, depth, and silhouette images are rendered from the 3D Gaussian map using a differentiable rasterizer, enabling gradient-based optimization.
3. **Silhouette-Guided Tracking:** The camera pose is estimated by minimizing the image and depth reconstruction error of the RGB-D frame with respect to camera pose parameters, evaluated only over pixels within the visible silhouette.
4. **Gaussian Densification:** New Gaussians are added to the map based on the rendered silhouette and input depth, increasing map capacity in areas that are not adequately represented.
5. **Map Update:**  The parameters of all Gaussians in the scene are updated by minimizing the RGB and depth errors over a subset of keyframes.

<details>
    <summary>主要方法</summary>
    <ul>
        <li><strong>3D高斯溅射</strong>：场景由3D高斯分布的集合表示，通过仅使用与视角无关的颜色和各向同性高斯分布来简化表示。</li>
        <li><strong>可微渲染</strong>：使用可微光栅化器从 3D 高斯地图渲染高保真颜色、深度和轮廓图像，从而实现基于梯度的优化。</li>
        <li><strong>轮廓引导跟踪</strong>：通过最小化 RGB-D 帧的图像和深度重建误差（相对于相机姿态参数）来估计相机姿态，仅评估可见轮廓内的像素。</li>
        <li><strong>高斯密集化</strong>：根据渲染的轮廓和输入深度将新的高斯分布添加到地图，从而增加未充分表示区域的地图容量。</li>
        <li><strong>地图更新</strong>：通过最小化一组关键帧上的 RGB 和深度误差来更新场景中所有高斯分布的参数。</li>
    </ul>
</details>

**Main Contributions:**

1.  **First Dense RGB-D SLAM with 3D Gaussians:**  This is the first dense RGB-D SLAM solution to utilize 3D Gaussian Splatting for high-fidelity scene reconstruction.
2.  **Silhouette-Guided Tracking:**  The use of a silhouette mask elegantly captures scene density and enables efficient camera tracking.
3.  **Explicit Volumetric Representation:**  The explicit volumetric representation allows for fast rendering, dense optimization, easy identification of previously mapped areas, and structured map expansion.
4.  **State-of-the-Art Performance:**  The approach achieves superior performance in camera pose estimation, map construction, and novel-view synthesis compared to existing methods.
5.  **Novel Evaluation Benchmark:**  The paper introduces a novel evaluation benchmark based on ScanNet++ for evaluating novel-view synthesis and SLAM simultaneously.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li><strong>首个使用 3D 高斯分布的密集 RGB-D SLAM</strong>：这是第一个利用 3D 高斯溅射进行高保真场景重建的密集 RGB-D SLAM 解决方案。</li>
        <li><strong>轮廓引导跟踪</strong>：使用轮廓掩模优雅地捕获场景密度，并实现高效的相机跟踪。</li>
        <li><strong>显式体积表示</strong>：显式体积表示允许快速渲染、密集优化、轻松识别先前映射的区域以及结构化的地图扩展。</li>
        <li><strong>最先进的性能</strong>：与现有方法相比，该方法在相机姿态估计、地图构建和新视角合成方面实现了卓越的性能。</li>
        <li><strong>新颖的评估基准</strong>：本文介绍了一种基于 ScanNet++ 的新颖评估基准，用于同时评估新视角合成和 SLAM。</li>
    </ul>
</details>