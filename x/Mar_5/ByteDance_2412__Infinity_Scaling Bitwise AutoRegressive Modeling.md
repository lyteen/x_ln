## Abstract

**Keywords:** Autoregressive Modeling, Text-to-Image Synthesis, High-Resolution Images, Bitwise Tokenization, Scalable Generation, Transformers.

<details>
    <summary>关键词</summary>
    <ul>
        自回归建模, 文本到图像合成, 高分辨率图像, 位式令牌化, 可扩展生成, 变换器
    </ul>
</details>

**Abstract:**

This paper introduces Infinity, a novel Bitwise Visual AutoRegressive Modeling approach for generating high-resolution, photorealistic images from text prompts. Infinity addresses limitations in visual autoregressive models by employing a bitwise token prediction framework featuring an infinite-vocabulary tokenizer & classifier and a bitwise self-correction mechanism. This significantly enhances generation capacity and detail. By scaling the tokenizer's vocabulary size and transformer size simultaneously, Infinity exhibits superior scaling capabilities compared to traditional VAR methods. The approach establishes a new state-of-the-art for autoregressive text-to-image models, outperforming diffusion models like SD3-Medium and SDXL in benchmarks like GenEval and ImageReward. Infinity also achieves faster generation speeds, creating high-quality 1024x1024 images in just 0.8 seconds.

<details>
    <summary>摘要</summary>
    <ul>
        本文介绍了一种新颖的位式视觉自回归建模方法 Infinity，用于从文本提示生成高分辨率、逼真的图像。Infinity 通过采用位式令牌预测框架来解决视觉自回归模型的局限性，该框架具有无限词汇标记器和分类器以及位式自校正机制。这显著提高了生成能力和细节。通过同时缩放令牌器的词汇量和Transformer大小，Infinity 表现出比传统VAR方法更优越的缩放能力。该方法为自回归文本到图像模型建立了新的技术水平，在 GenEval 和 ImageReward 等基准测试中，优于 SD3-Medium 和 SDXL 等扩散模型。Infinity 还实现了更快的生成速度，仅需 0.8 秒即可创建高质量的 1024x1024 图像。
    </ul>
</details>

**Main Methods:**

1.  **Bitwise Visual Autoregressive Modeling:**  Replaces index-wise tokens with bitwise tokens for image representation, improving reconstruction quality and detail.
2.  **Infinite-Vocabulary Tokenizer & Classifier:**  Scales the tokenizer vocabulary to effectively infinity using a bitwise approach, overcoming limitations of conventional classifiers.
3.  **Bitwise Self-Correction:**  Introduces a self-correction mechanism during training by randomly flipping bits and re-quantizing residual features to improve robustness and correct prediction mistakes.
4.  **Coarse-to-Fine Next-Scale Prediction:**  Employs a visual autoregressive approach for image generation by refining previous scale steps.
5.  **KV-Caching:** During inference stage, performs KV-caching to speed up inference and there's no need for masking

<details>
    <summary>主要方法</summary>
        <ul>
          <li>位式视觉自回归建模：用位式令牌替换索引式令牌以进行图像表示，从而提高重建质量和细节。</li>
          <li>无限词汇标记器和分类器：使用位式方法将标记器词汇表有效地扩展到无穷大，从而克服了传统分类器的局限性。</li>
          <li>位式自校正：通过随机翻转位和重新量化残差特征，在训练期间引入自校正机制，以提高鲁棒性并纠正预测错误。</li>
          <li>由粗到精的下一尺度预测：采用视觉自回归方法进行图像生成，通过细化先前的尺度步骤。</li>
	  <li>KV-Caching：在推理阶段，执行KV-Caching以加速推理，并且不需要掩蔽。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Novel Bitwise Modeling Framework:** Introduces a novel approach for visual autoregressive modeling that improves scaling and visual detail representation capabilities.
2.  **Near-Continuous Tokenizer Performance:** Achieves near-continuous tokenizer performance by scaling tokenizers and transformers.
3.  **Superior Text-to-Image Generation:** Enables a discrete autoregressive text-to-image model to achieve exceptional prompt adherence and superior image generation quality, along with fast inference speed.
4.  **Significant Performance Gains:**  Outperforms state-of-the-art diffusion models in benchmark scores and human evaluation, while also achieving faster generation speeds.
5.  **Open-Source Release:**  Plans to release models and code to promote further exploration of the "Infinity" approach.

<details>
    <summary>主要贡献</summary>
        <ul>
          <li>新颖的位式建模框架：引入了一种用于视觉自回归建模的新颖方法，该方法提高了缩放能力和视觉细节表示能力。</li>
          <li>接近连续的标记器性能：通过缩放标记器和Transformer，实现了接近连续的标记器性能。</li>
          <li>卓越的文本到图像生成：使离散自回归文本到图像模型能够实现卓越的提示遵循和卓越的图像生成质量，以及快速的推理速度。</li>
	  <li>显著的性能提升：在基准测试分数和人类评估方面优于最先进的扩散模型，同时还实现了更快的生成速度。</li>
          <li>开源发布：计划发布模型和代码，以促进对“Infinity”方法的进一步探索。</li>
        </ul>
</details>
