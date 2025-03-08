## Abstract

**Keywords:** Continual learning, Dual prompts, Knowledge distillation, Catastrophic forgetting

<details>
    <summary>关键词</summary>
    <ul>
        持续学习，双提示，知识蒸馏，灾难性遗忘
    <ul>
</details>

**Abstract:** Rehearsal-based continual learning methods often review a small number of representative samples to learn new content while retaining old knowledge. Existing methods often overlook that networks trained on limited, specific categories may have weaker generalization than those trained on large-scale, diverse datasets. They also impose strong knowledge distillation constraints that can hinder knowledge transfer. We propose a rehearsal-based continual learning method with dual prompts (DuPt) to alleviate these issues. We first introduce an input-aware prompt to utilize input priors for cue information, complementing inputs to generate rational and diverse distributions. Second, we introduce a proxy feature prompt to bridge the knowledge gap between teacher and student models, maintaining consistency in feature transfer, reinforcing feature plasticity and stability. This avoids feature conflicts due to differences between network features at new and old incremental stages. Extensive experiments validate the effectiveness of our method, which can seamlessly integrate with existing methods and lead to performance improvements.

<details>
    <summary>摘要</summary>
        基于排练的持续学习方法通常会回顾少量有代表性的样本，以便在保留旧知识的同时学习新内容。现有方法通常忽略了以下情况：在有限的特定类别上训练的网络可能比那些在大型、多样化数据集上训练的网络具有更弱的泛化能力。他们还施加了很强的知识提炼约束，这可能会阻碍知识转移。我们提出了一种基于排练的持续学习方法，该方法具有双提示（DuPt），以缓解这些问题。我们首先引入一个输入感知提示，利用输入先验来获取提示信息，补充输入以生成合理的和多样化的分布。其次，我们引入一个代理特征提示，以弥合教师模型和学生模型之间的知识差距，保持特征转移的一致性，加强特征可塑性和稳定性。这避免了由于新旧增量阶段的网络特征之间的差异而导致的特征冲突。大量的实验验证了我们方法的有效性，它可以无缝地与现有方法集成，并带来性能改进。
</details>

**Main Methods:**

1. **Input-Aware Prompt:** Uses an input-level cue that leverages an input prior to query valid cue information. This enhances the input samples, creating more rational and diverse distributions.

2. **Proxy Feature Prompt:** A feature-level cue that bridges the knowledge gap between teacher and student models, maintaining consistency in the feature transfer process and reinforcing feature plasticity and stability.

3. **Rehearsal-Based Learning:** Integrates both dual prompts with a rehearsal-based continual learning framework to retain old knowledge while learning new information.
    
4. **Knowledge Distillation:** Employs knowledge distillation, specifically modified through the proxy feature prompt, to transfer knowledge from the teacher model to the student model.

<details>
    <summary>主要方法</summary>
        <ul>
          <li>输入感知提示：使用输入级别提示，该提示利用输入先验来查询有效提示信息。这增强了输入样本，创造了更合理和多样化的分布。</li>
          <li>代理特征提示：一种特征级别提示，它弥合了教师模型和学生模型之间的知识差距，保持了特征转移过程的一致性，并加强了特征可塑性和稳定性。</li>
          <li>基于排练的学习：将双提示与基于排练的持续学习框架相结合，以在学习新信息的同时保留旧知识。</li>
	  <li>知识蒸馏：采用知识蒸馏，通过代理特征提示进行专门修改，以将知识从教师模型转移到学生模型。</li>
        </ul>
</details>

**Main Contributions:**

1.  **Introduction of Dual Prompts:** Proposes a rehearsal-based continual learning method with dual prompts (DuPt), enhancing new knowledge learning and old knowledge transfer while improving network plasticity and stability.

2.  **Input-Aware Prompt Design:** Introduces an input-aware prompt to improve the generation of more diverse and rational distributions for network inputs.

3.  **Proxy Feature Prompt Design:** Introduces a proxy feature prompt as a consistency regularization method that reduces feature inconsistencies caused by differences between network features.

4.  **Improved Performance:** Demonstrates through extensive experiments the effectiveness of the method, showcasing promising performance improvements over state-of-the-art rehearsal-based methods on challenging datasets.

5. **Strengthened Plasticity and Stability**: Enhances the plasticity and stability of the network from both input and feature perspectives.

<details>
    <summary>主要贡献</summary>
        <ul>
          <li>引入双提示：提出了一种基于排练的持续学习方法，该方法具有双提示（DuPt），增强了新知识的学习和旧知识的转移，同时提高了网络的可塑性和稳定性。</li>
          <li>输入感知提示设计：引入一种输入感知提示，以改进网络输入更多样化和合理的分布的生成。</li>
          <li>代理特征提示设计：引入代理特征提示作为一致性正则化方法，以减少由网络特征差异引起的特征不一致。</li>
	  <li>性能改进：通过大量实验证明了该方法的有效性，展示了在具有挑战性的数据集上优于最先进的基于排练的方法的有希望的性能改进。</li>
          <li>加强可塑性和稳定性：从输入和特征的角度加强网络的可塑性和稳定性。</li>
        </ul>
</details>