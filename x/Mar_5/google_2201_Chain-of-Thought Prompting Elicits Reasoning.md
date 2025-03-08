Okay, here is an abstract of the provided paper, following your guidelines.

## Abstract

**Keywords:** Chain-of-thought prompting, large language models, reasoning, arithmetic, commonsense, symbolic reasoning, few-shot learning, emergent abilities

<details>
    <summary>关键词</summary>
    <ul>
        链式思考提示，大型语言模型，推理，算术，常识，符号推理，少样本学习，涌现能力
    <ul>
</details>

**Main Methods:**

1.  **Chain-of-Thought Prompting:**  A few-shot prompting method where exemplars include a chain of intermediate natural language reasoning steps leading to the final answer.
2.  **In-Context Learning:** Leveraging the pre-trained knowledge within large language models without fine-tuning, relying solely on the provided exemplars.
3.  **Ablation Studies:**  Experimenting with variations like "equation only," "variable compute only," and "reasoning after answer" to understand the key aspects of chain-of-thought prompting.
4.  **Robustness Evaluations:** Testing the method with different annotators, varying exemplar orders, and varying numbers of exemplars to assess reliability.

<details>
    <summary>主要方法</summary>
    <ul>
        <li>链式思考提示：一种少样本提示方法，其中范例包括一系列引导至最终答案的中间自然语言推理步骤。</li>
        <li>上下文学习：利用大型语言模型中预训练的知识，无需微调，仅依赖于提供的范例。</li>
        <li>消融研究：实验诸如 “仅方程”、“仅可变计算” 和 “答案后推理” 等变体，以了解链式思考提示的关键方面。</li>
        <li>稳健性评估：通过不同的标注者、改变范例顺序和改变范例数量来测试该方法，以评估其可靠性。</li>
    <ul>
</details>

**Main Contributions:**

1.  **Demonstration of Chain-of-Thought Effectiveness:** Shows that chain-of-thought prompting significantly improves reasoning abilities in large language models on arithmetic, commonsense, and symbolic reasoning tasks.
2.  **Emergent Reasoning Ability:** Reveals that chain-of-thought reasoning is an emergent ability of model scale, only appearing in sufficiently large language models (~100B parameters).
3.  **State-of-the-Art Results:** Achieves state-of-the-art accuracy on the GSM8K benchmark for math word problems using chain-of-thought prompting with a 540B parameter PaLM model, surpassing even finetuned GPT-3 with a verifier.
4.  **Robustness Analysis:**  Demonstrates robustness to different annotators, exemplar variations, and language models.
5.  **Simplicity and Generality:** Provides a simple and broadly applicable method for enhancing reasoning in language models without task-specific training.

<details>
    <summary>主要贡献</summary>
    <ul>
        <li>证明链式思考的有效性：表明链式思考提示显著提高了大型语言模型在算术、常识和符号推理任务中的推理能力。</li>
        <li>涌现的推理能力：揭示了链式思考推理是模型规模的一种涌现能力，仅在足够大的语言模型（~100B 参数）中出现。</li>
        <li>最先进的结果：通过使用具有 540B 参数的 PaLM 模型进行链式思考提示，在数学文字问题 GSM8K 基准上实现了最先进的准确性，甚至超过了经过验证器微调的 GPT-3。</li>
        <li>稳健性分析：证明了对不同标注者、范例变体和语言模型的稳健性。</li>
        <li>简单性和通用性：提供了一种简单且广泛适用的方法来增强语言模型的推理能力，而无需特定于任务的训练。</li>
    <ul>
</details>

---

**Abstract (English):**

We explore how generating a *chain of thought*—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called *chain-of-thought prompting*, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

<details>
    <summary>摘要(中文)</summary>
我们探索了如何生成*思维链*——一系列中间推理步骤——从而显著提高大型语言模型执行复杂推理的能力。 特别是，我们展示了通过一种称为*思维链提示*的简单方法，这种推理能力如何在足够大的语言模型中自然地涌现，其中提供了一些思维链演示作为提示中的示例。 在三个大型语言模型上的实验表明，思维链提示提高了在一系列算术、常识和符号推理任务上的性能。 经验上的收益可能是惊人的。 例如，仅用八个思维链范例提示 PaLM 540B 即可在数学单词问题 GSM8K 基准上实现最先进的准确性，甚至超过了经过验证器微调的 GPT-3。
</details>