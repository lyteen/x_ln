## Abstract

**Keywords:** Deep reinforcement learning, sample efficiency, episodic memory, actor-critic algorithm

<details>
    <summary>关键词</summary>
    <ul>
        深度强化学习，样本效率，情景记忆，Actor-Critic算法
    <ul>
</details>

**Abstract:** Existing deep reinforcement learning (DRL) algorithms suffer from the problem of low sample efficiency. Episodic memory allows DRL algorithms to remember and use past experiences with high return, thereby improving sample efficiency. However, due to the high dimensionality of the state-action space in continuous action tasks, previous methods in continuous action tasks often only utilize the information stored in episodic memory, rather than directly employing episodic memory for action selection as done in discrete action tasks. We suppose that episodic memory retains the potential to guide action selection in continuous control tasks. Our objective is to enhance sample efficiency by leveraging episodic memory for action selection in such tasks either reducing the number of training steps required to achieve comparable performance or enabling the agent to obtain higher rewards within the same number of training steps. To this end, we propose an "Episodic Memory-Double Actor-Critic (EMDAC)" framework, which can use episodic memory for action selection in continuous action tasks. The critics and episodic memory evaluate the value of state-action pairs selected by the two actors to determine the final action. Meanwhile, we design an episodic memory based on a Kalman filter optimizer, which updates using the episodic rewards of collected state-action pairs. The Kalman filter optimizer assigns different weights to experiences collected at different time periods during the memory update process. In our episodic memory, state-action pair clusters are used as indices, recording both the occurrence frequency of these clusters and the value estimates for the corresponding state-action pairs. This enables the estimation of the value of state-action pair clusters by querying the episodic memory. After that, we design intrinsic reward based on the novelty of state-action pairs with episodic memory, defined by the occurrence frequency of state-action pair clusters, to enhance the exploration capability of the agent. Ultimately, we propose an "EMDAC-TD3" algorithm by applying this three modules to Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm within an Actor-Critic framework. Through evaluations in MuJoCo environments within the OpenAI Gym domain, EMDAC-TD3 achieves higher sample efficiency compared to baseline algorithms. EMDAC-TD3 demonstrates superior final performance compared to state-of-the-art episodic control algorithms and advanced Actor-Critic algorithms, by comparing the final rewards, Median, Interquartile Mean, Mean, and Optimality Gap. The final rewards can directly demonstrate the advantages of the algorithms. Based on the final rewards, EMDAC-TD3 achieves an average performance improvement of 11.01% over TD3, surpassing the current state-of-the-art algorithms in the same category.

<details>
    <summary>摘要</summary>
    <ul>
        现有的深度强化学习 (DRL) 算法存在样本效率低的问题。情景记忆允许 DRL 算法记住并使用过去的高回报经验，从而提高样本效率。然而，由于连续动作任务中状态-动作空间的高维度，以前的方法在连续动作任务中通常只利用存储在情景记忆中的信息，而不是像在离散动作任务中那样直接使用情景记忆进行动作选择。我们假设情景记忆保留了在连续控制任务中指导动作选择的潜力。我们的目标是通过利用情景记忆进行此类任务中的动作选择来提高样本效率，要么减少实现可比性能所需的训练步骤的数量，要么使智能体在相同数量的训练步骤内获得更高的奖励。为此，我们提出了一个“情景记忆-双重 Actor-Critic (EMDAC)”框架，该框架可以使用情景记忆来进行连续动作任务中的动作选择。评论员和情景记忆评估由两个actor选择的状态-行动对的值，以确定最终行动。同时，我们设计了一种基于卡尔曼滤波器优化器的情景记忆，该优化器使用收集的状态-行动对的情景奖励进行更新。卡尔曼滤波器优化器在记忆更新过程中为不同时间段收集的经验分配不同的权重。在我们的情景记忆中，状态-行动对聚类被用作索引，记录这些聚类的出现频率和相应状态-行动对的值估计。这使得可以通过查询情景记忆来估计状态-行动对聚类的值。之后，我们设计了基于情景记忆的状态-行动对的新颖性的内在奖励，该奖励由状态-行动对聚类的出现频率定义，以提高智能体的探索能力。最终，我们通过在 Actor-Critic 框架内将这三个模块应用于 Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法，从而提出了“EMDAC-TD3”算法。通过在 OpenAI Gym 域中的 MuJoCo 环境中进行评估，与基线算法相比，EMDAC-TD3 实现了更高的样本效率。通过比较最终奖励、中位数、四分位间距均值、平均值和最优性差距，EMDAC-TD3 显示出比最先进的情景控制算法和高级 Actor-Critic 算法更优越的最终性能。最终奖励可以直接证明算法的优势。基于最终奖励，EMDAC-TD3 比 TD3 平均性能提高了 11.01%，超过了同一类别中当前最先进的算法。
    <ul>
</details>

**Main Methods:**

1.  **Episodic Memory-Double Actor-Critic (EMDAC) Framework:** Introduces a novel framework for continuous action tasks where two alternative actions are selected by actors, and the final action is chosen based on a hybrid estimation combining episodic memory and double critics. This framework addresses the challenge of using episodic memory for action selection in continuous action spaces.

2.  **Kalman Filter Optimizer-Based Episodic Memory:** Designs an episodic memory that employs a Kalman filter optimizer for updating, assigning different weights to experiences collected at different time periods. This improves the accuracy of state-action value estimation. State-action pair clusters are used as indices, recording occurrence frequency and value estimates.

3.  **Intrinsic Reward Based on Episodic Memory:**  Introduces an intrinsic reward based on the novelty of state-action pairs, defined by the occurrence frequency of state-action pair clusters, to enhance the agent's exploration capability.

4.  **EMDAC-TD3 Algorithm:**  Integrates the EMDAC framework, Kalman filter-based episodic memory, and intrinsic reward into the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

<details>
    <summary>主要方法</summary>
    <ol>
        <li><strong>情景记忆-双重Actor-Critic (EMDAC) 框架：</strong> 为连续动作任务引入了一种新颖的框架，其中两个替代动作由行动者选择，并且最终动作基于结合情景记忆和双重评论家的混合估计来选择。该框架解决了在连续动作空间中使用情景记忆进行动作选择的挑战。</li>
        <li><strong>基于卡尔曼滤波器优化器的情景记忆：</strong> 设计了一种情景记忆，该情景记忆采用卡尔曼滤波器优化器进行更新，为不同时间段收集的经验分配不同的权重。这提高了状态-动作价值估计的准确性。状态-行动对聚类被用作索引，记录发生频率和价值估计。</li>
        <li><strong>基于情景记忆的内在奖励：</strong> 引入了基于状态-动作对新颖性的内在奖励，该新颖性由状态-动作对聚类的发生频率定义，以增强智能体的探索能力。</li>
        <li><strong>EMDAC-TD3 算法：</strong> 将 EMDAC 框架、基于卡尔曼滤波器的情景记忆和内在奖励集成到 Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法中。</li>
    </ol>
</details>

**Main Contributions:**

1.  **EMDAC Framework:** Proposes a novel EMDAC framework to enable episodic memory for action selection in continuous action tasks, improving sample efficiency.

2.  **Kalman Filter-Based Episodic Memory:** Introduces an episodic memory using a Kalman filter optimizer to accurately estimate the value of state-action pairs and assign different weights to collected experiences during memory update.

3.  **Intrinsic Reward for Enhanced Exploration:** Designs an intrinsic reward based on episodic memory to encourage exploration of novel state-action pairs, enhancing the agent's exploration capability.

4.  **High Sample Efficiency:** Demonstrates through evaluations that EMDAC-TD3 achieves higher sample efficiency compared to baseline algorithms and superior final performance compared to state-of-the-art algorithms.

5.  **Significant Performance Improvement:** Achieves an average performance improvement of 11.01% over TD3, surpassing current state-of-the-art algorithms in the same category.

<details>
    <summary>主要贡献</summary>
    <ol>
        <li><strong>EMDAC 框架：</strong> 提出了一种新颖的 EMDAC 框架，以支持在连续动作任务中使用情景记忆进行动作选择，从而提高样本效率。</li>
        <li><strong>基于卡尔曼滤波器的情景记忆：</strong> 引入了一种使用卡尔曼滤波器优化器的情景记忆，以准确估计状态-动作对的值，并在记忆更新期间为收集的经验分配不同的权重。</li>
        <li><strong>增强探索的内在奖励：</strong> 设计了一种基于情景记忆的内在奖励，以鼓励探索新的状态-动作对，从而增强了智能体的探索能力。</li>
        <li><strong>高样本效率：</strong> 通过评估证明，与基线算法相比，EMDAC-TD3 实现了更高的样本效率，并且与最先进的算法相比，最终性能更优越。</li>
        <li><strong>显著的性能提升：</strong> 与 TD3 相比，实现了平均 11.01% 的性能提升，超过了同类别中当前最先进的算法。</li>
    </ol>
</details>
