Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\models\eannendef.py`

好的，现在我们重新开始。我将对提供的代码进行分段解释，并为每一段附上中文描述和简单的使用示例。

**1. 模型定义 (EANN_ENDEFModel):**

```python
import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class EANN_ENDEFModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(EANN_ENDEFModel, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.embedding = self.bert.embeddings
        domain_num = 8

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.classifier = MLP(mlp_input_shape, mlp_dims, dropout)
        self.domain_classifier = nn.Sequential(MLP(mlp_input_shape, mlp_dims, dropout, False), torch.nn.ReLU(),
                        torch.nn.Linear(mlp_dims[-1], domain_num))
        
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp)
    
    def forward(self, alpha, **kwargs):
        inputs = kwargs['content']
        bert_feature = self.embedding(inputs)
        feature = self.convs(bert_feature)
        bias_pred = self.classifier(feature).squeeze(1)
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(feature, alpha))

        entity = kwargs['entity']
        entity_feature = self.embedding(entity)
        entity_prob = self.entity_net(entity_feature).squeeze(1)

        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), domain_pred, torch.sigmoid(bias_pred)
```

**描述:**  `EANN_ENDEFModel` 类定义了一个基于 BERT 的情感分析模型，结合了领域对抗和实体信息。

*   **初始化 (`__init__`)**:
    *   加载预训练的中文 BERT 模型 (`hfl/chinese-bert-wwm-ext`)，并冻结其参数。
    *   定义卷积层 (`self.convs`) 用于提取文本特征。
    *   定义 MLP 分类器 (`self.classifier`) 用于情感预测。
    *   定义领域分类器 (`self.domain_classifier`) 用于领域对抗训练。
    *   定义实体信息提取网络 (`self.entity_net`)，用于融合实体信息。
*   **前向传播 (`forward`)**:
    *   获取文本内容 (`inputs`)，并使用 BERT 的 embedding 层提取特征。
    *   使用卷积层提取文本特征 (`feature`)。
    *   使用分类器预测情感倾向 (`bias_pred`)。
    *   使用领域分类器预测领域信息 (`domain_pred`)，并使用 `ReverseLayerF` 实现领域对抗。
    *   获取实体信息 (`entity`)，提取实体特征 (`entity_feature`)，并使用 `entity_net` 预测实体情感倾向 (`entity_prob`)。
    *   融合文本情感倾向和实体情感倾向，并返回最终的情感预测结果、实体情感预测结果、领域预测结果和文本情感预测结果。

**中文描述:**  `EANN_ENDEFModel` 类是一个情感分析模型，它使用预训练的 BERT 模型来提取文本特征，并结合了领域对抗训练和实体信息。 模型包括一个文本情感分类器，一个领域分类器和一个实体情感分类器。 前向传播函数将文本内容和实体信息作为输入，并返回情感预测，领域预测和实体情感预测。

**使用示例:**

```python
# 假设已经定义了 emb_dim, mlp_dims, dropout 等参数
# 创建模型实例
model = EANN_ENDEFModel(emb_dim=768, mlp_dims=[256, 128], dropout=0.1)

# 创建虚拟输入数据
content = torch.randint(0, 21128, (32, 128))  # (batch_size, sequence_length)  假设词汇表大小是21128
entity = torch.randint(0, 21128, (32, 16))  # (batch_size, entity_length)

# 将输入数据打包成字典
kwargs = {'content': content, 'entity': entity}

# 前向传播
alpha = 0.5  # 领域对抗的超参数
pred, entity_pred, domain_pred, bias_pred = model(alpha=alpha, **kwargs)

# 打印输出结果的形状
print("情感预测结果:", pred.shape)
print("实体情感预测结果:", entity_pred.shape)
print("领域预测结果:", domain_pred.shape)
print("文本情感预测结果:", bias_pred.shape)
```

**2. 训练器 (Trainer):**

```python
class Trainer():
    def __init__(self,
                 config
                 ):
        self.config = config
        
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        
    def train(self, logger = None):
        print('lr:', self.config['lr'])
        if(logger):
            logger.info('start training......')
        self.model = EANN_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'])
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'], self.config['batchsize'], shuffle=True, use_endef=True, aug_prob=self.config['aug_prob'])
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()
            alpha = max(2. / (1. + np.exp(-10 * epoch / self.config['epoch'])) - 1, 1e-1)

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']
                domain_label = batch_data['year']

                pred, entity_pred, domain_pred, __ = self.model(**batch_data, alpha=alpha)
                loss_adv = F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label)
                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float()) + loss_adv
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_eannendef.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_eannendef.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('future results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_eannendef.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                _, __, ___, batch_pred = self.model(**batch_data, alpha = 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)
```

**描述:** `Trainer` 类用于训练和评估 `EANN_ENDEFModel` 模型。

*   **初始化 (`__init__`)**:
    *   接收一个 `config` 字典，包含训练所需的各种参数，例如学习率、batch size、模型保存路径等。
    *   根据 `config` 中的信息设置模型保存路径。
*   **训练 (`train`)**:
    *   从 `config` 中读取学习率等参数。
    *   创建 `EANN_ENDEFModel` 实例，并根据 `config` 中的 `use_cuda` 参数决定是否使用 CUDA。
    *   定义二元交叉熵损失函数 (`loss_fn`)。
    *   定义 Adam 优化器 (`optimizer`)，并设置学习率和权重衰减。
    *   创建 `Recorder` 实例用于记录验证集上的性能，并实现早停。
    *   加载验证集数据。
    *   进行多个 epoch 的训练：
        *   加载训练集数据。
        *   使用 `tqdm` 创建训练数据迭代器。
        *   定义 `Averager` 实例用于计算平均损失。
        *   根据当前 epoch 计算领域对抗训练的超参数 `alpha`。
        *   遍历训练数据：
            *   将 batch 数据移动到 GPU (如果 `use_cuda` 为 True)。
            *   获取标签 (`label`) 和领域标签 (`domain_label`)。
            *   将 batch 数据传递给模型，并获取预测结果 (`pred`, `entity_pred`, `domain_pred`)。
            *   计算领域对抗损失 (`loss_adv`)。
            *   计算总损失 (`loss`)，包括情感分类损失、实体情感分类损失和领域对抗损失。
            *   清空优化器的梯度。
            *   反向传播计算梯度。
            *   更新模型参数。
            *   更新平均损失。
        *   打印训练信息。
        *   在验证集上进行测试，并获取测试结果 (`results`)。
        *   使用 `Recorder` 记录测试结果，并判断是否需要保存模型或提前停止训练。
    *   加载最佳模型参数。
    *   在测试集上进行测试，并获取最终的测试结果 (`future_results`)。
    *   将测试结果和模型保存路径记录到日志中 (如果提供了 `logger`)。
    *   返回测试结果和模型保存路径。
*   **测试 (`test`)**:
    *   将模型设置为评估模式 (`self.model.eval()`)。
    *   创建数据迭代器。
    *   遍历测试数据：
        *   将 batch 数据移动到 GPU (如果 `use_cuda` 为 True)。
        *   获取标签 (`batch_label`)。
        *   将 batch 数据传递给模型，并获取预测结果 (`batch_pred`)。
        *   将预测结果和标签添加到列表中。
    *   使用 `metrics` 函数计算评估指标，并返回结果。

**中文描述:** `Trainer` 类负责模型的训练和评估。 它接收一个包含所有配置信息的字典，并使用这些信息来初始化模型，优化器，损失函数等。  训练函数循环遍历训练数据，计算损失，更新模型参数，并在验证集上评估模型性能。 如果验证集性能停止提高，则使用提前停止。  测试函数在测试集上评估训练后的模型。

**使用示例:**

```python
# 假设已经定义了 config 字典，其中包含了所有必要的参数
# 创建 Trainer 实例
trainer = Trainer(config)

# 开始训练
future_results, model_path = trainer.train()

# 打印测试结果
print("最终测试结果:", future_results)
print("模型保存路径:", model_path)
```

**3. 关键辅助函数:**

*   **`data2gpu(batch, use_cuda)` (来自 `utils.utils`)**:  将一个 batch 的数据移动到 GPU 上。如果 `use_cuda` 为 True，则将数据移动到 GPU；否则，保持在 CPU 上。
*   **`get_dataloader(path, max_len, batchsize, shuffle, use_endef, aug_prob)` (来自 `utils.dataloader`)**:  根据给定的参数创建一个 DataLoader，用于加载数据。  `path` 是数据文件的路径，`max_len` 是最大序列长度，`batchsize` 是 batch size，`shuffle` 决定是否打乱数据，`use_endef` 决定是否使用实体信息，`aug_prob` 是数据增强的概率。
*   **`metrics(label, pred)` (来自 `utils.utils`)**:  根据给定的标签和预测结果计算评估指标。
*   **`Recorder(early_stop)` (来自 `utils.utils`)**: 用于记录模型性能，实现早停机制。如果模型在一定 epoch 内性能没有提升，则停止训练。
*   **`cnn_extractor(feature_kernel, emb_dim)` (来自 `.layers`)**: 创建一个 CNN 特征提取器. `feature_kernel` 定义了不同kernel size的卷积核数量. `emb_dim` 是输入的embedding维度.
*   **`MLP(input_shape, mlp_dims, dropout, bn)` (来自 `.layers`)**:  创建一个多层感知机 (MLP) 网络. `input_shape` 是输入维度, `mlp_dims` 是 MLP 的各层维度, `dropout` 是 dropout 概率, `bn` 指示是否使用 batch normalization.
*    **`ReverseLayerF` (来自 `.layers`)**:  实现梯度反转层，用于领域对抗训练。 在前向传播过程中，它不做任何操作； 在反向传播过程中，它将梯度乘以一个负数，从而实现领域对抗。

**总结:**

这段代码实现了一个复杂的情感分析模型，结合了 BERT 特征、领域对抗训练和实体信息。 `Trainer` 类提供了训练和评估模型的功能。要使用这段代码，需要准备好数据，并根据实际情况配置 `config` 字典。 此外，还需要确保 `utils` 和 `.layers` 模块中的辅助函数已经正确实现。
