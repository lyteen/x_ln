Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\models\mdfendendef.py`

好的，现在重新开始，我将对代码的关键部分进行解释，并生成代码片段附带中文描述，并包含代码的使用方式和简单示例。

**1. MDFEND_ENDEFModel 类:**

```python
import os
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import *
from transformers import BertModel

class MDFEND_ENDEFModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super(MDFEND_ENDEFModel, self).__init__()
        self.domain_num = 8  # 领域数量
        self.num_expert = 5  # 专家数量
        self.emb_dim = emb_dim # 嵌入维度
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False) # 预训练 BERT 模型
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}  # CNN 核大小和数量
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))  # 创建多个 CNN 专家
        self.expert = nn.ModuleList(expert) # 专家列表

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.num_expert),
                                      nn.Softmax(dim = 1)) # 门控网络

        self.attention = MaskAttention(emb_dim) # 注意力机制

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim) # 领域嵌入
        self.classifier = MLP(320, mlp_dims, dropout) # 分类器

        self.entity_convs = cnn_extractor(feature_kernel, emb_dim) # 实体 CNN
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout) # 实体 MLP
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp) # 实体网络
```

**描述:**  `MDFEND_ENDEFModel` 类定义了模型结构，包括 BERT 编码器、多个 CNN 专家、门控网络、注意力机制、领域嵌入、分类器和实体网络。

**如何使用:** 首先实例化该类，然后将输入数据传递给 `forward` 方法。

**2. forward 方法:**

```python
    def forward(self, **kwargs):
        inputs = kwargs['content'] # 输入文本
        masks = kwargs['content_masks'] # 文本 Mask
        category = kwargs['year'] # 年份类别
        init_feature = self.bert(inputs, attention_mask = masks)[0] # BERT 编码
        gate_input_feature, _ = self.attention(init_feature, masks) # 注意力加权特征
        shared_feature = 0 # 共享特征
        if self.training == True:
            idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
            domain_embedding = self.domain_embedder(idxs).squeeze(1) # 训练时使用真实领域嵌入
        else:
            batchsize = inputs.size(0)
            domain_embedding = self.domain_embedder(torch.LongTensor(range(8)).cuda()).squeeze(1).mean(dim = 0, keepdim = True).expand(batchsize, self.emb_dim) # 测试时使用所有领域嵌入的平均

        gate_input = torch.cat([domain_embedding, gate_input_feature], dim = -1) # 门控网络输入
        gate_value = self.gate(gate_input) # 门控值
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature) # 专家提取特征
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1)) # 加权平均专家特征

        bias_pred = self.classifier(shared_feature).squeeze(1) # 偏见预测

        entity = kwargs['entity'] # 实体文本
        masks = kwargs['entity_masks'] # 实体 Mask
        entity_feature = self.bert(entity, attention_mask = masks)[0] # 实体 BERT 编码
        entity_prob = self.entity_net(entity_feature).squeeze(1) # 实体预测
        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), torch.sigmoid(bias_pred) # 融合预测结果
```

**描述:** `forward` 方法接收输入文本、Mask、年份类别和实体文本，并使用 BERT 编码文本。 然后，它使用注意力机制和门控网络来融合多个 CNN 专家的特征。  最后，它融合偏见预测和实体预测，输出最终预测结果。

**如何使用:**  将包含 `content`, `content_masks`, `year`, `entity`, `entity_masks` 等键的字典传递给 `forward` 方法。

**3. Trainer 类:**

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
```

**描述:** `Trainer` 类负责模型的训练和测试。  它接收一个配置字典，其中包含训练参数。

**如何使用:**  实例化 `Trainer` 类，传入配置字典。

**4. train 方法:**

```python
    def train(self, logger = None):
        print('lr:', self.config['lr'])
        if(logger):
            logger.info('start training......')
        self.model = MDFEND_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'])
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

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred, entity_pred, _ = self.model(**batch_data)
                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_mdfendendef.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
```

**描述:** `train` 方法执行模型的训练过程。 它加载训练数据，并使用 Adam 优化器更新模型参数。 它还使用 `Recorder` 类来实现早停策略。

**如何使用:** 调用 `train` 方法开始训练。需要确保配置字典中包含训练数据路径、batchsize、学习率等参数。

**5. test 方法:**

```python
    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                _, __, batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)
```

**描述:** `test` 方法评估模型在测试数据上的性能。  它加载测试数据，并计算指标。

**如何使用:**  调用 `test` 方法进行测试。需要提供测试数据的 `dataloader`。

**示例 Config:**

```python
config = {
    'emb_dim': 128,
    'model': {'mlp': {'dims': [256, 128], 'dropout': 0.2}},
    'use_cuda': True,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'early_stop': 5,
    'root_path': './data/', # 请替换成你的数据路径
    'max_len': 128,
    'batchsize': 32,
    'aug_prob': 0.5,
    'epoch': 10,
    'save_param_dir': './checkpoints',
    'model_name': 'MDFEND_ENDEF',
}
```

**如何使用:** 将此配置字典保存为 JSON 文件，并在训练和测试期间加载它。  确保替换 `root_path` 为实际的数据目录。
