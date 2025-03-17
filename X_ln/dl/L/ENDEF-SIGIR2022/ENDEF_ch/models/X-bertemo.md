Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\models\bertemo.py`

好的，我们重新开始。 考虑到您提供的代码，我将专注于几个可以改进的地方，主要包括：

1.  **模型结构的可配置性:**  允许更灵活地配置 `BERTEmoModel` 的各个组件，例如 RNN 的层数、Attention 机制的类型等。
2.  **训练过程的优化:**  加入学习率调度器 (Learning Rate Scheduler) 和梯度裁剪 (Gradient Clipping) 等技术，以提高训练的稳定性和效果。
3.  **代码结构的模块化:**  将训练循环、验证循环等部分拆分成独立的函数，使代码更易于阅读和维护。

**1. 更加灵活的模型结构:**

```python
import torch
import torch.nn as nn
from transformers import BertModel

class BERTEmoModel(nn.Module):
    def __init__(self, config):
        super(BERTEmoModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model']).requires_grad_(False)

        # 可配置的BERT层冻结
        for name, param in self.bert.named_parameters():
            if any(layer in name for layer in config['trainable_bert_layers']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = config['emb_dim']
        self.rnn = nn.GRU(input_size=config['emb_dim'],
                          hidden_size=self.fea_size,
                          num_layers=config['rnn_layers'],
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(config['emb_dim'] * 2)
        self.mlp = MLP(config['emb_dim'] * 2 + config['num_emotions'], config['mlp_dims'], config['dropout'])

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']
        bert_feature = self.bert(inputs, attention_mask=masks)[0]
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, emotion], dim=1))
        return torch.sigmoid(output.squeeze(1))
```

**描述:**

*   这个版本的 `BERTEmoModel` 接受一个 `config` 字典作为参数，而不是硬编码的参数。
*   `config['trainable_bert_layers']` 允许您指定哪些 BERT 层可以训练。例如，可以设置为 `["encoder.layer.11", "pooler"]`。
*   `config['rnn_layers']` 允许您控制 RNN 的层数。
*   `config['num_emotions']` 是情感特征的数量。

**示例配置:**

```python
model_config = {
    'bert_model': 'hfl/chinese-bert-wwm-ext',
    'trainable_bert_layers': ["encoder.layer.11"],
    'emb_dim': 768,
    'rnn_layers': 1,
    'num_emotions': 47,
    'mlp_dims': [512, 1],
    'dropout': 0.5
}
```

---

**2. 优化训练过程:**

```python
import torch
import torch.nn as nn
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import data2gpu, Averager, metrics, Recorder

class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        self.loss_fn = nn.BCELoss()
        self.optimizer = AdamW(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=self.config['total_steps']
        )
        self.recorder = Recorder(self.config['early_stop'])
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        os.makedirs(self.save_path, exist_ok=True) # 使用 exist_ok=True

    def train_epoch(self, train_loader):
        self.model.train()
        train_data_iter = tqdm.tqdm(train_loader)
        avg_loss = Averager()

        for step_n, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, self.config['use_cuda'])
            label = batch_data['label']
            pred = self.model(**batch_data)
            loss = self.loss_fn(pred, label.float())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])  # 梯度裁剪
            self.optimizer.step()
            self.scheduler.step()  # 学习率调度
            avg_loss.add(loss.item())

        return avg_loss.item()

    def eval_epoch(self, val_loader):
        return self.test(val_loader, eval_mode=True) # 复用 test 函数

    def test(self, dataloader, eval_mode=False):
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()
        pred = []
        label = []
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                batch_pred = self.model(**batch_data)
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
        return metrics(label, pred)

    def train(self, train_loader, val_loader, logger=None):
        if logger:
            logger.info('开始训练...')

        for epoch in range(self.config['epoch']):
            train_loss = self.train_epoch(train_loader)
            print(f'训练 Epoch {epoch + 1}; 损失 {train_loss};')

            results = self.eval_epoch(val_loader)
            mark = self.recorder.add(results)

            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'parameter_bertemo.pkl'))
            elif mark == 'esc':
                print("提前停止!")
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bertemo.pkl')))
        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)

        if logger:
            logger.info("开始测试...")
            logger.info(f"测试结果: {future_results}")
        print('测试结果:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bertemo.pkl')
```

**描述:**

*   **AdamW Optimizer (AdamW 优化器):**  使用 AdamW 优化器，它可以更好地处理权重衰减 (weight decay)。
*   **Linear Warmup Scheduler (线性预热调度器):**  使用线性预热调度器，可以先逐渐增加学习率，然后再逐渐减小，有助于模型更好地收敛。
*   **Gradient Clipping (梯度裁剪):**  添加梯度裁剪，可以防止梯度爆炸。
*   **函数分离:**  将训练循环和测试/验证循环分离成 `train_epoch`、`eval_epoch` 和 `test` 函数。
*   **目录创建:**  使用 `os.makedirs(self.save_path, exist_ok=True)` 安全地创建保存模型的目录。

**示例配置:**

```python
train_config = {
    'use_cuda': True,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'total_steps': 1000,
    'clip_grad': 1.0,
    'epoch': 10,
    'early_stop': 5,
    'save_param_dir': './saved_models',
    'model_name': 'bertemo_v2'
}
```

**3.  整合到训练脚本:**

```python
# 示例训练脚本 (example_train.py)
import torch
from utils.dataloader import get_dataloader  # 假设 dataloader 在 utils 模块中
from BERTEmoModel import BERTEmoModel  # 假设 BERTEmoModel 在 BERTEmoModel.py 文件中
from Trainer import Trainer  # 假设 Trainer 在 Trainer.py 文件中

# 加载配置 (从 config.py 或其他地方)
from config import model_config, train_config, data_config # 假设您有单独的配置文件

# 合并配置 (如果需要)
config = {**model_config, **train_config, **data_config}

# 创建模型
model = BERTEmoModel(config)

# 创建 Trainer
trainer = Trainer(config, model)

# 创建数据加载器
train_loader = get_dataloader(config['root_path'] + 'train.json', config['max_len'], config['batchsize'], shuffle=True, use_endef=False, aug_prob=config['aug_prob'])
val_loader = get_dataloader(config['root_path'] + 'val.json', config['max_len'], config['batchsize'], shuffle=False, use_endef=False, aug_prob=config['aug_prob'])

# 训练模型
results, model_path = trainer.train(train_loader, val_loader)

print("训练完成!")
print("结果:", results)
print("模型保存路径:", model_path)
```

**描述:**

*   此脚本演示了如何加载配置、创建模型和训练器，以及如何启动训练循环。
*   它假定你已经定义了 `get_dataloader` 函数以及 `BERTEmoModel` 和 `Trainer` 类。
*   你需要根据你的实际文件结构和配置进行调整。

**关键点:**

*   **配置文件:**  强烈建议将所有配置放在单独的文件 (例如 `config.py`) 中，以方便管理。
*   **模块化:**  将代码分成独立的模块 (例如 `BERTEmoModel.py`、`Trainer.py`、`dataloader.py`、`utils.py`)，以提高可读性和可维护性。

**总结:**

通过这些改进，您将拥有一个更灵活、更强大且更易于维护的情感分类模型训练框架。 请务必根据您的特定需求调整配置和代码。
