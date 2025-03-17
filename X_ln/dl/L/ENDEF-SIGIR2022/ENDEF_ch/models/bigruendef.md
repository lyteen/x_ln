Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\models\bigruendef.py`

Okay, I will provide a detailed explanation and code snippets for the provided code, with explanations in Chinese and example usages. I'll also try to keep the code snippets reasonably sized and well-commented.

**1. `BiGRU_ENDEFModel` 类 (Class):**

```python
import os
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import *
from transformers import BertModel

# 导入自定义的层和工具函数 (Import custom layers and utility functions)
from .layers import *
from utils.utils import data2gpu
#from utils.dataloader import get_dataloader  # 注释掉，后面会用到 (Commented out, will be used later)


class BiGRU_ENDEFModel(nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, num_layers):
        super(BiGRU_ENDEFModel, self).__init__()
        self.fea_size = emb_dim
        # 使用预训练的中文BERT模型 (Using pre-trained Chinese BERT model)
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.embedding = self.bert.embeddings # 使用bert的embedding层

        # 双向GRU (Bidirectional GRU)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)

        # 注意力机制 (Attention mechanism)
        input_shape = self.fea_size * 2  # 双向GRU输出维度翻倍 (Output dimension doubles for bidirectional GRU)
        self.attention = MaskAttention(input_shape)

        # 多层感知机 (Multi-layer Perceptron)
        self.mlp = MLP(input_shape, mlp_dims, dropout)

        # 实体特征提取 (Entity feature extraction)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = nn.Sequential(self.entity_convs, self.entity_mlp)

    def forward(self, **kwargs):
        # 从kwargs中获取输入数据 (Get input data from kwargs)
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        entity = kwargs['entity']

        # 使用BERT的embedding层 (Use BERT embedding layer)
        emb_feature = self.embedding(inputs)

        # GRU处理 (GRU processing)
        feature, _ = self.rnn(emb_feature)

        # 注意力机制 (Attention mechanism)
        feature, _ = self.attention(feature, masks)

        # MLP预测偏见 (MLP predicts bias)
        bias_pred = self.mlp(feature).squeeze(1)

        # 实体特征提取和预测 (Entity feature extraction and prediction)
        entity_feature = self.embedding(entity)
        entity_prob = self.entity_net(entity_feature).squeeze(1)

        # 最终预测 (Final prediction): 融合偏见预测和实体预测
        final_pred = torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob)
        entity_pred = torch.sigmoid(entity_prob)
        bias_pred = torch.sigmoid(bias_pred)

        return final_pred, entity_pred, bias_pred
```

**描述:**

*   `BiGRU_ENDEFModel` 是一个用于二元分类的模型，它结合了 BiGRU、注意力机制、MLP 和实体信息。
*   **初始化 (`__init__`)**: 加载预训练的中文 BERT 模型，定义 BiGRU 层、注意力层、MLP 层和实体特征提取层。
*   **前向传播 (`forward`)**: 接收文本内容和实体信息，通过 BERT embedding 层获取嵌入，然后依次通过 BiGRU、注意力机制和 MLP 进行处理，得到偏见预测和实体预测，最后将两者融合得到最终预测结果。`requires_grad_(False)`冻结了BERT的参数，防止训练时更改BERT参数。`nn.Sequential` 帮助组合多个层。

**如何使用:**

1.  实例化 `BiGRU_ENDEFModel` 类。
2.  将文本数据和实体数据以及对应的 masks 传入 `forward` 方法。
3.  获得最终的预测结果、实体预测结果和 bias 预测结果。

**Demo 使用示例:**

```python
# 假设已定义了 emb_dim, mlp_dims, dropout, num_layers
emb_dim = 768 # BERT 输出维度
mlp_dims = [256, 1]
dropout = 0.1
num_layers = 1

# 创建模型实例 (Create model instance)
model = BiGRU_ENDEFModel(emb_dim, mlp_dims, dropout, num_layers)

# 假设已准备好 content, content_masks, entity 数据 (Assume content, content_masks, entity data are ready)
# 这些应该是 PyTorch tensors
content = torch.randint(0, 1000, (32, 128))  # 32个样本，每个样本长度为128 (32 samples, each with length 128)
content_masks = torch.ones((32, 128), dtype=torch.bool)  # 32个样本，每个样本长度为128 (32 samples, each with length 128)
entity = torch.randint(0, 1000, (32, 20))    # 32个实体，每个实体长度为20  (32 entities, each with length 20)

# 构建输入字典 (Construct input dictionary)
input_data = {
    'content': content,
    'content_masks': content_masks,
    'entity': entity
}

# 进行前向传播 (Perform forward propagation)
final_pred, entity_pred, bias_pred = model(**input_data)

# 打印预测结果形状 (Print prediction result shape)
print("Final Prediction Shape:", final_pred.shape)  # Expected: torch.Size([32])
print("Entity Prediction Shape:", entity_pred.shape)  # Expected: torch.Size([32])
print("Bias Prediction Shape:", bias_pred.shape)  # Expected: torch.Size([32])
```

**2. `Trainer` 类 (Class):**

```python
import os
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import *
from transformers import BertModel

# 导入自定义的层和工具函数 (Import custom layers and utility functions)
from .layers import *
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader

class Trainer():
    def __init__(self, config):
        self.config = config

        # 设置保存路径 (Set save path)
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)

    def train(self, logger=None):
        print('lr:', self.config['lr'])
        if (logger):
            logger.info('start training......')

        # 初始化模型 (Initialize model)
        self.model = BiGRU_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'],
                                      self.config['model']['mlp']['dropout'], num_layers=1)
        if self.config['use_cuda']:
            self.model = self.model.cuda()

        # 定义损失函数和优化器 (Define loss function and optimizer)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

        # 初始化记录器 (Initialize recorder)
        recorder = Recorder(self.config['early_stop'])

        # 获取验证集数据加载器 (Get validation set data loader)
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'],
                                    self.config['batchsize'], shuffle=False, use_endef=True,
                                    aug_prob=self.config['aug_prob'])

        # 训练循环 (Training loop)
        for epoch in range(self.config['epoch']):
            self.model.train()

            # 获取训练集数据加载器 (Get training set data loader)
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'],
                                          self.config['batchsize'], shuffle=True, use_endef=True,
                                          aug_prob=self.config['aug_prob'])
            train_data_iter = tqdm.tqdm(train_loader)  # 使用 tqdm 显示进度条 (Use tqdm to display progress bar)
            avg_loss = Averager()

            # 批次训练 (Batch training)
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])  # 将数据移动到 GPU (Move data to GPU)
                label = batch_data['label']

                # 前向传播 (Forward propagation)
                pred, entity_pred, _ = self.model(**batch_data)

                # 计算损失 (Calculate loss)
                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float()) # 最终预测和实体预测的损失
                # 反向传播和优化 (Backward propagation and optimization)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            # 验证 (Validation)
            results = self.test(val_loader)
            mark = recorder.add(results) # 保存、提前停止等逻辑 (Save, early stopping, etc. logic)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_path, 'parameter_bigruendef.pkl'))
            elif mark == 'esc':
                break
            else:
                continue

        # 加载最佳模型 (Load best model)
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bigruendef.pkl')))

        # 在测试集上进行测试 (Test on the test set)
        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'],
                                             self.config['batchsize'], shuffle=False, use_endef=True,
                                             aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)

        # 记录测试结果 (Record test results)
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'],
                                                                                self.config['aug_prob'],
                                                                                future_results['metric']))
        print('future results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bigruendef.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()  # 设置为评估模式 (Set to evaluation mode)
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():  # 关闭梯度计算 (Turn off gradient calculation)
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']

                # 前向传播 (Forward propagation)
                batch_pred, _, _ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist()) #label转为list
                pred.extend(batch_pred.detach().cpu().numpy().tolist()) #pred转为list

        return metrics(label, pred) # 计算评估指标
```

**描述:**

*   `Trainer` 类负责模型的训练、验证和测试。
*   **初始化 (`__init__`)**: 接收配置信息，设置模型保存路径。
*   **训练 (`train`)**: 初始化模型、损失函数和优化器，循环遍历训练数据，进行前向传播、反向传播和优化，并在验证集上评估模型性能，根据验证集结果保存最佳模型，并在测试集上测试最终模型。
*   **测试 (`test`)**: 在给定的数据加载器上评估模型性能。

**如何使用:**

1.  创建包含训练参数的配置字典。
2.  实例化 `Trainer` 类，传入配置字典。
3.  调用 `train` 方法开始训练。

**Demo 使用示例:**

```python
# 导入需要的库 (Import required libraries)
import torch
from utils.utils import Averager, metrics, Recorder, data2gpu
from utils.dataloader import get_dataloader

# 假设已经定义了 BiGRU_ENDEFModel 类 (Assume BiGRU_ENDEFModel class is already defined)

class Trainer():
    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            os.makedirs(self.save_path)
            self.save_param_dir = self.save_path

    def train(self, logger=None):
        print('lr:', self.config['lr'])
        if (logger):
            logger.info('start training......')

        # 初始化模型 (Initialize model)
        self.model = BiGRU_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'],
                                      self.config['model']['mlp']['dropout'], num_layers=1)
        if self.config['use_cuda']:
            self.model = self.model.cuda()

        # 定义损失函数和优化器 (Define loss function and optimizer)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

        # 初始化记录器 (Initialize recorder)
        recorder = Recorder(self.config['early_stop'])

        # 获取验证集数据加载器 (Get validation set data loader)
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'],
                                    self.config['batchsize'], shuffle=False, use_endef=True,
                                    aug_prob=self.config['aug_prob'])

        # 训练循环 (Training loop)
        for epoch in range(self.config['epoch']):
            self.model.train()

            # 获取训练集数据加载器 (Get training set data loader)
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'],
                                          self.config['batchsize'], shuffle=True, use_endef=True,
                                          aug_prob=self.config['aug_prob'])
            train_data_iter = tqdm.tqdm(train_loader)  # 使用 tqdm 显示进度条 (Use tqdm to display progress bar)
            avg_loss = Averager()

            # 批次训练 (Batch training)
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])  # 将数据移动到 GPU (Move data to GPU)
                label = batch_data['label']

                # 前向传播 (Forward propagation)
                pred, entity_pred, _ = self.model(**batch_data)

                # 计算损失 (Calculate loss)
                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float()) # 最终预测和实体预测的损失
                # 反向传播和优化 (Backward propagation and optimization)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            # 验证 (Validation)
            results = self.test(val_loader)
            mark = recorder.add(results) # 保存、提前停止等逻辑 (Save, early stopping, etc. logic)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_path, 'parameter_bigruendef.pkl'))
            elif mark == 'esc':
                break
            else:
                continue

        # 加载最佳模型 (Load best model)
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bigruendef.pkl')))

        # 在测试集上进行测试 (Test on the test set)
        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'],
                                             self.config['batchsize'], shuffle=False, use_endef=True,
                                             aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)

        # 记录测试结果 (Record test results)
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'],
                                                                                self.config['aug_prob'],
                                                                                future_results['metric']))
        print('future results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bigruendef.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()  # 设置为评估模式 (Set to evaluation mode)
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():  # 关闭梯度计算 (Turn off gradient calculation)
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']

                # 前向传播 (Forward propagation)
                batch_pred, _, _ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist()) #label转为list
                pred.extend(batch_pred.detach().cpu().numpy().tolist()) #pred转为list

        return metrics(label, pred) # 计算评估指标

# 假设已经定义了 BiGRU_ENDEFModel 类 (Assume BiGRU_ENDEFModel class is already defined)
from torch.utils.data import Dataset, DataLoader
from utils.utils import Averager, metrics, Recorder, data2gpu

class DummyDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.emb_dim = 768

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 创造随机数据
        content = torch.randint(0, 1000, (50,))
        content_masks = torch.ones(50, dtype=torch.bool)
        entity = torch.randint(0, 1000, (10,))
        label = torch.randint(0, 2, (1,)).float()
        return {'content': content, 'content_masks': content_masks, 'entity': entity, 'label': label}

# Example Usage
if __name__ == '__main__':
    # 创建一个简单的配置字典 (Create a simple configuration dictionary)
    config = {
        'lr': 0.001,
        'weight_decay': 0.0001,
        'epoch': 2,
        'batchsize': 32,
        'use_cuda': False, # 更改为False，因为我们没有CUDA (Change to False because we don't have CUDA)
        'early_stop': 5,
        'save_param_dir': './saved_models',
        'model_name': 'bigru_endef',
        'root_path': './data/',
        'max_len': 128,
        'aug_prob': 0.0,
        'emb_dim': 768, # BERT 输出维度
        'model': {
            'mlp': {
                'dims': [256, 1],
                'dropout': 0.1
            }
        }
    }

    # 创建虚拟数据集和数据加载器 (Create dummy dataset and data loader)
    train_dataset = DummyDataset(length=100)
    val_dataset = DummyDataset(length=50)
    test_dataset = DummyDataset(length=50)

    def collate_fn(batch):
        content = [item['content'] for item in batch]
        content_masks = [item['content_masks'] for item in batch]
        entity = [item['entity'] for item in batch]
        label = [item['label'] for item in batch]

        # 填充到相同长度
        content = torch.nn.utils.rnn.pad_sequence(
            [item['content'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        content_masks = torch.nn.utils.rnn.pad_sequence(
            [item['content_masks'] for item in batch],
            batch_first=True,
            padding_value=False
        )
        entity = torch.nn.utils.rnn.pad_sequence(
            [item['entity'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        label = torch.cat(label)
        return {'content': content, 'content_masks': content_masks, 'entity': entity, 'label': label}

    train_loader = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batchsize'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batchsize'], shuffle=False, collate_fn=collate_fn)

    # 将数据加载器添加到配置中 (Add data loaders to the configuration)
    config['train_loader'] = train_loader
    config['val_loader'] = val_loader
    config['test_loader'] = test_loader

    def get_dataloader(path, max_len, batchsize, shuffle, use_endef, aug_prob):
      return config['train_loader']

    # 实例化 Trainer 类 (Instantiate Trainer class)
    trainer = Trainer(config)

    # 训练模型 (Train model)
    results, model_path = trainer.train()

    # 打印结果 (Print results)
    print("Training completed.")
    print("Results:", results)
    print("Model saved to:", model_path)
```

**关键点:**

*   **`config` 字典:** 存储训练所需的各种超参数和配置信息。
*   **虚拟数据 (`DummyDataset`):**  用于演示训练过程，避免依赖真实数据。
*   **`collate_fn` 函数:**  将不同长度的序列填充到相同长度，以便可以进行批处理。  `torch.nn.utils.rnn.pad_sequence`  函数用于填充序列。
*   **修改 `get_dataloader` 函数:** 由于这里使用DummyDataset，因此直接用config中的train_loader替换dataloader的返回.

**3. `data2gpu`, `Averager`, `metrics`, `Recorder` 类 (Classes):**

这些都是辅助类，用于数据处理、指标计算和训练过程记录。由于它们的代码没有提供，这里给出描述，需要在`utils.utils`中定义。

*   **`data2gpu`**:  将数据移动到 GPU 上。
*   **`Averager`**:  计算平均损失。
*   **`metrics`**:  计算评估指标（如准确率、精确率、召回率和 F1 值）。
*   **`Recorder`**:  记录训练过程中的验证集性能，并实现提前停止策略。

**如何在实际项目中使用:**

1.  准备真实的数据集，并使用 `utils.dataloader.get_dataloader` 函数创建数据加载器。
2.  根据实际情况调整配置字典中的参数。
3.  实例化 `Trainer` 类并调用 `train` 方法开始训练。
4.  训练完成后，使用测试集评估模型性能。

希望这个详细的解释和示例能帮助你理解代码。
