Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\models\bigruendef.py`

好的，我们重新开始。我将提供更智能、更模块化的代码，并附带中文描述和示例。

**1. 更灵活的注意力机制 (FlexibleAttention):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleAttention(nn.Module):
    def __init__(self, input_dim, attention_type='dot'):
        super().__init__()
        self.attention_type = attention_type
        self.input_dim = input_dim
        if attention_type == 'mlp':
            self.attention_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1)
            )
        elif attention_type == 'linear':
            self.attention_linear = nn.Linear(input_dim, 1)

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len)  Optional mask to ignore padding tokens.

        Returns:
            attended_vector: (batch_size, input_dim) Context vector.
            attention_weights: (batch_size, seq_len) Attention weights.
        """
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(inputs, inputs.transpose(1, 2)).mean(dim=2) # (B, L)
        elif self.attention_type == 'mlp':
            attention_weights = self.attention_mlp(inputs).squeeze(-1)
        elif self.attention_type == 'linear':
            attention_weights = self.attention_linear(inputs).squeeze(-1)
        else:
            raise ValueError("Invalid attention type")

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)  # Mask padding

        attention_weights = F.softmax(attention_weights, dim=1)
        attended_vector = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)

        return attended_vector, attention_weights

# 示例用法 (Demo Usage):
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    input_dim = 64

    # 创建一个随机输入张量 (Create a random input tensor)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # 创建一个可选的 mask (Create an optional mask)
    dummy_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

    # 使用不同的注意力类型 (Use different attention types)
    attention_dot = FlexibleAttention(input_dim, attention_type='dot')
    attended_vector_dot, weights_dot = attention_dot(dummy_input, dummy_mask)
    print(f"Dot Attention Output Shape: {attended_vector_dot.shape}") # 输出点积注意力向量的形状 (Output shape of dot attention vector)

    attention_mlp = FlexibleAttention(input_dim, attention_type='mlp')
    attended_vector_mlp, weights_mlp = attention_mlp(dummy_input, dummy_mask)
    print(f"MLP Attention Output Shape: {attended_vector_mlp.shape}") # 输出MLP注意力向量的形状 (Output shape of MLP attention vector)

    attention_linear = FlexibleAttention(input_dim, attention_type='linear')
    attended_vector_linear, weights_linear = attention_linear(dummy_input, dummy_mask)
    print(f"Linear Attention Output Shape: {attended_vector_linear.shape}") # 输出线性注意力向量的形状 (Output shape of linear attention vector)
```

**描述:**

这段代码定义了一个更灵活的注意力机制 `FlexibleAttention`。 它支持三种不同的注意力计算方式：

*   `dot` (点积注意力): 直接计算输入向量之间的点积。
*   `mlp` (MLP 注意力): 使用一个多层感知机 (MLP) 来学习注意力权重。
*   `linear` (线性注意力): 使用一个线性层来计算注意力权重。

**主要改进:**

*   **多种注意力类型 (Multiple Attention Types):** 允许你选择最适合你的任务的注意力计算方式。
*   **可选的 Mask (Optional Mask):** 可以使用 mask 来忽略填充的 token，从而提高性能。
*   **更清晰的接口 (Clearer Interface):** `forward` 方法的输入和输出更加清晰明了。

**如何使用:**

1.  初始化 `FlexibleAttention` 类，指定输入维度和注意力类型。
2.  将输入张量和可选的 mask 传递给 `forward` 方法。

---

**2. 更模块化的 MLP (ModularMLP):**

```python
import torch
import torch.nn as nn

class ModularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation='relu', batchnorm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(dims[i+1])) # Batch Normalization
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(dims[-1], output_dim)) # Output layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 示例用法 (Demo Usage):
if __name__ == '__main__':
    input_dim = 64
    hidden_dims = [128, 64]
    output_dim = 1

    # 创建一个随机输入张量 (Create a random input tensor)
    dummy_input = torch.randn(1, input_dim)

    # 使用不同的配置 (Use different configurations)
    mlp_relu = ModularMLP(input_dim, hidden_dims, output_dim)
    output_relu = mlp_relu(dummy_input)
    print(f"ReLU MLP Output Shape: {output_relu.shape}") # 输出ReLU MLP的形状 (Output shape of ReLU MLP)

    mlp_sigmoid = ModularMLP(input_dim, hidden_dims, output_dim, activation='sigmoid')
    output_sigmoid = mlp_sigmoid(dummy_input)
    print(f"Sigmoid MLP Output Shape: {output_sigmoid.shape}") # 输出Sigmoid MLP的形状 (Output shape of Sigmoid MLP)

    mlp_batchnorm = ModularMLP(input_dim, hidden_dims, output_dim, batchnorm=True)
    output_batchnorm = mlp_batchnorm(dummy_input)
    print(f"BatchNorm MLP Output Shape: {output_batchnorm.shape}") # 输出带BatchNorm的MLP的形状 (Output shape of MLP with BatchNorm)
```

**描述:**

这段代码定义了一个更模块化的多层感知机 `ModularMLP`。

**主要改进:**

*   **可配置的激活函数 (Configurable Activation Function):** 允许你选择不同的激活函数，如 ReLU, Sigmoid, Tanh。
*   **批归一化 (Batch Normalization):** 可以选择是否在每一层之后添加批归一化层。
*   **更清晰的结构 (Clearer Structure):** 使用 `nn.ModuleList` 来管理层，使代码更易于阅读和维护。

**如何使用:**

1.  初始化 `ModularMLP` 类，指定输入维度、隐藏维度、输出维度、dropout 概率和激活函数。
2.  将输入张量传递给 `forward` 方法。

---

**3. 更新后的 `BiGRU_ENDEFModel` 和 `Trainer`:**

```python
import os
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader
from typing import Dict, Any, Optional, Tuple

# 引入新的模块 (Import new modules)
from .attention import FlexibleAttention
from .mlp import ModularMLP

class BiGRU_ENDEFModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.emb_dim = config['emb_dim']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.num_layers = config['num_layers']
        self.attention_type = config['attention_type']
        self.entity_feature_kernel = config['entity_feature_kernel']

        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.embedding = self.bert.embeddings

        self.rnn = nn.GRU(input_size=self.emb_dim,
                          hidden_size=self.emb_dim,
                          num_layers=self.num_layers,
                          batch_first=True,
                          bidirectional=True)

        input_shape = self.emb_dim * 2
        self.attention = FlexibleAttention(input_shape, attention_type=self.attention_type)
        self.mlp = ModularMLP(input_shape, self.mlp_dims, 1, self.dropout)

        mlp_input_shape = sum([self.entity_feature_kernel[kernel] for kernel in self.entity_feature_kernel])
        self.entity_convs = cnn_extractor(self.entity_feature_kernel, self.emb_dim)
        self.entity_mlp = ModularMLP(mlp_input_shape, self.mlp_dims, 1, self.dropout)
        self.entity_net = nn.Sequential(self.entity_convs, self.entity_mlp)

    def forward(self, content: torch.Tensor, content_masks: torch.Tensor, entity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            content: (batch_size, seq_len) Input text content.
            content_masks: (batch_size, seq_len) Mask for the content.
            entity: (batch_size, entity_len) Input entity.

        Returns:
            bias_pred: (batch_size) Prediction for bias.
            entity_prob: (batch_size) Prediction based on entity.
            content_prob: (batch_size) Prediction based on content.
        """
        emb_feature = self.embedding(content)
        feature, _ = self.rnn(emb_feature)
        feature, _ = self.attention(feature, content_masks)
        bias_pred = self.mlp(feature).squeeze(1)

        entity_feature = self.embedding(entity)
        entity_prob = self.entity_net(entity_feature).squeeze(1)
        content_prob = self.mlp(feature).squeeze(1)

        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), torch.sigmoid(content_prob)


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, logger: Optional[Any] = None) -> Tuple[Dict[str, float], str]:
        print('lr:', self.config['lr'])
        if logger:
            logger.info('start training......')

        self.model = BiGRU_ENDEFModel(self.config) # Directly pass the config
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = nn.BCELoss()
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

                pred, entity_pred, content_pred = self.model(content=batch_data['content'], content_masks=batch_data['content_masks'], entity=batch_data['entity'])

                loss = loss_fn(pred, label.float()) + 0.2 * loss_fn(entity_pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'parameter_bigruendef.pkl'))
            elif mark == 'esc':
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bigruendef.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=True, aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)

        if logger:
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('future results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bigruendef.pkl')

    def test(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                pred_raw, _, batch_pred = self.model(content=batch_data['content'], content_masks=batch_data['content_masks'], entity=batch_data['entity'])

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)
```

**主要改进:**

*   **使用`FlexibleAttention`和`ModularMLP`**:  模型现在使用更灵活和模块化的注意力机制和MLP。
*   **配置驱动的模型**:  `BiGRU_ENDEFModel`现在接收一个配置字典，使其更容易定制。
*   **类型提示**: 添加了类型提示，以提高可读性和可维护性。
*   **清晰的输入/输出**:  `forward`方法具有明确的输入和输出参数。
*   **直接传递配置**: Trainer 直接传递 config 给模型。
*   **简化 forward**: 简化了`forward`方法中的代码。
*   **使用了`os.makedirs(exist_ok=True)`**: 这可以避免在目录已存在时出现错误。
*  **返回content_prob**: 现在返回基于content的预测`content_prob`，并将其用于测试中的预测。

**4. 示例配置 (Example Configuration):**

```python
config = {
    'emb_dim': 128,
    'model': {
        'mlp': {
            'dims': [256, 128],
            'dropout': 0.1
        }
    },
    'num_layers': 1,
    'attention_type': 'dot',  # 'dot', 'mlp', 'linear'
    'entity_feature_kernel': {1: 64, 2: 64, 3: 64, 5: 64, 10: 64},
    'lr': 0.001,
    'weight_decay': 1e-5,
    'early_stop': 5,
    'epoch': 10,
    'batchsize': 32,
    'max_len': 128,
    'use_cuda': torch.cuda.is_available(),
    'root_path': './data/',  # 替换为你的数据路径 (Replace with your data path)
    'save_param_dir': './checkpoints/',  # 替换为你的保存路径 (Replace with your save path)
    'model_name': 'BiGRU_ENDEF',
    'aug_prob': 0.2
}
```

**描述:**

这是一个示例配置字典，用于初始化`BiGRU_ENDEFModel`和`Trainer`。 确保将`root_path`和`save_param_dir`替换为你的实际路径。 你还可以修改其他参数以进行实验。  `attention_type` 控制注意力机制的类型。

**总结:**

这段代码提供了一个更加模块化、可配置和易于使用的`BiGRU_ENDEFModel`和`Trainer`。 使用`FlexibleAttention`和`ModularMLP`可以更轻松地尝试不同的模型架构。 配置文件驱动的设计使得修改模型参数变得更加简单。
