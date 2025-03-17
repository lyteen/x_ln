Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\utils\dataloader.py`

好的，我将重新开始，并详细解释代码的每个部分，并附上中文描述和使用示例。

**1. 字典定义 (Dictionary Definitions):**

```python
label_dict = {
    "real": 0,
    "fake": 1
}

category_dict = {
    "2010": 0,
    "2011": 1,
    "2012": 2,
    "2013": 3,
    "2014": 4,
    "2015": 5,
    "2016": 6,
    "2017": 7,
    "2018": 8,
    "2019": 9,
    "2020": 9,
    "2021": 9
}
```

**描述:**  `label_dict` 将字符串标签 "real" 和 "fake" 映射到整数 0 和 1，用于分类任务。 `category_dict` 将年份字符串映射到整数类别，可能是为了表示时间信息。 注意2020和2021年共享同一个类别9。

**如何使用:**  这两个字典用于将原始数据中的文本标签和年份转换为模型可以理解的数字表示。 例如，如果数据集中有一条数据的标签是 "fake"，你可以使用 `label_dict["fake"]` 获取对应的数字表示 1。

**2. `word2input` 函数:**

```python
from transformers import BertTokenizer
import torch

def word2input(texts, max_len):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks
```

**描述:** `word2input` 函数使用预训练的 BERT tokenizer (`hfl/chinese-bert-wwm-ext`) 将文本转换为 BERT 可以理解的输入格式。它将文本编码为 token ID，并生成 attention mask。

**详细解释:**

*   `tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')`:  加载预训练的中文 BERT tokenizer。
*   `tokenizer.encode(...)`:  将文本编码为 token ID。`max_length` 指定最大序列长度，`add_special_tokens=True` 添加 BERT 特殊 token (例如 `[CLS]` 和 `[SEP]`)，`padding='max_length'` 将序列填充到 `max_length`，`truncation=True` 截断超过 `max_length` 的序列。
*   `masks`:  生成 attention mask，用于指示哪些 token 是真实的 (值为 1)，哪些是填充的 (值为 0)。

**如何使用:**

```python
# 示例
texts = ["这是一段中文文本。", "这是另一段文本。"]
max_len = 128
token_ids, masks = word2input(texts, max_len)
print("Token IDs 形状:", token_ids.shape)  # 输出: Token IDs 形状: torch.Size([2, 128])
print("Masks 形状:", masks.shape)  # 输出: Masks 形状: torch.Size([2, 128])
```

**3. `get_entity` 函数:**

```python
def get_entity(entity_list):
    entity_content = []
    for item in entity_list:
        entity_content.append(item["entity"])
    entity_content = '[SEP]'.join(entity_content)
    return entity_content
```

**描述:**  `get_entity` 函数接收一个实体列表，提取每个实体的 "entity" 字段，并将它们用 `[SEP]` 分隔符连接成一个字符串。

**如何使用:**

```python
# 示例
entity_list = [{"entity": "苹果"}, {"entity": "三星"}]
entity_content = get_entity(entity_list)
print(entity_content)  # 输出: 苹果[SEP]三星
```

**4. `data_augment` 函数:**

```python
import random
import jieba

def data_augment(content, entity_list, aug_prob):
    entity_content = []
    random_num = random.randint(1,100)
    if random_num <= 50:
        for item in entity_list:
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                content = content.replace(item["entity"], '[MASK]')
            elif random_num <= int(2 * aug_prob * 100):
                content = content.replace(item["entity"], '')
            else:
                entity_content.append(item["entity"])
        entity_content = '[SEP]'.join(entity_content)
    else:
        content = list(jieba.cut(content))
        for index in range(len(content) - 1, -1, -1):
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):
                del content[index]
            elif random_num <= int(2 * aug_prob * 100):
                content[index] = '[MASK]'
        content = ''.join(content)
        entity_content = get_entity(entity_list)

    return content, entity_content
```

**描述:** `data_augment` 函数用于数据增强。它根据概率 `aug_prob` 随机地将实体替换为 `[MASK]` 或删除， 或者使用jieba对句子进行随机mask和删除token。 目标是增加数据的多样性，提高模型的泛化能力。

**详细解释:**

*   前50%概率使用实体增强。根据 `aug_prob`，实体可能会被替换为 `[MASK]`，删除，或者保持不变。如果实体保持不变，它会被添加到 `entity_content` 中。
*   后50%概率使用分词增强。使用 `jieba.cut` 对文本进行分词，然后根据 `aug_prob` 随机地删除或替换 token 为 `[MASK]`。

**如何使用:**

```python
# 示例
content = "苹果公司发布了新的iPhone手机。"
entity_list = [{"entity": "苹果公司"}, {"entity": "iPhone手机"}]
aug_prob = 0.2
augmented_content, entity_content = data_augment(content, entity_list, aug_prob)
print("增强后的内容:", augmented_content)
print("实体内容:", entity_content)
```

**5. `get_dataloader` 函数:**

```python
import torch
import pandas as pd
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_dataloader(path, max_len, batch_size, shuffle, use_endef, aug_prob):
    data_list = json.load(open(path, 'r',encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content','label'))
    for item in data_list:
        tmp_data = {}
        if shuffle == True and use_endef == True:
            tmp_data['content'], tmp_data['entity'] = data_augment(item['content'], item['entity_list'], aug_prob)
        else:
            tmp_data['content'] = item['content']
            tmp_data['entity'] = get_entity(item['entity_list'])
        tmp_data['label'] = item['label']
        tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        df_data = df_data.append(tmp_data, ignore_index=True)
    emotion = np.load(path.replace('.json', '_emo.npy')).astype('float32')
    emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    entity_content = df_data['entity'].to_numpy()
    label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
    year = torch.tensor(df_data['year'].apply(lambda c: category_dict[c]).astype(int).to_numpy())
    content_token_ids, content_masks = word2input(content, max_len)
    entity_token_ids, entity_masks = word2input(entity_content, 50)
    dataset = TensorDataset(content_token_ids,
                            content_masks,
                            entity_token_ids,
                            entity_masks,
                            label,
                            year,
                            emotion
                            )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader
```

**描述:** `get_dataloader` 函数加载数据，进行数据增强 (如果 `shuffle` 和 `use_endef` 都为 True)，将文本转换为 token ID，并创建 PyTorch DataLoader。

**详细解释:**

*   `data_list = json.load(open(path, 'r',encoding='utf-8'))`:  从 JSON 文件加载数据。
*   数据处理：从加载的JSON数据中提取content, entity, label和year。
*   `emotion = np.load(path.replace('.json', '_emo.npy')).astype('float32')`: 加载预先计算的情感特征。
*   `word2input(...)`:  使用 `word2input` 函数将文本和实体内容转换为 token ID 和 attention mask。
*   `TensorDataset`:  创建 PyTorch TensorDataset，将所有输入数据组合在一起。
*   `DataLoader`:  创建 PyTorch DataLoader，用于批量加载数据，进行 shuffle 等操作。

**如何使用:**

```python
# 示例
path = "data.json"  # 替换为你的数据文件路径
max_len = 128
batch_size = 32
shuffle = True
use_endef = True
aug_prob = 0.2

# 创建一个虚拟的 data.json 文件
data = [
    {"content": "苹果发布新手机", "entity_list": [{"entity": "苹果"}], "label": "real", "time": "2023-10-26 10:00:00"},
    {"content": "三星发布新平板", "entity_list": [{"entity": "三星"}], "label": "fake", "time": "2022-09-15 14:30:00"}
]
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# 创建一个虚拟的 data_emo.npy 文件
emotion = np.random.rand(2, 10)  # 假设情感特征是 2x10 的数组
np.save("data_emo.npy", emotion)


dataloader = get_dataloader(path, max_len, batch_size, shuffle, use_endef, aug_prob)

# 迭代 DataLoader
for batch in dataloader:
    content_token_ids, content_masks, entity_token_ids, entity_masks, label, year, emotion = batch
    print("Content Token IDs 形状:", content_token_ids.shape)  #例如: Content Token IDs 形状: torch.Size([32, 128])
    print("Label 形状:", label.shape)  #例如: Label 形状: torch.Size([32])
    break # 演示一个batch
```

**总结:**

这段代码提供了一个完整的数据加载和预处理流程，用于文本分类任务。它使用 BERT tokenizer 将文本转换为 token ID，进行数据增强，并创建 PyTorch DataLoader，方便模型训练。 示例代码和注释应该能帮助你理解每个部分的功能和使用方法。请确保安装了所有需要的库 (transformers, pandas, numpy, jieba, torch)。
