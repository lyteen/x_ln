Lan: `py` From`dl/L/ENDEF-SIGIR2022/ENDEF_ch\utils\dataloader.py`

好的，我们重新开始，并提供更智能、更模块化的代码改进方案，并附带中文解释和演示。我们将重点放在以下几个方面：

1.  **更高效的 Tokenization:** 使用更快的分词器，并考虑缓存分词结果。
2.  **更灵活的数据增强:** 提供更多的数据增强策略，并允许更精细的控制。
3.  **更清晰的数据加载:** 将数据加载和预处理逻辑分离，提高代码可读性和可维护性。
4.  **更智能的参数管理:** 使用配置文件来管理各种参数。

**1. 更高效的 Tokenization:**

```python
from transformers import AutoTokenizer
import torch
import os

class CachedTokenizer:
    def __init__(self, model_name, cache_dir="token_cache"):
        """
        使用预训练模型的分词器，并缓存分词结果以提高效率。

        Args:
            model_name (str): 预训练模型的名称 (例如 "hfl/chinese-bert-wwm-ext").
            cache_dir (str): 缓存目录.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # 使用AutoTokenizer
        self.cache = {} # 用于存储已分词的结果

    def tokenize(self, text, max_length, padding='max_length', truncation=True):
        """
        对文本进行分词，使用缓存来避免重复计算。

        Args:
            text (str): 要分词的文本.
            max_length (int): 最大序列长度.
            padding (str): padding策略.
            truncation (bool): 是否截断.

        Returns:
            torch.Tensor: 分词后的token IDs.
            torch.Tensor: attention mask.
        """
        cache_key = f"{text[:100]}_{max_length}_{padding}_{truncation}"  # 使用文本的一部分作为键
        if cache_key in self.cache:
            return self.cache[cache_key]

        encoded_inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"  # 返回PyTorch tensors
        )
        token_ids = encoded_inputs["input_ids"].squeeze() # 去掉batch维度
        attention_mask = encoded_inputs["attention_mask"].squeeze()
        self.cache[cache_key] = (token_ids, attention_mask)
        return token_ids, attention_mask

    def __call__(self, text, max_length, padding='max_length', truncation=True):
        return self.tokenize(text, max_length, padding, truncation)

# 演示用法
if __name__ == '__main__':
    tokenizer = CachedTokenizer("hfl/chinese-bert-wwm-ext")
    text = "这是一个测试句子，用于演示缓存分词器的用法。"
    token_ids, attention_mask = tokenizer(text, max_length=128)
    print(f"Token IDs: {token_ids.shape}, {token_ids[:10]}")
    print(f"Attention Mask: {attention_mask.shape}, {attention_mask[:10]}")

    # 第二次分词，会从缓存中读取
    token_ids, attention_mask = tokenizer(text, max_length=128)
    print("第二次分词（从缓存读取）完成。")
```

**描述:**

*   `CachedTokenizer` 类使用 `transformers.AutoTokenizer` 来自动加载指定模型的tokenizer。
*   它使用一个 `cache` 字典来存储已经分词的结果，这样下次遇到相同的文本时，就可以直接从缓存中读取，避免重复计算。
*   `tokenize` 方法负责分词的具体实现，它首先检查缓存中是否存在结果，如果不存在，则进行分词，并将结果存储到缓存中。
*  使用 `AutoTokenizer` 可以兼容更多的模型。

**2. 更灵活的数据增强:**

```python
import jieba
import random

def random_mask(text, prob=0.15):
    """随机mask文本中的词语."""
    words = list(jieba.cut(text))
    masked_words = []
    for word in words:
        if random.random() < prob:
            masked_words.append("[MASK]")
        else:
            masked_words.append(word)
    return "".join(masked_words)


def random_delete(text, prob=0.15):
    """随机删除文本中的词语."""
    words = list(jieba.cut(text))
    deleted_words = [word for word in words if random.random() >= prob]
    return "".join(deleted_words)


def random_replace(text, word_list, prob=0.15):
    """随机替换文本中的词语."""
    words = list(jieba.cut(text))
    replaced_words = []
    for word in words:
        if random.random() < prob:
            replaced_words.append(random.choice(word_list))  # 从提供的词汇列表中随机选择
        else:
            replaced_words.append(word)
    return "".join(replaced_words)

def data_augment(content, entity_list, aug_prob, entity_mask_prob=0.5, entity_delete_prob=0.2):
    """
    综合的数据增强策略。

    Args:
        content: 原始文本内容
        entity_list: 实体列表
        aug_prob: 整体增强概率 (决定是否应用增强)
        entity_mask_prob: 实体mask概率
        entity_delete_prob: 实体删除概率
    """
    if random.random() > aug_prob:
        return content  # 不进行数据增强

    # 1. 实体增强
    for entity_data in entity_list:
        entity = entity_data["entity"]
        rand_num = random.random()
        if rand_num < entity_mask_prob:
            content = content.replace(entity, "[MASK]")
        elif rand_num < entity_mask_prob + entity_delete_prob:
            content = content.replace(entity, "")

    # 2. 全局文本增强
    rand_num = random.random()
    if rand_num < 0.33:
        content = random_mask(content)
    elif rand_num < 0.66:
        content = random_delete(content)
    else:
        # 可以添加一个词汇表，或者简单使用[UNK]
        content = random_replace(content, ["[UNK]", "[MASK]"])  # 使用[UNK]作为替代词

    return content

# 演示用法
if __name__ == '__main__':
    text = "腾讯是一家伟大的公司，在深圳。"
    entity_list = [{"entity": "腾讯"}, {"entity": "深圳"}]
    augmented_text = data_augment(text, entity_list, aug_prob=0.7)
    print(f"原始文本: {text}")
    print(f"增强后的文本: {augmented_text}")
```

**描述:**

*   提供了 `random_mask`, `random_delete`, `random_replace` 三种常用的数据增强方法。
*   `data_augment` 函数整合了实体增强和全局文本增强。
*   允许配置实体mask、删除的概率。
*  可以在`random_replace` 中定义自己的词汇表用于替换，例如使用`[UNK]`。

**3. 更清晰的数据加载:**

```python
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, use_endef, aug_prob):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_endef = use_endef
        self.aug_prob = aug_prob
        self.label_dict = {"real": 0, "fake": 1}
        self.category_dict = {
            "2010": 0, "2011": 1, "2012": 2, "2013": 3, "2014": 4,
            "2015": 5, "2016": 6, "2017": 7, "2018": 8, "2019": 9,
            "2020": 9, "2021": 9
        }

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def get_entity(self, entity_list):
        entity_content = '[SEP]'.join([item["entity"] for item in entity_list])
        return entity_content

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        content = item['content']
        entity_content = self.get_entity(item['entity_list'])

        if self.use_endef:
            content = data_augment(content, item['entity_list'], self.aug_prob)

        content_token_ids, content_masks = self.tokenizer(content, max_length=self.max_len)
        entity_token_ids, entity_masks = self.tokenizer(entity_content, max_length=50)

        label = torch.tensor(self.label_dict[item['label']], dtype=torch.long)
        year = torch.tensor(self.category_dict[item['time'].split(' ')[0].split('-')[0]], dtype=torch.long)
        emotion = torch.tensor(np.load(data_path.replace('.json', '_emo.npy'))[idx].astype('float32')) # 直接索引

        return (
            content_token_ids,
            content_masks,
            entity_token_ids,
            entity_masks,
            label,
            year,
            emotion
        )

def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader


# 演示用法 (需要准备一个.json和一个_emo.npy文件)
if __name__ == '__main__':
    # 假设你有一个名为 "train.json" 和 "train_emo.npy" 的文件
    data_path = "train.json"  # 替换为你的json文件路径

    # 创建一个假的train.json和train_emo.npy用于演示
    fake_data = [{"content": "这是一个测试文本，关于苹果公司。", "entity_list": [{"entity": "苹果公司"}], "label": "real", "time": "2023-10-26 10:00:00"},
                 {"content": "这是一个假新闻，声称腾讯要收购微软。", "entity_list": [{"entity": "腾讯"}, {"entity": "微软"}], "label": "fake", "time": "2022-05-15 15:30:00"}]
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(fake_data, f, ensure_ascii=False)

    fake_emotions = np.random.rand(len(fake_data), 128)  # 假设emotion特征是128维
    np.save("train_emo.npy", fake_emotions)

    tokenizer = CachedTokenizer("hfl/chinese-bert-wwm-ext")
    max_len = 128
    use_endef = True
    aug_prob = 0.5
    batch_size = 2

    dataset = CustomDataset(data_path, tokenizer, max_len, use_endef, aug_prob)
    dataloader = get_dataloader(dataset, batch_size)

    for batch in dataloader:
        content_token_ids, content_masks, entity_token_ids, entity_masks, labels, years, emotions = batch
        print(f"Content Token IDs shape: {content_token_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Emotions shape: {emotions.shape}")
        break # 仅打印一个batch
```

**描述:**

*   `CustomDataset` 类继承自 `torch.utils.data.Dataset`，负责数据的加载和预处理。
*   它将数据加载、分词、数据增强等操作封装在一起，使得代码更清晰易懂。
*   `get_dataloader` 函数负责创建 `DataLoader`，用于批量加载数据。
*   数据加载器直接索引 `emotion` 数据，避免了不必要的 DataFrame 操作。

**4. 更智能的参数管理:**

```python
import json

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.__dict__ = json.load(f)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

# 演示用法
if __name__ == '__main__':
    # 创建一个配置文件 config.json
    config_data = {
        "model_name": "hfl/chinese-bert-wwm-ext",
        "max_len": 128,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "use_data_augmentation": True,
        "augmentation_probability": 0.5
    }
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

    config = Config("config.json")
    print(f"Model Name: {config.model_name}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Use Data Augmentation: {config.use_data_augmentation}")
```

**描述:**

*   `Config` 类从 JSON 文件中加载配置参数，并允许通过属性访问。
*   这使得参数管理更加方便，避免了在代码中硬编码参数。

**整合:**

现在，你可以将这些改进后的组件整合到你的代码中。例如，在 `get_dataloader` 函数中使用 `Config` 类来加载参数，使用 `CachedTokenizer` 类来进行分词，并使用改进后的数据增强策略。

**总结:**

通过以上改进，你的代码将变得更高效、更灵活、更易于维护。 这些仅仅是一些建议，可以根据你的具体需求进行调整。 记住，编写高质量代码的关键在于模块化、可读性和可维护性。
