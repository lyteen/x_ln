Lan: `py` From`X_ln\dl/DeepSeek-VL2\deepseek_vl2\__init__.html`

好的，重新开始。 这次我会更注重代码的简洁性和可读性，并使用更常用的库和模式。

**1. 改进的 Tokenizer（分词器）:**

```python
import re

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab else {}
        self.index_to_token = {i: token for token, i in self.vocab.items()}

    def train(self, text):
        """基于文本训练分词器，构建词汇表."""
        words = re.findall(r'\b\w+\b', text.lower())  # 简单地按单词分割
        self.vocab = {token: i for i, token in enumerate(sorted(set(words)))}
        self.index_to_token = {i: token for token, i in self.vocab.items()}


    def encode(self, text):
        """将文本编码为token ID列表."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(word, len(self.vocab)) for word in words] # OOV处理

    def decode(self, token_ids):
        """将token ID列表解码为文本."""
        return " ".join([self.index_to_token.get(idx, "<UNK>") for idx in token_ids])

# Demo Usage
if __name__ == '__main__':
    text = "This is a simple example. This example is simple."
    tokenizer = SimpleTokenizer()
    tokenizer.train(text) # 训练分词器

    encoded = tokenizer.encode(text)
    print(f"编码后的文本: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"解码后的文本: {decoded}")

    print(f"词汇表: {tokenizer.vocab}")
```

**描述:**  这个 `SimpleTokenizer` 类将文本分割成单词，并将其编码成 ID。

**主要特点:**

*   **简单易懂:** 使用正则表达式进行简单的单词分割。
*   **词汇表构建:** `train` 方法用于基于文本构建词汇表。
*   **OOV处理:**  `encode` 方法处理词汇表外的单词 (OOV)，返回词汇表长度作为 OOV 的 ID。
*   **双向映射:** 维护 `vocab` (token到ID) 和 `index_to_token` (ID到token) 两个字典，方便编码和解码。

**如何使用:** 创建 `SimpleTokenizer` 实例，调用 `train` 方法训练分词器，然后使用 `encode` 和 `decode` 方法进行编码和解码。

---

**2. 改进的Transformer Block:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        # Multi-Head Attention
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = x + self.dropout(attn_output) # Residual connection + Dropout
        x = self.norm1(x) # Layer Normalization

        # Feed Forward Network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output) # Residual connection + Dropout
        x = self.norm2(x) # Layer Normalization

        return x


# Demo Usage
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    ff_dim = 128

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    dummy_input = torch.randn(batch_size, seq_len, embed_dim)
    output = transformer_block(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
```

**描述:**  这个 `TransformerBlock` 类实现了一个标准的 Transformer 块。

**主要特点:**

*   **标准结构:** 包含 Multi-Head Attention、Layer Normalization 和 Feed Forward Network。
*   **残差连接:** 使用残差连接来避免梯度消失。
*   **Dropout:** 应用 Dropout 来正则化模型。
*   **Layer Normalization:** 在残差连接之后应用 Layer Normalization。

**如何使用:**  创建 `TransformerBlock` 实例，指定嵌入维度、头的数量、前馈网络的维度和 Dropout 率。  然后，将输入张量传递给 `forward` 方法。

---

**3. 改进的简单GPT模型:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(1024, embed_dim) # 假设最大序列长度为 1024
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(embed_dim, vocab_size) # logits

    def forward(self, x):
        """
        x: (batch_size, seq_len) - Token IDs
        """
        batch_size, seq_len = x.shape
        # Embedding
        x = self.embedding(x) # (batch_size, seq_len, embed_dim)
        # Positional Encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.pos_embedding(positions) # (batch_size, seq_len, embed_dim)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x) # (batch_size, seq_len, embed_dim)

        # Linear layer for prediction
        logits = self.linear(x) # (batch_size, seq_len, vocab_size)

        return logits

    def generate(self, prompt, max_length=50):
        """Generate text given a prompt."""
        self.eval()  # Set to evaluation mode
        device = next(self.parameters()).device  # Get device

        with torch.no_grad():
            tokenized_prompt = tokenizer.encode(prompt)
            input_sequence = torch.tensor([tokenized_prompt], dtype=torch.long).to(device)

            for _ in range(max_length):
                logits = self.forward(input_sequence[:, -1024:]) # Truncate to max length of pos embedding
                last_token_logits = logits[:, -1, :]  # Get logits for the last token
                probabilities = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)  # Sample the next token

                input_sequence = torch.cat([input_sequence, next_token], dim=1)

                if next_token.item() == len(tokenizer.vocab): # EOS token (OOV)
                    break

            generated_text = tokenizer.decode(input_sequence[0].tolist())
            return generated_text

# Demo Usage
if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 100 # 示例
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_layers = 2
    dropout = 0.1

    # Create model
    model = SimpleGPT(vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout)

    # Dummy Input
    batch_size = 2
    seq_len = 20
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}")

    # Training (示例)
    # 假设有一个损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 随机目标
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    print(f"Loss: {loss.item()}")

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Text Generation Example
    text = "The quick brown fox jumps over the lazy dog."
    tokenizer = SimpleTokenizer()
    tokenizer.train(text)
    model = SimpleGPT(len(tokenizer.vocab) + 1, embed_dim, num_heads, ff_dim, num_layers, dropout) # +1 for OOV/EOS

    # 初始化 Embedding 层 (重要!) 必须在训练后初始化，保证模型和分词器的 vocab size 一致
    model.embedding = nn.Embedding(len(tokenizer.vocab) + 1, embed_dim) #  + 1 for OOV token

    prompt = "The quick brown"
    generated_text = model.generate(prompt, max_length=20)
    print(f"Generated Text: {generated_text}")
```

**描述:**  这个 `SimpleGPT` 类实现了一个简化的 GPT 模型。

**主要特点:**

*   **Embedding Layer:** 使用 `nn.Embedding` 将 token ID 转换为嵌入向量。
*   **Positional Encoding:**  使用 `nn.Embedding` 添加位置编码。
*   **Transformer Blocks:** 使用多个 `TransformerBlock` 层。
*   **Linear Layer:** 使用线性层将 Transformer 块的输出转换为 logits。
*   **Generation Function:** 实现了一个简单的 `generate` 函数用于文本生成。
*   **完整的示例** 包含了 tokenizer 训练, 模型训练 (虽然只是一个 epoch 的演示), 以及文本生成.
*   **OOV 处理:** 在生成时，如果采样到 OOV token （对应词汇表的大小），则停止生成。

**关键点:**

*   **初始化 `nn.Embedding`:**  在训练后使用 `model.embedding = nn.Embedding(len(tokenizer.vocab) + 1, embed_dim)` 来同步模型和分词器的词汇表大小.  这非常重要!  如果词汇表大小不匹配，会报错或者生成错误的结果。
*   **生成长度限制:**  `generate` 函数中使用 `input_sequence[:, -1024:]` 来限制输入到模型的序列长度，防止超出位置编码的范围。
*   **EOS Token:**  在 `generate` 函数中，如果生成的 token 是 OOV token，则认为到达了句子的结尾(EOS)，停止生成.
*   **模型训练:** 虽然示例中只进行了一个 epoch 的训练，但是展示了如何设置损失函数，优化器，进行反向传播和参数更新。  实际训练需要更多的 epoch 和更大的数据集。

**如何使用:**

1.  创建 `SimpleTokenizer` 实例，并用训练数据进行训练。
2.  创建 `SimpleGPT` 实例，确保 `vocab_size` 与 `len(tokenizer.vocab) + 1` (加上OOV token) 相匹配。
3.  **重要:**  初始化 `model.embedding` 以匹配 tokenizer 的词汇表大小.
4.  使用训练数据训练模型。
5.  使用 `generate` 函数生成文本。

这些代码提供了更清晰、更实用的示例，并包含了一些重要的改进和解释。 希望对你有帮助！
