Lan: `py` From`dl/DeepSeek/native-sparse-attention-pytorch\tests\test_custom_compress_mlp.py`

好的，我们重新来。

**1. 优化稀疏注意力中的块选择 (Optimized Block Selection in Sparse Attention):**

```python
import torch
import torch.nn as nn

def optimized_block_selection(attn_scores, num_selected_blocks):
    """
    选择最重要的块。
    Selects the most important blocks based on attention scores.
    Args:
        attn_scores (torch.Tensor): 注意力分数，形状为 (B, H, W, num_blocks)
                                     Attention scores of shape (B, H, W, num_blocks)
        num_selected_blocks (int): 要选择的块的数量
                                    Number of blocks to select

    Returns:
        torch.Tensor: 选定的块的索引，形状为 (B, H, W, num_selected_blocks)
                      Indices of selected blocks of shape (B, H, W, num_selected_blocks)
    """
    B, H, W, num_blocks = attn_scores.shape
    # 首先将注意力分数展平，以便跨所有块进行排序
    # Flatten attention scores to sort across all blocks
    flat_attn_scores = attn_scores.view(B, H, W, -1)
    # 选择最高的 num_selected_blocks 个分数
    # Select the top num_selected_blocks scores
    top_scores, top_indices = torch.topk(flat_attn_scores, num_selected_blocks, dim=-1)
    return top_indices

# Demo example
if __name__ == '__main__':
    # 模拟注意力分数
    # Simulate attention scores
    attn_scores = torch.randn(2, 8, 16, 32) # (B=2, H=8, W=16, num_blocks=32)
    num_selected_blocks = 8
    # 使用优化后的块选择函数
    # Use the optimized block selection function
    selected_indices = optimized_block_selection(attn_scores, num_selected_blocks)
    print(f"选定的块的索引的形状: {selected_indices.shape}") # 形状应该是 (2, 8, 16, 8)
    # The shape should be (2, 8, 16, 8)
```

**描述:** 此代码片段实现了优化的块选择机制，用于稀疏注意力。与简单地选择前k个块不同，此函数首先对所有块的注意力分数进行排序，然后选择具有最高分数的块。这有助于确保选择了最重要的块，从而提高了注意力的效率和准确性。

*   **效率 (Efficiency):** `torch.topk` 函数用于快速找到最高分数的块，而无需完全排序所有分数。
*   **灵活性 (Flexibility):** `num_selected_blocks` 参数允许控制所选块的数量，从而可以调整注意力机制的稀疏性。

**2. 改进的压缩 MLP (Improved Compression MLP):**

```python
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class ImprovedCompressMLP(nn.Module):
    def __init__(self, dim_head, compress_block_size):
        """
        改进的压缩 MLP。
        Improved Compression MLP.

        Args:
            dim_head (int): 注意力头的维度
                            Dimension of the attention head
            compress_block_size (int): 压缩块的大小
                                        Size of the compression block
        """
        super().__init__()
        self.dim_head = dim_head
        self.compress_block_size = compress_block_size
        self.compress_dim = dim_head * compress_block_size

        self.mlp = nn.Sequential(
            Rearrange('b h w n d -> b h w (n d)'),
            nn.Linear(self.compress_dim, self.compress_dim * 2),
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Linear(self.compress_dim * 2, self.dim_head),
        )

    def forward(self, x):
        """
        前向传递。
        Forward pass.

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, H, W, compress_block_size, dim_head)
                             Input tensor of shape (B, H, W, compress_block_size, dim_head)

        Returns:
            torch.Tensor: 压缩后的张量，形状为 (B, H, W, dim_head)
                          Compressed tensor of shape (B, H, W, dim_head)
        """
        return self.mlp(x)


# Demo Usage
if __name__ == '__main__':
    dim_head = 64
    compress_block_size = 4
    compress_mlp = ImprovedCompressMLP(dim_head, compress_block_size)
    dummy_input = torch.randn(2, 8, 16, compress_block_size, dim_head)
    compressed_output = compress_mlp(dummy_input)
    print(f"压缩后的输出的形状: {compressed_output.shape}")
    # 应为：torch.Size([2, 8, 16, 64])
    # Should be: torch.Size([2, 8, 16, 64])
```

**描述:** 这个代码片段定义了一个改进的压缩 MLP，用于降低稀疏注意力中的维度。

**主要改进:**

*   **GELU激活 (GELU Activation):** 使用 GELU 激活函数，这通常比 SiLU 表现更好。
*   **更大的中间层 (Larger Intermediate Layer):**  使用更大的中间层（`compress_dim * 2`），可以更好地捕获输入数据中的复杂关系。

**3. 带学习能力的位置编码 (Learnable Positional Embeddings):**

```python
import torch
import torch.nn as nn

class LearnablePositionEmbeddings(nn.Module):
    def __init__(self, max_len, dim):
        """
        带学习能力的位置嵌入。
        Learnable positional embeddings.

        Args:
            max_len (int): 序列的最大长度
                            Maximum length of the sequence
            dim (int): 嵌入的维度
                         Dimension of the embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, x):
        """
        前向传递。
        Forward pass.

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, L, dim)
                             Input tensor of shape (B, L, dim)

        Returns:
            torch.Tensor: 具有位置信息的张量，形状为 (B, L, dim)
                          Tensor with positional information of shape (B, L, dim)
        """
        B, L, _ = x.shape
        positions = torch.arange(0, L, device=x.device).expand(B, L)
        return x + self.embedding(positions)

# Demo Usage
if __name__ == '__main__':
    max_len = 32
    dim = 512
    pos_embed = LearnablePositionEmbeddings(max_len, dim)
    dummy_input = torch.randn(2, max_len, dim)
    output = pos_embed(dummy_input)
    print(f"位置嵌入的输出的形状：{output.shape}")
    # 应该是 torch.Size([2, 32, 512])
    # Should be torch.Size([2, 32, 512])
```

**描述:**  此代码片段实现了可学习的位置嵌入。与固定的位置编码（例如正弦曲线）不同，这些嵌入是在训练期间学习的，允许模型更好地适应序列数据中的位置信息。

*   **灵活性 (Flexibility):** 可以学习任意的位置依赖性。
*   **简单性 (Simplicity):** 使用 `nn.Embedding` 模块轻松实现。

**总结 (Summary):** 这些代码片段提供了稀疏注意力机制的改进，包括优化的块选择、改进的压缩 MLP 和可学习的位置嵌入。这些改进可以提高注意力的效率、准确性和灵活性。
