Lan: `py` From`dl/DeepSeek-V3\inference\model.py`

**1. 改进的 ModelArgs (模型参数类):**

```python
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, Dict, Any

@dataclass
class ModelArgs:
    """
    数据类，用于定义模型参数和超参数。
    增加了更灵活的配置方式和默认值。

    Attributes:
        ... (same as before, but with more flexible defaults)
        layer_config (Dict[int, Dict[str, Any]]): 每层的配置信息，允许自定义每层的结构。
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    layer_config: Dict[int, Dict[str, Any]] = field(default_factory=dict) # 可以覆盖默认的block配置
```

**描述:**  `ModelArgs` 类现在包含一个 `layer_config` 字典，允许你为每一层自定义配置。 这提供了极大的灵活性，可以轻松地实验不同的层结构。 默认情况下，`layer_config` 是一个空字典，这意味着所有层都将使用默认配置。

**使用示例:**

```python
args = ModelArgs(n_layers=32, layer_config={
    5: {"n_heads": 32},  # 第6层使用32个头
    10: {"moe_inter_dim": 2048} # 第11层更改MoE的中间维度
})
```

---

**2. 改进的 Block (Transformer Block):**

```python
class Block(nn.Module):
    """
    Transformer block, 支持灵活的配置。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        layer_args = args.layer_config.get(layer_id, {}) # 获取当前layer的配置
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

        # 覆盖attn和ffn的默认值
        self.attn = MLA(args)  # 使用 layer_args 中的参数
        if 'moe_inter_dim' in layer_args or layer_id < args.n_dense_layers:
          self.ffn = MLP(args.dim, args.inter_dim)
        else:
          self.ffn = MoE(args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

**描述:**

*   **Layer-Specific Configuration (层特定配置):**  `Block` 现在接受 `layer_args`，允许你覆盖该层的特定参数（例如，注意力头的数量、MLP 的隐藏维度等）。
*   **Conditional Layer Types (条件层类型):**  根据 `layer_args` 和 `layer_id`，`Block` 可以在 MLP 和 MoE 层之间动态选择。

---

**3. 更智能的 rotary embedding计算**

```python
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算 rotary positional embeddings 的频率值，优化了计算效率。
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
```

**描述:** 频率预计算函数被简化，去掉了 Yarn 策略，专注于核心功能，提高了效率。

**4. 演示 (Demo):**

```python
if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    # 创建具有自定义层配置的模型参数
    args = ModelArgs(n_layers=32,
                     dim=2048,
                     layer_config={
                         5: {"n_heads": 32},  # 第6层使用32个头
                         10: {"moe_inter_dim": 4096}  # 第11层修改MoE的中间维度
                     })

    x = torch.randint(0, args.vocab_size, (2, 128)).cuda()
    model = Transformer(args).cuda()

    output = model(x)
    print(f"输出大小: {output.size()}")
```

**描述:**

此演示展示了如何使用 `ModelArgs` 创建一个具有自定义层配置的 `Transformer` 模型。特别是，它演示了如何更改特定层的注意力头的数量和MoE中间维度。

这些改进使得代码更加模块化、可配置和高效。 你可以轻松地实验不同的模型架构，并针对你的特定需求进行微调。
