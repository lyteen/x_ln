Lan: `py` From`dl/DeepSeek-V3\inference\generate.py`

**1.  改进的 `sample` 函数 (Improved `sample` function):**

```python
import torch
import torch.nn.functional as F

def sample(logits, temperature: float = 1.0, top_p: float = 0.0):
    """
    从 logits 中采样 token，使用温度缩放和 Top-p 采样。

    Args:
        logits (torch.Tensor): token 预测的 logits 张量。
        temperature (float, optional):  用于缩放 logits 的温度。 Defaults to 1.0.
        top_p (float, optional):  Top-p 采样的概率阈值。 Defaults to 0.0 (禁用 Top-p)。

    Returns:
        torch.Tensor: 采样的 token。
    """
    if temperature > 0:
        logits = logits / temperature
    else:
        # 避免除以零错误
        logits = logits / 1e-5

    probs = F.softmax(logits, dim=-1)  # 使用 softmax 将 logits 转换为概率

    if top_p > 0.0:
        # Top-p (nucleus) 采样
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)  # 重新映射到原始索引
    else:
        # 简单贪婪采样 (温度不为零时是带温度的采样)
        next_token = torch.multinomial(probs, num_samples=1) if temperature > 0 else torch.argmax(probs, dim=-1, keepdim=True)

    return next_token.squeeze(1)  # 删除多余的维度

# 演示用法 (Demo Usage):
if __name__ == '__main__':
    # 创建一些模拟 logits
    logits = torch.randn(1, 10)  # 假设有 10 个 token
    sampled_token = sample(logits, temperature=0.7, top_p=0.9)
    print(f"Sampled token: {sampled_token}")  # 打印采样到的 token 索引
```

**描述 (Description):**

*   这段代码改进了 `sample` 函数，增加了 Top-p 采样（也称为 nucleus sampling）选项。 Top-p 采样可以生成更多样化的文本，避免模型过于自信地选择概率最高的 token。
*   **Top-p 采样 (Top-p Sampling):**  只保留概率最高的 token，其概率总和达到 `top_p`，然后对这些 token 进行重新归一化并采样。
*   **Multinomial Sampling (多项式采样):** 使用 `torch.multinomial` 从概率分布中采样 token。
*   **温度缩放 (Temperature Scaling):**  仍然保留了温度缩放，允许控制生成文本的随机性。
*   **零温度时的贪婪解码 (Greedy Decoding at Zero Temperature):** 添加了在 `temperature=0` 时的贪婪解码路径，以进行确定性生成。
*   **避免除以零 (Avoid Division by Zero):** 增加了防止零温度时除以零的保护措施。

**2. 改进的 `generate` 函数 (Improved `generate` function):**

```python
import torch
from typing import List

@torch.inference_mode()
def generate(
    model: torch.nn.Module,  # 确保模型类型正确
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.0,
    pad_id: int = 0  # 添加 pad_id 参数
) -> List[List[int]]:
    """
    基于给定的 prompt token 使用指定的模型生成新的 token。

    Args:
        model (torch.nn.Module): 用于 token 生成的 Transformer 模型。
        prompt_tokens (List[List[int]]): 包含每个序列的 prompt token 的列表的列表。
        max_new_tokens (int): 要生成的新 token 的最大数量。
        eos_id (int): 序列结束 token ID。
        temperature (float, optional):  采样温度。 Defaults to 1.0.
        top_p (float, optional): Top-p 采样参数。 Defaults to 0.0.
        pad_id (int, optional): 用于填充序列的 token ID. Defaults to 0.

    Returns:
        List[List[int]]: 包含每个序列的生成 token 的列表的列表。
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    max_prompt_len = max(prompt_lens)  # 获取最长 prompt 的长度

    # 验证 Prompt 长度
    assert max_prompt_len <= model.max_seq_len, f"Prompt 长度超过模型最大序列长度 (max_seq_len={model.max_seq_len})"

    # 初始化 Token 张量
    total_len = min(model.max_seq_len, max_new_tokens + max_prompt_len)
    tokens = torch.full((len(prompt_tokens), total_len), pad_id, dtype=torch.long, device="cuda")

    # 填充 Prompt Token
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != pad_id  # 使用 pad_id 创建 mask

    # 生成循环
    for cur_pos in range(max_prompt_len, total_len): # 从最长prompt之后开始
        # 前向传播
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos) # 只传入当前需要计算的部分

        # 采样下一个 Token
        next_token = sample(logits[:, -1, :], temperature, top_p) # 从最后一个位置采样

        # 更新 Token 张量
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token) # 如果是prompt则保持不变
        tokens[:, cur_pos] = next_token

        # 检查是否完成
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id) # 检查非prompt部分是否生成了eos token
        prev_pos = cur_pos

        # 如果所有序列都已完成，则中断循环
        if finished.all():
            break

    # 提取生成 token
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:]  # 移除 prompt 部分
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]  # 截断到 EOS token
        completion_tokens.append(toks)

    return completion_tokens
```

**描述 (Description):**

*   **Pad ID (填充 ID):** 增加了 `pad_id` 参数，用于填充不同长度的 prompt，使它们能够批处理。 默认为0.
*   **Prompt Mask (Prompt 掩码):** 使用 `pad_id` 创建 prompt 掩码，以区分 prompt token 和生成的 token。
*   **Efficient Forward Pass (高效前向传播):**  在生成循环中，只将当前需要计算的部分 `tokens[:, prev_pos:cur_pos]` 传递给模型，这可以节省计算量。
*   **Top-p 采样支持 (Top-p Sampling Support):**  集成了新的 `sample` 函数，允许使用 Top-p 采样。
*   **Early Stopping (提前停止):** 如果所有序列都生成了 EOS token，则提前停止生成。
*   **Clearer Logic (更清晰的逻辑):** 使代码更易于理解和维护。
*   **类型提示 (Type Hints):** 添加了类型提示，以提高代码的可读性和可维护性。
*  **只从最后一个位置采样:**  `sample` 函数现在只从 logits 的最后一个位置采样 (`logits[:, -1, :]`)，因为我们一次只生成一个 token。

**3.  更新 `main` 函数 (Updated `main` function):**

```python
def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.0,  # 添加 top_p 参数
) -> None:
    """
    加载模型并执行交互式或批量文本生成的主函数。

    Args:
        ckpt_path (str): 模型 checkpoint 目录的路径。
        config (str): 模型配置文件的路径。
        input_file (str, optional): 包含输入 prompt 的文件的路径。 Defaults to "".
        interactive (bool, optional): 是否在交互模式下运行。 Defaults to True.
        max_new_tokens (int, optional): 要生成的新 token 的最大数量。 Defaults to 100.
        temperature (float, optional): 采样温度。 Defaults to 1.0.
        top_p (float, optional): Top-p 采样参数。 Defaults to 0.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)  # 确保 token 化
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, top_p, tokenizer.pad_token_id)  # 传递 pad_id
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"prompt 数量超过最大 batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True) for prompt in prompts] # 确保 token 化
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature, top_p, tokenizer.pad_token_id) # 传递 pad_id
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()
```

**4. 更新 `if __name__ == "__main__":` 部分 (Updated `if __name__ == '__main__':` section):**

```python
if __name__ == "__main__":
    """
    分布式文本生成的命令行界面。

    参数:
        --ckpt-path (str): 模型 checkpoint 目录的路径。
        --config (str): 模型配置文件的路径。
        --input-file (str, optional): 包含批量处理 prompt 的文件。
        --interactive (bool, optional): 启用交互模式以生成文本。
        --max-new-tokens (int, optional): 要生成的新 token 的最大数量。 Defaults to 200.
        --temperature (float, optional): 采样温度。 Defaults to 0.2.
        --top_p (float, optional): Top-p 采样参数. Defaults to 0.0.

    Raises:
        AssertionError: 如果未指定 input-file 并且未启用交互模式。
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.0)  # 添加 top_p 参数
    args = parser.parse_args()
    assert args.input_file or args.interactive, "必须指定 input-file 或启用交互模式"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.top_p) # 传递 top_p 参数
```

**主要更改摘要 (Summary of Main Changes):**

*   **Top-p 采样 (Top-p Sampling):**  集成了 Top-p 采样，以实现更多样化的文本生成。
*   **Pad ID (填充 ID):** 添加了 `pad_id`，以处理不同长度的 prompt。
*   **Efficient Forward Pass (高效前向传播):** 优化了前向传播过程，只传递必要的部分。
*   **Tokenization (Token 化):** 确保 prompt_tokens 在传递给 `generate` 函数之前被正确 token 化。 使用 `tokenizer.apply_chat_template(..., tokenize=True)`.
*  **传递 pad_id:** 将 `tokenizer.pad_token_id` 传递给 `generate` 函数。

这些改进应该使代码更加健壮、高效和灵活。 通过添加 Top-p 采样，可以生成更具创造性的文本。 通过处理不同长度的 prompt 和优化前向传播，代码可以更有效地处理各种输入。
