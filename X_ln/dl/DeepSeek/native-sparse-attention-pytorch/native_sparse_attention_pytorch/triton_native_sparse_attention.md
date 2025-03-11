Lan: `py` From`dl/DeepSeek/native-sparse-attention-pytorch\native_sparse_attention_pytorch\triton_native_sparse_attention.py`


**1. 辅助函数 (Helper Functions):**

```python
from functools import partial
import math
from math import ceil

import torch
from torch import Tensor, arange
import torch.nn.functional as F

import einx
from einops import repeat, rearrange, reduce

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(n, mult):
    return ceil(n / mult) * mult

def pad_at_dim(t, pad: tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_to_multiple(t, mult, *, dim):
    length = t.shape[dim]
    padded_length = round_up_multiple(length, mult)
    remainder = padded_length - length
    return pad_at_dim(t, (0, remainder), dim = dim)

def is_contiguous(x: Tensor):
    return x.stride(-1) == 1
```

**描述:** 这部分代码定义了一些辅助函数，用于处理张量、数值计算和一些存在性检查。例如 `exists` 检查变量是否为 `None`，`default` 提供默认值，`divisible_by` 检查整除性，`round_up_multiple` 将数字向上舍入到倍数，`pad_at_dim` 和 `pad_to_multiple` 用于张量填充，`is_contiguous` 检查张量是否连续。

**如何使用:** 这些函数在代码的其他部分广泛使用，用于确保数据的有效性、进行必要的填充和执行各种计算。例如，`round_up_multiple` 用于对序列长度进行填充，使其能够被 Triton block size 整除。
**2. Triton 环境检查 (Triton Environment Check):**

```python
TRITON_BLOCK_SIZE = 128 # some block size that allows triton not to break, at least half a year ago

INSTALL_COMMAND = 'pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly'

# make sure triton 2.1+ is installed

import packaging.version as pkg_version

import importlib
from importlib.metadata import version

try:
    triton_version = version('triton')
except:
    print(f'latest triton must be installed. `{INSTALL_COMMAND}` first')
    exit()

assert pkg_version.parse(triton_version) >= pkg_version.parse('3.0.0'), f'triton must be version 3.0.0 or above. `{INSTALL_COMMAND}` to upgrade'

import triton
import triton.language as tl
from triton.language.extra import libdevice
```

**描述:** 此代码块检查 Triton 是否已安装以及版本是否满足要求 (>=3.0.0)。 如果 Triton 未安装或版本过旧，它将打印安装命令并退出。 此外，它还导入了必要的 Triton 库。`TRITON_BLOCK_SIZE`定义了一个全局变量，用于指定 Triton 编译时候的块大小。

**如何使用:**  这段代码在脚本开始时执行，以确保 Triton 环境正确设置。 这对于使用 Triton 编写的自定义 CUDA 内核至关重要。
**3. Triton 内核: `reduce_avg`:**

```python
@triton.jit
def reduce_avg(x, y):
    return (x + y) / 2
```

**描述:** 这是一个简单的 Triton JIT 函数，用于计算两个数值的平均值。 它使用 `@triton.jit` 装饰器进行装饰，这会将 Python 函数编译为优化的 Triton 内核。

**如何使用:**  该函数在 attention 机制中用于平均来自不同 head 的信息。

**4. Triton 内核: `forward_kernel_causal_and_sparse`:**

```python
@triton.jit
def forward_kernel_causal_and_sparse(
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr
):
    # ... (kernel implementation) ...
```

**描述:** 这是前向传播的关键 Triton 内核。 它实现了因果和稀疏 attention 机制。 该内核使用 Triton 编程模型进行优化，以实现高性能。 它接收查询 (Q)、键 (K)、值 (V) 张量，以及用于指定稀疏 attention 模式的附加参数（`kv_block_indices`, `kv_block_mask`）。它还计算 log-sum-exp (`Lse`) 以进行数值稳定性。此kernel是基于块（Block）进行设计的，充分利用了硬件并行计算能力，从而加速计算过程。
  * `Q`, `K`, `V`: 查询，键，值张量
  * `kv_block_indices`: 选择的kv块的索引
  * `kv_block_mask`: 选择的kv块的掩码
  * `Out`: 输出张量
  * `Lse`: log-sum-exp
  * `softmax_scale`: softmax的缩放因子
  * `stride_*`: 各个张量的步长
  * `seqlen_q`, `seqlen_k`: 查询和键的序列长度
  * `headdim`: attention头的维度
  * `BLOCK_*`: 编译时的常量，指定块的大小
  * `NUM_SEL_KV_BLOCKS`: 选择的kv块的数量
  * `INCLUDE_BLOCK_CAUSAL`: 是否包含因果关系
  * `SLIDING`: 是否使用滑动窗口

**如何使用:** 此内核由 `native_sparse_attn_forward` 函数启动。 它在 CUDA 设备上并行计算 attention 输出。

**5. Triton 内核: `forward_kernel`:**

```python
@triton.heuristics(
    dict(
        EVEN_M = lambda args: divisible_by(args["seqlen_q"], args["BLOCK"]),
        EVEN_N = lambda args: divisible_by(args["seqlen_k"], args["BLOCK"]),
        EVEN_HEADDIM = lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        QUERY_EXPAND_DIM = lambda args: 16 // args['QUERY_HEAD_GROUPS']
    )
)
@triton.jit
def forward_kernel(
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    Out,
    SlidingOut,
    Lse,
    SlidingLse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    RETURN_SLIDING_OUT: tl.constexpr
):
    if RETURN_SLIDING_OUT:
        sliding = tl.program_id(2) == 0
        out_ptr = SlidingOut if sliding else Out
        lse_ptr = SlidingLse if sliding else Lse
        num_sel_kv_blocks = 0 if sliding else NUM_SEL_KV_BLOCKS
    else:
        sliding = False
        out_ptr = Out
        lse_ptr = Lse
        num_sel_kv_blocks = NUM_SEL_KV_BLOCKS

    forward_kernel_causal_and_sparse(
        Q,
        K,
        V,
        kv_block_indices,
        kv_block_mask,
        out_ptr,
        Lse,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_om,
        stride_kvbl_b,
        stride_kvbl_h,
        stride_kvbl_m,
        stride_lse_b,
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM,
        EVEN_M,
        EVEN_N,
        EVEN_HEADDIM,
        BLOCK,
        SEL_BLOCK,
        QUERY_HEAD_GROUPS,
        QUERY_EXPAND_DIM,
        num_sel_kv_blocks,
        INCLUDE_BLOCK_CAUSAL,
        sliding
    )
```

**描述:** 这个函数是对 `forward_kernel_causal_and_sparse` 的封装。它根据 `RETURN_SLIDING_OUT` 参数来选择使用哪个输出指针和 log-sum-exp 指针。如果 `RETURN_SLIDING_OUT` 为真，则会执行滑动窗口 attention。 `triton.heuristics` 根据输入参数，自动选择最优的block size，从而优化性能。

**如何使用:** 这是在前向传播中调用的主要入口点。它根据配置启动因果和稀疏 attention 内核。

**6. 前向传播函数: `native_sparse_attn_forward`:**

```python
def native_sparse_attn_forward(
    q,
    k,
    v,
    kv_block_indices,
    kv_block_mask,
    block_size = 128,
    include_block_causal = True,
    return_sliding_window_out = False
):
    q, k, v, kv_block_indices = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v, kv_block_indices)]

    batch, nheads, seqlen_q, dim, device = *q.shape, q.device
    _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(nheads, kv_heads)
    head_groups = nheads // kv_heads

    assert divisible_by(block_size, 16)

    num_blocks_per_sel = block_size // 16
    if num_blocks_per_sel > 1:
        kv_block_indices = einx.add('... sel, r -> ... (sel r)', kv_block_indices * num_blocks_per_sel, arange(num_blocks_per_sel, device = device))
        kv_block_mask = repeat(kv_block_mask, '... sel -> ... (sel r)', r = num_blocks_per_sel)

    num_selected_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    assert k.shape == (batch, kv_heads, seqlen_k, dim)
    assert v.shape == (batch, kv_heads, seqlen_k, dim)
    assert dim <= 128, "only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert all([t.is_cuda for t in (q, k, v)])

    softmax_scale = dim ** -0.5

    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    lse = torch.empty((batch, nheads, seqlen_q_rounded), device = device, dtype = torch.float32)
    sliding_lse = torch.empty((batch, nheads, seqlen_q_rounded), device = device, dtype = torch.float32)

    o = torch.empty_like(q)
    slide_o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    num_warps = 4 if dim <= 64 else 8

    grid = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK"]),
        batch * kv_heads,
        (2 if return_sliding_window_out else 1)
    ) # kv heads here, as grouped query heads all loaded, following the paper

    forward_kernel[grid](
        q,
        k,
        v,
        kv_block_indices,
        kv_block_mask,
        o,
        slide_o,
        lse,
        sliding_lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_block_indices.stride(0),
        kv_block_indices.stride(1),
        kv_block_indices.stride(2),
        lse.stride(0),
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        dim,
        seqlen_q // 32,
        seqlen_k // 32,
        BLOCK_HEADDIM,
        BLOCK = 16,
        SEL_BLOCK = block_size,
        QUERY_HEAD_GROUPS = head_groups,
        NUM_SEL_KV_BLOCKS = num_selected_fine_blocks,
        INCLUDE_BLOCK_CAUSAL = include_block_causal,
        RETURN_SLIDING_OUT = return_sliding_window_out,
        num_warps = num_warps,
        num_stages = 1,
    )

    return o, slide_o, lse
```

**描述:** 此函数是前向传播的 Python 入口点。它执行以下操作：
  *   检查输入张量的连续性，如不连续则进行contiguous操作。
  *   计算 softmax 缩放因子。
  *   根据配置，使用 `forward_kernel` 启动 Triton 内核。
  *   分配输出张量 (`o`) 和 `log-sum-exp` 张量 (`lse`)。
  *   计算 attention 输出。

**如何使用:** 你可以使用此函数来计算给定查询、键和值张量的稀疏 attention。

**7. 反向传播预处理内核: `backward_preprocess_do_o_dot`:**

```python
@triton.jit
def backward_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    qheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    # ... (kernel implementation) ...
```

**描述:** 此 Triton 内核用于反向传播的预处理步骤。它计算输出和输出梯度 (`Out`, `DO`) 的点积，并将结果存储在 `Delta` 张量中。

**如何使用:** 此内核由 `native_sparse_attn_backward` 函数启动。 它为反向传播计算必要的梯度信息。

**8. 反向传播存储内核: `backward_store_dk_dv`:**

```python
@triton.jit
def backward_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # ... (kernel implementation) ...
```

**描述:** 该内核负责原子地将计算出的 `dk` 和 `dv` 值写回到内存中的相应位置。原子操作对于避免并发写入引起的竞争条件是必要的。

**如何使用:** 它会被反向传播的kernel调用，以存储计算出的梯度。
**9. 反向传播单列块内核 (稀疏): `backward_kernel_one_col_block_sparse`:**

```python
@triton.jit
def backward_kernel_one_col_block_sparse(
    start_n,
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_kvbl_m,
    stride_qh,
    stride_doh,
    stride_dqh,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    RETURN_SEL_GRADS: tl.constexpr,
    OFF_SEL_KV_BLOCKS: tl.constexpr,
    BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr,
):
    # ... (kernel implementation) ...
```

**描述:** 这是反向传播的关键 Triton 内核之一，专门用于稀疏注意力机制。它计算查询梯度 (`DQ`)、键梯度 (`DK`) 和值梯度 (`DV`)。 此内核处理 `kv_block_indices` 和 `kv_block_mask` 以进行稀疏 attention。

**如何使用:** 该内核由 `backward_kernel` 函数启动，用于计算稀疏 attention 的梯度。

**10. 反向传播单列块内核 (因果): `backward_kernel_one_col_block_causal`:**

```python
@triton.jit
def backward_kernel_one_col_block_causal(
    start_n,
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_kvbl_m,
    stride_qh,
    stride_doh,
    stride_dqh,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    SLIDING: tl.constexpr
):
    # ... (kernel implementation) ...
```

**描述:** 这是另一个反向传播的关键 Triton 内核，专用于因果 attention 机制。 它与 `backward_kernel_one_col_block_sparse` 类似，但它还考虑了因果关系。

**如何使用:**  此内核由 `backward_kernel` 函数启动，用于计算因果 attention 的梯度。

**11. 反向传播内核: `backward_kernel`:**

```python
@triton.heuristics(
    dict(
        QUERY_EXPAND_DIM = lambda args: 16 // args['QUERY_HEAD_GROUPS']
    )
)
@triton.jit
def backward_kernel(
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    stride_D_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    RETURN_SEL_GRADS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads
    off_qh = off_h * QUERY_HEAD_GROUPS

    OFF_SEL_KV_BLOCKS = tl.program_id(0) - int(INCLUDE_BLOCK_CAUSAL)
    IS_CAUSAL = INCLUDE_BLOCK_CAUSAL and tl.program_id(0) == 0

    # offset pointers for batch/head

    Q += off_b * stride_qb + off_qh * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_qh * stride_doh
    DQ += off_b * stride_dqb + off_qh * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh

    # offset pointers for batch/head for selected kv block related

    kv_block_indices += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_mask += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_grads += off_b * stride_kvbl_b + off_h * stride_kvbl_h

    # pointer to row-wise quantities in value-like data

    D += (
        off_b * stride_D_b +
        off_qh * seqlen_q_rounded
    )

    LSE += (
        off_b * stride_lse_b +
        off_qh * seqlen_q_rounded
    )

    num_block_n = tl.cdiv(seqlen_k, BLOCK)

    if IS_CAUSAL:
        for start_n in range(0, num_block_n):
            backward_kernel_one_col_block_causal(
                start_n,
                Q,
                K,
                V,
                kv_block_indices,
                kv_block_mask,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                stride_kvbl_m,
                stride_qh,
                stride_doh,
                stride_dqh,
                seqlen_q,
                seqlen_k,
                seqlen_q_rounded,
                headdim,
                BLOCK_HEADDIM = BLOCK_HEADDIM,
                EVEN_M = EVEN_M,
                EVEN_N = EVEN_N,
                EVEN_HEADDIM = EVEN_HEADDIM,
                BLOCK = BLOCK,
                SEL_BLOCK = SEL_BLOCK,
                QUERY_HEAD_GROUPS = QUERY_HEAD_GROUPS,
                QUERY_EXPAND_DIM = QUERY_EXPAND_DIM,
                SLIDING = SLIDING
            )
    else:
        for start_n in range(0, num_block_n):
            backward_kernel_one_col_block_sparse(
                start_n,
                Q,
                K,
                V,
                kv_block_indices,
                kv_block_mask,
                kv_block_grads,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                stride_kvbl_m,
                stride_qh,
                stride_doh,
                stride_dqh,
                seqlen_q,
                seqlen_k,
                seqlen_q_rounded,
                headdim,
                BLOCK_HEADDIM = BLOCK_HEADDIM,
                EVEN_M = EVEN_M,
                EVEN_N = EVEN_N,
                EVEN_HEADDIM = EVEN_HEADDIM,
                BLOCK = BLOCK,
                QUERY_HEAD_GROUPS = QUERY_HEAD_GROUPS,
                QUERY_EXPAND_DIM = QUERY_EXPAND_DIM,
                RETURN_SEL_GRADS = RETURN_SEL_GRADS,
                OFF_SEL_KV_BLOCKS = OFF_SEL_KV_BLOCKS,
                BLOCK_DV_USE_DOT = BLOCK_DV_USE_DOT,
                BLOCK_DK_USE_DOT = BLOCK_DK_USE_DOT,
            )
```

**描述:** 此函数是反向传播的主要入口点。 它根据配置选择使用 `backward_kernel_one_col_block_causal` 或 `backward_kernel_one_col_block_sparse` 内核。

**如何使用:**  你可以使用此函数来计算给定输出梯度、查询、键和值张量的稀疏 attention 的梯度。

**12. 反向传播函数: `native_sparse_attn_backward`:**

```python
def native_sparse_attn_backward(
    do,
    q, k, v,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    o,
    lse,
    dq, dk, dv,
    block_size = 128,
    include_block_causal = True,
    return_sel_grads = False,
    sliding = False,
    block_dk_dv_use_dot = None
):
    device = do.device

    # Make sure that the last dimension is contiguous
    if not is_contiguous(do):
        do = do.contiguous()

    batch, q_heads, seqlen_q, dim = q.shape

    _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(q_heads, kv_heads)
    head_groups = q_heads // kv_heads
    assert divisible_by(16, head_groups)

    assert divisible_by(block_size, 16)

    num_blocks_per_sel = block_size // 16

    orig_kv_block_grads = kv_block_grads

    if num_blocks_per_sel > 1:
        kv_block_indices = einx.add('... sel, r -> ... (sel r)', kv_block_indices * num_blocks_per_sel, arange(num_blocks_per_sel, device = device))
        kv_block_mask = repeat(kv_block_mask, '... sel -> ... (sel r)', r = num_blocks_per_sel)
        kv_block_grads = repeat(kv_block_grads, '... sel -> ... (sel r)', r = num_blocks_per_sel)

    num_sel_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    # assert d in {16, 32, 64, 128}
    assert dim <= 128
    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    assert lse.shape == (batch, q_heads, seqlen_q_rounded)
    assert all([is_contiguous(t) for t in (q, k, v, o, dq, dk, dv)])

    softmax_scale = dim ** -0.5

    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)

    delta = torch.empty_like(lse)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK"]), batch * q_heads)

    backward_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        q_heads,
        seqlen_q,
        seqlen_q_rounded,
        dim,
        BLOCK = block_size,
        BLOCK_HEADDIM = BLOCK_HEADDIM,
    )

    grid = lambda META: (
        num_sel_fine_blocks + int(include_block_causal),
        batch * kv_heads
    )

    backward_kernel[grid](
        q,
        k,
        v,
        kv_block_indices,
        kv_block_mask,
        kv_block_grads,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        kv_block_indices.stride(0),
        