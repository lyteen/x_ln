Lan: `py` From`dl/DeepSeek-V3\inference\kernel.py`

**1. 激活量化 (Activation Quantization):**

```python
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    使用 Triton 量化激活值的 CUDA Kernel 函数。

    这个 Kernel 函数将输入张量 `x_ptr` 量化，并将量化后的值存储在 `y_ptr` 中，
    将缩放因子（scaling factor）存储在 `s_ptr` 中。

    参数：
        x_ptr (triton.Pointer): 指向输入张量的指针。
        y_ptr (triton.Pointer): 指向输出张量的指针，存储量化后的值。
        s_ptr (triton.Pointer): 指向输出张量的指针，存储缩放因子。
        BLOCK_SIZE (tl.constexpr): 每个 Triton 程序实例处理的块大小。
    """
    pid = tl.program_id(axis=0)  # 获取当前程序的 ID
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算当前程序处理的数据偏移量
    x = tl.load(x_ptr + offs).to(tl.float32)  # 从全局内存加载数据，并转换为 float32
    s = tl.max(tl.abs(x)) / 448.  # 计算缩放因子，使用最大绝对值
    y = x / s  # 进行量化
    y = y.to(y_ptr.dtype.element_ty)  # 将量化后的值转换为目标数据类型
    tl.store(y_ptr + offs, y)  # 将量化后的值存储到全局内存
    tl.store(s_ptr + pid, s)  # 将缩放因子存储到全局内存


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用分块量化的方式量化输入张量 `x`。

    参数：
        x (torch.Tensor): 要量化的输入张量。必须是连续的，并且其最后一个维度的大小必须能被 `block_size` 整除。
        block_size (int, optional): 用于量化的块大小。默认为 128。

    返回值：
        Tuple[torch.Tensor, torch.Tensor]: 一个包含以下元素的元组：
            - 量化后的张量，数据类型为 `torch.float8_e4m3fn`。
            - 缩放因子张量，数据类型为 `torch.float32`。
    """
    assert x.is_contiguous(), '输入张量必须是连续的'
    assert x.size(-1) % block_size == 0, f'最后一个维度的大小必须能被 block_size 整除 (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)  # 创建一个与输入张量形状相同的空张量，用于存储量化后的值
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)  # 创建一个用于存储缩放因子的张量
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )  # 定义 Triton Kernel 函数的网格大小
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)  # 启动 Triton Kernel 函数
    return y, s  # 返回量化后的张量和缩放因子


# Demo 使用示例
if __name__ == '__main__':
    x = torch.randn(1, 128, 256, requires_grad=False, device='cuda')  # 创建一个随机张量
    y, s = act_quant(x)  # 进行量化
    print(f"原始张量形状: {x.shape}, 数据类型: {x.dtype}")
    print(f"量化后张量形状: {y.shape}, 数据类型: {y.dtype}")
    print(f"缩放因子形状: {s.shape}, 数据类型: {s.dtype}")

```

**描述:**

*   `act_quant_kernel`:  Triton Kernel 函数，负责实际的量化操作。它计算每个块的最大绝对值，然后用这个值作为缩放因子来量化数据。
*   `act_quant`:  Python 函数，负责调用 Triton Kernel 函数。它检查输入张量的形状和连续性，创建输出张量，然后启动 Kernel 函数。
*   **中文解释:**  代码中添加了详细的中文注释，解释了每个步骤的作用和参数的含义。
*   **示例:**  提供了一个简单的示例，演示了如何使用 `act_quant` 函数。

**2. 权重解量化 (Weight Dequantization):**

```python
import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    使用 Triton 解量化权重的 CUDA Kernel 函数。

    这个 Kernel 函数使用提供的缩放因子解量化权重，并将结果存储在 `y_ptr` 中。

    参数：
        x_ptr (tl.pointer): 指向量化权重的指针。
        s_ptr (tl.pointer): 指向缩放因子的指针。
        y_ptr (tl.pointer): 指向输出缓冲区的指针，用于存储解量化后的权重。
        M (int): 权重矩阵的行数。
        N (int): 权重矩阵的列数。
        BLOCK_SIZE (tl.constexpr): 用于分块处理的块大小。
    """
    pid_m = tl.program_id(axis=0)  # 获取行方向的程序 ID
    pid_n = tl.program_id(axis=1)  # 获取列方向的程序 ID
    n = tl.cdiv(N, BLOCK_SIZE)  # 计算列方向的块数
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算行方向的偏移量
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算列方向的偏移量
    offs = offs_m[:, None] * N + offs_n[None, :]  # 计算全局偏移量
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)  # 创建一个掩码，用于处理边界情况
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)  # 从全局内存加载量化后的权重，并转换为 float32
    s = tl.load(s_ptr + pid_m * n + pid_n)  # 从全局内存加载缩放因子
    y = x * s  # 进行解量化
    tl.store(y_ptr + offs, y, mask=mask)  # 将解量化后的权重存储到全局内存


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    使用提供的缩放因子解量化给定的权重张量。

    参数：
        x (torch.Tensor): 量化后的权重张量，形状为 (M, N)。
        s (torch.Tensor): 缩放因子张量，形状为 (M, N)。
        block_size (int, optional): 用于解量化的块大小。默认为 128。

    返回值：
        torch.Tensor: 解量化后的权重张量，形状与 `x` 相同。

    抛出：
        AssertionError: 如果 `x` 或 `s` 不是连续的，或者它们的维度不是 2。
    """
    assert x.is_contiguous() and s.is_contiguous(), '输入张量必须是连续的'
    assert x.dim() == 2 and s.dim() == 2, '输入张量必须有 2 个维度'
    M, N = x.size()  # 获取权重矩阵的形状
    y = torch.empty_like(x, dtype=torch.get_default_dtype())  # 创建一个与输入张量形状相同的空张量，用于存储解量化后的权重
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))  # 定义 Triton Kernel 函数的网格大小
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)  # 启动 Triton Kernel 函数
    return y  # 返回解量化后的权重张量


# Demo 使用示例
if __name__ == '__main__':
    x = torch.randn(64, 128, dtype=torch.float8_e4m3fn, device='cuda')  # 创建一个随机的量化权重张量
    s = torch.randn(64, 128, device='cuda')  # 创建一个随机的缩放因子张量
    y = weight_dequant(x, s)  # 进行解量化
    print(f"量化后权重张量形状: {x.shape}, 数据类型: {x.dtype}")
    print(f"缩放因子张量形状: {s.shape}, 数据类型: {s.dtype}")
    print(f"解量化后权重张量形状: {y.shape}, 数据类型: {y.dtype}")
```

**描述:**

*   `weight_dequant_kernel`:  Triton Kernel 函数，负责实际的解量化操作。它使用提供的缩放因子来解量化权重。
*   `weight_dequant`:  Python 函数，负责调用 Triton Kernel 函数。它检查输入张量的形状和连续性，创建输出张量，然后启动 Kernel 函数。
*   **中文解释:**  代码中添加了详细的中文注释，解释了每个步骤的作用和参数的含义。
*   **示例:**  提供了一个简单的示例，演示了如何使用 `weight_dequant` 函数。

**3. FP8 GEMM (通用矩阵乘法):**

```python
import torch
import triton
import triton.language as tl
from triton import Config


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    使用 Triton 在 FP8 矩阵上执行矩阵乘法操作的 CUDA Kernel 函数。

    参数：
        a_ptr (tl.tensor): 指向第一个输入矩阵 A 的指针。
        b_ptr (tl.tensor): 指向第二个输入矩阵 B 的指针。
        c_ptr (tl.tensor): 指向输出矩阵 C 的指针。
        a_s_ptr (tl.tensor): 指向矩阵 A 的缩放因子的指针。
        b_s_ptr (tl.tensor): 指向矩阵 B 的缩放因子的指针。
        M (int): 矩阵 A 和 C 的行数。
        N (tl.constexpr): 矩阵 B 和 C 的列数。
        K (tl.constexpr): 矩阵 A 的列数和矩阵 B 的行数。
        BLOCK_SIZE_M (tl.constexpr): M 维度的块大小。
        BLOCK_SIZE_N (tl.constexpr): N 维度的块大小。
        BLOCK_SIZE_K (tl.constexpr): K 维度的块大小。
    """
    pid_m = tl.program_id(axis=0)  # 获取行方向的程序 ID
    pid_n = tl.program_id(axis=1)  # 获取列方向的程序 ID
    k = tl.cdiv(K, BLOCK_SIZE_K)  # 计算 K 维度的块数
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # 计算 M 维度的偏移量
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # 计算 N 维度的偏移量
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # 计算 K 维度的偏移量
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]  # 计算矩阵 A 的指针
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]  # 计算矩阵 B 的指针
    a_s_ptrs = a_s_ptr + offs_m * k  # 计算矩阵 A 的缩放因子的指针
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k  # 计算矩阵 B 的缩放因子的指针

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 初始化累加器
    for i in range(k):  # 循环遍历 K 维度的块
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)  # 从全局内存加载矩阵 A 的数据
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)  # 从全局内存加载矩阵 B 的数据
        a_s = tl.load(a_s_ptrs)  # 从全局内存加载矩阵 A 的缩放因子
        b_s = tl.load(b_s_ptrs)  # 从全局内存加载矩阵 B 的缩放因子
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]  # 执行矩阵乘法和缩放
        a_ptrs += BLOCK_SIZE_K  # 更新矩阵 A 的指针
        b_ptrs += BLOCK_SIZE_K  # 更新矩阵 B 的指针
        a_s_ptrs += 1  # 更新矩阵 A 的缩放因子的指针
        b_s_ptrs += 1  # 更新矩阵 B 的缩放因子的指针
    c = accumulator.to(c_ptr.dtype.element_ty)  # 将累加器转换为输出数据类型
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 计算 M 维度的偏移量
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 计算 N 维度的偏移量
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]  # 计算矩阵 C 的指针
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)  # 创建一个掩码，用于处理边界情况
    tl.store(c_ptrs, c, mask=mask)  # 将结果存储到全局内存


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    使用 FP8 精度执行矩阵乘法。

    参数：
        a (torch.Tensor): 第一个输入矩阵，必须是连续的。
        a_s (torch.Tensor): 第一个输入矩阵的缩放因子，必须是连续的。
        b (torch.Tensor): 第二个输入矩阵，必须是连续的。
        b_s (torch.Tensor): 第二个输入矩阵的缩放因子，必须是连续的。

    返回值：
        torch.Tensor: 矩阵乘法的结果。
    """
    assert a.is_contiguous() and b.is_contiguous(), '输入张量必须是连续的'
    assert a_s.is_contiguous() and b_s.is_contiguous(), '缩放因子张量必须是连续的'
    K = a.size(-1)  # 获取矩阵 A 的列数
    M = a.numel() // K  # 获取矩阵 A 的行数
    N = b.size(0)  # 获取矩阵 B 的行数
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())  # 创建一个用于存储结果的空张量
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))  # 定义 Triton Kernel 函数的网格大小
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)  # 启动 Triton Kernel 函数
    return c  # 返回结果

# Demo 使用示例
if __name__ == '__main__':
    M, K, N = 64, 128, 32  # 定义矩阵的形状
    a = torch.randn(M, K, dtype=torch.float8_e4m3fn, device='cuda')  # 创建一个随机的 FP8 矩阵 A
    a_s = torch.randn(M, K // 16, device='cuda')  # 创建一个随机的缩放因子张量 A
    b = torch.randn(N, K, dtype=torch.float8_e4m3fn, device='cuda')  # 创建一个随机的 FP8 矩阵 B
    b_s = torch.randn(N, K // 16, device='cuda')  # 创建一个随机的缩放因子张量 B
    c = fp8_gemm(a, a_s, b, b_s)  # 执行矩阵乘法
    print(f"矩阵 A 形状: {a.shape}, 数据类型: {a.dtype}")
    print(f"矩阵 B 形状: {b.shape}, 数据类型: {b.dtype}")
    print(f"矩阵 C 形状: {c.shape}, 数据类型: {c.dtype}")
```

**描述:**

*   `fp8_gemm_kernel`:  Triton Kernel 函数，负责实际的 FP8 矩阵乘法操作。它使用分块的方式计算矩阵乘法，并使用提供的缩放因子进行缩放。
*   `fp8_gemm`:  Python 函数，负责调用 Triton Kernel 函数。它检查输入张量的形状和连续性，创建输出张量，然后启动 Kernel 函数。
*   `fp8_gemm_configs`:  一个配置列表，用于 Triton 的自动调优功能。它定义了不同的块大小和执行阶段数，Triton 会自动选择最佳的配置。
*   `@triton.autotune`:  Triton 的自动调优装饰器，用于自动选择最佳的 Kernel 函数配置。
*   **中文解释:**  代码中添加了详细的中文注释，解释了每个步骤的作用和参数的含义。
*   **示例:**  提供了一个简单的示例，演示了如何使用 `fp8_gemm` 函数。

**总结:**

这三个代码段展示了如何使用 Triton 执行激活量化、权重解量化和 FP8 矩阵乘法。 这些代码都经过了改进，具有更好的可读性和易懂性，并附带了详细的中文解释和示例。 Triton 的自动调优功能可以帮助你选择最佳的 Kernel 函数配置，从而提高性能。 请注意，这些代码需要在支持 CUDA 的 GPU 上运行。
