Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_dma_ex.h`

好的，重新开始。 这次我会更详细地解释代码，提供更多示例，并用中文进行注释。

**文件： stm32f1xx_hal_dma_ex.h**

此头文件是 STM32F1 系列微控制器的 HAL (硬件抽象层) 库的一部分。它定义了 DMA (直接存储器访问) 扩展模块相关的宏和函数。 DMA 允许外设和存储器之间的数据传输，无需 CPU 的干预，从而提高了效率。`_ex` 表示这是对标准 DMA 功能的扩展，提供了一些额外的便利宏。

**1. 头文件保护:**

```c
#ifndef __STM32F1xx_HAL_DMA_EX_H
#define __STM32F1xx_HAL_DMA_EX_H
```

*   **功能:** 这是标准的头文件保护机制。它确保头文件只被包含一次，防止重复定义错误。
*   **解释:**
    *   `#ifndef __STM32F1xx_HAL_DMA_EX_H`:  如果 `__STM32F1xx_HAL_DMA_EX_H` 宏未定义，则执行下面的代码。
    *   `#define __STM32F1xx_HAL_DMA_EX_H`:  定义 `__STM32F1xx_HAL_DMA_EX_H` 宏。  这会在后续的包含中阻止头文件内容被再次包含。

**2. C++ 兼容性:**

```c
#ifdef __cplusplus
 extern "C" {
#endif
```

*   **功能:** 允许在 C++ 代码中使用此 C 头文件。
*   **解释:**
    *   `#ifdef __cplusplus`:  如果定义了 `__cplusplus` 宏（表示 C++ 编译器），则执行下面的代码。
    *   `extern "C" {`:  告诉 C++ 编译器使用 C 的链接规则。 这对于确保 C++ 代码可以正确地链接到 C 代码是必要的，因为 C++ 有名称修饰 (name mangling) 的特性，而 C 没有。

**3. 包含 HAL 定义:**

```c
#include "stm32f1xx_hal_def.h"
```

*   **功能:** 包含 HAL 库的基本定义，如数据类型和通用宏。
*   **解释:** `stm32f1xx_hal_def.h`  包含了 HAL 库中使用的基本数据类型定义、通用宏定义以及其他必要的预定义。

**4. HAL 驱动组和 DMAEx 组:**

```c
/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @defgroup DMAEx DMAEx
  * @{
  */
```

*   **功能:** 定义 HAL 驱动和 DMA 扩展模块的组，方便组织文档。
*   **解释:** 这两个注释用于生成 HAL 库的文档。`@addtogroup`  将当前代码添加到 `STM32F1xx_HAL_Driver`  组中，`@defgroup`  定义一个名为 `DMAEx`  的组。

**5. 导出类型和常量:**

```c
/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
```

*   **功能:** 此处原本应该定义导出的数据类型和常量，但该文件中没有实际定义。通常，DMA 相关的结构体（例如 DMA 初始化结构体）和一些预定义的常量会在这里声明。
*   **解释:** 这部分留空，表明此文件主要关注宏定义，而不是数据类型或常量。

**6. DMAEx 导出的宏:**

```c
/* Exported macro ------------------------------------------------------------*/
/** @defgroup DMAEx_Exported_Macros DMA Extended Exported Macros
  * @{
  */
/* Interrupt & Flag management */
```

*   **功能:**  定义与 DMA 扩展模块相关的宏，主要用于中断和标志管理。
*   **解释:** 这部分开始定义一些方便使用的宏，用于操作 DMA 通道的状态标志和中断。

**7. 设备区分 (高密度/中低密度):**

```c
#if defined (STM32F100xE) || defined (STM32F101xE) || defined (STM32F101xG) || defined (STM32F103xE) || \
    defined (STM32F103xG) || defined (STM32F105xC) || defined (STM32F107xC)
/** @defgroup DMAEx_High_density_XL_density_Product_devices DMAEx High density and XL density product devices
  * @{
  */

// ... (高密度/超高密度设备的宏定义)

#else
/** @defgroup DMA_Low_density_Medium_density_Product_devices DMA Low density and Medium density product devices
  * @{
  */

// ... (低密度/中密度设备的宏定义)

#endif
```

*   **功能:**  根据 STM32F1 系列的不同型号（高密度/中低密度）提供不同的宏定义。这是因为不同型号的 STM32F1 在 DMA 寄存器和通道数量上可能存在差异。
*   **解释:**
    *   `#if defined (...) || defined (...) ...`:  根据预定义的宏（例如 `STM32F100xE`）来选择不同的代码块。这些宏通常在编译时由 STM32 的库定义。
    *   高密度和超高密度型号通常具有更多的 DMA 通道（例如，DMA2），因此需要不同的宏来处理这些额外的通道。
    *   中低密度型号通常只有 DMA1。

**8. 获取传输完成标志索引宏 (__HAL_DMA_GET_TC_FLAG_INDEX):**

```c
#define __HAL_DMA_GET_TC_FLAG_INDEX(__HANDLE__) \
(((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel1))? DMA_FLAG_TC1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel2))? DMA_FLAG_TC2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel3))? DMA_FLAG_TC3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel4))? DMA_FLAG_TC4 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel5))? DMA_FLAG_TC5 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel6))? DMA_FLAG_TC6 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel7))? DMA_FLAG_TC7 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel1))? DMA_FLAG_TC1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel2))? DMA_FLAG_TC2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel3))? DMA_FLAG_TC3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel4))? DMA_FLAG_TC4 :\
   DMA_FLAG_TC5)
```

*   **功能:**  根据 DMA 句柄 (`__HANDLE__`) 确定 DMA 通道对应的传输完成 (TC) 标志。
*   **解释:**
    *   `__HANDLE__`:  这是一个指向 DMA 句柄结构体的指针，包含了 DMA 通道的配置信息。
    *   `(__HANDLE__)->Instance`:  指向 DMA 通道寄存器组的指针 (例如 `DMA1_Channel1`, `DMA2_Channel3`)。
    *   `DMA_FLAG_TCx`:  定义在 HAL 库中的宏，表示 DMA 通道 `x` 的传输完成标志 (例如 `DMA_FLAG_TC1`, `DMA_FLAG_TC2`)。
    *   这个宏通过一系列的条件判断，将 DMA 句柄与特定的 DMA 通道相关联，并返回对应的传输完成标志。
*   **如何使用:**  在检查 DMA 传输是否完成时，可以使用这个宏来获取正确的标志位。例如:

    ```c
    DMA_HandleTypeDef hdma_spi1_rx; // DMA句柄

    // 初始化 DMA ...

    if (__HAL_DMA_GET_TC_FLAG_INDEX(&hdma_spi1_rx) == DMA_FLAG_TC2) {
        // DMA 通道 2 传输完成
        // ...
    }
    ```

**9. 获取半传输完成标志索引宏 (__HAL_DMA_GET_HT_FLAG_INDEX):**

```c
#define __HAL_DMA_GET_HT_FLAG_INDEX(__HANDLE__)\
(((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel1))? DMA_FLAG_HT1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel2))? DMA_FLAG_HT2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel3))? DMA_FLAG_HT3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel4))? DMA_FLAG_HT4 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel5))? DMA_FLAG_HT5 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel6))? DMA_FLAG_HT6 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel7))? DMA_FLAG_HT7 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel1))? DMA_FLAG_HT1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel2))? DMA_FLAG_HT2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel3))? DMA_FLAG_HT3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel4))? DMA_FLAG_HT4 :\
   DMA_FLAG_HT5)
```

*   **功能:** 根据 DMA 句柄确定 DMA 通道对应的半传输完成 (HT) 标志。
*   **解释:**  与 `__HAL_DMA_GET_TC_FLAG_INDEX` 类似，但针对的是半传输完成标志。  半传输完成标志在 DMA 循环模式中非常有用，允许在 DMA 传输一半数据后触发中断。
*   **如何使用:**

    ```c
    DMA_HandleTypeDef hdma_usart1_tx;

    // 初始化 DMA ...

    if (__HAL_DMA_GET_HT_FLAG_INDEX(&hdma_usart1_tx) == DMA_FLAG_HT1) {
        // DMA 通道 1 半传输完成
        // ...
    }
    ```

**10. 获取传输错误标志索引宏 (__HAL_DMA_GET_TE_FLAG_INDEX):**

```c
#define __HAL_DMA_GET_TE_FLAG_INDEX(__HANDLE__)\
(((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel1))? DMA_FLAG_TE1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel2))? DMA_FLAG_TE2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel3))? DMA_FLAG_TE3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel4))? DMA_FLAG_TE4 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel5))? DMA_FLAG_TE5 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel6))? DMA_FLAG_TE6 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel7))? DMA_FLAG_TE7 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel1))? DMA_FLAG_TE1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel2))? DMA_FLAG_TE2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel3))? DMA_FLAG_TE3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel4))? DMA_FLAG_TE4 :\
   DMA_FLAG_TE5)
```

*   **功能:**  根据 DMA 句柄确定 DMA 通道对应的传输错误 (TE) 标志。
*   **解释:**  用于检查 DMA 传输是否发生错误。 常见的 DMA 错误包括访问无效的存储器地址或外设错误。
*   **如何使用:**

    ```c
    DMA_HandleTypeDef hdma_adc1;

    // 初始化 DMA ...

    if (__HAL_DMA_GET_TE_FLAG_INDEX(&hdma_adc1) == DMA_FLAG_TE1) {
        // DMA 通道 1 传输错误
        // 处理错误 ...
    }
    ```

**11. 获取全局中断标志索引宏 (__HAL_DMA_GET_GI_FLAG_INDEX):**

```c
#define __HAL_DMA_GET_GI_FLAG_INDEX(__HANDLE__)\
(((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel1))? DMA_FLAG_GL1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel2))? DMA_FLAG_GL2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel3))? DMA_FLAG_GL3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel4))? DMA_FLAG_GL4 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel5))? DMA_FLAG_GL5 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel6))? DMA_FLAG_GL6 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA1_Channel7))? DMA_FLAG_GL7 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel1))? DMA_FLAG_GL1 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel2))? DMA_FLAG_GL2 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel3))? DMA_FLAG_GL3 :\
 ((uint32_t)((__HANDLE__)->Instance) == ((uint32_t)DMA2_Channel4))? DMA_FLAG_GL4 :\
   DMA_FLAG_GL5)
```

*   **功能:** 根据 DMA 句柄确定 DMA 通道对应的全局中断 (GI) 标志。
*   **解释:**  全局中断标志是 DMA 通道所有中断标志的总和。  如果设置了传输完成、半传输完成或传输错误中的任何一个标志，则全局中断标志也会被设置。
*   **如何使用:**

    ```c
    DMA_HandleTypeDef hdma_uart2_rx;

    // 初始化 DMA ...

    if (__HAL_DMA_GET_GI_FLAG_INDEX(&hdma_uart2_rx) == DMA_FLAG_GL1) {
        // DMA 通道 1 发生了中断 (传输完成, 半传输完成或错误)
        // ...
    }
    ```

**12. 获取标志宏 (__HAL_DMA_GET_FLAG):**

```c
#define __HAL_DMA_GET_FLAG(__HANDLE__, __FLAG__)\
(((uint32_t)((__HANDLE__)->Instance) > (uint32_t)DMA1_Channel7)? (DMA2->ISR & (__FLAG__)) :\
  (DMA1->ISR & (__FLAG__)))
```

*   **功能:**  读取指定的 DMA 通道标志。
*   **解释:**
    *   `__HANDLE__`: DMA 句柄
    *   `__FLAG__`: 要读取的标志 (例如 `DMA_FLAG_TC1`, `DMA_FLAG_HT2`, `DMA_FLAG_TE3`)。
    *   `DMA1->ISR`: DMA1 的中断状态寄存器。
    *   `DMA2->ISR`: DMA2 的中断状态寄存器。
    *   此宏根据 DMA 句柄判断使用哪个 DMA 控制器 (DMA1 或 DMA2)，然后从对应的中断状态寄存器中读取指定的标志位。
*   **如何使用:**

    ```c
    DMA_HandleTypeDef hdma_spi2_tx;

    // 初始化 DMA ...

    if (__HAL_DMA_GET_FLAG(&hdma_spi2_tx, DMA_FLAG_TC2)) {
        // DMA 通道 2 传输完成
        // ...
    }
    ```

**13. 清除标志宏 (__HAL_DMA_CLEAR_FLAG):**

```c
#define __HAL_DMA_CLEAR_FLAG(__HANDLE__, __FLAG__) \
(((uint32_t)((__HANDLE__)->Instance) > (uint32_t)DMA1_Channel7)? (DMA2->IFCR = (__FLAG__)) :\
  (DMA1->IFCR = (__FLAG__)))
```

*   **功能:**  清除指定的 DMA 通道标志。
*   **解释:**
    *   `__HANDLE__`: DMA 句柄
    *   `__FLAG__`: 要清除的标志 (例如 `DMA_FLAG_TC1`, `DMA_FLAG_HT2`, `DMA_FLAG_TE3`)。
    *   `DMA1->IFCR`: DMA1 的中断标志清除寄存器。
    *   `DMA2->IFCR`: DMA2 的中断标志清除寄存器。
    *   此宏根据 DMA 句柄判断使用哪个 DMA 控制器 (DMA1 或 DMA2)，然后将指定的标志位写入对应的中断标志清除寄存器，从而清除标志。
*   **如何使用:**

    ```c
    DMA_HandleTypeDef hdma_uart3_rx;

    // 初始化 DMA ...

    // 清除 DMA 通道 3 的传输完成标志
    __HAL_DMA_CLEAR_FLAG(&hdma_uart3_rx, DMA_FLAG_TC3);
    ```

**9.  其他设备型号的宏定义 (中低密度):**

`#else` 分支中的宏定义与高密度设备类似，只是它们只针对 DMA1 的通道（因为中低密度设备通常只有一个 DMA 控制器）。它们的用法与高密度设备的宏相同。

**10.  文件结尾:**

```c
/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif /* STM32F100xE || STM32F101xE || STM32F101xG || STM32F103xE || */
       /* STM32F103xG || STM32F105xC || STM32F107xC */

#endif /* __STM32F1xx_HAL_DMA_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
```

*   **功能:**  关闭 C++ 兼容性块，结束条件编译，并添加版权声明。

**总结:**

`stm32f1xx_hal_dma_ex.h`  文件提供了一组宏，用于方便地访问和操作 STM32F1 系列微控制器的 DMA 通道标志。 这些宏根据设备型号自动选择正确的寄存器和标志位，从而简化了 DMA 编程。 这些宏极大地简化了使用 DMA 的过程，允许开发人员以更简洁和可读的方式处理 DMA 中断和状态标志。 它们是 HAL 库 DMA 驱动的重要组成部分。 通过理解这些宏，可以更好地掌握 STM32F1 系列微控制器的 DMA 功能。
