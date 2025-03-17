Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\core_cm23.h`

好的，现在我将对 `core_cm23.h` 文件进行分析，并为你生成每个代码片段的解释，以及中文描述、使用方法和演示代码。

**总览:**

`core_cm23.h` 是 CMSIS (Cortex Microcontroller Software Interface Standard) 的一部分，专门用于 ARM Cortex-M23 处理器的头文件。它提供了一组定义和函数，方便开发者访问和控制 Cortex-M23 内核的各种外设和功能，例如中断控制器 (NVIC)、系统控制块 (SCB)、系统滴答定时器 (SysTick) 等。它还处理了一些编译器兼容性问题，并定义了用于安全扩展的结构体和函数（如果可用）。

**文件结构划分:**

1.  **文件头部和保护:** 包含了版权信息、许可证声明，以及用于避免重复包含头文件的宏定义 `__CORE_CM23_H_GENERIC`。
2.  **MISRA 异常:** 声明了为了代码效率和底层硬件访问而违反的一些 MISRA-C 规则。
3.  **CMSIS 定义:** 定义了 CMSIS 版本信息、Cortex-M 内核编号、是否使用 FPU 等宏。
4.  **IO 定义:** 定义了用于访问外设寄存器的 I/O 类型限定符 ( `__I`, `__O`, `__IO`, `__IM`, `__OM`, `__IOM` )。
5.  **寄存器抽象:** 定义了用于访问内核寄存器（如 APSR, IPSR, xPSR, CONTROL）、NVIC、SCB、SysTick、DWT、TPI、MPU、SAU 和 CoreDebug 的结构体类型。
6.  **硬件抽象层 (HAL):** 提供了用于操作内核外设的内联函数，例如 NVIC 控制、SysTick 配置、MPU 配置 (如果存在) 和 SAU 配置 (如果存在)。

**现在，我们将逐个部分进行详细解释和生成代码片段：**

**1. 文件头部和保护：**

```c
/**************************************************************************//**
 * @file     core_cm23.h
 * @brief    CMSIS Cortex-M23 Core Peripheral Access Layer Header File
 * @version  V5.0.7
 * @date     22. June 2018
 ******************************************************************************/
/*
 * Copyright (c) 2009-2018 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header   /* treat file as system include file */
#endif

#ifndef __CORE_CM23_H_GENERIC
#define __CORE_CM23_H_GENERIC
```

*   **描述:** 这段代码是头文件的标准开头，包含了文件说明、版权信息、许可证声明。`#ifndef __CORE_CM23_H_GENERIC` 和 `#define __CORE_CM23_H_GENERIC` 用于防止头文件被重复包含。`#pragma system_include` 和 `#pragma clang system_header` 是编译器指令，告诉编译器将此文件视为系统头文件，用于 MISRA 检查和其他优化。
*   **如何使用:** 无需手动修改或使用这段代码。它是头文件不可分割的一部分。

**2. MISRA 异常：**

```c
/**
  \page CMSIS_MISRA_Exceptions  MISRA-C:2004 Compliance Exceptions
  CMSIS violates the following MISRA-C:2004 rules:

   \li Required Rule 8.5, object/function definition in header file.<br>
     Function definitions in header files are used to allow 'inlining'.

   \li Required Rule 18.4, declaration of union type or object of union type: '{...}'.<br>
     Unions are used for effective representation of core registers.

   \li Advisory Rule 19.7, Function-like macro defined.<br>
     Function-like macros are used to allow more efficient code.
 */
```

*   **描述:** 这段代码注释说明了 CMSIS 为了效率和底层访问，违反了一些 MISRA-C 规则，包括：
    *   在头文件中定义对象/函数（为了内联）。
    *   使用联合体 (union) 来表示内核寄存器。
    *   使用函数式宏。
*   **如何使用:** 这段代码是信息性的，无需手动修改或使用。理解这些例外情况可以帮助你更好地理解 CMSIS 的设计。

**3. CMSIS 定义：**

```c
#include "cmsis_version.h"

/*  CMSIS definitions */
#define __CM23_CMSIS_VERSION_MAIN  (__CM_CMSIS_VERSION_MAIN)                   /*!< \deprecated [31:16] CMSIS HAL main version */
#define __CM23_CMSIS_VERSION_SUB   (__CM_CMSIS_VERSION_SUB)                    /*!< \deprecated [15:0]  CMSIS HAL sub version */
#define __CM23_CMSIS_VERSION       ((__CM23_CMSIS_VERSION_MAIN << 16U) | \
                                     __CM23_CMSIS_VERSION_SUB           )      /*!< \deprecated CMSIS HAL version number */

#define __CORTEX_M                 (23U)                                       /*!< Cortex-M Core */

/** __FPU_USED indicates whether an FPU is used or not.
    This core does not support an FPU at all
*/
#define __FPU_USED       0U
```

*   **描述:**
    *   `#include "cmsis_version.h"`: 包含了 CMSIS 版本信息的头文件。
    *   `__CM23_CMSIS_VERSION_MAIN`, `__CM23_CMSIS_VERSION_SUB`, `__CM23_CMSIS_VERSION`: 定义了 CMSIS HAL 的主版本号、子版本号和完整版本号。这些宏已经被弃用 (deprecated)。
    *   `__CORTEX_M`: 定义了 Cortex-M 内核的编号 (23)。
    *   `__FPU_USED`: 定义了是否使用浮点单元 (FPU)。对于 Cortex-M23，通常不包含 FPU，所以设置为 0。
*   **如何使用:** 这些宏主要用于库的版本控制和条件编译。你可能在代码中用到 `__CORTEX_M` 来判断当前使用的内核类型。

**4. IO 定义：**

```c
/* IO definitions (access restrictions to peripheral registers) */
/**
    \defgroup CMSIS_glob_defs CMSIS Global Defines

    <strong>IO Type Qualifiers</strong> are used
    \li to specify the access to peripheral variables.
    \li for automatic generation of peripheral register debug information.
*/
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

/* following defines should be used for structure members */
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */
```

*   **描述:**
    *   这些宏定义了 I/O 类型限定符，用于指定对外设寄存器的访问权限。`volatile` 关键字告诉编译器不要对外设寄存器的访问进行优化，确保每次都从实际地址读取或写入。
    *   `__I`: 只读 (Read Only)
    *   `__O`: 只写 (Write Only)
    *   `__IO`: 读写 (Read / Write)
    *   `__IM`, `__OM`, `__IOM`: 用于结构体成员的只读、只写、读写权限。
*   **如何使用:** 这些限定符用于定义外设寄存器的结构体，例如：

    ```c
    typedef struct {
        __IO uint32_t CTRL;  // Control Register (读写)
        __I  uint32_t STATUS; // Status Register (只读)
    } MyPeripheral_Type;
    ```

**5. 寄存器抽象：**

这部分代码定义了用于访问 Cortex-M23 内核寄存器和外设的结构体类型。由于篇幅原因，这里只给出 APSR (应用程序状态寄存器) 的定义作为示例，其他的定义方式类似。

```c
/**
  \brief  Union type to access the Application Program Status Register (APSR).
 */
typedef union
{
  struct
  {
    uint32_t _reserved0:28;              /*!< bit:  0..27  Reserved */
    uint32_t V:1;                        /*!< bit:     28  Overflow condition code flag */
    uint32_t C:1;                        /*!< bit:     29  Carry condition code flag */
    uint32_t Z:1;                        /*!< bit:     30  Zero condition code flag */
    uint32_t N:1;                        /*!< bit:     31  Negative condition code flag */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} APSR_Type;

/* APSR Register Definitions */
#define APSR_N_Pos                         31U                                            /*!< APSR: N Position */
#define APSR_N_Msk                         (1UL << APSR_N_Pos)                            /*!< APSR: N Mask */

#define APSR_Z_Pos                         30U                                            /*!< APSR: Z Position */
#define APSR_Z_Msk                         (1UL << APSR_Z_Pos)                            /*!< APSR: Z Mask */

#define APSR_C_Pos                         29U                                            /*!< APSR: C Position */
#define APSR_C_Msk                         (1UL << APSR_C_Pos)                            /*!< APSR: C Mask */

#define APSR_V_Pos                         28U                                            /*!< APSR: V Position */
#define APSR_V_Msk                         (1UL << APSR_V_Pos)                            /*!< APSR: V Mask */
```

*   **描述:**
    *   `APSR_Type` 是一个联合体 (union)，允许你以位域 (bit field) 的方式 (`b` 成员) 或以整个 32 位字 (`w` 成员) 的方式访问 APSR 寄存器。
    *   位域结构体 `b` 定义了 APSR 寄存器中各个标志位 (N, Z, C, V) 的名称和位置。`_reserved0` 用于填充未使用的位。
    *   `APSR_N_Pos`, `APSR_N_Msk` 等宏定义了每个标志位的位置和掩码，方便你操作特定的标志位。
*   **如何使用:**

    ```c
    APSR_Type apsr_value;

    // 读取 APSR 寄存器的值
    apsr_value.w = __get_PSR(); // 使用 CMSIS 内联函数读取 PSR

    // 检查 N (负数) 标志位
    if (apsr_value.b.N) {
        // 执行相应的操作
    }

    // 使用宏来设置 Z (零) 标志位 (不推荐直接修改 APSR)
    // apsr_value.w |= APSR_Z_Msk;  // 错误！不能直接修改 APSR
    ```

    **注意:**  通常情况下，不应该直接修改 APSR 寄存器，而是通过指令执行来改变它的值。

**6. 硬件抽象层 (HAL)：**

这部分代码提供了用于操作内核外设的内联函数。这里以 NVIC (嵌套向量中断控制器) 的使能中断函数为例：

```c
/**
  \brief   Enable Interrupt
  \details Enables a device specific interrupt in the NVIC interrupt controller.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void __NVIC_EnableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ISER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}
```

*   **描述:**
    *   `__NVIC_EnableIRQ` 是一个内联函数，用于使能指定的中断。
    *   `IRQn_Type` 是中断号的类型 (通常是一个枚举类型，在设备特定的头文件中定义)。
    *   该函数首先检查中断号是否有效 (非负数)。
    *   然后，它通过访问 `NVIC->ISER` 寄存器来设置相应的中断使能位。`(((uint32_t)IRQn) >> 5UL)` 计算出要访问的 `ISER` 数组的索引，`(1UL << (((uint32_t)IRQn) & 0x1FUL))` 计算出要设置的位掩码。
    *   `__STATIC_INLINE`  指示编译器尝试将该函数内联到调用处，以提高性能。
*   **如何使用:**

    ```c
    #include "core_cm23.h" // 包含 CMSIS 内核头文件
    #include "stm32l5xx.h" // 包含设备特定的头文件 (例如，STM32L5 系列)

    void MyInterruptHandler(void) {
        // 中断处理代码
    }

    int main(void) {
        // ... 其他初始化代码 ...

        // 使能 USART1 中断
        NVIC_EnableIRQ(USART1_IRQn);

        // 设置中断优先级 (可选)
        NVIC_SetPriority(USART1_IRQn, 3);

        // ...
        while (1) {
            // 主循环
        }
    }
    ```

    在这个例子中，我们首先包含了 `core_cm23.h` 和设备特定的头文件。然后，我们使用 `NVIC_EnableIRQ` 函数来使能 USART1 的中断。`USART1_IRQn` 是一个在设备特定的头文件中定义的枚举常量，表示 USART1 中断的中断号。

**其他说明:**

*   `__DSB()` 和 `__ISB()`：这些是内联汇编指令，用于确保数据同步和指令同步。在修改 NVIC 寄存器后，通常需要使用这些指令来确保更改生效。
*   `__NO_RETURN`:  这个宏用于标记那些不会返回的函数，例如 `__NVIC_SystemReset()`。

总的来说，`core_cm23.h` 是一个非常重要的头文件，它为开发者提供了访问和控制 Cortex-M23 内核的底层接口。理解这个头文件中的定义和函数，可以帮助你更好地开发嵌入式应用程序。记住，在实际使用时，还需要包含设备特定的头文件，以获取正确的外设地址和中断号定义。
