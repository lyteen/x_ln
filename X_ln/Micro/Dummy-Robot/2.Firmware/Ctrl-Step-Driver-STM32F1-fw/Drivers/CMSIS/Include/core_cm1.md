Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\core_cm1.h`

好的，我们重新开始，分析 `core_cm1.h` 文件的关键部分，并用中文解释，添加代码片段，并提供使用示例。

**1. 文件头和版权声明:**

```c
/**************************************************************************//**
 * @file     core_cm1.h
 * @brief    CMSIS Cortex-M1 Core Peripheral Access Layer Header File
 * @version  V1.0.0
 * @date     23. July 2018
 ******************************************************************************/
/*
 * Copyright (c) 2009-2018 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0
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
```

**描述:**  这是文件的开头，包含文件名，简短描述，版本信息和版权声明。  它指明这个头文件是CMSIS (Cortex Microcontroller Software Interface Standard) 的一部分，专门为Cortex-M1内核设计的。  SPDX-License-Identifier 指定了许可证类型 (Apache 2.0)。

**如何使用:**  通常情况下，你不需要修改这部分内容。 这部分是文件的元数据。

**2. 预处理器指令和包含头文件:**

```c
#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header   /* treat file as system include file */
#endif

#ifndef __CORE_CM1_H_GENERIC
#define __CORE_CM1_H_GENERIC

#include <stdint.h>

#ifdef __cplusplus
 extern "C" {
#endif
```

**描述:**  这部分代码使用了预处理器指令来处理不同的编译器。  `#pragma system_include` 和 `#pragma clang system_header` 指示编译器将此文件视为系统头文件，可能影响某些警告的处理（例如 MISRA 检查）。 `#ifndef __CORE_CM1_H_GENERIC`  和 `#define __CORE_CM1_H_GENERIC`  用于防止头文件被重复包含。  `#include <stdint.h>`  包含了标准整数类型定义（如 `uint32_t`）。  `#ifdef __cplusplus`  和 `extern "C"`  用于支持 C++ 编译环境，确保 C 函数的链接方式正确。

**如何使用:**  这些指令由编译器自动处理。 你不需要手动修改它们。

**3. MISRA 异常说明:**

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

**描述:**  这部分文档说明了 CMSIS 代码违反了 MISRA-C:2004 规范中的某些规则。  MISRA-C 是一套 C 语言的编码标准，旨在提高代码的安全性、可靠性和可维护性。 CMSIS 为了性能和效率，选择性地忽略了一些 MISRA 规则，并在此处进行了说明。

**如何使用:**  如果你需要遵循严格的 MISRA 规范，你需要了解这些异常并进行评估。  在大多数情况下，可以忽略此部分。

**4. CMSIS 定义:**

```c
/**
  \ingroup Cortex_M1
  @{
 */

#include "cmsis_version.h"

/*  CMSIS CM1 definitions */
#define __CM1_CMSIS_VERSION_MAIN  (__CM_CMSIS_VERSION_MAIN)              /*!< \deprecated [31:16] CMSIS HAL main version */
#define __CM1_CMSIS_VERSION_SUB   (__CM_CMSIS_VERSION_SUB)               /*!< \deprecated [15:0]  CMSIS HAL sub version */
#define __CM1_CMSIS_VERSION       ((__CM1_CMSIS_VERSION_MAIN << 16U) | \
                                    __CM1_CMSIS_VERSION_SUB           )  /*!< \deprecated CMSIS HAL version number */

#define __CORTEX_M                (1U)                                   /*!< Cortex-M Core */

/** __FPU_USED indicates whether an FPU is used or not.
    This core does not support an FPU at all
*/
#define __FPU_USED       0U

// ... 其他 FPU 相关检查 ...

#include "cmsis_compiler.h"               /* CMSIS compiler specific defines */
```

**描述:**  这部分定义了一些重要的宏。 `__CM1_CMSIS_VERSION_MAIN`， `__CM1_CMSIS_VERSION_SUB` 和 `__CM1_CMSIS_VERSION`  定义了 CMSIS 版本号 (注意 `\deprecated` 注释，说明这些宏可能不推荐使用)。 `__CORTEX_M`  定义了 Cortex-M 内核的类型 (1 代表 Cortex-M1)。 `__FPU_USED`  指示是否使用了浮点单元 (FPU)。 在 Cortex-M1 中，FPU 不可用，所以定义为 0。  `cmsis_compiler.h`  包含了针对不同编译器的特定定义。

**如何使用:**  这些宏主要用于条件编译。 例如，你可以使用 `__FPU_USED`  来判断是否包含 FPU 相关的代码。

**5. IO 类型限定符:**

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

**描述:**  这些宏定义了 I/O 类型限定符，用于指定对外设寄存器的访问权限。 `__I`  表示只读， `__O`  表示只写， `__IO`  表示可读写。 `volatile` 关键字告诉编译器，该变量的值可能会被外部因素（例如中断）修改，因此每次访问该变量时都要从内存中读取，而不是使用缓存的值。  `__IM`, `__OM` 和 `__IOM`  用于结构体成员。

**如何使用:**  这些限定符用于定义外设寄存器的结构体。 例如：

```c
typedef struct {
  __IO uint32_t DATA;   // 数据寄存器，可读写
  __I  uint32_t STATUS; // 状态寄存器，只读
} MyPeripheral_Type;
```

**6. 核心寄存器结构体定义 (APSR, IPSR, xPSR, CONTROL):**

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

// ... 其他状态寄存器 (IPSR, xPSR) 和控制寄存器 (CONTROL) 的定义 ...
```

**描述:**  这些 `typedef union`  定义了访问 Cortex-M1 内核状态和控制寄存器的结构。 `APSR_Type`  是应用程序状态寄存器，包含 N（负数标志）、Z（零标志）、C（进位标志）和 V（溢出标志）等标志位。  使用 `union` 的目的是允许以位域 (`b` 成员) 或整个 32 位字 (`w` 成员) 的方式访问寄存器。

**如何使用:**  虽然可以直接访问这些寄存器，但通常建议使用 CMSIS 提供的函数来操作它们，以保证代码的可移植性和兼容性。  直接操作寄存器可能需要深入了解 Cortex-M1 的架构。

**7. NVIC 结构体和函数:**

```c
/**
  \brief  Structure type to access the Nested Vectored Interrupt Controller (NVIC).
 */
typedef struct
{
  __IOM uint32_t ISER[1U];               /*!< Offset: 0x000 (R/W)  Interrupt Set Enable Register */
        uint32_t RESERVED0[31U];
  __IOM uint32_t ICER[1U];               /*!< Offset: 0x080 (R/W)  Interrupt Clear Enable Register */
        uint32_t RSERVED1[31U];
  __IOM uint32_t ISPR[1U];               /*!< Offset: 0x100 (R/W)  Interrupt Set Pending Register */
        uint32_t RESERVED2[31U];
  __IOM uint32_t ICPR[1U];               /*!< Offset: 0x180 (R/W)  Interrupt Clear Pending Register */
        uint32_t RESERVED3[31U];
        uint32_t RESERVED4[64U];
  __IOM uint32_t IP[8U];                 /*!< Offset: 0x300 (R/W)  Interrupt Priority Register */
}  NVIC_Type;

#define __NVIC_EnableIRQ(IRQn)  { if ((int32_t)(IRQn) >= 0) { NVIC->ISER[0U] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL)); } }

// ... 其他 NVIC 函数 ...
```

**描述:** `NVIC_Type` 定义了访问 NVIC (Nested Vectored Interrupt Controller，嵌套向量中断控制器) 寄存器的结构。  `ISER`  用于使能中断， `ICER`  用于禁用中断， `ISPR`  用于设置中断挂起状态， `ICPR`  用于清除中断挂起状态， `IP`  用于设置中断优先级。  `__NVIC_EnableIRQ`  是一个内联函数，用于使能特定的中断。

**如何使用:**

```c
#include "core_cm1.h"

// 使能 UART1 中断
NVIC_EnableIRQ(UART1_IRQn);

// 设置 UART1 中断优先级为 2
NVIC_SetPriority(UART1_IRQn, 2);
```

**8. SCB 结构体和函数:**

```c
/**
  \brief  Structure type to access the System Control Block (SCB).
 */
typedef struct
{
  __IM  uint32_t CPUID;                  /*!< Offset: 0x000 (R/ )  CPUID Base Register */
  __IOM uint32_t ICSR;                   /*!< Offset: 0x004 (R/W)  Interrupt Control and State Register */
        uint32_t RESERVED0;
  __IOM uint32_t AIRCR;                  /*!< Offset: 0x00C (R/W)  Application Interrupt and Reset Control Register */
  __IOM uint32_t SCR;                    /*!< Offset: 0x010 (R/W)  System Control Register */
  __IOM uint32_t CCR;                    /*!< Offset: 0x014 (R/W)  Configuration Control Register */
        uint32_t RESERVED1;
  __IOM uint32_t SHP[2U];                /*!< Offset: 0x01C (R/W)  System Handlers Priority Registers. [0] is RESERVED */
  __IOM uint32_t SHCSR;                  /*!< Offset: 0x024 (R/W)  System Handler Control and State Register */
} SCB_Type;

#define __NVIC_SystemReset() ... // 代码已经提供
```

**描述:** `SCB_Type` 定义了访问 SCB (System Control Block，系统控制块) 寄存器的结构。 `CPUID`  包含 CPU 的 ID 信息， `ICSR`  是中断控制和状态寄存器， `AIRCR`  是应用中断和复位控制寄存器， `SCR`  是系统控制寄存器。`__NVIC_SystemReset()` 函数用于触发系统复位。

**如何使用:**

```c
#include "core_cm1.h"

// 触发系统复位
NVIC_SystemReset();
```

**9. SysTick 结构体和函数:**

```c
/**
  \brief  Structure type to access the System Timer (SysTick).
 */
typedef struct
{
  __IOM uint32_t CTRL;                   /*!< Offset: 0x000 (R/W)  SysTick Control and Status Register */
  __IOM uint32_t LOAD;                   /*!< Offset: 0x004 (R/W)  SysTick Reload Value Register */
  __IOM uint32_t VAL;                    /*!< Offset: 0x008 (R/W)  SysTick Current Value Register */
  __IM  uint32_t CALIB;                  /*!< Offset: 0x00C (R/ )  SysTick Calibration Register */
} SysTick_Type;

#if defined (__Vendor_SysTickConfig) && (__Vendor_SysTickConfig == 0U)
__STATIC_INLINE uint32_t SysTick_Config(uint32_t ticks) { ... }
#endif
```

**描述:** `SysTick_Type`  定义了访问 SysTick 定时器的结构。 `CTRL`  是控制和状态寄存器， `LOAD`  是重载值寄存器， `VAL`  是当前值寄存器， `CALIB`  是校准寄存器。`SysTick_Config()`  函数用于配置 SysTick 定时器。

**如何使用:**

```c
#include "core_cm1.h"

// 配置 SysTick 定时器，使其每 1ms 产生一次中断
SysTick_Config(SystemCoreClock / 1000);

// 在 SysTick 中断处理函数中执行一些操作
void SysTick_Handler(void) {
  // ...
}
```

总而言之，`core_cm1.h` 文件为 Cortex-M1 内核提供了一组标准化的接口，用于访问内核的寄存器和控制系统功能。 了解这些定义和函数对于编写 Cortex-M1 平台的嵌入式应用程序至关重要。