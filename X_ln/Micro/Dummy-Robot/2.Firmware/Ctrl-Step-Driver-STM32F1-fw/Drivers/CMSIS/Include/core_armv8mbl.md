Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\core_armv8mbl.h`

好的，我们重新开始。

这段代码是一个 CMSIS (Cortex Microcontroller Software Interface Standard) 核心外设访问层头文件，专门为 Armv8-M Baseline 架构设计的。它提供了一种标准化的方式来访问 Cortex-M 处理器的核心外设，例如 NVIC（嵌套向量中断控制器）、SCB（系统控制块）和 SysTick 定时器。以下是代码的关键部分的解释，每个部分都附带中文描述，使用说明和一个简单的演示例子。

**1. 文件头和版权声明:**

```c
/**************************************************************************//**
 * @file     core_armv8mbl.h
 * @brief    CMSIS Armv8-M Baseline Core Peripheral Access Layer Header File
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
```

**描述:** 这部分定义了文件名、功能描述、版本信息、日期以及版权和许可协议。 这确保了代码的版权和使用遵循 Apache 2.0 许可。

**中文描述:** 这部分是文件的头部信息，包含了文件名，简要描述，版本，日期，以及版权声明和许可协议。 它就像一个文件的身份证，告诉我们这个文件是做什么的，谁拥有它，以及我们如何使用它。

**2. 预处理器指令和包含头文件:**

```c
#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header   /* treat file as system include file */
#endif

#ifndef __CORE_ARMV8MBL_H_GENERIC
#define __CORE_ARMV8MBL_H_GENERIC

#include <stdint.h>

#ifdef __cplusplus
 extern "C" {
#endif
```

**描述:** 这部分代码使用预处理器指令来处理不同的编译器，并包含标准整数类型头文件 `<stdint.h>`。 `extern "C"` 确保 C++ 代码可以正确链接到 C 代码。

**中文描述:** 这部分使用了预处理指令来适配不同的编译器环境，比如 IAR 和 Clang。它还包含了 `<stdint.h>` 头文件，定义了标准整数类型，例如 `uint32_t`。`extern "C"` 的作用是让 C++ 编译器知道这段代码是用 C 语言编写的，从而保证 C++ 代码可以正确地调用 C 语言的函数。

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

**描述:** 这部分声明了 CMSIS 代码违反的 MISRA-C 规则。 这通常是由于内联函数、联合和函数式宏的使用，这些都是为了性能而进行的优化。

**中文描述:** 这部分说明了 CMSIS 代码中违反 MISRA-C (一种 C 语言编码规范) 的地方。 这些违反通常是为了提高代码的效率，比如使用内联函数、联合体和函数式宏。

**4. CMSIS 定义:**

```c
#include "cmsis_version.h"

/*  CMSIS definitions */
#define __ARMv8MBL_CMSIS_VERSION_MAIN  (__CM_CMSIS_VERSION_MAIN)
#define __ARMv8MBL_CMSIS_VERSION_SUB   (__CM_CMSIS_VERSION_SUB)
#define __ARMv8MBL_CMSIS_VERSION       ((__ARMv8MBL_CMSIS_VERSION_MAIN << 16U) | \
                                         __ARMv8MBL_CMSIS_VERSION_SUB           )

#define __CORTEX_M                     ( 2U)

/** __FPU_USED indicates whether an FPU is used or not.
    This core does not support an FPU at all
*/
#define __FPU_USED       0U
```

**描述:** 此部分包含 CMSIS 版本信息和核心定义。 `__FPU_USED` 被设置为 0，表示该核心不支持 FPU（浮点单元）。

**中文描述:** 这部分定义了 CMSIS 的版本信息以及一些核心的配置。`__FPU_USED` 被设置为 0，说明这个 Cortex-M 核心没有使用浮点运算单元 (FPU)。

**5. 编译器检查:**

```c
#if defined ( __CC_ARM )
  #if defined __TARGET_FPU_VFP
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif
#endif
// ... 更多编译器检查
```

**描述:** 这部分包含一系列编译器检查，以确保编译器不会为没有 FPU 的设备生成 FPU 指令。

**中文描述:**  这段代码会根据使用的编译器，检查是否生成了针对 FPU 的指令，但是当前内核又没有 FPU 单元。如果出现这种情况，编译器会报错，提示开发者检查配置。

**6. I/O 定义:**

```c
#ifdef __cplusplus
  #define   __I     volatile
#else
  #define   __I     volatile const
#endif
#define     __O     volatile
#define     __IO    volatile

#define     __IM     volatile const
#define     __OM     volatile
#define     __IOM    volatile
```

**描述:** 这些宏定义了 I/O 访问限定符。`__I` 表示只读，`__O` 表示只写，`__IO` 表示可读写。 这些限定符可以帮助编译器进行优化，并提供关于寄存器访问的信息。

**中文描述:** 这些宏定义了 I/O 访问权限。`__I` 代表只读，`__O` 代表只写，`__IO` 代表可读写。使用这些定义可以帮助编译器进行优化，并且方便开发者了解寄存器的访问权限。

**7. 寄存器结构体定义 (例: NVIC_Type):**

```c
typedef struct
{
  __IOM uint32_t ISER[16U];              /*!< Offset: 0x000 (R/W)  Interrupt Set Enable Register */
        uint32_t RESERVED0[16U];
  __IOM uint32_t ICER[16U];              /*!< Offset: 0x080 (R/W)  Interrupt Clear Enable Register */
        uint32_t RSERVED1[16U];
  __IOM uint32_t ISPR[16U];              /*!< Offset: 0x100 (R/W)  Interrupt Set Pending Register */
        uint32_t RESERVED2[16U];
  __IOM uint32_t ICPR[16U];              /*!< Offset: 0x180 (R/W)  Interrupt Clear Pending Register */
        uint32_t RESERVED3[16U];
  __IOM uint32_t IABR[16U];              /*!< Offset: 0x200 (R/W)  Interrupt Active bit Register */
        uint32_t RESERVED4[16U];
  __IOM uint32_t ITNS[16U];              /*!< Offset: 0x280 (R/W)  Interrupt Non-Secure State Register */
        uint32_t RESERVED5[16U];
  __IOM uint32_t IPR[124U];              /*!< Offset: 0x300 (R/W)  Interrupt Priority Register */
}  NVIC_Type;
```

**描述:**  这个结构体定义了 NVIC (Nested Vectored Interrupt Controller) 的寄存器布局。 每个成员都标有 `__IOM`，表明它是可读写的。`RESERVED` 用于填充结构体以匹配硬件布局。

**中文描述:** 这个结构体 `NVIC_Type` 定义了嵌套向量中断控制器 (NVIC) 的所有寄存器。 `__IOM` 表示这些寄存器是可读写的。`RESERVED` 用于填充结构体，保证结构体的大小和硬件的寄存器布局一致。

**用法演示:**

```c
#include "core_armv8mbl.h"

int main() {
  // Enable interrupt number 5
  NVIC->ISER[0] = (1 << 5);

  // Set priority for interrupt number 5
  NVIC->IPR[1] = (5 << 24); // Assuming 8 priority bits, setting priority to 5
  return 0;
}
```

这段代码演示了如何使用 `NVIC_Type` 结构体来启用中断并设置其优先级。

**8. 寄存器位域定义 (例: SCB_AIRCR 寄存器):**

```c
/* SCB Application Interrupt and Reset Control Register Definitions */
#define SCB_AIRCR_VECTKEY_Pos              16U
#define SCB_AIRCR_VECTKEY_Msk              (0xFFFFUL << SCB_AIRCR_VECTKEY_Pos)

#define SCB_AIRCR_SYSRESETREQ_Pos           2U
#define SCB_AIRCR_SYSRESETREQ_Msk          (1UL << SCB_AIRCR_SYSRESETREQ_Pos)
```

**描述:** 这些宏定义了 `SCB->AIRCR`（应用程序中断和重置控制寄存器）中各个位域的位置和掩码。 例如，`SCB_AIRCR_SYSRESETREQ_Pos` 定义了 SYSRESETREQ 位的位置，而 `SCB_AIRCR_SYSRESETREQ_Msk` 定义了它的掩码。

**中文描述:** 这些宏定义了 `SCB->AIRCR` 寄存器中各个位的位置和掩码。例如，`SCB_AIRCR_SYSRESETREQ_Pos` 定义了 SYSRESETREQ 位的位置，而 `SCB_AIRCR_SYSRESETREQ_Msk` 定义了它的掩码。通过使用这些宏，开发者可以方便地设置和读取寄存器中的特定位。

**9. 核心外设基地址定义:**

```c
  #define SCS_BASE            (0xE000E000UL)
  #define SysTick_BASE        (SCS_BASE +  0x0010UL)
  #define NVIC_BASE           (SCS_BASE +  0x0100UL)
  #define SCB_BASE            (SCS_BASE +  0x0D00UL)
```

**描述:** 这些宏定义了核心外设的基地址。 例如，`NVIC_BASE` 定义了 NVIC 的基地址。

**中文描述:** 这些宏定义了核心外设的起始地址。例如，`NVIC_BASE` 定义了 NVIC (嵌套向量中断控制器) 的起始地址。 开发者可以通过这些地址来访问对应的外设寄存器。

**10. 外设访问结构体定义:**

```c
  #define SCB                 ((SCB_Type       *)     SCB_BASE         )
  #define SysTick             ((SysTick_Type   *)     SysTick_BASE     )
  #define NVIC                ((NVIC_Type      *)     NVIC_BASE        )
```

**描述:** 这些宏将基地址转换为指向相应结构体类型的指针，允许使用结构体成员语法来访问外设寄存器。

**中文描述:** 这些宏将外设的基地址转换为对应结构体类型的指针。 这样，开发者就可以使用结构体成员的方式来访问外设的寄存器，例如 `NVIC->ISER[0]`。

**11. 内联函数 (例: __NVIC_EnableIRQ):**

```c
__STATIC_INLINE void __NVIC_EnableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ISER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}
```

**描述:**  这是一个内联函数，用于启用 NVIC 中的特定中断。 它直接写入 `NVIC->ISER` 寄存器。

**中文描述:** 这是一个内联函数，用于启用 NVIC 中的特定中断。它通过直接写入 `NVIC->ISER` 寄存器来实现。内联函数可以减少函数调用的开销，提高代码执行效率。

**用法演示:**

```c
#include "core_armv8mbl.h"

int main() {
  // Enable IRQ number 5
  __NVIC_EnableIRQ(5);
  return 0;
}
```

这段代码演示了如何使用 `__NVIC_EnableIRQ` 函数来启用中断。

**总结:**

这个头文件是嵌入式开发中非常重要的一个组件，它提供了一种标准化的方法来访问 Cortex-M 处理器的核心外设。 通过使用这个头文件，开发者可以编写出更具可移植性和可维护性的代码。 它定义了寄存器结构体、位域宏和内联函数，使得开发者可以方便地访问和控制核心外设。
