Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\core_armv8mml.h`

好的，重新开始，我将为您解释 `core_armv8mml.h` 文件的关键部分，并提供每个小代码块的解释和用例。

```c
/**************************************************************************//**
 * @file     core_armv8mml.h
 * @brief    CMSIS Armv8-M Mainline Core Peripheral Access Layer Header File
 * @version  V5.0.7
 * @date     06. July 2018
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

**描述:**  这部分是文件的头注释，提供了文件的名称、简要描述、版本信息、日期以及版权和许可信息。  `SPDX-License-Identifier: Apache-2.0` 表明该代码使用 Apache 2.0 许可。

```c
#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header   /* treat file as system include file */
#endif

#ifndef __CORE_ARMV8MML_H_GENERIC
#define __CORE_ARMV8MML_H_GENERIC

#include <stdint.h>

#ifdef __cplusplus
 extern "C" {
#endif
```

**描述:**
*   这段代码是编译器的预处理指令。它检查是否定义了 `__ICCARM__` （IAR 编译器）或 `__clang__`。 如果定义了其中一个，则使用 `#pragma system_include` 或 `#pragma clang system_header` 指令来将该文件视为系统包含文件，以便进行 MISRA 检查（一种代码质量标准）。
*   `#ifndef __CORE_ARMV8MML_H_GENERIC` 和 `#define __CORE_ARMV8MML_H_GENERIC` 用于防止头文件被重复包含。
*   `#include <stdint.h>` 包含标准整数类型定义，例如 `uint32_t`。
*   `#ifdef __cplusplus` 和 `extern "C" {` 用于确保 C++ 代码可以正确链接 C 代码。

**用例:**  这些代码确保头文件只被包含一次，并且可以被 C 和 C++ 代码使用，同时确保代码质量。

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

**描述:**  这段注释描述了 CMSIS 代码对 MISRA-C:2004 规则的一些例外情况。 由于性能原因，CMSIS 代码有时会违反某些 MISRA 规则。  例如，为了内联函数，会在头文件中定义函数。

**用例:**  这段代码告诉开发者 CMSIS 代码在哪些方面不符合 MISRA 标准，以及这样做的原因是出于性能考虑。

```c
/*******************************************************************************
 *                 CMSIS definitions
 ******************************************************************************/
/**
  \ingroup Cortex_ARMv8MML
  @{
 */

#include "cmsis_version.h"

/*  CMSIS Armv8MML definitions */
#define __ARMv8MML_CMSIS_VERSION_MAIN  (__CM_CMSIS_VERSION_MAIN)                   /*!< \deprecated [31:16] CMSIS HAL main version */
#define __ARMv8MML_CMSIS_VERSION_SUB   (__CM_CMSIS_VERSION_SUB)                    /*!< \deprecated [15:0]  CMSIS HAL sub version */
#define __ARMv8MML_CMSIS_VERSION       ((__ARMv8MML_CMSIS_VERSION_MAIN << 16U) | \
                                         __ARMv8MML_CMSIS_VERSION_SUB           )  /*!< \deprecated CMSIS HAL version number */

#define __CORTEX_M                     (81U)                                       /*!< Cortex-M Core */
```

**描述:**
*   `#include "cmsis_version.h"` 包含 CMSIS 版本信息。
*   `__ARMv8MML_CMSIS_VERSION_MAIN`, `__ARMv8MML_CMSIS_VERSION_SUB`, `__ARMv8MML_CMSIS_VERSION` 定义了 CMSIS HAL 的主要版本、子版本和完整版本号。  这些定义已被弃用。
*   `__CORTEX_M` 定义了 Cortex-M 核心的类型，`81U` 对应于 Armv8-M。

**用例:**  这些定义提供了有关 CMSIS 库版本和目标核心的信息。

```c
/** __FPU_USED indicates whether an FPU is used or not.
    For this, __FPU_PRESENT has to be checked prior to making use of FPU specific registers and functions.
*/
#if defined ( __CC_ARM )
  #if defined __TARGET_FPU_VFP
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined(__ARM_FEATURE_DSP)
    #if defined(__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  #if defined __ARM_PCS_VFP
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #warning "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined(__ARM_FEATURE_DSP)
    #if defined(__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined ( __GNUC__ )
  #if defined (__VFP_FP__) && !defined(__SOFTFP__)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined(__ARM_FEATURE_DSP)
    #if defined(__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined ( __ICCARM__ )
  #if defined __ARMVFP__
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED         0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined(__ARM_FEATURE_DSP)
    #if defined(__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined ( __TI_ARM__ )
  #if defined __TI_VFP_SUPPORT__
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED         0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

#elif defined ( __TASKING__ )
  #if defined __FPU_VFP__
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

#elif defined ( __CSMC__ )
  #if ( __CSMC__ & 0x400U)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

#endif

#include "cmsis_compiler.h"               /* CMSIS compiler specific defines */
```

**描述:**

*   这段代码检查编译器是否使用了 FPU（浮点单元）和 DSP (数字信号处理器)指令集。
*   `__FPU_USED` 宏定义指示编译器是否生成 FPU 指令。如果编译器生成 FPU 指令，但设备没有 FPU，则会产生错误。
*   `__DSP_USED` 宏定义指示编译器是否生成 DSP (SIMD) 指令。如果编译器生成 DSP 指令，但设备没有 DSP 扩展，则会产生错误。
*   代码针对不同的编译器 (如 `__CC_ARM`, `__GNUC__`, `__ICCARM__` 等) 提供了不同的检查方法。
*   `#include "cmsis_compiler.h"` 包含了与编译器相关的定义，例如属性定义。

**用例:** 这些宏定义确保编译器生成的代码与目标硬件的功能匹配，防止生成无效或错误的指令。

```c
#ifdef __cplusplus
}
#endif

#endif /* __CORE_ARMV8MML_H_GENERIC */

#ifndef __CMSIS_GENERIC

#ifndef __CORE_ARMV8MML_H_DEPENDANT
#define __CORE_ARMV8MML_H_DEPENDANT

#ifdef __cplusplus
 extern "C" {
#endif

/* check device defines and use defaults */
#if defined __CHECK_DEVICE_DEFINES
  #ifndef __ARMv8MML_REV
    #define __ARMv8MML_REV               0x0000U
    #warning "__ARMv8MML_REV not defined in device header file; using default!"
  #endif

  #ifndef __FPU_PRESENT
    #define __FPU_PRESENT             0U
    #warning "__FPU_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __MPU_PRESENT
    #define __MPU_PRESENT             0U
    #warning "__MPU_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __SAUREGION_PRESENT
    #define __SAUREGION_PRESENT       0U
    #warning "__SAUREGION_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __DSP_PRESENT
    #define __DSP_PRESENT             0U
    #warning "__DSP_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __NVIC_PRIO_BITS
    #define __NVIC_PRIO_BITS          3U
    #warning "__NVIC_PRIO_BITS not defined in device header file; using default!"
  #endif

  #ifndef __Vendor_SysTickConfig
    #define __Vendor_SysTickConfig    0U
    #warning "__Vendor_SysTickConfig not defined in device header file; using default!"
  #endif
#endif
```

**描述:**

*   `#ifndef __CMSIS_GENERIC` 和 `#ifndef __CORE_ARMV8MML_H_DEPENDANT` 用于确保代码只被包含一次。
*   `#if defined __CHECK_DEVICE_DEFINES`： 这部分代码检查设备相关的宏定义是否在设备头文件中定义。如果没有定义，则使用默认值，并发出警告。这包括:
    *   `__ARMv8MML_REV`：Armv8-M 微控制器的修订版。
    *   `__FPU_PRESENT`：指示是否存在浮点单元（FPU）。
    *   `__MPU_PRESENT`：指示是否存在内存保护单元（MPU）。
    *   `__SAUREGION_PRESENT`：指示是否存在安全属性单元（SAU)。
    *   `__DSP_PRESENT`：指示是否存在数字信号处理 (DSP) 扩展。
    *   `__NVIC_PRIO_BITS`：中断优先级位数。
    *   `__Vendor_SysTickConfig`：指示是否使用供应商提供的 SysTick 配置。

**用例:** 这段代码确保即使设备头文件缺少某些定义，CMSIS 也能正常工作，并提供合理的默认值。

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

/*@} end of group ARMv8MML */
```

**描述:**

*   这段代码定义了 I/O 访问限定符，用于指定对外设寄存器的访问权限。
    *   `__I`：只读 (Read-only)。
    *   `__O`：只写 (Write-only)。
    *   `__IO`：读写 (Read-write)。
*   对于结构体成员，定义了 `__IM`, `__OM`, `__IOM`，作用相同，但应用于结构体成员。
*   `volatile` 关键字确保每次访问都直接从内存读取，防止编译器优化掉对这些寄存器的访问。 `const` 确保该变量不能被修改。

**用例:** 这些限定符用于定义外设寄存器的访问权限，防止程序错误地写入只读寄存器，或者读取只写寄存器。它们也有助于生成调试信息。

```c
/*******************************************************************************
 *                 Register Abstraction
  Core Register contain:
  - Core Register
  - Core NVIC Register
  - Core SCB Register
  - Core SysTick Register
  - Core Debug Register
  - Core MPU Register
  - Core SAU Register
  - Core FPU Register
 ******************************************************************************/
/**
  \defgroup CMSIS_core_register Defines and Type Definitions
  \brief Type definitions and defines for Cortex-M processor based devices.
*/

/**
  \ingroup    CMSIS_core_register
  \defgroup   CMSIS_CORE  Status and Control Registers
  \brief      Core Register type definitions.
  @{
 */

/**
  \brief  Union type to access the Application Program Status Register (APSR).
 */
typedef union
{
  struct
  {
    uint32_t _reserved0:16;              /*!< bit:  0..15  Reserved */
    uint32_t GE:4;                       /*!< bit: 16..19  Greater than or Equal flags */
    uint32_t _reserved1:7;               /*!< bit: 20..26  Reserved */
    uint32_t Q:1;                        /*!< bit:     27  Saturation condition flag */
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

/* ... (类似地定义了 C, V, Q, GE 的 Pos 和 Msk) ... */
```

**描述:**

*   这部分代码开始定义核心寄存器的抽象。
*   `APSR_Type` 是一个联合体，用于访问应用程序程序状态寄存器 (APSR)。 它可以按位访问（使用 `b` 结构体）或按字访问（使用 `w` 成员）。
*   `APSR_N_Pos`, `APSR_N_Msk` 等定义了 APSR 寄存器中各个位的偏移量和掩码。  `Pos` 表示位的位置，`Msk` 表示用于提取该位的掩码。

**用例:** 联合体允许灵活地访问寄存器，既可以作为一个整体的 32 位值，也可以单独访问各个标志位。例如，可以使用 `APSR_Type` 来检查上次运算是否产生负数结果（通过访问 `APSR.b.N`）。

```c
/**
  \brief  Union type to access the Interrupt Program Status Register (IPSR).
 */
typedef union
{
  struct
  {
    uint32_t ISR:9;                      /*!< bit:  0.. 8  Exception number */
    uint32_t _reserved0:23;              /*!< bit:  9..31  Reserved */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} IPSR_Type;

/* IPSR Register Definitions */
#define IPSR_ISR_Pos                        0U                                            /*!< IPSR: ISR Position */
#define IPSR_ISR_Msk                       (0x1FFUL /*<< IPSR_ISR_Pos*/)                  /*!< IPSR: ISR Mask */
```

**描述:**

*   `IPSR_Type` 是一个联合体，用于访问中断程序状态寄存器 (IPSR)。
*   `ISR` 字段包含异常编号。
*   `IPSR_ISR_Pos` 和 `IPSR_ISR_Msk` 定义了 `ISR` 字段的偏移量和掩码。

**用例:** 可以使用 `IPSR_Type` 来确定当前正在执行的中断或异常的编号。

```c
/**
  \brief  Union type to access the Special-Purpose Program Status Registers (xPSR).
 */
typedef union
{
  struct
  {
    uint32_t ISR:9;                      /*!< bit:  0.. 8  Exception number */
    uint32_t _reserved0:7;               /*!< bit:  9..15  Reserved */
    uint32_t GE:4;                       /*!< bit: 16..19  Greater than or Equal flags */
    uint32_t _reserved1:4;               /*!< bit: 20..23  Reserved */
    uint32_t T:1;                        /*!< bit:     24  Thumb bit        (read 0) */
    uint32_t IT:2;                       /*!< bit: 25..26  saved IT state   (read 0) */
    uint32_t Q:1;                        /*!< bit:     27  Saturation condition flag */
    uint32_t V:1;                        /*!< bit:     28  Overflow condition code flag */
    uint32_t C:1;                        /*!< bit:     29  Carry condition code flag */
    uint32_t Z:1;                        /*!< bit:     30  Zero condition code flag */
    uint32_t N:1;                        /*!< bit:     31  Negative condition code flag */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} xPSR_Type;

/* xPSR Register Definitions */
#define xPSR_N_Pos                         31U                                            /*!< xPSR: N Position */
#define xPSR_N_Msk                         (1UL << xPSR_N_Pos)                            /*!< xPSR: N Mask */

/* ... (类似地定义了 Z, C, V, Q, IT, T, GE, ISR 的 Pos 和 Msk) ... */
```

**描述:**

*   `xPSR_Type` 是一个联合体，用于访问特殊用途程序状态寄存器 (xPSR)。
*   它包含了 APSR 和 IPSR 的组合，以及其他标志位，例如 T (Thumb bit) 和 IT (If-Then block state)。
*   定义了各个标志位的偏移量和掩码。

**用例:** `xPSR_Type` 提供了对处理器状态的全面访问。

```c
/**
  \brief  Union type to access the Control Registers (CONTROL).
 */
typedef union
{
  struct
  {
    uint32_t nPRIV:1;                    /*!< bit:      0  Execution privilege in Thread mode */
    uint32_t SPSEL:1;                    /*!< bit:      1  Stack-pointer select */
    uint32_t FPCA:1;                     /*!< bit:      2  Floating-point context active */
    uint32_t SFPA:1;                     /*!< bit:      3  Secure floating-point active */
    uint32_t _reserved1:28;              /*!< bit:  4..31  Reserved */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} CONTROL_Type;

/* CONTROL Register Definitions */
#define CONTROL_SFPA_Pos                    3U                                            /*!< CONTROL: SFPA Position */
#define CONTROL_SFPA_Msk                   (1UL << CONTROL_SFPA_Pos)                      /*!< CONTROL: SFPA Mask */

/* ... (类似地定义了 FPCA, SPSEL, nPRIV 的 Pos 和 Msk) ... */
```

**描述:**

*   `CONTROL_Type` 是一个联合体，用于访问控制寄存器 (CONTROL)。
*   `nPRIV` 位指示线程模式下的执行权限。
*   `SPSEL` 位选择要使用的堆栈指针 (MSP 或 PSP)。
*   `FPCA` 位指示浮点上下文是否处于活动状态。
*	`SFPA` 位指示安全浮点上下文是否处于活动状态.

**用例:** 可以使用 `CONTROL_Type` 来切换线程模式下的权限级别，选择堆栈指针，或检查 FPU 上下文是否处于活动状态。

```c
/**
  \ingroup    CMSIS_core_register
  \defgroup   CMSIS_NVIC  Nested Vectored Interrupt Controller (NVIC)
  \brief      Type definitions for the NVIC Registers
  @{
 */

/**
  \brief  Structure type to access the Nested Vectored Interrupt Controller (NVIC).
 */
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
  __IOM uint8_t  IPR[496U];              /*!< Offset: 0x300 (R/W)  Interrupt Priority Register (8Bit wide) */
        uint32_t RESERVED6[580U];
  __OM  uint32_t STIR;                   /*!< Offset: 0xE00 ( /W)  Software Trigger Interrupt Register */
}  NVIC_Type;

/* Software Triggered Interrupt Register Definitions */
#define NVIC_STIR_INTID_Pos                 0U                                         /*!< STIR: INTLINESNUM Position */
#define NVIC_STIR_INTID_Msk                (0x1FFUL /*<< NVIC_STIR_INTID_Pos*/)        /*!< STIR: INTLINESNUM Mask */
```

**描述:**

*   `NVIC_Type` 是一个结构体，用于访问嵌套向量中断控制器 (NVIC)。
*   `ISER` (Interrupt Set Enable Register)：用于启用中断。
*   `ICER` (Interrupt Clear Enable Register)：用于禁用中断。
*   `ISPR` (Interrupt Set Pending Register)：用于将中断设置为挂起状态。
*   `ICPR` (Interrupt Clear Pending Register)：用于清除中断的挂起状态。
*   `IABR` (Interrupt Active Bit Register)：用于读取中断的激活状态。
*   `ITNS` (Interrupt Target Non-Secure Register)：用于配置中断的目标安全状态.
*   `IPR` (Interrupt Priority Register)：用于设置中断优先级。
*   `STIR` (Software Trigger Interrupt Register)：用于软件触发中断。
*   `NVIC_STIR_INTID_Pos` 和 `NVIC_STIR_INTID_Msk` 定义了 `STIR` 寄存器中中断 ID 的偏移量和掩码。

**用例:**  可以使用 `NVIC_Type` 来管理中断，例如启用特定中断、设置中断优先级或软件触发中断。

```c
/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_SCB     System Control Block (SCB)
  \brief    Type definitions for the System Control Block Registers
  @{
 */

/**
  \brief  Structure type to access the System Control Block (SCB).
 */
typedef struct
{
  __IM  uint32_t CPUID;                  /*!< Offset: 0x000 (R/ )  CPUID Base Register */
  __IOM uint32_t ICSR;                   /*!< Offset: 0x004 (R/W)  Interrupt Control and State Register */
  __IOM uint32_t VTOR;                   /*!< Offset: 0x008 (R/W)  Vector Table Offset Register */
  __IOM uint32_t AIRCR;                  /*!< Offset: 0x00C (R/W)  Application Interrupt and Reset Control Register */
  __IOM uint32_t SCR;                    /*!< Offset: 0x010 (R/W)  System Control Register */
  __IOM uint32_t CCR;                    /*!< Offset: 0x014 (R/W)  Configuration Control Register */
  __IOM uint8_t  SHPR[12U];              /*!< Offset: 0x018 (R/W)  System Handlers Priority Registers (4-7, 8-11, 12-15) */
  __IOM uint32_t SHCSR;                  /*!< Offset: 0x024 (R/W)  System Handler Control and State Register */
  __IOM uint32_t CFSR;                   /*!< Offset: 0x028 (R/W)  Configurable Fault Status Register */
  __IOM uint32_t HFSR;                   /*!< Offset: 0x02C (R/W)  HardFault Status Register */
  __IOM uint32_t DFSR;                   /*!< Offset: 0x030 (R/W)  Debug Fault Status Register */
  __IOM uint32_t MMFAR;                  /*!< Offset: 0x034 (R/W)  MemManage Fault Address Register */
  __IOM uint32_t BFAR;                   /*!< Offset: 0x038 (R/W)  BusFault Address Register */
  __IOM uint32_t AFSR;                   /*!< Offset: 0x03C (R/W)  Auxiliary Fault Status Register */
  __IM  uint32_t ID_PFR[2U];             /*!< Offset: 0x040 (R/ )  Processor Feature Register */
  __IM  uint32_t ID_DFR;                 /*!< Offset: 0x048 (R/ )  Debug Feature Register */
  __IM  uint32_t ID_ADR;                 /*!< Offset: 0x04C (R/ )  Auxiliary Feature Register */
  __IM  uint32_t ID_MMFR[4U];            /*!< Offset: 0x050 (R/ )  Memory Model Feature Register */
  __IM  uint32_t ID_ISAR[6U];            /*!< Offset: 0x060 (R/ )  Instruction Set Attributes Register */
  __IM  uint32_t CLIDR;                  /*!< Offset: 0x078 (R/ )  Cache Level ID register */
  __IM  uint32_t CTR;                    /*!< Offset: 0x07C (R/ )  Cache Type register */
  __IM  uint32_t CCSIDR;                 /*!< Offset: 0x080 (R/ )  Cache Size ID Register */
  __IOM uint32_t CSSELR;                 /*!< Offset: 0x084 (R/W)  Cache Size Selection Register */
  __IOM uint32_t CPACR;                  /*!< Offset: 0x088 (R/W)  Coprocessor Access Control Register */
  __IOM uint32_t NSACR;                  /*!< Offset: 0x08C (R/W)  Non-Secure Access Control Register */
        uint32_t RESERVED3[92U];
  __OM  uint32_t STIR;                   /*!< Offset: 0x200 ( /W)  Software Triggered Interrupt Register */
        uint32_t RESERVED4[15U];
  __IM  uint32_t MVFR0;                  /*!< Offset: 0x240 (R/ )  Media and VFP Feature Register 0 */
  __IM  uint32_t MVFR1;                  /*!< Offset: 0x244 (R/ )  Media and VFP Feature Register 1 */
  __IM  uint32_t MVFR2;                  /*!< Offset: 0x248 (R/ )  Media and VFP Feature Register 2 */
        uint32_t RESERVED5[1U];
  __OM  uint32_t ICIALLU;                /*!< Offset: 0x250 ( /W)  I-Cache Invalidate All to PoU */