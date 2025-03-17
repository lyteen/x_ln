Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\mpu_armv8.h`

好的，我们重新开始，针对提供的 `mpu_armv8.h` 头文件，给出更智能的版本，并附带详细的解释和示例。

**目标:**  针对 Armv8-M MPU 的 CMSIS MPU API，旨在简化 MPU 的配置和使用。 为了使代码更易于阅读和使用，可以改进以下方面：

*   **类型安全:** 使用更强的类型来减少错误。
*   **易用性:** 提供更高级的 API，简化常见任务。
*   **可读性:** 改进代码的结构和注释。

**1. 类型安全的定义和枚举:**

```c
#ifndef ARM_MPU_ARMV8_H
#define ARM_MPU_ARMV8_H

#include <stdint.h>
#include <stdbool.h> // 引入 bool 类型

// 定义内存属性枚举
typedef enum {
    ARM_MPU_ATTR_DEVICE = 0U,                           // 设备内存
    ARM_MPU_ATTR_NON_CACHEABLE = 4U,                    // 不可缓存的普通内存
    ARM_MPU_ATTR_MEMORY_NORMAL = 8U                     // 普通内存 (示例值，实际使用时根据需要修改)
} ARM_MPU_MemAttr_t;

// 定义共享属性枚举
typedef enum {
    ARM_MPU_SH_NON = 0U,                                // 非共享
    ARM_MPU_SH_OUTER = 2U,                              // 外部共享
    ARM_MPU_SH_INNER = 3U                               // 内部共享
} ARM_MPU_Shareable_t;

// 定义访问权限枚举
typedef enum {
    ARM_MPU_AP_READ_ONLY = (1U << 1U),                  // 只读
    ARM_MPU_AP_NON_PRIVILEGED = (1U << 0U),            // 非特权
    ARM_MPU_AP_READ_WRITE = 0U                         // 读写 (没有设置 RO 和 NP)
} ARM_MPU_AccessPerm_t;
```

**描述:**

*   **`ARM_MPU_MemAttr_t`**:  枚举了常用的内存属性，使代码更具可读性。 避免了直接使用数字，增强了代码的可维护性。
*   **`ARM_MPU_Shareable_t`**:  枚举了共享属性，同样提高了代码的可读性。
*   **`ARM_MPU_AccessPerm_t`**:  枚举了访问权限，使得在配置 MPU 区域时更容易指定访问权限。
*   **`#include <stdbool.h>`**:  为了使用 `bool` 类型, 更清晰地表达使能/禁止的状态

**2. 改进的宏定义：**

```c
/** \brief Region Base Address Register value
* \param BASE The base address bits [31:5] of a memory region. The value is zero extended. Effective address gets 32 byte aligned.
* \param SH Defines the Shareability domain for this memory region.
* \param AP Access Permission (combined RO and NP).  Use ARM_MPU_AccessPerm_t enum.
* \oaram XN eXecute Never: Set to 1 for a non-executable memory region.
*/
#define ARM_MPU_RBAR(BASE, SH, AP, XN) \
  ((BASE & MPU_RBAR_BASE_Msk) | \
  ((SH << MPU_RBAR_SH_Pos) & MPU_RBAR_SH_Msk) | \
  (AP & MPU_RBAR_AP_Msk) | \
  ((XN << MPU_RBAR_XN_Pos) & MPU_RBAR_XN_Msk))

/** \brief Region Limit Address Register value
* \param LIMIT The limit address bits [31:5] for this memory region. The value is one extended.
* \param IDX The attribute index to be associated with this memory region.
*/
#define ARM_MPU_RLAR(LIMIT, IDX) \
  ((LIMIT & MPU_RLAR_LIMIT_Msk) | \
  ((IDX << MPU_RLAR_AttrIndx_Pos) & MPU_RLAR_AttrIndx_Msk) | \
  (MPU_RLAR_EN_Msk))

```

**描述:**

*   **`ARM_MPU_RBAR`**:  将 `RO` 和 `NP` 参数合并为一个 `AP` (Access Permission) 参数，并使用 `ARM_MPU_AccessPerm_t` 枚举，使代码更清晰。

**3. 更高级的API:**

```c
/**
 * @brief 配置 MPU 区域。
 *
 * @param rnr   区域编号。
 * @param base  基地址。
 * @param limit 限制地址。
 * @param mem_attr 内存属性 (使用 ARM_MPU_MemAttr_t)。
 * @param shareable 共享属性 (使用 ARM_MPU_Shareable_t)。
 * @param access_perm 访问权限 (使用 ARM_MPU_AccessPerm_t)。
 * @param execute_never  禁止执行。
 */
__STATIC_INLINE void ARM_MPU_ConfigureRegion(uint32_t rnr,
                                            uint32_t base,
                                            uint32_t limit,
                                            ARM_MPU_MemAttr_t mem_attr,
                                            ARM_MPU_Shareable_t shareable,
                                            ARM_MPU_AccessPerm_t access_perm,
                                            bool execute_never) {
    uint32_t rbar = ARM_MPU_RBAR(base, shareable, access_perm, execute_never);
    uint32_t rlar = ARM_MPU_RLAR(limit, mem_attr); //假设mem_attr可以直接作为index使用，具体要看硬件定义

    MPU->RNR  = rnr;
    MPU->RBAR = rbar;
    MPU->RLAR = rlar;
}

/**
 * @brief 使能 MPU.
 *
 * @param privileged_default 使能特权访问默认权限。
 */
__STATIC_INLINE void ARM_MPU_EnableWithDefaults(bool privileged_default) {
    uint32_t ctrl = (privileged_default ? MPU_CTRL_PRIVDEFENA_Msk : 0U) | MPU_CTRL_ENABLE_Msk;
    __DSB();
    __ISB();
    MPU->CTRL = ctrl;
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
    SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
#endif
}
```

**描述:**

*   **`ARM_MPU_ConfigureRegion`**:  提供了一个更方便的函数来配置 MPU 区域。  它接受所有必要的参数，并调用 `ARM_MPU_RBAR` 和 `ARM_MPU_RLAR` 来生成寄存器值。 这样可以避免手动计算寄存器值，减少出错的可能性。
*   **`ARM_MPU_EnableWithDefaults`**: 提供了一个更方便的函数来使能 MPU，并可选择使能特权访问默认权限。

**4. 示例代码 (假设基于 CMSIS-Core):**

```c
#include "stm32g4xx.h"  // 替换为你使用的 MCU 的头文件
#include "mpu_armv8.h"

int main(void) {
    // 1.  Disable MPU (before configuration)
    ARM_MPU_Disable();

    // 2.  Configure Memory Attributes (MAIR register)
    ARM_MPU_SetMemAttr(0, ARM_MPU_ATTR_NORMAL_MEMORY);    // Index 0: Normal memory
    ARM_MPU_SetMemAttr(1, ARM_MPU_ATTR_DEVICE);          // Index 1: Device memory
    ARM_MPU_SetMemAttr(2, ARM_MPU_ATTR_NON_CACHEABLE);     // Index 2: Non-cacheable

    // 3. Configure MPU Region (Region 0 for Flash)
    uint32_t flash_base = 0x08000000U; // 假设Flash起始地址
    uint32_t flash_limit = 0x0803FFFFU; // 假设Flash结束地址
    ARM_MPU_ConfigureRegion(0,                      // Region number
                            flash_base,             // Base address
                            flash_limit,            // Limit address
                            0,                    // Memory attribute index (Normal)
                            ARM_MPU_SH_NON,          // Not shareable
                            ARM_MPU_AP_READ_ONLY,    // Read-only
                            true);                  // Execute Never

    // 4. Configure MPU Region (Region 1 for SRAM)
    uint32_t sram_base = 0x20000000U;  // 假设SRAM起始地址
    uint32_t sram_limit = 0x2000FFFFU; // 假设SRAM结束地址
    ARM_MPU_ConfigureRegion(1,                      // Region number
                            sram_base,              // Base address
                            sram_limit,             // Limit address
                            0,                    // Memory attribute index (Normal)
                            ARM_MPU_SH_NON,          // Not shareable
                            ARM_MPU_AP_READ_WRITE,   // Read-write
                            false);                 // Execute Allowed

    // 5. Enable MPU
    ARM_MPU_EnableWithDefaults(false);  // Disable privileged default access

    while (1) {
        // 你的主循环代码
    }
}
```

**描述:**

*   **示例场景:**  这段代码展示了如何配置 MPU 来保护 Flash 和 SRAM。
*   **Flash 配置:** 将 Flash 区域配置为只读和禁止执行，防止意外写入和代码注入。
*   **SRAM 配置:** 将 SRAM 区域配置为可读写，允许程序存储数据。
*   **`ARM_MPU_EnableWithDefaults(false)`**:  禁用了特权默认访问权限，增强了安全性。

**总结:**

通过使用枚举类型、改进的宏定义和更高级的 API，可以使 `mpu_armv8.h` 头文件更易于使用、可读性更高，并且减少了出错的可能性。  示例代码展示了如何在实际项目中使用这些改进来配置 MPU。

**中文解释：**

这个改进后的 `mpu_armv8.h` 头文件旨在简化 Armv8-M MPU 的配置。  主要通过以下方式改进：

*   **类型安全：** 使用枚举类型来表示内存属性、共享属性和访问权限，避免直接使用数字，减少错误。
*   **易用性：** 提供 `ARM_MPU_ConfigureRegion` 函数，简化了 MPU 区域的配置过程。  `ARM_MPU_EnableWithDefaults` 函数简化了 MPU 的使能过程。
*   **可读性：** 代码结构更清晰，注释更详细，更容易理解和维护。

示例代码演示了如何使用这些改进后的 API 来配置 Flash 和 SRAM 区域的 MPU，从而提高系统的安全性和可靠性。  例如，可以将 Flash 区域配置为只读和禁止执行，防止恶意代码修改 Flash 中的程序。

希望这个改进的版本能帮助你更好地理解和使用 Armv8-M MPU。
