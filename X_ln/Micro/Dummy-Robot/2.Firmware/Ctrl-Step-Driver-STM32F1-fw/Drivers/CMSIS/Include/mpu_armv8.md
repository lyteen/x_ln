Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\mpu_armv8.h`

好的，重新开始。

这段代码定义了用于ARMv8-M架构的存储器保护单元（MPU）的CMSIS API。它允许配置内存区域的访问权限和属性，从而提高系统的安全性和可靠性。下面是代码关键部分的解释，以及使用示例：

**1. 定义常量（宏）:**

```c
/** \brief Attribute for device memory (outer only) */
#define ARM_MPU_ATTR_DEVICE                           ( 0U )

/** \brief Attribute for non-cacheable, normal memory */
#define ARM_MPU_ATTR_NON_CACHEABLE                    ( 4U )

/** \brief Attribute for normal memory (outer and inner)
* \param NT Non-Transient: Set to 1 for non-transient data.
* \param WB Write-Back: Set to 1 to use write-back update policy.
* \param RA Read Allocation: Set to 1 to use cache allocation on read miss.
* \param WA Write Allocation: Set to 1 to use cache allocation on write miss.
*/
#define ARM_MPU_ATTR_MEMORY_(NT, WB, RA, WA) \
  (((NT & 1U) << 3U) | ((WB & 1U) << 2U) | ((RA & 1U) << 1U) | (WA & 1U))

/** \brief Device memory type non Gathering, non Re-ordering, non Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_nGnRnE (0U)

/** \brief Device memory type non Gathering, non Re-ordering, Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_nGnRE  (1U)

/** \brief Device memory type non Gathering, Re-ordering, Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_nGRE   (2U)

/** \brief Device memory type Gathering, Re-ordering, Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_GRE    (3U)

/** \brief Memory Attribute
* \param O Outer memory attributes
* \param I O == ARM_MPU_ATTR_DEVICE: Device memory attributes, else: Inner memory attributes
*/
#define ARM_MPU_ATTR(O, I) (((O & 0xFU) << 4U) | (((O & 0xFU) != 0U) ? (I & 0xFU) : ((I & 0x3U) << 2U)))

/** \brief Normal memory non-shareable  */
#define ARM_MPU_SH_NON   (0U)

/** \brief Normal memory outer shareable  */
#define ARM_MPU_SH_OUTER (2U)

/** \brief Normal memory inner shareable  */
#define ARM_MPU_SH_INNER (3U)

/** \brief Memory access permissions
* \param RO Read-Only: Set to 1 for read-only memory.
* \param NP Non-Privileged: Set to 1 for non-privileged memory.
*/
#define ARM_MPU_AP_(RO, NP) (((RO & 1U) << 1U) | (NP & 1U))

/** \brief Region Base Address Register value
* \param BASE The base address bits [31:5] of a memory region. The value is zero extended. Effective address gets 32 byte aligned.
* \param SH Defines the Shareability domain for this memory region.
* \param RO Read-Only: Set to 1 for a read-only memory region.
* \param NP Non-Privileged: Set to 1 for a non-privileged memory region.
* \oaram XN eXecute Never: Set to 1 for a non-executable memory region.
*/
#define ARM_MPU_RBAR(BASE, SH, RO, NP, XN) \
  ((BASE & MPU_RBAR_BASE_Msk) | \
  ((SH << MPU_RBAR_SH_Pos) & MPU_RBAR_SH_Msk) | \
  ((ARM_MPU_AP_(RO, NP) << MPU_RBAR_AP_Pos) & MPU_RBAR_AP_Msk) | \
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

*   **描述:** 这些宏定义了用于配置MPU的各种属性和寄存器值的常量。例如：`ARM_MPU_ATTR_DEVICE`定义了设备内存的属性，`ARM_MPU_RBAR`和`ARM_MPU_RLAR`分别用于设置区域基地址寄存器和区域限制地址寄存器的值。
*   **用法:** 使用这些宏可以方便地设置内存区域的属性和访问权限。 例如，将一个区域设置为只读，非特权访问，不可执行，可以这样：

    ```c
    #define MY_REGION_BASE 0x20000000
    #define MY_REGION_LIMIT 0x20001FFF
    #define MY_REGION_NUMBER 0

    MPU->RNR = MY_REGION_NUMBER;
    MPU->RBAR = ARM_MPU_RBAR(MY_REGION_BASE, ARM_MPU_SH_NON, 1, 1, 1); // RO, NP, XN
    MPU->RLAR = ARM_MPU_RLAR(MY_REGION_LIMIT, 0); // 使用属性索引0
    ```

    这段代码将从`0x20000000`到`0x20001FFF`的内存区域设置为只读，非特权访问，不可执行。

**2. 数据结构定义:**

```c
/**
* Struct for a single MPU Region
*/
typedef struct {
  uint32_t RBAR;                   /*!< Region Base Address Register value */
  uint32_t RLAR;                   /*!< Region Limit Address Register value */
} ARM_MPU_Region_t;
```

*   **描述:** 定义了一个结构体`ARM_MPU_Region_t`，用于表示一个MPU区域的配置信息，包含RBAR（区域基地址寄存器）和RLAR（区域限制地址寄存器）的值。
*   **用法:** 可以使用此结构体来组织一组MPU区域的配置信息，然后使用`ARM_MPU_Load`函数一次性加载多个区域的配置。

**3. 使能和禁用MPU的函数:**

```c
/** Enable the MPU.
* \param MPU_Control Default access permissions for unconfigured regions.
*/
__STATIC_INLINE void ARM_MPU_Enable(uint32_t MPU_Control)
{
  __DSB();
  __ISB();
  MPU->CTRL = MPU_Control | MPU_CTRL_ENABLE_Msk;
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
  SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
#endif
}

/** Disable the MPU.
*/
__STATIC_INLINE void ARM_MPU_Disable(void)
{
  __DSB();
  __ISB();
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
  SCB->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk;
#endif
  MPU->CTRL  &= ~MPU_CTRL_ENABLE_Msk;
}
```

*   **描述:**  `ARM_MPU_Enable`函数用于使能MPU，并设置未配置区域的默认访问权限。`ARM_MPU_Disable`函数用于禁用MPU。`__DSB()`和`__ISB()`是数据同步屏障和指令同步屏障，用于确保MPU配置的正确应用。
*   **用法:** 在配置完MPU区域后，调用`ARM_MPU_Enable`函数启用MPU。在修改MPU配置前，建议先调用`ARM_MPU_Disable`禁用MPU。

**4. 设置内存属性的函数:**

```c
/** Set the memory attribute encoding to the given MPU.
* \param mpu Pointer to the MPU to be configured.
* \param idx The attribute index to be set [0-7]
* \param attr The attribute value to be set.
*/
__STATIC_INLINE void ARM_MPU_SetMemAttrEx(MPU_Type* mpu, uint8_t idx, uint8_t attr)
{
  const uint8_t reg = idx / 4U;
  const uint32_t pos = ((idx % 4U) * 8U);
  const uint32_t mask = 0xFFU << pos;
  
  if (reg >= (sizeof(mpu->MAIR) / sizeof(mpu->MAIR[0]))) {
    return; // invalid index
  }
  
  mpu->MAIR[reg] = ((mpu->MAIR[reg] & ~mask) | ((attr << pos) & mask));
}

/** Set the memory attribute encoding.
* \param idx The attribute index to be set [0-7]
* \param attr The attribute value to be set.
*/
__STATIC_INLINE void ARM_MPU_SetMemAttr(uint8_t idx, uint8_t attr)
{
  ARM_MPU_SetMemAttrEx(MPU, idx, attr);
}
```

*   **描述:** `ARM_MPU_SetMemAttrEx`和`ARM_MPU_SetMemAttr`函数用于设置内存属性索引对应的属性值。  `ARM_MPU_SetMemAttrEx` 允许指定要配置的MPU，而 `ARM_MPU_SetMemAttr` 默认配置主MPU。这些属性控制着缓存策略和内存行为。
*   **用法:** 在配置MPU区域前，需要先使用这些函数设置属性索引对应的属性值。例如：

    ```c
    ARM_MPU_SetMemAttr(0, ARM_MPU_ATTR(ARM_MPU_ATTR_NON_CACHEABLE, ARM_MPU_ATTR_NON_CACHEABLE)); // 属性索引0：Non-cacheable
    ARM_MPU_SetMemAttr(1, ARM_MPU_ATTR(ARM_MPU_ATTR_MEMORY_(1,1,1,1), ARM_MPU_ATTR_MEMORY_(1,1,1,1))); // 属性索引1：Normal memory，Write-Back, Read/Write Allocate
    ```

**5. 清除、配置和加载MPU区域的函数:**

```c
/** Clear and disable the given MPU region of the given MPU.
* \param mpu Pointer to MPU to be used.
* \param rnr Region number to be cleared.
*/
__STATIC_INLINE void ARM_MPU_ClrRegionEx(MPU_Type* mpu, uint32_t rnr)
{
  mpu->RNR = rnr;
  mpu->RLAR = 0U;
}

/** Clear and disable the given MPU region.
* \param rnr Region number to be cleared.
*/
__STATIC_INLINE void ARM_MPU_ClrRegion(uint32_t rnr)
{
  ARM_MPU_ClrRegionEx(MPU, rnr);
}

/** Configure the given MPU region of the given MPU.
* \param mpu Pointer to MPU to be used.
* \param rnr Region number to be configured.
* \param rbar Value for RBAR register.
* \param rlar Value for RLAR register.
*/   
__STATIC_INLINE void ARM_MPU_SetRegionEx(MPU_Type* mpu, uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  mpu->RNR = rnr;
  mpu->RBAR = rbar;
  mpu->RLAR = rlar;
}

/** Configure the given MPU region.
* \param rnr Region number to be configured.
* \param rbar Value for RBAR register.
* \param rlar Value for RLAR register.
*/   
__STATIC_INLINE void ARM_MPU_SetRegion(uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  ARM_MPU_SetRegionEx(MPU, rnr, rbar, rlar);
}

/** Memcopy with strictly ordered memory access, e.g. for register targets.
* \param dst Destination data is copied to.
* \param src Source data is copied from.
* \param len Amount of data words to be copied.
*/
__STATIC_INLINE void orderedCpy(volatile uint32_t* dst, const uint32_t* __RESTRICT src, uint32_t len)
{
  uint32_t i;
  for (i = 0U; i < len; ++i) 
  {
    dst[i] = src[i];
  }
}

/** Load the given number of MPU regions from a table to the given MPU.
* \param mpu Pointer to the MPU registers to be used.
* \param rnr First region number to be configured.
* \param table Pointer to the MPU configuration table.
* \param cnt Amount of regions to be configured.
*/
__STATIC_INLINE void ARM_MPU_LoadEx(MPU_Type* mpu, uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt) 
{
  const uint32_t rowWordSize = sizeof(ARM_MPU_Region_t)/4U;
  if (cnt == 1U) {
    mpu->RNR = rnr;
    orderedCpy(&(mpu->RBAR), &(table->RBAR), rowWordSize);
  } else {
    uint32_t rnrBase   = rnr & ~(MPU_TYPE_RALIASES-1U);
    uint32_t rnrOffset = rnr % MPU_TYPE_RALIASES;
    
    mpu->RNR = rnrBase;
    while ((rnrOffset + cnt) > MPU_TYPE_RALIASES) {
      uint32_t c = MPU_TYPE_RALIASES - rnrOffset;
      orderedCpy(&(mpu->RBAR)+(rnrOffset*2U), &(table->RBAR), c*rowWordSize);
      table += c;
      cnt -= c;
      rnrOffset = 0U;
      rnrBase += MPU_TYPE_RALIASES;
      mpu->RNR = rnrBase;
    }
    
    orderedCpy(&(mpu->RBAR)+(rnrOffset*2U), &(table->RBAR), cnt*rowWordSize);
  }
}

/** Load the given number of MPU regions from a table.
* \param rnr First region number to be configured.
* \param table Pointer to the MPU configuration table.
* \param cnt Amount of regions to be configured.
*/
__STATIC_INLINE void ARM_MPU_Load(uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt) 
{
  ARM_MPU_LoadEx(MPU, rnr, table, cnt);
}
```

*   **描述:**
    *   `ARM_MPU_ClrRegionEx`和`ARM_MPU_ClrRegion`函数用于清除并禁用指定的MPU区域。
    *   `ARM_MPU_SetRegionEx`和`ARM_MPU_SetRegion`函数用于配置指定的MPU区域的RBAR和RLAR寄存器值。
    *   `orderedCpy` 函数用于进行严格有序的内存拷贝，这对于访问寄存器等目标特别重要，以保证配置的正确性。
    *   `ARM_MPU_LoadEx`和`ARM_MPU_Load`函数用于从一个配置表加载多个MPU区域的配置。
*   **用法:**
    1.  **清除区域:**  在重新配置一个区域之前，通常先清除它：

        ```c
        ARM_MPU_ClrRegion(0); // 清除区域0
        ```
    2.  **配置区域:**  使用`ARM_MPU_SetRegion` 设置单个区域，或者使用`ARM_MPU_Load`一次性设置多个区域。

        ```c
        // 设置单个区域
        ARM_MPU_SetRegion(0,
                          ARM_MPU_RBAR(0x20000000, ARM_MPU_SH_NON, 0, 0, 0), // Base address, Shareability, RO/NP/XN
                          ARM_MPU_RLAR(0x20001FFF, 1)); // Limit address, Attribute index
        ```
    3.  **加载多个区域:**

        ```c
        ARM_MPU_Region_t mpu_config[] = {
            {ARM_MPU_RBAR(0x20000000, ARM_MPU_SH_NON, 0, 0, 0), ARM_MPU_RLAR(0x20001FFF, 1)}, // 区域0
            {ARM_MPU_RBAR(0x20002000, ARM_MPU_SH_NON, 1, 1, 1), ARM_MPU_RLAR(0x20003FFF, 0)}  // 区域1
        };

        ARM_MPU_Load(0, mpu_config, sizeof(mpu_config) / sizeof(ARM_MPU_Region_t)); // 从区域0开始加载，加载所有区域
        ```

**6. 非安全MPU (MPU_NS) 函数:**

代码中也包含了针对非安全MPU (MPU_NS) 的函数，例如`ARM_MPU_Enable_NS`、`ARM_MPU_Disable_NS`、`ARM_MPU_SetMemAttr_NS`、`ARM_MPU_ClrRegion_NS`、`ARM_MPU_SetRegion_NS`、`ARM_MPU_Load_NS`。这些函数的功能与安全MPU对应的函数类似，但作用于非安全MPU，用于配置非安全区域的访问权限。

**完整示例:**

```c
#include "stm32f4xx.h" // 替换为你的 MCU 头文件
#include "mpu_armv8.h"

#define FLASH_START 0x08000000
#define FLASH_END   0x080FFFFF

#define SRAM_START  0x20000000
#define SRAM_END    0x2001FFFF

int main(void) {
  // 1. 禁用MPU
  ARM_MPU_Disable();

  // 2. 设置内存属性
  ARM_MPU_SetMemAttr(0, ARM_MPU_ATTR(ARM_MPU_ATTR_DEVICE, ARM_MPU_ATTR_DEVICE)); // 属性索引0: Device memory
  ARM_MPU_SetMemAttr(1, ARM_MPU_ATTR(ARM_MPU_ATTR_NON_CACHEABLE, ARM_MPU_ATTR_NON_CACHEABLE)); // 属性索引1: Non-cacheable
  ARM_MPU_SetMemAttr(2, ARM_MPU_ATTR(ARM_MPU_ATTR_MEMORY_(1,1,1,1), ARM_MPU_ATTR_MEMORY_(1,1,1,1))); // 属性索引2: Normal memory, Write-Back, Read/Write Allocate

  // 3. 配置MPU区域
  ARM_MPU_Region_t mpu_config[] = {
    {ARM_MPU_RBAR(FLASH_START, ARM_MPU_SH_NON, 1, 1, 1), ARM_MPU_RLAR(FLASH_END, 2)}, // 区域0: Flash, RO, NP, XN, Normal memory
    {ARM_MPU_RBAR(SRAM_START, ARM_MPU_SH_NON, 0, 0, 0), ARM_MPU_RLAR(SRAM_END, 2)}    // 区域1: SRAM, RW, Privileged, Executable, Normal memory
  };
  ARM_MPU_Load(0, mpu_config, sizeof(mpu_config) / sizeof(ARM_MPU_Region_t));

  // 4. 使能MPU
  ARM_MPU_Enable(0); // 使能MPU，默认权限为No access

  while (1) {
    // 你的代码
  }
}
```

**总结:**

这段代码提供了一套方便的API，用于配置ARMv8-M架构的MPU。通过使用这些API，可以定义内存区域的访问权限和属性，从而提高系统的安全性、可靠性和性能。理解这些API以及它们如何与MPU寄存器交互，对于开发安全可靠的嵌入式系统至关重要。
