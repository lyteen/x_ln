Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\mpu_armv7.h`

好的，我们重新开始，针对 `mpu_armv7.h` 头文件，我将提供改进建议，并附带中文描述和简单示例。

**目标:**

*   **更强的类型安全:** 使用枚举 (enum) 替代 `#define`，减少类型错误的可能性。
*   **更清晰的 API 设计:**  使用结构体来组合 MPU 配置信息，提高代码可读性。
*   **更易用的接口:**  提供更高级别的函数，简化 MPU 配置过程。
*   **更好的可维护性:**  代码结构更清晰，方便修改和扩展。

**1. Region Size 定义 (区域大小定义):**

```c
typedef enum {
    ARM_MPU_REGION_SIZE_32B      = 0x04U,  /*!< MPU 区域大小 32 字节 */
    ARM_MPU_REGION_SIZE_64B      = 0x05U,  /*!< MPU 区域大小 64 字节 */
    ARM_MPU_REGION_SIZE_128B     = 0x06U,  /*!< MPU 区域大小 128 字节 */
    ARM_MPU_REGION_SIZE_256B     = 0x07U,  /*!< MPU 区域大小 256 字节 */
    ARM_MPU_REGION_SIZE_512B     = 0x08U,  /*!< MPU 区域大小 512 字节 */
    ARM_MPU_REGION_SIZE_1KB      = 0x09U,  /*!< MPU 区域大小 1 千字节 */
    ARM_MPU_REGION_SIZE_2KB      = 0x0AU,  /*!< MPU 区域大小 2 千字节 */
    ARM_MPU_REGION_SIZE_4KB      = 0x0BU,  /*!< MPU 区域大小 4 千字节 */
    ARM_MPU_REGION_SIZE_8KB      = 0x0CU,  /*!< MPU 区域大小 8 千字节 */
    ARM_MPU_REGION_SIZE_16KB     = 0x0DU,  /*!< MPU 区域大小 16 千字节 */
    ARM_MPU_REGION_SIZE_32KB     = 0x0EU,  /*!< MPU 区域大小 32 千字节 */
    ARM_MPU_REGION_SIZE_64KB     = 0x0FU,  /*!< MPU 区域大小 64 千字节 */
    ARM_MPU_REGION_SIZE_128KB    = 0x10U,  /*!< MPU 区域大小 128 千字节 */
    ARM_MPU_REGION_SIZE_256KB    = 0x11U,  /*!< MPU 区域大小 256 千字节 */
    ARM_MPU_REGION_SIZE_512KB    = 0x12U,  /*!< MPU 区域大小 512 千字节 */
    ARM_MPU_REGION_SIZE_1MB      = 0x13U,  /*!< MPU 区域大小 1 兆字节 */
    ARM_MPU_REGION_SIZE_2MB      = 0x14U,  /*!< MPU 区域大小 2 兆字节 */
    ARM_MPU_REGION_SIZE_4MB      = 0x15U,  /*!< MPU 区域大小 4 兆字节 */
    ARM_MPU_REGION_SIZE_8MB      = 0x16U,  /*!< MPU 区域大小 8 兆字节 */
    ARM_MPU_REGION_SIZE_16MB     = 0x17U,  /*!< MPU 区域大小 16 兆字节 */
    ARM_MPU_REGION_SIZE_32MB     = 0x18U,  /*!< MPU 区域大小 32 兆字节 */
    ARM_MPU_REGION_SIZE_64MB     = 0x19U,  /*!< MPU 区域大小 64 兆字节 */
    ARM_MPU_REGION_SIZE_128MB    = 0x1AU,  /*!< MPU 区域大小 128 兆字节 */
    ARM_MPU_REGION_SIZE_256MB    = 0x1BU,  /*!< MPU 区域大小 256 兆字节 */
    ARM_MPU_REGION_SIZE_512MB    = 0x1CU,  /*!< MPU 区域大小 512 兆字节 */
    ARM_MPU_REGION_SIZE_1GB      = 0x1DU,  /*!< MPU 区域大小 1 吉字节 */
    ARM_MPU_REGION_SIZE_2GB      = 0x1EU,  /*!< MPU 区域大小 2 吉字节 */
    ARM_MPU_REGION_SIZE_4GB      = 0x1FU   /*!< MPU 区域大小 4 吉字节 */
} ARM_MPU_RegionSize_t;
```

**描述:** 使用 `enum` 类型 `ARM_MPU_RegionSize_t` 来定义区域大小，增加了类型安全性。  以前使用 `#define`，编译器不会检查类型，可能导致错误。 `enum` 可以确保你只能使用预定义的值。

**2. Access Permission 定义 (访问权限定义):**

```c
typedef enum {
    ARM_MPU_AP_NONE = 0U,   /*!< MPU 访问权限：无访问权限 */
    ARM_MPU_AP_PRIV = 1U,   /*!< MPU 访问权限：仅特权访问 */
    ARM_MPU_AP_URO  = 2U,   /*!< MPU 访问权限：非特权只读 */
    ARM_MPU_AP_FULL = 3U,   /*!< MPU 访问权限：完全访问 */
    ARM_MPU_AP_PRO  = 5U,   /*!< MPU 访问权限：特权只读 */
    ARM_MPU_AP_RO   = 6U    /*!< MPU 访问权限：只读访问 */
} ARM_MPU_AccessPermission_t;
```

**描述:** 同样，使用 `enum` 来定义访问权限，提高类型安全性和可读性。

**3. Access Attribute 定义 (访问属性定义):**

```c
typedef enum {
    ARM_MPU_CACHEP_NOCACHE = 0U,  /*!< MPU 内存访问属性：不可缓存 */
    ARM_MPU_CACHEP_WB_WRA = 1U,   /*!< MPU 内存访问属性：写回，写和读分配 */
    ARM_MPU_CACHEP_WT_NWA = 2U,   /*!< MPU 内存访问属性：写通，无写分配 */
    ARM_MPU_CACHEP_WB_NWA = 3U    /*!< MPU 内存访问属性：写回，无写分配 */
} ARM_MPU_CachePolicy_t;

#define ARM_MPU_ACCESS_ORDERED ARM_MPU_ACCESS_(0U, 1U, 0U, 0U)
#define ARM_MPU_ACCESS_DEVICE(IsShareable) ((IsShareable) ? ARM_MPU_ACCESS_(0U, 1U, 0U, 1U) : ARM_MPU_ACCESS_(2U, 0U, 0U, 0U))
#define ARM_MPU_ACCESS_NORMAL(OuterCp, InnerCp, IsShareable) ARM_MPU_ACCESS_((4U | (OuterCp)), IsShareable, ((InnerCp) & 2U), ((InnerCp) & 1U))

```

**描述:**  定义了缓存策略的 `enum`，并保留了之前的宏定义，方便使用。

**4. MPU Region 配置结构体 (MPU 区域配置结构体):**

```c
typedef struct {
    uint32_t base_address;         /*!< 区域基地址 */
    ARM_MPU_RegionSize_t size;      /*!< 区域大小 */
    ARM_MPU_AccessPermission_t access_permission; /*!< 访问权限 */
    uint32_t  type_extension;     /*!<  类型扩展字段  */
    uint32_t  is_shareable;        /*!<  区域是否可共享  */
    uint32_t  is_cacheable;        /*!<  区域是否可缓存  */
    uint32_t  is_bufferable;       /*!<  区域是否可缓冲  */
    uint32_t sub_region_disable;  /*!< 子区域禁用位掩码 */
    uint32_t disable_execute;      /*!< 禁止执行标志 */
    uint32_t region_number;        /*!< 区域编号 (0-15) */
} ARM_MPU_RegionConfig_t;
```

**描述:**  使用结构体 `ARM_MPU_RegionConfig_t` 来存储单个 MPU 区域的所有配置信息。  这使得配置更清晰，更易于传递和管理。  不再需要手动构建 `RBAR` 和 `RASR`，而是将所有信息放在一个结构体中。

**5. 辅助函数 (Helper Functions):**

```c
/**
 * @brief  配置 MPU 区域.
 * @param  config: 指向 MPU 区域配置结构体的指针.
 * @retval None
 */
__STATIC_INLINE void ARM_MPU_ConfigureRegion(const ARM_MPU_RegionConfig_t *config) {
  //检查区域编号是否有效
  if (config->region_number > 15) {
      return; // 或者返回错误码
  }

  // 构建 RBAR 寄存器值
  uint32_t rbar = (config->base_address & MPU_RBAR_ADDR_Msk) |
                  ((config->region_number << MPU_RBAR_REGION_Pos) & MPU_RBAR_REGION_Msk) |
                  MPU_RBAR_VALID_Msk;

  // 构建 RASR 寄存器值
  uint32_t rasr = (((config->disable_execute) << MPU_RASR_XN_Pos) & MPU_RASR_XN_Msk) |
                  (((config->access_permission) << MPU_RASR_AP_Pos) & MPU_RASR_AP_Msk) |
                  (((config->type_extension) << MPU_RASR_TEX_Pos) & MPU_RASR_TEX_Msk) |
                  (((config->is_shareable) << MPU_RASR_S_Pos) & MPU_RASR_S_Msk) |
                  (((config->is_cacheable) << MPU_RASR_C_Pos) & MPU_RASR_C_Msk) |
                  (((config->is_bufferable) << MPU_RASR_B_Pos) & MPU_RASR_B_Msk) |
                  ((config->sub_region_disable << MPU_RASR_SRD_Pos) & MPU_RASR_SRD_Msk) |
                  ((config->size << MPU_RASR_SIZE_Pos) & MPU_RASR_SIZE_Msk) |
                  MPU_RASR_ENABLE_Msk; //使能区域

  // 配置 MPU 寄存器
  MPU->RNR = config->region_number;
  MPU->RBAR = rbar;
  MPU->RASR = rasr;
}


/**
 * @brief  使能 MPU
 * @param  default_access_permission:  未配置区域的默认访问权限
 * @retval None
 */
__STATIC_INLINE void ARM_MPU_EnableEx(uint32_t default_access_permission) {
    __DSB();
    __ISB();
    MPU->CTRL = default_access_permission | MPU_CTRL_ENABLE_Msk | MPU_CTRL_HFNMIENA_Msk; // 使能 MPU，允许在 HardFault 和 NMI 中使用 MPU
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
    SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk; // 使能内存错误处理
#endif
}


#define MPU_DEFAULT_DISABLE_EXECUTE_ACCESS  0x01U  // 定义默认禁止执行的宏
#define MPU_DEFAULT_PRIVILEGED_READ_ONLY    0x05U  // 定义默认特权只读的宏


/**
 * @brief  禁用 MPU
 * @retval None
 */
__STATIC_INLINE void ARM_MPU_DisableEx(void) {
    __DSB();
    __ISB();
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
    SCB->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk;  // 禁用内存错误处理
#endif
    MPU->CTRL &= ~MPU_CTRL_ENABLE_Msk;          // 禁用 MPU
}
```

**描述:**

*   `ARM_MPU_ConfigureRegion`:  接受一个 `ARM_MPU_RegionConfig_t` 结构体指针，并根据结构体中的信息配置 MPU 区域。  它负责构建 `RBAR` 和 `RASR` 寄存器的值，并将其写入 MPU 硬件寄存器。 这样做减少了出错的可能性，简化了代码，可读性更好。
*   `ARM_MPU_EnableEx`: 接受一个默认权限参数，并使能MPU。
*   `ARM_MPU_DisableEx`:  禁用 MPU。

**6. 示例代码 (Demo):**

```c
#include "stm32f4xx.h" // 替换成你的 MCU 头文件
#include "mpu_armv7.h"

int main(void) {
    // 1. 定义 MPU 区域配置
    ARM_MPU_RegionConfig_t region_config;

    region_config.region_number = 0;  // 区域 0
    region_config.base_address = 0x20000000; // SRAM 开始地址
    region_config.size = ARM_MPU_REGION_SIZE_32KB; // 32KB 区域
    region_config.access_permission = ARM_MPU_AP_FULL; // 完全访问权限 (读写)
    region_config.type_extension = 0;
    region_config.is_shareable = 0;
    region_config.is_cacheable = 0;
    region_config.is_bufferable = 0;
    region_config.sub_region_disable = 0;
    region_config.disable_execute = 1; // 禁止在此区域执行代码

    // 2. 配置 MPU 区域
    ARM_MPU_ConfigureRegion(&region_config);

    // 3. 使能 MPU (设置默认权限为仅特权访问)
    ARM_MPU_EnableEx(MPU_DEFAULT_PRIVILEGED_READ_ONLY);

    while (1) {
        // 你的代码
    }
}
```

**描述:**

*   这段代码演示了如何使用新的 API 配置 MPU 区域。
*   它首先定义一个 `ARM_MPU_RegionConfig_t` 类型的变量，并填充区域配置信息 (基地址、大小、访问权限等)。
*   然后，调用 `ARM_MPU_ConfigureRegion` 函数来配置 MPU 区域。
*   最后，调用 `ARM_MPU_EnableEx` 函数来启用 MPU。

**完整的 `mpu_armv7.h` 文件 (改进后的版本):**

```c
#ifndef ARM_MPU_ARMV7_H
#define ARM_MPU_ARMV7_H

#include <stdint.h>

// 确保包含 MPU 相关的寄存器定义 (根据你的 MCU)
#include "core_cm4.h" // 这是一个例子，你需要替换成你的 MCU 的 CMSIS 头文件

// 1. Region Size 定义
typedef enum {
    ARM_MPU_REGION_SIZE_32B      = 0x04U,  /*!< MPU 区域大小 32 字节 */
    ARM_MPU_REGION_SIZE_64B      = 0x05U,  /*!< MPU 区域大小 64 字节 */
    ARM_MPU_REGION_SIZE_128B     = 0x06U,  /*!< MPU 区域大小 128 字节 */
    ARM_MPU_REGION_SIZE_256B     = 0x07U,  /*!< MPU 区域大小 256 字节 */
    ARM_MPU_REGION_SIZE_512B     = 0x08U,  /*!< MPU 区域大小 512 字节 */
    ARM_MPU_REGION_SIZE_1KB      = 0x09U,  /*!< MPU 区域大小 1 千字节 */
    ARM_MPU_REGION_SIZE_2KB      = 0x0AU,  /*!< MPU 区域大小 2 千字节 */
    ARM_MPU_REGION_SIZE_4KB      = 0x0BU,  /*!< MPU 区域大小 4 千字节 */
    ARM_MPU_REGION_SIZE_8KB      = 0x0CU,  /*!< MPU 区域大小 8 千字节 */
    ARM_MPU_REGION_SIZE_16KB     = 0x0DU,  /*!< MPU 区域大小 16 千字节 */
    ARM_MPU_REGION_SIZE_32KB     = 0x0EU,  /*!< MPU 区域大小 32 千字节 */
    ARM_MPU_REGION_SIZE_64KB     = 0x0FU,  /*!< MPU 区域大小 64 千字节 */
    ARM_MPU_REGION_SIZE_128KB    = 0x10U,  /*!< MPU 区域大小 128 千字节 */
    ARM_MPU_REGION_SIZE_256KB    = 0x11U,  /*!< MPU 区域大小 256 千字节 */
    ARM_MPU_REGION_SIZE_512KB    = 0x12U,  /*!< MPU 区域大小 512 千字节 */
    ARM_MPU_REGION_SIZE_1MB      = 0x13U,  /*!< MPU 区域大小 1 兆字节 */
    ARM_MPU_REGION_SIZE_2MB      = 0x14U,  /*!< MPU 区域大小 2 兆字节 */
    ARM_MPU_REGION_SIZE_4MB      = 0x15U,  /*!< MPU 区域大小 4 兆字节 */
    ARM_MPU_REGION_SIZE_8MB      = 0x16U,  /*!< MPU 区域大小 8 兆字节 */
    ARM_MPU_REGION_SIZE_16MB     = 0x17U,  /*!< MPU 区域大小 16 兆字节 */
    ARM_MPU_REGION_SIZE_32MB     = 0x18U,  /*!< MPU 区域大小 32 兆字节 */
    ARM_MPU_REGION_SIZE_64MB     = 0x19U,  /*!< MPU 区域大小 64 兆字节 */
    ARM_MPU_REGION_SIZE_128MB    = 0x1AU,  /*!< MPU 区域大小 128 兆字节 */
    ARM_MPU_REGION_SIZE_256MB    = 0x1BU,  /*!< MPU 区域大小 256 兆字节 */
    ARM_MPU_REGION_SIZE_512MB    = 0x1CU,  /*!< MPU 区域大小 512 兆字节 */
    ARM_MPU_REGION_SIZE_1GB      = 0x1DU,  /*!< MPU 区域大小 1 吉字节 */
    ARM_MPU_REGION_SIZE_2GB      = 0x1EU,  /*!< MPU 区域大小 2 吉字节 */
    ARM_MPU_REGION_SIZE_4GB      = 0x1FU   /*!< MPU 区域大小 4 吉字节 */
} ARM_MPU_RegionSize_t;

// 2. Access Permission 定义
typedef enum {
    ARM_MPU_AP_NONE = 0U,   /*!< MPU 访问权限：无访问权限 */
    ARM_MPU_AP_PRIV = 1U,   /*!< MPU 访问权限：仅特权访问 */
    ARM_MPU_AP_URO  = 2U,   /*!< MPU 访问权限：非特权只读 */
    ARM_MPU_AP_FULL = 3U,   /*!< MPU 访问权限：完全访问 */
    ARM_MPU_AP_PRO  = 5U,   /*!< MPU 访问权限：特权只读 */
    ARM_MPU_AP_RO   = 6U    /*!< MPU 访问权限：只读访问 */
} ARM_MPU_AccessPermission_t;

// 3. Cache Policy 定义
typedef enum {
    ARM_MPU_CACHEP_NOCACHE = 0U,  /*!< MPU 内存访问属性：不可缓存 */
    ARM_MPU_CACHEP_WB_WRA = 1U,   /*!< MPU 内存访问属性：写回，写和读分配 */
    ARM_MPU_CACHEP_WT_NWA = 2U,   /*!< MPU 内存访问属性：写通，无写分配 */
    ARM_MPU_CACHEP_WB_NWA = 3U    /*!< MPU 内存访问属性：写回，无写分配 */
} ARM_MPU_CachePolicy_t;

//宏定义方便使用
#define ARM_MPU_ACCESS_ORDERED ARM_MPU_ACCESS_(0U, 1U, 0U, 0U)
#define ARM_MPU_ACCESS_DEVICE(IsShareable) ((IsShareable) ? ARM_MPU_ACCESS_(0U, 1U, 0U, 1U) : ARM_MPU_ACCESS_(2U, 0U, 0U, 0U))
#define ARM_MPU_ACCESS_NORMAL(OuterCp, InnerCp, IsShareable) ARM_MPU_ACCESS_((4U | (OuterCp)), IsShareable, ((InnerCp) & 2U), ((InnerCp) & 1U))

#define ARM_MPU_ACCESS_(TypeExtField, IsShareable, IsCacheable, IsBufferable)   \
  ((((TypeExtField ) << MPU_RASR_TEX_Pos) & MPU_RASR_TEX_Msk)                 | \
   (((IsShareable ) << MPU_RASR_S_Pos) & MPU_RASR_S_Msk)                      | \
   (((IsCacheable ) << MPU_RASR_C_Pos) & MPU_RASR_C_Msk)                      | \
   (((IsBufferable ) << MPU_RASR_B_Pos) & MPU_RASR_B_Msk))

// 4. MPU Region 配置结构体
typedef struct {
    uint32_t base_address;         /*!< 区域基地址 */
    ARM_MPU_RegionSize_t size;      /*!< 区域大小 */
    ARM_MPU_AccessPermission_t access_permission; /*!< 访问权限 */
    uint32_t  type_extension;     /*!<  类型扩展字段  */
    uint32_t  is_shareable;        /*!<  区域是否可共享  */
    uint32_t  is_cacheable;        /*!<  区域是否可缓存  */
    uint32_t  is_bufferable;       /*!<  区域是否可缓冲  */
    uint32_t sub_region_disable;  /*!< 子区域禁用位掩码 */
    uint32_t disable_execute;      /*!< 禁止执行标志 */
    uint32_t region_number;        /*!< 区域编号 (0-15) */
} ARM_MPU_RegionConfig_t;

// 5. 辅助函数
/**
 * @brief  配置 MPU 区域.
 * @param  config: 指向 MPU 区域配置结构体的指针.
 * @retval None
 */
__STATIC_INLINE void ARM_MPU_ConfigureRegion(const ARM_MPU_RegionConfig_t *config) {
  //检查区域编号是否有效
  if (config->region_number > 15) {
      return; // 或者返回错误码
  }

  // 构建 RBAR 寄存器值
  uint32_t rbar = (config->base_address & MPU_RBAR_ADDR_Msk) |
                  ((config->region_number << MPU_RBAR_REGION_Pos) & MPU_RBAR_REGION_Msk) |
                  MPU_RBAR_VALID_Msk;

  // 构建 RASR 寄存器值
  uint32_t rasr = (((config->disable_execute) << MPU_RASR_XN_Pos) & MPU_RASR_XN_Msk) |
                  (((config->access_permission) << MPU_RASR_AP_Pos) & MPU_RASR_AP_Msk) |
                  (((config->type_extension) << MPU_RASR_TEX_Pos) & MPU_RASR_TEX_Msk) |
                  (((config->is_shareable) << MPU_RASR_S_Pos) & MPU_RASR_S_Msk) |
                  (((config->is_cacheable) << MPU_RASR_C_Pos) & MPU_RASR_C_Msk) |
                  (((config->is_bufferable) << MPU_RASR_B_Pos) & MPU_RASR_B_Msk) |
                  ((config->sub_region_disable << MPU_RASR_SRD_Pos) & MPU_RASR_SRD_Msk) |
                  ((config->size << MPU_RASR_SIZE_Pos) & MPU_RASR_SIZE_Msk) |
                  MPU_RASR_ENABLE_Msk; //使能区域

  // 配置 MPU 寄存器
  MPU->RNR = config->region_number;
  MPU->RBAR = rbar;
  MPU->RASR = rasr;
}


/**
 * @brief  使能 MPU
 * @param  default_access_permission:  未配置区域的默认访问权限
 * @retval None
 */
__STATIC_INLINE void ARM_MPU_EnableEx(uint32_t default_access_permission) {
    __DSB();
    __ISB();
    MPU->CTRL = default_access_permission | MPU_CTRL_ENABLE_Msk | MPU_CTRL_HFNMIENA_Msk; // 使能 MPU，允许在 HardFault 和 NMI 中使用 MPU
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
    SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk; // 使能内存错误处理
#endif
}


#define MPU_DEFAULT_DISABLE_EXECUTE_ACCESS  0x01U  // 定义默认禁止执行的宏
#define MPU_DEFAULT_PRIVILEGED_READ_ONLY    0x05U  // 定义默认特权只读的宏


/**
 * @brief  禁用 MPU
 * @retval None
 */
__STATIC_INLINE void ARM_MPU_DisableEx(void) {
    __DSB();
    __ISB();
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
    SCB->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk;  // 禁用内存错误处理
#endif
    MPU->CTRL &= ~MPU_CTRL_ENABLE_Msk;          // 禁用 MPU
}


#endif // ARM_MPU_ARMV7_H
```

**总结:**

这些改进使得 MPU 配置更加安全、清晰和易于管理。 使用 `enum` 增加了类型安全性，使用结构体 `ARM_MPU_RegionConfig_t` 组织配置信息，并提供辅助函数来简化配置过程。  示例代码演示了如何使用这些新的 API 来配置 MPU 区域。  总的来说，这是一种更现代、更易于维护的 MPU 驱动程序设计方法。
