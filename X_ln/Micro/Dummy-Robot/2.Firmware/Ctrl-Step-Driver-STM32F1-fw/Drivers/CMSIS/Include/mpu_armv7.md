Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\mpu_armv7.h`

好的，我们重新开始。这次我会更仔细地解释代码，并提供更清晰的示例。

**1. 区域大小定义 (Region Size Definitions)**

```c
#define ARM_MPU_REGION_SIZE_32B      ((uint8_t)0x04U) ///!< MPU Region Size 32 Bytes
#define ARM_MPU_REGION_SIZE_64B      ((uint8_t)0x05U) ///!< MPU Region Size 64 Bytes
// ... (其他区域大小定义)
#define ARM_MPU_REGION_SIZE_4GB      ((uint8_t)0x1FU) ///!< MPU Region Size 4 GBytes
```

**描述:**  这部分定义了一系列宏，用于指定 MPU 区域的大小。每个宏都代表一个特定的尺寸，从 32 字节到 4GB。 这些值实际上代表了在RASR寄存器中编码区域大小的位域的值。

**如何使用:**  在配置 MPU 区域时，可以使用这些宏来设置区域的大小。 例如，`ARM_MPU_REGION_SIZE_4KB` 表示将区域大小设置为 4KB。

**例子：**
```c
#include "stm32f4xx.h" // 假设使用 STM32F4 系列单片机，需要包含相应的头文件
#include "mpu_armv7.h"

int main(void) {
  // ... 初始化代码

  // 配置 MPU 区域 0，地址为 0x20000000，大小为 4KB
  uint32_t rbar = ARM_MPU_RBAR(0, 0x20000000); // 区域 0，基地址 0x20000000
  uint32_t rasr = ARM_MPU_RASR(1, ARM_MPU_AP_FULL, 0, 0, 0, 0, 0, ARM_MPU_REGION_SIZE_4KB); // 禁止执行，完全访问权限，4KB 大小

  ARM_MPU_SetRegion(rbar, rasr);

  // 启用 MPU
  ARM_MPU_Enable(MPU_CTRL_PRIVDEFENA_Msk);  // 允许在未配置区域的默认访问权限

  while (1) {
    // ... 你的程序逻辑
  }
}
```

**2. 访问权限定义 (Access Permission Definitions)**

```c
#define ARM_MPU_AP_NONE 0U ///!< MPU Access Permission no access
#define ARM_MPU_AP_PRIV 1U ///!< MPU Access Permission privileged access only
#define ARM_MPU_AP_URO  2U ///!< MPU Access Permission unprivileged access read-only
#define ARM_MPU_AP_FULL 3U ///!< MPU Access Permission full access
#define ARM_MPU_AP_PRO  5U ///!< MPU Access Permission privileged access read-only
#define ARM_MPU_AP_RO   6U ///!< MPU Access Permission read-only access
```

**描述:**  这些宏定义了 MPU 区域的访问权限。 它们控制着特权模式和非特权模式下对内存区域的读取和写入访问。

*   `ARM_MPU_AP_NONE`: 禁止所有访问。
*   `ARM_MPU_AP_PRIV`: 只有特权模式可以访问。
*   `ARM_MPU_AP_URO`: 非特权模式只能读取，特权模式可以读写。
*   `ARM_MPU_AP_FULL`: 允许所有访问 (读写)。
*   `ARM_MPU_AP_PRO`: 非特权模式禁止访问，特权模式只读。
*   `ARM_MPU_AP_RO`: 允许所有读取，禁止所有写入。

**如何使用:**  在配置 MPU 区域时，选择适当的访问权限宏来限制对特定内存区域的访问。

**3.  `ARM_MPU_RBAR` 宏**

```c
#define ARM_MPU_RBAR(Region, BaseAddress) \
  (((BaseAddress) & MPU_RBAR_ADDR_Msk) |  \
   ((Region) & MPU_RBAR_REGION_Msk)    |  \
   (MPU_RBAR_VALID_Msk))
```

**描述:**  这个宏用于构造 MPU 区域基地址寄存器 (RBAR) 的值。它将基地址、区域号和有效位组合在一起。

*   `Region`:  要配置的区域的编号 (0-15)。
*   `BaseAddress`: 区域的起始地址。

**如何使用:**  使用这个宏来计算 RBAR 寄存器的值，然后将其写入 MPU 硬件。

**4. `ARM_MPU_ACCESS_` 宏**

```c
#define ARM_MPU_ACCESS_(TypeExtField, IsShareable, IsCacheable, IsBufferable)   \
  ((((TypeExtField ) << MPU_RASR_TEX_Pos) & MPU_RASR_TEX_Msk)                 | \
   (((IsShareable ) << MPU_RASR_S_Pos) & MPU_RASR_S_Msk)                      | \
   (((IsCacheable ) << MPU_RASR_C_Pos) & MPU_RASR_C_Msk)                      | \
   (((IsBufferable ) << MPU_RASR_B_Pos) & MPU_RASR_B_Msk))
```

**描述:** 此宏定义了MPU内存访问属性。它允许配置诸如是否可共享、是否可缓存以及是否可缓冲等属性。
*   `TypeExtField`:  类型扩展字段，允许配置内存访问类型 (例如，强顺序、外设)。
*   `IsShareable`:  区域是否可以在多个总线主设备之间共享。
*   `IsCacheable`:  区域是否可以被缓存 (即，其值是否可以保存在缓存中)。
*   `IsBufferable`:  区域是否可以被缓冲 (即，使用写回缓存)。 可缓存但不可缓冲的区域使用直写策略。

**如何使用:**  使用此宏定义访问属性。

**5. `ARM_MPU_RASR_EX` 宏**

```c
#define ARM_MPU_RASR_EX(DisableExec, AccessPermission, AccessAttributes, SubRegionDisable, Size)      \
  ((((DisableExec ) << MPU_RASR_XN_Pos) & MPU_RASR_XN_Msk)                                          | \
   (((AccessPermission) << MPU_RASR_AP_Pos) & MPU_RASR_AP_Msk)                                      | \
   (((AccessAttributes) ) & (MPU_RASR_TEX_Msk | MPU_RASR_S_Msk | MPU_RASR_C_Msk | MPU_RASR_B_Msk)))
```

**描述:**  这个宏用于构造 MPU 区域属性和大小寄存器 (RASR) 的值。  它包含了是否禁止执行指令、访问权限、内存访问属性、子区域禁用和区域大小等信息。

*   `DisableExec`:  指令访问禁止位。 1 = 禁止指令读取。
*   `AccessPermission`: 数据访问权限。
*   `AccessAttributes`: 内存访问属性，使用 `ARM_MPU_ACCESS_` 定义。
*   `SubRegionDisable`: 子区域禁用字段。
*   `Size`:  区域大小，使用 `ARM_MPU_REGION_SIZE_` 宏定义。

**如何使用:**  使用这个宏来计算 RASR 寄存器的值，然后将其写入 MPU 硬件。

**6. `ARM_MPU_RASR` 宏**

```c
#define ARM_MPU_RASR(DisableExec, AccessPermission, TypeExtField, IsShareable, IsCacheable, IsBufferable, SubRegionDisable, Size) \
  ARM_MPU_RASR_EX(DisableExec, AccessPermission, ARM_MPU_ACCESS_(TypeExtField, IsShareable, IsCacheable, IsBufferable), SubRegionDisable, Size)
```

**描述:**
这个宏是`ARM_MPU_RASR_EX`的简化版本。它直接接受存储器访问属性的各个组件，而不是像`ARM_MPU_RASR_EX`那样接受预定义的访问属性。

**7. 使能/禁用 MPU 的函数**

```c
__STATIC_INLINE void ARM_MPU_Enable(uint32_t MPU_Control) { ... }
__STATIC_INLINE void ARM_MPU_Disable(void) { ... }
```

**描述:**  `ARM_MPU_Enable` 函数使能 MPU。 `MPU_Control` 参数指定未配置区域的默认访问权限。 `ARM_MPU_Disable` 函数禁用 MPU。

**如何使用:**  在配置完 MPU 区域后，调用 `ARM_MPU_Enable` 来激活 MPU。  在需要禁用 MPU 时，调用 `ARM_MPU_Disable`。

**8. 其他辅助函数**

*   `ARM_MPU_ClrRegion`: 清除并禁用给定的 MPU 区域。
*   `ARM_MPU_SetRegion`:  配置一个 MPU 区域。
*   `ARM_MPU_SetRegionEx`:  配置给定的 MPU 区域（指定区域号）。
*   `orderedCpy`:  使用严格排序的内存访问进行内存复制。
*   `ARM_MPU_Load`: 从表中加载给定数量的 MPU 区域。

**总结和示例**

这些宏和函数提供了一个方便的接口来配置 Armv7-M MPU。  通过正确配置 MPU，你可以保护你的系统免受内存访问错误的影响，并提高系统的安全性。

**完整的示例 (假设使用 STM32F4 系列单片机):**

```c
#include "stm32f4xx.h" // 假设使用 STM32F4 系列单片机，需要包含相应的头文件
#include "mpu_armv7.h"

int main(void) {
  // 1. 初始化系统 (例如，时钟配置)
  SystemInit(); // 调用系统初始化函数，这个函数通常由ST提供。

  // 2. 禁用 MPU (可选，但推荐在配置前禁用)
  ARM_MPU_Disable();

  // 3. 配置 MPU 区域 0：用于 Flash (只读，禁止执行)
  uint32_t rbar0 = ARM_MPU_RBAR(0, FLASH_BASE); // FLASH 起始地址
  uint32_t rasr0 = ARM_MPU_RASR(1, ARM_MPU_AP_RO, 0, 0, 1, 0, 0, ARM_MPU_REGION_SIZE_1MB); // 禁止执行, 只读，可缓存，1MB

  // 4. 配置 MPU 区域 1：用于 SRAM (可读写，可执行)
  uint32_t rbar1 = ARM_MPU_RBAR(1, SRAM_BASE); // SRAM 起始地址
  uint32_t rasr1 = ARM_MPU_RASR(0, ARM_MPU_AP_FULL, 0, 0, 1, 1, 0, ARM_MPU_REGION_SIZE_256KB); // 允许执行，完全访问，可缓存，可缓冲，256KB

  // 5. 应用配置
  ARM_MPU_SetRegionEx(0, rbar0, rasr0); // 使用 Ex 版本指定区域号
  ARM_MPU_SetRegionEx(1, rbar1, rasr1);

  // 6. 启用 MPU
  ARM_MPU_Enable(MPU_CTRL_PRIVDEFENA_Msk);  // 允许在未配置区域的默认访问权限

  while (1) {
    // 你的程序逻辑...
    // 例如，尝试从 Flash 写入数据，应该会触发 MPU 错误。
    // 例如，在 SRAM 中运行代码。
  }
}
```

**重要注意事项:**

*   **`stm32f4xx.h`:**  这个头文件特定于 STM32F4 系列单片机。你需要根据你使用的单片机型号包含相应的头文件。
*   **`FLASH_BASE` 和 `SRAM_BASE`:**  这些宏定义了 Flash 和 SRAM 的起始地址。 它们也定义在特定于单片机的头文件中。
*  **SystemInit()**: stm32f4xx的库函数，用于初始化系统时钟等等。
*   **MPU 错误处理:**  你需要设置一个 MPU 错误处理程序来捕获 MPU 错误。  当发生 MPU 错误时，会调用这个处理程序。  这通常涉及配置 System Control Block (SCB) 中的相关位。

这个更详细的解释和示例应该可以帮助你理解和使用 Armv7-M MPU API。 请记住根据你使用的特定单片机调整代码。