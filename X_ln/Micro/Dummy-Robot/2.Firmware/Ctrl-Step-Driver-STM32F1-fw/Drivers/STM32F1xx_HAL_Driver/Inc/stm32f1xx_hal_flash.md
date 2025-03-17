Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_flash.h`

好的， 让我们重新开始。我将对代码的关键部分进行解释，生成代码片段，并附上简短的中文描述，包括代码用法说明和简单示例。

**总览**

这个 `stm32f1xx_hal_flash.h` 文件是 STM32F1 系列微控制器的 Flash HAL (Hardware Abstraction Layer) 模块的头文件。  它定义了用于控制和访问 STM32F1 内部 Flash 存储器的函数、宏、结构体和枚举类型。HAL 的目的是提供一个抽象层，使应用程序代码能够以更通用的方式与硬件交互，而无需直接处理底层寄存器。

**1. 头文件保护 (Header Guard)**

```c
#ifndef __STM32F1xx_HAL_FLASH_H
#define __STM32F1xx_HAL_FLASH_H

// ... 文件内容 ...

#endif /* __STM32F1xx_HAL_FLASH_H */
```

**描述:** 这段代码使用预处理器指令来防止头文件被重复包含。 如果 `__STM32F1xx_HAL_FLASH_H` 还没有被定义，则定义它，并包含头文件的内容。 否则，跳过头文件的内容。 这可以避免编译错误。
**用途:**  防止多次包含头文件导致重复定义错误。

**2. C++ 兼容性 (C++ Compatibility)**

```c
#ifdef __cplusplus
 extern "C" {
#endif

// ...

#ifdef __cplusplus
}
#endif
```

**描述:**  这段代码用于确保 C 头文件可以被 C++ 代码包含。  `extern "C"`  指示编译器使用 C 链接约定，这对于混合 C 和 C++ 代码非常重要。
**用途:** 允许 C++ 代码调用 C 函数。

**3. 包含必要的头文件 (Includes)**

```c
#include "stm32f1xx_hal_def.h"
```

**描述:**  包含了 `stm32f1xx_hal_def.h` 头文件，它定义了 HAL 库中常用的数据类型、宏定义和其他基础结构。
**用途:** 提供 HAL 库的基础定义。

**4. 超时值 (Timeout Value)**

```c
#define FLASH_TIMEOUT_VALUE              50000U /* 50 s */
```

**描述:**  定义了 Flash 操作的默认超时时间，单位通常是系统时钟周期。这里设置为 50 秒。
**用途:**  在 Flash 操作中，如果超过这个时间仍然没有完成，则认为操作失败。

**5. FLASH 程序类型宏 (FLASH Program Type Macros)**

```c
#define IS_FLASH_TYPEPROGRAM(VALUE)  (((VALUE) == FLASH_TYPEPROGRAM_HALFWORD) || \
                                      ((VALUE) == FLASH_TYPEPROGRAM_WORD)     || \
                                      ((VALUE) == FLASH_TYPEPROGRAM_DOUBLEWORD))
```

**描述:**  这是一个宏，用于检查给定的 `VALUE` 是否是有效的 Flash 程序类型。
**用途:**  在函数参数检查中使用，确保程序类型是允许的。

**6. FLASH 延迟宏 (FLASH Latency Macros)**

```c
#if   defined(FLASH_ACR_LATENCY)
#define IS_FLASH_LATENCY(__LATENCY__) (((__LATENCY__) == FLASH_LATENCY_0) || \
                                       ((__LATENCY__) == FLASH_LATENCY_1) || \
                                       ((__LATENCY__) == FLASH_LATENCY_2))

#else
#define IS_FLASH_LATENCY(__LATENCY__)   ((__LATENCY__) == FLASH_LATENCY_0)
#endif /* FLASH_ACR_LATENCY */
```

**描述:**  这些宏用于检查给定的延迟值 `__LATENCY__` 是否是有效的 Flash 延迟。 延迟设置与系统时钟频率有关，用于确保 Flash 访问的正确时序。`FLASH_ACR_LATENCY` 的定义决定了是否支持多个延迟选项。
**用途:**  在函数参数检查中使用，确保延迟设置是允许的。

**7. FLASH 过程类型枚举 (FLASH Procedure Type Enumeration)**

```c
typedef enum 
{
  FLASH_PROC_NONE              = 0U, 
  FLASH_PROC_PAGEERASE         = 1U,
  FLASH_PROC_MASSERASE         = 2U,
  FLASH_PROC_PROGRAMHALFWORD   = 3U,
  FLASH_PROC_PROGRAMWORD       = 4U,
  FLASH_PROC_PROGRAMDOUBLEWORD = 5U
} FLASH_ProcedureTypeDef;
```

**描述:**  定义了 Flash 操作的类型，例如无操作、页擦除、整片擦除、半字编程、字编程和双字编程。
**用途:** 用于记录当前正在进行的Flash操作，以便在中断处理程序中进行管理。

**8. FLASH 进程类型结构体 (FLASH Process Type Structure)**

```c
typedef struct
{
  __IO FLASH_ProcedureTypeDef ProcedureOnGoing; /*!< Internal variable to indicate which procedure is ongoing or not in IT context */
  
  __IO uint32_t               DataRemaining;    /*!< Internal variable to save the remaining pages to erase or half-word to program in IT context */

  __IO uint32_t               Address;          /*!< Internal variable to save address selected for program or erase */

  __IO uint64_t               Data;             /*!< Internal variable to save data to be programmed */

  HAL_LockTypeDef             Lock;             /*!< FLASH locking object                */

  __IO uint32_t               ErrorCode;        /*!< FLASH error code                    
                                                     This parameter can be a value of @ref FLASH_Error_Codes  */
} FLASH_ProcessTypeDef;
```

**描述:**  定义了一个结构体，用于保存 Flash 操作的状态信息，例如正在进行的操作类型、剩余数据量、地址、数据、锁状态和错误代码。  `__IO` 表示这个变量是易变的，可能会被中断服务程序修改。
**用途:**  用于在中断上下文中管理 Flash 操作，保持状态。

**9. FLASH 错误代码 (FLASH Error Codes)**

```c
#define HAL_FLASH_ERROR_NONE      0x00U  /*!< No error */
#define HAL_FLASH_ERROR_PROG      0x01U  /*!< Programming error */
#define HAL_FLASH_ERROR_WRP       0x02U  /*!< Write protection error */
#define HAL_FLASH_ERROR_OPTV      0x04U  /*!< Option validity error */
```

**描述:**  定义了 Flash 操作可能产生的错误代码。
**用途:**  在 Flash 操作完成后，可以通过检查错误代码来判断操作是否成功。

**10. FLASH 程序类型 (FLASH Program Types)**

```c
#define FLASH_TYPEPROGRAM_HALFWORD             0x01U  /*!<Program a half-word (16-bit) at a specified address.*/
#define FLASH_TYPEPROGRAM_WORD                 0x02U  /*!<Program a word (32-bit) at a specified address.*/
#define FLASH_TYPEPROGRAM_DOUBLEWORD           0x03U  /*!<Program a double word (64-bit) at a specified address*/
```

**描述:**  定义了 Flash 编程操作支持的数据类型，例如半字、字和双字。
**用途:**  作为 `HAL_FLASH_Program` 函数的参数，指定要写入的数据类型。

**11. FLASH 延迟定义 (FLASH Latency Definitions)**

```c
#if   defined(FLASH_ACR_LATENCY)
/** @defgroup FLASH_Latency FLASH Latency
  * @{
  */
#define FLASH_LATENCY_0            0x00000000U               /*!< FLASH Zero Latency cycle */
#define FLASH_LATENCY_1            FLASH_ACR_LATENCY_0       /*!< FLASH One Latency cycle */
#define FLASH_LATENCY_2            FLASH_ACR_LATENCY_1       /*!< FLASH Two Latency cycles */

/**
  * @}
  */

#else
/** @defgroup FLASH_Latency FLASH Latency
  * @{
  */
#define FLASH_LATENCY_0            0x00000000U    /*!< FLASH Zero Latency cycle */

/**
  * @}
  */

#endif /* FLASH_ACR_LATENCY */
```

**描述:** 定义了不同的 Flash 延迟周期。 延迟周期数取决于微控制器的时钟频率。  选择合适的延迟可以确保 Flash 访问的正确性。
**用途:**  在初始化 Flash 时设置延迟。

**12. FLASH 半周期访问宏 (FLASH Half Cycle Access Macros)**

```c
/**
  * @brief  Enable the FLASH half cycle access.
  * @note   half cycle access can only be used with a low-frequency clock of less than
            8 MHz that can be obtained with the use of HSI or HSE but not of PLL.
  * @retval None
  */
#define __HAL_FLASH_HALF_CYCLE_ACCESS_ENABLE()  (FLASH->ACR |= FLASH_ACR_HLFCYA)

/**
  * @brief  Disable the FLASH half cycle access.
  * @note   half cycle access can only be used with a low-frequency clock of less than
            8 MHz that can be obtained with the use of HSI or HSE but not of PLL.
  * @retval None
  */
#define __HAL_FLASH_HALF_CYCLE_ACCESS_DISABLE() (FLASH->ACR &= (~FLASH_ACR_HLFCYA))
```

**描述:**  这两个宏用于启用或禁用 Flash 的半周期访问模式。 半周期访问只能在低频时钟下使用 (低于 8MHz) 。
**用途:**  在低频应用中，可以使用半周期访问来降低功耗。

**13. FLASH 延迟设置宏 (FLASH Latency Setting Macros)**

```c
#if defined(FLASH_ACR_LATENCY)
/** @defgroup FLASH_EM_Latency FLASH Latency
 *  @brief macros to handle FLASH Latency
 * @{
 */ 
  
/**
  * @brief  Set the FLASH Latency.
  * @param  __LATENCY__ FLASH Latency                   
  *         The value of this parameter depend on device used within the same series
  * @retval None
  */ 
#define __HAL_FLASH_SET_LATENCY(__LATENCY__)    (FLASH->ACR = (FLASH->ACR&(~FLASH_ACR_LATENCY)) | (__LATENCY__))


/**
  * @brief  Get the FLASH Latency.
  * @retval FLASH Latency                   
  *         The value of this parameter depend on device used within the same series
  */ 
#define __HAL_FLASH_GET_LATENCY()     (READ_BIT((FLASH->ACR), FLASH_ACR_LATENCY))

/**
  * @}
  */

#endif /* FLASH_ACR_LATENCY */
```

**描述:**  这些宏用于设置和获取 Flash 的延迟。`__HAL_FLASH_SET_LATENCY` 用于设置延迟，`__HAL_FLASH_GET_LATENCY` 用于获取当前延迟。
**用途:**  根据系统时钟频率配置 Flash 延迟。

**14. FLASH 预取缓冲宏 (FLASH Prefetch Buffer Macros)**

```c
/** @defgroup FLASH_Prefetch FLASH Prefetch
 *  @brief macros to handle FLASH Prefetch buffer
 * @{
 */   
/**
  * @brief  Enable the FLASH prefetch buffer.
  * @retval None
  */ 
#define __HAL_FLASH_PREFETCH_BUFFER_ENABLE()    (FLASH->ACR |= FLASH_ACR_PRFTBE)

/**
  * @brief  Disable the FLASH prefetch buffer.
  * @retval None
  */
#define __HAL_FLASH_PREFETCH_BUFFER_DISABLE()   (FLASH->ACR &= (~FLASH_ACR_PRFTBE))
```

**描述:**  这两个宏用于启用或禁用 Flash 的预取缓冲。 预取缓冲可以提高 Flash 的读取速度。
**用途:**  在需要高性能 Flash 读取的应用中，可以启用预取缓冲。

**15. 包含 FLASH 扩展头文件 (Include FLASH Extended Header File)**

```c
#include "stm32f1xx_hal_flash_ex.h"
```

**描述:**  包含了 Flash HAL 的扩展头文件，其中定义了更多的 Flash 操作函数，例如 OTP (One-Time Programmable) 操作和用户扇区配置。
**用途:**  提供额外的 Flash 功能。

**16. 导出函数 (Exported Functions)**

```c
HAL_StatusTypeDef HAL_FLASH_Program(uint32_t TypeProgram, uint32_t Address, uint64_t Data);
HAL_StatusTypeDef HAL_FLASH_Program_IT(uint32_t TypeProgram, uint32_t Address, uint64_t Data);

/* FLASH IRQ handler function */
void       HAL_FLASH_IRQHandler(void);
/* Callbacks in non blocking modes */ 
void       HAL_FLASH_EndOfOperationCallback(uint32_t ReturnValue);
void       HAL_FLASH_OperationErrorCallback(uint32_t ReturnValue);

HAL_StatusTypeDef HAL_FLASH_Unlock(void);
HAL_StatusTypeDef HAL_FLASH_Lock(void);
HAL_StatusTypeDef HAL_FLASH_OB_Unlock(void);
HAL_StatusTypeDef HAL_FLASH_OB_Lock(void);
void HAL_FLASH_OB_Launch(void);

uint32_t HAL_FLASH_GetError(void);
```

**描述:**  声明了 Flash HAL 库提供的函数。  这些函数包括：
    *   `HAL_FLASH_Program`:  用于将数据写入 Flash。
    *   `HAL_FLASH_Program_IT`:  用于在中断模式下将数据写入 Flash。
    *   `HAL_FLASH_IRQHandler`: Flash 中断处理函数。
    *   `HAL_FLASH_EndOfOperationCallback`:  Flash 操作完成回调函数。
    *   `HAL_FLASH_OperationErrorCallback`:  Flash 操作错误回调函数。
    *   `HAL_FLASH_Unlock`:  解锁 Flash，允许写入。
    *   `HAL_FLASH_Lock`:  锁定 Flash，防止意外写入。
    *   `HAL_FLASH_OB_Unlock`: 解锁选项字节（Option Bytes）。
    *   `HAL_FLASH_OB_Lock`: 锁定选项字节。
    *   `HAL_FLASH_OB_Launch`: 启动选项字节的更改。
    *   `HAL_FLASH_GetError`:  获取 Flash 错误代码。
**用途:**  提供应用程序访问 Flash 存储器的接口。

**17. 私有函数 (Private Functions)**

```c
HAL_StatusTypeDef       FLASH_WaitForLastOperation(uint32_t Timeout);
#if defined(FLASH_BANK2_END)
HAL_StatusTypeDef       FLASH_WaitForLastOperationBank2(uint32_t Timeout);
#endif /* FLASH_BANK2_END */
```

**描述:**  声明了 Flash HAL 库内部使用的函数。`FLASH_WaitForLastOperation` 用于等待上一次 Flash 操作完成。  `FLASH_WaitForLastOperationBank2` 用于多 Bank Flash 芯片，等待 Bank2 的操作完成。
**用途:**  在 HAL 函数内部使用，用于控制 Flash 操作的时序。

**示例代码 (Demo):**

以下是一个简单的 Flash 编程示例：

```c
#include "stm32f1xx_hal.h"

#define FLASH_USER_START_ADDR   ((uint32_t)0x08008000) /* Start address for user program : 32KBytes from flash start */

int main(void) {
  HAL_Init();

  // 假设已经完成了时钟和其他外设的初始化

  uint32_t address = FLASH_USER_START_ADDR;
  uint64_t data = 0x123456789ABCDEF0;

  // 1. 解锁 Flash
  HAL_FLASH_Unlock();

  // 2. 擦除要写入的页 (可选，如果该页之前已经写入过数据)
  //   (需要包含 stm32f1xx_hal_flash_ex.h 并调用 HAL_FLASHEx_Erase)

  // 3. 写入数据
  HAL_StatusTypeDef status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, address, data);

  // 4. 锁定 Flash
  HAL_FLASH_Lock();

  if (status == HAL_OK) {
    // 写入成功
    // 可以读取 Flash 地址验证写入是否正确
  } else {
    // 写入失败
    // 可以调用 HAL_FLASH_GetError() 获取错误代码
  }

  while (1) {
    // ...
  }
}
```

**说明:**

1.  **包含头文件:** 包含必要的 HAL 头文件。
2.  **解锁 Flash:**  在写入 Flash 之前，必须先解锁 Flash。
3.  **擦除页 (可选):** 如果要写入的 Flash 区域之前已经写入过数据，需要先擦除该页。 这通常需要调用 `HAL_FLASHEx_Erase` 函数，定义在 `stm32f1xx_hal_flash_ex.h` 中。
4.  **写入数据:** 调用 `HAL_FLASH_Program` 函数写入数据。 第一个参数指定要写入的数据类型 (例如 `FLASH_TYPEPROGRAM_DOUBLEWORD`)，第二个参数是要写入的地址，第三个参数是要写入的数据。
5.  **锁定 Flash:**  在写入完成后，建议锁定 Flash，以防止意外写入。
6.  **错误处理:** 检查 `HAL_FLASH_Program` 函数的返回值，判断写入是否成功。 如果写入失败，可以调用 `HAL_FLASH_GetError` 函数获取错误代码。

**重要的提示:**

*   在进行 Flash 操作之前，务必阅读 STM32F1 系列微控制器的参考手册，了解 Flash 的特性和限制。
*   确保正确配置系统时钟，并根据时钟频率设置 Flash 延迟。
*   在擦除 Flash 之前，请务必备份重要数据。
*   注意 Flash 的写入次数限制。 频繁写入 Flash 可能会缩短 Flash 的寿命。

希望这个详细的解释和示例能够帮助你理解 `stm32f1xx_hal_flash.h` 文件。
