Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_flash_ex.h`

好的，我们重新开始，对提供的 C 头文件 `stm32f1xx_hal_flash_ex.h` 进行逐段解释，并为关键部分提供代码示例和中文说明。

**总览**

该头文件 `stm32f1xx_hal_flash_ex.h` 是 STM32F1 系列微控制器 HAL (Hardware Abstraction Layer) 库中 Flash 扩展模块的头文件。 它定义了用于扩展 FLASH 存储器操作的类型、常量、宏和函数。 这些扩展操作包括擦除、选项字节 (Option Bytes) 的编程和读取等。选项字节用于配置诸如读保护、写保护、看门狗等功能。

**1. 头文件保护和 C++ 兼容性**

```c
#ifndef __STM32F1xx_HAL_FLASH_EX_H
#define __STM32F1xx_HAL_FLASH_EX_H

#ifdef __cplusplus
 extern "C" {
#endif
```

*   `#ifndef __STM32F1xx_HAL_FLASH_EX_H`:  防止头文件被重复包含。
*   `#define __STM32F1xx_HAL_FLASH_EX_H`:  定义宏，表示头文件已被包含。
*   `#ifdef __cplusplus ... extern "C" { ... #endif`: 允许 C++ 代码调用 C 函数。  `extern "C"` 确保 C++ 编译器使用 C 的链接规则，因为 C 和 C++ 的函数命名方式可能不同。

**2. 包含头文件**

```c
#include "stm32f1xx_hal_def.h"
```

*   `#include "stm32f1xx_hal_def.h"`:  包含 HAL 库的定义文件，如基本数据类型 ( `uint32_t` 等) 和常用宏。

**3. 外设驱动分组**

```c
/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @addtogroup FLASHEx
  * @{
  */
```

*   `@addtogroup STM32F1xx_HAL_Driver`: 将文件添加到 STM32F1xx HAL 驱动程序组。
*   `@addtogroup FLASHEx`:  将文件添加到 FLASHEx 扩展模块组。 这些是文档生成工具（如 Doxygen）使用的标记，用于组织和生成 HAL 库的文档。

**4. 私有常量**

```c
/** @addtogroup FLASHEx_Private_Constants
  * @{
  */

#define FLASH_SIZE_DATA_REGISTER     0x1FFFF7E0U
#define OBR_REG_INDEX                1U
#define SR_FLAG_MASK                 ((uint32_t)(FLASH_SR_BSY | FLASH_SR_PGERR | FLASH_SR_WRPRTERR | FLASH_SR_EOP))

/**
  * @}
  */
```

*   `FLASH_SIZE_DATA_REGISTER`: 包含 FLASH 大小的寄存器地址。 通过读取该地址，可以确定芯片上 FLASH 的大小。
*   `OBR_REG_INDEX`: 选项字节寄存器 (Option Bytes Register) 的索引。
*   `SR_FLAG_MASK`:  状态寄存器 (Status Register) 中相关标志的掩码。 这些标志指示 FLASH 操作的状态（忙、编程错误、写保护错误、操作结束）。

**示例代码：读取 FLASH 大小**

```c
uint16_t flash_size = *((uint16_t *)FLASH_SIZE_DATA_REGISTER);
// flash_size now contains the FLASH size in KB
// 例如，如果 flash_size 等于 128，则表示有 128KB 的 FLASH 存储器.
```

**5. 私有宏**

```c
/** @addtogroup FLASHEx_Private_Macros
  * @{
  */

#define IS_FLASH_TYPEERASE(VALUE)   (((VALUE) == FLASH_TYPEERASE_PAGES) || ((VALUE) == FLASH_TYPEERASE_MASSERASE))
#define IS_OPTIONBYTE(VALUE)        (((VALUE) <= (OPTIONBYTE_WRP | OPTIONBYTE_RDP | OPTIONBYTE_USER | OPTIONBYTE_DATA)))
#define IS_WRPSTATE(VALUE)          (((VALUE) == OB_WRPSTATE_DISABLE) || ((VALUE) == OB_WRPSTATE_ENABLE))
#define IS_OB_RDP_LEVEL(LEVEL)      (((LEVEL) == OB_RDP_LEVEL_0) || ((LEVEL) == OB_RDP_LEVEL_1))
#define IS_OB_DATA_ADDRESS(ADDRESS) (((ADDRESS) == OB_DATA_ADDRESS_DATA0) || ((ADDRESS) == OB_DATA_ADDRESS_DATA1))
#define IS_OB_IWDG_SOURCE(SOURCE)   (((SOURCE) == OB_IWDG_SW) || ((SOURCE) == OB_IWDG_HW))
#define IS_OB_STOP_SOURCE(SOURCE)   (((SOURCE) == OB_STOP_NO_RST) || ((SOURCE) == OB_STOP_RST))
#define IS_OB_STDBY_SOURCE(SOURCE)  (((SOURCE) == OB_STDBY_NO_RST) || ((SOURCE) == OB_STDBY_RST))

#if defined(FLASH_BANK2_END)
#define IS_OB_BOOT1(BOOT1)         (((BOOT1) == OB_BOOT1_RESET) || ((BOOT1) == OB_BOOT1_SET))
#endif /* FLASH_BANK2_END */

// ... (其他 IS_FLASH_NB_PAGES 和 IS_FLASH_PROGRAM_ADDRESS 宏)

/**
  * @}
  */
```

这些宏用于参数检查。 它们验证传递给 HAL 函数的参数是否在允许的范围内。

*   `IS_FLASH_TYPEERASE(VALUE)`: 检查擦除类型是否有效 (页擦除或全部擦除).
*   `IS_OPTIONBYTE(VALUE)`: 检查选项字节类型是否有效.
*   `IS_WRPSTATE(VALUE)`: 检查写保护状态是否有效 (启用或禁用).
*   `IS_OB_RDP_LEVEL(LEVEL)`: 检查读保护级别是否有效.
*   `IS_OB_DATA_ADDRESS(ADDRESS)`: 检查选项字节数据地址是否有效.
*   `IS_OB_IWDG_SOURCE(SOURCE)`: 检查独立看门狗源是否有效 (软件或硬件).
*   `IS_OB_STOP_SOURCE(SOURCE)`: 检查 STOP 模式复位源是否有效.
*   `IS_OB_STDBY_SOURCE(SOURCE)`: 检查 STANDBY 模式复位源是否有效.
*   `IS_OB_BOOT1(BOOT1)`: (如果定义了 `FLASH_BANK2_END` ) 检查 BOOT1 配置是否有效.
*   `IS_FLASH_NB_PAGES(ADDRESS, NBPAGES)`: 检查给定的地址和页数是否在 FLASH 范围内。不同密度的设备有不同的 FLASH 范围，宏会根据 FLASH_SIZE_DATA_REGISTER 寄存器来判断.
*   `IS_FLASH_PROGRAM_ADDRESS(ADDRESS)`: 检查给定的地址是否是有效的 FLASH 编程地址。

**示例代码：使用参数检查宏**

```c
FLASH_EraseInitTypeDef EraseInitStruct;
EraseInitStruct.TypeErase = FLASH_TYPEERASE_PAGES; // 页擦除

if (!IS_FLASH_TYPEERASE(EraseInitStruct.TypeErase)) {
  //  处理错误：无效的擦除类型
  printf("Error: Invalid erase type!\n");
  return HAL_ERROR;
}

EraseInitStruct.PageAddress = 0x08004000; // 从地址 0x08004000 开始

if (!IS_FLASH_PROGRAM_ADDRESS(EraseInitStruct.PageAddress)) {
   // 处理错误：无效的Flash 编程地址
   printf("Error: Invalid Flash program address!\n");
   return HAL_ERROR;
}
```

**6. 导出类型定义 (typedef)**

```c
/** @defgroup FLASHEx_Exported_Types FLASHEx Exported Types
  * @{
  */

/**
  * @brief  FLASH Erase structure definition
  */
typedef struct
{
  uint32_t TypeErase;
  uint32_t Banks;
  uint32_t PageAddress;
  uint32_t NbPages;
} FLASH_EraseInitTypeDef;

/**
  * @brief  FLASH Options bytes program structure definition
  */
typedef struct
{
  uint32_t OptionType;
  uint32_t WRPState;
  uint32_t WRPPage;
  uint32_t Banks;
  uint8_t RDPLevel;
  uint8_t USERConfig;
  uint32_t DATAAddress;
  uint8_t DATAData;
} FLASH_OBProgramInitTypeDef;

/**
  * @}
  */
```

*   `FLASH_EraseInitTypeDef`: 用于配置 FLASH 擦除操作的结构体。
    *   `TypeErase`:  擦除类型 (页擦除或全部擦除)。
    *   `Banks`: 选择要擦除的存储体（Bank）。
    *   `PageAddress`: 起始页地址。
    *   `NbPages`: 要擦除的页数。
*   `FLASH_OBProgramInitTypeDef`: 用于配置选项字节编程的结构体。
    *   `OptionType`: 要配置的选项字节类型 (WRP, RDP, USER, DATA).
    *   `WRPState`: 写保护状态 (启用或禁用).
    *   `WRPPage`: 要写保护的页。
    *   `Banks`: 选择要操作的存储体。
    *   `RDPLevel`: 读保护级别。
    *   `USERConfig`: 用户配置 (IWDG, STOP, STDBY, BOOT1 配置).
    *   `DATAAddress`: 数据选项字节的地址。
    *   `DATAData`: 要写入数据选项字节的数据。

**示例代码：配置 FLASH_EraseInitTypeDef**

```c
FLASH_EraseInitTypeDef EraseInitStruct;
uint32_t PageError = 0;

/* Fill EraseInit structure*/
EraseInitStruct.TypeErase   = FLASH_TYPEERASE_PAGES;  // 选择页擦除
EraseInitStruct.PageAddress = 0x08004000;            // 设置起始地址为 0x08004000
EraseInitStruct.NbPages     = 2;                      // 擦除 2 页
EraseInitStruct.Banks       = FLASH_BANK_1;             // 选择BANK1

/* Note: If an erase operation in Flash memory also concerns data in the data Flash memory,
   the data Flash memory will be erased and initialized to the value 0xFF.
   When a Mass Erase is requested, for all user Flash memory, the nWRP bits are discarded. */
if (HAL_FLASHEx_Erase(&EraseInitStruct, &PageError) != HAL_OK)
{
  /*
  Error occurred while page erase.
  Add some user code to manage this error.
  */
  printf("Error occurred during Flash erase!\n");
  return HAL_ERROR;
}
```

**7. 导出常量**

```c
/** @defgroup FLASHEx_Exported_Constants FLASHEx Exported Constants
  * @{
  */

/** @defgroup FLASHEx_Constants FLASH Constants
  * @{
  */

/** @defgroup FLASHEx_Page_Size Page Size
  * @{
  */
#if (defined(STM32F101x6) || defined(STM32F102x6) || defined(STM32F103x6) || defined(STM32F100xB) || defined(STM32F101xB) || defined(STM32F102xB) || defined(STM32F103xB))
#define FLASH_PAGE_SIZE          0x400U //1024
#endif /* STM32F101x6 || STM32F102x6 || STM32F103x6 */
       /* STM32F100xB || STM32F101xB || STM32F102xB || STM32F103xB */

#if (defined(STM32F100xE) || defined(STM32F101xE) || defined(STM32F103xE) || defined(STM32F101xG) || defined(STM32F103xG) || defined(STM32F105xC) || defined(STM32F107xC))
#define FLASH_PAGE_SIZE          0x800U //2048
#endif /* STM32F100xB || STM32F101xB || STM32F102xB || STM32F103xB */
       /* STM32F101xG || STM32F103xG */
       /* STM32F105xC || STM32F107xC */

/**
  * @}
  */

/** @defgroup FLASHEx_Type_Erase Type Erase
  * @{
  */
#define FLASH_TYPEERASE_PAGES     0x00U  /*!<Pages erase only*/
#define FLASH_TYPEERASE_MASSERASE 0x02U  /*!<Flash mass erase activation*/

/**
  * @}
  */

/** @defgroup FLASHEx_Banks Banks
  * @{
  */
#if defined(FLASH_BANK2_END)
#define FLASH_BANK_1     1U /*!< Bank 1   */
#define FLASH_BANK_2     2U /*!< Bank 2   */
#define FLASH_BANK_BOTH  ((uint32_t)FLASH_BANK_1 | FLASH_BANK_2) /*!< Bank1 and Bank2  */

#else
#define FLASH_BANK_1     1U /*!< Bank 1   */
#endif
/**
  * @}
  */

/**
  * @}
  */

/** @defgroup FLASHEx_OptionByte_Constants Option Byte Constants
  * @{
  */

/** @defgroup FLASHEx_OB_Type Option Bytes Type
  * @{
  */
#define OPTIONBYTE_WRP            0x01U  /*!<WRP option byte configuration*/
#define OPTIONBYTE_RDP            0x02U  /*!<RDP option byte configuration*/
#define OPTIONBYTE_USER           0x04U  /*!<USER option byte configuration*/
#define OPTIONBYTE_DATA           0x08U  /*!<DATA option byte configuration*/

/**
  * @}
  */

/** @defgroup FLASHEx_OB_WRP_State Option Byte WRP State
  * @{
  */
#define OB_WRPSTATE_DISABLE       0x00U  /*!<Disable the write protection of the desired pages*/
#define OB_WRPSTATE_ENABLE        0x01U  /*!<Enable the write protection of the desired pagess*/

/**
  * @}
  */

// ... (其他常量定义，如写保护页定义，读保护级别，看门狗源，STOP/STANDBY 复位源, BOOT1 配置, 数据地址)

/**
  * @}
  */

/** @addtogroup FLASHEx_Constants
  * @{
  */

/** @defgroup FLASH_Flag_definition Flag definition
  * @brief Flag definition
  * @{
  */
#if defined(FLASH_BANK2_END)
 #define FLASH_FLAG_BSY             FLASH_FLAG_BSY_BANK1       /*!< FLASH Bank1 Busy flag                   */
 #define FLASH_FLAG_PGERR           FLASH_FLAG_PGERR_BANK1     /*!< FLASH Bank1 Programming error flag      */
 #define FLASH_FLAG_WRPERR          FLASH_FLAG_WRPERR_BANK1    /*!< FLASH Bank1 Write protected error flag  */
 #define FLASH_FLAG_EOP             FLASH_FLAG_EOP_BANK1       /*!< FLASH Bank1 End of Operation flag       */

 #define FLASH_FLAG_BSY_BANK1       FLASH_SR_BSY               /*!< FLASH Bank1 Busy flag                   */
 #define FLASH_FLAG_PGERR_BANK1     FLASH_SR_PGERR             /*!< FLASH Bank1 Programming error flag      */
 #define FLASH_FLAG_WRPERR_BANK1    FLASH_SR_WRPRTERR          /*!< FLASH Bank1 Write protected error flag  */
 #define FLASH_FLAG_EOP_BANK1       FLASH_SR_EOP               /*!< FLASH Bank1 End of Operation flag       */

 #define FLASH_FLAG_BSY_BANK2       (FLASH_SR2_BSY << 16U)      /*!< FLASH Bank2 Busy flag                   */
 #define FLASH_FLAG_PGERR_BANK2     (FLASH_SR2_PGERR << 16U)    /*!< FLASH Bank2 Programming error flag      */
 #define FLASH_FLAG_WRPERR_BANK2    (FLASH_SR2_WRPRTERR << 16U) /*!< FLASH Bank2 Write protected error flag  */
 #define FLASH_FLAG_EOP_BANK2       (FLASH_SR2_EOP << 16U)      /*!< FLASH Bank2 End of Operation flag       */

#else

 #define FLASH_FLAG_BSY             FLASH_SR_BSY              /*!< FLASH Busy flag                          */
 #define FLASH_FLAG_PGERR           FLASH_SR_PGERR            /*!< FLASH Programming error flag             */
 #define FLASH_FLAG_WRPERR          FLASH_SR_WRPRTERR         /*!< FLASH Write protected error flag         */
 #define FLASH_FLAG_EOP             FLASH_SR_EOP              /*!< FLASH End of Operation flag              */

#endif
 #define FLASH_FLAG_OPTVERR         ((OBR_REG_INDEX << 8U | FLASH_OBR_OPTERR)) /*!< Option Byte Error        */
/**
  * @}
  */

/** @defgroup FLASH_Interrupt_definition Interrupt definition
  * @brief FLASH Interrupt definition
  * @{
  */
#if defined(FLASH_BANK2_END)
 #define FLASH_IT_EOP               FLASH_IT_EOP_BANK1        /*!< End of FLASH Operation Interrupt source Bank1 */
 #define FLASH_IT_ERR               FLASH_IT_ERR_BANK1        /*!< Error Interrupt source Bank1                  */

 #define FLASH_IT_EOP_BANK1         FLASH_CR_EOPIE            /*!< End of FLASH Operation Interrupt source Bank1 */
 #define FLASH_IT_ERR_BANK1         FLASH_CR_ERRIE            /*!< Error Interrupt source Bank1                  */

 #define FLASH_IT_EOP_BANK2         (FLASH_CR2_EOPIE << 16U)   /*!< End of FLASH Operation Interrupt source Bank2 */
 #define FLASH_IT_ERR_BANK2         (FLASH_CR2_ERRIE << 16U)   /*!< Error Interrupt source Bank2                  */

#else

 #define FLASH_IT_EOP               FLASH_CR_EOPIE          /*!< End of FLASH Operation Interrupt source */
 #define FLASH_IT_ERR               FLASH_CR_ERRIE          /*!< Error Interrupt source                  */

#endif
/**
  * @}
  */

/**
  * @}
  */


/**
  * @}
  */

```

这部分定义了大量的常量，包括：

*   `FLASH_PAGE_SIZE`: FLASH 页大小，根据不同的 STM32F1 设备而不同 (1KB 或 2KB)。
*   `FLASH_TYPEERASE_PAGES` 和 `FLASH_TYPEERASE_MASSERASE`:  FLASH 擦除类型 (页擦除或全部擦除)。
*   `FLASH_BANK_1`, `FLASH_BANK_2`, `FLASH_BANK_BOTH`:  用于选择 FLASH 存储体（Bank），仅在具有多个存储体的设备上有效。
*   选项字节相关的常量，如 `OPTIONBYTE_WRP`, `OPTIONBYTE_RDP`, `OPTIONBYTE_USER`, `OPTIONBYTE_DATA`, `OB_WRPSTATE_DISABLE`, `OB_WRPSTATE_ENABLE`, `OB_RDP_LEVEL_0`, `OB_RDP_LEVEL_1` 等，这些常量用于配置 FLASH 的保护和用户设置。
*   状态标志 (Status Flags)，用于指示 FLASH 操作的状态，如 `FLASH_FLAG_BSY` (忙), `FLASH_FLAG_PGERR` (编程错误), `FLASH_FLAG_WRPERR` (写保护错误), `FLASH_FLAG_EOP` (操作结束)。
*   中断使能位，例如 `FLASH_IT_EOP` (结束操作中断), `FLASH_IT_ERR` (错误中断)。

**8. 导出宏**

```c
/** @defgroup FLASHEx_Exported_Macros FLASHEx Exported Macros
  * @{
  */

/** @defgroup FLASH_Interrupt Interrupt
 *  @brief macros to handle FLASH interrupts
 * @{
 */

#if defined(FLASH_BANK2_END)
/**
  * @brief  Enable the specified FLASH interrupt.
  * @param  __INTERRUPT__  FLASH interrupt
  *     This parameter can be any combination of the following values:
  *     @arg @ref FLASH_IT_EOP_BANK1 End of FLASH Operation Interrupt on bank1
  *     @arg @ref FLASH_IT_ERR_BANK1 Error Interrupt on bank1
  *     @arg @ref FLASH_IT_EOP_BANK2 End of FLASH Operation Interrupt on bank2
  *     @arg @ref FLASH_IT_ERR_BANK2 Error Interrupt on bank2
  * @retval none
  */
#define __HAL_FLASH_ENABLE_IT(__INTERRUPT__)  do { \
                          /* Enable Bank1 IT */ \
                          SET_BIT(FLASH->CR, ((__INTERRUPT__) & 0x0000FFFFU)); \
                          /* Enable Bank2 IT */ \
                          SET_BIT(FLASH->CR2, ((__INTERRUPT__) >> 16U)); \
                    } while(0U)

// ... (其他宏定义，如禁用中断，获取标志位，清除标志位)

#else
/**
  * @brief  Enable the specified FLASH interrupt.
  * @param  __INTERRUPT__  FLASH interrupt
  *         This parameter can be any combination of the following values:
  *     @arg @ref FLASH_IT_EOP End of FLASH Operation Interrupt
  *     @arg @ref FLASH_IT_ERR Error Interrupt
  * @retval none
  */
#define __HAL_FLASH_ENABLE_IT(__INTERRUPT__)  (FLASH->CR |= (__INTERRUPT__))

// ... (其他宏定义，如禁用中断，获取标志位，清除标志位)

#endif

/**
  * @}
  */

/**
  * @}
  */
```

这些宏用于操作 FLASH 相关的寄存器，简化了中断使能、标志位读取和清除等操作。

*   `__HAL_FLASH_ENABLE_IT(__INTERRUPT__)`:  使能指定的 FLASH 中断。根据设备是否支持多个存储体 (FLASH_BANK2_END)，操作不同的寄存器 (FLASH->CR 或 FLASH->CR 和 FLASH->CR2)。
*   `__HAL_FLASH_DISABLE_IT(__INTERRUPT__)`: 禁用指定的 FLASH 中断。
*   `__HAL_FLASH_GET_FLAG(__FLAG__)`: 获取指定的 FLASH 标志位状态。
*   `__HAL_FLASH_CLEAR_FLAG(__FLAG__)`: 清除指定的 FLASH 标志位。

**示例代码：使能 FLASH 结束操作中断**

```c
__HAL_FLASH_ENABLE_IT(FLASH_IT_EOP); // 使能 FLASH 结束操作中断
// 之后需要在中断处理函数中处理 FLASH 操作完成事件.
```

**9. 导出函数声明**

```c
/** @addtogroup FLASHEx_Exported_Functions
  * @{
  */

/** @addtogroup FLASHEx_Exported_Functions_Group1
  * @{
  */
/* IO operation functions *****************************************************/
HAL_StatusTypeDef  HAL_FLASHEx_Erase(FLASH_EraseInitTypeDef *pEraseInit, uint32_t *PageError);
HAL_StatusTypeDef  HAL_FLASHEx_Erase_IT(FLASH_EraseInitTypeDef *pEraseInit);

/**
  * @}
  */

/** @addtogroup FLASHEx_Exported_Functions_Group2
  * @{
  */
/* Peripheral Control functions ***********************************************/
HAL_StatusTypeDef  HAL_FLASHEx_OBErase(void);
HAL_StatusTypeDef  HAL_FLASHEx_OBProgram(FLASH_OBProgramInitTypeDef *pOBInit);
void               HAL_FLASHEx_OBGetConfig(FLASH_OBProgramInitTypeDef *pOBInit);
uint32_t           HAL_FLASHEx_OBGetUserData(uint32_t DATAAdress);
/**
  * @}
  */

/**
  * @}
  */
```

这部分声明了 FLASHEx 模块提供的函数。

*   `HAL_FLASHEx_Erase()`: 擦除 FLASH 存储器。
*   `HAL_FLASHEx_Erase_IT()`: 以中断方式擦除 FLASH 存储器。
*   `HAL_FLASHEx_OBErase()`: 擦除选项字节。
*   `HAL_FLASHEx_OBProgram()`: 编程选项字节。
*   `HAL_FLASHEx_OBGetConfig()`: 获取选项字节配置。
*   `HAL_FLASHEx_OBGetUserData()`: 获取用户数据选项字节。

**示例代码：使用 HAL_FLASHEx_OBProgram() 配置读保护**

```c
FLASH_OBProgramInitTypeDef OBProgramInitStruct;
HAL_StatusTypeDef status;

/* Unlock the Flash to enable the flash control register access *************/
HAL_FLASH_Unlock();

/* Unlock the Options Bytes *************************************************/
HAL_FLASH_OB_Unlock();

/* Get the Option bytes configuration **********************************************/
HAL_FLASHEx_OBGetConfig(&OBProgramInitStruct);

/* Fill the structure */
OBProgramInitStruct.OptionType = OPTIONBYTE_RDP;
OBProgramInitStruct.RDPLevel = OB_RDP_LEVEL_1;

status = HAL_FLASHEx_OBProgram(&OBProgramInitStruct);

if (status == HAL_OK)
{
  /* Start the Option Bytes programming process ***********************************/
  status = HAL_FLASH_OB_Launch();

  /* FLASH_OB_Launch() can return ERROR if the Option Bytes are locked, the error code can be retrieved with: */
  /* errorCode = HAL_FLASH_GetError()*/
}

/* Lock the Options Bytes *************************************************/
HAL_FLASH_OB_Lock();

/* Lock the Flash to disable the flash control register access (recommended
  to protect the FLASH memory against possible unwanted operation) *********/
HAL_FLASH_Lock();

if (status != HAL_OK) {
    printf("Error programming option bytes!\n");
    return HAL_ERROR;
}
```

**总结**

`stm32f1xx_hal_flash_ex.h` 头文件为 STM32F1 系列微控制器的 FLASH 存储器扩展功能提供了必要的定义和函数声明。  通过使用该头文件中定义的结构体、常量和函数，可以方便地进行 FLASH 擦除、选项字节编程和读取等操作，从而配置微控制器的 FLASH 存储器和相关功能。  需要注意的是，对 FLASH 进行操作有风险，务必仔细阅读相关文档，并在充分理解后进行操作，避免造成数据丢失或设备损坏。
