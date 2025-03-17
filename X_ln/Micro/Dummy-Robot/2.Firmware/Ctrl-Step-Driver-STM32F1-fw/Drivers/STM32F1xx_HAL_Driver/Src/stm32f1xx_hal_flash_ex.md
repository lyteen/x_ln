Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_flash_ex.c`

好的，重新开始。我将逐步解释 STM32F1xx HAL 库中 `stm32f1xx_hal_flash_ex.c` 文件的关键部分，并提供中文解释和代码示例。

**文件总览 (File Overview)**

`stm32f1xx_hal_flash_ex.c` 是 STM32F1 系列单片机 FLASH 存储器扩展 HAL (Hardware Abstraction Layer) 驱动程序的一部分。它提供了对标准 FLASH HAL 驱动程序功能的补充，例如设置/重置写保护，编程用户选项字节和获取读取保护级别。 让我们逐步分解代码。

**1. 包含头文件 (Include Header File)**

```c
#include "stm32f1xx_hal.h"
```

**解释:**
*   `#include "stm32f1xx_hal.h"`:  包含 STM32F1xx HAL 库的主头文件。这个头文件定义了访问底层硬件寄存器所需的各种数据结构、宏和函数。

**2. 模块使能宏 (Module Enable Macro)**

```c
#ifdef HAL_FLASH_MODULE_ENABLED
```

**解释:**
*   `#ifdef HAL_FLASH_MODULE_ENABLED`: 这是一个预处理器指令，检查是否定义了 `HAL_FLASH_MODULE_ENABLED` 宏。 如果定义了，则编译该文件中的代码。 这是 HAL 库中常用的方法，用于有条件地编译模块，从而允许用户启用或禁用特定的外设驱动程序。

**3. 组定义 (Group Definitions)**

```c
/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @addtogroup FLASH
  * @{
  */

/** @addtogroup FLASH_Private_Variables
 * @{
 */
/* Variables used for Erase pages under interruption*/
extern FLASH_ProcessTypeDef pFlash;
/**
  * @}
  */

/**
  * @}
  */

/** @defgroup FLASHEx FLASHEx
  * @brief FLASH HAL Extension module driver
  * @{
  */
```

**解释:**

*   `@addtogroup`, `@defgroup`：这些是 Doxygen 标记，用于组织和记录代码。 它们定义了模块、子模块和代码组之间的层次结构。`FLASHEx` 组表示 FLASH HAL 扩展驱动程序。
*   `extern FLASH_ProcessTypeDef pFlash;`: 声明了一个外部变量 `pFlash`，类型为 `FLASH_ProcessTypeDef`。 这很可能是一个结构体，用于跟踪 FLASH 操作的状态（例如，擦除或编程），尤其是在中断上下文中。`extern` 关键字表示该变量在其他地方定义。

**4. 私有常量 (Private Constants)**

```c
/** @defgroup FLASHEx_Private_Constants FLASHEx Private Constants
 * @{
 */
#define FLASH_POSITION_IWDGSW_BIT        FLASH_OBR_IWDG_SW_Pos
#define FLASH_POSITION_OB_USERDATA0_BIT  FLASH_OBR_DATA0_Pos
#define FLASH_POSITION_OB_USERDATA1_BIT  FLASH_OBR_DATA1_Pos
/**
  * @}
  */
```

**解释:**
*   `#define`: 定义了一些宏常量。 这些常量用于访问 FLASH 选项字节 (Option Bytes) 寄存器中的特定位域。例如，`FLASH_POSITION_IWDGSW_BIT` 表示 IWDG（独立看门狗）软件启用位在选项字节寄存器中的位置。
*   `FLASH_OBR_IWDG_SW_Pos`，`FLASH_OBR_DATA0_Pos`，`FLASH_OBR_DATA1_Pos`：这些常量可能是在其他头文件中定义的，表示相应位域在寄存器中的位偏移量。

**5. 私有函数原型 (Private Function Prototypes)**

```c
/** @defgroup FLASHEx_Private_Functions FLASHEx Private Functions
 * @{
 */
/* Erase operations */
static void              FLASH_MassErase(uint32_t Banks);
void    FLASH_PageErase(uint32_t PageAddress);

/* Option bytes control */
static HAL_StatusTypeDef FLASH_OB_EnableWRP(uint32_t WriteProtectPage);
static HAL_StatusTypeDef FLASH_OB_DisableWRP(uint32_t WriteProtectPage);
static HAL_StatusTypeDef FLASH_OB_RDP_LevelConfig(uint8_t ReadProtectLevel);
static HAL_StatusTypeDef FLASH_OB_UserConfig(uint8_t UserConfig);
static HAL_StatusTypeDef FLASH_OB_ProgramData(uint32_t Address, uint8_t Data);
static uint32_t          FLASH_OB_GetWRP(void);
static uint32_t          FLASH_OB_GetRDP(void);
static uint8_t           FLASH_OB_GetUser(void);

/**
  * @}
  */
```

**解释:**
*   `static`: 这些函数原型声明了 `FLASHEx` 模块内部使用的私有函数。 `static` 关键字限制了这些函数的作用域，使它们只能在当前文件中访问。
*   **Erase Operations (擦除操作):**
    *   `FLASH_MassErase`: 用于执行全片擦除操作。
    *   `FLASH_PageErase`: 用于擦除指定的 FLASH 页。
*   **Option Bytes Control (选项字节控制):**
    *   `FLASH_OB_EnableWRP`: 启用写保护。
    *   `FLASH_OB_DisableWRP`: 禁用写保护。
    *   `FLASH_OB_RDP_LevelConfig`: 配置读保护级别。
    *   `FLASH_OB_UserConfig`: 配置用户选项字节。
    *   `FLASH_OB_ProgramData`: 编程选项字节数据。
    *   `FLASH_OB_GetWRP`: 获取写保护状态。
    *   `FLASH_OB_GetRDP`: 获取读保护级别。
    *   `FLASH_OB_GetUser`: 获取用户选项字节值。

**6. 导出函数 (Exported Functions)**

这部分代码定义了可以从 HAL 驱动程序外部调用的函数。

**6.1 FLASH 擦除函数 (FLASH Erasing Functions)**

```c
/** @defgroup FLASHEx_Exported_Functions_Group1 FLASHEx Memory Erasing functions
 *  @brief   FLASH Memory Erasing functions
  * @{
  */

/**
  * @brief  Perform a mass erase or erase the specified FLASH memory pages
  * @param  pEraseInit pointer to an FLASH_EraseInitTypeDef structure
  * @param  PageError pointer to variable that contains configuration information
  * @retval HAL_StatusTypeDef HAL Status
  */
HAL_StatusTypeDef HAL_FLASHEx_Erase(FLASH_EraseInitTypeDef *pEraseInit, uint32_t *PageError) { ... }

/**
  * @brief  Perform a mass erase or erase the specified FLASH memory pages with interrupt enabled
  * @param  pEraseInit pointer to an FLASH_EraseInitTypeDef structure
  * @retval HAL_StatusTypeDef HAL Status
  */
HAL_StatusTypeDef HAL_FLASHEx_Erase_IT(FLASH_EraseInitTypeDef *pEraseInit) { ... }

/**
  * @}
  */
```

**解释:**

*   `HAL_FLASHEx_Erase`:  同步 FLASH 擦除函数。它接受一个 `FLASH_EraseInitTypeDef` 结构体指针，该结构体定义了擦除操作的类型（全片擦除或页擦除），以及要擦除的页的地址和数量。  `PageError` 参数是一个输出参数，用于存储擦除过程中出错的页的地址。
*   `HAL_FLASHEx_Erase_IT`:  中断驱动的 FLASH 擦除函数。 它以非阻塞方式执行擦除操作，并使用中断来通知应用程序擦除操作已完成。

**示例 (HAL_FLASHEx_Erase 示例):**

```c
#include "stm32f1xx_hal.h"

void FlashEraseExample(uint32_t StartPageAddress, uint32_t NumberOfPages) {
  HAL_FLASH_Unlock(); // 解锁 FLASH 访问

  FLASH_EraseInitTypeDef EraseInitStruct;
  uint32_t PageError = 0;

  EraseInitStruct.TypeErase   = FLASH_TYPEERASE_PAGES; // 页擦除
  EraseInitStruct.PageAddress = StartPageAddress; // 起始页地址
  EraseInitStruct.NbPages     = NumberOfPages;      // 要擦除的页数
  EraseInitStruct.Banks       = FLASH_BANK_1;         // FLASH Bank 1

  if (HAL_FLASHEx_Erase(&EraseInitStruct, &PageError) != HAL_OK) {
    // 擦除出错处理
    if (PageError != 0xFFFFFFFF) {
      // 在地址 PageError 发生错误
      printf("Flash Erase Error at Page: 0x%08lX\r\n", PageError);
    }
  } else {
    printf("Flash Erase Successful!\r\n");
  }

  HAL_FLASH_Lock(); // 锁定 FLASH 访问
}

// 如何使用：
int main(void) {
  // ... 初始化代码

  // 假设要从地址 0x08004000 开始擦除 4 页
  FlashEraseExample(0x08004000, 4);

  while (1) {
    // ...
  }
}
```

**代码解释:**

1.  **包含头文件:** 包含必要的 HAL 库头文件。
2.  **解锁 FLASH:**  `HAL_FLASH_Unlock()` 函数解锁 FLASH 存储器的访问，允许进行擦除和编程操作。
3.  **配置擦除结构体:**  `FLASH_EraseInitTypeDef` 结构体用于配置擦除操作。
    *   `TypeErase = FLASH_TYPEERASE_PAGES;`：指定执行页擦除。
    *   `PageAddress = StartPageAddress;`：指定要擦除的起始页的地址。
    *   `NbPages = NumberOfPages;`：指定要擦除的页数。
    *   `Banks = FLASH_BANK_1;`： 选择BANK1。
4.  **执行擦除操作:**  `HAL_FLASHEx_Erase()` 函数执行实际的 FLASH 擦除操作。 它接受配置的 `EraseInitStruct` 结构体和 `PageError` 指针作为参数。
5.  **错误处理:**  如果擦除操作失败，`HAL_FLASHEx_Erase()` 函数返回 `HAL_ERROR`。 `PageError` 变量将包含擦除过程中出错的页的地址。
6.  **锁定 FLASH:**  `HAL_FLASH_Lock()` 函数重新锁定 FLASH 存储器的访问，防止意外的写入操作。

**6.2 选项字节编程函数 (Option Bytes Programming Functions)**

```c
/** @defgroup FLASHEx_Exported_Functions_Group2 Option Bytes Programming functions
 *  @brief   Option Bytes Programming functions
  * @{
  */

/**
  * @brief  Erases the FLASH option bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FLASHEx_OBErase(void) { ... }

/**
  * @brief  Program option bytes
  * @param  pOBInit pointer to an FLASH_OBInitStruct structure
  * @retval HAL_StatusTypeDef HAL Status
  */
HAL_StatusTypeDef HAL_FLASHEx_OBProgram(FLASH_OBProgramInitTypeDef *pOBInit) { ... }

/**
  * @brief  Get the Option byte configuration
  * @param  pOBInit pointer to an FLASH_OBInitStruct structure
  * @retval None
  */
void HAL_FLASHEx_OBGetConfig(FLASH_OBProgramInitTypeDef *pOBInit) { ... }

/**
  * @brief  Get the Option byte user data
  * @param  DATAAdress Address of the option byte DATA
  * @retval Value programmed in USER data
  */
uint32_t HAL_FLASHEx_OBGetUserData(uint32_t DATAAdress) { ... }

/**
  * @}
  */
```

**解释:**

*   `HAL_FLASHEx_OBErase()`: 擦除 FLASH 选项字节。这将重置所有选项字节为它们的默认值（除了读保护级别）。
*   `HAL_FLASHEx_OBProgram()`: 编程 FLASH 选项字节。 它接受一个 `FLASH_OBProgramInitTypeDef` 结构体指针，该结构体定义了要编程的选项字节的类型（写保护、读保护、用户配置），以及要设置的值。
*   `HAL_FLASHEx_OBGetConfig()`: 获取当前选项字节配置。它填充一个 `FLASH_OBProgramInitTypeDef` 结构体，其中包含当前选项字节的值。
*   `HAL_FLASHEx_OBGetUserData()`: 获取用户选项字节数据。

**示例 (HAL_FLASHEx_OBProgram 示例):**

```c
#include "stm32f1xx_hal.h"

void OptionBytesProgramExample(void) {
  HAL_FLASH_Unlock();       // 解锁 FLASH 访问
  HAL_FLASH_OB_Unlock();    // 解锁选项字节访问

  FLASH_OBProgramInitTypeDef OBProgramInitStruct;

  // 配置选项字节
  OBProgramInitStruct.OptionType = OPTIONBYTE_USER; // 配置用户选项字节
  OBProgramInitStruct.USERConfig = OB_IWDG_SW | OB_STOP_NO_RST | OB_STDBY_NO_RST; // 配置 IWDG 源和复位行为

  if (HAL_FLASHEx_OBProgram(&OBProgramInitStruct) != HAL_OK) {
    // 编程出错处理
    printf("Option Bytes Program Error!\r\n");
  } else {
    printf("Option Bytes Program Successful!\r\n");

    // 启动选项字节加载 (需要复位)
    HAL_FLASH_OB_Launch();
  }

  HAL_FLASH_OB_Lock();      // 锁定选项字节访问
  HAL_FLASH_Lock();         // 锁定 FLASH 访问
}

int main(void) {
  // ... 初始化代码

  OptionBytesProgramExample();

  while (1) {
    // ...
  }
}
```

**代码解释:**

1.  **包含头文件:** 包含必要的 HAL 库头文件。
2.  **解锁 FLASH 和选项字节:** `HAL_FLASH_Unlock()` 解锁 FLASH 存储器的访问，`HAL_FLASH_OB_Unlock()` 解锁选项字节的访问。
3.  **配置选项字节结构体:** `FLASH_OBProgramInitTypeDef` 结构体用于配置要编程的选项字节。
    *   `OptionType = OPTIONBYTE_USER;`:  指定要配置用户选项字节。
    *   `USERConfig = OB_IWDG_SW | OB_STOP_NO_RST | OB_STDBY_NO_RST;`:
        *   `OB_IWDG_SW`: 配置独立看门狗 (IWDG) 由软件启用。
        *   `OB_STOP_NO_RST`:  配置在 STOP 模式下不复位。
        *   `OB_STDBY_NO_RST`: 配置在 STANDBY 模式下不复位。
4.  **编程选项字节:** `HAL_FLASHEx_OBProgram()` 函数执行实际的选项字节编程。
5.  **启动选项字节加载:** `HAL_FLASH_OB_Launch()` 函数启动选项字节的加载。  **注意：** 这通常会导致系统复位，因为选项字节的更改需要重新启动才能生效。
6.  **锁定 FLASH 和选项字节:** `HAL_FLASH_OB_Lock()` 和 `HAL_FLASH_Lock()` 函数重新锁定 FLASH 存储器和选项字节的访问，防止意外的写入操作。

**7. 私有函数实现 (Private Function Implementations)**

这部分代码实现了前面声明的私有函数。 这些函数执行实际的 FLASH 操作和选项字节编程。

**示例 (FLASH\_MassErase 函数):**

```c
static void FLASH_MassErase(uint32_t Banks) {
  /* Check the parameters */
  assert_param(IS_FLASH_BANK(Banks));

  /* Clean the error context */
  pFlash.ErrorCode = HAL_FLASH_ERROR_NONE;

#if defined(FLASH_BANK2_END)
  if (Banks == FLASH_BANK_BOTH) {
    /* bank1 & bank2 will be erased*/
    SET_BIT(FLASH->CR, FLASH_CR_MER);
    SET_BIT(FLASH->CR2, FLASH_CR2_MER);
    SET_BIT(FLASH->CR, FLASH_CR_STRT);
    SET_BIT(FLASH->CR2, FLASH_CR2_STRT);
  } else if (Banks == FLASH_BANK_2) {
    /*Only bank2 will be erased*/
    SET_BIT(FLASH->CR2, FLASH_CR2_MER);
    SET_BIT(FLASH->CR2, FLASH_CR2_STRT);
  } else {
#endif /* FLASH_BANK2_END */
#if !defined(FLASH_BANK2_END)
    /* Prevent unused argument(s) compilation warning */
    UNUSED(Banks);
#endif /* FLASH_BANK2_END */
    /* Only bank1 will be erased*/
    SET_BIT(FLASH->CR, FLASH_CR_MER);
    SET_BIT(FLASH->CR, FLASH_CR_STRT);
#if defined(FLASH_BANK2_END)
  }
#endif /* FLASH_BANK2_END */
}
```

**解释:**

1.  **参数检查:**  `assert_param(IS_FLASH_BANK(Banks));` 宏用于检查 `Banks` 参数的有效性。 `IS_FLASH_BANK` 应该是一个宏，用于验证 `Banks` 是否是有效值（例如，`FLASH_BANK_1`，`FLASH_BANK_2`，`FLASH_BANK_BOTH`）。
2.  **清除错误上下文:** `pFlash.ErrorCode = HAL_FLASH_ERROR_NONE;` 清除全局 `pFlash` 变量中的任何先前错误。
3.  **双 Bank 支持 (如果可用):** `#if defined(FLASH_BANK2_END)` 块中的代码处理具有两个 FLASH Bank 的 STM32F1 器件。 根据 `Banks` 参数的值，它设置适当的控制寄存器位来执行全片擦除。
4.  **单 Bank 器件:**  `#if !defined(FLASH_BANK2_END)` 块中的代码处理只有一个 FLASH Bank 的器件。在这种情况下，它设置 BANK1的寄存器
5.  **设置控制位:**
    *   `SET_BIT(FLASH->CR, FLASH_CR_MER);`：设置 FLASH 控制寄存器 (CR) 中的 `MER`（Mass Erase）位，以启用全片擦除模式。
    *   `SET_BIT(FLASH->CR, FLASH_CR_STRT);`：设置 FLASH 控制寄存器 (CR) 中的 `STRT`（Start）位，以启动擦除操作。

**总结 (Summary)**

`stm32f1xx_hal_flash_ex.c` 文件提供了一组扩展函数，用于控制 STM32F1 系列微控制器的 FLASH 存储器。 这些函数允许你擦除 FLASH 存储器、编程选项字节以及配置读写保护。 了解如何使用这些函数对于安全地存储和更新 STM32F1 微控制器的固件至关重要。

希望这个更详细的解释对你有所帮助！
