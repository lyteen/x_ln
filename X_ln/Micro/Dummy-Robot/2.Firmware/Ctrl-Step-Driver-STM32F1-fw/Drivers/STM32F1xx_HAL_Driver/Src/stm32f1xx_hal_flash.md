Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_flash.c`

好的，我们重新开始。我会对这段代码的关键部分进行解释，生成每一小段代码，并附上简短的中文描述。我也会包含代码如何使用的简短描述，并提供简单的示例。

**总览**

这段代码是STM32F1xx系列单片机 FLASH HAL (Hardware Abstraction Layer) 模块的驱动程序，用于管理芯片内部 FLASH 存储器的编程、擦除、保护等操作。它提供了一组函数和宏，方便用户在应用层进行 FLASH 操作。

**1. 头文件包含 (Includes)**

```c
#include "stm32f1xx_hal.h"
```

**描述:** 包含STM32F1xx的HAL库头文件。所有HAL库的函数，类型定义和宏定义都可以在这个文件中找到。
**用途:** 这是所有使用HAL库的程序都必须包含的，相当于C语言标准库的`stdio.h`。

**2. 模块使能宏 (Module Enable Macro)**

```c
#ifdef HAL_FLASH_MODULE_ENABLED
...
#endif /* HAL_FLASH_MODULE_ENABLED */
```

**描述:** 使用预编译宏，只有当`HAL_FLASH_MODULE_ENABLED`被定义时，才会编译这段代码。
**用途:** 这种机制允许用户在编译时选择是否包含某个HAL模块，从而减小代码体积，并避免不必要的依赖。

**3. 类型定义和宏定义 (Private typedef & define)**

```c
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @defgroup FLASH_Private_Constants FLASH Private Constants
  * @{
  */
/**
  * @}
  */

/* Private macro ---------------------------- ---------------------------------*/
/** @defgroup FLASH_Private_Macros FLASH Private Macros
  * @{
  */
 
/**
  * @}
  */
```

**描述:**  这部分代码定义了 FLASH 模块内部使用的私有类型和常量、宏，但在这个代码片段中，它们是空的，可能是为以后的扩展预留。
**用途:** 私有类型和宏常量仅供模块内部使用，对外部用户隐藏，有助于模块的封装和信息隐藏。

**4. 私有变量 (Private Variables)**

```c
/* Private variables ---------------------------------------------------------*/
/** @defgroup FLASH_Private_Variables FLASH Private Variables
  * @{
  */
/* Variables used for Erase pages under interruption*/
FLASH_ProcessTypeDef pFlash;
/**
  * @}
  */
```

**描述:** 定义了`pFlash`变量，类型为`FLASH_ProcessTypeDef`，用于在中断模式下擦除页面的操作。
**用途:** `pFlash`是一个全局变量，用于保存FLASH操作的状态信息，例如当前操作的地址、数据、剩余数据量等。在中断处理函数中，需要访问这些信息来继续执行FLASH操作。

**5. 私有函数原型 (Private Function Prototypes)**

```c
/* Private function prototypes -----------------------------------------------*/
/** @defgroup FLASH_Private_Functions FLASH Private Functions
  * @{
  */
static  void   FLASH_Program_HalfWord(uint32_t Address, uint16_t Data);
static  void   FLASH_SetErrorCode(void);
extern void    FLASH_PageErase(uint32_t PageAddress);
/**
  * @}
  */
```

**描述:** 声明了FLASH模块内部使用的私有函数。
*   `FLASH_Program_HalfWord`: 用于将半字（16位）数据编程到指定的FLASH地址。
*   `FLASH_SetErrorCode`: 用于设置FLASH错误代码，记录发生的错误类型。
*   `FLASH_PageErase`: 用于擦除指定的FLASH页面。

**用途:** 这些函数是模块内部实现的细节，对外部用户隐藏，只能通过提供的公共函数来调用。

**6. HAL_FLASH_Program 函数 (公共函数)**

```c
HAL_StatusTypeDef HAL_FLASH_Program(uint32_t TypeProgram, uint32_t Address, uint64_t Data)
{
  HAL_StatusTypeDef status = HAL_ERROR;
  uint8_t index = 0;
  uint8_t nbiterations = 0;
  
  /* Process Locked */
  __HAL_LOCK(&pFlash);

  /* Check the parameters */
  assert_param(IS_FLASH_TYPEPROGRAM(TypeProgram));
  assert_param(IS_FLASH_PROGRAM_ADDRESS(Address));

#if defined(FLASH_BANK2_END)
  if(Address <= FLASH_BANK1_END)
  {
#endif /* FLASH_BANK2_END */
    /* Wait for last operation to be completed */
    status = FLASH_WaitForLastOperation(FLASH_TIMEOUT_VALUE);
#if defined(FLASH_BANK2_END)
  }
  else
  {
    /* Wait for last operation to be completed */
    status = FLASH_WaitForLastOperationBank2(FLASH_TIMEOUT_VALUE);
  }
#endif /* FLASH_BANK2_END */
  
  if(status == HAL_OK)
  {
    if(TypeProgram == FLASH_TYPEPROGRAM_HALFWORD)
    {
      /* Program halfword (16-bit) at a specified address. */
      nbiterations = 1U;
    }
    else if(TypeProgram == FLASH_TYPEPROGRAM_WORD)
    {
      /* Program word (32-bit = 2*16-bit) at a specified address. */
      nbiterations = 2U;
    }
    else
    {
      /* Program double word (64-bit = 4*16-bit) at a specified address. */
      nbiterations = 4U;
    }

    for (index = 0U; index < nbiterations; index++)
    {
      FLASH_Program_HalfWord((Address + (2U*index)), (uint16_t)(Data >> (16U*index)));

#if defined(FLASH_BANK2_END)
      if(Address <= FLASH_BANK1_END)
      {
#endif /* FLASH_BANK2_END */
        /* Wait for last operation to be completed */
        status = FLASH_WaitForLastOperation(FLASH_TIMEOUT_VALUE);
    
        /* If the program operation is completed, disable the PG Bit */
        CLEAR_BIT(FLASH->CR, FLASH_CR_PG);
#if defined(FLASH_BANK2_END)
      }
      else
      {
        /* Wait for last operation to be completed */
        status = FLASH_WaitForLastOperationBank2(FLASH_TIMEOUT_VALUE);
        
        /* If the program operation is completed, disable the PG Bit */
        CLEAR_BIT(FLASH->CR2, FLASH_CR2_PG);
      }
#endif /* FLASH_BANK2_END */
      /* In case of error, stop programation procedure */
      if (status != HAL_OK)
      {
        break;
      }
    }
  }

  /* Process Unlocked */
  __HAL_UNLOCK(&pFlash);

  return status;
}
```

**描述:**  该函数用于将半字、字或双字的数据编程到指定的 FLASH 地址。 它接受编程类型、地址和数据作为输入。  它首先锁定 FLASH 接口，然后等待上一个操作完成。 根据编程类型，它会调用私有函数 `FLASH_Program_HalfWord` 多次来完成编程。 最后，它解锁 FLASH 接口。

**中文解释：**
*   `HAL_StatusTypeDef HAL_FLASH_Program(uint32_t TypeProgram, uint32_t Address, uint64_t Data)`: 这是编程操作的公共函数，外部用户通过调用此函数来实现FLASH的编程。
*   `TypeProgram`: 指示编程类型，例如半字（16位）、字（32位）或双字（64位）。
*   `Address`: 指定要编程的FLASH地址。
*   `Data`:  指定要编程的数据。
*   `__HAL_LOCK(&pFlash)` 和 `__HAL_UNLOCK(&pFlash)`:  用于保护临界区，防止多个任务同时访问 FLASH 资源导致冲突。
*   `FLASH_WaitForLastOperation(FLASH_TIMEOUT_VALUE)`: 等待上一次FLASH操作完成，确保当前操作不会被干扰。
*   `FLASH_Program_HalfWord((Address + (2U*index)), (uint16_t)(Data >> (16U*index)))`: 调用私有函数，将数据按半字编程到指定的FLASH地址。
*   `CLEAR_BIT(FLASH->CR, FLASH_CR_PG)`: 清除FLASH控制寄存器中的编程位，表示编程操作完成。

**用途:**  这是最常用的 FLASH 编程函数，应用程序通过调用它来实现数据的写入。

**示例代码：**

```c
#include "stm32f1xx_hal.h"

extern void SystemClock_Config(void); // 假设你有一个配置系统时钟的函数

int main(void) {
  HAL_Init(); // 初始化HAL库
  SystemClock_Config(); // 配置系统时钟

  // 定义要编程的地址和数据
  uint32_t address = 0x08008000; // 示例地址，需要确保该地址是可写的
  uint64_t data = 0x123456789ABCDEF0; // 示例数据

  // 1. 解锁FLASH
  if (HAL_FLASH_Unlock() != HAL_OK) {
    // 解锁失败，处理错误
    while (1); // 停在这里，指示错误
  }

  // 2. 编程FLASH
  if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, address, data) != HAL_OK) {
    // 编程失败，处理错误
    while (1); // 停在这里，指示错误
  }

  // 3. 锁定FLASH
  if (HAL_FLASH_Lock() != HAL_OK) {
    // 锁定失败，处理错误
    while (1); // 停在这里，指示错误
  }

  // 编程成功
  while (1) {
    // 应用程序的其他逻辑
  }
}
```

**解释：**

1.  先初始化HAL库和系统时钟。
2.  定义要写入的FLASH地址和数据。注意，这里的地址 `0x08008000` 只是一个例子，你需要根据你的STM32F1xx芯片的FLASH地址范围进行调整。 并且，编程之前通常需要先擦除对应的FLASH页。
3.  调用`HAL_FLASH_Unlock()`解锁FLASH，允许写入操作。
4.  调用`HAL_FLASH_Program()`进行编程，这里使用`FLASH_TYPEPROGRAM_DOUBLEWORD`表示写入双字数据。
5.  调用`HAL_FLASH_Lock()`锁定FLASH，防止误写入。
6.  如果所有步骤都成功，程序会进入一个无限循环，在这里可以添加你的应用程序的其他逻辑。  如果任何步骤失败，程序会停在一个无限循环中，提示发生错误。

**7. HAL_FLASH_Program_IT 函数 (公共函数 - 中断模式)**

```c
HAL_StatusTypeDef HAL_FLASH_Program_IT(uint32_t TypeProgram, uint32_t Address, uint64_t Data)
{
  HAL_StatusTypeDef status = HAL_OK;
  
  /* Process Locked */
  __HAL_LOCK(&pFlash);

  /* Check the parameters */
  assert_param(IS_FLASH_TYPEPROGRAM(TypeProgram));
  assert_param(IS_FLASH_PROGRAM_ADDRESS(Address));

#if defined(FLASH_BANK2_END)
  /* If procedure already ongoing, reject the next one */
  if (pFlash.ProcedureOnGoing != FLASH_PROC_NONE)
  {
    return HAL_ERROR;
  }
  
  if(Address <= FLASH_BANK1_END)
  {
    /* Enable End of FLASH Operation and Error source interrupts */
    __HAL_FLASH_ENABLE_IT(FLASH_IT_EOP_BANK1 | FLASH_IT_ERR_BANK1);

  }else
  {
    /* Enable End of FLASH Operation and Error source interrupts */
    __HAL_FLASH_ENABLE_IT(FLASH_IT_EOP_BANK2 | FLASH_IT_ERR_BANK2);
  }
#else
  /* Enable End of FLASH Operation and Error source interrupts */
  __HAL_FLASH_ENABLE_IT(FLASH_IT_EOP | FLASH_IT_ERR);
#endif /* FLASH_BANK2_END */
  
  pFlash.Address = Address;
  pFlash.Data = Data;

  if(TypeProgram == FLASH_TYPEPROGRAM_HALFWORD)
  {
    pFlash.ProcedureOnGoing = FLASH_PROC_PROGRAMHALFWORD;
    /* Program halfword (16-bit) at a specified address. */
    pFlash.DataRemaining = 1U;
  }
  else if(TypeProgram == FLASH_TYPEPROGRAM_WORD)
  {
    pFlash.ProcedureOnGoing = FLASH_PROC_PROGRAMWORD;
    /* Program word (32-bit : 2*16-bit) at a specified address. */
    pFlash.DataRemaining = 2U;
  }
  else
  {
    pFlash.ProcedureOnGoing = FLASH_PROC_PROGRAMDOUBLEWORD;
    /* Program double word (64-bit : 4*16-bit) at a specified address. */
    pFlash.DataRemaining = 4U;
  }

  /* Program halfword (16-bit) at a specified address. */
  FLASH_Program_HalfWord(Address, (uint16_t)Data);

  return status;
}
```

**描述:** 该函数以中断驱动的方式编程 FLASH。 它首先锁定 FLASH 接口，检查是否有其他程序正在进行，然后启用 FLASH 中断（完成中断和错误中断）。 然后，它将编程类型、地址和数据存储到全局变量 `pFlash` 中。最后，它调用私有函数 `FLASH_Program_HalfWord` 开始编程。

**中文解释:**
*   `HAL_FLASH_Program_IT(uint32_t TypeProgram, uint32_t Address, uint64_t Data)`:  这是中断模式下的 FLASH 编程函数。
*   `pFlash.ProcedureOnGoing != FLASH_PROC_NONE`:  检查是否已经有FLASH操作正在进行，避免冲突。
*   `__HAL_FLASH_ENABLE_IT(FLASH_IT_EOP | FLASH_IT_ERR)`:  使能FLASH中断，包括操作完成中断（EOP）和错误中断（ERR）。
*   `pFlash.Address = Address; pFlash.Data = Data;`:  将编程的地址和数据保存到全局变量`pFlash`中，供中断处理函数使用。
*   `pFlash.ProcedureOnGoing = FLASH_PROC_PROGRAMHALFWORD; ... pFlash.DataRemaining = ...`:  设置当前操作类型和剩余数据量。

**用途:** 这种中断模式的编程方式，可以提高CPU的利用率，在FLASH编程的同时，还可以执行其他的任务。

**示例代码：**

```c
#include "stm32f1xx_hal.h"

extern void SystemClock_Config(void);

// FLASH操作完成的回调函数
void HAL_FLASH_EndOfOperationCallback(uint32_t ReturnValue) {
  // FLASH编程完成，在这里可以添加你的逻辑
  // 例如：设置一个标志，通知主循环FLASH编程完成
  static uint8_t flash_programmed = 1;
  (void)ReturnValue; // 避免编译警告
}

// FLASH操作出错的回调函数
void HAL_FLASH_OperationErrorCallback(uint32_t ReturnValue) {
  // FLASH编程出错，在这里可以添加你的逻辑
  // 例如：打印错误信息，尝试重新编程
  (void)ReturnValue; // 避免编译警告
  while(1); // 停在这里指示错误
}

int main(void) {
  HAL_Init();
  SystemClock_Config();

  // 定义要编程的地址和数据
  uint32_t address = 0x08008000;
  uint64_t data = 0x123456789ABCDEF0;

  // 1. 解锁FLASH
  if (HAL_FLASH_Unlock() != HAL_OK) {
    // 解锁失败，处理错误
    while (1);
  }

  // 2. 编程FLASH (中断模式)
  if (HAL_FLASH_Program_IT(FLASH_TYPEPROGRAM_DOUBLEWORD, address, data) != HAL_OK) {
    // 编程启动失败，处理错误
    while (1);
  }

  // 3.  等待中断处理完成，或者在主循环中做其他的事情
  while(HAL_FLASH_GetError() == HAL_FLASH_ERROR_NONE) {
    // 在这里可以执行其他的任务
  }

  // 4.  如果出错，这里会执行
  while(1);

}
```

**解释：**

1.  包含必要的头文件，并定义了系统时钟配置函数`SystemClock_Config()`。
2.  定义了两个回调函数：`HAL_FLASH_EndOfOperationCallback()`和`HAL_FLASH_OperationErrorCallback()`。这两个函数分别在FLASH操作完成和出错时被调用。
3.  在`main()`函数中，首先初始化HAL库和系统时钟。
4.  定义要写入的FLASH地址和数据。
5.  调用`HAL_FLASH_Unlock()`解锁FLASH，允许写入操作。
6.  调用`HAL_FLASH_Program_IT()`启动FLASH编程，并启用中断。
7.  在主循环中，程序可以执行其他任务，或者等待FLASH操作完成。  通过`HAL_FLASH_GetError()`可以检查是否发生错误。
8.  如果在回调函数中设置了标志`flash_programmed`，则可以在主循环中检测该标志，以确定FLASH编程是否完成。

**8. HAL_FLASH_IRQHandler 函数 (中断处理函数)**

```c
void HAL_FLASH_IRQHandler(void)
{
  uint32_t addresstmp = 0U;
  
  /* Check FLASH operation error flags */
#if defined(FLASH_BANK2_END)
  if(__HAL_FLASH_GET_FLAG(FLASH_FLAG_WRPERR_BANK1) || __HAL_FLASH_GET_FLAG(FLASH_FLAG_PGERR_BANK1) || \
    (__HAL_FLASH_GET_FLAG(FLASH_FLAG_WRPERR_BANK2) || __HAL_FLASH_GET_FLAG(FLASH_FLAG_PGERR_BANK2)))
#else
  if(__HAL_FLASH_GET_FLAG(FLASH_FLAG_WRPERR) ||__HAL_FLASH_GET_FLAG(FLASH_FLAG_PGERR))
#endif /* FLASH_BANK2_END */
  {
    /* Return the faulty address */
    addresstmp = pFlash.Address;
    /* Reset address */
    pFlash.Address = 0xFFFFFFFFU;
  
    /* Save the Error code */
    FLASH_SetErrorCode();
    
    /* FLASH error interrupt user callback */
    HAL_FLASH_OperationErrorCallback(addresstmp);

    /* Stop the procedure ongoing */
    pFlash.ProcedureOnGoing = FLASH_PROC_NONE;
  }

  /* Check FLASH End of Operation flag  */
#if defined(FLASH_BANK2_END)
  if(__HAL_FLASH_GET_FLAG(FLASH_FLAG_EOP_BANK1))
  {
    /* Clear FLASH End of Operation pending bit */
    __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP_BANK1);
#else
  if(__HAL_FLASH_GET_FLAG(FLASH_FLAG_EOP))
  {
    /* Clear FLASH End of Operation pending bit */
    __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP);
#endif /* FLASH_BANK2_END */
    
    /* Process can continue only if no error detected */
    if(pFlash.ProcedureOnGoing != FLASH_PROC_NONE)
    {
      if(pFlash.ProcedureOnGoing == FLASH_PROC_PAGEERASE)
      {
        /* Nb of pages to erased can be decreased */
        pFlash.DataRemaining--;

        /* Check if there are still pages to erase */
        if(pFlash.DataRemaining != 0U)
        {
          addresstmp = pFlash.Address;
          /*Indicate user which sector has been erased */
          HAL_FLASH_EndOfOperationCallback(addresstmp);

          /*Increment sector number*/
          addresstmp = pFlash.Address + FLASH_PAGE_SIZE;
          pFlash.Address = addresstmp;

          /* If the erase operation is completed, disable the PER Bit */
          CLEAR_BIT(FLASH->CR, FLASH_CR_PER);

          FLASH_PageErase(addresstmp);
        }
        else
        {
          /* No more pages to Erase, user callback can be called. */
          /* Reset Sector and stop Erase pages procedure */
          pFlash.Address = addresstmp = 0xFFFFFFFFU;
          pFlash.ProcedureOnGoing = FLASH_PROC_NONE;
          /* FLASH EOP interrupt user callback */
          HAL_FLASH_EndOfOperationCallback(addresstmp);
        }
      }
      else if(pFlash.ProcedureOnGoing == FLASH_PROC_MASSERASE)
      {
        /* Operation is completed, disable the MER Bit */
        CLEAR_BIT(FLASH->CR, FLASH_CR_MER);

#if defined(FLASH_BANK2_END)
        /* Stop Mass Erase procedure if no pending mass erase on other bank */
        if (HAL_IS_BIT_CLR(FLASH->CR2, FLASH_CR2_MER))
        {
#endif /* FLASH_BANK2_END */
          /* MassErase ended. Return the selected bank */
          /* FLASH EOP interrupt user callback */
          HAL_FLASH_EndOfOperationCallback(0U);

          /* Stop Mass Erase procedure*/
          pFlash.ProcedureOnGoing = FLASH_PROC_NONE;
        }
#if defined(FLASH_BANK2_END)
      }
#endif /* FLASH_BANK2_END */
      else
      {
        /* Nb of 16-bit data to program can be decreased */
        pFlash.DataRemaining--;
        
        /* Check if there are still 16-bit data to program */
        if(pFlash.DataRemaining != 0U)
        {
          /* Increment address to 16-bit */
          pFlash.Address += 2U;
          addresstmp = pFlash.Address;
          
          /* Shift to have next 16-bit data */
          pFlash.Data = (pFlash.Data >> 16U);
          
          /* Operation is completed, disable the PG Bit */
          CLEAR_BIT(FLASH->CR, FLASH_CR_PG);

          /*Program halfword (16-bit) at a specified address.*/
          FLASH_Program_HalfWord(addresstmp, (uint16_t)pFlash.Data);
        }
        else
        {
          /* Program ended. Return the selected address */
          /* FLASH EOP interrupt user callback */
          if (pFlash.ProcedureOnGoing == FLASH_PROC_PROGRAMHALFWORD)
          {
            HAL_FLASH_EndOfOperationCallback(pFlash.Address);
          }
          else if (pFlash.ProcedureOnGoing == FLASH_PROC_PROGRAMWORD)
          {
            HAL_FLASH_EndOfOperationCallback(pFlash.Address - 2U);
          }
          else 
          {
            HAL_FLASH_EndOfOperationCallback(pFlash.Address - 6U);
          }
        
          /* Reset Address and stop Program procedure */
          pFlash.Address = 0xFFFFFFFFU;
          pFlash.ProcedureOnGoing = FLASH_PROC_NONE;
        }
      }
    }
  }
  
#if defined(FLASH_BANK2_END)
  /* Check FLASH End of Operation flag  */
  if(__HAL_FLASH_GET_FLAG( FLASH_FLAG_EOP_BANK2))
  {
    /* Clear FLASH End of Operation pending bit */
    __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP_BANK2);
    
    /* Process can continue only if no error detected */
    if(pFlash.ProcedureOnGoing != FLASH_PROC_NONE)
    {
      if(pFlash.ProcedureOnGoing == FLASH_PROC_PAGEERASE)
      {
        /* Nb of pages to erased can be decreased */
        pFlash.DataRemaining--;
        
        /* Check if there are still pages to erase*/
        if(pFlash.DataRemaining != 0U)
        {
          /* Indicate user which page address has been erased*/
          HAL_FLASH_EndOfOperationCallback(pFlash.Address);
        
          /* Increment page address to next page */
          pFlash.Address += FLASH_PAGE_SIZE;
          addresstmp = pFlash.Address;

          /* Operation is completed, disable the PER Bit */
          CLEAR_BIT(FLASH->CR2, FLASH_CR2_PER);

          FLASH_PageErase(addresstmp);
        }
        else
        {
          /*No more pages to Erase*/
          
          /*Reset Address and stop Erase pages procedure*/
          pFlash.Address = 0xFFFFFFFFU;
          pFlash.ProcedureOnGoing = FLASH_PROC_NONE;

          /* FLASH EOP interrupt user callback */
          HAL_FLASH_EndOfOperationCallback(pFlash.Address);
        }
      }
      else if(pFlash.ProcedureOnGoing == FLASH_PROC_MASSERASE)
      {
        /* Operation is completed, disable the MER Bit */
        CLEAR_BIT(FLASH->CR2, FLASH_CR2_MER);

        if (HAL_IS_BIT_CLR(FLASH->CR, FLASH_CR_MER))
        {
          /* MassErase ended. Return the selected bank*/
          /* FLASH EOP interrupt user callback */
          HAL_FLASH_EndOfOperationCallback(0U);
        
          pFlash.ProcedureOnGoing = FLASH_PROC_NONE;
        }
      }
      else
      {
        /* Nb of 16-bit data to program can be decreased */
        pFlash.DataRemaining--;
        
        /* Check if there are still 16-bit data to program */
        if(pFlash.DataRemaining != 0U)
        {
          /* Increment address to 16-bit */
          pFlash.Address += 2U;
          addresstmp = pFlash.Address;
          
          /* Shift to have next 16-bit data */
          pFlash.Data = (pFlash.Data >> 16U);
          
          /* Operation is completed, disable the PG Bit */
          CLEAR_BIT(FLASH->CR2, FLASH_CR2_PG);

          /*Program halfword (16-bit) at a specified address.*/
          FLASH_Program_HalfWord(addresstmp, (uint16_t)pFlash.Data);
        }
        else
        {
          /*Program ended. Return the selected address*/
          /* FLASH EOP interrupt user callback */
          if (pFlash.ProcedureOnGoing == FLASH_PROC_PROGRAMHALFWORD)
          {
            HAL_FLASH_EndOfOperationCallback(pFlash.Address);
          }
          else if (pFlash.ProcedureOnGoing == FLASH_PROC_PROGRAMWORD)
          {
            HAL_FLASH_EndOfOperationCallback(pFlash.Address-2U);
          }
          else 
          {
            HAL_FLASH_EndOfOperationCallback(pFlash.Address-6U);
          }
          
          /* Reset Address and stop Program procedure*/
          pFlash.Address = 0xFFFFFFFFU;
          pFlash.ProcedureOnGoing = FLASH_PROC_NONE;
        }
      }
    }
  }
#endif 

  if(pFlash.ProcedureOnGoing == FLASH_PROC_NONE)
  {
#if defined(FLASH_BANK2_END)
    /* Operation is completed, disable the PG, PER and MER Bits for both bank */
    CLEAR_BIT(FLASH->CR, (FLASH_CR_PG | FLASH_CR_PER | FLASH_CR_MER));
    CLEAR_BIT(FLASH->CR2, (FLASH_CR2_PG | FLASH_CR2_PER | FLASH_CR2_MER));  
  
    /* Disable End of FLASH Operation and Error source interrupts for both banks */
    __HAL_FLASH_DISABLE_IT(FLASH_IT_EOP_BANK1 | FLASH_IT_ERR_BANK1 | FLASH_IT_EOP_BANK2 | FLASH_IT_ERR_BANK2);
#else
    /* Operation is completed, disable the PG, PER and MER Bits */
    CLEAR_BIT(FLASH->CR, (FLASH_CR_PG | FLASH_CR_PER | FLASH_CR_MER));

    /* Disable End of FLASH Operation and Error source interrupts */
    __HAL_FLASH_DISABLE_IT(FLASH_IT_EOP | FLASH_IT_ERR);
#endif /* FLASH_BANK2_END */

    /* Process Unlocked */
    __HAL_UNLOCK(&pFlash);
  }
}
```

**描述:** 这是 FLASH 中断处理函数。  当发生 FLASH 事件（完成或出错）时，会调用此函数。 该函数检查错误标志，并在出错时调用错误回调。 如果操作完成，它会清除完成标志，并根据当前过程（编程、页面擦除或 mass erase）继续执行或完成操作。

**中文解释:**

*   `void HAL_FLASH_IRQHandler(void)`:  这是FLASH中断服务例程，当FLASH操作完成或出错时，CPU会跳转到这个函数执行。
*   `__HAL_FLASH_GET_FLAG(FLASH_FLAG_WRPERR) || __HAL_FLASH_GET_FLAG(FLASH_FLAG_PGERR)`:  检查是否发生了写保护错误或编程错误。
*   `FLASH_SetErrorCode()`:  如果发生错误，调用此函数设置错误代码。
*   `HAL_FLASH_OperationErrorCallback(addresstmp)`: 调用用户提供的错误回调函数，通知用户发生了错误。
*   `__HAL_FLASH_GET_FLAG(FLASH_FLAG_EOP)`: 检查FLASH操作是否完成。
*   `pFlash.ProcedureOnGoing == FLASH_PROC_PAGEERASE ... FLASH_PROC_MASSERASE ...`: 根据当前操作类型进行不同的处理。
*   `HAL_FLASH_EndOfOperationCallback(addresstmp)`: 调用用户提供的完成回调函数，通知用户操作完成。
*   `CLEAR_BIT(FLASH->CR, (FLASH_CR_PG | FLASH_CR_PER | FLASH_CR_MER))`: 清除FLASH控制寄存器中的编程位、页面擦除位和全擦除位，表示操作完成。
*   `__HAL_FLASH_DISABLE_IT(FLASH_IT_EOP | FLASH_IT_ERR)`: 禁用FLASH中断。

**用途:**  这是非常关键的一个函数，它处理FLASH操作的中断，并调用用户的回调函数，通知用户操作的结果。

**9. HAL_FLASH_EndOfOperationCallback 和 HAL_FLASH_OperationErrorCallback (回调函数)**

```c
__weak void HAL_FLASH_EndOfOperationCallback(uint32_t ReturnValue)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(ReturnValue);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_FLASH_EndOfOperationCallback could be implemented in the user file
   */ 
}

__weak void HAL_FLASH_OperationErrorCallback(uint32_t ReturnValue)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(ReturnValue);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_FLASH_OperationErrorCallback could be implemented in the user file
   */ 
}
```

**描述:** 这是两个弱回调函数。  `HAL_FLASH_EndOfOperationCallback` 在 FLASH 操作成功完成时调用，`HAL_FLASH_OperationErrorCallback` 在发生错误时调用。 用户可以重新定义这些函数来执行自定义操作。

**中文解释:**

*   `__weak`: 表示这是一个弱函数，用户可以在自己的代码中重新定义它，覆盖HAL库提供的默认实现。
*   `HAL_FLASH_EndOfOperationCallback(uint32_t ReturnValue)`:  FLASH操作完成时，会调用这个函数。 `ReturnValue` 包含操作完成时的状态信息。
*   `HAL_FLASH_OperationErrorCallback(uint32_t ReturnValue)`: FLASH操作出错时，会调用这个函数。 `ReturnValue` 包含错误发生时的状态信息。

**用途:**  这两个回调函数允许用户自定义FLASH操作完成或出错时的处理逻辑。

**10. 其他控制函数 (Peripheral Control Functions)**

```c
HAL_StatusTypeDef HAL_FLASH_Unlock(void);
HAL_StatusTypeDef HAL_FLASH_Lock(void);
HAL_StatusTypeDef HAL_FLASH_OB_Unlock(void);
HAL_StatusTypeDef HAL_FLASH_OB_Lock(void);
void HAL_FLASH_OB_Launch(void);
```

**描述:** 这些函数用于控制 FLASH 外设，包括解锁/锁定 FLASH 控制寄存器、解锁/锁定 Option Bytes 控制寄存器，以及启动 Option Bytes 加载。

**中文解释:**

*   `HAL_FLASH_Unlock()`: 解锁FLASH控制寄存器，允许写入操作。
*   `HAL_FLASH_Lock()`: 锁定FLASH控制寄存器，防止误写入操作。
*   `HAL_FLASH_OB_Unlock()`: 解锁Option Bytes控制寄存器，允许修改Option Bytes。
*   `HAL_FLASH_OB_Lock()`: 锁定Option Bytes控制寄存器，防止误修改Option Bytes。
*   `HAL_FLASH_OB_Launch()`: 启动Option Bytes加载，通常用于更新Option Bytes的配置。

**用途:** 这些函数是FLASH操作的基础，必须先解锁FLASH，才能进行编程、擦除等操作。

**11. 错误处理函数 (Peripheral Errors Functions)**

```c
uint32_t HAL_FLASH_GetError(void);
```

**描述:**  此函数返回特定的 FLASH 错误标志，允许用户在运行时获取 FLASH 外设的错误。

**中文解释:**

*   `HAL_FLASH_GetError()`:  获取FLASH错误代码，指示发生的错误类型。

**用途:**  应用程序可以使用此函数来检查FLASH操作是否出错，并采取相应的措施。

**12. FLASH_Program_HalfWord 函数 (私有函数)**

```c
static void FLASH_Program_HalfWord(uint32_t Address, uint16_t Data)
{
  /* Clean the error context */
  pFlash.ErrorCode = HAL_FLASH_ERROR_NONE;
  
#if defined(FLASH_BANK2_END)
  if(Address <= FLASH_BANK1_END)
  {
#endif /* FLASH_BANK2_END */
    /* Proceed to program the new data */
    SET_BIT(FLASH->CR, FLASH_CR_PG);
#if defined(FLASH_BANK2_END)
  }
  else
  {
    /* Proceed to program the new data */
    SET_BIT(FLASH->CR2, FLASH_CR2_PG);
  }
#endif /* FLASH_BANK2_END */

  /* Write data in the address */
  *(__IO uint16_t*)Address = Data;
}
```

**描述