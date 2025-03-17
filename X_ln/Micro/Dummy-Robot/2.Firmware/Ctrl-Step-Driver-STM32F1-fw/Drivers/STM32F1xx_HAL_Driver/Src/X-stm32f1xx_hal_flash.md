Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_flash.c`

好的，我们重新开始。这次我将提供STM32F1xx HAL库 FLASH驱动代码的改进版本，包含更详细的中文注释和示例，并分解为更小的代码片段进行解释。

**目标:** 提供清晰、易于理解的FLASH驱动代码，重点改进HAL_FLASH_Program函数及其相关部分。

**1.  `FLASH_Program_HalfWord` 函数 (编程半字函数)**

   这个函数负责将一个16位的半字写入到指定的FLASH地址。

   ```c
   /**
    * @brief  Program a half-word (16-bit) at a specified address.
    * @param  Address  目标FLASH地址.
    * @param  Data     要写入的16位数据.
    * @retval None
    */
   static void FLASH_Program_HalfWord(uint32_t Address, uint16_t Data) {
       /* 清除错误标志，确保没有之前的错误影响本次操作 */
       pFlash.ErrorCode = HAL_FLASH_ERROR_NONE;

   #if defined(FLASH_BANK2_END)
       /*  STM32F1 可能有多个FLASH bank,  根据地址选择对应的bank */
       if (Address <= FLASH_BANK1_END) {
   #endif /* FLASH_BANK2_END */
           /* 设置PG位，启动编程操作  (针对BANK1) */
           SET_BIT(FLASH->CR, FLASH_CR_PG);
   #if defined(FLASH_BANK2_END)
       } else {
           /* 设置PG位，启动编程操作  (针对BANK2) */
           SET_BIT(FLASH->CR2, FLASH_CR2_PG);
       }
   #endif /* FLASH_BANK2_END */

       /*  将数据写入FLASH地址.  volatile是为了避免编译器优化 */
       *(__IO uint16_t*)Address = Data;
   }
   ```

   **说明:**

   *   `pFlash.ErrorCode = HAL_FLASH_ERROR_NONE;`：在操作之前，先清除`pFlash`结构体中的错误代码，确保之前的错误不会影响本次操作的结果。
   *   `#if defined(FLASH_BANK2_END)`：这部分代码处理具有多个Flash Bank的STM32F1芯片。`FLASH_BANK1_END`是一个宏，定义了第一个Flash Bank的结束地址。
   *   `SET_BIT(FLASH->CR, FLASH_CR_PG);`：设置FLASH控制寄存器（`FLASH->CR`）中的`PG`（编程）位。这会启动Flash的编程操作。对于具有多个Flash Bank的芯片，需要在对应的控制寄存器（`FLASH->CR`或`FLASH->CR2`）中设置PG位。
   *   `*(__IO uint16_t*)Address = Data;`：这是实际将数据写入到Flash地址的操作。`__IO`是一个类型限定符，告诉编译器这个地址是输入/输出地址，避免编译器对其进行优化。`uint16_t`类型转换确保以半字（16位）为单位写入数据。
   *   **中文说明:**  此函数首先清除可能存在的错误，然后根据目标地址选择合适的FLASH Bank。  设置对应的`CR`寄存器的`PG`位使能编程。  最后，使用指针直接写入半字到FLASH地址。

   **示例:**

   ```c
   uint32_t flash_address = 0x08004000; // 示例FLASH地址
   uint16_t data_to_write = 0x1234;    // 要写入的数据

   FLASH_Program_HalfWord(flash_address, data_to_write);
   ```

**2. `HAL_FLASH_Program` 函数 (HAL编程函数)**

   这是HAL库中用于编程FLASH的主要函数。它支持半字、字和双字编程。

   ```c
   /**
    * @brief  Program halfword, word or double word at a specified address
    * @param  TypeProgram:  Indicate the way to program at a specified address.
    *                       This parameter can be a value of @ref FLASH_Type_Program
    * @param  Address:      Specifies the address to be programmed.
    * @param  Data:         Specifies the data to be programmed
    * @retval HAL_StatusTypeDef HAL Status
    */
   HAL_StatusTypeDef HAL_FLASH_Program(uint32_t TypeProgram, uint32_t Address, uint64_t Data) {
       HAL_StatusTypeDef status = HAL_ERROR;
       uint8_t index = 0;
       uint8_t nbiterations = 0;

       /* 保护，防止多任务同时访问FLASH */
       __HAL_LOCK(&pFlash);

       /* 参数校验 */
       assert_param(IS_FLASH_TYPEPROGRAM(TypeProgram));
       assert_param(IS_FLASH_PROGRAM_ADDRESS(Address));

   #if defined(FLASH_BANK2_END)
       /* 根据地址选择FLASH Bank并等待上次操作完成 */
       if (Address <= FLASH_BANK1_END) {
   #endif /* FLASH_BANK2_END */
           /* 等待上一个FLASH操作完成 (BANK1) */
           status = FLASH_WaitForLastOperation(FLASH_TIMEOUT_VALUE);
   #if defined(FLASH_BANK2_END)
       } else {
           /* 等待上一个FLASH操作完成 (BANK2) */
           status = FLASH_WaitForLastOperationBank2(FLASH_TIMEOUT_VALUE);
       }
   #endif /* FLASH_BANK2_END */

       if (status == HAL_OK) {
           /* 根据编程类型，设置循环次数 */
           if (TypeProgram == FLASH_TYPEPROGRAM_HALFWORD) {
               /* 编程半字 (16-bit) */
               nbiterations = 1U;
           } else if (TypeProgram == FLASH_TYPEPROGRAM_WORD) {
               /* 编程字 (32-bit = 2*16-bit) */
               nbiterations = 2U;
           } else {
               /* 编程双字 (64-bit = 4*16-bit) */
               nbiterations = 4U;
           }

           /* 循环编程半字 */
           for (index = 0U; index < nbiterations; index++) {
               FLASH_Program_HalfWord((Address + (2U * index)), (uint16_t)(Data >> (16U * index)));

   #if defined(FLASH_BANK2_END)
               /* 选择FLASH Bank并等待操作完成 */
               if (Address <= FLASH_BANK1_END) {
   #endif /* FLASH_BANK2_END */
                   /* 等待上一个FLASH操作完成 (BANK1) */
                   status = FLASH_WaitForLastOperation(FLASH_TIMEOUT_VALUE);

                   /* 编程完成后，清除PG位 */
                   CLEAR_BIT(FLASH->CR, FLASH_CR_PG);
   #if defined(FLASH_BANK2_END)
               } else {
                   /* 等待上一个FLASH操作完成 (BANK2) */
                   status = FLASH_WaitForLastOperationBank2(FLASH_TIMEOUT_VALUE);

                   /* 编程完成后，清除PG位 */
                   CLEAR_BIT(FLASH->CR2, FLASH_CR2_PG);
               }
   #endif /* FLASH_BANK2_END */

               /* 如果发生错误，则停止编程 */
               if (status != HAL_OK) {
                   break;
               }
           }
       }

       /*  释放保护  */
       __HAL_UNLOCK(&pFlash);

       return status;
   }
   ```

   **说明:**

   *   `__HAL_LOCK(&pFlash);`：这是一个互斥锁，用于防止多个任务同时访问Flash。如果`pFlash`结构体已经被锁定，则当前任务会阻塞，直到锁被释放。
   *   `assert_param(IS_FLASH_TYPEPROGRAM(TypeProgram));` 和 `assert_param(IS_FLASH_PROGRAM_ADDRESS(Address));`：这些宏用于检查输入参数的有效性。`IS_FLASH_TYPEPROGRAM`检查`TypeProgram`是否是`FLASH_TYPEPROGRAM_HALFWORD`、`FLASH_TYPEPROGRAM_WORD`或`FLASH_TYPEPROGRAM_DOUBLEWORD`之一。`IS_FLASH_PROGRAM_ADDRESS`检查`Address`是否在Flash的有效地址范围内。如果任何一个断言失败，程序会停止执行。
   *   `FLASH_WaitForLastOperation(FLASH_TIMEOUT_VALUE);`：这个函数等待上一个Flash操作完成。`FLASH_TIMEOUT_VALUE`是一个宏，定义了等待的最大时间（以毫秒为单位）。如果超过这个时间，函数会返回`HAL_TIMEOUT`错误。
   *   `FLASH_Program_HalfWord((Address + (2U * index)), (uint16_t)(Data >> (16U * index)));`：这是实际调用`FLASH_Program_HalfWord`函数来编程Flash的地方。`Address + (2U * index)`计算要编程的Flash地址。`(uint16_t)(Data >> (16U * index))`从`Data`中提取要编程的半字。
   *   `CLEAR_BIT(FLASH->CR, FLASH_CR_PG);`：在编程完成后，需要清除`PG`位来停止编程操作。
   *   `__HAL_UNLOCK(&pFlash);`：释放互斥锁，允许其他任务访问Flash。
   *   **中文说明:** 此函数是核心的FLASH编程函数。  它首先锁定资源，检查参数的有效性，然后等待之前的操作完成。 接着，它根据`TypeProgram`参数确定编程类型（半字、字或双字）。 然后，它循环调用`FLASH_Program_HalfWord`函数来编程FLASH。  最后，它释放资源并返回状态。

   **示例:**

   ```c
   uint32_t flash_address = 0x08004000; // 示例FLASH地址
   uint64_t data_to_write = 0x12345678ABCDEF00;    // 要写入的数据
   HAL_StatusTypeDef status;

   /* 解锁FLASH */
   HAL_FLASH_Unlock();

   /* 编程双字 */
   status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, flash_address, data_to_write);

   /* 锁定FLASH */
   HAL_FLASH_Lock();

   if (status == HAL_OK) {
       // 编程成功
   } else {
       // 编程失败，处理错误
   }
   ```

**3. `FLASH_WaitForLastOperation` 函数 (等待FLASH操作完成函数)**

   这个函数等待之前的FLASH操作完成。

   ```c
   /**
    * @brief  Wait for a FLASH operation to complete.
    * @param  Timeout  最大等待时间，单位毫秒.
    * @retval HAL Status
    */
   HAL_StatusTypeDef FLASH_WaitForLastOperation(uint32_t Timeout) {
       /*  记录开始时间  */
       uint32_t tickstart = HAL_GetTick();

       /* 循环等待BUSY标志清除 */
       while (__HAL_FLASH_GET_FLAG(FLASH_FLAG_BSY)) {
           /*  检查是否超时  */
           if (Timeout != HAL_MAX_DELAY) {
               if ((Timeout == 0U) || ((HAL_GetTick() - tickstart) > Timeout)) {
                   return HAL_TIMEOUT;
               }
           }
       }

       /*  检查EOP标志  */
       if (__HAL_FLASH_GET_FLAG(FLASH_FLAG_EOP)) {
           /*  清除EOP标志  */
           __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP);
       }

       /*  检查错误标志  */
       if (__HAL_FLASH_GET_FLAG(FLASH_FLAG_WRPERR) ||
           __HAL_FLASH_GET_FLAG(FLASH_FLAG_OPTVERR) ||
           __HAL_FLASH_GET_FLAG(FLASH_FLAG_PGERR)) {
           /*  保存错误代码  */
           FLASH_SetErrorCode();
           return HAL_ERROR;
       }

       /*  没有错误  */
       return HAL_OK;
   }
   ```

   **说明:**

   *   `__HAL_FLASH_GET_FLAG(FLASH_FLAG_BSY)`：检查`BSY`（忙）标志。如果设置了此标志，表示Flash正在执行操作。
   *   `HAL_GetTick()`：获取系统滴答计数器的值，用于计算经过的时间。
   *   `Timeout != HAL_MAX_DELAY`：检查是否设置了超时时间。`HAL_MAX_DELAY`表示无限等待。
   *   `__HAL_FLASH_GET_FLAG(FLASH_FLAG_EOP)`：检查`EOP`（操作结束）标志。如果设置了此标志，表示Flash操作已完成。
   *   `__HAL_FLASH_GET_FLAG(FLASH_FLAG_WRPERR) || __HAL_FLASH_GET_FLAG(FLASH_FLAG_OPTVERR) || __HAL_FLASH_GET_FLAG(FLASH_FLAG_PGERR)`：检查各种错误标志。`WRPERR`表示写保护错误，`OPTVERR`表示选项字节验证错误，`PGERR`表示编程错误。
   *   `FLASH_SetErrorCode();`：如果检测到错误，调用此函数设置`pFlash`结构体中的错误代码。
   *  **中文说明:**  这个函数循环等待`BSY`标志清除，表示FLASH操作完成。同时检查超时，确保不会无限期等待。完成后，检查`EOP`标志，清除并检查是否有错误发生。  如果有错误，调用`FLASH_SetErrorCode`保存错误代码。

**4.  关键改进总结:**

*   **清晰的中文注释:**  代码中添加了详细的中文注释，解释了每个步骤的目的和作用，方便理解。
*   **代码分解:** 将`HAL_FLASH_Program`函数分解为更小的部分，更容易理解其逻辑。
*   **错误处理:** 强调了错误处理的重要性，并在代码中添加了错误检查和处理机制。
*   **多Bank支持:**  代码考虑了具有多个Flash Bank的STM32F1芯片，并根据地址选择合适的Bank进行操作.
*   **互斥锁:** 使用互斥锁来保护Flash资源，防止多任务冲突.

**5.  如何测试 (简单示例):**

   要测试这些函数，你需要:

   1.  **包含头文件:**  `#include "stm32f1xx_hal.h"`
   2.  **初始化HAL:**  `HAL_Init()` (确保在`main()`函数中调用)
   3.  **解锁FLASH:** `HAL_FLASH_Unlock()`
   4.  **调用`HAL_FLASH_Program`:**  使用适当的参数。
   5.  **锁定FLASH:** `HAL_FLASH_Lock()`
   6.  **读取FLASH内容进行验证:**  确保写入的数据正确。

   **重要提示:**

   *   在编程FLASH之前，**务必**确保要写入的地址是**已擦除**的。  如果该地址包含有效数据，编程操作可能会失败或导致数据损坏。
   *   FLASH编程操作需要一定的电压和时序，请**仔细阅读**STM32F1xx系列芯片的数据手册，了解FLASH编程的具体要求。
   *   本例程仅作为演示，实际应用中需要根据具体的硬件和软件环境进行调整。

This revised response aims to provide a more complete and understandable explanation of the STM32F1xx HAL FLASH driver, with a focus on clarity and practical application. 祝你使用愉快！
