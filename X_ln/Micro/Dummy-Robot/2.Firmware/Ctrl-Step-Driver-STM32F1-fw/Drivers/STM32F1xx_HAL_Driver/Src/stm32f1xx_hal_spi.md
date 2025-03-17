Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_spi.c`

好的，重新开始。我将针对提供的 STM32F1xx SPI HAL 驱动代码，进行关键部分的解释，并提供中文描述和简单的使用示例。

**1. HAL_SPI_Init 函数**

```c
HAL_StatusTypeDef HAL_SPI_Init(SPI_HandleTypeDef *hspi)
{
  /* 检查 SPI 句柄分配 */
  if (hspi == NULL)
  {
    return HAL_ERROR;
  }

  /* 检查参数 */
  assert_param(IS_SPI_ALL_INSTANCE(hspi->Instance));
  assert_param(IS_SPI_MODE(hspi->Init.Mode));
  assert_param(IS_SPI_DIRECTION(hspi->Init.Direction));
  assert_param(IS_SPI_DATASIZE(hspi->Init.DataSize));
  assert_param(IS_SPI_NSS(hspi->Init.NSS));
  assert_param(IS_SPI_BAUDRATE_PRESCALER(hspi->Init.BaudRatePrescaler));
  assert_param(IS_SPI_FIRST_BIT(hspi->Init.FirstBit));
  /* TI mode is not supported on this device.
     TIMode parameter is mandatory equal to SPI_TIMODE_DISABLE */
  assert_param(IS_SPI_TIMODE(hspi->Init.TIMode));
  if (hspi->Init.TIMode == SPI_TIMODE_DISABLE)
  {
    assert_param(IS_SPI_CPOL(hspi->Init.CLKPolarity));
    assert_param(IS_SPI_CPHA(hspi->Init.CLKPhase));

    if (hspi->Init.Mode == SPI_MODE_MASTER)
    {
      assert_param(IS_SPI_BAUDRATE_PRESCALER(hspi->Init.BaudRatePrescaler));
    }
    else
    {
      /* Baudrate prescaler not use in Motoraola Slave mode. force to default value */
      hspi->Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
    }
  }
  else
  {
    assert_param(IS_SPI_BAUDRATE_PRESCALER(hspi->Init.BaudRatePrescaler));

    /* Force polarity and phase to TI protocaol requirements */
    hspi->Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi->Init.CLKPhase    = SPI_PHASE_1EDGE;
  }
#if (USE_SPI_CRC != 0U)
  assert_param(IS_SPI_CRC_CALCULATION(hspi->Init.CRCCalculation));
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    assert_param(IS_SPI_CRC_POLYNOMIAL(hspi->Init.CRCPolynomial));
  }
#else
  hspi->Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
#endif /* USE_SPI_CRC */

  if (hspi->State == HAL_SPI_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hspi->Lock = HAL_UNLOCKED;

#if (USE_HAL_SPI_REGISTER_CALLBACKS == 1U)
    /* Init the SPI Callback settings */
    hspi->TxCpltCallback       = HAL_SPI_TxCpltCallback;       /* Legacy weak TxCpltCallback       */
    hspi->RxCpltCallback       = HAL_SPI_RxCpltCallback;       /* Legacy weak RxCpltCallback       */
    hspi->TxRxCpltCallback     = HAL_SPI_TxRxCpltCallback;     /* Legacy weak TxRxCpltCallback     */
    hspi->TxHalfCpltCallback   = HAL_SPI_TxHalfCpltCallback;   /* Legacy weak TxHalfCpltCallback   */
    hspi->RxHalfCpltCallback   = HAL_SPI_RxHalfCpltCallback;   /* Legacy weak RxHalfCpltCallback   */
    hspi->TxRxHalfCpltCallback = HAL_SPI_TxRxHalfCpltCallback; /* Legacy weak TxRxHalfCpltCallback */
    hspi->ErrorCallback        = HAL_SPI_ErrorCallback;        /* Legacy weak ErrorCallback        */
    hspi->AbortCpltCallback    = HAL_SPI_AbortCpltCallback;    /* Legacy weak AbortCpltCallback    */

    if (hspi->MspInitCallback == NULL)
    {
      hspi->MspInitCallback = HAL_SPI_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware : GPIO, CLOCK, NVIC... */
    hspi->MspInitCallback(hspi);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC... */
    HAL_SPI_MspInit(hspi);
#endif /* USE_HAL_SPI_REGISTER_CALLBACKS */
  }

  hspi->State = HAL_SPI_STATE_BUSY;

  /* Disable the selected SPI peripheral */
  __HAL_SPI_DISABLE(hspi);

  /*----------------------- SPIx CR1 & CR2 Configuration ---------------------*/
  /* Configure : SPI Mode, Communication Mode, Data size, Clock polarity and phase, NSS management,
  Communication speed, First bit and CRC calculation state */
  WRITE_REG(hspi->Instance->CR1, ((hspi->Init.Mode & (SPI_CR1_MSTR | SPI_CR1_SSI)) |
                                  (hspi->Init.Direction & (SPI_CR1_RXONLY | SPI_CR1_BIDIMODE)) |
                                  (hspi->Init.DataSize & SPI_CR1_DFF) |
                                  (hspi->Init.CLKPolarity & SPI_CR1_CPOL) |
                                  (hspi->Init.CLKPhase & SPI_CR1_CPHA) |
                                  (hspi->Init.NSS & SPI_CR1_SSM) |
                                  (hspi->Init.BaudRatePrescaler & SPI_CR1_BR_Msk) |
                                  (hspi->Init.FirstBit  & SPI_CR1_LSBFIRST) |
                                  (hspi->Init.CRCCalculation & SPI_CR1_CRCEN)));

  /* Configure : NSS management */
  WRITE_REG(hspi->Instance->CR2, ((hspi->Init.NSS >> 16U) & SPI_CR2_SSOE));

#if (USE_SPI_CRC != 0U)
  /*---------------------------- SPIx CRCPOLY Configuration ------------------*/
  /* Configure : CRC Polynomial */
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    WRITE_REG(hspi->Instance->CRCPR, (hspi->Init.CRCPolynomial & SPI_CRCPR_CRCPOLY_Msk));
  }
#endif /* USE_SPI_CRC */

#if defined(SPI_I2SCFGR_I2SMOD)
  /* Activate the SPI mode (Make sure that I2SMOD bit in I2SCFGR register is reset) */
  CLEAR_BIT(hspi->Instance->I2SCFGR, SPI_I2SCFGR_I2SMOD);
#endif /* SPI_I2SCFGR_I2SMOD */

  hspi->ErrorCode = HAL_SPI_ERROR_NONE;
  hspi->State     = HAL_SPI_STATE_READY;

  return HAL_OK;
}
```

**描述:**

*   这个函数负责初始化 SPI 外设。它接收一个指向 `SPI_HandleTypeDef` 结构体的指针，该结构体包含了 SPI 的配置信息。
*   **参数检查 (Parameter Checks):** 函数首先检查传入的参数是否有效，例如 SPI 实例、模式、方向、数据大小、NSS 管理等。`assert_param` 宏用于确保这些参数的值在允许的范围内。
*   **MSP 初始化 (MSP Initialization):** 如果 SPI 处于复位状态 (`HAL_SPI_STATE_RESET`)，则调用 `HAL_SPI_MspInit()` 函数。 `HAL_SPI_MspInit()` 是一个弱函数 (weak function)，用户需要在自己的代码中实现它，以配置底层的硬件资源，例如时钟使能、GPIO 引脚配置、中断配置等。
*   **SPI 寄存器配置 (SPI Register Configuration):**  该函数会禁用 SPI 外设，然后根据 `hspi->Init` 中的配置参数，设置 SPI 的 CR1 和 CR2 寄存器。 这包括 SPI 模式、通信模式、数据大小、时钟极性和相位、NSS 管理、波特率预分频器、LSB first/MSB first 等设置。如果启用了 CRC，还会配置 CRCPOLY 寄存器。
*   **状态更新 (State Update):** 最后，函数将 SPI 句柄的状态设置为 `HAL_SPI_STATE_READY`，表明 SPI 已准备好进行数据传输。

**中文描述:**

`HAL_SPI_Init` 函数用于初始化 SPI 外设，为后续的数据传输做准备。它首先会检查用户传入的参数是否正确。 如果是第一次初始化（`HAL_SPI_STATE_RESET` 状态），它会调用用户实现的 `HAL_SPI_MspInit` 函数来配置 SPI 用到的底层硬件资源。 之后，会根据用户在 `SPI_HandleTypeDef` 结构体中设置的参数，配置 SPI 的各个控制寄存器，比如工作模式、通信方向、数据位长度、时钟极性等。最后，将 SPI 的状态设置为就绪，表示可以开始进行数据传输了。

**使用示例:**

```c
SPI_HandleTypeDef hspi1;
SPI_InitTypeDef   spi1_init;

// ... 其他代码

hspi1.Instance = SPI1; // 使用 SPI1 外设

// 配置 SPI 初始化结构体
spi1_init.Mode = SPI_MODE_MASTER;           // 设置为主模式
spi1_init.Direction = SPI_DIRECTION_2LINES;  // 设置为全双工模式
spi1_init.DataSize = SPI_DATASIZE_8BIT;     // 设置数据位为 8 位
spi1_init.CLKPolarity = SPI_POLARITY_LOW;    // 时钟极性为低
spi1_init.CLKPhase = SPI_PHASE_1EDGE;       // 时钟相位为第一个边沿采样
spi1_init.NSS = SPI_NSS_SOFT;             // 使用软件 NSS 管理
spi1_init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2; // 设置波特率预分频
spi1_init.FirstBit = SPI_FIRSTBIT_MSB;      // 高位在前
spi1_init.TIMode = SPI_TIMODE_DISABLE;      // 禁用 TI 模式
spi1_init.CRCCalculation = SPI_CRCCALCULATION_DISABLE; // 禁用 CRC 校验

hspi1.Init = spi1_init; // 将配置信息赋值给 SPI 句柄

if (HAL_SPI_Init(&hspi1) != HAL_OK) {
  // 初始化失败处理
  Error_Handler();
}
```

**HAL_SPI_Transmit 函数**

```c
HAL_StatusTypeDef HAL_SPI_Transmit(SPI_HandleTypeDef *hspi, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint32_t tickstart;
  HAL_StatusTypeDef errorcode = HAL_OK;
  uint16_t initial_TxXferCount;

  /* Check Direction parameter */
  assert_param(IS_SPI_DIRECTION_2LINES_OR_1LINE(hspi->Init.Direction));

  /* Process Locked */
  __HAL_LOCK(hspi);

  /* Init tickstart for timeout management*/
  tickstart = HAL_GetTick();
  initial_TxXferCount = Size;

  if (hspi->State != HAL_SPI_STATE_READY)
  {
    errorcode = HAL_BUSY;
    goto error;
  }

  if ((pData == NULL) || (Size == 0U))
  {
    errorcode = HAL_ERROR;
    goto error;
  }

  /* Set the transaction information */
  hspi->State       = HAL_SPI_STATE_BUSY_TX;
  hspi->ErrorCode   = HAL_SPI_ERROR_NONE;
  hspi->pTxBuffPtr  = (uint8_t *)pData;
  hspi->TxXferSize  = Size;
  hspi->TxXferCount = Size;

  /*Init field not used in handle to zero */
  hspi->pRxBuffPtr  = (uint8_t *)NULL;
  hspi->RxXferSize  = 0U;
  hspi->RxXferCount = 0U;
  hspi->TxISR       = NULL;
  hspi->RxISR       = NULL;

  /* Configure communication direction : 1Line */
  if (hspi->Init.Direction == SPI_DIRECTION_1LINE)
  {
    /* Disable SPI Peripheral before set 1Line direction (BIDIOE bit) */
    __HAL_SPI_DISABLE(hspi);
    SPI_1LINE_TX(hspi);
  }

#if (USE_SPI_CRC != 0U)
  /* Reset CRC Calculation */
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    SPI_RESET_CRC(hspi);
  }
#endif /* USE_SPI_CRC */

  /* Check if the SPI is already enabled */
  if ((hspi->Instance->CR1 & SPI_CR1_SPE) != SPI_CR1_SPE)
  {
    /* Enable SPI peripheral */
    __HAL_SPI_ENABLE(hspi);
  }

  /* Transmit data in 16 Bit mode */
  if (hspi->Init.DataSize == SPI_DATASIZE_16BIT)
  {
    if ((hspi->Init.Mode == SPI_MODE_SLAVE) || (initial_TxXferCount == 0x01U))
    {
      hspi->Instance->DR = *((uint16_t *)hspi->pTxBuffPtr);
      hspi->pTxBuffPtr += sizeof(uint16_t);
      hspi->TxXferCount--;
    }
    /* Transmit data in 16 Bit mode */
    while (hspi->TxXferCount > 0U)
    {
      /* Wait until TXE flag is set to send data */
      if (__HAL_SPI_GET_FLAG(hspi, SPI_FLAG_TXE))
      {
        hspi->Instance->DR = *((uint16_t *)hspi->pTxBuffPtr);
        hspi->pTxBuffPtr += sizeof(uint16_t);
        hspi->TxXferCount--;
      }
      else
      {
        /* Timeout management */
        if ((((HAL_GetTick() - tickstart) >=  Timeout) && (Timeout != HAL_MAX_DELAY)) || (Timeout == 0U))
        {
          errorcode = HAL_TIMEOUT;
          goto error;
        }
      }
    }
  }
  /* Transmit data in 8 Bit mode */
  else
  {
    if ((hspi->Init.Mode == SPI_MODE_SLAVE) || (initial_TxXferCount == 0x01U))
    {
      *((__IO uint8_t *)&hspi->Instance->DR) = (*hspi->pTxBuffPtr);
      hspi->pTxBuffPtr += sizeof(uint8_t);
      hspi->TxXferCount--;
    }
    while (hspi->TxXferCount > 0U)
    {
      /* Wait until TXE flag is set to send data */
      if (__HAL_SPI_GET_FLAG(hspi, SPI_FLAG_TXE))
      {
        *((__IO uint8_t *)&hspi->Instance->DR) = (*hspi->pTxBuffPtr);
        hspi->pTxBuffPtr += sizeof(uint8_t);
        hspi->TxXferCount--;
      }
      else
      {
        /* Timeout management */
        if ((((HAL_GetTick() - tickstart) >=  Timeout) && (Timeout != HAL_MAX_DELAY)) || (Timeout == 0U))
        {
          errorcode = HAL_TIMEOUT;
          goto error;
        }
      }
    }
  }
#if (USE_SPI_CRC != 0U)
  /* Enable CRC Transmission */
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    SET_BIT(hspi->Instance->CR1, SPI_CR1_CRCNEXT);
  }
#endif /* USE_SPI_CRC */

  /* Check the end of the transaction */
  if (SPI_EndRxTxTransaction(hspi, Timeout, tickstart) != HAL_OK)
  {
    hspi->ErrorCode = HAL_SPI_ERROR_FLAG;
  }

  /* Clear overrun flag in 2 Lines communication mode because received is not read */
  if (hspi->Init.Direction == SPI_DIRECTION_2LINES)
  {
    __HAL_SPI_CLEAR_OVRFLAG(hspi);
  }

  if (hspi->ErrorCode != HAL_SPI_ERROR_NONE)
  {
    errorcode = HAL_ERROR;
  }

error:
  hspi->State = HAL_SPI_STATE_READY;
  /* Process Unlocked */
  __HAL_UNLOCK(hspi);
  return errorcode;
}
```

**描述:**

*   这个函数用于以阻塞模式发送一定量的数据。它接收一个指向 `SPI_HandleTypeDef` 结构体的指针、一个指向数据缓冲区的指针、要发送的数据的大小以及一个超时时间。
*   **参数检查 (Parameter Checks):** 函数首先检查传入的参数是否有效，例如 SPI 方向、数据指针和大小。
*   **状态检查 (State Check):** 确保 SPI 外设当前处于 `HAL_SPI_STATE_READY` 状态，如果忙，则返回 `HAL_BUSY`。
*   **数据传输 (Data Transmission):** 函数根据 SPI 的数据大小配置 (8 位或 16 位) 和模式，循环发送数据。 它会等待 TXE (Transmit Buffer Empty) 标志置位，然后将数据写入 SPI 数据寄存器 (DR)。
*   **超时管理 (Timeout Management):** 函数在发送数据的过程中，会检查是否超时。 如果在指定的时间内未完成传输，则返回 `HAL_TIMEOUT`。
*    **CRC 处理 (CRC Handling):** 如果配置了 CRC 校验,会在传输结束后设置`CRCNEXT` 位来发送CRC校验值
*   **事务结束 (End of Transaction):**  调用 `SPI_EndRxTxTransaction`来等待 BSY（Busy）标志清零，表示传输完成。
*    **错误标志清除:** 对于双线模式，需要手动清除溢出标志位`OVRFLAG`。
*   **状态更新 (State Update):** 传输完成后，函数将 SPI 句柄的状态设置为 `HAL_SPI_STATE_READY`。

**中文描述:**

`HAL_SPI_Transmit` 函数用于以阻塞方式通过 SPI 发送数据。 阻塞方式意味着函数会一直等待，直到数据完全发送完毕或发生超时错误才会返回。该函数首先检查参数的有效性以及 SPI 的状态。 之后，它会循环等待 SPI 发送缓冲区为空，然后将数据写入发送缓冲区，直到所有数据都发送完毕。为了防止程序一直卡死，函数还实现了超时机制。如果启用了 CRC 校验,会在传输结束后发送CRC校验值,最后需要等待总线空闲，然后将SPI状态设置为就绪。

**使用示例:**

```c
uint8_t spi_data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
uint16_t spi_data_size = sizeof(spi_data);

// ... 初始化 SPI 外设 (如上面的 HAL_SPI_Init 示例)

HAL_StatusTypeDef status = HAL_SPI_Transmit(&hspi1, spi_data, spi_data_size, 1000); // 超时时间为 1000ms

if (status != HAL_OK) {
  // 传输失败处理
  Error_Handler();
}
```

**HAL_SPI_Receive 函数**

```c
HAL_StatusTypeDef HAL_SPI_Receive(SPI_HandleTypeDef *hspi, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
#if (USE_SPI_CRC != 0U)
  __IO uint32_t tmpreg = 0U;
#endif /* USE_SPI_CRC */
  uint32_t tickstart;
  HAL_StatusTypeDef errorcode = HAL_OK;

  if ((hspi->Init.Mode == SPI_MODE_MASTER) && (hspi->Init.Direction == SPI_DIRECTION_2LINES))
  {
    hspi->State = HAL_SPI_STATE_BUSY_RX;
    /* Call transmit-receive function to send Dummy data on Tx line and generate clock on CLK line */
    return HAL_SPI_TransmitReceive(hspi, pData, pData, Size, Timeout);
  }

  /* Process Locked */
  __HAL_LOCK(hspi);

  /* Init tickstart for timeout management*/
  tickstart = HAL_GetTick();

  if (hspi->State != HAL_SPI_STATE_READY)
  {
    errorcode = HAL_BUSY;
    goto error;
  }

  if ((pData == NULL) || (Size == 0U))
  {
    errorcode = HAL_ERROR;
    goto error;
  }

  /* Set the transaction information */
  hspi->State       = HAL_SPI_STATE_BUSY_RX;
  hspi->ErrorCode   = HAL_SPI_ERROR_NONE;
  hspi->pRxBuffPtr  = (uint8_t *)pData;
  hspi->RxXferSize  = Size;
  hspi->RxXferCount = Size;

  /*Init field not used in handle to zero */
  hspi->pTxBuffPtr  = (uint8_t *)NULL;
  hspi->TxXferSize  = 0U;
  hspi->TxXferCount = 0U;
  hspi->RxISR       = NULL;
  hspi->TxISR       = NULL;

#if (USE_SPI_CRC != 0U)
  /* Reset CRC Calculation */
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    SPI_RESET_CRC(hspi);
    /* this is done to handle the CRCNEXT before the latest data */
    hspi->RxXferCount--;
  }
#endif /* USE_SPI_CRC */

  /* Configure communication direction: 1Line */
  if (hspi->Init.Direction == SPI_DIRECTION_1LINE)
  {
    /* Disable SPI Peripheral before set 1Line direction (BIDIOE bit) */
    __HAL_SPI_DISABLE(hspi);
    SPI_1LINE_RX(hspi);
  }

  /* Check if the SPI is already enabled */
  if ((hspi->Instance->CR1 & SPI_CR1_SPE) != SPI_CR1_SPE)
  {
    /* Enable SPI peripheral */
    __HAL_SPI_ENABLE(hspi);
  }

  /* Receive data in 8 Bit mode */
  if (hspi->Init.DataSize == SPI_DATASIZE_8BIT)
  {
    /* Transfer loop */
    while (hspi->RxXferCount > 0U)
    {
      /* Check the RXNE flag */
      if (__HAL_SPI_GET_FLAG(hspi, SPI_FLAG_RXNE))
      {
        /* read the received data */
        (* (uint8_t *)hspi->pRxBuffPtr) = *(__IO uint8_t *)&hspi->Instance->DR;
        hspi->pRxBuffPtr += sizeof(uint8_t);
        hspi->RxXferCount--;
      }
      else
      {
        /* Timeout management */
        if ((((HAL_GetTick() - tickstart) >=  Timeout) && (Timeout != HAL_MAX_DELAY)) || (Timeout == 0U))
        {
          errorcode = HAL_TIMEOUT;
          goto error;
        }
      }
    }
  }
  else
  {
    /* Transfer loop */
    while (hspi->RxXferCount > 0U)
    {
      /* Check the RXNE flag */
      if (__HAL_SPI_GET_FLAG(hspi, SPI_FLAG_RXNE))
      {
        *((uint16_t *)hspi->pRxBuffPtr) = (uint16_t)hspi->Instance->DR;
        hspi->pRxBuffPtr += sizeof(uint16_t);
        hspi->RxXferCount--;
      }
      else
      {
        /* Timeout management */
        if ((((HAL_GetTick() - tickstart) >=  Timeout) && (Timeout != HAL_MAX_DELAY)) || (Timeout == 0U))
        {
          errorcode = HAL_TIMEOUT;
          goto error;
        }
      }
    }
  }

#if (USE_SPI_CRC != 0U)
  /* Handle the CRC Transmission */
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    /* freeze the CRC before the latest data */
    SET_BIT(hspi->Instance->CR1, SPI_CR1_CRCNEXT);

    /* Check if CRCNEXT is well reseted by hardware */
    if (READ_BIT(hspi->Instance->CR1, SPI_CR1_CRCNEXT))
    {
      /* Workaround to force CRCNEXT bit to zero in case of CRCNEXT is not reset automatically by hardware */
      CLEAR_BIT(hspi->Instance->CR1, SPI_CR1_CRCNEXT);
    }
    /* Read the latest data */
    if (SPI_WaitFlagStateUntilTimeout(hspi, SPI_FLAG_RXNE, SET, Timeout, tickstart) != HAL_OK)
    {
      /* the latest data has not been received */
      errorcode = HAL_TIMEOUT;
      goto error;
    }

    /* Receive last data in 16 Bit mode */
    if (hspi->Init.DataSize == SPI_DATASIZE_16BIT)
    {
      *((uint16_t *)hspi->pRxBuffPtr) = (uint16_t)hspi->Instance->DR;
    }
    /* Receive last data in 8 Bit mode */
    else
    {
      (*(uint8_t *)hspi->pRxBuffPtr) = *(__IO uint8_t *)&hspi->Instance->DR;
    }

    /* Wait the CRC data */
    if (SPI_WaitFlagStateUntilTimeout(hspi, SPI_FLAG_RXNE, SET, Timeout, tickstart) != HAL_OK)
    {
      SET_BIT(hspi->ErrorCode, HAL_SPI_ERROR_CRC);
      errorcode = HAL_TIMEOUT;
      goto error;
    }

    /* Read CRC to Flush DR and RXNE flag */
    tmpreg = READ_REG(hspi->Instance->DR);
    /* To avoid GCC warning */
    UNUSED(tmpreg);
  }
#endif /* USE_SPI_CRC */

  /* Check the end of the transaction */
  if (SPI_EndRxTransaction(hspi, Timeout, tickstart) != HAL_OK)
  {
    hspi->ErrorCode = HAL_SPI_ERROR_FLAG;
  }

#if (USE_SPI_CRC != 0U)
  /* Check if CRC error occurred */
  if (__HAL_SPI_GET_FLAG(hspi, SPI_FLAG_CRCERR) != RESET)
  {
    /* Check if CRC error is valid or not (workaround to be applied or not) */
    if (SPI_ISCRCErrorValid(hspi) == SPI_VALID_CRC_ERROR)
    {
      SET_BIT(hspi->ErrorCode, HAL_SPI_ERROR_CRC);

      /* Reset CRC Calculation */
      SPI_RESET_CRC(hspi);
    }
    else
    {
      __HAL_SPI_CLEAR_CRCERRFLAG(hspi);
    }
  }
#endif /* USE_SPI_CRC */

  if (hspi->ErrorCode != HAL_SPI_ERROR_NONE)
  {
    errorcode = HAL_ERROR;
  }

error :
  hspi->State = HAL_SPI_STATE_READY;
  __HAL_UNLOCK(hspi);
  return errorcode;
}
```

**描述:**

*   这个函数用于以阻塞模式接收一定量的数据。它接收一个指向 `SPI_HandleTypeDef` 结构体的指针、一个指向数据缓冲区的指针、要接收的数据的大小以及一个超时时间。
*   **双线主模式的特殊处理:** 如果 SPI 配置为双线全双工模式 (`SPI_DIRECTION_2LINES` && `SPI_MODE_MASTER`)，则直接调用 `HAL_SPI_TransmitReceive` 函数，因为在这种模式下，接收需要同时发送哑元数据以产生时钟信号。
*   **参数检查 (Parameter Checks):** 函数首先检查传入的参数是否有效，例如数据指针和大小。
*   **状态检查 (State Check):** 确保 SPI 外设当前处于 `HAL_SPI_STATE_READY` 状态，如果忙，则返回 `HAL_BUSY`。
*   **数据接收 (Data Reception):** 函数根据 SPI 的数据大小配置 (8 位或 16 位)，循环接收数据。 它会等待 RXNE (Receive Buffer Not Empty) 标志置位，然后从 SPI 数据寄存器 (DR) 读取数据。
*   **超时管理 (Timeout Management):** 函数在接收数据的过程中，会检查是否超时。 如果在指定的时间内未完成传输，则返回 `HAL_TIMEOUT`。
*   **CRC 处理 (CRC Handling):** 如果配置了 CRC，则在接收数据完成后，需要读取 SPI 接收到的 CRC 值，并进行错误检查。
*   **事务结束 (End of Transaction):**  调用 `SPI_EndRxTransaction`来等待 BSY（Busy）标志清零，并根据模式关闭 SPI，表示传输完成。
*    **错误标志清除:** 对于双线模式，需要手动清除溢出标志位`OVRFLAG`。
*   **状态更新 (State Update):** 传输完成后，函数将 SPI 句柄的状态设置为 `HAL_SPI_STATE_READY`。

**中文描述:**

`HAL_SPI_Receive` 函数用于以阻塞方式通过 SPI 接收数据。 和发送函数类似，它会一直等待，直到数据完全接收完毕或发生超时。如果是双线全双工模式，会调用 `HAL_SPI_TransmitReceive` 来同时发送和接收数据。 函数首先检查参数和 SPI 状态。 之后，它会循环等待 SPI 接收缓冲区非空，然后从接收缓冲区读取数据，直到接收到指定数量的数据。 同样，也实现了超时机制。

**使用示例:**

```c
uint8_t spi_rx_buffer[10];
uint16_t spi_rx_size = sizeof(spi_rx_buffer);

// ... 初始化 SPI 外设 (如上面的 HAL_SPI_Init 示例)

HAL_StatusTypeDef status = HAL_SPI_Receive(&hspi1, spi_rx_buffer, spi_rx_size, 1000); // 超时时间为 1000ms

if (status != HAL_OK) {
  // 接收失败处理
  Error_Handler();
}
// spi_rx_buffer 中现在包含了接收到的数据
```
**HAL_SPI_TransmitReceive 函数**

```c
HAL_StatusTypeDef HAL_SPI_TransmitReceive(SPI_HandleTypeDef *hspi, uint8_t *pTxData, uint8_t *pRxData, uint16_t Size,
                                          uint32_t Timeout)
{
  uint16_t             initial_TxXferCount;
  uint32_t             tmp_mode;
  HAL_SPI_StateTypeDef tmp_state;
  uint32_t             tickstart;
#if (USE_SPI_CRC != 0U)
  __IO uint32_t tmpreg = 0U;
#endif /* USE_SPI_CRC */

  /* Variable used to alternate Rx and Tx during transfer */
  uint32_t             txallowed = 1U;
  HAL_StatusTypeDef    errorcode = HAL_OK;

  /* Check Direction parameter */
  assert_param(IS_SPI_DIRECTION_2LINES(hspi->Init.Direction));

  /* Process Locked */
  __HAL_LOCK(hspi);

  /* Init tickstart for timeout management*/
  tickstart = HAL_GetTick();

  /* Init temporary variables */
  tmp_state           = hspi->State;
  tmp_mode            = hspi->Init.Mode;
  initial_TxXferCount = Size;

  if (!((tmp_state == HAL_SPI_STATE_READY) || \
        ((tmp_mode == SPI_MODE_MASTER) && (hspi->Init.Direction == SPI_DIRECTION_2LINES) && (tmp_state == HAL_SPI_STATE_BUSY_RX))))
  {
    errorcode = HAL_BUSY;
    goto error;
  }

  if ((pTxData == NULL) || (pRxData == NULL) || (Size == 0U))
  {
    errorcode = HAL_ERROR;
    goto error;
  }

  /* Don't overwrite in case of HAL_SPI_STATE_BUSY_RX */
  if (hspi->State != HAL_SPI_STATE_BUSY_RX)
  {
    hspi->State = HAL_SPI_STATE_BUSY_TX_RX;
  }

  /* Set the transaction information */
  hspi->ErrorCode   = HAL_SPI_ERROR_NONE;
  hspi->pRxBuffPtr  = (uint8_t *)pRxData;
  hspi->RxXferCount = Size;
  hspi->RxXferSize  = Size;
  hspi->pTxBuffPtr  = (uint8_t *)pTxData;
  hspi->TxXferCount = Size;
  hspi->TxXferSize  = Size;

  /*Init field not used in handle to zero */
  hspi->RxISR       = NULL;
  hspi->TxISR       = NULL;

#if (USE_SPI_CRC != 0U)
  /* Reset CRC Calculation */
  if (hspi->Init.CRCCalculation == SPI_CRCCALCULATION_ENABLE)
  {
    SPI_RESET_CRC(hspi);
  }
#endif /* USE_SPI_CRC */

  /* Check if the SPI is already enabled */
  if ((hspi->