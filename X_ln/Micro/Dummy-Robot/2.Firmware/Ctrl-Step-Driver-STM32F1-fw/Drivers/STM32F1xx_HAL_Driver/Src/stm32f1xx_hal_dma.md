Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_dma.c`

好的，我们重新开始，对代码的关键部分进行解释，生成小段代码并附上中文描述，并包含代码的使用说明和简单的演示。

**总览:**

这份代码是 STM32F1xx 系列微控制器的 DMA (直接内存访问) HAL (硬件抽象层) 驱动。它提供了一系列函数，用于配置、启动、停止和管理 DMA 传输。DMA 允许外设和内存之间的数据传输，无需 CPU 的干预，从而提高系统的效率。

**1. `HAL_DMA_Init()` 函数:**

```c
HAL_StatusTypeDef HAL_DMA_Init(DMA_HandleTypeDef *hdma)
{
  uint32_t tmp = 0U;

  /* Check the DMA handle allocation */
  if(hdma == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_DMA_ALL_INSTANCE(hdma->Instance));
  assert_param(IS_DMA_DIRECTION(hdma->Init.Direction));
  assert_param(IS_DMA_PERIPHERAL_INC_STATE(hdma->Init.PeriphInc));
  assert_param(IS_DMA_MEMORY_INC_STATE(hdma->Init.MemInc));
  assert_param(IS_DMA_PERIPHERAL_DATA_SIZE(hdma->Init.PeriphDataAlignment));
  assert_param(IS_DMA_MEMORY_DATA_SIZE(hdma->Init.MemDataAlignment));
  assert_param(IS_DMA_MODE(hdma->Init.Mode));
  assert_param(IS_DMA_PRIORITY(hdma->Init.Priority));

#if defined (DMA2)
  /* calculation of the channel index */
  if ((uint32_t)(hdma->Instance) < (uint32_t)(DMA2_Channel1))
  {
    /* DMA1 */
    hdma->ChannelIndex = (((uint32_t)hdma->Instance - (uint32_t)DMA1_Channel1) / ((uint32_t)DMA1_Channel2 - (uint32_t)DMA1_Channel1)) << 2;
    hdma->DmaBaseAddress = DMA1;
  }
  else
  {
    /* DMA2 */
    hdma->ChannelIndex = (((uint32_t)hdma->Instance - (uint32_t)DMA2_Channel1) / ((uint32_t)DMA2_Channel2 - (uint32_t)DMA2_Channel1)) << 2;
    hdma->DmaBaseAddress = DMA2;
  }
#else
  /* DMA1 */
  hdma->ChannelIndex = (((uint32_t)hdma->Instance - (uint32_t)DMA1_Channel1) / ((uint32_t)DMA1_Channel2 - (uint32_t)DMA1_Channel1)) << 2;
  hdma->DmaBaseAddress = DMA1;
#endif /* DMA2 */

  /* Change DMA peripheral state */
  hdma->State = HAL_DMA_STATE_BUSY;

  /* Get the CR register value */
  tmp = hdma->Instance->CCR;

  /* Clear PL, MSIZE, PSIZE, MINC, PINC, CIRC and DIR bits */
  tmp &= ((uint32_t)~(DMA_CCR_PL    | DMA_CCR_MSIZE  | DMA_CCR_PSIZE  | \
                      DMA_CCR_MINC  | DMA_CCR_PINC   | DMA_CCR_CIRC   | \
                      DMA_CCR_DIR));

  /* Prepare the DMA Channel configuration */
  tmp |=  hdma->Init.Direction        |
          hdma->Init.PeriphInc           | hdma->Init.MemInc           |
          hdma->Init.PeriphDataAlignment | hdma->Init.MemDataAlignment |
          hdma->Init.Mode                | hdma->Init.Priority;

  /* Write to DMA Channel CR register */
  hdma->Instance->CCR = tmp;

  /* Initialise the error code */
  hdma->ErrorCode = HAL_DMA_ERROR_NONE;

  /* Initialize the DMA state*/
  hdma->State = HAL_DMA_STATE_READY;
  /* Allocate lock resource and initialize it */
  hdma->Lock = HAL_UNLOCKED;

  return HAL_OK;
}
```

**描述:**

*   该函数用于初始化 DMA 通道。
*   它接收一个 `DMA_HandleTypeDef` 结构体的指针，该结构体包含了 DMA 通道的配置信息。
*   函数首先检查句柄是否为空，并进行参数校验，确保配置的参数是有效的。
*   根据 DMA 控制器的型号（DMA1 或 DMA2），计算 DMA 通道的索引和基地址。
*   然后，清除 DMA 控制寄存器 (CCR) 中的相关位，并根据 `DMA_InitTypeDef` 结构体中的设置，配置传输方向、地址增量模式、数据大小、传输模式和优先级。
*   最后，设置 DMA 的状态为 `HAL_DMA_STATE_READY`，并初始化错误代码和锁。

**使用方法:**

1.  创建一个 `DMA_HandleTypeDef` 结构体，并填充 `Init` 成员，包含所需的 DMA 配置。
2.  调用 `HAL_DMA_Init()` 函数，传入 `DMA_HandleTypeDef` 结构体的指针。

**示例:**

```c
DMA_HandleTypeDef hdma_usart1_rx;

void HAL_UART_MspInit(UART_HandleTypeDef* huart)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(huart->Instance==USART1)
  {
  /* Peripheral clock enable */
  __HAL_RCC_USART1_CLK_ENABLE();

  __HAL_RCC_GPIOA_CLK_ENABLE();
  /**USART1 GPIO Configuration
  PA9     ------> USART1_TX
  PA10    ------> USART1_RX
  */
  GPIO_InitStruct.Pin = GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_10;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* USART1 DMA Init */
    hdma_usart1_rx.Instance = DMA1_Channel5;
    hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart1_rx.Init.Mode = DMA_NORMAL;
    hdma_usart1_rx.Init.Priority = DMA_PRIORITY_LOW;

    HAL_DMA_Init(&hdma_usart1_rx);

    __HAL_LINKDMA(huart,hdmarx,hdma_usart1_rx);

    /* USART1 interrupt Init */
    HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
  }
}
```

**2. `HAL_DMA_Start_IT()` 函数:**

```c
HAL_StatusTypeDef HAL_DMA_Start_IT(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_DMA_BUFFER_SIZE(DataLength));

  /* Process locked */
  __HAL_LOCK(hdma);

  if(HAL_DMA_STATE_READY == hdma->State)
  {
    /* Change DMA peripheral state */
    hdma->State = HAL_DMA_STATE_BUSY;
    hdma->ErrorCode = HAL_DMA_ERROR_NONE;

    /* Disable the peripheral */
    __HAL_DMA_DISABLE(hdma);

    /* Configure the source, destination address and the data length & clear flags*/
    DMA_SetConfig(hdma, SrcAddress, DstAddress, DataLength);

    /* Enable the transfer complete interrupt */
    /* Enable the transfer Error interrupt */
    if(NULL != hdma->XferHalfCpltCallback)
    {
      /* Enable the Half transfer complete interrupt as well */
      __HAL_DMA_ENABLE_IT(hdma, (DMA_IT_TC | DMA_IT_HT | DMA_IT_TE));
    }
    else
    {
      __HAL_DMA_DISABLE_IT(hdma, DMA_IT_HT);
      __HAL_DMA_ENABLE_IT(hdma, (DMA_IT_TC | DMA_IT_TE));
    }
    /* Enable the Peripheral */
    __HAL_DMA_ENABLE(hdma);
  }
  else
  {
    /* Process Unlocked */
    __HAL_UNLOCK(hdma);

    /* Remain BUSY */
    status = HAL_BUSY;
  }
  return status;
}
```

**描述:**

*   此函数启动 DMA 传输，并使能中断。
*   它接收一个 `DMA_HandleTypeDef` 结构体的指针，源地址、目标地址和数据长度。
*   函数首先检查参数的有效性，并尝试获取锁，防止多个任务同时访问 DMA 资源。
*   然后，将 DMA 的状态设置为 `HAL_DMA_STATE_BUSY`，禁用 DMA 通道，调用 `DMA_SetConfig()` 函数配置源地址、目标地址和数据长度，并清除 DMA 标志。
*   根据是否定义了半传输完成回调函数 `XferHalfCpltCallback`，使能不同的中断：
    *   如果定义了 `XferHalfCpltCallback`，则使能传输完成中断 (DMA_IT_TC)、半传输完成中断 (DMA_IT_HT) 和传输错误中断 (DMA_IT_TE)。
    *   否则，禁用半传输完成中断，使能传输完成中断和传输错误中断。
*   最后，使能 DMA 通道。

**使用方法:**

1.  确保已经调用 `HAL_DMA_Init()` 函数初始化 DMA 通道。
2.  配置源地址、目标地址和数据长度。
3.  调用 `HAL_DMA_Start_IT()` 函数，启动 DMA 传输。

**示例:**

```c
uint8_t rx_buffer[100];

HAL_UART_Receive_DMA(&huart1, rx_buffer, 100); // UART接收数据到rx_buffer，使用DMA

// 或者
HAL_DMA_Start_IT(&hdma_usart1_rx, (uint32_t)&USART1->DR, (uint32_t)rx_buffer, 100); // 同样的效果
```

**3. `HAL_DMA_IRQHandler()` 函数:**

```c
void HAL_DMA_IRQHandler(DMA_HandleTypeDef *hdma)
{
  uint32_t flag_it = hdma->DmaBaseAddress->ISR;
  uint32_t source_it = hdma->Instance->CCR;

  /* Half Transfer Complete Interrupt management ******************************/
  if (((flag_it & (DMA_FLAG_HT1 << hdma->ChannelIndex)) != RESET) && ((source_it & DMA_IT_HT) != RESET))
  {
    /* Disable the half transfer interrupt if the DMA mode is not CIRCULAR */
    if((hdma->Instance->CCR & DMA_CCR_CIRC) == 0U)
    {
      /* Disable the half transfer interrupt */
      __HAL_DMA_DISABLE_IT(hdma, DMA_IT_HT);
    }
    /* Clear the half transfer complete flag */
    __HAL_DMA_CLEAR_FLAG(hdma, __HAL_DMA_GET_HT_FLAG_INDEX(hdma));

    /* DMA peripheral state is not updated in Half Transfer */
    /* but in Transfer Complete case */

    if(hdma->XferHalfCpltCallback != NULL)
    {
      /* Half transfer callback */
      hdma->XferHalfCpltCallback(hdma);
    }
  }

  /* Transfer Complete Interrupt management ***********************************/
  else if (((flag_it & (DMA_FLAG_TC1 << hdma->ChannelIndex)) != RESET) && ((source_it & DMA_IT_TC) != RESET))
  {
    if((hdma->Instance->CCR & DMA_CCR_CIRC) == 0U)
    {
      /* Disable the transfer complete and error interrupt */
      __HAL_DMA_DISABLE_IT(hdma, DMA_IT_TE | DMA_IT_TC);

      /* Change the DMA state */
      hdma->State = HAL_DMA_STATE_READY;
    }
    /* Clear the transfer complete flag */
      __HAL_DMA_CLEAR_FLAG(hdma, __HAL_DMA_GET_TC_FLAG_INDEX(hdma));

    /* Process Unlocked */
    __HAL_UNLOCK(hdma);

    if(hdma->XferCpltCallback != NULL)
    {
      /* Transfer complete callback */
      hdma->XferCpltCallback(hdma);
    }
  }

  /* Transfer Error Interrupt management **************************************/
  else if (( RESET != (flag_it & (DMA_FLAG_TE1 << hdma->ChannelIndex))) && (RESET != (source_it & DMA_IT_TE)))
  {
    /* When a DMA transfer error occurs */
    /* A hardware clear of its EN bits is performed */
    /* Disable ALL DMA IT */
    __HAL_DMA_DISABLE_IT(hdma, (DMA_IT_TC | DMA_IT_HT | DMA_IT_TE));

    /* Clear all flags */
    hdma->DmaBaseAddress->IFCR = (DMA_ISR_GIF1 << hdma->ChannelIndex);

    /* Update error code */
    hdma->ErrorCode = HAL_DMA_ERROR_TE;

    /* Change the DMA state */
    hdma->State = HAL_DMA_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hdma);

    if (hdma->XferErrorCallback != NULL)
    {
      /* Transfer error callback */
      hdma->XferErrorCallback(hdma);
    }
  }
  return;
}
```

**描述:**

*   这个函数是 DMA 中断处理函数。当 DMA 传输完成、半传输完成或发生错误时，会调用此函数。
*   它首先读取 DMA 的中断状态寄存器 (ISR) 和控制寄存器 (CCR)，以确定触发中断的原因。
*   然后，根据中断原因执行相应的操作：
    *   **半传输完成中断 (HT)**: 如果使能了半传输完成中断，并且发生了半传输完成事件，则清除半传输完成标志，并调用 `XferHalfCpltCallback` 回调函数。  如果DMA设置为非循环模式，会禁用半传输中断。
    *   **传输完成中断 (TC)**: 如果使能了传输完成中断，并且发生了传输完成事件，则清除传输完成标志，将 DMA 状态设置为 `HAL_DMA_STATE_READY`，释放锁，并调用 `XferCpltCallback` 回调函数。 如果设置为非循环模式，会禁用传输完成和错误中断。
    *   **传输错误中断 (TE)**: 如果使能了传输错误中断，并且发生了传输错误事件，则禁用所有 DMA 中断，清除所有标志，设置 DMA 错误代码，将 DMA 状态设置为 `HAL_DMA_STATE_READY`，释放锁，并调用 `XferErrorCallback` 回调函数。

**使用方法:**

1.  在中断向量表中注册 DMA 中断处理函数。  通常，这个函数的名字是 `DMA1_Channelx_IRQHandler`，其中 `x` 是 DMA 通道号。
2.  在 DMA 中断处理函数中调用 `HAL_DMA_IRQHandler()` 函数，传入 `DMA_HandleTypeDef` 结构体的指针。

**示例:**

```c
void DMA1_Channel5_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&hdma_usart1_rx);
}
```

**4. `DMA_SetConfig()` 函数:**

```c
static void DMA_SetConfig(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength)
{
  /* Clear all flags */
  hdma->DmaBaseAddress->IFCR = (DMA_ISR_GIF1 << hdma->ChannelIndex);

  /* Configure DMA Channel data length */
  hdma->Instance->CNDTR = DataLength;

  /* Memory to Peripheral */
  if((hdma->Init.Direction) == DMA_MEMORY_TO_PERIPH)
  {
    /* Configure DMA Channel destination address */
    hdma->Instance->CPAR = DstAddress;

    /* Configure DMA Channel source address */
    hdma->Instance->CMAR = SrcAddress;
  }
  /* Peripheral to Memory */
  else
  {
    /* Configure DMA Channel source address */
    hdma->Instance->CPAR = SrcAddress;

    /* Configure DMA Channel destination address */
    hdma->Instance->CMAR = DstAddress;
  }
}
```

**描述:**

*   此函数用于配置 DMA 的传输参数，例如源地址、目标地址和数据长度。
*   它接收一个 `DMA_HandleTypeDef` 结构体的指针，源地址、目标地址和数据长度。
*   函数首先清除 DMA 的所有标志。
*   然后，配置 DMA 通道的数据长度寄存器 (CNDTR)，源地址和目标地址寄存器 (CPAR 和 CMAR)，根据传输方向 (内存到外设或外设到内存) 设置相应的寄存器。

**使用方法:**

1.  在调用 `HAL_DMA_Start()` 或 `HAL_DMA_Start_IT()` 函数之前，必须先调用 `DMA_SetConfig()` 函数配置传输参数。  通常 `HAL_DMA_Start` 和 `HAL_DMA_Start_IT` 内部会调用这个函数。

**5. DMA HAL 驱动宏:**

*   `__HAL_DMA_ENABLE(hdma)`: 使能指定的 DMA 通道。
*   `__HAL_DMA_DISABLE(hdma)`: 禁用指定的 DMA 通道。
*   `__HAL_DMA_GET_FLAG(hdma, flag)`: 获取指定的 DMA 通道标志。
*   `__HAL_DMA_CLEAR_FLAG(hdma, flag)`: 清除指定的 DMA 通道标志。
*   `__HAL_DMA_ENABLE_IT(hdma, it)`: 使能指定的 DMA 通道中断。
*   `__HAL_DMA_DISABLE_IT(hdma, it)`: 禁用指定的 DMA 通道中断。

**总结:**

这份 DMA HAL 驱动提供了一套完整的 API，用于在 STM32F1xx 系列微控制器上配置和使用 DMA。  通过使用这些函数，可以简化 DMA 编程，并提高系统的效率。

**简单 Demo (UART 接收):**

这个 Demo 展示了如何使用 DMA 从 UART 接收数据到一个缓冲区。

```c
#include "stm32f1xx_hal.h"

UART_HandleTypeDef huart1;
DMA_HandleTypeDef  hdma_usart1_rx;
uint8_t            rx_buffer[100];

void SystemClock_Config(void); // 时钟配置
void Error_Handler(void);      // 错误处理

int main(void)
{
  HAL_Init();
  SystemClock_Config();

  // 1. 初始化 UART
  huart1.Instance          = USART1;
  huart1.Init.BaudRate     = 115200;
  huart1.Init.WordLength   = UART_WORDLENGTH_8B;
  huart1.Init.StopBits     = UART_STOPBITS_1;
  huart1.Init.Parity       = UART_PARITY_NONE;
  huart1.Init.Mode         = UART_MODE_RX; // 只接收
  huart1.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK) {
    Error_Handler();
  }

  // 2.  关联 DMA 和 UART (在 HAL_UART_MspInit 中完成)
  // 请看前面 HAL_UART_MspInit 的例子

  // 3. 启动 DMA 接收
  HAL_UART_Receive_DMA(&huart1, rx_buffer, sizeof(rx_buffer));

  while (1) {
    // 循环等待数据接收，或者做其他事情
    // 数据接收完成后，中断服务程序会处理数据 (如果定义了回调函数)
  }
}

// 中断服务程序 (可选)
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART1) {
    // 在这里处理接收到的数据，例如：
    // printf("Received data: %s\n", rx_buffer);
    // 然后重新启动 DMA 接收，循环接收数据
    HAL_UART_Receive_DMA(&huart1, rx_buffer, sizeof(rx_buffer));
  }
}

void USART1_IRQHandler(void)
{
  HAL_UART_IRQHandler(&huart1);
}

void DMA1_Channel5_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&hdma_usart1_rx);
}

// 时钟配置和错误处理
void SystemClock_Config(void) { /* ... */ }
void Error_Handler(void) { /* ... */ }
```

请注意：这个 Demo 需要相应的时钟配置，GPIO 配置，UART配置，以及错误处理函数。 为了代码的简洁，这些函数没有完全展开，需要根据实际情况进行填充。 另外需要在 `stm32f1xx_it.c` 中手动添加 `USART1_IRQHandler` 和 `DMA1_Channel5_IRQHandler` 中断处理函数。