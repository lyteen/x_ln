Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_dma.h`

好的，重新开始。 这次我会更详细地解释 STM32F1xx HAL 库 DMA 模块的头文件 `stm32f1xx_hal_dma.h`，并提供代码示例以及中文解释。

**`stm32f1xx_hal_dma.h` 文件的主要作用：**

该头文件定义了 STM32F1xx 系列微控制器的 DMA (Direct Memory Access, 直接存储器访问) 外设的硬件抽象层 (HAL)。 它提供了一组函数和数据结构，用于配置和控制 DMA，从而实现高效的数据传输，而无需 CPU 的持续干预。

**关键部分详解 (带有中文注释和代码示例):**

**1. 头文件保护和 C++ 兼容:**

```c
#ifndef __STM32F1xx_HAL_DMA_H
#define __STM32F1xx_HAL_DMA_H

#ifdef __cplusplus
 extern "C" {
#endif
```

*   `#ifndef __STM32F1xx_HAL_DMA_H` 和 `#define __STM32F1xx_HAL_DMA_H`：防止头文件被重复包含，避免编译错误。
*   `#ifdef __cplusplus` 和 `extern "C" { ... #endif`：当使用 C++ 编译器时，确保 C 函数的链接方式与 C 兼容，允许 C++ 代码调用 C 函数。

**2. 包含必要的头文件:**

```c
#include "stm32f1xx_hal_def.h"
```

*   `#include "stm32f1xx_hal_def.h"`： 包含 HAL 库的通用定义，例如 `HAL_StatusTypeDef` (函数返回值类型) 和一些常用的宏。

**3. DMA 配置结构体 (`DMA_InitTypeDef`):**

```c
typedef struct
{
  uint32_t Direction;                 /*!< 数据传输方向 */
  uint32_t PeriphInc;                 /*!< 外设地址是否自增 */
  uint32_t MemInc;                    /*!< 存储器地址是否自增 */
  uint32_t PeriphDataAlignment;       /*!< 外设数据宽度 */
  uint32_t MemDataAlignment;          /*!< 存储器数据宽度 */
  uint32_t Mode;                      /*!< DMA 模式 (Normal/Circular) */
  uint32_t Priority;                  /*!< DMA 优先级 */
} DMA_InitTypeDef;
```

*   **`Direction` (传输方向):**
    *   `DMA_PERIPH_TO_MEMORY`：外设到存储器 (例如，ADC 读取数据到内存).
    *   `DMA_MEMORY_TO_PERIPH`：存储器到外设 (例如，内存数据通过 SPI 发送).
    *   `DMA_MEMORY_TO_MEMORY`：存储器到存储器.
*   **`PeriphInc` (外设地址自增):** `DMA_PINC_ENABLE` 或 `DMA_PINC_DISABLE`。 如果使能，每次传输后，外设地址会自动增加。
*   **`MemInc` (存储器地址自增):** `DMA_MINC_ENABLE` 或 `DMA_MINC_DISABLE`。 如果使能，每次传输后，存储器地址会自动增加。
*   **`PeriphDataAlignment` (外设数据宽度):**  `DMA_PDATAALIGN_BYTE`, `DMA_PDATAALIGN_HALFWORD`, `DMA_PDATAALIGN_WORD` (8位，16位，32位)。
*   **`MemDataAlignment` (存储器数据宽度):** `DMA_MDATAALIGN_BYTE`, `DMA_MDATAALIGN_HALFWORD`, `DMA_MDATAALIGN_WORD` (8位，16位，32位)。
*   **`Mode` (DMA 模式):**
    *   `DMA_NORMAL`：正常模式，传输完成后停止 DMA。
    *   `DMA_CIRCULAR`：循环模式，传输完成后自动重新开始，用于连续数据流。
*   **`Priority` (DMA 优先级):**  `DMA_PRIORITY_LOW`, `DMA_PRIORITY_MEDIUM`, `DMA_PRIORITY_HIGH`, `DMA_PRIORITY_VERY_HIGH`。  当多个 DMA 通道同时请求时，优先级高的通道先执行。

**代码示例 (初始化 DMA):**

```c
DMA_HandleTypeDef hdma_usart1_rx; // USART1 接收 DMA 句柄

void MX_DMA_Init(void)
{
  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel5_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel5_IRQn);

}

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
    PA10     ------> USART1_RX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* DMA controller clock enable */
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* USART1 DMA Init */
    hdma_usart1_rx.Instance = DMA1_Channel5;
    hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart1_rx.Init.Mode = DMA_CIRCULAR;  // 使用循环模式，连续接收
    hdma_usart1_rx.Init.Priority = DMA_PRIORITY_LOW;

    HAL_DMA_Init(&hdma_usart1_rx);

    __HAL_LINKDMA(huart, hdmarx, hdma_usart1_rx);

    /* USART1 interrupt Init */
    HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
  }

}
```

**中文解释:**

1.  定义一个 `DMA_HandleTypeDef` 类型的变量 `hdma_usart1_rx`，作为 USART1 接收 DMA 的句柄。
2.  在`MX_DMA_Init`函数中，使能 DMA1 的时钟，并配置 DMA 中断。
3.  在 `HAL_UART_MspInit` 函数（USART 初始化函数，由 HAL 库提供）中：

    *   使能 USART1 和 GPIO 的时钟。
    *   配置 USART1 的 TX 和 RX 引脚。
    *   使能 DMA1 的时钟。
    *   初始化 `hdma_usart1_rx` 句柄：
        *   `Instance = DMA1_Channel5`：指定 DMA 通道为 DMA1 的通道 5。
        *   `Direction = DMA_PERIPH_TO_MEMORY`：数据传输方向是从 USART1 (外设) 到内存。
        *   `PeriphInc = DMA_PINC_DISABLE`：外设地址不自增 (USART1 的数据寄存器地址是固定的).
        *   `MemInc = DMA_MINC_ENABLE`：存储器地址自增 (将接收到的数据存储到连续的内存地址).
        *   `PeriphDataAlignment = DMA_PDATAALIGN_BYTE`：外设数据宽度为字节 (USART1 每次接收一个字节).
        *   `MemDataAlignment = DMA_MDATAALIGN_BYTE`：存储器数据宽度为字节.
        *   `Mode = DMA_CIRCULAR`：DMA 模式为循环模式，可以持续接收数据。
        *   `Priority = DMA_PRIORITY_LOW`：DMA 优先级为低。
    *   调用 `HAL_DMA_Init(&hdma_usart1_rx)` 初始化 DMA。
    *   使用 `__HAL_LINKDMA(huart, hdmarx, hdma_usart1_rx)` 将 DMA 句柄链接到 UART 句柄，以便 UART 驱动程序可以使用 DMA。
4.  配置并使能 USART1 的中断。

**4. HAL DMA 状态 (`HAL_DMA_StateTypeDef`) 和错误代码:**

```c
typedef enum
{
  HAL_DMA_STATE_RESET             = 0x00U,  /*!< DMA 未初始化或禁用 */
  HAL_DMA_STATE_READY             = 0x01U,  /*!< DMA 初始化完成，准备就绪 */
  HAL_DMA_STATE_BUSY              = 0x02U,  /*!< DMA 传输正在进行中 */
  HAL_DMA_STATE_TIMEOUT           = 0x03U   /*!< DMA 超时 */
}HAL_DMA_StateTypeDef;

typedef enum
{
  HAL_DMA_ERROR_NONE                     = 0x00000000U,    /*!< 无错误 */
  HAL_DMA_ERROR_TE                       = 0x00000001U,    /*!< 传输错误 */
  HAL_DMA_ERROR_NO_XFER                  = 0x00000004U,    /*!< 没有正在进行的传输 */
  HAL_DMA_ERROR_TIMEOUT                  = 0x00000020U,    /*!< 超时错误 */
  HAL_DMA_ERROR_NOT_SUPPORTED            = 0x00000100U     /*!< 不支持的模式 */
}HAL_DMA_ErrorTypeDef;
```

*   `HAL_DMA_StateTypeDef`：定义了 DMA 的状态，用于检查 DMA 是否处于空闲、忙碌或错误状态。
*   `HAL_DMA_ErrorTypeDef`：定义了 DMA 可能发生的错误类型。

**5. HAL DMA 回调函数 (`HAL_DMA_CallbackIDTypeDef`):**

```c
typedef enum
{
  HAL_DMA_XFER_CPLT_CB_ID          = 0x00U,    /*!< 完整传输完成 */
  HAL_DMA_XFER_HALFCPLT_CB_ID      = 0x01U,    /*!< 半传输完成 */
  HAL_DMA_XFER_ERROR_CB_ID         = 0x02U,    /*!< 错误 */
  HAL_DMA_XFER_ABORT_CB_ID         = 0x03U,    /*!< 中止 */
  HAL_DMA_XFER_ALL_CB_ID           = 0x04U     /*!< 所有 */

}HAL_DMA_CallbackIDTypeDef;
```

*   `HAL_DMA_CallbackIDTypeDef`：定义了 DMA 事件的回调函数 ID。  你可以注册这些回调函数，以便在 DMA 传输完成、发生错误或中止时执行特定的操作。

**代码示例 (注册 DMA 完成回调函数):**

```c
void DMA1_Channel5_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&hdma_usart1_rx);
}

void HAL_DMA_XferCpltCallback(DMA_HandleTypeDef *hdma)
{
  if (hdma->Instance == DMA1_Channel5)
  {
    // DMA 传输完成，处理接收到的数据
    // 例如，设置一个标志位，通知主循环处理数据
    usart1_rx_complete = 1;
  }
}

int main(void) {
  // ... 初始化代码 ...
  HAL_DMA_RegisterCallback(&hdma_usart1_rx, HAL_DMA_XFER_CPLT_CB_ID, HAL_DMA_XferCpltCallback);

  // ... 启动 DMA 接收 ...
  HAL_UART_Receive_DMA(&huart1, usart1_rx_buffer, USART1_RX_BUFFER_SIZE);

  while (1) {
    if (usart1_rx_complete) {
      // 处理接收到的数据 (usart1_rx_buffer)
      usart1_rx_complete = 0; // 清除标志位
      // 重新启动 DMA 接收 (如果需要持续接收)
      HAL_UART_Receive_DMA(&huart1, usart1_rx_buffer, USART1_RX_BUFFER_SIZE);
    }
  }
}
```

**中文解释:**

1.  `DMA1_Channel5_IRQHandler` 是 DMA1 通道 5 的中断处理函数。 在此函数中，调用 `HAL_DMA_IRQHandler` 来处理 DMA 中断事件。
2.  `HAL_DMA_XferCpltCallback` 是 DMA 传输完成的回调函数。 当 DMA 传输完成时，HAL 库会自动调用此函数。 在此函数中，你可以执行一些操作，例如设置一个标志位，通知主循环处理接收到的数据。
3.  `HAL_DMA_RegisterCallback` 函数用于注册 DMA 回调函数。 将 `HAL_DMA_XferCpltCallback` 注册为 `HAL_DMA_XFER_CPLT_CB_ID` (传输完成) 的回调函数。
4.  `HAL_UART_Receive_DMA` 函数启动 USART1 的 DMA 接收。
5.  在主循环中，检查 `usart1_rx_complete` 标志位。 如果该标志位被设置，则处理接收到的数据，并重新启动 DMA 接收。

**6. DMA 函数 (`HAL_DMA_Init`, `HAL_DMA_Start`, `HAL_DMA_Abort`, 等):**

头文件中声明了许多 DMA 函数，用于初始化、启动、停止和控制 DMA 传输。

*   `HAL_DMA_Init(DMA_HandleTypeDef *hdma)`: 初始化 DMA 通道。  需要传递一个指向 `DMA_HandleTypeDef` 结构体的指针，该结构体包含 DMA 通道的配置信息。
*   `HAL_DMA_DeInit(DMA_HandleTypeDef *hdma)`: 释放 DMA 通道。
*   `HAL_DMA_Start(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength)`: 以阻塞模式启动 DMA 传输。  程序会在此函数中等待，直到 DMA 传输完成。
*   `HAL_DMA_Start_IT(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength)`: 以中断模式启动 DMA 传输。  DMA 传输在后台运行，完成后会触发中断。
*   `HAL_DMA_Abort(DMA_HandleTypeDef *hdma)`: 停止 DMA 传输 (阻塞模式)。
*   `HAL_DMA_Abort_IT(DMA_HandleTypeDef *hdma)`: 停止 DMA 传输 (中断模式)。
*   `HAL_DMA_PollForTransfer(DMA_HandleTypeDef *hdma, uint32_t CompleteLevel, uint32_t Timeout)`:  轮询 DMA 传输是否完成或发生错误。
*   `HAL_DMA_IRQHandler(DMA_HandleTypeDef *hdma)`: DMA 中断处理函数。  需要在 DMA 中断服务例程中调用此函数，以处理 DMA 中断事件。
*   `HAL_DMA_GetState(DMA_HandleTypeDef *hdma)`: 获取 DMA 的当前状态。
*   `HAL_DMA_GetError(DMA_HandleTypeDef *hdma)`: 获取 DMA 的错误代码。
*   `HAL_DMA_RegisterCallback()`: 注册 DMA 回调函数。
*   `HAL_DMA_UnRegisterCallback()`: 取消注册 DMA 回调函数。

**7. 宏定义:**

头文件中还定义了许多宏，用于简化 DMA 配置和控制。 例如：

*   `__HAL_DMA_ENABLE(DMA_HandleTypeDef *hdma)`: 启用 DMA 通道。
*   `__HAL_DMA_DISABLE(DMA_HandleTypeDef *hdma)`: 禁用 DMA 通道。
*   `__HAL_DMA_ENABLE_IT(DMA_HandleTypeDef *hdma, uint32_t __INTERRUPT__)`: 启用 DMA 中断。
*   `__HAL_DMA_DISABLE_IT(DMA_HandleTypeDef *hdma, uint32_t __INTERRUPT__)`: 禁用 DMA 中断。
*   `__HAL_DMA_GET_COUNTER(DMA_HandleTypeDef *hdma)`: 获取剩余的 DMA 传输计数。

**总结:**

`stm32f1xx_hal_dma.h` 头文件是 STM32F1xx HAL 库中 DMA 模块的核心。 它定义了用于配置和控制 DMA 传输的数据结构、函数和宏。 通过使用这些 API，你可以方便地实现高效的数据传输，减轻 CPU 的负担。  重要的是要理解 `DMA_InitTypeDef` 结构体的每个成员的含义，以及如何使用 `HAL_DMA_Init`、`HAL_DMA_Start_IT` 和 DMA 中断回调函数来实现特定的 DMA 传输场景。