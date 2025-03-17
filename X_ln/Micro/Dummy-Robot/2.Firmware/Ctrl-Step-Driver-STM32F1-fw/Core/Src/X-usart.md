Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\usart.c`

好的，这次我们重新开始，并以更细致的方式来改进你的USART驱动代码，提供更清晰的代码片段和中文解释。

**目标:**

1.  **更高效的DMA接收:** 使用循环DMA接收，避免频繁启动DMA。
2.  **空闲中断处理优化:** 更可靠地处理空闲中断，确定接收数据长度。
3.  **代码可读性:** 提升代码可读性，添加注释。
4.  **提供示例:**  展示如何使用回调函数处理接收到的数据。

**改进后的代码:**

**1. `usart.h` 头文件 (保持不变, 除非你想添加新的宏定义):**

```c
#ifndef __USART_H
#define __USART_H

#include "stm32f1xx_hal.h"  // 替换为你的STM32系列对应的头文件

#define BUFFER_SIZE 256       // 接收缓冲区大小 (根据你的需求调整)

extern UART_HandleTypeDef huart1;
extern DMA_HandleTypeDef hdma_usart1_rx;
extern DMA_HandleTypeDef hdma_usart1_tx;

void MX_USART1_UART_Init(void);
void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle);
void HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle);

void Uart_SetRxCpltCallBack(void(* xerc)(uint8_t*, uint16_t));

#endif /* __USART_H */
```

**2. `usart.c` 源文件:**

```c
/* Includes ------------------------------------------------------------------*/
#include "usart.h"
#include "common_inc.h"  // 你自定义的包含文件，例如包含Error_Handler
#include "Platform/retarget.h" // 如果你使用了printf重定向

/* USER CODE BEGIN 0 */
#include <string.h> // For memset

volatile uint8_t rxLen = 0;                     // 接收到的数据长度
uint8_t rx_buffer[BUFFER_SIZE] = {0};           // 接收缓冲区
void (* OnRecvEnd)(uint8_t* data, uint16_t len); // 回调函数指针

/* USER CODE END 0 */

UART_HandleTypeDef huart1; // UART句柄
DMA_HandleTypeDef hdma_usart1_rx; // DMA接收句柄
DMA_HandleTypeDef hdma_usart1_tx; // DMA发送句柄

/* USART1 init function */
void MX_USART1_UART_Init(void)
{
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;

    if (HAL_UART_Init(&huart1) != HAL_OK) {
        Error_Handler(); // 处理初始化错误
    }

    /* USER CODE BEGIN USART1_Init 2 */
    __HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE); // 使能空闲中断

    RetargetInit(&huart1); // 初始化printf重定向 (如果使用)
    Uart_SetRxCpltCallBack(OnUartCmd); // 设置接收完成回调函数

    // 启动循环DMA接收
    HAL_UART_Receive_DMA(&huart1, rx_buffer, BUFFER_SIZE);
    /* USER CODE END USART1_Init 2 */
}


void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    if (uartHandle->Instance == USART1) {
        /* USER CODE BEGIN USART1_MspInit 0 */
        /* USER CODE END USART1_MspInit 0 */

        /* Peripheral clock enable */
        __HAL_RCC_USART1_CLK_ENABLE();
        __HAL_RCC_GPIOB_CLK_ENABLE();

        /**USART1 GPIO Configuration
        PB6     ------> USART1_TX
        PB7     ------> USART1_RX
        */
        GPIO_InitStruct.Pin = GPIO_PIN_6;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

        GPIO_InitStruct.Pin = GPIO_PIN_7;
        GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

        __HAL_AFIO_REMAP_USART1_ENABLE();

        /* USART1 DMA Init */
        /* USART1_RX Init */
        hdma_usart1_rx.Instance = DMA1_Channel5;
        hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
        hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
        hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
        hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
        hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
        hdma_usart1_rx.Init.Mode = DMA_CIRCULAR;  // 使用循环DMA
        hdma_usart1_rx.Init.Priority = DMA_PRIORITY_MEDIUM;

        if (HAL_DMA_Init(&hdma_usart1_rx) != HAL_OK) {
            Error_Handler();
        }

        __HAL_LINKDMA(uartHandle, hdmarx, hdma_usart1_rx);

        /* USART1_TX Init */
        hdma_usart1_tx.Instance = DMA1_Channel4;
        hdma_usart1_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
        hdma_usart1_tx.Init.PeriphInc = DMA_PINC_DISABLE;
        hdma_usart1_tx.Init.MemInc = DMA_MINC_ENABLE;
        hdma_usart1_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
        hdma_usart1_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
        hdma_usart1_tx.Init.Mode = DMA_NORMAL;
        hdma_usart1_tx.Init.Priority = DMA_PRIORITY_MEDIUM;

        if (HAL_DMA_Init(&hdma_usart1_tx) != HAL_OK) {
            Error_Handler();
        }

        __HAL_LINKDMA(uartHandle, hdmatx, hdma_usart1_tx);

        /* USART1 interrupt Init */
        HAL_NVIC_SetPriority(USART1_IRQn, 3, 0);
        HAL_NVIC_EnableIRQ(USART1_IRQn);

        /* USER CODE BEGIN USART1_MspInit 1 */
        /* USER CODE END USART1_MspInit 1 */
    }
}


void HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle)
{
    if (uartHandle->Instance == USART1) {
        /* USER CODE BEGIN USART1_MspDeInit 0 */
        /* USER CODE END USART1_MspDeInit 0 */

        /* Peripheral clock disable */
        __HAL_RCC_USART1_CLK_DISABLE();

        /**USART1 GPIO Configuration
        PB6     ------> USART1_TX
        PB7     ------> USART1_RX
        */
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6 | GPIO_PIN_7);

        /* USART1 DMA DeInit */
        HAL_DMA_DeInit(uartHandle->hdmarx);
        HAL_DMA_DeInit(uartHandle->hdmatx);

        /* USART1 interrupt Deinit */
        HAL_NVIC_DisableIRQ(USART1_IRQn);

        /* USER CODE BEGIN USART1_MspDeInit 1 */
        /* USER CODE END USART1_MspDeInit 1 */
    }
}


/* USER CODE BEGIN 1 */
void Uart_SetRxCpltCallBack(void(* xerc)(uint8_t*, uint16_t))
{
    OnRecvEnd = xerc;
}

/*
 *  USART1 中断服务函数
 */
void USART1_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart1);
}

/*
 *  UART 空闲中断回调函数
 */
void HAL_UART_RxIdleCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        // 1.  计算接收到的数据长度
        // 获取剩余未接收的数据量 (以字节为单位)
        uint32_t remaining_bytes = __HAL_DMA_GET_COUNTER(huart->hdmarx);

        // 接收到的数据长度 = 总缓冲区大小 - 剩余未接收的数据量
        rxLen = BUFFER_SIZE - remaining_bytes;

        // 2.  停止DMA接收 (可选，但推荐，特别是对于非循环模式)
        HAL_UART_DMAStop(huart);

        // 3.  调用回调函数处理数据
        if (OnRecvEnd != NULL) {
            OnRecvEnd(rx_buffer, rxLen);
        }

        // 4. 清空接收缓冲区, 避免旧数据干扰 (重要!)
        memset(rx_buffer, 0, BUFFER_SIZE);

        // 5. 重新启动DMA接收 (对于循环DMA，这一步不是必须的，但对于非循环DMA是必须的)
        HAL_UART_Receive_DMA(huart, rx_buffer, BUFFER_SIZE);
    }
}

/* USER CODE END 1 */
```

**3. 添加到 `stm32f1xx_it.c` (或者你的中断处理文件):**

```c
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32f1xx_it.h"
#include "usart.h" // 包含 usart.h

/* ... 其他代码 ... */

/* USART1 global interrupt handler */
void USART1_IRQHandler(void)
{
  /* USER CODE BEGIN USART1_IRQn 0 */

  /* USER CODE END USART1_IRQn 0 */
  HAL_UART_IRQHandler(&huart1);
  /* USER CODE BEGIN USART1_IRQn 1 */

  /* USER CODE END USART1_IRQn 1 */
}
```

**4.  `common_inc.h` (或者你存放回调函数定义的文件):**

```c
#ifndef __COMMON_INC_H
#define __COMMON_INC_H

#include <stdint.h>

void OnUartCmd(uint8_t* data, uint16_t len);
void Error_Handler(void);

#endif
```

**5. 实现回调函数 (示例，放到你的主程序或其他源文件中):**

```c
#include "main.h" // 包含 main.h  (为了使用 HAL 库的函数)
#include "usart.h"
#include <stdio.h> // 包含 printf

void OnUartCmd(uint8_t* data, uint16_t len)
{
    printf("Received data: %s, length: %d\r\n", data, len);
    // 在这里处理接收到的数据，例如解析命令、更新状态等
}

void Error_Handler(void)
{
    /* USER CODE BEGIN Error_Handler_Debug */
    /* User can add his own implementation to report the HAL error return state */
    printf("Error occurred!\r\n");
    while(1)
    {
    }
    /* USER CODE END Error_Handler_Debug */
}
```

**代码解释 (中文):**

*   **循环DMA:** `hdma_usart1_rx.Init.Mode = DMA_CIRCULAR;`  将DMA设置为循环模式，这意味着DMA在接收完缓冲区后会自动从头开始，不需要手动重新启动DMA。  这大大简化了接收流程。
*   **空闲中断:**  `__HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE);`  使能了UART的空闲中断。  当UART线路空闲一段时间后 (没有数据传输)，会触发这个中断。  这个中断很重要，因为它可以告诉我们何时接收完成了一个完整的数据包。
*   **`HAL_UART_RxIdleCallback()`:**  这是空闲中断的回调函数。  在这个函数中，我们计算接收到的数据长度，停止DMA，调用回调函数处理数据，清空缓冲区，然后重新启动DMA。
    *   **计算长度:** `rxLen = BUFFER_SIZE - __HAL_DMA_GET_COUNTER(huart->hdmarx);`  `__HAL_DMA_GET_COUNTER()` 返回DMA剩余未传输的数据量。  用缓冲区总大小减去这个值，就得到实际接收到的数据长度。
    *   **停止DMA (可选):** `HAL_UART_DMAStop(huart);`  虽然循环DMA不需要停止，但停止DMA可以防止在处理数据期间，新的数据覆盖旧的数据。对于非循环DMA，必须停止。
    *   **清空缓冲区:** `memset(rx_buffer, 0, BUFFER_SIZE);` **非常重要!**  如果不清空缓冲区，下次接收到的数据会追加到旧数据后面，导致错误。
    *   **重新启动DMA:** `HAL_UART_Receive_DMA(huart, rx_buffer, BUFFER_SIZE);`  对于非循环DMA，在处理完数据后，必须重新启动DMA，才能继续接收数据。循环DMA本身会一直运行，但为了保证数据的准确性，在回调函数中还是重新启动一次。
*   **回调函数:**  `Uart_SetRxCpltCallBack(OnUartCmd);`  允许你设置一个回调函数，当接收到数据后，这个函数会被调用。  这使得你可以将接收数据的处理逻辑与UART驱动程序分离，使代码更模块化。
*   **`Error_Handler()`:**  一个简单的错误处理函数，当初始化或其他操作失败时，会调用这个函数。  你可以根据你的需求修改这个函数，例如输出错误信息到串口，或者重启系统。

**使用示例 (中文):**

1.  **初始化:** 在你的 `main()` 函数中，调用 `MX_USART1_UART_Init()` 初始化USART1。
2.  **设置回调:** 使用 `Uart_SetRxCpltCallBack(你的回调函数);` 设置你的数据处理函数。  例如，`Uart_SetRxCpltCallBack(OnUartCmd);`
3.  **发送数据:**  使用串口调试工具或其他设备向你的STM32发送数据。
4.  **处理数据:**  当STM32接收到数据后，`OnUartCmd()` 函数会被调用，你可以在这个函数中处理接收到的数据。

**优势:**

*   **高效:**  循环DMA最大限度地减少了CPU的干预，提高了接收效率。
*   **可靠:**  空闲中断处理能够准确地检测数据包的结束。
*   **灵活:**  回调函数机制使得数据处理更加灵活和模块化。
*   **可读:**  代码注释清晰，易于理解和修改。

**注意事项:**

*   **缓冲区大小:** `BUFFER_SIZE`  应该根据你的应用需求进行调整。  如果你的数据包很大，你需要增加缓冲区的大小。
*   **中断优先级:**  确保USART1中断的优先级设置正确，避免与其他中断冲突。
*   **错误处理:**  `Error_Handler()`  应该根据你的需求进行完善，例如添加更详细的错误信息。
*   **同步:**  在多线程环境中，需要考虑对 `rx_buffer`  的访问进行同步，以避免竞争条件。

这个改进后的代码应该能够提供更可靠和高效的USART通信功能。  记住，根据你的具体应用场景，你可能需要进行一些调整。
