Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Inc\usart.h`

好的，我们重新开始，这次我将专注于提供更清晰、更安全、更易于理解的串口（USART）头文件，并且会提供更详细的中文注释和示例代码。

**目标:** 设计一个更健壮、可扩展的 `usart.h` 头文件，方便在STM32项目中进行串口通信。

**1. 更清晰的宏定义和类型定义:**

```c
#ifndef __USART_H__
#define __USART_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"  // 包含 STM32 HAL 库的头文件

// ***  配置参数  ***
#define USART_BUFFER_SIZE 256  // 增加缓冲区大小，提升数据处理能力。串口接收缓冲区大小

// ***  状态标志  ***
typedef enum {
    USART_IDLE = 0,       // 空闲状态
    USART_RECEIVING,      // 正在接收数据
    USART_TRANSMITTING,   // 正在发送数据
    USART_ERROR           // 发生错误
} USART_StateTypeDef;

// ***  错误代码  ***
typedef enum {
    USART_OK = 0,                 // 成功
    USART_ERROR_OVERRUN,          // 溢出错误
    USART_ERROR_FRAMING,          // 帧错误
    USART_ERROR_PARITY,           // 奇偶校验错误
    USART_ERROR_DMA               // DMA传输错误
} USART_ErrorCodeTypeDef;

// ***  接收完成回调函数类型  ***
typedef void (*USART_RxCpltCallback)(uint8_t *data, uint16_t len);


/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

extern UART_HandleTypeDef huart1;

/* USER CODE BEGIN Private defines */
// #define BUFFER_SIZE  128  //  已经被 USART_BUFFER_SIZE 替代

extern DMA_HandleTypeDef hdma_usart1_rx;
extern DMA_HandleTypeDef hdma_usart1_tx;
extern volatile uint8_t rxLen;  // 用处待定，可以考虑去除，或者明确用途
extern volatile uint8_t recv_end_flag; // 用处待定，可以用状态机代替

// extern uint8_t rx_buffer[BUFFER_SIZE]; // 已经被 USART_BUFFER_SIZE 替代，并且移到 .c 文件中


/* USER CODE END Private defines */

void MX_USART1_UART_Init(void);

/* USER CODE BEGIN Prototypes */
// extern void (*OnRecvEnd)(uint8_t *data, uint16_t len); // 已被 USART_RxCpltCallback 替代
void Uart_SetRxCpltCallBack(USART_RxCpltCallback xerc);  // 更清晰的回调函数设置

/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __USART_H__ */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
```

**改进说明:**

*   **使用枚举类型 (`USART_StateTypeDef`, `USART_ErrorCodeTypeDef`)**:  代替简单的宏定义，提高了代码的可读性和类型安全性。  例如, `USART_IDLE`, `USART_RECEIVING` 更易于理解。
*   **typedef 回调函数**: 使用 `typedef` 定义回调函数类型 `USART_RxCpltCallback`，使代码更清晰、更易于维护。
*   **统一命名规范**:  变量和宏定义使用一致的命名规范（例如： `USART_BUFFER_SIZE`）。
*   **更详细的注释**:  添加了更多的中文注释，解释了每个宏定义和函数的作用。
*   **BUFFER_SIZE移除**: rx_buffer的大小不应该在头文件中定义，应该在.c中定义，这是为了避免在多个文件中引用时造成冲突。

**2. 对应的 `usart.c` 文件 (示例):**

```c
#include "usart.h"

// 接收缓冲区，定义在 .c 文件中，避免头文件暴露大小信息
uint8_t rx_buffer[USART_BUFFER_SIZE];

// 接收完成回调函数指针
static USART_RxCpltCallback RxCpltCallback = NULL; // 静态变量，只能在本文件访问

//设置回调函数
void Uart_SetRxCpltCallBack(USART_RxCpltCallback xerc) {
    RxCpltCallback = xerc;
}

// 假设在 DMA 中断处理函数中调用
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance == USART1) { // 确保是 USART1 的中断
    if (RxCpltCallback != NULL) {
      RxCpltCallback(rx_buffer, USART_BUFFER_SIZE -  __HAL_DMA_GET_COUNTER(huart->hdmarx)); // 传递数据和长度
    }

    // 重新启动 DMA 接收 (如果需要连续接收)
    HAL_UART_Receive_DMA(&huart1, rx_buffer, USART_BUFFER_SIZE);
  }
}


// 可能的串口接收处理函数示例(需要在 main.c 或者其他文件中初始化)
void MyUsart_ProcessData(uint8_t *data, uint16_t len) {
  // 在这里处理接收到的数据，例如：
  // - 解析命令
  // - 更新状态变量
  // - 发送响应
  for (int i = 0; i < len; i++){
    printf("%c",data[i]); //将接收到的数据打印到控制台
  }

  printf("\r\n");
}
```

**说明:**

*   **`rx_buffer` 定义在 `.c` 文件中**:  避免在头文件中定义数组大小，防止重复定义和减小编译依赖。
*   **静态回调函数指针**:  `RxCpltCallback` 是一个 `static` 变量，只能在 `usart.c` 中访问，实现了更好的封装性。
*   **HAL 库回调函数**:  使用了 STM32 HAL 库提供的中断回调函数 `HAL_UART_RxCpltCallback`，简化了中断处理。
*   **重新启动 DMA 接收**:  在回调函数中重新启动 DMA 接收，实现连续接收数据。
*   **示例数据处理函数**:  `MyUsart_ProcessData` 函数是一个示例，展示了如何处理接收到的数据。

**3. 使用示例 (在 `main.c` 中):**

```c
#include "main.h"
#include "usart.h"
#include <stdio.h>  // 使用 printf 函数

extern UART_HandleTypeDef huart1; // 声明外部变量 huart1

int main(void)
{
  HAL_Init();

  SystemClock_Config();

  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();

  // 设置接收完成回调函数
  Uart_SetRxCpltCallBack(MyUsart_ProcessData);

  // 启动 DMA 接收
  HAL_UART_Receive_DMA(&huart1, rx_buffer, USART_BUFFER_SIZE);

  while (1)
  {
    // 主循环，可以执行其他任务
    HAL_Delay(100);
  }
}

// 重定向 printf 函数到 USART1 (可选)
int fputc(int ch, FILE *f)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
  return ch;
}
```

**使用步骤和说明 (中文):**

1.  **包含头文件**:  在 `main.c` 中包含 `usart.h` 头文件。
2.  **初始化 USART1**:  调用 `MX_USART1_UART_Init()` 函数初始化 USART1 (由 STM32CubeIDE 生成)。
3.  **设置回调函数**:  使用 `Uart_SetRxCpltCallBack()` 函数设置接收完成后的回调函数 (例如: `MyUsart_ProcessData`)。  这个回调函数会在每次接收到数据后被调用。
4.  **启动 DMA 接收**:  使用 `HAL_UART_Receive_DMA()` 函数启动 DMA 接收，将接收到的数据存储到 `rx_buffer` 中。
5.  **实现回调函数**:  实现你自己的数据处理函数 (`MyUsart_ProcessData`)，在函数中处理接收到的数据。
6.  **重定向 `printf` 函数 (可选)**:  如果你想使用 `printf` 函数通过串口输出调试信息，需要重定向 `printf` 函数。

**关键点总结:**

*   **代码清晰**: 使用枚举类型和 `typedef` 提高了代码的可读性。
*   **类型安全**: 使用枚举类型和函数指针增加了类型安全性。
*   **封装性**: `rx_buffer` 和 `RxCpltCallback` 定义在 `.c` 文件中，实现了更好的封装性。
*   **易于使用**:  通过 `Uart_SetRxCpltCallBack()` 函数设置回调函数，简化了串口通信的配置。
*   **可扩展**:  可以很容易地添加新的状态和错误代码，以满足不同的需求。

这个改进后的 `usart.h` 头文件和相关的示例代码，提供了一个更健壮、可扩展和易于使用的串口通信解决方案。 希望对你有帮助！
