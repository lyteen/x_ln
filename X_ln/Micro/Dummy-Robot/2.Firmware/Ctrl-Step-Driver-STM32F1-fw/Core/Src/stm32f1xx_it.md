Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\stm32f1xx_it.c`

好的，我们重新开始分析这段 STM32 中断服务例程代码。我会对关键部分进行解释，并提供代码片段和中文描述，以及代码用途的简单说明和示例。

**1. 头文件包含 (Includes)**

```c
#include "main.h"
#include "stm32f1xx_it.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <memory.h>
#include "common_inc.h"
#include "usart.h"
/* USER CODE END Includes */
```

**描述:**  这段代码包含了必要的头文件。`main.h` 和 `stm32f1xx_it.h` 是 STM32 HAL 库自动生成的文件，定义了硬件抽象层和中断向量表。`<memory.h>` 提供了内存操作函数（如 `memset`）， `common_inc.h` 和 `usart.h` 是用户自定义的头文件，很可能包含一些常用的定义和函数声明，特别是串口相关的。

**用途:** 包含了程序依赖的头文件，方便后续调用头文件中声明的函数和变量。

**2. 外部变量声明 (External Variables)**

```c
extern DMA_HandleTypeDef hdma_adc1;
extern CAN_HandleTypeDef hcan;
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim3;
extern TIM_HandleTypeDef htim4;
extern DMA_HandleTypeDef hdma_usart1_rx;
extern DMA_HandleTypeDef hdma_usart1_tx;
extern UART_HandleTypeDef huart1;
```

**描述:**  `extern` 关键字声明了在其他文件中定义的全局变量。这里声明了 ADC DMA句柄、CAN 句柄、定时器句柄和串口 DMA 句柄。 这些句柄是在其他地方（通常是 `main.c` 或其他初始化文件中）创建和配置的。

**用途:**  中断处理程序需要访问这些句柄来操作外设和管理 DMA 传输。

**3.  NMI_Handler, HardFault_Handler, MemManage_Handler, BusFault_Handler, UsageFault_Handler**

```c
void NMI_Handler(void)
{
  /* USER CODE BEGIN NonMaskableInt_IRQn 0 */

  /* USER CODE END NonMaskableInt_IRQn 0 */
  /* USER CODE BEGIN NonMaskableInt_IRQn 1 */
    while (1)
    {
    }
  /* USER CODE END NonMaskableInt_IRQn 1 */
}

void HardFault_Handler(void)
{
  while (1)
  {
  }
}
// ... 其他 Fault Handler类似
```

**描述:**  这些是 Cortex-M3 内核的异常处理函数。`NMI_Handler` 处理不可屏蔽中断，其他 `...Fault_Handler`  处理各种错误情况，如硬件错误，内存管理错误，总线错误等。  `while(1)`  表示进入无限循环，通常在发生严重错误时使用，防止程序继续运行。

**用途:**  用于捕获和处理系统错误，防止程序崩溃。 在调试阶段，可以在这些处理函数中添加调试代码来定位错误原因。

**4. SysTick_Handler**

```c
void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
  /* USER CODE BEGIN SysTick_IRQn 1 */
  /* USER CODE END SysTick_IRQn 1 */
}
```

**描述:**  `SysTick_Handler` 是系统滴答定时器的中断处理函数。 `HAL_IncTick()`  是 HAL 库提供的函数，用于更新系统时钟计数器 `uwTick`，该计数器用于 HAL 库的延时函数（`HAL_Delay()`）。

**用途:**  提供时间基准，供操作系统或应用程序使用。

**5. DMA1_Channel1_IRQHandler (ADC DMA中断)**

```c
void DMA1_Channel1_IRQHandler(void)
{
  /* USER CODE BEGIN DMA1_Channel1_IRQn 0 */

  /* USER CODE END DMA1_Channel1_IRQn 0 */
  HAL_DMA_IRQHandler(&hdma_adc1);
  /* USER CODE BEGIN DMA1_Channel1_IRQn 1 */

  /* USER CODE END DMA1_Channel1_IRQn 1 */
}
```

**描述:**  这是 DMA1 通道 1 的中断处理函数，通常与 ADC 转换完成相关联。 `HAL_DMA_IRQHandler(&hdma_adc1)`  调用 HAL 库提供的 DMA 中断处理函数，该函数会清除中断标志，并执行与 DMA 完成相关的回调函数（如果在 DMA 初始化时设置了回调函数）。

**用途:**  处理 ADC 转换完成后的 DMA 传输，例如将 ADC 数据从 DMA 缓冲区搬运到应用程序的缓冲区。

**6. DMA1_Channel4_IRQHandler & DMA1_Channel5_IRQHandler (USART1 DMA中断)**

```c
void DMA1_Channel4_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&hdma_usart1_tx);
}

void DMA1_Channel5_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&hdma_usart1_rx);
}
```

**描述:**  分别对应USART1的发送和接收DMA中断。当DMA传输完成或发生错误时，会触发这些中断。`HAL_DMA_IRQHandler`负责清除中断标志并调用相应的回调函数。

**用途:**  用于处理USART1的DMA发送和接收。可以实现非阻塞的串口通信，提高数据传输效率。

**7. USB_HP_CAN1_TX_IRQHandler, USB_LP_CAN1_RX0_IRQHandler, CAN1_RX1_IRQHandler, CAN1_SCE_IRQHandler (CAN 中断)**

```c
void USB_HP_CAN1_TX_IRQHandler(void)
{
  HAL_CAN_IRQHandler(&hcan);
}

void USB_LP_CAN1_RX0_IRQHandler(void)
{
  HAL_CAN_IRQHandler(&hcan);
}

void CAN1_RX1_IRQHandler(void)
{
  HAL_CAN_IRQHandler(&hcan);
}

void CAN1_SCE_IRQHandler(void)
{
  HAL_CAN_IRQHandler(&hcan);
}
```

**描述:**  这些是 CAN1 外设的中断处理函数，分别对应 CAN 的发送、接收和状态错误中断。 `HAL_CAN_IRQHandler(&hcan)`  调用 HAL 库提供的 CAN 中断处理函数。

**用途:**  处理 CAN 总线的发送和接收事件，以及 CAN 控制器的错误状态。

**8. EXTI9_5_IRQHandler (外部中断)**

```c
void EXTI9_5_IRQHandler(void)
{
  /* USER CODE BEGIN EXTI9_5_IRQn 0 */
    return;
  /* USER CODE END EXTI9_5_IRQn 0 */
  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_7);
  /* USER CODE BEGIN EXTI9_5_IRQn 1 */

  /* USER CODE END EXTI9_5_IRQn 1 */
}
```

**描述:**  这是外部中断线 9-5 的中断处理函数。 `HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_7)`  调用 HAL 库提供的 GPIO 外部中断处理函数，用于处理 GPIO 引脚 7 上的中断事件。  `GpioPin7InterruptCallback()` 会在`main.c`或其他文件中定义，处理具体的GPIO中断事件。 注意到代码中有`return;`语句，这意味着该中断服务例程实际功能由`HAL_GPIO_EXTI_IRQHandler`和`GpioPin7InterruptCallback()`共同完成。

**用途:**  响应外部事件，例如按键按下，传感器信号变化等。

**9. TIM1_UP_IRQHandler, TIM3_IRQHandler, TIM4_IRQHandler (定时器中断)**

```c
void TIM1_UP_IRQHandler(void)
{
    Tim1Callback100Hz();
    return;
  HAL_TIM_IRQHandler(&htim1);
}

void TIM3_IRQHandler(void)
{
    return;
  HAL_TIM_IRQHandler(&htim3);
}

void TIM4_IRQHandler(void)
{
    Tim4Callback20kHz();
    return;
  HAL_TIM_IRQHandler(&htim4);
}
```

**描述:**  这些是定时器 1、3、4 的中断处理函数。分别对应不同的中断事件 (例如，更新中断)。  `HAL_TIM_IRQHandler(&htimx)`  调用 HAL 库提供的定时器中断处理函数。`Tim1Callback100Hz()`, `Tim3CaptureCallback()`和 `Tim4Callback20kHz()`是在其他地方定义的回调函数，处理具体的定时器事件。 注意到代码中有`return;`语句，这意味着这些中断服务例程实际功能由`HAL_TIM_IRQHandler`和回调函数共同完成。

**用途:**  用于定时执行任务，例如周期性采样数据，控制 PWM 输出等。

**10. USART1_IRQHandler (串口中断)**

```c
void USART1_IRQHandler(void)
{
    if ((__HAL_UART_GET_FLAG(&huart1, UART_FLAG_IDLE) != RESET))
    {
        __HAL_UART_CLEAR_IDLEFLAG(&huart1);
        HAL_UART_DMAStop(&huart1);
        uint32_t temp = __HAL_DMA_GET_COUNTER(&hdma_usart1_rx);
        rxLen = BUFFER_SIZE - temp;

        OnRecvEnd(rx_buffer, rxLen);

        memset(rx_buffer, 0, rxLen);
        rxLen = 0;

        HAL_UART_Receive_DMA(&huart1, rx_buffer, BUFFER_SIZE);
    }
  HAL_UART_IRQHandler(&huart1);
}
```

**描述:**  这是 USART1 串口的中断处理函数。这段代码检查是否发生了空闲线路检测中断（IDLE）。 如果是，则停止 DMA 接收，计算接收到的数据长度，调用 `OnRecvEnd()` 函数处理接收到的数据，然后重新启动 DMA 接收。 `HAL_UART_IRQHandler(&huart1)`  调用 HAL 库提供的 UART 中断处理函数。

**用途:**  实现串口数据的接收，特别是在使用 DMA 的情况下，可以实现非阻塞的串口接收。

**代码示例 (USART1_IRQHandler 使用说明):**

假设你希望通过串口接收数据，并且使用 DMA 来提高接收效率。 首先，你需要配置 USART1 和 DMA。在 `main.c` 中：

```c
// main.c

#define BUFFER_SIZE 256
uint8_t rx_buffer[BUFFER_SIZE];
uint16_t rxLen = 0;

void OnRecvEnd(uint8_t *data, uint16_t len) {
  // 在这里处理接收到的数据
  printf("Received %d bytes: %s\r\n", len, data);
}

int main(void) {
  // ... 初始化代码 ...

  // 启动串口 DMA 接收
  HAL_UART_Receive_DMA(&huart1, rx_buffer, BUFFER_SIZE);

  while (1) {
    // 主循环
  }
}
```

在这个例子中，当串口接收到数据并且空闲一段时间后，会触发空闲线路检测中断，进入 `USART1_IRQHandler`，`OnRecvEnd`函数会被调用，你可以在`OnRecvEnd`函数中处理接收到的数据。 通过 DMA，串口接收数据时 CPU 不需要参与，提高了效率。
