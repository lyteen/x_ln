Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\usart.c`

好的，没问题，让我们重新开始，这次我会更注重代码的解释和演示。

**这段代码是关于STM32单片机中使用USART1（通用同步异步收发器1）进行串口通信的配置。主要包括以下几个部分：**

1.  **头文件包含和用户代码区：**

```c
/* Includes ------------------------------------------------------------------*/
#include "usart.h"

/* USER CODE BEGIN 0 */
#include "common_inc.h"
#include "Platform/retarget.h"

volatile uint8_t rxLen = 0;
uint8_t rx_buffer[BUFFER_SIZE] = {0};
void (* OnRecvEnd)(uint8_t* data, uint16_t len);
/* USER CODE END 0 */
```

*   `#include "usart.h"`：包含USART相关的头文件，定义了USART的配置参数和函数。
*   `/* USER CODE BEGIN 0 */` 和 `/* USER CODE END 0 */`：用户代码区，用于添加自定义的代码。
*   `#include "common_inc.h"`：包含常用的头文件，例如数据类型定义等。
*   `#include "Platform/retarget.h"`：包含重定向printf函数的头文件，可以将printf函数输出到串口。
*   `volatile uint8_t rxLen = 0;`：声明一个volatile类型的uint8\_t变量rxLen，用于记录接收到的数据长度。volatile关键字表示该变量的值可能被外部因素修改，例如中断。
*   `uint8_t rx_buffer[BUFFER_SIZE] = {0};`：声明一个uint8\_t类型的数组rx\_buffer，作为接收缓冲区，用于存储接收到的数据。BUFFER\_SIZE定义了缓冲区的大小。
*   `void (* OnRecvEnd)(uint8_t* data, uint16_t len);`：声明一个函数指针OnRecvEnd，指向一个函数，该函数接受一个uint8\_t类型的指针data和一个uint16\_t类型的len作为参数，没有返回值。这个函数指针用于在接收完成时调用用户自定义的回调函数。

**简单Demo：**

假设 `common_inc.h` 中定义了 `BUFFER_SIZE` 为 256。  `Platform/retarget.h` 实现了 `RetargetInit` 函数，用于将 `printf` 重定向到串口。
`OnUartCmd` 是一个用户自定义的函数，用于处理接收到的串口数据。

2.  **USART1句柄和DMA句柄：**

```c
UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;
DMA_HandleTypeDef hdma_usart1_tx;
```

*   `UART_HandleTypeDef huart1;`：声明一个UART\_HandleTypeDef类型的变量huart1，作为USART1的句柄。句柄是一个指向结构体的指针，结构体中包含了USART1的配置参数和状态信息。
*   `DMA_HandleTypeDef hdma_usart1_rx;`：声明一个DMA\_HandleTypeDef类型的变量hdma\_usart1\_rx，作为USART1接收DMA的句柄。
*   `DMA_HandleTypeDef hdma_usart1_tx;`：声明一个DMA\_HandleTypeDef类型的变量hdma\_usart1\_tx，作为USART1发送DMA的句柄。
    DMA（Direct Memory Access）允许外设（例如USART）直接访问存储器，而无需CPU的干预，从而提高数据传输效率。

3.  **USART1初始化函数：**

```c
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
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  __HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE);
  RetargetInit(&huart1);
  Uart_SetRxCpltCallBack(OnUartCmd);
  HAL_UART_Receive_DMA(&huart1, rx_buffer, BUFFER_SIZE);
}
```

*   `huart1.Instance = USART1;`：指定USART1的实例。
*   `huart1.Init.BaudRate = 115200;`：设置波特率为115200。波特率是串口通信的速率，表示每秒传输的比特数。
*   `huart1.Init.WordLength = UART_WORDLENGTH_8B;`：设置字长为8位。字长是每次传输的数据的位数。
*   `huart1.Init.StopBits = UART_STOPBITS_1;`：设置停止位为1位。停止位用于标识数据传输的结束。
*   `huart1.Init.Parity = UART_PARITY_NONE;`：设置校验位为无校验。校验位用于检测数据传输过程中是否发生错误。
*   `huart1.Init.Mode = UART_MODE_TX_RX;`：设置模式为发送和接收模式。
*   `huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;`：设置硬件流控制为无流控制。硬件流控制用于防止数据溢出。
*   `huart1.Init.OverSampling = UART_OVERSAMPLING_16;`：设置过采样为16倍。过采样可以提高数据接收的可靠性。
*   `if (HAL_UART_Init(&huart1) != HAL_OK)`：调用HAL\_UART\_Init函数初始化USART1。HAL\_UART\_Init函数是HAL库提供的函数，用于初始化USART。如果初始化失败，则调用Error\_Handler函数处理错误。
*   `__HAL_UART_ENABLE_IT(&huart1, UART_IT_IDLE);`：使能USART1的空闲中断。空闲中断是指在接收数据期间，总线空闲一段时间后触发的中断。可以用来判断一帧数据是否接收完成。
*   `RetargetInit(&huart1);`：调用RetargetInit函数，将printf函数重定向到USART1。
*   `Uart_SetRxCpltCallBack(OnUartCmd);`：调用Uart\_SetRxCpltCallBack函数，设置接收完成回调函数为OnUartCmd。
*   `HAL_UART_Receive_DMA(&huart1, rx_buffer, BUFFER_SIZE);`：调用HAL\_UART\_Receive\_DMA函数，启动DMA接收。HAL\_UART\_Receive\_DMA函数是HAL库提供的函数，用于使用DMA接收数据。

**简单Demo：**

该函数配置USART1为115200波特率，8位数据位，1位停止位，无校验，使能发送和接收，无硬件流控制，16倍过采样。  然后初始化USART1，使能空闲中断，将printf重定向到串口，设置接收完成回调函数，启动DMA接收。

4.  **USART1 MSP初始化函数：**

```c
void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(uartHandle->Instance==USART1)
  {
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    __HAL_AFIO_REMAP_USART1_ENABLE();

    hdma_usart1_rx.Instance = DMA1_Channel5;
    hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart1_rx.Init.Mode = DMA_NORMAL;
    hdma_usart1_rx.Init.Priority = DMA_PRIORITY_MEDIUM;
    if (HAL_DMA_Init(&hdma_usart1_rx) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(uartHandle,hdmarx,hdma_usart1_rx);

    hdma_usart1_tx.Instance = DMA1_Channel4;
    hdma_usart1_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_usart1_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart1_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart1_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart1_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart1_tx.Init.Mode = DMA_NORMAL;
    hdma_usart1_tx.Init.Priority = DMA_PRIORITY_MEDIUM;
    if (HAL_DMA_Init(&hdma_usart1_tx) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(uartHandle,hdmatx,hdma_usart1_tx);

    HAL_NVIC_SetPriority(USART1_IRQn, 3, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);
  }
}
```

*   `HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)`：是HAL库提供的MSP（MCU Support Package）初始化函数，用于初始化USART的底层硬件资源，例如时钟、GPIO和DMA。
*   `__HAL_RCC_USART1_CLK_ENABLE();`：使能USART1的时钟。
*   `__HAL_RCC_GPIOB_CLK_ENABLE();`：使能GPIOB的时钟。USART1的TX和RX引脚通常连接到GPIOB。
*   `GPIO_InitStruct.Pin = GPIO_PIN_6;`：设置TX引脚为PB6。
*   `GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;`：设置TX引脚的模式为复用推挽输出。复用是指该引脚可以作为USART1的TX引脚使用。推挽输出是指该引脚可以输出高电平和低电平。
*   `GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;`：设置TX引脚的速率为高速。
*   `HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);`：初始化TX引脚。
*   `GPIO_InitStruct.Pin = GPIO_PIN_7;`：设置RX引脚为PB7。
*   `GPIO_InitStruct.Mode = GPIO_MODE_INPUT;`：设置RX引脚的模式为输入。
*   `GPIO_InitStruct.Pull = GPIO_NOPULL;`：设置RX引脚为无上拉/下拉。
*   `HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);`：初始化RX引脚。
*   `__HAL_AFIO_REMAP_USART1_ENABLE();`：使能USART1的引脚重映射。有些STM32芯片的USART1引脚不是默认的PB6和PB7，需要通过引脚重映射将USART1的引脚映射到PB6和PB7。
*   `hdma_usart1_rx.Instance = DMA1_Channel5;`：设置USART1接收DMA的实例为DMA1的通道5。
*   `hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;`：设置DMA的传输方向为外设到存储器，即从USART1接收数据到存储器。
*   `hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;`：设置外设地址不递增，即USART1的地址保持不变。
*   `hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;`：设置存储器地址递增，即数据存储到存储器后，存储器地址自动递增。
*   `hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;`：设置外设数据对齐方式为字节对齐。
*   `hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;`：设置存储器数据对齐方式为字节对齐。
*   `hdma_usart1_rx.Init.Mode = DMA_NORMAL;`：设置DMA的模式为普通模式，即传输完成后停止DMA。
*   `hdma_usart1_rx.Init.Priority = DMA_PRIORITY_MEDIUM;`：设置DMA的优先级为中等。
*   `if (HAL_DMA_Init(&hdma_usart1_rx) != HAL_OK)`：初始化DMA。
*   `__HAL_LINKDMA(uartHandle,hdmarx,hdma_usart1_rx);`：将DMA句柄链接到USART句柄，方便USART使用DMA。
*   `hdma_usart1_tx.Instance = DMA1_Channel4;`：设置USART1发送DMA的实例为DMA1的通道4。
*   `...`：USART1发送DMA的初始化过程与接收DMA类似，只是传输方向为存储器到外设。
*   `HAL_NVIC_SetPriority(USART1_IRQn, 3, 0);`：设置USART1中断的优先级。
*   `HAL_NVIC_EnableIRQ(USART1_IRQn);`：使能USART1中断。

**简单Demo：**

该函数使能USART1的时钟和GPIOB的时钟，配置PB6和PB7为USART1的TX和RX引脚，初始化DMA，将DMA句柄链接到USART句柄，设置USART1中断的优先级，使能USART1中断。

5.  **USART1 MSP反初始化函数：**

```c
void HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle)
{
  if(uartHandle->Instance==USART1)
  {
    __HAL_RCC_USART1_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6|GPIO_PIN_7);
    HAL_DMA_DeInit(uartHandle->hdmarx);
    HAL_DMA_DeInit(uartHandle->hdmatx);
    HAL_NVIC_DisableIRQ(USART1_IRQn);
  }
}
```

*   `HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle)`：是HAL库提供的MSP反初始化函数，用于反初始化USART的底层硬件资源，例如关闭时钟、禁用GPIO和DMA。
*   `__HAL_RCC_USART1_CLK_DISABLE();`：禁用USART1的时钟。
*   `HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6|GPIO_PIN_7);`：禁用PB6和PB7的GPIO。
*   `HAL_DMA_DeInit(uartHandle->hdmarx);`：禁用USART1接收DMA。
*   `HAL_DMA_DeInit(uartHandle->hdmatx);`：禁用USART1发送DMA。
*   `HAL_NVIC_DisableIRQ(USART1_IRQn);`：禁用USART1中断。

6.  **设置接收完成回调函数：**

```c
void Uart_SetRxCpltCallBack(void(* xerc)(uint8_t*, uint16_t))
{
  OnRecvEnd = xerc;
}
```

*   `Uart_SetRxCpltCallBack(void(* xerc)(uint8_t*, uint16_t))`：设置接收完成回调函数。
*   `OnRecvEnd = xerc;`：将用户自定义的回调函数指针赋值给OnRecvEnd。

**简单Demo：**

该函数允许用户设置一个在串口接收完成时调用的函数。 例如，可以这样使用：

```c
void MyUartCallback(uint8_t* data, uint16_t len) {
  // 在这里处理接收到的数据
  printf("Received %d bytes: %s\r\n", len, data);
}

int main() {
  // ...其他初始化代码...
  MX_USART1_UART_Init(); // 初始化串口
  Uart_SetRxCpltCallBack(MyUartCallback); // 设置回调函数
  while (1) {
    // 主循环
  }
}
```

在这个例子中，`MyUartCallback` 函数会在串口接收到数据后被调用。

**整体使用流程：**

1.  在 `main.c` 中调用 `MX_USART1_UART_Init()` 初始化串口。
2.  通过 `Uart_SetRxCpltCallBack()` 设置一个回调函数，用于处理接收到的数据。
3.  当串口接收到数据并且触发空闲中断时，HAL库会自动调用中断服务函数，该函数会将接收到的数据存储到 `rx_buffer` 中，并且调用用户自定义的回调函数 `OnRecvEnd`。
4.  在回调函数中，用户可以处理接收到的数据。

**一个完整的示例：**

`main.c`:

```c
#include "main.h"
#include "usart.h"

void SystemClock_Config(void); // 假设有系统时钟配置函数

void MyUartCallback(uint8_t* data, uint16_t len) {
  // 处理接收到的数据
  printf("Received %d bytes: %s\r\n", len, data);
  HAL_UART_Receive_DMA(&huart1, rx_buffer, BUFFER_SIZE); // 重新启动DMA接收
}

int main(void) {
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init(); // 假设有GPIO初始化函数
  MX_USART1_UART_Init(); // 初始化串口

  Uart_SetRxCpltCallBack(MyUartCallback); // 设置回调函数

  printf("USART1 Initialized. Ready to receive data.\r\n");

  while (1) {
    // 主循环
  }
}
```

在这个例子中，`MyUartCallback` 会在接收到数据后打印接收到的内容，并且重新启动DMA接收，以便接收下一帧数据。`SystemClock_Config()` 和 `MX_GPIO_Init()` 是为了配合STM32 HAL库的初始化而存在的，需要根据你的具体项目进行配置。  务必实现 `USART1_IRQHandler`  USART1的中断函数.
```c
void USART1_IRQHandler(void)
{
  HAL_UART_IRQHandler(&huart1);
}
```

这段代码是STM32 HAL库中USART串口通信配置的关键部分。 通过理解每个部分的功能和使用方法，可以轻松地在STM32单片机中使用USART进行串口通信。 关键是正确配置USART的参数、GPIO和DMA，并且设置好中断和回调函数，以便在接收到数据时及时处理。
