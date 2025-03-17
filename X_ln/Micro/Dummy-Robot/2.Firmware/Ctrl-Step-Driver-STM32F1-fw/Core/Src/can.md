Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\can.c`

好的，重新开始。 针对给出的CAN总线驱动代码，我将对其关键部分进行解释，并提供代码片段和中文描述，包含使用方法和简单示例。

**1. 头文件包含 (Includes):**

```c
/* Includes ------------------------------------------------------------------*/
#include "can.h"
#include "common_inc.h"
#include "configurations.h"
```

**描述:**  这段代码包含了 CAN 驱动需要的头文件。 `can.h` 包含了 CAN 相关的定义， `common_inc.h` 包含通用的定义，`configurations.h` 包含板子的配置信息，比如CAN节点的ID。

**使用方法:**  在使用 CAN 驱动之前，需要包含这些头文件。

**2. 全局变量定义 (Global Variables):**

```c
CAN_TxHeaderTypeDef TxHeader;
CAN_RxHeaderTypeDef RxHeader;
uint8_t TxData[8];
uint8_t RxData[8];
uint32_t TxMailbox;
```

**描述:**  这些变量用于存储 CAN 消息的头部信息、数据和发送邮箱。 `TxHeader` 和 `RxHeader` 分别用于配置发送和接收消息的头部，`TxData` 和 `RxData` 是发送和接收的数据缓冲区，`TxMailbox` 用于存储发送邮箱的信息。

**使用方法:**  在发送 CAN 消息之前，需要配置 `TxHeader` 和 `TxData`，然后在发送函数中使用它们。接收 CAN 消息时，消息内容将被存放在 `RxHeader` 和 `RxData` 中。

**3. CAN 初始化函数 (MX_CAN_Init):**

```c
void MX_CAN_Init(void)
{
  hcan.Instance = CAN1;
  hcan.Init.Prescaler = 4;
  hcan.Init.Mode = CAN_MODE_NORMAL;
  hcan.Init.SyncJumpWidth = CAN_SJW_1TQ;
  hcan.Init.TimeSeg1 = CAN_BS1_5TQ;
  hcan.Init.TimeSeg2 = CAN_BS2_3TQ;
  hcan.Init.TimeTriggeredMode = DISABLE;
  hcan.Init.AutoBusOff = DISABLE;
  hcan.Init.AutoWakeUp = ENABLE;
  hcan.Init.AutoRetransmission = DISABLE;
  hcan.Init.ReceiveFifoLocked = DISABLE;
  hcan.Init.TransmitFifoPriority = DISABLE;
  if (HAL_CAN_Init(&hcan) != HAL_OK)
  {
    Error_Handler();
  }

  CAN_FilterTypeDef sFilterConfig;
  sFilterConfig.FilterBank = 0;
  sFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
  sFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
  sFilterConfig.FilterIdHigh = 0x0000;
  sFilterConfig.FilterIdLow = 0x0000;
  sFilterConfig.FilterMaskIdHigh = 0x0000;
  sFilterConfig.FilterMaskIdLow = 0x0000;
  sFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0;
  sFilterConfig.FilterActivation = ENABLE;
  sFilterConfig.SlaveStartFilterBank = 14;
  if (HAL_CAN_ConfigFilter(&hcan, &sFilterConfig) != HAL_OK)
  {
    Error_Handler();
  }

  HAL_CAN_Start(&hcan);

  HAL_CAN_ActivateNotification(&hcan,
                               CAN_IT_TX_MAILBOX_EMPTY |
                               CAN_IT_RX_FIFO0_MSG_PENDING | CAN_IT_RX_FIFO1_MSG_PENDING |
                               CAN_IT_RX_FIFO0_FULL | CAN_IT_RX_FIFO1_FULL |
                               CAN_IT_RX_FIFO0_OVERRUN | CAN_IT_RX_FIFO1_OVERRUN |
                               CAN_IT_WAKEUP | CAN_IT_SLEEP_ACK |
                               CAN_IT_ERROR_WARNING | CAN_IT_ERROR_PASSIVE |
                               CAN_IT_BUSOFF | CAN_IT_LAST_ERROR_CODE |
                               CAN_IT_ERROR);

  TxHeader.StdId = boardConfig.canNodeId;
  TxHeader.ExtId = 0x00;
  TxHeader.RTR = CAN_RTR_DATA;
  TxHeader.IDE = CAN_ID_STD;
  TxHeader.DLC = 8;
  TxHeader.TransmitGlobalTime = DISABLE;
}
```

**描述:**  这个函数初始化 CAN1 外设。 它配置了波特率预分频器、工作模式（正常模式）、同步跳转宽度、时间段 1 和 2、过滤器以及中断。`HAL_CAN_Init` 初始化CAN外设, `HAL_CAN_ConfigFilter` 配置过滤器以接收特定的消息， `HAL_CAN_Start` 启动CAN， `HAL_CAN_ActivateNotification`  使能各种中断， 最后的几行代码初始化了 `TxHeader`, 设置了 CAN ID, 数据长度等等。

**使用方法:**  在程序开始时调用此函数以配置 CAN 外设。 例如，`hcan.Init.Prescaler = 4` 设置了波特率预分频器，影响了 CAN 总线的通信速度。  `sFilterConfig` 的设置则允许接收所有ID的消息.

**4. CAN GPIO 初始化函数 (HAL_CAN_MspInit):**

```c
void HAL_CAN_MspInit(CAN_HandleTypeDef* canHandle)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(canHandle->Instance==CAN1)
  {
    __HAL_RCC_CAN1_CLK_ENABLE();

    __HAL_RCC_GPIOB_CLK_ENABLE();

    GPIO_InitStruct.Pin = GPIO_PIN_8;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    __HAL_AFIO_REMAP_CAN1_2();

    HAL_NVIC_SetPriority(USB_HP_CAN1_TX_IRQn, 3, 0);
    HAL_NVIC_EnableIRQ(USB_HP_CAN1_TX_IRQn);
    HAL_NVIC_SetPriority(USB_LP_CAN1_RX0_IRQn, 3, 0);
    HAL_NVIC_EnableIRQ(USB_LP_CAN1_RX0_IRQn);
    HAL_NVIC_SetPriority(CAN1_RX1_IRQn, 3, 0);
    HAL_NVIC_EnableIRQ(CAN1_RX1_IRQn);
    HAL_NVIC_SetPriority(CAN1_SCE_IRQn, 3, 0);
    HAL_NVIC_EnableIRQ(CAN1_SCE_IRQn);
  }
}
```

**描述:**  此函数初始化 CAN1 的 GPIO 引脚 (PB8 和 PB9) 用于接收 (RX) 和发送 (TX) 数据。 它还使能了 CAN1 时钟并配置了中断。  `__HAL_RCC_CAN1_CLK_ENABLE()` 使能 CAN1 的时钟，`HAL_GPIO_Init()`  配置 GPIO 引脚， `__HAL_AFIO_REMAP_CAN1_2()` 用于引脚重映射(如果需要)， `HAL_NVIC_EnableIRQ` 使能中断。

**使用方法:**  这个函数由 HAL 库在 `HAL_CAN_Init()` 内部调用，用于初始化 CAN 通信所需的底层硬件资源。  确保使用的 GPIO 引脚与硬件连接匹配。

**5. CAN 发送函数 (CAN_Send):**

```c
void CAN_Send(CAN_TxHeaderTypeDef* pHeader, uint8_t* data)
{
    if (HAL_CAN_AddTxMessage(&hcan, pHeader, data, &TxMailbox) != HAL_OK)
    {
        Error_Handler();
    }
}
```

**描述:**  此函数发送 CAN 消息。 它使用 `HAL_CAN_AddTxMessage` 函数将消息添加到发送邮箱中。 如果发送失败，则调用 `Error_Handler` 函数。

**使用方法:**  首先，填充 `TxHeader` 和 `TxData` 变量，然后调用此函数以发送消息。
例如：

```c
TxHeader.StdId = 0x123; // 设置 CAN ID
TxData[0] = 0x01;
TxData[1] = 0x02;
TxData[2] = 0x03;
TxData[3] = 0x04;
TxHeader.DLC = 4; // 设置数据长度
CAN_Send(&TxHeader, TxData); // 发送消息
```

**6. CAN 接收中断回调函数 (HAL_CAN_RxFifo0MsgPendingCallback):**

```c
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef* CanHandle)
{
    if (HAL_CAN_GetRxMessage(CanHandle, CAN_RX_FIFO0, &RxHeader, RxData) != HAL_OK)
    {
        Error_Handler();
    }

    uint8_t id = (RxHeader.StdId >> 7); // 4Bits ID & 7Bits Msg
    uint8_t cmd = RxHeader.StdId & 0x7F; // 4Bits ID & 7Bits Msg
    if (id == 0 || id == boardConfig.canNodeId)
    {
        OnCanCmd(cmd, RxData, RxHeader.DLC);
    }
}
```

**描述:**  当接收到 CAN 消息时，HAL 库会调用此回调函数。 该函数从接收 FIFO 中获取消息，并将其存储在 `RxHeader` 和 `RxData` 变量中。 然后，它解析 CAN ID, 并调用 `OnCanCmd` 函数来处理接收到的命令。

**使用方法:**  此函数是中断服务程序的一部分，无需手动调用。 在 `MX_CAN_Init` 函数中使能了接收中断后，当接收到 CAN 消息时，此函数会自动执行。
代码中将标准ID分解为高4位ID和低7位CMD, 然后根据ID判断是否是发给自己的消息, 如果是就调用`OnCanCmd`函数处理. `OnCanCmd` 函数通常需要用户自己实现，用于执行特定的操作，比如控制电机，读取传感器数据等。

**简单 Demo:**

假设 `OnCanCmd` 函数的实现如下：

```c
void OnCanCmd(uint8_t cmd, uint8_t* data, uint8_t len) {
  // 根据 cmd 的值执行不同的操作
  switch (cmd) {
    case 0x01: // 控制 LED
      // ...
      break;
    case 0x02: // 读取传感器数据
      // ...
      break;
    default:
      // 未知命令
      break;
  }
}
```

在该示例中，如果接收到的 `cmd` 值为 `0x01`，则执行控制 LED 的操作；如果 `cmd` 值为 `0x02`，则执行读取传感器数据的操作。

总结来说，这段 CAN 驱动代码提供了一个在 STM32 微控制器上使用 CAN 总线进行通信的基本框架。 通过配置初始化函数，设置过滤器，以及实现发送和接收函数，可以方便地实现 CAN 通信功能。 开发者需要根据具体的应用场景，实现相应的消息处理逻辑，比如 `OnCanCmd` 函数。
