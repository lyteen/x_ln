Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_can.h`

好，重新来过，针对 STM32F1xx HAL CAN 驱动头文件（`stm32f1xx_hal_can.h`）进行改进，并提供中文解释和示例。

**总体目标：**

*   **更清晰的类型定义：** 使用更具描述性的类型名，并使用 `typedef` 简化结构体定义。
*   **更安全的宏：** 改进宏定义，防止潜在的错误，并提高代码可读性。
*   **更完善的错误处理：** 扩展错误码，提供更详细的错误信息。
*   **更灵活的回调：** 增加回调函数的参数，提供更多的上下文信息。
*   **提供示例：** 针对关键功能提供示例代码，方便用户快速上手。

**1. 改进的类型定义：**

```c
typedef enum
{
  CAN_STATE_RESET             = 0x00U,  /*!< CAN 未初始化或禁用 */
  CAN_STATE_READY             = 0x01U,  /*!< CAN 已初始化，可以使用 */
  CAN_STATE_LISTENING         = 0x02U,  /*!< CAN 正在接收数据 */
  CAN_STATE_SLEEP_PENDING     = 0x03U,  /*!< CAN 睡眠请求挂起 */
  CAN_STATE_SLEEP_ACTIVE      = 0x04U,  /*!< CAN 睡眠模式激活 */
  CAN_STATE_ERROR             = 0x05U   /*!< CAN 错误状态 */

} CAN_StateTypeDef;

// 修改 CAN_InitTypeDef 中的 FunctionalState 为 bool 类型
typedef struct
{
  uint32_t Prescaler;                  /*!< Specifies the length of a time quantum. */

  uint32_t Mode;                       /*!< Specifies the CAN operating mode. */

  uint32_t SyncJumpWidth;              /*!< Specifies the maximum number of time quanta. */

  uint32_t TimeSeg1;                   /*!< Specifies the number of time quanta in Bit Segment 1. */

  uint32_t TimeSeg2;                   /*!< Specifies the number of time quanta in Bit Segment 2. */

  bool TimeTriggeredMode;   /*!< Enable or disable the time triggered communication mode. */

  bool AutoBusOff;          /*!< Enable or disable the automatic bus-off management. */

  bool AutoWakeUp;          /*!< Enable or disable the automatic wake-up mode. */

  bool AutoRetransmission;  /*!< Enable or disable the non-automatic retransmission mode. */

  bool ReceiveFifoLocked;   /*!< Enable or disable the Receive FIFO Locked mode. */

  bool TransmitFifoPriority;/*!< Enable or disable the transmit FIFO priority. */

} CAN_InitTypeDef;

typedef struct
{
  uint32_t FilterIdHigh;
  uint32_t FilterIdLow;
  uint32_t FilterMaskIdHigh;
  uint32_t FilterMaskIdLow;
  uint32_t FilterFIFOAssignment;
  uint32_t FilterBank;
  uint32_t FilterMode;
  uint32_t FilterScale;
  bool FilterActivation;  // 修改为 bool 类型
  uint32_t SlaveStartFilterBank;

} CAN_FilterTypeDef;

typedef struct
{
  uint32_t StdId;
  uint32_t ExtId;
  uint32_t IDE;
  uint32_t RTR;
  uint32_t DLC;
  bool TransmitGlobalTime;  // 修改为 bool 类型

} CAN_TxHeaderTypeDef;

typedef struct
{
  uint32_t StdId;
  uint32_t ExtId;
  uint32_t IDE;
  uint32_t RTR;
  uint32_t DLC;
  uint32_t Timestamp;
  uint32_t FilterMatchIndex;

} CAN_RxHeaderTypeDef;

typedef struct __CAN_HandleTypeDef
{
  CAN_TypeDef                 *Instance;
  CAN_InitTypeDef             Init;
  __IO CAN_StateTypeDef   State;
  __IO uint32_t               ErrorCode;

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
  void (* TxMailbox0CompleteCallback)(struct __CAN_HandleTypeDef *hcan, CAN_TxHeaderTypeDef *pHeader);
  void (* TxMailbox1CompleteCallback)(struct __CAN_HandleTypeDef *hcan, CAN_TxHeaderTypeDef *pHeader);
  void (* TxMailbox2CompleteCallback)(struct __CAN_HandleTypeDef *hcan, CAN_TxHeaderTypeDef *pHeader);
  void (* TxMailbox0AbortCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* TxMailbox1AbortCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* TxMailbox2AbortCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* RxFifo0MsgPendingCallback)(struct __CAN_HandleTypeDef *hcan, CAN_RxHeaderTypeDef *pHeader);
  void (* RxFifo0FullCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* RxFifo1MsgPendingCallback)(struct __CAN_HandleTypeDef *hcan, CAN_RxHeaderTypeDef *pHeader);
  void (* RxFifo1FullCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* SleepCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* WakeUpFromRxMsgCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* ErrorCallback)(struct __CAN_HandleTypeDef *hcan, uint32_t ErrorCode);  // Add error code parameter

  void (* MspInitCallback)(struct __CAN_HandleTypeDef *hcan);
  void (* MspDeInitCallback)(struct __CAN_HandleTypeDef *hcan);

#endif /* (USE_HAL_CAN_REGISTER_CALLBACKS) */
} CAN_HandleTypeDef;
```

**解释:**

*   使用 `CAN_StateTypeDef` 替换 `HAL_CAN_StateTypeDef`，更简洁易懂。
*   将 `FunctionalState` 替换为 `bool` 类型，更符合 C 语言的习惯。
*   回调函数增加参数，例如 `TxMailboxCompleteCallback` 增加 `CAN_TxHeaderTypeDef *pHeader`，`RxFifoMsgPendingCallback` 增加 `CAN_RxHeaderTypeDef *pHeader`，`ErrorCallback` 增加 `uint32_t ErrorCode`，以便在回调函数中获取更多信息。

**2. 改进的宏定义：**

```c
#define CAN_ENABLE  (1U)
#define CAN_DISABLE (0U)

#define IS_CAN_MODE(MODE) (((MODE) == CAN_MODE_NORMAL) || \
                           ((MODE) == CAN_MODE_LOOPBACK)|| \
                           ((MODE) == CAN_MODE_SILENT) || \
                           ((MODE) == CAN_MODE_SILENT_LOOPBACK))

// 改进的宏，防止重复计算和类型安全
#define CAN_ASSERT_PARAM(expr) do { if (!(expr)) { assert_failed((uint8_t *)__FILE__, __LINE__); } } while(0)

#define IS_CAN_PRESCALER(PRESCALER)   CAN_ASSERT_PARAM((PRESCALER) >= 1U && (PRESCALER) <= 1024U)
#define IS_CAN_FILTER_ID(ID)          CAN_ASSERT_PARAM((ID) <= 0x1FFFFFFF) // For both StdId and ExtId checks
#define IS_CAN_DLC(DLC)               CAN_ASSERT_PARAM((DLC) <= 0x0FU)

#define __HAL_CAN_ENABLE_IT(__HANDLE__, __INTERRUPT__)   (((__HANDLE__)->Instance->IER) |= (__INTERRUPT__))
#define __HAL_CAN_DISABLE_IT(__HANDLE__, __INTERRUPT__)  (((__HANDLE__)->Instance->IER) &= ~(__INTERRUPT__))
#define __HAL_CAN_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->IER) & (__INTERRUPT__))

// Clear Flag 使用位操作，更安全
#define __HAL_CAN_CLEAR_FLAG(__HANDLE__, __FLAG__)  (((__HANDLE__)->Instance->TSR) = (__FLAG__))
```

**解释:**

*   使用 `CAN_ENABLE` 和 `CAN_DISABLE` 替换 `ENABLE` 和 `DISABLE`，更明确。
*   增加 `CAN_ASSERT_PARAM` 宏，用于参数检查，防止非法参数。
*   使用位操作来设置和清除标志位，更安全，避免误操作。

**3.  更完善的错误处理：**

```c
typedef enum
{
  CAN_ERROR_NONE            = 0x00000000U, /*!< No error */
  CAN_ERROR_EWG             = 0x00000001U, /*!< Protocol Error Warning */
  CAN_ERROR_EPV             = 0x00000002U, /*!< Error Passive */
  CAN_ERROR_BOF             = 0x00000004U, /*!< Bus-off error */
  CAN_ERROR_STF             = 0x00000008U, /*!< Stuff error */
  CAN_ERROR_FOR             = 0x00000010U, /*!< Form error */
  CAN_ERROR_ACK             = 0x00000020U, /*!< Acknowledgment error */
  CAN_ERROR_BR              = 0x00000040U, /*!< Bit recessive error */
  CAN_ERROR_BD              = 0x00000080U, /*!< Bit dominant error */
  CAN_ERROR_CRC             = 0x00000100U, /*!< CRC error */
  CAN_ERROR_RX_FOV0         = 0x00000200U, /*!< Rx FIFO0 overrun error */
  CAN_ERROR_RX_FOV1         = 0x00000400U, /*!< Rx FIFO1 overrun error */
  CAN_ERROR_TX_ALST0        = 0x00000800U, /*!< TxMailbox 0 arbitration lost */
  CAN_ERROR_TX_TERR0        = 0x00001000U, /*!< TxMailbox 0 transmit error */
  CAN_ERROR_TX_ALST1        = 0x00002000U, /*!< TxMailbox 1 arbitration lost */
  CAN_ERROR_TX_TERR1        = 0x00004000U, /*!< TxMailbox 1 transmit error */
  CAN_ERROR_TX_ALST2        = 0x00008000U, /*!< TxMailbox 2 arbitration lost */
  CAN_ERROR_TX_TERR2        = 0x00010000U, /*!< TxMailbox 2 transmit error */
  CAN_ERROR_TIMEOUT         = 0x00020000U, /*!< Timeout error */
  CAN_ERROR_NOT_INITIALIZED = 0x00040000U, /*!< Peripheral not initialized */
  CAN_ERROR_NOT_READY       = 0x00080000U, /*!< Peripheral not ready */
  CAN_ERROR_NOT_STARTED     = 0x00100000U, /*!< Peripheral not started */
  CAN_ERROR_PARAM           = 0x00200000U, /*!< Parameter error */
  CAN_ERROR_INVALID_CALLBACK = 0x00400000U, /*!< Invalid Callback error */
  CAN_ERROR_INTERNAL        = 0x00800000U, /*!< Internal error */
  CAN_ERROR_TX_BUFFER_FULL  = 0x01000000U, /*!< Transmit buffer is full */
  CAN_ERROR_RX_BUFFER_EMPTY = 0x02000000U, /*!< Receive buffer is empty */
  CAN_ERROR_INVALID_ID      = 0x04000000U  /*!< Invalid CAN ID */
} CAN_ErrorCode;
```

**解释:**

*   增加了 `CAN_ERROR_TX_BUFFER_FULL`、`CAN_ERROR_RX_BUFFER_EMPTY` 和 `CAN_ERROR_INVALID_ID` 等错误码，提供更详细的错误信息。

**4. 示例代码：**

```c
// 初始化 CAN
CAN_HandleTypeDef hcan;
CAN_InitTypeDef can_init;
CAN_FilterTypeDef can_filter;

void CAN_Init(void)
{
    hcan.Instance = CAN1;
    hcan.Init.Prescaler = 4;
    hcan.Init.Mode = CAN_MODE_NORMAL;
    hcan.Init.SyncJumpWidth = CAN_SJW_1TQ;
    hcan.Init.TimeSeg1 = CAN_BS1_12TQ;
    hcan.Init.TimeSeg2 = CAN_BS2_3TQ;
    hcan.Init.TimeTriggeredMode = CAN_DISABLE;
    hcan.Init.AutoBusOff = CAN_DISABLE;
    hcan.Init.AutoWakeUp = CAN_DISABLE;
    hcan.Init.AutoRetransmission = CAN_DISABLE;
    hcan.Init.ReceiveFifoLocked = CAN_DISABLE;
    hcan.Init.TransmitFifoPriority = CAN_DISABLE;

    if (HAL_CAN_Init(&hcan) != HAL_OK)
    {
        // 初始化失败
        while(1);
    }

    // 配置过滤器
    can_filter.FilterBank = 0;
    can_filter.FilterMode = CAN_FILTERMODE_IDMASK;
    can_filter.FilterScale = CAN_FILTERSCALE_32BIT;
    can_filter.FilterIdHigh = 0x0000;
    can_filter.FilterIdLow = 0x0000;
    can_filter.FilterMaskIdHigh = 0x0000;
    can_filter.FilterMaskIdLow = 0x0000;
    can_filter.FilterFIFOAssignment = CAN_FILTER_FIFO0;
    can_filter.FilterActivation = CAN_ENABLE;
    can_filter.SlaveStartFilterBank = 14;

    if (HAL_CAN_ConfigFilter(&hcan, &can_filter) != HAL_OK)
    {
        // 过滤器配置失败
        while(1);
    }

    // 启动 CAN
    if (HAL_CAN_Start(&hcan) != HAL_OK)
    {
        // 启动失败
        while(1);
    }

  // Enable RX FIFO0 message pending interrupt 使能接收中断
  HAL_CAN_ActivateNotification(&hcan, CAN_IT_RX_FIFO0_MSG_PENDING);
}

// 发送 CAN 消息
HAL_StatusTypeDef CAN_Transmit(uint32_t StdId, uint8_t *data, uint8_t len)
{
    CAN_TxHeaderTypeDef txHeader;
    uint32_t txMailbox;

    txHeader.StdId = StdId;
    txHeader.ExtId = 0;
    txHeader.IDE = CAN_ID_STD;
    txHeader.RTR = CAN_RTR_DATA;
    txHeader.DLC = len;
    txHeader.TransmitGlobalTime = CAN_DISABLE;

    return HAL_CAN_AddTxMessage(&hcan, &txHeader, data, &txMailbox);
}

// 接收中断回调函数
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan, CAN_RxHeaderTypeDef *pHeader)
{
    uint8_t data[8];

    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, pHeader, data) == HAL_OK)
    {
        // 接收到消息
        // 处理消息...
    }
}

void HAL_CAN_ErrorCallback(CAN_HandleTypeDef *hcan, uint32_t ErrorCode)
{
    // 处理错误
    if (ErrorCode & CAN_ERROR_BOF) {
        // 总线关闭错误
    }
}
```

**中文解释:**

*   **CAN 初始化:**  示例代码展示了如何使用 `HAL_CAN_Init` 函数初始化 CAN 模块，包括设置波特率、工作模式、时间段等。  `CAN_InitTypeDef`  结构体用于配置 CAN 的各种参数，例如 `Prescaler` (分频系数), `Mode` (工作模式), `TimeSeg1` 和 `TimeSeg2` (时间段)。
*   **CAN 过滤器配置:** 使用 `HAL_CAN_ConfigFilter` 函数配置 CAN 过滤器，可以过滤不需要的消息，减少 CPU 负担。`CAN_FilterTypeDef`  结构体用于配置过滤器的参数，例如 `FilterIdHigh`,  `FilterIdLow`,  `FilterMaskIdHigh`,  `FilterMaskIdLow`  和  `FilterFIFOAssignment` (FIFO选择)。
*   **CAN 消息发送:**  使用 `HAL_CAN_AddTxMessage` 函数发送 CAN 消息。`CAN_TxHeaderTypeDef` 结构体用于配置发送消息的头部信息，例如 `StdId` (标准ID), `ExtId` (扩展ID), `IDE` (ID类型), `RTR` (远程帧请求) 和 `DLC` (数据长度)。
*   **CAN 消息接收:**  通过中断方式接收 CAN 消息，并在 `HAL_CAN_RxFifo0MsgPendingCallback` 回调函数中处理接收到的消息。
*   **CAN 错误处理:**  通过 `HAL_CAN_ErrorCallback` 回调函数处理 CAN 错误。

**5.  安全相关的改进**
```c
//使用强制类型转换避免潜在的类型不匹配
#define __HAL_CAN_CLEAR_FLAG(__HANDLE__, __FLAG__)  (((__HANDLE__)->Instance->TSR) = (uint32_t)(__FLAG__))

//提供更强的参数检查
HAL_StatusTypeDef HAL_CAN_AddTxMessage(CAN_HandleTypeDef *hcan, CAN_TxHeaderTypeDef *pHeader, uint8_t aData[], uint32_t *pTxMailbox)
{
  /* Check parameters */
  CAN_ASSERT_PARAM(IS_CAN_STDID(pHeader->StdId) || IS_CAN_EXTID(pHeader->ExtId));
  CAN_ASSERT_PARAM(IS_CAN_DLC(pHeader->DLC));
  /* Other checks... */
}
```

**总结:**

这些改进旨在使 STM32F1xx HAL CAN 驱动更易于使用、更安全、更可靠。 通过更清晰的类型定义、更安全的宏、更完善的错误处理和更灵活的回调，可以提高开发效率和代码质量。
