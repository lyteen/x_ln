Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_spi.h`

好的，我们重新开始。这次我会更详细地解释每个部分，并提供更完整的中文注释和示例，希望能帮助你更好地理解这段代码。

**1. 头文件保护和 C++ 兼容性 (Header Guard and C++ Compatibility):**

```c
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef STM32F1xx_HAL_SPI_H
#define STM32F1xx_HAL_SPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal_def.h"
```

**描述:**

*   `#ifndef STM32F1xx_HAL_SPI_H`, `#define STM32F1xx_HAL_SPI_H`, `#endif`：这部分代码是头文件保护机制，防止头文件被重复包含，避免编译错误。  如果 `STM32F1xx_HAL_SPI_H` 没有被定义，就定义它，并包含头文件的内容。 如果已经被定义了，就忽略头文件中的内容，直接跳到 `#endif`。
*   `#ifdef __cplusplus extern "C" { #endif` 和 `#ifdef __cplusplus } #endif`：  这是 C++ 兼容性处理。 因为 C 和 C++ 的函数调用约定不同，当 C++ 代码调用 C 代码时，需要使用 `extern "C"` 来告诉编译器，按照 C 的方式进行编译和链接。

**如何使用:**

这部分代码一般不需要手动修改，是由开发环境自动生成的。 它的作用是确保头文件只被包含一次。

**2. 包含头文件 (Includes):**

```c
/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal_def.h"
```

**描述:**

*   `#include "stm32f1xx_hal_def.h"`：包含 `stm32f1xx_hal_def.h` 头文件，这个头文件通常定义了一些公共的类型定义、宏定义和其他 HAL 库需要的基础结构。

**如何使用:**

`stm32f1xx_hal_def.h`  文件通常需要手动包含。 它提供了 STM32 HAL 库的基础设施，使得 SPI 驱动程序可以使用 HAL 库提供的功能。

**3. 外设驱动分组 (Peripheral Driver Grouping):**

```c
/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @addtogroup SPI
  * @{
  */
```

**描述:**

*   `/** @addtogroup STM32F1xx_HAL_Driver` 和 `/** @addtogroup SPI`：这些是文档注释，用于将 HAL 驱动程序组织成逻辑组。  `STM32F1xx_HAL_Driver`  是 HAL 驱动程序的顶层组， `SPI` 是该组下的一个子组。  这种组织方式可以方便开发者查找和理解 HAL 库的功能。

**如何使用:**

这部分代码是文档生成工具（例如 Doxygen）使用的，用来生成 API 文档。 开发者通常不需要手动修改。

**4. 导出类型定义 (Exported Types):**

```c
/* Exported types ------------------------------------------------------------*/
/** @defgroup SPI_Exported_Types SPI Exported Types
  * @{
  */

/**
  * @brief  SPI Configuration Structure definition
  */
typedef struct
{
  uint32_t Mode;                /*!< Specifies the SPI operating mode.
                                     This parameter can be a value of @ref SPI_Mode */

  uint32_t Direction;           /*!< Specifies the SPI bidirectional mode state.
                                     This parameter can be a value of @ref SPI_Direction */

  uint32_t DataSize;            /*!< Specifies the SPI data size.
                                     This parameter can be a value of @ref SPI_Data_Size */

  uint32_t CLKPolarity;         /*!< Specifies the serial clock steady state.
                                     This parameter can be a value of @ref SPI_Clock_Polarity */

  uint32_t CLKPhase;            /*!< Specifies the clock active edge for the bit capture.
                                     This parameter can be a value of @ref SPI_Clock_Phase */

  uint32_t NSS;                 /*!< Specifies whether the NSS signal is managed by
                                     hardware (NSS pin) or by software using the SSI bit.
                                     This parameter can be a value of @ref SPI_Slave_Select_management */

  uint32_t BaudRatePrescaler;   /*!< Specifies the Baud Rate prescaler value which will be
                                     used to configure the transmit and receive SCK clock.
                                     This parameter can be a value of @ref SPI_BaudRate_Prescaler
                                     @note The communication clock is derived from the master
                                     clock. The slave clock does not need to be set. */

  uint32_t FirstBit;            /*!< Specifies whether data transfers start from MSB or LSB bit.
                                     This parameter can be a value of @ref SPI_MSB_LSB_transmission */

  uint32_t TIMode;              /*!< Specifies if the TI mode is enabled or not.
                                     This parameter can be a value of @ref SPI_TI_mode */

  uint32_t CRCCalculation;      /*!< Specifies if the CRC calculation is enabled or not.
                                     This parameter can be a value of @ref SPI_CRC_Calculation */

  uint32_t CRCPolynomial;       /*!< Specifies the polynomial used for the CRC calculation.
                                     This parameter must be an odd number between Min_Data = 1 and Max_Data = 65535 */
} SPI_InitTypeDef;
```

**描述:**

*   `typedef struct { ... } SPI_InitTypeDef;`：定义了一个结构体 `SPI_InitTypeDef`，用于配置 SPI 外设的各种参数。  结构体成员包括：
    *   `Mode`：SPI 的工作模式（主模式或从模式）。
    *   `Direction`：SPI 的数据传输方向（单向、双向）。
    *   `DataSize`：SPI 的数据位长度（8 位或 16 位）。
    *   `CLKPolarity`：SPI 时钟极性。
    *   `CLKPhase`：SPI 时钟相位。
    *   `NSS`：NSS 信号的管理方式（硬件或软件）。
    *   `BaudRatePrescaler`：波特率预分频器，用于设置 SPI 的通信速率。
    *   `FirstBit`：数据传输的起始位（MSB 或 LSB）。
    *   `TIMode`：TI 模式是否启用。
    *   `CRCCalculation`：CRC 校验是否启用。
    *   `CRCPolynomial`：CRC 校验的多项式。

**如何使用:**

在初始化 SPI 外设时，需要创建一个 `SPI_InitTypeDef` 结构体变量，并设置其中的成员。 然后，将该结构体变量传递给 `HAL_SPI_Init()` 函数，用于配置 SPI 外设。

**代码示例:**

```c
SPI_HandleTypeDef hspi1;
SPI_InitTypeDef spi_init;

// 配置 SPI1
hspi1.Instance = SPI1;
spi_init.Mode = SPI_MODE_MASTER;
spi_init.Direction = SPI_DIRECTION_2LINES;
spi_init.DataSize = SPI_DATASIZE_8BIT;
spi_init.CLKPolarity = SPI_POLARITY_LOW;
spi_init.CLKPhase = SPI_PHASE_1EDGE;
spi_init.NSS = SPI_NSS_SOFT;
spi_init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
spi_init.FirstBit = SPI_FIRSTBIT_MSB;
spi_init.TIMode = SPI_TIMODE_DISABLE;
spi_init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
spi_init.CRCPolynomial = 10;

hspi1.Init = spi_init;
HAL_SPI_Init(&hspi1);  // 使用 HAL_SPI_Init 函数初始化 SPI1
```

```c
/**
  * @brief  HAL SPI State structure definition
  */
typedef enum
{
  HAL_SPI_STATE_RESET      = 0x00U,    /*!< Peripheral not Initialized                         */
  HAL_SPI_STATE_READY      = 0x01U,    /*!< Peripheral Initialized and ready for use           */
  HAL_SPI_STATE_BUSY       = 0x02U,    /*!< an internal process is ongoing                     */
  HAL_SPI_STATE_BUSY_TX    = 0x03U,    /*!< Data Transmission process is ongoing               */
  HAL_SPI_STATE_BUSY_RX    = 0x04U,    /*!< Data Reception process is ongoing                  */
  HAL_SPI_STATE_BUSY_TX_RX = 0x05U,    /*!< Data Transmission and Reception process is ongoing */
  HAL_SPI_STATE_ERROR      = 0x06U,    /*!< SPI error state                                    */
  HAL_SPI_STATE_ABORT      = 0x07U     /*!< SPI abort is ongoing                               */
} HAL_SPI_StateTypeDef;
```

**描述:**

*   `typedef enum { ... } HAL_SPI_StateTypeDef;`：定义了一个枚举类型 `HAL_SPI_StateTypeDef`，表示 SPI 外设的状态。
    *   `HAL_SPI_STATE_RESET`：SPI 外设未初始化。
    *   `HAL_SPI_STATE_READY`：SPI 外设已初始化，可以进行数据传输。
    *   `HAL_SPI_STATE_BUSY`：SPI 外设忙碌，正在进行内部操作。
    *   `HAL_SPI_STATE_BUSY_TX`：SPI 外设忙碌，正在发送数据。
    *   `HAL_SPI_STATE_BUSY_RX`：SPI 外设忙碌，正在接收数据。
    *   `HAL_SPI_STATE_BUSY_TX_RX`：SPI 外设忙碌，正在同时发送和接收数据。
    *   `HAL_SPI_STATE_ERROR`：SPI 外设出现错误。
    *   `HAL_SPI_STATE_ABORT`：SPI 中止正在进行。

**如何使用:**

可以使用 `HAL_SPI_GetState()` 函数获取 SPI 外设的当前状态。 根据状态，可以判断 SPI 外设是否可以进行数据传输，或者是否出现了错误。

**代码示例:**

```c
SPI_HandleTypeDef hspi1;
HAL_SPI_StateTypeDef spi_state;

spi_state = HAL_SPI_GetState(&hspi1);

if (spi_state == HAL_SPI_STATE_READY) {
  // 可以进行数据传输
} else if (spi_state == HAL_SPI_STATE_ERROR) {
  // 出现了错误
  uint32_t error_code = HAL_SPI_GetError(&hspi1);
  // 处理错误
}
```

```c
/**
  * @brief  SPI handle Structure definition
  */
typedef struct __SPI_HandleTypeDef
{
  SPI_TypeDef                *Instance;      /*!< SPI registers base address               */

  SPI_InitTypeDef            Init;           /*!< SPI communication parameters             */

  uint8_t                    *pTxBuffPtr;    /*!< Pointer to SPI Tx transfer Buffer        */

  uint16_t                   TxXferSize;     /*!< SPI Tx Transfer size                     */

  __IO uint16_t              TxXferCount;    /*!< SPI Tx Transfer Counter                  */

  uint8_t                    *pRxBuffPtr;    /*!< Pointer to SPI Rx transfer Buffer        */

  uint16_t                   RxXferSize;     /*!< SPI Rx Transfer size                     */

  __IO uint16_t              RxXferCount;    /*!< SPI Rx Transfer Counter                  */

  void (*RxISR)(struct __SPI_HandleTypeDef *hspi);   /*!< function pointer on Rx ISR       */

  void (*TxISR)(struct __SPI_HandleTypeDef *hspi);   /*!< function pointer on Tx ISR       */

  DMA_HandleTypeDef          *hdmatx;        /*!< SPI Tx DMA Handle parameters             */

  DMA_HandleTypeDef          *hdmarx;        /*!< SPI Rx DMA Handle parameters             */

  HAL_LockTypeDef            Lock;           /*!< Locking object                           */

  __IO HAL_SPI_StateTypeDef  State;          /*!< SPI communication state                  */

  __IO uint32_t              ErrorCode;      /*!< SPI Error code                           */

#if (USE_HAL_SPI_REGISTER_CALLBACKS == 1U)
  void (* TxCpltCallback)(struct __SPI_HandleTypeDef *hspi);             /*!< SPI Tx Completed callback          */
  void (* RxCpltCallback)(struct __SPI_HandleTypeDef *hspi);             /*!< SPI Rx Completed callback          */
  void (* TxRxCpltCallback)(struct __SPI_HandleTypeDef *hspi);           /*!< SPI TxRx Completed callback        */
  void (* TxHalfCpltCallback)(struct __SPI_HandleTypeDef *hspi);         /*!< SPI Tx Half Completed callback     */
  void (* RxHalfCpltCallback)(struct __SPI_HandleTypeDef *hspi);         /*!< SPI Rx Half Completed callback     */
  void (* TxRxHalfCpltCallback)(struct __SPI_HandleTypeDef *hspi);       /*!< SPI TxRx Half Completed callback   */
  void (* ErrorCallback)(struct __SPI_HandleTypeDef *hspi);              /*!< SPI Error callback                 */
  void (* AbortCpltCallback)(struct __SPI_HandleTypeDef *hspi);          /*!< SPI Abort callback                 */
  void (* MspInitCallback)(struct __SPI_HandleTypeDef *hspi);            /*!< SPI Msp Init callback              */
  void (* MspDeInitCallback)(struct __SPI_HandleTypeDef *hspi);          /*!< SPI Msp DeInit callback            */

#endif  /* USE_HAL_SPI_REGISTER_CALLBACKS */
} SPI_HandleTypeDef;
```

**描述:**

*   `typedef struct __SPI_HandleTypeDef { ... } SPI_HandleTypeDef;`：定义了一个结构体 `SPI_HandleTypeDef`，它是 SPI 驱动程序的核心数据结构，用于保存 SPI 外设的各种信息。
    *   `Instance`：指向 SPI 外设寄存器地址的指针。 例如，`SPI1`，`SPI2` 等。
    *   `Init`：`SPI_InitTypeDef` 结构体变量，保存 SPI 的配置参数。
    *   `pTxBuffPtr`：指向发送缓冲区的指针。
    *   `TxXferSize`：发送数据的大小。
    *   `TxXferCount`：发送计数器，记录已经发送的数据量。
    *   `pRxBuffPtr`：指向接收缓冲区的指针。
    *   `RxXferSize`：接收数据的大小。
    *   `RxXferCount`：接收计数器，记录已经接收的数据量。
    *   `RxISR`：接收中断服务例程指针。
    *   `TxISR`：发送中断服务例程指针。
    *   `hdmatx`：指向发送 DMA 句柄的指针。
    *   `hdmarx`：指向接收 DMA 句柄的指针。
    *   `Lock`：锁对象，用于保护 SPI 资源。
    *   `State`：SPI 外设的当前状态，使用 `HAL_SPI_StateTypeDef` 枚举类型表示。
    *   `ErrorCode`：SPI 错误代码。
    *   `TxCpltCallback`，`RxCpltCallback`，`TxRxCpltCallback`，`TxHalfCpltCallback`，`RxHalfCpltCallback`，`TxRxHalfCpltCallback`，`ErrorCallback`，`AbortCpltCallback`，`MspInitCallback`，`MspDeInitCallback`：这些是回调函数指针，用于在 SPI 数据传输完成、发生错误或中止时，调用用户自定义的函数。  这些回调函数只有在 `USE_HAL_SPI_REGISTER_CALLBACKS` 被定义为 1 时才有效。

**如何使用:**

在使用 SPI 驱动程序时，需要创建一个 `SPI_HandleTypeDef` 结构体变量，并设置其中的成员。  然后，将该结构体变量传递给 HAL 库提供的 SPI 函数，例如 `HAL_SPI_Init()`、`HAL_SPI_Transmit()`、`HAL_SPI_Receive()` 等。

**代码示例:**

```c
SPI_HandleTypeDef hspi1;

// 配置 SPI1
hspi1.Instance = SPI1;
hspi1.Init.Mode = SPI_MODE_MASTER;
hspi1.Init.Direction = SPI_DIRECTION_2LINES;
// ... 其他配置

uint8_t tx_buffer[10];
uint8_t rx_buffer[10];

// 发送数据
HAL_SPI_Transmit(&hspi1, tx_buffer, 10, 1000);

// 接收数据
HAL_SPI_Receive(&hspi1, rx_buffer, 10, 1000);
```

```c
#if (USE_HAL_SPI_REGISTER_CALLBACKS == 1U)
/**
  * @brief  HAL SPI Callback ID enumeration definition
  */
typedef enum
{
  HAL_SPI_TX_COMPLETE_CB_ID             = 0x00U,    /*!< SPI Tx Completed callback ID         */
  HAL_SPI_RX_COMPLETE_CB_ID             = 0x01U,    /*!< SPI Rx Completed callback ID         */
  HAL_SPI_TX_RX_COMPLETE_CB_ID          = 0x02U,    /*!< SPI TxRx Completed callback ID       */
  HAL_SPI_TX_HALF_COMPLETE_CB_ID        = 0x03U,    /*!< SPI Tx Half Completed callback ID    */
  HAL_SPI_RX_HALF_COMPLETE_CB_ID        = 0x04U,    /*!< SPI Rx Half Completed callback ID    */
  HAL_SPI_TX_RX_HALF_COMPLETE_CB_ID     = 0x05U,    /*!< SPI TxRx Half Completed callback ID  */
  HAL_SPI_ERROR_CB_ID                   = 0x06U,    /*!< SPI Error callback ID                */
  HAL_SPI_ABORT_CB_ID                   = 0x07U,    /*!< SPI Abort callback ID                */
  HAL_SPI_MSPINIT_CB_ID                 = 0x08U,    /*!< SPI Msp Init callback ID             */
  HAL_SPI_MSPDEINIT_CB_ID               = 0x09U     /*!< SPI Msp DeInit callback ID           */

} HAL_SPI_CallbackIDTypeDef;

/**
  * @brief  HAL SPI Callback pointer definition
  */
typedef  void (*pSPI_CallbackTypeDef)(SPI_HandleTypeDef *hspi); /*!< pointer to an SPI callback function */

#endif /* USE_HAL_SPI_REGISTER_CALLBACKS */
```

**描述:**

*   `typedef enum { ... } HAL_SPI_CallbackIDTypeDef;`：定义了一个枚举类型 `HAL_SPI_CallbackIDTypeDef`，表示 SPI 回调函数的 ID。
    *   `HAL_SPI_TX_COMPLETE_CB_ID`：发送完成回调函数 ID。
    *   `HAL_SPI_RX_COMPLETE_CB_ID`：接收完成回调函数 ID。
    *   `HAL_SPI_TX_RX_COMPLETE_CB_ID`：发送和接收完成回调函数 ID。
    *   `HAL_SPI_TX_HALF_COMPLETE_CB_ID`：发送半完成回调函数 ID。
    *   `HAL_SPI_RX_HALF_COMPLETE_CB_ID`：接收半完成回调函数 ID。
    *   `HAL_SPI_TX_RX_HALF_COMPLETE_CB_ID`：发送和接收半完成回调函数 ID。
    *   `HAL_SPI_ERROR_CB_ID`：错误回调函数 ID。
    *   `HAL_SPI_ABORT_CB_ID`：中止回调函数 ID。
    *    `HAL_SPI_MSPINIT_CB_ID`：MSP 初始化回调函数 ID。
    *   `HAL_SPI_MSPDEINIT_CB_ID`：MSP 反初始化回调函数 ID。
*   `typedef void (*pSPI_CallbackTypeDef)(SPI_HandleTypeDef *hspi);`：定义了一个函数指针类型 `pSPI_CallbackTypeDef`，表示 SPI 回调函数的类型。

**如何使用:**

如果定义了 `USE_HAL_SPI_REGISTER_CALLBACKS` 为 1，可以使用 `HAL_SPI_RegisterCallback()` 函数注册 SPI 回调函数。  注册后，当 SPI 数据传输完成、发生错误或中止时，HAL 库会自动调用注册的回调函数。

**代码示例:**

```c
SPI_HandleTypeDef hspi1;

void SPI1_TxCpltCallback(SPI_HandleTypeDef *hspi) {
  // 发送完成的处理
}

int main(void) {
  // 初始化 SPI1
  hspi1.Instance = SPI1;
  // ... 其他初始化

#if (USE_HAL_SPI_REGISTER_CALLBACKS == 1U)
  // 注册发送完成回调函数
  HAL_SPI_RegisterCallback(&hspi1, HAL_SPI_TX_COMPLETE_CB_ID, SPI1_TxCpltCallback);
#endif

  // 发送数据
  uint8_t tx_buffer[10];
  HAL_SPI_Transmit_IT(&hspi1, tx_buffer, 10); // 使用中断方式发送数据

  while (1) {
    // ...
  }
}
```

**5. 导出常量 (Exported Constants):**

```c
/* Exported constants --------------------------------------------------------*/
/** @defgroup SPI_Exported_Constants SPI Exported Constants
  * @{
  */

/** @defgroup SPI_Error_Code SPI Error Code
  * @{
  */
#define HAL_SPI_ERROR_NONE              (0x00000000U)   /*!< No error                               */
#define HAL_SPI_ERROR_MODF              (0x00000001U)   /*!< MODF error                             */
#define HAL_SPI_ERROR_CRC               (0x00000002U)   /*!< CRC error                              */
#define HAL_SPI_ERROR_OVR               (0x00000004U)   /*!< OVR error                              */
#define HAL_SPI_ERROR_DMA               (0x00000010U)   /*!< DMA transfer error                     */
#define HAL_SPI_ERROR_FLAG              (0x00000020U)   /*!< Error on RXNE/TXE/BSY Flag             */
#define HAL_SPI_ERROR_ABORT             (0x00000040U)   /*!< Error during SPI Abort procedure       */
#if (USE_HAL_SPI_REGISTER_CALLBACKS == 1U)
#define HAL_SPI_ERROR_INVALID_CALLBACK  (0x00000080U)   /*!< Invalid Callback error                 */
#endif /* USE_HAL_SPI_REGISTER_CALLBACKS */
/**
  * @}
  */
```

**描述:**

*   定义了一组 SPI 错误代码常量，用于表示 SPI 驱动程序中可能出现的各种错误。
    *   `HAL_SPI_ERROR_NONE`：没有错误。
    *   `HAL_SPI_ERROR_MODF`：模式错误（Mode Fault Error）。
    *   `HAL_SPI_ERROR_CRC`：CRC 校验错误。
    *   `HAL_SPI_ERROR_OVR`：溢出错误（Overrun Error）。
    *   `HAL_SPI_ERROR_DMA`：DMA 传输错误。
    *   `HAL_SPI_ERROR_FLAG`：RXNE/TXE/BSY 标志错误。
    *   `HAL_SPI_ERROR_ABORT`：SPI 中止过程中的错误。
    *    `HAL_SPI_ERROR_INVALID_CALLBACK`: 无效的回调函数

**如何使用:**

可以使用 `HAL_SPI_GetError()` 函数获取 SPI 驱动程序的错误代码。  根据错误代码，可以判断 SPI 驱动程序中出现了什么错误，并采取相应的处理措施。

**代码示例:**

```c
SPI_HandleTypeDef hspi1;
HAL_StatusTypeDef status;

// 发送数据
uint8_t tx_buffer[10];
status = HAL_SPI_Transmit(&hspi1, tx_buffer, 10, 1000);

if (status != HAL_OK) {
  // 出现了错误
  uint32_t error_code = HAL_SPI_GetError(&hspi1);

  if (error_code & HAL_SPI_ERROR_OVR) {
    // 处理溢出错误
  } else if (error_code & HAL_SPI_ERROR_CRC) {
    // 处理 CRC 校验错误
  }
}
```

```c
/** @defgroup SPI_Mode SPI Mode
  * @{
  */
#define SPI_MODE_SLAVE                  (0x00000000U)
#define SPI_MODE_MASTER                 (SPI_CR1_MSTR | SPI_CR1_SSI)
/**
  * @}
  */

/** @defgroup SPI_Direction SPI Direction Mode
  * @{
  */
#define SPI_DIRECTION_2LINES            (0x00000000U)
#define SPI_DIRECTION_2LINES_RXONLY     SPI_CR1_RXONLY
#define SPI_DIRECTION_1LINE             SPI_CR1_BIDIMODE
/**
  * @}
  */

/** @defgroup SPI_Data_Size SPI Data Size
  * @{
  */
#define SPI_DATASIZE_8BIT               (0x00000000U)
#define SPI_DATASIZE_16BIT              SPI_CR1_DFF
/**
  * @}
  */

/** @defgroup SPI_Clock_Polarity SPI Clock Polarity
  * @{
  */
#define SPI_POLARITY_LOW                (0x00000000U)
#define SPI_POLARITY_HIGH               SPI_CR1_CPOL
/**
  * @}
  */

/** @defgroup SPI_Clock_Phase SPI Clock Phase
  * @{
  */
#define SPI_PHASE_1EDGE                 (0x00000000U)
#define SPI_PHASE_2EDGE                 SPI_CR1_CPHA
/**
  * @}
  */

/** @defgroup SPI_Slave_Select_management SPI Slave Select Management
  * @{
  */
#define SPI_NSS_SOFT                    SPI_CR1_SSM
#define SPI_NSS_HARD_INPUT              (0x00000000U)
#define SPI_NSS_HARD_OUTPUT             (SPI_CR2_SSOE << 16U)
/**
  * @}
  */

/** @defgroup SPI_BaudRate_Prescaler SPI BaudRate Prescaler
  * @{
  */
#define SPI_BAUDRATEPRESCALER_2         (0x00000000U)
#define SPI_BAUDRATEPRESCALER_4         (SPI_CR1_BR_0)
#define SPI_BAUDRATEPRESCALER_8         (SPI_CR1_BR_1)
#define SPI_BAUDRATEPRESCALER_16        (SPI_CR1_BR_1 | SPI_CR1_BR_0)
#define SPI_BAUDRATEPRESCALER_32        (SPI_CR1_BR_2)
#define SPI_BAUDRATEPRESCALER_64        (SPI_CR1_BR_2 | SPI_CR1_BR_0)
#define SPI_BAUDRATEPRESCALER_128       (SPI_CR1_BR_2 | SPI_CR1_BR_1)
#define SPI_BAUDRATEPRESCALER_256       (SPI_CR1_BR_2 | SPI_CR1_BR_1 | SPI_CR1_BR_0)
/**
  * @}
  */

/** @defgroup SPI_MSB_LSB_transmission SPI MSB LSB Transmission
  * @{
  */
#define SPI_FIRSTBIT_MSB                (0x00000000U)
#define SPI_FIRSTBIT_LSB                SPI_CR1_LSBFIRST
/**
  * @}
  */

/** @defgroup SPI_TI_mode SPI TI Mode
  * @{
  */
#define SPI_TIMODE_DISABLE              (0x00000000U)
/**
  * @}
  */

/** @defgroup SPI_CRC_Calculation SPI CRC Calculation
  * @{
  */
#define SPI_CRCCALCULATION_DISABLE      (0x00000000U)
#define SPI_CRCCALCULATION_ENABLE       SPI_CR1_CRCEN
/**
  * @}
  */
```

**描述:**

*   定义了一组常量，用于配置 `SPI_InitTypeDef` 结构体中的成员。
    *   `SPI_MODE_SLAVE` 和 `SPI_MODE_MASTER`：SPI 工作模式（从模式或主模式）。
    *   `SPI_DIRECTION_2LINES`、`SPI_DIRECTION_2LINES_RXONLY` 和 `SPI_DIRECTION_1LINE`：SPI 数据传输方向（双向、单向接收、单向）。
    *   `SPI_DATASIZE_8BIT` 和 `SPI_DATASIZE_16BIT`：SPI 数据位长度（8 位或 16 位）。
    *   `SPI_POLARITY_LOW` 和 `SPI_POLARITY_HIGH`：SPI 时钟极性（低电平或高电平）。
    *   `SPI_PHASE_1EDGE` 和 `SPI_PHASE_2EDGE`：SPI 时钟相位（第一个边沿或第二个边沿）。
    *   `SPI_NSS_SOFT`、`SPI_NSS_HARD_INPUT` 和 `SPI_NSS_HARD_OUTPUT`：NSS 信号的管理方式（软件、硬件输入、硬件输出）。
    *   `SPI_BAUDRATEPRESCALER_2` 到 `SPI_BAUDRATEPRESCALER_256`：波特率预分频器，用于设置 SPI 的通信速率。
    *   `SPI_FIRSTBIT_MSB` 和 `SPI_FIRSTBIT_LSB`：数据传输的起始位（MSB 或 LSB）。
    *   `SPI_TIMODE_DISABLE`：TI 模式禁用
    *   `SPI_CRCCALCULATION_DISABLE` 和 `SPI_CRCCALCULATION_ENABLE`：CRC 校验是否启用。

**如何使用:**

在初始化 SPI 外设时，需要使用这些常量来配置 `SPI_InitTypeDef` 结构体中的成员。

**代码示例:**

```c
SPI_InitTypeDef spi_init;

spi_init.Mode = SPI_MODE_MASTER;
spi_init.Direction = SPI_DIRECTION_2LINES;
spi_init.DataSize = SPI_DATASIZE_8BIT;
spi_init.CLKPolarity = SPI_POLARITY_LOW;
spi_init.CLKPhase = SPI_PHASE_1EDGE;
spi_init.NSS = SPI_NSS_SOFT;
spi_init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
spi_init.FirstBit = SPI_FIRSTBIT_MSB;
spi_init.TIMode = SPI_TIMODE_DISABLE;
spi_init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
```

```c
/** @defgroup SPI_Interrupt_definition SPI Interrupt Definition
  * @{
  */
#define SPI_IT_TXE                      SPI_CR2_TXEIE
#define SPI_IT_RXNE                     SPI_CR2_RXNEIE
#define SPI_IT_ERR                      SPI_CR2_ERRIE
/**
  * @}
  */

/** @defgroup SPI_Flags_definition SPI Flags Definition
  * @{
  */
#define SPI_FLAG_RXNE                   SPI_SR_RXNE   /* SPI status flag: Rx buffer not empty flag       */
#define SPI_FLAG_TXE                    SPI_SR_TXE    /* SPI status flag: Tx buffer empty flag           */
#define SPI_FLAG_BSY                    SPI_SR_BSY    /* SPI status flag: Busy flag                      */
#define SPI_FLAG_CRCERR                 SPI_SR_CRCERR /* SPI Error flag: CRC error flag                  */
#define SPI_FLAG_MODF                   SPI_SR_MODF   /* SPI Error flag: Mode fault flag                 */
#define SPI_FLAG_OVR                    SPI_SR_OVR    /* SPI Error flag: Overrun flag                    */
#define SPI_FLAG_MASK                   (SPI_SR_RXNE | SPI_SR_TXE | SPI_SR_BSY\
                                         | SPI_SR_CRCERR | SPI_SR_MODF | SPI_SR_OVR)
/**
  * @}
  */
```

**描述:**

*   定义了一组常量，用于控制 SPI 中断和标志位。
    *   `SPI_IT_TXE`：发送缓冲区空中断使能。
    *   `SPI_IT_RXNE`：接收缓冲区非空中断使能。
    *   `SPI_IT_ERR`：错误中断使能。
    *   `SPI_FLAG_RXNE`：接收缓冲区非空标志。
    *   `SPI_FLAG_TXE`：发送缓冲区空标志。
    *   `SPI_FLAG_BSY`：SPI 忙标志。
    *   `SPI_FLAG_CRCERR`：CRC 校验错误标志。
    *   `SPI_FLAG_MODF`：模式错误标志。
    *   `SPI_FLAG_OVR`：溢出错误标志。
    *   `SPI_FLAG_MASK`：用于屏蔽不需要的标志位

**如何使用:**

*   可以使用 `__HAL_SPI_ENABLE_IT()` 和 `__HAL_SPI_DISABLE_IT()` 宏来使能和禁止 SPI 中断。
*   可以使用 `__HAL_SPI_GET_FLAG()` 宏来读取 SPI 标志位。
*   可以使用 `__HAL_SPI_CLEAR_CRCERRFLAG()`， `__HAL_SPI_CLEAR_MODFFLAG()` 和 `__HAL_SPI_CLEAR_OVRFLAG()` 来清除对应的错误标志位.

**代码示例:**

```c
SPI_HandleTypeDef hspi1;

// 使能接收缓冲区非空中断
__HAL_SPI_ENABLE_IT(&hspi1, SPI_IT_RXNE);

// 检查接收缓冲区是否非空
if (__HAL_SPI_GET_FLAG(&hspi1, SPI_FLAG_RXNE)) {
  // 读取接收缓冲区数据
}
```

**6. 导出宏 (Exported Macros):**

```c
/* Exported macros -----------------------------------------------------------*/
/** @defgroup SPI_Exported_Macros SPI Exported Macros
  * @{
