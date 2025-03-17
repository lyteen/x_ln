Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\spi.c`

Okay, restarting with a focus on providing smaller, more modular code snippets with Chinese explanations, and a simple demo at the end.  Let's aim for clarity and ease of understanding.

**1.  SPI Initialization Structure (`spi.h`)**

```c
#ifndef __SPI_H
#define __SPI_H

#ifdef __cplusplus
 extern "C" {
#endif

#include "stm32f1xx_hal.h"  //  根据你的STM32型号更改
// 根据你的STM32型号更改，例如 stm32f4xx_hal.h 或其他

/*  Exported types ------------------------------------------------------------*/
/*  Exported constants --------------------------------------------------------*/
/*  Exported macro ------------------------------------------------------------*/
/*  Exported functions prototypes ---------------------------------------------*/

void MX_SPI1_Init(void);
void HAL_SPI_MspInit(SPI_HandleTypeDef* spiHandle);
void HAL_SPI_MspDeInit(SPI_HandleTypeDef* spiHandle);

extern SPI_HandleTypeDef hspi1;

#ifdef __cplusplus
}
#endif

#endif /* __SPI_H */
```

**中文描述:**

*   `#ifndef __SPI_H ... #endif`:  这是一个头文件保护，防止重复包含 `spi.h` 文件。
*   `#include "stm32f1xx_hal.h"`: 包含 STM32 HAL 库的头文件。 **重要:** 你需要根据你使用的STM32型号更改这个文件! 例如如果是STM32F4系列，就改成 `"stm32f4xx_hal.h"`。
*   `void MX_SPI1_Init(void);`:  SPI1 初始化函数的声明。这个函数负责配置 SPI1 外设。
*   `void HAL_SPI_MspInit(SPI_HandleTypeDef* spiHandle);`:  底层 SPI 初始化函数声明 (MSP: MCU Specific Package)。  它负责配置 SPI 使用的引脚 (GPIO) 和时钟。
*   `void HAL_SPI_MspDeInit(SPI_HandleTypeDef* spiHandle);`:  底层 SPI 反初始化函数声明。
*   `extern SPI_HandleTypeDef hspi1;`:  声明 SPI1 句柄变量。这个变量在 `spi.c` 文件中定义。

**2. SPI Initialization Function (`spi.c`)**

```c
#include "spi.h"

SPI_HandleTypeDef hspi1;

void MX_SPI1_Init(void) {
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;  // 修改为 8 位数据
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;   // 修改为低电平有效
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;    // 修改为第一个边沿采样
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16; // 更慢的速度
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;

  if (HAL_SPI_Init(&hspi1) != HAL_OK) {
    Error_Handler(); // 错误处理函数 (需要你自己实现)
  }
}
```

**中文描述:**

*   `#include "spi.h"`: 包含 `spi.h` 头文件，声明了所需的函数和变量。
*   `SPI_HandleTypeDef hspi1;`: 定义 SPI1 句柄变量。  HAL 库使用句柄来管理 SPI 外设。
*   `MX_SPI1_Init()`:  SPI1 初始化函数。
    *   `hspi1.Instance = SPI1;`:  设置 SPI 实例为 SPI1。
    *   `hspi1.Init.Mode = SPI_MODE_MASTER;`:  配置 SPI 为主机模式。
    *   `hspi1.Init.Direction = SPI_DIRECTION_2LINES;`:  配置为双线全双工模式。
    *   `hspi1.Init.DataSize = SPI_DATASIZE_8BIT;`: **重要修改:**  配置数据大小为 8 位。  更常见也更易于理解。
    *   `hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;`: **重要修改:** 配置时钟极性为低电平。  空闲时钟线为低电平。
    *   `hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;`: **重要修改:** 配置时钟相位为第一个边沿采样。  在时钟的上升沿或下降沿采集数据，取决于极性。
    *   `hspi1.Init.NSS = SPI_NSS_SOFT;`:  使用软件控制 NSS (片选) 信号。
    *   `hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;`: **重要修改:**  使用更低的波特率预分频值。  这样 SPI 通信速度会更慢，更稳定，尤其是在调试阶段。
    *   `HAL_SPI_Init(&hspi1)`:  调用 HAL 库函数初始化 SPI 外设。
    *   `Error_Handler()`:  如果初始化失败，调用错误处理函数。**你需要自己实现这个函数，例如在 `main.c` 中。**

**3.  MSP Initialization Function (`spi.c`)**

```c
void HAL_SPI_MspInit(SPI_HandleTypeDef* spiHandle) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  if (spiHandle->Instance == SPI1) {
    /* SPI1 clock enable */
    __HAL_RCC_SPI1_CLK_ENABLE();

    /* GPIO Configuration */
    __HAL_RCC_GPIOA_CLK_ENABLE(); // 或者 GPIOB，根据你的硬件连接

    GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;  // SCK, MISO, MOSI on PA5, PA6, PA7 (示例)
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;        // Alternate Function Push Pull
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);      // 或者 GPIOB

    //  如果使用软件 NSS，需要配置 NSS 引脚 (可选)
    //  GPIO_InitStruct.Pin = GPIO_PIN_4; // 假设 NSS 在 PA4
    //  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    //  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    //  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // __HAL_AFIO_REMAP_SPI1_ENABLE(); // 如果需要重映射，取消注释
  }
}
```

**中文描述:**

*   `HAL_SPI_MspInit()`:  底层 SPI 初始化函数。  它配置 SPI 使用的 GPIO 引脚和时钟。
    *   `__HAL_RCC_SPI1_CLK_ENABLE();`:  使能 SPI1 时钟。
    *   `__HAL_RCC_GPIOA_CLK_ENABLE();`:  使能 GPIOA 时钟 (或者 GPIOB，取决于你的硬件连接)。
    *   `GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;`:  配置 GPIO 引脚。  **你需要根据你的硬件连接修改引脚编号!**  这里假设 SCK 在 PA5, MISO 在 PA6, MOSI 在 PA7。
    *   `GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;`:  配置 GPIO 为复用推挽输出模式。
    *   `HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);`:  初始化 GPIO 引脚。
    *   (可选) 如果使用软件 NSS，配置 NSS 引脚为输出。
    *   `__HAL_AFIO_REMAP_SPI1_ENABLE();`:  如果需要重映射 SPI1 引脚，取消注释。

**4. MSP De-initialization Function (`spi.c`)**

```c
void HAL_SPI_MspDeInit(SPI_HandleTypeDef* spiHandle) {
  if (spiHandle->Instance == SPI1) {
    /* Peripheral clock disable */
    __HAL_RCC_SPI1_CLK_DISABLE();

    /* SPI1 GPIO Configuration */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7); // 确保引脚号正确

    // 如果使用了软件 NSS，也要反初始化 NSS 引脚
    // HAL_GPIO_DeInit(GPIOA, GPIO_PIN_4);
  }
}
```

**中文描述:**

*   `HAL_SPI_MspDeInit()`:  底层 SPI 反初始化函数。 它释放 SPI 使用的资源。
    *   `__HAL_RCC_SPI1_CLK_DISABLE();`:  禁用 SPI1 时钟。
    *   `HAL_GPIO_DeInit(GPIOA, GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7);`:  反初始化 GPIO 引脚。**确保引脚号与 `HAL_SPI_MspInit` 中配置的相同!**

**5.  SPI Transmit/Receive Function (Example in `main.c`)**

```c
#include "main.h" // 包含 Error_Handler 的定义
#include "spi.h"

uint8_t spi_transmit_receive(uint8_t data_to_send) {
  uint8_t received_data;
  HAL_SPI_TransmitReceive(&hspi1, &data_to_send, &received_data, 1, HAL_MAX_DELAY);
  return received_data;
}

void Error_Handler(void) {
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1) {
    // 可以添加错误指示代码，例如闪烁 LED
  }
}
```

**中文描述:**

*   `spi_transmit_receive()`:  一个简单的 SPI 发送/接收函数。
    *   `HAL_SPI_TransmitReceive()`:  HAL 库提供的 SPI 发送/接收函数。  它同时发送一个字节并接收一个字节。
    *   `&hspi1`:  SPI 句柄。
    *   `&data_to_send`:  指向要发送的数据的指针。
    *   `&received_data`:  指向接收数据的指针。
    *   `1`:  要发送/接收的数据的字节数。
    *   `HAL_MAX_DELAY`:  超时时间。  `HAL_MAX_DELAY` 表示无限等待。

**6.  Simple Demo (in `main.c`)**

```c
#include "main.h"
#include "spi.h"

int main(void) {
  HAL_Init();
  SystemClock_Config(); // 配置系统时钟 (需要你自己实现)
  MX_GPIO_Init();      // 初始化 GPIO (需要你自己实现，例如 LED 引脚)
  MX_SPI1_Init();

  uint8_t data_to_send = 0xAA; //  要发送的数据
  uint8_t received_data;

  while (1) {
    received_data = spi_transmit_receive(data_to_send);

    //  检查接收到的数据 (例如，与期望值比较)
    if (received_data == 0x55) {
      HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET); // 点亮 LED (假设 LED 已正确配置)
    } else {
      HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET); // 熄灭 LED
    }
    HAL_Delay(100);
  }
}
```

**中文描述:**

*   `main()`:  主函数。
    *   `HAL_Init()`:  初始化 HAL 库。
    *   `SystemClock_Config()`:  配置系统时钟。**你需要自己实现这个函数，根据你的硬件设置选择合适的时钟源和频率。**
    *   `MX_GPIO_Init()`:  初始化 GPIO。**你需要自己实现这个函数，配置 LED 引脚。**
    *   `MX_SPI1_Init()`:  初始化 SPI1 外设。
    *   `data_to_send = 0xAA;`:  设置要发送的数据。
    *   `received_data = spi_transmit_receive(data_to_send);`:  调用 `spi_transmit_receive()` 函数发送数据并接收数据。
    *   如果接收到的数据是 `0x55`，点亮 LED；否则，熄灭 LED。 这只是一个简单的示例，用于演示 SPI 通信。你需要根据你的应用场景修改代码。

**重要注意事项:**

*   **时钟配置 (`SystemClock_Config()`):**  确保你的系统时钟配置正确。 SPI 的速度依赖于系统时钟。
*   **GPIO 配置 (`MX_GPIO_Init()` and in `HAL_SPI_MspInit()`):**  仔细检查 GPIO 引脚的配置，确保它们与你的硬件连接匹配。  错误的 GPIO 配置是 SPI 通信问题的常见原因。
*   **错误处理 (`Error_Handler()`):**  实现 `Error_Handler()` 函数，以便在发生错误时能够及时发现和处理。
*   **硬件连接:**  确保 SPI 设备 (例如传感器或存储器) 已正确连接到 STM32 的 SPI 引脚。
*   **片选 (NSS) 管理:**  根据你的硬件连接，选择合适的 NSS 管理方式 (软件或硬件)。 如果使用软件 NSS，需要手动控制 NSS 引脚的电平。

这个例子将SPI的初始化配置，数据发送接收都细化了。你可以根据你的实际硬件配置和应用需求，修改这些代码。 请记住，调试嵌入式系统需要耐心和细致的分析。
