Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\Legacy\stm32_hal_legacy.h`

好的，我们重新开始。 这次，我将重点放在改进后的STM32 HAL库的实例代码，用中文解释代码，并且提供简短的演示。

**目标：** 优化现有的`stm32_hal_legacy.h`文件中的别名定义，并提供更清晰、更具实用性的示例代码，方便开发者理解和使用STM32 HAL库。

**主要内容:**

*   **改进的HAL库别名:** 减少冗余和错误，确保与最新HAL库的兼容性。
*   **示例代码:** 提供I2C，ADC，UART等常用外设的使用示例。
*   **中文注释:** 每段代码都有详细的中文注释，方便理解。
*   **Demo介绍:** 每段代码都附带简单的中文Demo介绍，方便上手。

**以下是优化的`stm32_hal_legacy.h`头文件片段和示例代码:**

```c
#ifndef STM32_HAL_LEGACY
#define STM32_HAL_LEGACY

#ifdef __cplusplus
 extern "C" {
#endif

/* ------------------ ADC Aliased Defines ------------------- */
#define ADC_RESOLUTION_12B       ADC_RESOLUTION_12B  // 12位分辨率
#define ADC_EOC_SINGLE           ADC_EOC_SINGLE // 单次转换结束标志
#define ADC_REGULAR_GROUP        ADC_REGULAR_GROUP // 规则组
#define ADC_AWD_EVENT            ADC_AWD_EVENT     // 模拟看门狗事件
#define ADC_CLOCK_SYNC_PCLK_DIV2 ADC_CLOCK_SYNC_PCLK_DIV2 // 时钟分频

/* ------------------ I2C Aliased Defines ------------------- */
#define I2C_DUALADDRESS_ENABLE   I2C_DUALADDRESS_ENABLE // 双地址使能
#define I2C_GENERALCALL_ENABLE  I2C_GENERALCALL_ENABLE  // 广播呼叫使能
#define I2C_ANALOGFILTER_ENABLE  I2C_ANALOGFILTER_ENABLE // 模拟滤波器使能
#define HAL_I2C_STATE_BUSY_TX    HAL_I2C_STATE_BUSY_TX // I2C状态 - 发送忙

/* ------------------ UART Aliased Defines ------------------ */
#define UART_ONE_BIT_SAMPLE_ENABLE UART_ONE_BIT_SAMPLE_ENABLE // 单比特采样使能

/* ------------------ GPIO Aliased Defines ------------------ */
#define GPIO_SPEED_FREQ_LOW      GPIO_SPEED_FREQ_LOW // 低速GPIO

#ifdef __cplusplus
}
#endif

#endif /* STM32_HAL_LEGACY */
```

**示例代码1：ADC配置及读取 (ADC Configuration and Reading)**

```c
#include "stm32f4xx_hal.h"  // 包含HAL库头文件
#include <stdio.h>

ADC_HandleTypeDef hadc1; // ADC句柄

// ADC初始化函数
void ADC1_Init(void) {
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4; // 时钟分频
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;  // 12位分辨率
    hadc1.Init.ScanConvMode = DISABLE; // 禁止扫描模式
    hadc1.Init.ContinuousConvMode = ENABLE; // 连续转换模式
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.NbrOfConversion = 1;  // 转换通道数量
    hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE; // 外部触发
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;  // 数据右对齐
    hadc1.Init.NbrOfDiscConversion = 0;
    HAL_ADC_Init(&hadc1); // 初始化ADC
}

// 添加通道配置 (配置为通道0)
void ADC1_ChannelConf(void) {
    ADC_ChannelConfTypeDef sConfig = {0};
    sConfig.Channel = ADC_CHANNEL_0;
    sConfig.Rank = 1;  // 优先级
    sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES; // 采样时间
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);
}

int main(void) {
  HAL_Init(); // 初始化HAL库
  ADC1_Init(); // 初始化ADC1
  ADC1_ChannelConf(); // 配置ADC通道

  HAL_ADC_Start(&hadc1); // 启动ADC转换
  while (1) {
    HAL_ADC_PollForConversion(&hadc1, 10); // 等待转换完成 (10ms超时)
    if ((HAL_ADC_GetState(&hadc1) & HAL_ADC_STATE_EOC_REG) == HAL_ADC_STATE_EOC_REG) {
      uint32_t value = HAL_ADC_GetValue(&hadc1); // 读取ADC值
      printf("ADC Value: %lu\r\n", value);  // 打印ADC值
      HAL_Delay(500); // 延时500ms
    }
  }
}

// 中文Demo介绍：
// 此示例代码展示了如何配置和使用STM32的ADC。 ADC1被配置为连续转换通道0上的电压，并将读取的值通过串口打印出来。 代码包括初始化ADC，配置通道，启动ADC转换和读取ADC值的部分。

```

**示例代码2：I2C发送数据 (I2C Sending Data)**

```c
#include "stm32f4xx_hal.h"
#include <stdio.h>

I2C_HandleTypeDef hi2c1;

// I2C初始化函数
void I2C1_Init(void) {
    hi2c1.Instance = I2C1;
    hi2c1.Init.ClockSpeed = 100000;   // 100kHz
    hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    HAL_I2C_Init(&hi2c1);
}

int main(void) {
  HAL_Init();
  I2C1_Init();

  uint8_t data[] = "Hello I2C!"; // 待发送的数据
  uint16_t slave_address = 0x68; // 从机地址 (7-bit)

  HAL_StatusTypeDef status = HAL_I2C_Master_Transmit(&hi2c1, slave_address << 1, data, sizeof(data) - 1, 100); // 发送数据
    //slave_address << 1 左移一位是写入地址，要发送数据

  if (status != HAL_OK) {
    printf("I2C Transmission Error\r\n");
  } else {
    printf("I2C Transmission Successful\r\n");
  }

  while (1) {
    HAL_Delay(1000);
  }
}

// 中文Demo介绍：
// 此示例代码演示了如何使用I2C向从设备发送数据。 I2C1被初始化为100kHz，然后向地址为0x68的从设备发送字符串 "Hello I2C!"。 如果发送成功，会打印“I2C Transmission Successful”，否则打印错误信息。

```

**示例代码3：UART发送和接收数据 (UART Send and Receive Data)**

```c
#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <string.h>

UART_HandleTypeDef huart2; // UART句柄

// UART初始化函数
void UART2_Init(void) {
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200; // 波特率
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX; // 收发模式
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  HAL_UART_Init(&huart2); // 初始化UART
}

int main(void) {
  HAL_Init();
  UART2_Init();

  char tx_data[] = "Hello UART!\r\n"; // 待发送的数据
  uint8_t rx_data[32]; // 接收缓冲区
  memset(rx_data, 0, sizeof(rx_data)); // 清空接收缓冲区

  HAL_UART_Transmit(&huart2, (uint8_t*)tx_data, strlen(tx_data), 100); // 发送数据
  printf("Data sent over UART\r\n");

  HAL_UART_Receive(&huart2, rx_data, sizeof(rx_data) - 1, 1000); // 接收数据
  printf("Received data: %s\r\n", rx_data); //打印接收的数据

  while (1) {
    HAL_Delay(1000);
  }
}

// 中文Demo介绍：
// 此示例代码演示了如何使用UART发送和接收数据。 UART2被初始化为115200波特率，8位数据位，无校验，单停止位。 首先，发送字符串 "Hello UART!\r\n"，然后尝试接收最多31个字节的数据。 接收的数据通过串口打印。

```

**说明：**

*   这些示例代码需要在具体的STM32开发环境中编译和运行。
*   请根据你的STM32型号修改外设的句柄和引脚配置。
*   需要在`HAL_MspInit()`函数中配置GPIO和时钟。

这些例子希望能帮助你理解STM32 HAL库的基本使用方法。  希望这些对你有帮助！