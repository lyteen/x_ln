Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\adc.c`

好的，我们重新来，这次我会给出更智能的ADC配置代码，并附带更详细的中文解释和示例。

**目标:**  改进ADC配置，使其更灵活、更易于维护，并加入一些常用的优化技巧。

**1. adc.h 头文件**

```c
#ifndef __ADC_H
#define __ADC_H

#include "stm32f1xx_hal.h" // 根据你的芯片型号修改
#include <stdint.h>
#include <stdbool.h>

// ADC配置结构体
typedef struct {
  ADC_HandleTypeDef handle;
  DMA_HandleTypeDef dma_handle;
  uint16_t *data_buffer; // 数据缓冲区
  uint32_t buffer_size;  // 缓冲区大小 (采样点数 * 通道数)
  uint32_t sampling_time; // 采样时间
  uint32_t channels[2];  // 需要采样的通道列表，最多支持2个通道
  uint8_t num_channels;   // 通道数量
  bool use_dma; // 是否使用DMA
  void (*callback)(uint16_t *data, uint32_t size); // 数据采集完成回调函数
} ADC_ConfigTypeDef;

// 函数声明
bool ADC_Init(ADC_ConfigTypeDef *config);
void ADC_Start(ADC_ConfigTypeDef *config);
void ADC_Stop(ADC_ConfigTypeDef *config);
float ADC_GetVoltage(uint16_t adc_value, float vref); // 将ADC值转换为电压
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc); // 中断回调函数，必须实现

extern ADC_ConfigTypeDef g_adc_config; // 全局ADC配置变量

#endif
```

**描述:** 这个头文件定义了ADC配置结构体 `ADC_ConfigTypeDef`，包含了ADC句柄、DMA句柄、数据缓冲区、采样时间、通道列表、通道数量、是否使用DMA、回调函数等。  同时声明了初始化、启动、停止、电压转换等函数。

**2. adc.c 源文件**

```c
#include "adc.h"
#include "main.h" // 包含Error_Handler()

// 全局ADC配置变量
ADC_ConfigTypeDef g_adc_config;

// 初始化ADC
bool ADC_Init(ADC_ConfigTypeDef *config) {
  HAL_StatusTypeDef status;
  ADC_ChannelConfTypeDef sConfig = {0};

  // 1. 配置ADC句柄
  config->handle.Instance = ADC1;  // 使用ADC1，可以根据需要修改
  config->handle.Init.ScanConvMode = ENABLE;       // 扫描模式
  config->handle.Init.ContinuousConvMode = ENABLE; // 连续转换模式
  config->handle.Init.DiscontinuousConvMode = DISABLE;
  config->handle.Init.ExternalTrigConv = ADC_SOFTWARE_START; // 软件触发
  config->handle.Init.DataAlign = ADC_DATAALIGN_RIGHT;       // 右对齐
  config->handle.Init.NbrOfConversion = config->num_channels; // 通道数量

  status = HAL_ADC_Init(&config->handle);
  if (status != HAL_OK) {
    Error_Handler();
    return false;
  }

  // 2. 配置ADC通道
  for (uint8_t i = 0; i < config->num_channels; i++) {
    sConfig.Channel = config->channels[i];
    sConfig.Rank = i + 1; // 通道排名
    sConfig.SamplingTime = config->sampling_time;

    status = HAL_ADC_ConfigChannel(&config->handle, &sConfig);
    if (status != HAL_OK) {
      Error_Handler();
      return false;
    }
  }

  // 3. 配置DMA (如果使用)
  if (config->use_dma) {
    config->dma_handle.Instance = DMA1_Channel1; // 使用DMA1通道1，根据需要修改
    config->dma_handle.Init.Direction = DMA_PERIPH_TO_MEMORY;
    config->dma_handle.Init.PeriphInc = DMA_PINC_DISABLE;    // 外设地址不自增
    config->dma_handle.Init.MemInc = DMA_MINC_ENABLE;       // 内存地址自增
    config->dma_handle.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD; // 半字
    config->dma_handle.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;   // 半字
    config->dma_handle.Init.Mode = DMA_CIRCULAR;    // 循环模式
    config->dma_handle.Init.Priority = DMA_PRIORITY_HIGH; // 优先级

    status = HAL_DMA_Init(&config->dma_handle);
    if (status != HAL_OK) {
      Error_Handler();
      return false;
    }

    __HAL_LINKDMA(&config->handle, DMA_Handle, config->dma_handle); // 关联ADC和DMA
  }

  // 4. 校准ADC
  HAL_ADCEx_Calibration_Start(&config->handle);

  return true;
}

// 启动ADC
void ADC_Start(ADC_ConfigTypeDef *config) {
  if (config->use_dma) {
    HAL_ADC_Start_DMA(&config->handle, (uint32_t *)config->data_buffer, config->buffer_size);
  } else {
    HAL_ADC_Start(&config->handle);
  }
}

// 停止ADC
void ADC_Stop(ADC_ConfigTypeDef *config) {
  if (config->use_dma) {
    HAL_ADC_Stop_DMA(&config->handle);
  } else {
    HAL_ADC_Stop(&config->handle);
  }
}

// ADC中断回调函数 (DMA传输完成时调用)
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
  if (hadc == &g_adc_config.handle) {
    // 数据采集完成，调用回调函数
    if (g_adc_config.callback != NULL) {
      g_adc_config.callback(g_adc_config.data_buffer, g_adc_config.buffer_size);
    }
  }
}

// 将ADC值转换为电压 (假设VREF = 3.3V)
float ADC_GetVoltage(uint16_t adc_value, float vref) {
  return (float)adc_value * vref / 4095.0f; // 12位ADC，最大值为4095
}

// ADC MSP初始化 (在HAL库内部调用)
void HAL_ADC_MspInit(ADC_HandleTypeDef *adcHandle) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if (adcHandle->Instance == ADC1) {
    // 使能ADC1时钟
    __HAL_RCC_ADC1_CLK_ENABLE();

    // 使能GPIOA时钟
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // 配置ADC引脚为模拟模式 (PA0, PA1)
    GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1; // 根据实际使用的引脚修改
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // 使能DMA1时钟 (如果使用DMA)
    __HAL_RCC_DMA1_CLK_ENABLE();

    // 配置DMA中断 (如果使用DMA)
    HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);  // 设置优先级
    HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);           // 使能中断
  }
}

// ADC MSP反初始化
void HAL_ADC_MspDeInit(ADC_HandleTypeDef *adcHandle) {
  if (adcHandle->Instance == ADC1) {
    // 禁用ADC1时钟
    __HAL_RCC_ADC1_CLK_DISABLE();

    // 禁用GPIOA时钟
    __HAL_RCC_GPIOA_CLK_DISABLE();

    // 反初始化ADC引脚
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_0 | GPIO_PIN_1); // 根据实际使用的引脚修改

    // 禁用DMA1时钟
    __HAL_RCC_DMA1_CLK_DISABLE();

    // 禁用DMA中断
    HAL_NVIC_DisableIRQ(DMA1_Channel1_IRQn);
  }
}
```

**描述:**

1.  **灵活的配置结构体:** 使用`ADC_ConfigTypeDef`结构体，将ADC的各种配置参数集中管理，方便修改和维护。  避免了硬编码，提高了代码的可读性和可移植性。
2.  **DMA支持:**  可以方便地选择是否使用DMA。  使用DMA可以显著提高ADC的采样效率，降低CPU占用率。
3.  **中断回调:**  通过回调函数，在ADC数据采集完成后执行特定任务，例如数据处理、显示等。  回调函数的机制使得ADC的使用更加灵活。
4.  **错误处理:**  使用`Error_Handler()`函数统一处理错误，方便调试。
5.  **电压转换:**  提供`ADC_GetVoltage()`函数将ADC值转换为电压，方便应用。
6.  **MSP初始化/反初始化:** `HAL_ADC_MspInit()` 和 `HAL_ADC_MspDeInit()` 函数用于初始化和反初始化与ADC相关的硬件资源，例如时钟、GPIO引脚和DMA。

**3. main.c 中的使用示例**

```c
#include "main.h"
#include "adc.h"
#include <stdio.h> // printf

// 数据缓冲区 (必须足够大，至少能容纳 buffer_size 个 uint16_t 数据)
uint16_t adc_data[24];  // 2个通道，每个通道12个采样点，总共 2*12=24 个采样点

// 数据处理回调函数
void ADC_DataReadyCallback(uint16_t *data, uint32_t size) {
  // 在这里处理ADC数据，例如计算平均值、滤波等
  // 示例：打印ADC值
  printf("ADC Data:\r\n");
  for (uint32_t i = 0; i < size; i++) {
    printf("Channel %ld: %d\r\n", (i % 2), data[i]); // 假设交替采样两个通道
  }
  printf("\r\n");
  HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin); // LED翻转，指示数据更新
}

int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init(); // 初始化GPIO (包括LED)

  // 配置ADC
  g_adc_config.handle.Instance = ADC1;
  g_adc_config.data_buffer = adc_data;
  g_adc_config.buffer_size = sizeof(adc_data) / sizeof(adc_data[0]); // 缓冲区大小
  g_adc_config.sampling_time = ADC_SAMPLETIME_239CYCLES_5;
  g_adc_config.channels[0] = ADC_CHANNEL_0; // PA0
  g_adc_config.channels[1] = ADC_CHANNEL_1; // PA1
  g_adc_config.num_channels = 2;
  g_adc_config.use_dma = true;
  g_adc_config.callback = ADC_DataReadyCallback;

  if (!ADC_Init(&g_adc_config)) {
    Error_Handler();
  }

  // 启动ADC
  ADC_Start(&g_adc_config);

  while (1) {
    // 主循环可以做其他事情，ADC数据采集在后台进行 (DMA模式)
    HAL_Delay(1000); // 延时1秒
  }
}

//  重定向printf到串口 (需要先初始化串口)
#ifdef __GNUC__
  /* With GCC/RAE, small printf (option LD Linker->Libraries->Small printf
     set to 'Yes') calls __io_putchar() */
  #define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
  #define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif /* __GNUC__ */

/**
  * @brief  Retargets the C library printf function to the USART.
  * @param  None
  * @retval None
  */
PUTCHAR_PROTOTYPE
{
  /* Place your implementation of fputc here */
  /* e.g. write a character to the USART1 and Loop until the end of transmission */
  HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xFFFF);  // huart1 是串口句柄，需要提前初始化
  return ch;
}

void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
    HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
    HAL_Delay(200);
  }
  /* USER CODE END Error_Handler_Debug */
}
```

**描述:**

1.  **配置结构体初始化:**  在 `main()` 函数中，初始化 `g_adc_config` 结构体的各个成员，包括ADC实例、数据缓冲区、采样时间、通道列表、是否使用DMA、回调函数等。
2.  **ADC初始化和启动:**  调用 `ADC_Init()` 函数初始化ADC，然后调用 `ADC_Start()` 函数启动ADC。
3.  **数据处理回调函数:**  定义 `ADC_DataReadyCallback()` 函数，用于处理ADC采集到的数据。  在这个函数中，可以对数据进行滤波、计算平均值等操作。
4.  **主循环:**  主循环可以执行其他任务，ADC数据采集在后台通过DMA完成。  每隔1秒，LED翻转一次，指示数据更新。
5. **Error Handler:**  在发生错误时，Error_Handler 会被调用，这里使用LED闪烁来指示错误。
6. **Printf重定向:** 使用Printf函数输出调试信息需要重定向到串口，确保huart1串口句柄已经正确初始化。

**4.  stm32f1xx_it.c (中断服务函数)**

```c
#include "stm32f1xx_it.h"
#include "adc.h"

extern ADC_HandleTypeDef hadc1;

void DMA1_Channel1_IRQHandler(void)
{
  HAL_DMA_IRQHandler(hadc1.DMA_Handle);
}
```

**解释:**

* **DMA1_Channel1_IRQHandler:**  DMA1通道1的中断服务函数，当DMA传输完成时，会调用此函数。
* **HAL_DMA_IRQHandler:**  调用HAL库提供的DMA中断处理函数。  此函数会调用 `HAL_ADC_ConvCpltCallback()` 函数，从而触发数据处理回调函数。

**重要的中文解释:**

*   **扫描模式 (ScanConvMode):**  如果需要采样多个通道，需要开启扫描模式。  ADC会依次采样配置的每个通道。
*   **连续转换模式 (ContinuousConvMode):**  开启后，ADC会连续不断地进行转换。  关闭后，ADC只进行一次转换。
*   **DMA (Direct Memory Access):**  直接内存访问。  使用DMA，ADC可以将采集到的数据直接传输到内存，无需CPU干预，大大提高了效率。
*   **采样时间 (SamplingTime):**  采样时间越长，精度越高，但采样速度越慢。  需要根据应用场景选择合适的采样时间。
*   **中断回调函数 (Callback):**  当ADC完成一次或多次转换后，会触发中断，并在中断处理函数中调用回调函数。  回调函数用于处理采集到的数据。
*   **错误处理 (Error Handling):**  在发生错误时，需要进行适当的处理，例如停止ADC、记录错误信息、重启系统等。

**总结:**

这个改进后的ADC配置代码更加灵活、易于维护，并加入了一些常用的优化技巧。  通过配置结构体、DMA支持、中断回调函数等，可以方便地实现各种ADC应用。  记住，需要根据你的具体硬件和应用场景调整配置参数。  确保正确配置了GPIO、DMA和中断。  最后，通过printf函数打印ADC数据，并使用LED指示数据更新，可以帮助你调试代码。
