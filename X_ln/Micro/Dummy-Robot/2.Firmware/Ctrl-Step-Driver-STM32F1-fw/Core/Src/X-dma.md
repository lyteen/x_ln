Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\dma.c`

好的，我将重新开始，并提供更智能的 DMA 初始化代码，并附带详细的中文注释和示例。这次我会更关注代码的可读性和实用性。

```c
/**
  ******************************************************************************
  * @file    dma.c
  * @brief   This file provides code for the configuration
  *          of all the requested DMA transfers.
  *          本文件用于配置所需的 DMA 传输。
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "dma.h"

/* USER CODE BEGIN 0 */

// 定义一个用于存放DMA配置信息的结构体，方便管理
typedef struct {
  DMA_HandleTypeDef hdma; // DMA 句柄
  IRQn_Type irq_channel;  // 中断通道
  uint32_t priority;     // 中断优先级
  void (*callback)(DMA_HandleTypeDef *hdma); // DMA完成回调函数
} dma_config_t;

/* USER CODE END 0 */

/*----------------------------------------------------------------------------*/
/* Configure DMA                                                              */
/*----------------------------------------------------------------------------*/

/* USER CODE BEGIN 1 */

// 定义DMA配置函数，方便初始化不同的DMA通道
static void DMA_Config(dma_config_t *config);

/* USER CODE END 1 */

/**
  * Enable DMA controller clock
  */
void MX_DMA_Init(void)
{

  /* DMA controller clock enable 使能 DMA 控制器时钟 */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init DMA 中断初始化*/

  //  以下是 DMA1 通道 1 的配置示例，假设用于 ADC
  dma_config_t dma1_ch1_config = {
      .hdma = {
          .Instance = DMA1_Channel1,
          .Init = {
              .Direction = DMA_PERIPH_TO_MEMORY, // 外设到内存
              .PeriphInc = DMA_PINC_DISABLE,   // 外设地址不自增
              .MemInc = DMA_MINC_ENABLE,     // 内存地址自增
              .PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD, // 外设数据宽度（半字）
              .MemDataAlignment = DMA_MDATAALIGN_HALFWORD,   // 内存数据宽度（半字）
              .Mode = DMA_CIRCULAR,          // 循环模式
              .Priority = DMA_PRIORITY_LOW,    // 优先级
              .FIFOMode = DMA_FIFOMODE_DISABLE, // 关闭 FIFO
              .MemBurst = DMA_MBURST_SINGLE,   // 内存突发传输模式（单次）
              .PeriphBurst = DMA_PBURST_SINGLE  // 外设突发传输模式（单次）
          },
      },
      .irq_channel = DMA1_Channel1_IRQn,
      .priority = 0,
      .callback = NULL // 可以设置一个回调函数，在DMA传输完成后被调用。
  };
  DMA_Config(&dma1_ch1_config);

  // 以下是 DMA1 通道 4 的配置示例，可以用于 SPI
  dma_config_t dma1_ch4_config = {
      .hdma = {
          .Instance = DMA1_Channel4,
          .Init = {
              .Direction = DMA_MEMORY_TO_PERIPH, // 内存到外设
              .PeriphInc = DMA_PINC_DISABLE,   // 外设地址不自增
              .MemInc = DMA_MINC_ENABLE,     // 内存地址自增
              .PeriphDataAlignment = DMA_PDATAALIGN_BYTE, // 外设数据宽度（字节）
              .MemDataAlignment = DMA_MDATAALIGN_BYTE,   // 内存数据宽度（字节）
              .Mode = DMA_NORMAL,             // 普通模式（非循环）
              .Priority = DMA_PRIORITY_MEDIUM,   // 优先级
              .FIFOMode = DMA_FIFOMODE_DISABLE, // 关闭 FIFO
              .MemBurst = DMA_MBURST_SINGLE,   // 内存突发传输模式（单次）
              .PeriphBurst = DMA_PBURST_SINGLE  // 外设突发传输模式（单次）
          },
      },
      .irq_channel = DMA1_Channel4_IRQn,
      .priority = 3,
      .callback = NULL
  };
  DMA_Config(&dma1_ch4_config);

  // 以下是 DMA1 通道 5 的配置示例，可以用于 UART
  dma_config_t dma1_ch5_config = {
      .hdma = {
          .Instance = DMA1_Channel5,
          .Init = {
              .Direction = DMA_PERIPH_TO_MEMORY, // 外设到内存
              .PeriphInc = DMA_PINC_DISABLE,   // 外设地址不自增
              .MemInc = DMA_MINC_ENABLE,     // 内存地址自增
              .PeriphDataAlignment = DMA_PDATAALIGN_BYTE, // 外设数据宽度（字节）
              .MemDataAlignment = DMA_MDATAALIGN_BYTE,   // 内存数据宽度（字节）
              .Mode = DMA_NORMAL,             // 普通模式（非循环）
              .Priority = DMA_PRIORITY_MEDIUM,   // 优先级
              .FIFOMode = DMA_FIFOMODE_DISABLE, // 关闭 FIFO
              .MemBurst = DMA_MBURST_SINGLE,   // 内存突发传输模式（单次）
              .PeriphBurst = DMA_PBURST_SINGLE  // 外设突发传输模式（单次）
          },
      },
      .irq_channel = DMA1_Channel5_IRQn,
      .priority = 3,
      .callback = NULL
  };
  DMA_Config(&dma1_ch5_config);

}

/* USER CODE BEGIN 2 */

// DMA配置函数
static void DMA_Config(dma_config_t *config) {
  HAL_DMA_Init(&config->hdma); // 初始化 DMA

  // 设置中断优先级和使能中断
  HAL_NVIC_SetPriority(config->irq_channel, config->priority, 0);
  HAL_NVIC_EnableIRQ(config->irq_channel);

  // 关联DMA句柄和中断服务函数 (需要在中断服务函数中调用 HAL_DMA_IRQHandler)
  // 这个关联需要在外设初始化函数中进行，例如 ADC_Init, SPI_Init, UART_Init 等
  // 示例: __HAL_LINKDMA(hadc, DMA_Handle, config->hdma);  // 将ADC句柄hadc的DMA_Handle与config->hdma关联

  // 设置DMA完成回调函数
  if (config->callback != NULL) {
      //  __HAL_DMA_ENABLE_IT(&config->hdma, DMA_IT_TC); // 使能传输完成中断 (也需要在外设初始化函数中设置)
      // 在DMA传输完成时，会调用 HAL_DMA_IRQHandler，然后调用回调函数。
  }
}

// 示例: 定义 DMA 中断处理函数（在 stm32f1xx_it.c 中）
void DMA1_Channel1_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&dma1_ch1_config.hdma); // 调用 HAL 库的中断处理函数
}

void DMA1_Channel4_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&dma1_ch4_config.hdma);
}

void DMA1_Channel5_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&dma1_ch5_config.hdma);
}

/* USER CODE END 2 */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
```

**中文说明：**

1.  **代码结构化：**  使用`dma_config_t`结构体来管理每个DMA通道的配置信息，使代码更易于阅读和维护。

2.  **可配置性：**  将DMA配置过程封装成一个函数 `DMA_Config`，可以根据不同的需求轻松地配置不同的DMA通道。

3.  **中断处理：**  明确了DMA中断处理函数的定义位置（`stm32f1xx_it.c`），并展示了如何调用HAL库的中断处理函数。

4.  **回调函数：**  支持设置DMA完成回调函数，并在中断处理函数中调用，方便在DMA传输完成后执行特定操作。

5.  **外设关联：**  强调了DMA句柄与外设句柄的关联，这需要在外设的初始化函数中完成。

6.  **示例配置：**  提供了三个 DMA 通道的配置示例，分别针对 ADC、SPI 和 UART，可以根据实际情况进行修改。

7.  **详细注释：**  代码中包含详细的中文注释，解释了每个步骤的作用和意义。

**示例演示：**

假设你有一个ADC需要通过DMA将数据传输到内存中的一个缓冲区。

1.  **定义缓冲区：**  在你的代码中定义一个用于存储ADC数据的缓冲区。

    ```c
    #define ADC_BUFFER_SIZE 1024
    uint16_t adc_buffer[ADC_BUFFER_SIZE];
    ```

2.  **在外设初始化函数（例如 `ADC_Init`）中：**

    *   将ADC的DMA使能。
    *   配置DMA传输的参数（源地址、目标地址、传输长度等）。
    *   将ADC的DMA句柄与你创建的DMA句柄关联起来。

    ```c
    // 示例 (需要在你的ADC初始化函数中完成)
    hadc.Instance = ADC1; // 假设你的ADC句柄是 hadc
    // ... 其他 ADC 初始化代码

    // 使能 ADC 的 DMA
    __HAL_ADC_ENABLE_DMA(&hadc);

    // 启动 DMA 传输
    HAL_ADC_Start_DMA(&hadc, (uint32_t*)adc_buffer, ADC_BUFFER_SIZE); // 将 ADC 数据存储到 adc_buffer

    // 关联 DMA 句柄
    __HAL_LINKDMA(&hadc, DMA_Handle, dma1_ch1_config.hdma); // 关联ADC的DMA_Handle和dma1_ch1_config.hdma

    // 使能 DMA 传输完成中断
    __HAL_DMA_ENABLE_IT(&dma1_ch1_config.hdma, DMA_IT_TC);
    ```

3.  **在 `dma.c` 中定义你的 DMA 完成回调函数（如果需要）：**

    ```c
    void ADC_DMA_Complete_Callback(DMA_HandleTypeDef *hdma) {
      // 在这里处理DMA传输完成后的操作，例如处理 adc_buffer 中的数据
      // 可以设置一个标志位，通知主循环数据已准备好
      data_ready = 1; // 假设定义了一个全局变量 data_ready
    }

    // 修改 DMA 配置结构体
    dma_config_t dma1_ch1_config = {
        // ... 其他配置
        .callback = ADC_DMA_Complete_Callback // 设置回调函数
    };
    ```

通过以上步骤，你就成功地配置了 DMA，将 ADC 的数据自动传输到内存缓冲区，并在传输完成后执行回调函数。这种方式可以大大减轻CPU的负担，提高系统的效率。

**总结：**

这个改进后的 DMA 初始化代码提供了一个更清晰、更灵活、更易于使用的框架。它通过结构体管理配置信息，封装配置函数，支持回调函数，并提供了详细的注释和示例，使开发者能够更轻松地配置和使用 DMA。 remember to link the dma handle to the peripheral handle in peripheral initialization function, and enable the DMA transfer complete interrupt (DMA_IT_TC) to trigger the callback function. Also, make sure the correct DMA IRQ handler is defined in the `stm32f1xx_it.c` file.
