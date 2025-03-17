Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\adc.c`

好的，我们重新开始，详细解释这份 ADC 的配置代码，并提供带中文解释的片段和使用示例。

**代码总体概述:**

这段代码是为 STM32 微控制器的 ADC1 (模数转换器) 实例进行配置的。它初始化 ADC，配置通道，并使用 DMA (直接内存访问) 来传输转换后的数据。以下是对代码主要部分的详细解释，并附带中文注释、使用方法和简单演示。

**1. 头文件包含:**

```c
#include "adc.h"
```

*   **解释:** 包含 `adc.h` 头文件，该文件很可能包含 ADC 相关的定义、结构体和函数声明。
*   **使用:** 这是标准的 C 代码包含头文件的方式，必须包含。

**2. 用户代码段 (USER CODE BEGIN/END):**

```c
/* USER CODE BEGIN 0 */
uint16_t whole_adc_data[2][12];

/* USER CODE END 0 */
```

*   **解释:**  `USER CODE BEGIN/END` 标记用于在生成代码时，允许用户插入自定义代码，而不会被代码生成器覆盖。 在这里，`whole_adc_data` 是一个二维数组，用于存储 ADC 转换后的数据。 `[2][12]` 表示有两个缓冲区，每个缓冲区可以存储 12 个 ADC 通道的数据（虽然实际上代码只配置了2个通道，这个数组大小可能预留给将来扩展使用）。`uint16_t` 表示数据类型是16位无符号整数。
*   **使用:**  你可以在 `USER CODE BEGIN 0` 和 `USER CODE END 0` 之间添加变量声明、函数定义或其他自定义代码。
*   **中文解释:**  `/* 用户代码开始 0 */` 和 `/* 用户代码结束 0 */`  之间的代码是你可以自定义添加的，不会被自动生成的代码覆盖。`whole_adc_data` 是一个用来存放 ADC 转换结果的数组。

**3. ADC 和 DMA 句柄:**

```c
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;
```

*   **解释:**  `ADC_HandleTypeDef` 和 `DMA_HandleTypeDef` 是结构体类型，用于存储 ADC1 和 DMA 通道的配置信息。  `hadc1` 是 ADC1 的句柄，`hdma_adc1` 是 DMA 的句柄。
*   **使用:**  这些句柄在后面的 ADC 和 DMA 初始化函数中使用。
*   **中文解释:**  `ADC_HandleTypeDef` 和 `DMA_HandleTypeDef` 就像是 ADC1 和 DMA 的“身份证”，记录了它们的配置信息。

**4. ADC1 初始化函数 (`MX_ADC1_Init`):**

```c
void MX_ADC1_Init(void)
{
  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */
  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 2;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_239CYCLES_5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_1;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */
  HAL_ADCEx_Calibration_Start(&hadc1);
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)&whole_adc_data[0][0], 2);

  /* USER CODE END ADC1_Init 2 */

}
```

*   **解释:**
    *   `hadc1.Instance = ADC1;`:  指定使用 ADC1 实例。
    *   `hadc1.Init.ScanConvMode = ADC_SCAN_ENABLE;`: 启用扫描模式，ADC 会自动依次转换配置的通道。
    *   `hadc1.Init.ContinuousConvMode = ENABLE;`: 启用连续转换模式，ADC 会一直进行转换。
    *   `hadc1.Init.DiscontinuousConvMode = DISABLE;`: 禁用不连续转换模式。
    *   `hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;`:  使用软件触发启动转换。
    *   `hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;`:  数据右对齐。
    *   `hadc1.Init.NbrOfConversion = 2;`:  配置转换通道数为2。
    *   `HAL_ADC_Init(&hadc1);`: 使用配置信息初始化 ADC。
    *   `sConfig.Channel = ADC_CHANNEL_0;` 和 `sConfig.Channel = ADC_CHANNEL_1;`:  配置 ADC 通道 0 和通道 1。
    *   `sConfig.Rank = ADC_REGULAR_RANK_1;` 和 `sConfig.Rank = ADC_REGULAR_RANK_2;`:  设置通道的转换顺序。
    *   `sConfig.SamplingTime = ADC_SAMPLETIME_239CYCLES_5;`:  设置采样时间（影响转换精度和速度）。
    *   `HAL_ADC_ConfigChannel(&hadc1, &sConfig);`:  配置 ADC 通道。
    *   `HAL_ADCEx_Calibration_Start(&hadc1);`: 启动 ADC 校准。
    *   `HAL_ADC_Start_DMA(&hadc1, (uint32_t*)&whole_adc_data[0][0], 2);`:  启动 DMA 传输，将 ADC 转换后的数据存储到 `whole_adc_data` 数组中。`2` 代表传输 2 个数据（对应配置的2个转换通道）。
*   **使用:**  这个函数必须在程序启动时调用一次，用于初始化 ADC。
*   **中文解释:**  这个函数是配置 ADC1 的核心。它设置了 ADC 的各种工作模式，包括扫描模式、连续转换模式，以及转换通道。 `HAL_ADC_Start_DMA` 函数启动 DMA，使 ADC 转换的数据能够自动存储到 `whole_adc_data` 数组中，而不需要 CPU 干预。

**5. ADC MSP 初始化函数 (`HAL_ADC_MspInit`):**

```c
void HAL_ADC_MspInit(ADC_HandleTypeDef* adcHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(adcHandle->Instance==ADC1)
  {
  /* USER CODE BEGIN ADC1_MspInit 0 */

  /* USER CODE END ADC1_MspInit 0 */
    /* ADC1 clock enable */
    __HAL_RCC_ADC1_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**ADC1 GPIO Configuration
    PA0-WKUP     ------> ADC1_IN0
    PA1     ------> ADC1_IN1
    */
    GPIO_InitStruct.Pin = POWER_U_Pin|DRV_TEMP_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* ADC1 DMA Init */
    /* ADC1 Init */
    hdma_adc1.Instance = DMA1_Channel1;
    hdma_adc1.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_adc1.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_adc1.Init.MemInc = DMA_MINC_ENABLE;
    hdma_adc1.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    hdma_adc1.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
    hdma_adc1.Init.Mode = DMA_CIRCULAR;
    hdma_adc1.Init.Priority = DMA_PRIORITY_VERY_HIGH;
    if (HAL_DMA_Init(&hdma_adc1) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(adcHandle,DMA_Handle,hdma_adc1);

  /* USER CODE BEGIN ADC1_MspInit 1 */

  /* USER CODE END ADC1_MspInit 1 */
  }
}
```

*   **解释:**
    *   `__HAL_RCC_ADC1_CLK_ENABLE();`:  使能 ADC1 的时钟。
    *   `__HAL_RCC_GPIOA_CLK_ENABLE();`: 使能 GPIOA 的时钟，因为 ADC 的输入引脚通常连接到 GPIOA。
    *   `GPIO_InitStruct.Pin = POWER_U_Pin|DRV_TEMP_Pin;`:  配置 PA0 和 PA1 引脚为模拟输入模式 (`GPIO_MODE_ANALOG`)。`POWER_U_Pin` 和 `DRV_TEMP_Pin` 应该是定义在头文件中的宏，对应 PA0 和 PA1。
    *   `HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);`: 初始化 GPIO 引脚。
    *   配置 DMA (DMA1_Channel1) 将 ADC 数据传输到内存。  `DMA_PERIPH_TO_MEMORY` 表示数据从外设（ADC）传输到内存。 `DMA_CIRCULAR` 表示循环模式，DMA 会一直传输数据。
    *   `__HAL_LINKDMA(adcHandle,DMA_Handle,hdma_adc1);`:  将 DMA 句柄链接到 ADC 句柄。
*   **使用:**  这个函数由 HAL 库调用，用于初始化 ADC 的底层硬件资源，如时钟、GPIO 和 DMA。
*   **中文解释:**  这个函数负责配置 ADC1 的外围硬件，包括开启时钟、设置 GPIO 引脚为模拟输入模式，以及配置 DMA 通道。 DMA 的配置非常重要，因为它允许 ADC 在没有 CPU 干预的情况下，自动将转换后的数据存储到内存中。

**6. ADC MSP 反初始化函数 (`HAL_ADC_MspDeInit`):**

```c
void HAL_ADC_MspDeInit(ADC_HandleTypeDef* adcHandle)
{

  if(adcHandle->Instance==ADC1)
  {
  /* USER CODE BEGIN ADC1_MspDeInit 0 */

  /* USER CODE END ADC1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_ADC1_CLK_DISABLE();

    /**ADC1 GPIO Configuration
    PA0-WKUP     ------> ADC1_IN0
    PA1     ------> ADC1_IN1
    */
    HAL_GPIO_DeInit(GPIOA, POWER_U_Pin|DRV_TEMP_Pin);

    /* ADC1 DMA DeInit */
    HAL_DMA_DeInit(adcHandle->DMA_Handle);
  /* USER CODE BEGIN ADC1_MspDeInit 1 */

  /* USER CODE END ADC1_MspDeInit 1 */
  }
}
```

*   **解释:**  这个函数用于反初始化 ADC 的硬件资源，例如禁用时钟、取消 GPIO 配置和禁用 DMA。
*   **使用:**  在不需要使用 ADC 时，可以调用此函数来释放资源。
*   **中文解释:**  与 `HAL_ADC_MspInit` 相反，这个函数负责关闭 ADC1 的外围硬件，例如关闭时钟、将 GPIO 引脚恢复默认状态，以及停止 DMA 通道。

**7. 用户代码段 (USER CODE BEGIN/END):**

```c
/* USER CODE BEGIN 1 */

/* USER CODE END 1 */
```

*   **解释:**  另一个用户代码段，你可以在这里添加自定义代码。
*   **使用:**  例如，你可以在这里添加读取 `whole_adc_data` 数组并进行处理的代码。
*   **中文解释:**  `/* 用户代码开始 1 */` 和 `/* 用户代码结束 1 */` 之间的代码是你可以自定义添加的，不会被自动生成的代码覆盖。

**如何使用这段代码和简单演示:**

1.  **包含文件:** 确保 `adc.c` 和 `adc.h` 文件包含在你的 STM32 工程中。
2.  **初始化:** 在你的 `main.c` 文件中，调用 `MX_ADC1_Init()` 函数来初始化 ADC1。
3.  **读取 ADC 数据:**  在主循环或其他地方，你可以访问 `whole_adc_data` 数组来读取 ADC 转换后的数据。

**简单演示 (main.c):**

```c
#include "main.h"
#include "adc.h"  // 包含 ADC 驱动头文件
#include <stdio.h> // 用于 printf

extern uint16_t whole_adc_data[2][12]; // 声明在 adc.c 中定义的全局变量

int main(void)
{
  HAL_Init();  // 初始化 HAL 库
  SystemClock_Config(); // 配置系统时钟

  MX_GPIO_Init();   // 初始化 GPIO (假设你有一个 GPIO 初始化函数)
  MX_ADC1_Init();   // 初始化 ADC1

  /* Infinite loop */
  while (1)
  {
    // 简单延时
    HAL_Delay(100);

    // 打印 ADC 通道 0 和通道 1 的数据
    printf("ADC Channel 0: %d, ADC Channel 1: %d\r\n", whole_adc_data[0][0], whole_adc_data[0][1]);
  }
}
```

**注意事项:**

*   **错误处理:**  代码中使用了 `Error_Handler()` 函数，你应该实现这个函数来处理初始化错误。
*   **时钟配置:**  确保正确配置了系统时钟，ADC 需要一个稳定的时钟源。
*   **GPIO 配置:**  确保正确配置了连接到 ADC 输入引脚的 GPIO 引脚。
*   **printf:**  要使用 `printf` 函数，你需要配置 UART（或其它输出方式）并将 `stdout` 重定向到 UART。

**总结:**

这段代码提供了一个完整的 STM32 ADC1 配置示例，使用了 DMA 传输，可以高效地获取模拟信号的数字值。  通过理解代码的各个部分，你可以根据自己的需要进行修改和扩展。 请务必查看 STM32 的参考手册，以获得更详细的信息。
