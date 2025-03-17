Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_adc_ex.c`

好的，我们重新开始，对 STM32F1xx HAL ADC 扩展库 (stm32f1xx_hal_adc_ex.c) 提供更智能的改进建议，并附带中文描述和 Demo 示例。

**目标:**  在保证代码功能不变的前提下，提高代码的可读性、可维护性和效率。

**改进方向:**

1.  **简化宏定义和常量:**  对于重复使用的计算，使用宏或内联函数进行封装。
2.  **代码结构优化:**  将相关的代码段组织在一起，例如将标志位清除操作放在一起。
3.  **错误处理增强:**  添加更详细的错误信息，方便调试。
4.  **减少冗余代码:**  去除不必要的代码，例如重复的参数检查。
5.  **优化多 ADC 同步 (Multimode) 的代码:**  简化多 ADC 同步的配置和启动过程。
6.  **添加注释:**  对关键代码段添加更详细的注释，解释代码的功能和实现原理。

**示例改进 (HAL_ADCEx_Calibration_Start 函数):**

```c
/**
  * @brief  执行 ADC 自动自校准。
  *         校准前提：ADC 必须处于禁用状态（在 HAL_ADC_Start() 之前或 HAL_ADC_Stop() 之后执行此函数）。
  *         在校准过程中，ADC 会被启用。函数完成时，ADC 保持启用状态。
  * @param  hadc: ADC 句柄
  * @retval HAL 状态
  */
HAL_StatusTypeDef HAL_ADCEx_Calibration_Start(ADC_HandleTypeDef* hadc) {
    HAL_StatusTypeDef tmp_hal_status = HAL_OK;
    uint32_t tickstart;
    __IO uint32_t wait_loop_index = 0U;

    /* 参数检查 */
    if (hadc == NULL) {
        return HAL_ERROR; // 添加空指针检查
    }
    assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

    /* 进程锁定 */
    __HAL_LOCK(hadc);

    /* 1. 校准前提条件：ADC 必须禁用至少两个 ADC 时钟周期 */
    /* 停止所有正在进行的转换，禁用 ADC 外设 */
    tmp_hal_status = ADC_ConversionStop_Disable(hadc);

    /* 检查 ADC 是否已成功禁用 */
    if (tmp_hal_status == HAL_OK) {
        /* 设置 ADC 状态 */
        ADC_STATE_CLR_SET(hadc->State,
                          HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                          HAL_ADC_STATE_BUSY_INTERNAL);

        /* 硬件要求：在开始校准之前需要延迟 */
        /* 计算对应于 ADC 时钟周期的 CPU 时钟周期数 */
        wait_loop_index = ((SystemCoreClock / HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_ADC)) *
                           ADC_PRECALIBRATION_DELAY_ADCCLOCKCYCLES);

        while (wait_loop_index != 0U) {
            wait_loop_index--;
        }

        /* 2. 启用 ADC 外设 */
        ADC_Enable(hadc);

        /* 3. 复位 ADC 校准寄存器 */
        SET_BIT(hadc->Instance->CR2, ADC_CR2_RSTCAL);

        tickstart = HAL_GetTick();

        /* 等待校准复位完成 */
        while (HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_RSTCAL)) {
            if ((HAL_GetTick() - tickstart) > ADC_CALIBRATION_TIMEOUT) {
                /* 超时检查，避免 preempt 导致错误 */
                if (HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_RSTCAL)) {
                    /* 更新 ADC 状态机为错误 */
                    ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_BUSY_INTERNAL,
                                      HAL_ADC_STATE_ERROR_INTERNAL);

                    /* 进程解锁 */
                    __HAL_UNLOCK(hadc);
                    return HAL_ERROR;
                }
            }
        }

        /* 4. 启动 ADC 校准 */
        SET_BIT(hadc->Instance->CR2, ADC_CR2_CAL);

        tickstart = HAL_GetTick();

        /* 等待校准完成 */
        while (HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_CAL)) {
            if ((HAL_GetTick() - tickstart) > ADC_CALIBRATION_TIMEOUT) {
                /* 超时检查，避免 preempt 导致错误 */
                if (HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_CAL)) {
                    /* 更新 ADC 状态机为错误 */
                    ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_BUSY_INTERNAL,
                                      HAL_ADC_STATE_ERROR_INTERNAL);

                    /* 进程解锁 */
                    __HAL_UNLOCK(hadc);
                    return HAL_ERROR;
                }
            }
        }

        /* 设置 ADC 状态 */
        ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_BUSY_INTERNAL, HAL_ADC_STATE_READY);
    }

    /* 进程解锁 */
    __HAL_UNLOCK(hadc);

    /* 返回函数状态 */
    return tmp_hal_status;
}
```

**中文描述:**

*   **参数检查:**  添加了对 `hadc` 指针的空指针检查，避免空指针访问。
*   **注释增强:**  对每个步骤添加了更详细的中文注释，解释代码的功能和实现原理。
*   **错误处理:**  在超时检测中，保留了原有的再次检查的机制，避免因系统抢占导致的误判。
*   **状态机:** 使用 ADC_STATE_CLR_SET 宏，清晰的表明状态的改变

**Demo 示例 (配合 STM32CubeIDE):**

1.  **创建 STM32CubeIDE 工程:**  创建一个基于 STM32F103 的 STM32CubeIDE 工程。
2.  **配置 ADC:**  使用 STM32CubeMX 配置 ADC，选择一个通道，设置时钟和中断。
3.  **添加代码:**  在 `main.c` 文件中添加以下代码：

```c
#include "main.h"

ADC_HandleTypeDef hadc1;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_ADC1_Init();

    /*  校准 ADC */
    if (HAL_ADCEx_Calibration_Start(&hadc1) != HAL_OK) {
        Error_Handler(); // 处理校准失败的情况
    }

    /* 启动 ADC 转换 */
    if (HAL_ADC_Start(&hadc1) != HAL_OK) {
        Error_Handler(); // 处理启动失败的情况
    }

    while (1) {
        /*  轮询等待转换完成 */
        HAL_ADC_PollForConversion(&hadc1, 10);
        /* 读取 ADC 值 */
        uint32_t adc_value = HAL_ADC_GetValue(&hadc1);

        /*  打印 ADC 值 (需要配置 UART) */
        printf("ADC Value: %lu\r\n", adc_value);

        HAL_Delay(100);
    }
}

static void MX_ADC1_Init(void) {
  ADC_ChannelConfTypeDef sConfig = {0};

  hadc1.Instance = ADC1;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

}

void Error_Handler(void){
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
```

**注意:**

*   需要根据实际硬件连接修改 `ADC_CHANNEL_0`。
*   需要在 STM32CubeIDE 中配置 UART，才能使用 `printf` 函数。
*   `SystemClock_Config()` 和 `MX_GPIO_Init()` 函数由 STM32CubeIDE 自动生成。

**Multimode 示例 (基于 STM32F103xE 假设有两个 ADC):**

```c
#include "main.h"

ADC_HandleTypeDef hadc1;
ADC_HandleTypeDef hadc2;
DMA_HandleTypeDef hdma_adc1;
uint32_t ADC_values[2];  // 保存 ADC1 和 ADC2 的值

static void MX_ADC1_Init(void);
static void MX_ADC2_Init(void);
static void MX_DMA_Init(void);
static void HAL_ADC_MspInit(ADC_HandleTypeDef* adcHandle);  // 修正后的声明
static void HAL_ADC_MspDeInit(ADC_HandleTypeDef* adcHandle);

int main(void) {
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();

  ADC_MultiModeTypeDef multimode_config = {0};

  // 配置 ADC1 为 Master，Dual Mode
  multimode_config.Mode = ADC_MODE_DUAL_REG_SIMULT; //或者ADC_MODE_DUAL_REG_INTERL
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode_config) != HAL_OK)
  {
    Error_Handler();
  }

  // 启动 Multimode DMA
  if (HAL_ADCEx_MultiModeStart_DMA(&hadc1, ADC_values, 2) != HAL_OK)
  {
    Error_Handler();
  }

  while (1) {
    // ADC_values[0] 包含 ADC1 的值，ADC_values[1] 包含 ADC2 的值
    printf("ADC1 Value: %lu, ADC2 Value: %lu\r\n", ADC_values[0], ADC_values[1]);
    HAL_Delay(100);
  }
}


// 修正后的HAL_ADC_MspInit函数
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
    PA0     ------> ADC1_IN0
    */
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* ADC1 DMA Init */
    /* ADC1 Init */
    hdma_adc1.Instance = DMA1_Channel1;
    hdma_adc1.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_adc1.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_adc1.Init.MemInc = DMA_MINC_ENABLE;
    hdma_adc1.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
    hdma_adc1.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
    hdma_adc1.Init.Mode = DMA_CIRCULAR;
    hdma_adc1.Init.Priority = DMA_PRIORITY_LOW;
    if (HAL_DMA_Init(&hdma_adc1) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(adcHandle,DMA_Handle,hdma_adc1);

  /* USER CODE BEGIN ADC1_MspInit 1 */

  /* USER CODE END ADC1_MspInit 1 */
  }
  else if(adcHandle->Instance==ADC2)
  {
    /* Peripheral clock enable */
    __HAL_RCC_ADC2_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**ADC2 GPIO Configuration
    PA1     ------> ADC2_IN1
    */
    GPIO_InitStruct.Pin = GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  }
}

//修正后的HAL_ADC_MspDeInit函数
void HAL_ADC_MspDeInit(ADC_HandleTypeDef* adcHandle)
{
  if(adcHandle->Instance==ADC1)
  {
  /* USER CODE BEGIN ADC1_MspDeInit 0 */

  /* USER CODE END ADC1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_ADC1_CLK_DISABLE();

    /**ADC1 GPIO Configuration
    PA0     ------> ADC1_IN0
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_0);

    /* ADC1 DMA DeInit */
    HAL_DMA_DeInit(adcHandle->DMA_Handle);
  /* USER CODE BEGIN ADC1_MspDeInit 1 */

  /* USER CODE END ADC1_MspDeInit 1 */
  }
  else if(adcHandle->Instance==ADC2)
  {
    /* Peripheral clock disable */
    __HAL_RCC_ADC2_CLK_DISABLE();

    /**ADC2 GPIO Configuration
    PA1     ------> ADC2_IN1
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_1);
  }
}

static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

static void MX_ADC2_Init(void)
{

  /* USER CODE BEGIN ADC2_Init 0 */

  /* USER CODE END ADC2_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC2_Init 1 */

  /* USER CODE END ADC2_Init 1 */

  /** Common config
  */
  hadc2.Instance = ADC2;
  hadc2.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc2.Init.ContinuousConvMode = DISABLE;
  hadc2.Init.DiscontinuousConvMode = DISABLE;
  hadc2.Init.ExternalTrigConv = ADC_SOFTWARE_START;  // 必须为软件触发
  hadc2.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc2.Init.NbrOfConversion = 1;
  if (HAL_ADC_Init(&hadc2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_1;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC2_Init 2 */

  /* USER CODE END ADC2_Init 2 */

}

static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);

}
```

**重要提示:**

*   **硬件支持:** 确保你的 STM32F103 芯片有两个 ADC，并且引脚配置正确。
*   **DMA 配置:** DMA 的配置非常关键，需要仔细检查 DMA 的通道、方向、增量模式和数据对齐方式。
*   **ADC2 软件触发:**  ADC2 (Slave) 必须配置为软件触发，这是 STM32F1 的硬件限制。
*   **HAL_ADC_MspInit 修改:**  务必按照代码示例修改 `HAL_ADC_MspInit` 函数，配置 ADC1 和 ADC2 的 GPIO 和 DMA。  否则程序不能正常运行
*   **HAL_ADC_MspDeInit  修改:**  务必按照代码示例修改 `HAL_ADC_MspDeInit` 函数，配置 ADC1 和 ADC2 的 GPIO 和 DMA。  否则程序不能正常运行

**中文描述:**

*   **配置 Master/Slave:**  将 ADC1 配置为 Master，ADC2 配置为 Slave。
*   **Multimode 配置:**  使用 `ADC_MODE_DUAL_REG_SIMULT` 模式，同时采样 ADC1 和 ADC2。
*   **DMA 传输:**  使用 DMA 将 ADC1 和 ADC2 的采样结果存储到 `ADC_values` 数组中。
*    **HAL_ADC_MspInit 修改:**  务必按照代码示例修改 `HAL_ADC_MspInit` 函数，配置 ADC1 和 ADC2 的 GPIO 和 DMA。  否则程序不能正常运行
*   **HAL_ADC_MspDeInit 修改:**  务必按照代码示例修改 `HAL_ADC_MspDeInit` 函数，配置 ADC1 和 ADC2 的 GPIO 和 DMA。  否则程序不能正常运行

这个示例代码能够演示如何使用 Multimode DMA 功能，同时读取 ADC1 和 ADC2 的值，并通过串口打印出来。
