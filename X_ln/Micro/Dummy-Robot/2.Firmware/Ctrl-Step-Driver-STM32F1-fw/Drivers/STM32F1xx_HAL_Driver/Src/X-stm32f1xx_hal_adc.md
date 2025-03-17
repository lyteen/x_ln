Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_adc.c`

好的，我们重新开始，针对STM32F1xx HAL库的ADC驱动，我将提供更智能化的代码示例，并附带中文解释和简单的演示。

**目标：**

1.  **代码更清晰、模块化：** 将大函数拆分成更小的、易于理解和维护的函数。
2.  **错误处理更完善：**  在关键步骤增加错误检查，并提供更详细的错误信息。
3.  **可配置性更高：**  使用宏或结构体来实现更多的可配置选项。
4.  **演示更实用：**  提供一个简单的示例，展示如何使用ADC读取模拟信号。

**改进策略：**

*   **ADC时钟配置：** 将时钟配置部分从`HAL_ADC_Init`中分离出来，形成独立的`ADC_ClockConfig`函数，方便复用和修改。
*   **ADC状态管理：**  细化状态管理，增加更多的状态检查，避免在错误状态下进行操作。
*   **通道配置：**  提供更灵活的通道配置方式，例如允许一次配置多个通道。
*   **DMA配置：**  简化DMA配置，提供默认的DMA配置选项，方便快速上手。
*   **中断处理：**  改进中断处理函数，确保在发生错误时能够及时处理。

**1. ADC时钟配置函数 (ADC Clock Configuration Function)**

```c
/**
 * @brief 配置ADC的时钟。
 *
 * @param hadc ADC句柄
 * @param clockSource 时钟源 (例如：RCC_ADCPCLK2_DIV2, RCC_ADCPCLK2_DIV4, ...)。
 * @retval HAL状态
 */
HAL_StatusTypeDef ADC_ClockConfig(ADC_HandleTypeDef *hadc, uint32_t clockSource) {
  RCC_PeriphCLKInitTypeDef  PeriphClkInit;

  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  assert_param(IS_RCC_ADCPCLK2(clockSource)); // 确保时钟源是有效值

  __HAL_RCC_ADC1_CLK_ENABLE(); // 总是使能ADC1时钟，可以根据实际情况修改

  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInit.AdcClockSelection = clockSource;

  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK) {
    /* 时钟配置失败 */
    return HAL_ERROR;
  }

  return HAL_OK;
}

/* 使用示例 (Example Usage): 在 HAL_ADC_MspInit 中调用 */
void HAL_ADC_MspInit(ADC_HandleTypeDef* hadc) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(hadc->Instance==ADC1)
  {
    /* ADC1 clock enable */
    __HAL_RCC_ADC1_CLK_ENABLE();

    /* 使能GPIOA时钟 */
    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* 配置ADC引脚PA0为模拟输入 */
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

	/*配置ADC时钟*/
	ADC_ClockConfig(hadc, RCC_ADCPCLK2_DIV6);
  }
}
/*中文描述：
这段代码将ADC时钟配置从HAL_ADC_Init函数中分离出来，形成一个独立的函数ADC_ClockConfig。这样做的好处是可以更方便地重用时钟配置代码，并且可以更容易地修改时钟源。
函数首先使能ADC1的时钟（__HAL_RCC_ADC1_CLK_ENABLE()）。然后，它使用RCC_PeriphCLKInitTypeDef结构体来配置ADC时钟源。最后，它调用HAL_RCCEx_PeriphCLKConfig()函数来应用时钟配置。如果配置失败，函数返回HAL_ERROR。

*/
```

**2.  改进的HAL\_ADC\_Init()函数 (Improved HAL\_ADC\_Init() Function)**

```c
/**
  * @brief  Initializes the ADC peripheral and regular group according to
  *         parameters specified in structure "ADC_InitTypeDef".
  * @param  hadc: ADC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_Init(ADC_HandleTypeDef* hadc) {
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;
  uint32_t tmp_cr1 = 0U;
  uint32_t tmp_cr2 = 0U;
  uint32_t tmp_sqr1 = 0U;

  /* 检查ADC句柄 */
  if (hadc == NULL) {
    return HAL_ERROR;
  }

  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  assert_param(IS_ADC_DATA_ALIGN(hadc->Init.DataAlign));
  assert_param(IS_ADC_SCAN_MODE(hadc->Init.ScanConvMode));
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_EXTTRIG(hadc->Init.ExternalTrigConv));

  if (hadc->Init.ScanConvMode != ADC_SCAN_DISABLE) {
    assert_param(IS_ADC_REGULAR_NB_CONV(hadc->Init.NbrOfConversion));
    assert_param(IS_FUNCTIONAL_STATE(hadc->Init.DiscontinuousConvMode));
    if (hadc->Init.DiscontinuousConvMode != DISABLE) {
      assert_param(IS_ADC_REGULAR_DISCONT_NUMBER(hadc->Init.NbrOfDiscConversion));
    }
  }

  /* 如果ADC处于复位状态，则执行以下操作 */
  if (hadc->State == HAL_ADC_STATE_RESET) {
    /* 初始化ADC错误代码 */
    ADC_CLEAR_ERRORCODE(hadc);

    /* 分配并初始化锁资源 */
    hadc->Lock = HAL_UNLOCKED;

    #if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
    /* 初始化ADC回调设置 */
    hadc->ConvCpltCallback = HAL_ADC_ConvCpltCallback;
    hadc->ConvHalfCpltCallback = HAL_ADC_ConvHalfCpltCallback;
    hadc->LevelOutOfWindowCallback = HAL_ADC_LevelOutOfWindowCallback;
    hadc->ErrorCallback = HAL_ADC_ErrorCallback;
    hadc->InjectedConvCpltCallback = HAL_ADCEx_InjectedConvCpltCallback;

    if (hadc->MspInitCallback == NULL) {
      hadc->MspInitCallback = HAL_ADC_MspInit;
    }

    /* 初始化底层硬件 */
    hadc->MspInitCallback(hadc);
    #else
    /* 初始化底层硬件 */
    HAL_ADC_MspInit(hadc);
    #endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
  }

  /* 停止可能正在进行的转换（常规组和注入组） */
  /* 禁用ADC外设 */
  tmp_hal_status = ADC_ConversionStop_Disable(hadc);

  /* 如果之前的初步操作正确完成，则配置ADC参数 */
  if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL) &&
      (tmp_hal_status == HAL_OK)) {
    /* 设置ADC状态 */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_BUSY_INTERNAL);

    /* 设置ADC参数 */

    /* 配置ADC：
       - 数据对齐方式
       - 外部触发器以启动转换
       - 外部触发器极性（始终设置为1，因为所有触发器都需要：外部触发器或软件启动）
       - 连续转换模式
       注意：外部触发器极性（ADC_CR2_EXTTRIG）在HAL_ADC_Start_xxx函数中设置，
            因为如果在此函数中设置，则注入组上的转换也会在启用ADC后启动常规组上的转换。*/
    tmp_cr2 |= (hadc->Init.DataAlign |
                ADC_CFGR_EXTSEL(hadc, hadc->Init.ExternalTrigConv) |
                ADC_CR2_CONTINUOUS((uint32_t)hadc->Init.ContinuousConvMode));

    /* 配置ADC：
       - 扫描模式
       - 禁止/启用不连续模式
       - 不连续模式转换次数 */
    tmp_cr1 |= (ADC_CR1_SCAN_SET(hadc->Init.ScanConvMode));

    /* 只有在禁用连续模式时才启用不连续模式 */
    /* 注意：如果参数“Init.ScanConvMode”设置为禁用，则无论如何都会设置参数不连续，但对ADC硬件没有影响。*/
    if (hadc->Init.DiscontinuousConvMode == ENABLE) {
      if (hadc->Init.ContinuousConvMode == DISABLE) {
        /* 启用选定的ADC常规不连续模式 */
        /* 设置以不连续模式转换的通道数 */
        SET_BIT(tmp_cr1, ADC_CR1_DISCEN |
                         ADC_CR1_DISCONTINUOUS_NUM(hadc->Init.NbrOfDiscConversion));
      } else {
        /* ADC常规组设置连续模式和序列不连续模式不能同时启用。*/

        /* 更新ADC状态机为错误状态 */
        SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_CONFIG);

        /* 设置ADC错误代码为ADC IP内部错误 */
        SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);
      }
    }

    /* 使用之前的设置更新ADC配置寄存器CR1 */
    MODIFY_REG(hadc->Instance->CR1,
               ADC_CR1_SCAN |
               ADC_CR1_DISCEN |
               ADC_CR1_DISCNUM,
               tmp_cr1);

    /* 使用之前的设置更新ADC配置寄存器CR2 */
    MODIFY_REG(hadc->Instance->CR2,
               ADC_CR2_ALIGN |
               ADC_CR2_EXTSEL |
               ADC_CR2_EXTTRIG |
               ADC_CR2_CONT,
               tmp_cr2);

    /* 配置常规组序列发生器：
       - 如果禁用扫描模式，则常规通道序列长度设置为0x00：转换1个通道（常规排名1上的通道）
         参数“NbrOfConversion”将被丢弃。
         注意：扫描模式在此设备上由硬件提供，如果禁用，则会自动丢弃转换次数。
              无论如何，为了在所有STM32设备上对齐，强制将转换次数设置为0x00。
       - 如果启用扫描模式，则常规通道序列长度设置为参数“NbrOfConversion” */
    if (ADC_CR1_SCAN_SET(hadc->Init.ScanConvMode) == ADC_SCAN_ENABLE) {
      tmp_sqr1 = ADC_SQR1_L_SHIFT(hadc->Init.NbrOfConversion);
    }

    MODIFY_REG(hadc->Instance->SQR1,
               ADC_SQR1_L,
               tmp_sqr1);

    /* 检查ADC寄存器是否已有效配置，以确保没有ADC内核IP时钟的潜在问题。*/
    /* 通过寄存器CR2检查（排除其他函数中设置的位：执行控制位（ADON、JSWSTART、SWSTART），常规组位（DMA），
       注入组位（JEXTTRIG和JEXTSEL），通道内部测量路径位（TSVREFE）。*/
    if (READ_BIT(hadc->Instance->CR2, ~(ADC_CR2_ADON | ADC_CR2_DMA |
                                        ADC_CR2_SWSTART | ADC_CR2_JSWSTART |
                                        ADC_CR2_JEXTTRIG | ADC_CR2_JEXTSEL |
                                        ADC_CR2_TSVREFE)) == tmp_cr2) {
      /* 将ADC错误代码设置为无 */
      ADC_CLEAR_ERRORCODE(hadc);

      /* 设置ADC状态 */
      ADC_STATE_CLR_SET(hadc->State,
                        HAL_ADC_STATE_BUSY_INTERNAL,
                        HAL_ADC_STATE_READY);
    } else {
      /* 更新ADC状态机为错误状态 */
      ADC_STATE_CLR_SET(hadc->State,
                        HAL_ADC_STATE_BUSY_INTERNAL,
                        HAL_ADC_STATE_ERROR_INTERNAL);

      /* 设置ADC错误代码为ADC IP内部错误 */
      SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);

      tmp_hal_status = HAL_ERROR;
    }

  } else {
    /* 更新ADC状态机为错误状态 */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);

    tmp_hal_status = HAL_ERROR;
  }

  /* 返回函数状态 */
  return tmp_hal_status;
}
/*
这段代码是STM32F1xx HAL库中ADC初始化函数HAL_ADC_Init的改进版本。它负责根据ADC_InitTypeDef结构体中的参数初始化ADC外设和常规组。

主要功能包括：

参数检查： 验证ADC句柄和输入参数的有效性。
时钟配置： 确保ADC时钟已在RCC级别配置。
状态管理： 管理ADC状态机，确保初始化在正确状态下进行。
寄存器配置： 设置ADC控制寄存器（CR1、CR2、SQR1）以配置数据对齐、扫描模式、连续转换模式、外部触发等。
错误处理： 检查配置过程中的错误，并在发生错误时更新ADC状态和错误代码。

*/
```

**3.  简化通道配置 (Simplified Channel Configuration)**

```c
/**
 * @brief  配置一个或多个通道的常规组。
 *
 * @param  hadc: ADC句柄
 * @param  sConfig: ADC通道配置结构体数组。
 * @param  numChannels: 要配置的通道数量。
 * @retval HAL状态
 */
HAL_StatusTypeDef HAL_ADC_ConfigChannels(ADC_HandleTypeDef* hadc, ADC_ChannelConfTypeDef sConfig[], uint32_t numChannels) {
  HAL_StatusTypeDef status = HAL_OK;
  for (uint32_t i = 0; i < numChannels; i++) {
    status = HAL_ADC_ConfigChannel(hadc, &sConfig[i]);
    if (status != HAL_OK) {
      /* 配置失败 */
      return status;
    }
  }
  return status;
}

/* 使用示例 (Example Usage): */
ADC_ChannelConfTypeDef channelConfig[2];

channelConfig[0].Channel = ADC_CHANNEL_0;
channelConfig[0].Rank = 1;
channelConfig[0].SamplingTime = ADC_SAMPLETIME_1CYCLE_5;

channelConfig[1].Channel = ADC_CHANNEL_1;
channelConfig[1].Rank = 2;
channelConfig[1].SamplingTime = ADC_SAMPLETIME_1CYCLE_5;

HAL_ADC_ConfigChannels(&hadc1, channelConfig, 2);

/*中文描述：
这段代码提供了一个更方便的函数HAL_ADC_ConfigChannels，用于一次配置多个ADC通道。该函数接收一个ADC通道配置结构体数组和一个要配置的通道数量作为输入。它遍历数组，并使用HAL_ADC_ConfigChannel函数配置每个通道。如果任何通道的配置失败，函数将立即返回一个错误状态。
*/
```

**4. 改进中断处理 (Improved Interrupt Handling)**

```c
/**
 * @brief  ADC中断处理函数。
 *
 * @param  hadc: ADC句柄
 * @retval None
 */
void HAL_ADC_IRQHandler(ADC_HandleTypeDef* hadc) {
  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* 常规组转换完成中断 */
  if (__HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_EOC) && (__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_EOC))) {
    /* 清除中断标志 */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC);

    /* 检查是否有错误 */
    if (HAL_IS_BIT_SET(hadc->State, HAL_ADC_STATE_ERROR_OVR)) {
      /* 发生溢出错误 */
      HAL_ADC_ErrorCallback(hadc);
      return; // 立即返回
    }

    /* 调用回调函数 */
    HAL_ADC_ConvCpltCallback(hadc);

    /* 禁用中断，如果不需要连续转换 */
    if (hadc->Init.ContinuousConvMode == DISABLE) {
      __HAL_ADC_DISABLE_IT(hadc, ADC_IT_EOC);
    }
  }

  /* 模拟看门狗中断 */
  if (__HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_AWD) && (__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_AWD))) {
    /* 清除中断标志 */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_AWD);

    /* 调用回调函数 */
    HAL_ADC_LevelOutOfWindowCallback(hadc);
  }

  /* 注入组转换完成中断 */
  if (__HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_JEOC) && (__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_JEOC)))
  {
    /* 清除中断标志 */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_JEOC);

    /* 调用回调函数 */
    HAL_ADCEx_InjectedConvCpltCallback(hadc);
  }

   /* 其他中断处理 (Overrun, etc.) - 根据需要添加 */

  /* 如果发生错误，调用错误回调函数 */
  if (HAL_ADC_GetError(hadc) != HAL_ADC_ERROR_NONE) {
    HAL_ADC_ErrorCallback(hadc);
  }
}
/*中文描述：
这段代码改进了ADC中断处理函数HAL_ADC_IRQHandler。改进包括：

清晰的中断源检查： 代码使用__HAL_ADC_GET_IT_SOURCE()宏来检查中断源，使用__HAL_ADC_GET_FLAG()宏来检查中断标志。这使得代码更容易阅读和理解。
错误处理： 代码检查是否有溢出错误（HAL_ADC_STATE_ERROR_OVR）。如果发生溢出错误，代码将调用HAL_ADC_ErrorCallback()函数并立即返回。
回调函数调用： 代码在中断发生时调用相应的回调函数（HAL_ADC_ConvCpltCallback()、HAL_ADC_LevelOutOfWindowCallback()等）。
中断禁用： 如果不需要连续转换，代码将禁用中断（__HAL_ADC_DISABLE_IT(hadc, ADC_IT_EOC)）。
其他中断处理： 代码包括一个用于处理其他中断（例如，Overrun）的占位符。
总的来说，这段代码提供了一个更健壮和更易于维护的ADC中断处理函数。
*/
```

**5. 简化ADC开启与停止 (Simplified ADC Start and Stop)**

```c
HAL_StatusTypeDef ADC_Start(ADC_HandleTypeDef* hadc) {
 HAL_StatusTypeDef tmp_hal_status = HAL_OK;
 
 /* 检查ADC句柄 */
  if (hadc == NULL) {
   return HAL_ERROR;
  }
 
  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
 
  /* 进程锁定 */
  __HAL_LOCK(hadc);
  
  /* 如果 ADC 没有被使能 */
  if (ADC_IS_ENABLE(hadc) == RESET)
  {
   /* 使能 ADC 外设 */
   __HAL_ADC_ENABLE(hadc);
 
   /* 等待 ADC 稳定时间 */
   HAL_Delay(1); // 短暂的延迟
  }
  else{
      //如果ADC已经开启，直接UNLOCK，并返回OK
      __HAL_UNLOCK(hadc);
      return HAL_OK;
  }
 
  /* 如果软件启动被选择,开始转换 */
  if (ADC_IS_SOFTWARE_START_REGULAR(hadc))
  {
   /* 清除常规组转换标志 */
   __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC);
   
   /* 启动 ADC 常规组转换 */
   SET_BIT(hadc->Instance->CR2, ADC_CR2_SWSTART);
  }
  
  /* 进程解锁 */
  __HAL_UNLOCK(hadc);
 
  /* 返回函数状态 */
  return tmp_hal_status;
}
 
HAL_StatusTypeDef ADC_Stop(ADC_HandleTypeDef* hadc) {
 HAL_StatusTypeDef tmp_hal_status = HAL_OK;
 
  /* 检查ADC句柄 */
  if (hadc == NULL) {
   return HAL_ERROR;
  }
 
  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
 
  /* 进程锁定 */
  __HAL_LOCK(hadc);
 
  /* 检查ADC是否正在运行 */
  if (ADC_IS_ENABLE(hadc) != RESET)
  {
   /* 停止 ADC 转换 */
   CLEAR_BIT(hadc->Instance->CR2, ADC_CR2_SWSTART); // 停止软件启动
 
   /* 关闭 ADC */
   __HAL_ADC_DISABLE(hadc);
   
   /* 等待 ADC 关闭 */
   HAL_Delay(1); // 短暂的延迟
  }
  else{
      //如果ADC已经关闭，直接UNLOCK，并返回OK
      __HAL_UNLOCK(hadc);
      return HAL_OK;
  }
 
  /* 进程解锁 */
  __HAL_UNLOCK(hadc);
 
  /* 返回函数状态 */
  return tmp_hal_status;
}
/*中文描述：
这段代码简化了ADC开启和停止的操作。
ADC_Start函数会先检查ADC是否已经使能，如果未使能，则使能ADC并等待稳定。
ADC_Stop函数会检查ADC是否正在运行，如果是，则停止转换并关闭ADC。
两个函数都包含了简单的错误处理和锁定机制，以确保安全的操作。

*/
```

**6. 简单演示示例 (Simple Demo Example)**

```c
/* USER CODE BEGIN Includes */
#include "stdio.h"
/* USER CODE END Includes */

ADC_HandleTypeDef hadc1;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_ADC1_Init(void);

/* USER CODE BEGIN PFP */
uint16_t ADC_value;

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int fputc(int ch, FILE *f)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
  return ch;
}
/* USER CODE END 0 */

int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration----------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  /* USER CODE BEGIN 2 */
  HAL_ADC_Start(&hadc1);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    HAL_ADC_PollForConversion(&hadc1,100);
    if((HAL_ADC_GetState(&hadc1) & HAL_ADC_STATE_EOC_REG) == HAL_ADC_STATE_EOC_REG)
    {
      ADC_value = HAL_ADC_GetValue(&hadc1);
      printf("ADC Value : %d\r\n",ADC_value);
    }
    HAL_Delay(500);
  }
  /* USER CODE END 3 */
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
  hadc1.Init.ContinuousConvMode = ENABLE;
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
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/* USER CODE END 4 */

/*中文描述：
这段代码是一个简单的ADC演示示例。它配置了ADC1来读取通道0上的模拟信号，并定期将ADC值打印到串口。
主要步骤包括：

初始化： 系统时钟和GPIO引脚。
ADC配置： 配置ADC1进行连续转换，禁用扫描模式，使用软件触发，并将数据右对齐。
通道配置： 配置ADC通道0，使用1.5个周期的采样时间。
启动ADC： 启动ADC进行转换。
主循环： 定期轮询ADC转换完成标志，读取ADC值，并通过printf函数将值打印到串口。
这个示例可以帮助你快速开始使用STM32F1xx上的ADC。

*/
```

**总结：**

以上代码示例提供了一些改进的思路，包括更清晰的结构、更完善的错误处理、更灵活的配置选项和更实用的演示示例。  在实际应用中，可以根据具体需求进行调整和扩展。