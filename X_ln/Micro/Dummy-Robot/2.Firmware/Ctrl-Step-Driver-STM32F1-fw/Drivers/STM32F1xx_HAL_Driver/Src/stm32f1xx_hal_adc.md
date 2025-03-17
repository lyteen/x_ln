Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_adc.c`

好的，没问题，让我们重新开始，我会更详细地解释关键代码，并附上更清晰的使用说明和例子。

**目标：**

我们的目标是对 STM32F1xx HAL 库中 `stm32f1xx_hal_adc.c` 文件的核心功能进行中文解释，并提供代码片段和使用示例。我们将重点关注以下几个方面：

1.  **ADC 初始化和配置（HAL\_ADC\_Init）**
2.  **ADC 规则组通道配置（HAL\_ADC\_ConfigChannel）**
3.  **ADC 启动转换（HAL\_ADC\_Start）和停止（HAL\_ADC\_Stop）**
4.  **ADC 中断处理（HAL\_ADC\_IRQHandler）**
5.  **ADC DMA 模式（HAL\_ADC\_Start\_DMA, HAL\_ADC\_Stop\_DMA）**

**1. ADC 初始化和配置 (HAL\_ADC\_Init)**

```c
HAL_StatusTypeDef HAL_ADC_Init(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;
  uint32_t tmp_cr1 = 0U;
  uint32_t tmp_cr2 = 0U;
  uint32_t tmp_sqr1 = 0U;

  /* 检查 ADC 句柄 */
  if(hadc == NULL)
  {
    return HAL_ERROR;
  }

  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance)); // 确保是有效的 ADC 实例
  assert_param(IS_ADC_DATA_ALIGN(hadc->Init.DataAlign)); // 数据对齐方式（左对齐或右对齐）
  assert_param(IS_ADC_SCAN_MODE(hadc->Init.ScanConvMode)); // 扫描模式使能
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode)); // 连续转换模式使能
  assert_param(IS_ADC_EXTTRIG(hadc->Init.ExternalTrigConv)); // 外部触发使能

  /* 更多参数检查... */

  /* ADC 时钟必须先在 RCC 层配置好 */
  /* 参考头文件中的时钟使能步骤 */

  /* 如果 ADC 处于复位状态，则执行 MSP 初始化 */
  if (hadc->State == HAL_ADC_STATE_RESET)
  {
    /* 初始化 ADC 错误代码 */
    ADC_CLEAR_ERRORCODE(hadc);

    /* 分配锁资源并初始化 */
    hadc->Lock = HAL_UNLOCKED;

    /* 调用 MSP 初始化函数 */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
    /* 如果使用回调注册机制，则使用注册的回调函数 */
    if (hadc->MspInitCallback == NULL)
    {
      hadc->MspInitCallback = HAL_ADC_MspInit; /* 默认 MSP 初始化 */
    }
    hadc->MspInitCallback(hadc);
#else
    /* 如果没有使用回调注册机制，则使用默认 MSP 初始化 */
    HAL_ADC_MspInit(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
  }

  /* 停止潜在的转换，禁用 ADC */
  tmp_hal_status = ADC_ConversionStop_Disable(hadc);

  /* 配置 ADC 参数，如果之前的操作都成功 */
  if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL) &&
      (tmp_hal_status == HAL_OK)                                  )
  {
    /* 设置 ADC 状态 */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_BUSY_INTERNAL);

    /* 设置 ADC 参数 */
    tmp_cr2 |= (hadc->Init.DataAlign                                          |
                ADC_CFGR_EXTSEL(hadc, hadc->Init.ExternalTrigConv)            |
                ADC_CR2_CONTINUOUS((uint32_t)hadc->Init.ContinuousConvMode)   );

    tmp_cr1 |= (ADC_CR1_SCAN_SET(hadc->Init.ScanConvMode));

    /* 更新 ADC 配置寄存器 CR1 和 CR2 */
    MODIFY_REG(hadc->Instance->CR1, /* ... */, tmp_cr1);
    MODIFY_REG(hadc->Instance->CR2, /* ... */, tmp_cr2);

    /* 配置规则组序列 */
    if (ADC_CR1_SCAN_SET(hadc->Init.ScanConvMode) == ADC_SCAN_ENABLE)
    {
      tmp_sqr1 = ADC_SQR1_L_SHIFT(hadc->Init.NbrOfConversion);
    }
    MODIFY_REG(hadc->Instance->SQR1, ADC_SQR1_L, tmp_sqr1);

    /* 检查 ADC 寄存器是否配置成功 */
    if (READ_BIT(hadc->Instance->CR2, ~(ADC_CR2_ADON | ADC_CR2_DMA | /* ... */)) == tmp_cr2)
    {
      /* 设置 ADC 状态为就绪 */
      ADC_CLEAR_ERRORCODE(hadc);
      ADC_STATE_CLR_SET(hadc->State,
                        HAL_ADC_STATE_BUSY_INTERNAL,
                        HAL_ADC_STATE_READY);
    }
    else
    {
      /* 设置 ADC 状态为错误 */
      SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);
      SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);
      tmp_hal_status = HAL_ERROR;
    }
  }
  else
  {
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);
    tmp_hal_status = HAL_ERROR;
  }

  /* 返回函数状态 */
  return tmp_hal_status;
}
```

**解释：**

*   **功能：** `HAL_ADC_Init` 函数负责初始化 ADC 外设，包括配置数据对齐、扫描模式、连续转换模式、外部触发等参数。
*   **前提条件：** ADC 的时钟必须在 RCC 层提前配置。
*   **流程：**
    1.  检查 ADC 句柄和参数的有效性。
    2.  如果 ADC 处于复位状态，则调用 MSP 初始化函数 (`HAL_ADC_MspInit`) 进行底层硬件初始化（例如，使能 ADC 时钟、配置 GPIO 引脚）。  `HAL_ADC_MspInit` 需要用户根据具体硬件连接自行实现。
    3.  停止可能正在进行的转换，并禁用 ADC。
    4.  配置 ADC 控制寄存器 CR1 和 CR2，以及规则组序列寄存器 SQR1，设置相应的参数。
    5.  进行最后的寄存器检查，以确保配置正确。

**如何使用：**

```c
ADC_HandleTypeDef hadc1;
ADC_InitTypeDef adc_init;

void HAL_ADC_MspInit(ADC_HandleTypeDef* hadc) { //需要用户自行实现的底层初始化函数
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  __HAL_RCC_ADC1_CLK_ENABLE(); // 使能 ADC1 时钟

  __HAL_RCC_GPIOA_CLK_ENABLE(); // 使能 GPIOA 时钟
  GPIO_InitStruct.Pin = GPIO_PIN_0; // 使用 PA0 作为 ADC 输入
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG; // 设置为模拟输入模式
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

int main(void) {
  HAL_Init();

  // 初始化 ADC_InitTypeDef 结构体
  adc_init.DataAlign = ADC_DATAALIGN_RIGHT; // 右对齐
  adc_init.ScanConvMode = ADC_SCAN_DISABLE; // 禁用扫描模式
  adc_init.ContinuousConvMode = DISABLE; // 禁用连续转换模式
  adc_init.ExternalTrigConv = ADC_SOFTWARE_START; // 软件触发
  adc_init.NbrOfConversion = 1; // 转换通道数量
  adc_init.DiscontinuousConvMode = DISABLE;
  adc_init.NbrOfDiscConversion = 0;

  // 初始化 ADC_HandleTypeDef 结构体
  hadc1.Instance = ADC1;
  hadc1.Init = adc_init;
  hadc1.State = HAL_ADC_STATE_RESET; // 确保ADC处于复位状态

  // 调用 HAL_ADC_Init 函数
  if (HAL_ADC_Init(&hadc1) != HAL_OK) {
    // 初始化出错处理
    Error_Handler();
  }

  while (1) {
    // 主循环
  }
}
```

**2. ADC 规则组通道配置 (HAL\_ADC\_ConfigChannel)**

```c
HAL_StatusTypeDef HAL_ADC_ConfigChannel(ADC_HandleTypeDef* hadc, ADC_ChannelConfTypeDef* sConfig)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;
  __IO uint32_t wait_loop_index = 0U;

  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  assert_param(IS_ADC_CHANNEL(sConfig->Channel)); // 确保是有效的 ADC 通道
  assert_param(IS_ADC_REGULAR_RANK(sConfig->Rank)); // 确保是有效的规则组排序
  assert_param(IS_ADC_SAMPLE_TIME(sConfig->SamplingTime)); // 确保是有效的采样时间

  /* 锁定流程 */
  __HAL_LOCK(hadc);

  /* 规则序列配置 */
  if (sConfig->Rank < 7U)
  {
    MODIFY_REG(hadc->Instance->SQR3                        ,
               ADC_SQR3_RK(ADC_SQR3_SQ1, sConfig->Rank)    ,
               ADC_SQR3_RK(sConfig->Channel, sConfig->Rank) );
  }
  /* 对于 Rank 7 到 12，以及 Rank 13 到 16 类似 */

  /* 通道采样时间配置 */
  if (sConfig->Channel >= ADC_CHANNEL_10)
  {
    MODIFY_REG(hadc->Instance->SMPR1                             ,
               ADC_SMPR1(ADC_SMPR1_SMP10, sConfig->Channel)      ,
               ADC_SMPR1(sConfig->SamplingTime, sConfig->Channel) );
  }
  else /* 对于通道 0 到 9 */
  {
    MODIFY_REG(hadc->Instance->SMPR2                             ,
               ADC_SMPR2(ADC_SMPR2_SMP0, sConfig->Channel)       ,
               ADC_SMPR2(sConfig->SamplingTime, sConfig->Channel) );
  }

  /* 如果选择 ADC1 的通道 16 或 17，则使能温度传感器和 VREFINT 测量路径 */
  if ((sConfig->Channel == ADC_CHANNEL_TEMPSENSOR) ||
      (sConfig->Channel == ADC_CHANNEL_VREFINT)      )
  {
    /* STM32F1 只有 ADC1 可以访问内部通道 */
    if (hadc->Instance == ADC1)
    {
      if (READ_BIT(hadc->Instance->CR2, ADC_CR2_TSVREFE) == RESET)
      {
        SET_BIT(hadc->Instance->CR2, ADC_CR2_TSVREFE);

        if (sConfig->Channel == ADC_CHANNEL_TEMPSENSOR)
        {
          /* 温度传感器稳定时间延迟 */
          wait_loop_index = (ADC_TEMPSENSOR_DELAY_US * (SystemCoreClock / 1000000U));
          while(wait_loop_index != 0U)
          {
            wait_loop_index--;
          }
        }
      }
    }
    else
    {
      /* 如果使用其他 ADC 访问内部通道，则报错 */
      SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_CONFIG);
      tmp_hal_status = HAL_ERROR;
    }
  }

  /* 流程解锁 */
  __HAL_UNLOCK(hadc);

  /* 返回函数状态 */
  return tmp_hal_status;
}
```

**解释：**

*   **功能：** `HAL_ADC_ConfigChannel` 函数配置 ADC 的规则组通道，包括选择通道、设置采样时间和指定在序列中的排序。
*   **流程：**
    1.  检查参数的有效性，如通道号、排序和采样时间。
    2.  根据通道的排序（Rank），将通道号写入相应的 SQR 寄存器（SQR1、SQR2 或 SQR3）。
    3.  根据通道号，设置相应的 SMPR 寄存器（SMPR1 或 SMPR2）中的采样时间。
    4.  如果选择了温度传感器或内部参考电压通道，则使能相应的内部路径（仅适用于 ADC1）。

**如何使用：**

```c
ADC_ChannelConfTypeDef sConfig = {0};

// 配置 ADC 通道
sConfig.Channel = ADC_CHANNEL_0; // 选择 ADC 通道 0
sConfig.Rank = 1; // 在序列中排第 1 位
sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5; // 采样时间为 1.5 个周期

if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) {
  // 配置出错处理
  Error_Handler();
}
```

**3. ADC 启动 (HAL\_ADC\_Start) 和停止 (HAL\_ADC\_Stop) 转换**

```c
HAL_StatusTypeDef HAL_ADC_Start(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* 锁定流程 */
  __HAL_LOCK(hadc);

  /* 使能 ADC 外设 */
  tmp_hal_status = ADC_Enable(hadc);

  /* 如果 ADC 使能成功，则启动转换 */
  if (tmp_hal_status == HAL_OK)
  {
    /* 设置 ADC 状态 */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_READY | HAL_ADC_STATE_REG_EOC,
                      HAL_ADC_STATE_REG_BUSY);

    /* 如果规则组的转换也触发注入组，则更新 ADC 状态 */
    if (READ_BIT(hadc->Instance->CR1, ADC_CR1_JAUTO) != RESET)
    {
      ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_INJ_EOC, HAL_ADC_STATE_INJ_BUSY);
    }

    /* 清除规则组转换标志 */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC);

    /* 如果选择了软件启动，则立即启动转换 */
    if (ADC_IS_SOFTWARE_START_REGULAR(hadc))
    {
      /* 使用软件启动，启动规则组转换 */
      SET_BIT(hadc->Instance->CR2, (ADC_CR2_SWSTART | ADC_CR2_EXTTRIG));
    }
    else
    {
      /* 使用外部触发，启动规则组转换 */
      SET_BIT(hadc->Instance->CR2, ADC_CR2_EXTTRIG);
    }
  }
  else
  {
    /* 流程解锁 */
    __HAL_UNLOCK(hadc);
  }

  /* 返回函数状态 */
  return tmp_hal_status;
}

HAL_StatusTypeDef HAL_ADC_Stop(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* 锁定流程 */
  __HAL_LOCK(hadc);

  /* 停止潜在的转换，禁用 ADC */
  tmp_hal_status = ADC_ConversionStop_Disable(hadc);

  /* 如果 ADC 禁用成功，则设置 ADC 状态 */
  if (tmp_hal_status == HAL_OK)
  {
    /* 设置 ADC 状态 */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_READY);
  }

  /* 流程解锁 */
  __HAL_UNLOCK(hadc);

  /* 返回函数状态 */
  return tmp_hal_status;
}

```

**解释：**

*   **功能：** `HAL_ADC_Start` 函数启动 ADC 规则组的转换，而 `HAL_ADC_Stop` 函数停止转换并禁用 ADC。
*   **`HAL_ADC_Start` 流程：**
    1.  检查参数的有效性。
    2.  使能 ADC 外设。
    3.  清除规则组转换标志。
    4.  如果配置为软件启动，则设置 `SWSTART` 位来启动转换；否则，等待外部触发信号。
*   **`HAL_ADC_Stop` 流程：**
    1.  检查参数的有效性。
    2.  停止 ADC 转换并禁用 ADC 外设。

**如何使用：**

```c
// 启动 ADC 转换
if (HAL_ADC_Start(&hadc1) != HAL_OK) {
  // 启动出错处理
  Error_Handler();
}

// 等待转换完成 (轮询模式)
HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);

// 获取 ADC 值
uint32_t adc_value = HAL_ADC_GetValue(&hadc1);

// 停止 ADC 转换
if (HAL_ADC_Stop(&hadc1) != HAL_OK) {
  // 停止出错处理
  Error_Handler();
}
```

**4. ADC 中断处理 (HAL\_ADC\_IRQHandler)**

```c
void HAL_ADC_IRQHandler(ADC_HandleTypeDef* hadc)
{
  /* 检查参数 */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_REGULAR_NB_CONV(hadc->Init.NbrOfConversion));

  /* 检查规则组转换结束标志 */
  if(__HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_EOC))
  {
    if(__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_EOC) )
    {
      /* 更新状态机 */
      if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL))
      {
        SET_BIT(hadc->State, HAL_ADC_STATE_REG_EOC);
      }

      /* 如果不是连续转换，则禁用 EOC 中断 */
      if(ADC_IS_SOFTWARE_START_REGULAR(hadc)        &&
         (hadc->Init.ContinuousConvMode == DISABLE)   )
      {
        __HAL_ADC_DISABLE_IT(hadc, ADC_IT_EOC);
        CLEAR_BIT(hadc->State, HAL_ADC_STATE_REG_BUSY);
        SET_BIT(hadc->State, HAL_ADC_STATE_READY);
      }

      /* 转换完成回调 */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
      hadc->ConvCpltCallback(hadc);
#else
      HAL_ADC_ConvCpltCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */

      /* 清除规则组转换标志 */
      __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_STRT | ADC_FLAG_EOC);
    }
  }

  /* 检查注入组转换结束标志 */
  // 省略注入组处理代码

  /* 检查模拟看门狗标志 */
  // 省略模拟看门狗处理代码
}
```

**解释：**

*   **功能：** `HAL_ADC_IRQHandler` 函数是 ADC 中断服务例程 (ISR)，负责处理 ADC 转换完成、注入组转换完成和模拟看门狗事件的中断。
*   **流程：**
    1.  检查中断源，确定是哪个事件触发了中断。
    2.  如果是规则组转换完成中断，则更新 ADC 状态，禁用 EOC 中断（如果不是连续转换），并调用转换完成回调函数 (`HAL_ADC_ConvCpltCallback`)。
    3.  清除相应的标志位。

**如何使用：**

1.  在 STM32 的启动文件中找到 ADC 的中断向量表，并将 `HAL_ADC_IRQHandler` 函数链接到相应的 ADC 中断向量。
2.  在 `main.c` 中定义 `HAL_ADC_ConvCpltCallback` 函数，以处理转换完成后的操作（例如，读取 ADC 值，进行数据处理）。

```c
// 在 stm32f1xx_it.c 中定义中断处理函数
void ADC1_2_IRQHandler(void)
{
  HAL_ADC_IRQHandler(&hadc1);
}

// 在 main.c 中定义回调函数
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
  // 读取 ADC 值
  adc_value = HAL_ADC_GetValue(hadc);

  // 在这里可以进行数据处理或其他操作
}
```

**5. ADC DMA 模式 (HAL\_ADC\_Start\_DMA, HAL\_ADC\_Stop\_DMA)**

```c
HAL_StatusTypeDef HAL_ADC_Start_DMA(ADC_HandleTypeDef* hadc, uint32_t* pData, uint32_t Length)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* 检查参数 */
  assert_param(IS_ADC_DMA_CAPABILITY_INSTANCE(hadc->Instance));

  /* 检查多重模式是否禁用 */
  if(ADC_MULTIMODE_IS_ENABLE(hadc) == RESET)
  {
    /* 锁定流程 */
    __HAL_LOCK(hadc);

    /* 使能 ADC 外设 */
    tmp_hal_status = ADC_Enable(hadc);

    /* 如果 ADC 使能成功，则启动转换 */
    if (tmp_hal_status == HAL_OK)
    {
      /* 设置 ADC 状态 */
      ADC_STATE_CLR_SET(hadc->State,
                        HAL_ADC_STATE_READY | HAL_ADC_STATE_REG_EOC | HAL_ADC_STATE_REG_OVR | HAL_ADC_STATE_REG_EOSMP,
                        HAL_ADC_STATE_REG_BUSY);

      /* 设置 DMA 回调函数 */
      hadc->DMA_Handle->XferCpltCallback = ADC_DMAConvCplt;
      hadc->DMA_Handle->XferHalfCpltCallback = ADC_DMAHalfConvCplt;
      hadc->DMA_Handle->XferErrorCallback = ADC_DMAError;

      /* 清除规则组转换标志 */
      __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC);

      /* 使能 ADC DMA 模式 */
      SET_BIT(hadc->Instance->CR2, ADC_CR2_DMA);

      /* 启动 DMA 通道 */
      HAL_DMA_Start_IT(hadc->DMA_Handle, (uint32_t)&hadc->Instance->DR, (uint32_t)pData, Length);

      /* 如果选择了软件启动，则立即启动转换 */
      if (ADC_IS_SOFTWARE_START_REGULAR(hadc))
      {
        /* 使用软件启动，启动规则组转换 */
        SET_BIT(hadc->Instance->CR2, (ADC_CR2_SWSTART | ADC_CR2_EXTTRIG));
      }
      else
      {
        /* 使用外部触发，启动规则组转换 */
        SET_BIT(hadc->Instance->CR2, ADC_CR2_EXTTRIG);
      }
    }
    else
    {
      /* 流程解锁 */
      __HAL_UNLOCK(hadc);
    }
  }
  else
  {
    tmp_hal_status = HAL_ERROR;
  }

  /* 返回函数状态 */
  return tmp_hal_status;
}

HAL_StatusTypeDef HAL_ADC_Stop_DMA(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* 检查参数 */
  assert_param(IS_ADC_DMA_CAPABILITY_INSTANCE(hadc->Instance));

  /* 锁定流程 */
  __HAL_LOCK(hadc);

  /* 停止潜在的转换，禁用 ADC */
  tmp_hal_status = ADC_ConversionStop_Disable(hadc);

  /* 如果 ADC 禁用成功，则禁用 DMA 模式 */
  if (tmp_hal_status == HAL_OK)
  {
    /* 禁用 ADC DMA 模式 */
    CLEAR_BIT(hadc->Instance->CR2, ADC_CR2_DMA);

    /* 停止 DMA 通道 */
    if (hadc->DMA_Handle->State == HAL_DMA_STATE_BUSY)
    {
      tmp_hal_status = HAL_DMA_Abort(hadc->DMA_Handle);

      /* 如果 DMA 通道禁用成功，则设置 ADC 状态 */
      if (tmp_hal_status == HAL_OK)
      {
        ADC_STATE_CLR_SET(hadc->State,
                          HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                          HAL_ADC_STATE_READY);
      }
      else
      {
        /* DMA 出错，则更新状态机 */
        SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_DMA);
      }
    }
  }

  /* 流程解锁 */
  __HAL_UNLOCK(hadc);

  /* 返回函数状态 */
  return tmp_hal_status;
}
```

**解释：**

*   **功能：** `HAL_ADC_Start_DMA` 函数启动 ADC 转换，并将结果通过 DMA 传输到指定的内存缓冲区，而 `HAL_ADC_Stop_DMA` 函数停止转换并禁用 DMA 传输。
*   **`HAL_ADC_Start_DMA` 流程：**
    1.  检查参数的有效性。
    2.  使能 ADC 外设。
    3.  设置 DMA 回调函数（传输完成、半传输完成和错误处理）。
    4.  使能 ADC DMA 模式。
    5.  启动 DMA 通道，将 ADC 数据寄存器 (`DR`) 中的数据传输到内存缓冲区。
    6.  如果配置为软件启动，则设置 `SWSTART` 位来启动转换；否则，等待外部触发信号。
*   **`HAL_ADC_Stop_DMA` 流程：**
    1.  检查参数的有效性。
    2.  停止 ADC 转换并禁用 ADC 外设。
    3.  禁用 ADC DMA 模式。
    4.  停止 DMA 通道。

**如何使用：**

1.  初始化 DMA 通道，并将 DMA 句柄与 ADC 句柄关联。
2.  定义用于存储 ADC 数据的缓冲区。
3.  调用 `HAL_ADC_Start_DMA` 函数启动 ADC 转换和 DMA 传输。
4.  在 DMA 回调函数中处理接收到的数据。
5.  调用 `HAL_ADC_Stop_DMA` 函数停止转换和 DMA 传输。

```c
DMA_HandleTypeDef hdma_adc1;
uint32_t adc_buffer[10]; // 存储 ADC 数据的缓冲区

// 初始化 DMA
void DMA1_Channel1_IRQHandler(void)
{
  HAL_DMA_IRQHandler(&hdma_adc1);
}

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
    hdma_adc1.Instance = DMA1_Channel1;
    hdma_adc1.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_adc1.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_adc1.Init.MemInc = DMA_MINC_ENABLE;
    hdma_adc1.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
    hdma_adc1.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
    hdma_adc1.Init.Mode = DMA_CIRCULAR;
    hdma_adc1.Init.Priority = DMA_PRIORITY_LOW;
    HAL_DMA_Init(&hdma_adc1);

    __HAL_LINKDMA(adcHandle,DMA_Handle,hdma_adc1);
    HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);
  }
}

// 启动 DMA 传输
HAL_ADC_Start_DMA(&hadc1, adc_buffer, 10);

// 在 DMA 完成回调函数中处理数据
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
  // adc_buffer 包含了 DMA 传输完成的 ADC 值
  // 在这里进行数据处理
}

// 停止 DMA 传输
HAL_ADC_Stop_DMA(&hadc1);
```

**总结：**

本文提供了对 STM32F1xx HAL 库中 `stm32f1xx_hal_adc.c` 文件核心功能的详细中文解释和使用示例。通过学习这些内容，你可以更好地理解和使用 STM32F1xx 的 ADC 外设，并根据自己的需求进行配置。请记住，实际的代码需要根据你的具体硬件连接和应用场景进行修改。