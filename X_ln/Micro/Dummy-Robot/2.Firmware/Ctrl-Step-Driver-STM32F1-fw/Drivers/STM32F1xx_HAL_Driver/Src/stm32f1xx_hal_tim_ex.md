Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_tim_ex.c`

好的，重新开始，我将把代码分解成更小的部分，并用中文详细解释每个部分的功能和使用方法，并附上简单的示例。

**1. HAL_TIMEx_HallSensor_Init 函数 (霍尔传感器接口初始化)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Init(TIM_HandleTypeDef *htim, TIM_HallSensor_InitTypeDef *sConfig)
{
  TIM_OC_InitTypeDef OC_Config;

  /* 检查 TIM 句柄是否有效 */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* 检查配置参数的有效性 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));
  assert_param(IS_TIM_IC_POLARITY(sConfig->IC1Polarity));
  assert_param(IS_TIM_IC_PRESCALER(sConfig->IC1Prescaler));
  assert_param(IS_TIM_IC_FILTER(sConfig->IC1Filter));

  /* 如果定时器处于复位状态，则初始化 */
  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* 分配锁资源并初始化 */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* 重置中断回调函数为默认值 */
    TIM_ResetCallback(htim);

    if (htim->HallSensor_MspInitCallback == NULL)
    {
      htim->HallSensor_MspInitCallback = HAL_TIMEx_HallSensor_MspInit;
    }
    /* 初始化底层硬件：GPIO、时钟、NVIC */
    htim->HallSensor_MspInitCallback(htim);
#else
    /* 初始化底层硬件：GPIO、时钟、NVIC 和 DMA */
    HAL_TIMEx_HallSensor_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* 设置 TIM 的状态为忙碌 */
  htim->State = HAL_TIM_STATE_BUSY;

  /* 配置定时器的基本时基，例如计数模式、分频等 */
  TIM_Base_SetConfig(htim->Instance, &htim->Init);

  /* 配置通道 1 作为输入通道，用于连接霍尔传感器的三个输出 */
  TIM_TI1_SetConfig(htim->Instance, sConfig->IC1Polarity, TIM_ICSELECTION_TRC, sConfig->IC1Filter);

  /* 重置 IC1PSC 位 */
  htim->Instance->CCMR1 &= ~TIM_CCMR1_IC1PSC;
  /* 设置 IC1PSC 值 (输入捕获预分频器)*/
  htim->Instance->CCMR1 |= sConfig->IC1Prescaler;

  /* 使能霍尔传感器接口（三个输入的异或功能）*/
  htim->Instance->CR2 |= TIM_CR2_TI1S;

  /* 选择 TIM_TS_TI1F_ED 信号作为定时器的输入触发源 */
  htim->Instance->SMCR &= ~TIM_SMCR_TS;
  htim->Instance->SMCR |= TIM_TS_TI1F_ED;

  /* 使用 TIM_TS_TI1F_ED 信号在每次边沿检测时重置定时器计数器 */
  htim->Instance->SMCR &= ~TIM_SMCR_SMS;
  htim->Instance->SMCR |= TIM_SLAVEMODE_RESET;

  /* 将通道 2 配置为 PWM 2 模式，并设置所需的换向延迟 (Commutation_Delay)*/
  OC_Config.OCFastMode = TIM_OCFAST_DISABLE;
  OC_Config.OCIdleState = TIM_OCIDLESTATE_RESET;
  OC_Config.OCMode = TIM_OCMODE_PWM2;
  OC_Config.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  OC_Config.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  OC_Config.OCPolarity = TIM_OCPOLARITY_HIGH;
  OC_Config.Pulse = sConfig->Commutation_Delay;

  TIM_OC2_SetConfig(htim->Instance, &OC_Config);

  /* 选择 OC2REF 作为 TRGO 上的触发输出：在 TIMx_CR2 寄存器中将 MMS 位写入 101 */
  htim->Instance->CR2 &= ~TIM_CR2_MMS;
  htim->Instance->CR2 |= TIM_TRGO_OC2REF;

  /* 初始化 DMA 突发操作状态 */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* 初始化 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* 初始化 TIM 状态 */
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}
```

**描述:** `HAL_TIMEx_HallSensor_Init` 函数用于初始化定时器，使其工作在霍尔传感器接口模式。它配置输入捕获通道，设置触发源，并配置PWM通道用于换向延迟。简单来说，就是配置定时器来读取霍尔传感器的信号。

**如何使用:**

1.  **初始化 `TIM_HandleTypeDef` 结构体:** 包含定时器的基本配置信息，例如计数模式、时钟分频等。
2.  **初始化 `TIM_HallSensor_InitTypeDef` 结构体:** 包含霍尔传感器接口的配置信息，例如输入捕获的极性、预分频器、滤波器以及换向延迟。
3.  **调用 `HAL_TIMEx_HallSensor_Init` 函数:**  将上面两个结构体的指针作为参数传递给该函数。

**示例:**

```c
TIM_HandleTypeDef htim3; //假设使用TIM3
TIM_HallSensor_InitTypeDef sHallConfig;

void HAL_TIMEx_HallSensor_MspInit(TIM_HandleTypeDef* htim) //底层初始化函数
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(htim->Instance==TIM3)
  {
    /* Peripheral clock enable */
    __HAL_RCC_TIM3_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**TIM3 GPIO Configuration
    PA6     ------> TIM3_CH1
    PA7     ------> TIM3_CH2
    PB0     ------> TIM3_CH3
    */
    GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  }

}

int main(void)
{
  HAL_Init(); // 初始化HAL库

  // ... 其他初始化代码 ...

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 71; // 1MHz时钟
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_HallSensor_Init(&htim3) != HAL_OK)
  {
    Error_Handler(); // 错误处理
  }

  sHallConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sHallConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sHallConfig.IC1Filter = 0;
  sHallConfig.Commutation_Delay = 1000;

  if (HAL_TIMEx_HallSensor_Init(&htim3, &sHallConfig) != HAL_OK)
  {
    Error_Handler(); // 错误处理
  }

  // ... 其他代码 ...
}
```

**关键点:**

*   **`HAL_TIMEx_HallSensor_MspInit(TIM_HandleTypeDef *htim)`:**  这是一个弱函数 (weak function)，需要在你的代码中重新定义。 该函数负责初始化与定时器相关的 GPIO 引脚、时钟和 NVIC (中断控制器)。
*   **时钟配置:**  确保定时器的时钟正确配置，以便进行准确的测量。
*   **GPIO 配置:** 将霍尔传感器的输出引脚配置为定时器的输入捕获通道。

**2. HAL_TIMEx_HallSensor_DeInit 函数 (霍尔传感器接口反初始化)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_DeInit(TIM_HandleTypeDef *htim)
{
  /* 检查参数 */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* 禁用 TIM 外设时钟 */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->HallSensor_MspDeInitCallback == NULL)
  {
    htim->HallSensor_MspDeInitCallback = HAL_TIMEx_HallSensor_MspDeInit;
  }
  /* 反初始化底层硬件 */
  htim->HallSensor_MspDeInitCallback(htim);
#else
  /* 反初始化底层硬件：GPIO、时钟、NVIC */
  HAL_TIMEx_HallSensor_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* 更改 DMA 突发操作状态 */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* 更改 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_RESET);

  /* 更改 TIM 状态 */
  htim->State = HAL_TIM_STATE_RESET;

  /* 释放锁 */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}
```

**描述:** `HAL_TIMEx_HallSensor_DeInit` 函数用于反初始化霍尔传感器接口，它禁用定时器时钟，并反初始化相关的GPIO。

**如何使用:** 将 `TIM_HandleTypeDef` 结构体的指针传递给该函数。

**示例:**

```c
//假设htim3已经初始化
HAL_TIMEx_HallSensor_DeInit(&htim3);
```

**关键点:**

*   **`HAL_TIMEx_HallSensor_MspDeInit(TIM_HandleTypeDef *htim)`:** 这也是一个弱函数，需要在你的代码中重新定义。 该函数负责反初始化与定时器相关的 GPIO 引脚、时钟和 NVIC 。

**3. HAL_TIMEx_HallSensor_Start 函数 (启动霍尔传感器接口)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Start(TIM_HandleTypeDef *htim)
{
  uint32_t tmpsmcr;
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* 检查参数 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));

  /* 检查 TIM 通道状态 */
  if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
  {
    return HAL_ERROR;
  }

  /* 设置 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);

  /* 使能输入捕获通道 1 */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);

  /* 使能外设，除非处于触发模式，否则使能会自动完成 */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* 返回函数状态 */
  return HAL_OK;
}
```

**描述:** `HAL_TIMEx_HallSensor_Start` 函数用于启动霍尔传感器接口，它使能输入捕获通道和定时器外设。

**如何使用:** 将 `TIM_HandleTypeDef` 结构体的指针传递给该函数。

**示例:**

```c
//假设htim3已经初始化并配置
HAL_TIMEx_HallSensor_Start(&htim3);
```

**4. HAL_TIMEx_HallSensor_Stop 函数 (停止霍尔传感器接口)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Stop(TIM_HandleTypeDef *htim)
{
  /* 检查参数 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));

  /* 禁用输入捕获通道 1 */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);

  /* 禁用外设 */
  __HAL_TIM_DISABLE(htim);

  /* 设置 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* 返回函数状态 */
  return HAL_OK;
}
```

**描述:** `HAL_TIMEx_HallSensor_Stop` 函数用于停止霍尔传感器接口，它禁用输入捕获通道和定时器外设。

**如何使用:** 将 `TIM_HandleTypeDef` 结构体的指针传递给该函数。

**示例:**

```c
//假设htim3已经启动
HAL_TIMEx_HallSensor_Stop(&htim3);
```

**5.  HAL_TIMEx_HallSensor_Start_IT 函数 (中断模式启动霍尔传感器接口)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Start_IT(TIM_HandleTypeDef *htim)
{
  uint32_t tmpsmcr;
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* 检查参数 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));

  /* 检查 TIM 通道状态 */
  if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
  {
    return HAL_ERROR;
  }

  /* 设置 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);

  /* 使能捕获比较中断 1 事件 */
  __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);

  /* 使能输入捕获通道 1 */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);

  /* 使能外设，除非处于触发模式，否则使能会自动完成 */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* 返回函数状态 */
  return HAL_OK;
}
```

**描述:**  `HAL_TIMEx_HallSensor_Start_IT` 函数启动在中断模式下的霍尔传感器接口。当霍尔传感器产生一个输入捕获事件时，会触发中断。

**如何使用:**
1. 确保在 `HAL_TIMEx_HallSensor_Init` 之后调用此函数。
2.  实现 `HAL_TIM_IC_CaptureCallback` 函数来处理捕获事件。

**示例:**
```c
// 在 main.c 中
void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM3) {
    // 处理霍尔传感器输入捕获事件
    uint32_t capture_value = HAL_TIM_ReadCapturedValue(htim, TIM_CHANNEL_1);
    // ... 你的处理代码 ...
  }
}
```

**6. HAL_TIMEx_HallSensor_Stop_IT 函数 (停止中断模式霍尔传感器接口)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Stop_IT(TIM_HandleTypeDef *htim)
{
  /* 检查参数 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));

  /* 禁用输入捕获通道 1 */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);

  /* 禁用捕获比较中断事件 */
  __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);

  /* 禁用外设 */
  __HAL_TIM_DISABLE(htim);

  /* 设置 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* 返回函数状态 */
  return HAL_OK;
}
```

**描述:**  `HAL_TIMEx_HallSensor_Stop_IT` 函数停止中断模式下的霍尔传感器接口，禁用中断和外设。

**如何使用:**  确保在不再需要中断时调用此函数。

**7. HAL_TIMEx_HallSensor_Start_DMA 函数 (DMA 模式启动霍尔传感器接口)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Start_DMA(TIM_HandleTypeDef *htim, uint32_t *pData, uint16_t Length)
{
  uint32_t tmpsmcr;
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);

  /* 检查参数 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));

  /* 设置 TIM 通道状态 */
  if ((channel_1_state == HAL_TIM_CHANNEL_STATE_BUSY)
      || (complementary_channel_1_state == HAL_TIM_CHANNEL_STATE_BUSY))
  {
    return HAL_BUSY;
  }
  else if ((channel_1_state == HAL_TIM_CHANNEL_STATE_READY)
           && (complementary_channel_1_state == HAL_TIM_CHANNEL_STATE_READY))
  {
    if ((pData == NULL) && (Length > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else
  {
    return HAL_ERROR;
  }

  /* 使能输入捕获通道 1 */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);

  /* 设置 DMA 输入捕获 1 回调函数 */
  htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMACaptureCplt;
  htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;
  /* 设置 DMA 错误回调函数 */
  htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

  /* 使能捕获 1 的 DMA 通道 */
  if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)&htim->Instance->CCR1, (uint32_t)pData, Length) != HAL_OK)
  {
    /* 返回错误状态 */
    return HAL_ERROR;
  }
  /* 使能捕获比较 1 中断 */
  __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC1);

  /* 使能外设，除非处于触发模式，否则使能会自动完成 */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* 返回函数状态 */
  return HAL_OK;
}
```

**描述:** `HAL_TIMEx_HallSensor_Start_DMA` 函数启动在 DMA 模式下的霍尔传感器接口。 定时器捕获的值通过 DMA 传输到内存中的缓冲区。

**如何使用:**

1.  确保在 `HAL_TIMEx_HallSensor_Init` 之后调用此函数。
2.  配置 DMA 通道，使其与定时器输入捕获通道关联。
3.  提供一个指向内存缓冲区的指针 (`pData`) 和要传输的数据长度 (`Length`)。
4.  实现 `HAL_TIM_IC_CaptureCallback` 和/或 `HAL_TIM_IC_CaptureHalfCpltCallback` 来处理 DMA 传输完成事件。

**示例:**

```c
#define HALL_DATA_LENGTH 10
uint32_t hall_data[HALL_DATA_LENGTH];

// 在初始化部分：
DMA_HandleTypeDef hdma_tim3_ch1;  // DMA句柄
void HAL_TIMEx_HallSensor_MspInit(TIM_HandleTypeDef* htim) //底层初始化函数
{
  // ... (之前的代码不变) ...

  /* TIM3 DMA Init */
  /* TIM3_CH1 Init */
  hdma_tim3_ch1.Instance = DMA1_Channel2;
  hdma_tim3_ch1.Init.Direction = DMA_PERIPH_TO_MEMORY;
  hdma_tim3_ch1.Init.PeriphInc = DMA_PINC_DISABLE;
  hdma_tim3_ch1.Init.MemInc = DMA_MINC_ENABLE;
  hdma_tim3_ch1.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
  hdma_tim3_ch1.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
  hdma_tim3_ch1.Init.Mode = DMA_CIRCULAR;
  hdma_tim3_ch1.Init.Priority = DMA_PRIORITY_LOW;
  if (HAL_DMA_Init(&hdma_tim3_ch1) != HAL_OK)
  {
    Error_Handler();
  }

  __HAL_LINKDMA(htim,hdma[TIM_DMA_ID_CC1],hdma_tim3_ch1);

  /* Peripheral interrupt init */
  HAL_NVIC_SetPriority(TIM3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(TIM3_IRQn);

}
//在主函数中
HAL_TIMEx_HallSensor_Start_DMA(&htim3, hall_data, HALL_DATA_LENGTH);
```

**8. HAL_TIMEx_HallSensor_Stop_DMA 函数 (停止 DMA 模式霍尔传感器接口)**

```c
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Stop_DMA(TIM_HandleTypeDef *htim)
{
  /* 检查参数 */
  assert_param(IS_TIM_HALL_SENSOR_INTERFACE_INSTANCE(htim->Instance));

  /* 禁用输入捕获通道 1 */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);

  /* 禁用捕获比较中断 1 事件 */
  __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC1);

  (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);

  /* 禁用外设 */
  __HAL_TIM_DISABLE(htim);

  /* 设置 TIM 通道状态 */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);

  /* 返回函数状态 */
  return HAL_OK;
}
```

**描述:**  `HAL_TIMEx_HallSensor_Stop_DMA` 函数停止 DMA 模式下的霍尔传感器接口，禁用 DMA 请求和外设。

**如何使用:**

1.  确保 DMA 传输已启动。
2.  调用此函数停止 DMA 传输。

**关键点:**

*   **`TIM_DMA_ID_CC1`:**  这是用于指定与捕获通道 1 关联的 DMA 通道的 ID。

总之，这些函数提供了一组用于配置和控制 STM32 定时器外设以连接霍尔传感器的工具。记住始终正确初始化和反初始化 MSP 函数。
