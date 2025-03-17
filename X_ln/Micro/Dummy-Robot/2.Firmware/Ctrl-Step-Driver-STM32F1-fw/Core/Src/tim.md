Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\tim.c`

好的，重新开始。以下是该代码的分解，附带中文解释和示例：

```c
/**
  ******************************************************************************
  * @file    tim.c
  * @brief   This file provides code for the configuration
  *          of the TIM instances.
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
#include "tim.h"

/* USER CODE BEGIN 0 */
#include "common_inc.h"

/* USER CODE END 0 */

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
```

**解释:**

*   `#include "tim.h"`: 包含了定时器相关的头文件，定义了定时器相关的结构体和函数。
*   `#include "common_inc.h"`: 包含了项目常用的头文件，例如定义了`Error_Handler()`函数。
*   `TIM_HandleTypeDef htim1; ... TIM_HandleTypeDef htim4;`:  声明了四个定时器句柄变量，分别对应 TIM1, TIM2, TIM3, TIM4。 这些句柄用于配置和控制相应的定时器外设。 `TIM_HandleTypeDef` 是一个结构体类型，包含了定时器的所有配置信息。

**作用:**

这部分代码主要完成的是引入必要的头文件和定义全局的定时器句柄变量。 这些句柄变量是后续定时器配置和控制的基础。

---

```c
/* TIM1 init function */
void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 71;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 9999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}
```

**解释:**

*   `void MX_TIM1_Init(void)`:  TIM1 的初始化函数。 `MX_` 前缀通常表示是由 STM32CubeMX 生成的代码。
*   `TIM_ClockConfigTypeDef sClockSourceConfig = {0};`: 定义时钟源配置结构体变量，并初始化为0.
*   `TIM_MasterConfigTypeDef sMasterConfig = {0};`: 定义主模式配置结构体变量，并初始化为0.
*   `htim1.Instance = TIM1;`:  设置定时器实例为 TIM1。
*   `htim1.Init.Prescaler = 71;`: 设置预分频器为 71。 这会将定时器时钟频率降低到  `SystemCoreClock / (Prescaler + 1)`。 例如，如果 `SystemCoreClock` 是 72MHz，那么定时器时钟频率将是 1MHz。
*   `htim1.Init.CounterMode = TIM_COUNTERMODE_UP;`: 设置计数器模式为向上计数。
*   `htim1.Init.Period = 9999;`: 设置自动重载值（ARR）为 9999。 当计数器达到这个值时，会重新从 0 开始计数，并触发中断（如果使能了中断）。
*   `htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;`: 设置时钟分频因子，这里设置为不分频。
*   `htim1.Init.RepetitionCounter = 0;`:  设置重复计数器。 通常用于高级定时器，这里设置为 0。
*   `htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;`: 启用自动重载预加载。 这意味着新的 ARR 值会先加载到一个影子寄存器中，然后在下一个更新事件时才生效，以避免在计数过程中修改 ARR 值导致问题。
*   `HAL_TIM_Base_Init(&htim1)`: 调用 HAL 库函数初始化定时器。
*   `sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;`: 选择内部时钟源。
*   `HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig)`: 配置时钟源。
*   `sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;`: 配置主输出触发。 这里设置为复位模式。
*   `sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;`:  禁用主从模式。
*   `HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig)`: 配置主模式同步。

**作用:**

该函数配置 TIM1 为一个基本的定时器，具有指定的预分频器、计数模式、自动重载值等。它还配置了时钟源和主模式。

**示例:**

假设 `SystemCoreClock` 是 72MHz。 通过设置 `htim1.Init.Prescaler = 71` 和 `htim1.Init.Period = 9999`，可以计算出 TIM1 的中断频率：

*   定时器时钟频率: 72MHz / (71 + 1) = 1MHz
*   中断频率: 1MHz / (9999 + 1) = 100Hz

这意味着 TIM1 每秒会产生 100 次中断。

---

```c
/* TIM2 init function */
void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 1023;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}
```

**解释:**

*   `void MX_TIM2_Init(void)`: TIM2 的初始化函数。
*   `TIM_OC_InitTypeDef sConfigOC = {0};`: 定义输出比较配置结构体变量，并初始化为0.
*   `htim2.Instance = TIM2;`: 设置定时器实例为 TIM2。
*   `htim2.Init.Prescaler = 0;`:  设置预分频器为 0，意味着不分频。
*   `htim2.Init.CounterMode = TIM_COUNTERMODE_UP;`: 设置计数器模式为向上计数。
*   `htim2.Init.Period = 1023;`: 设置自动重载值（ARR）为 1023。
*   `htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;`: 设置时钟分频因子，这里设置为不分频。
*   `htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;`: 禁用自动重载预加载。
*   `HAL_TIM_PWM_Init(&htim2)`: 调用 HAL 库函数初始化定时器为 PWM 模式。
*   `sConfigOC.OCMode = TIM_OCMODE_PWM1;`: 设置输出比较模式为 PWM 模式 1。  在 PWM 模式 1 中，当计数器值小于比较值 (Pulse) 时，输出为有效电平；当计数器值大于比较值时，输出为无效电平。
*   `sConfigOC.Pulse = 0;`: 设置脉冲宽度（占空比）为 0。  这个值会决定 PWM 波形的占空比。 初始值为0，意味着初始占空比为 0%。
*   `sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;`: 设置输出极性为高电平有效。
*   `sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;`: 禁用快速模式。
*   `HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_3)`:  配置 TIM2 的通道 3 为 PWM 输出。
*   `HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_4)`:  配置 TIM2 的通道 4 为 PWM 输出。
*   `HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);`: 启动 TIM2 的通道 3 的 PWM 输出。
*   `HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4);`: 启动 TIM2 的通道 4 的 PWM 输出。
*   `HAL_TIM_MspPostInit(&htim2)`: 调用 HAL 库函数执行 TIM2 的 MSP 后期初始化，主要是配置 GPIO。

**作用:**

该函数配置 TIM2 为 PWM 发生器，并通过通道 3 和通道 4 输出 PWM 信号。  PWM 可用于控制电机速度、调节 LED 亮度等。

**示例:**

假设 `SystemCoreClock` 是 72MHz。 由于 `htim2.Init.Prescaler = 0` 和 `htim2.Init.Period = 1023`，可以计算出 PWM 的频率：

*   PWM 频率: 72MHz / (1023 + 1) = 大约 70.31kHz

要改变 PWM 信号的占空比，需要修改 `sConfigOC.Pulse` 的值， 然后调用 `HAL_TIM_PWM_ConfigChannel()` 和 `HAL_TIM_PWM_Start()` 更新 PWM 输出。例如，要设置通道 3 的占空比为 50%，可以将 `sConfigOC.Pulse` 设置为 `1023 / 2 = 511`。

---

```c
/* TIM3 init function */
void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_SlaveConfigTypeDef sSlaveConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_IC_InitTypeDef sConfigIC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 71;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_IC_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sSlaveConfig.SlaveMode = TIM_SLAVEMODE_RESET;
  sSlaveConfig.InputTrigger = TIM_TS_TI1FP1;
  sSlaveConfig.TriggerPolarity = TIM_INPUTCHANNELPOLARITY_RISING;
  sSlaveConfig.TriggerFilter = 0;
  if (HAL_TIM_SlaveConfigSynchro(&htim3, &sSlaveConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_RISING;
  sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;
  sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;
  sConfigIC.ICFilter = 0;
  if (HAL_TIM_IC_ConfigChannel(&htim3, &sConfigIC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}
```

**解释:**

*   `void MX_TIM3_Init(void)`: TIM3 的初始化函数。
*   `TIM_SlaveConfigTypeDef sSlaveConfig = {0};`: 定义从模式配置结构体变量，并初始化为0.
*   `TIM_MasterConfigTypeDef sMasterConfig = {0};`: 定义主模式配置结构体变量，并初始化为0.
*   `TIM_IC_InitTypeDef sConfigIC = {0};`: 定义输入捕获配置结构体变量，并初始化为0.
*   `htim3.Instance = TIM3;`: 设置定时器实例为 TIM3。
*   `htim3.Init.Prescaler = 71;`: 设置预分频器为 71。
*   `htim3.Init.CounterMode = TIM_COUNTERMODE_UP;`: 设置计数器模式为向上计数。
*   `htim3.Init.Period = 65535;`: 设置自动重载值（ARR）为 65535，这是 16 位定时器的最大值。
*   `htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;`: 设置时钟分频因子，这里设置为不分频。
*   `htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;`: 禁用自动重载预加载。
*   `HAL_TIM_Base_Init(&htim3)`: 调用 HAL 库函数初始化定时器。
*   `HAL_TIM_IC_Init(&htim3)`: 调用 HAL 库函数初始化定时器为输入捕获模式。
*   `sSlaveConfig.SlaveMode = TIM_SLAVEMODE_RESET;`: 设置从模式为复位模式。 在复位模式下，当触发输入信号有效时，计数器会被复位为 0。
*   `sSlaveConfig.InputTrigger = TIM_TS_TI1FP1;`: 设置触发输入源为 TIM3 的通道 1 (TI1FP1)。
*   `sSlaveConfig.TriggerPolarity = TIM_INPUTCHANNELPOLARITY_RISING;`: 设置触发极性为上升沿。 也就是说，当 TIM3_CH1 引脚检测到上升沿时，会触发复位。
*   `sSlaveConfig.TriggerFilter = 0;`:  设置触发滤波器，这里设置为无滤波器。
*   `HAL_TIM_SlaveConfigSynchro(&htim3, &sSlaveConfig)`: 配置从模式同步。
*   `sConfigIC.ICPolarity = TIM_INPUTCHANNELPOLARITY_RISING;`: 设置输入捕获极性为上升沿。
*   `sConfigIC.ICSelection = TIM_ICSELECTION_DIRECTTI;`: 设置输入捕获选择为直接输入。
*   `sConfigIC.ICPrescaler = TIM_ICPSC_DIV1;`: 设置输入捕获预分频器为 1，即不分频。
*   `sConfigIC.ICFilter = 0;`: 设置输入捕获滤波器，这里设置为无滤波器。
*   `HAL_TIM_IC_ConfigChannel(&htim3, &sConfigIC, TIM_CHANNEL_1)`: 配置 TIM3 的通道 1 为输入捕获。

**作用:**

该函数配置 TIM3 为输入捕获模式，并将其设置为从模式，由通道 1 的上升沿触发复位。  这可以用于测量外部信号的频率或者脉冲宽度。

**示例:**

TIM3 可以用来测量输入到 PA6 引脚的信号的频率。 每当 PA6 上出现一个上升沿，TIM3 的计数器就会被复位，然后开始计数。 在下一个上升沿到来时，读取计数器的值，这个值就代表了两个上升沿之间的时间间隔。  通过计算时间间隔的倒数，就可以得到信号的频率。

---

```c
/* TIM4 init function */
void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 71;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 49;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim4, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}
```

**解释:**

*   `void MX_TIM4_Init(void)`: TIM4 的初始化函数。
*   `TIM_ClockConfigTypeDef sClockSourceConfig = {0};`: 定义时钟源配置结构体变量，并初始化为0.
*   `TIM_MasterConfigTypeDef sMasterConfig = {0};`: 定义主模式配置结构体变量，并初始化为0.
*   `htim4.Instance = TIM4;`: 设置定时器实例为 TIM4。
*   `htim4.Init.Prescaler = 71;`: 设置预分频器为 71。
*   `htim4.Init.CounterMode = TIM_COUNTERMODE_UP;`: 设置计数器模式为向上计数。
*   `htim4.Init.Period = 49;`: 设置自动重载值（ARR）为 49。
*   `htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;`: 设置时钟分频因子，这里设置为不分频。
*   `htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;`: 启用自动重载预加载。
*   `HAL_TIM_Base_Init(&htim4)`: 调用 HAL 库函数初始化定时器。
*   `sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;`: 选择内部时钟源。
*   `HAL_TIM_ConfigClockSource(&htim4, &sClockSourceConfig)`: 配置时钟源。
*   `sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;`: 配置主输出触发。 这里设置为复位模式。
*   `sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;`:  禁用主从模式。
*   `HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig)`: 配置主模式同步。

**作用:**

该函数配置 TIM4 为一个基本的定时器，具有指定的预分频器、计数模式和自动重载值。

**示例:**

假设 `SystemCoreClock` 是 72MHz。 通过设置 `htim4.Init.Prescaler = 71` 和 `htim4.Init.Period = 49`，可以计算出 TIM4 的中断频率：

*   定时器时钟频率: 72MHz / (71 + 1) = 1MHz
*   中断频率: 1MHz / (49 + 1) = 20kHz

这意味着 TIM4 每秒会产生 20000 次中断。  这个定时器通常用于高频率的事件触发，例如控制 LED 的闪烁或者进行 ADC 采样。

---

```c
void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* tim_baseHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(tim_baseHandle->Instance==TIM1)
  {
  /* USER CODE BEGIN TIM1_MspInit 0 */

  /* USER CODE END TIM1_MspInit 0 */
    /* TIM1 clock enable */
    __HAL_RCC_TIM1_CLK_ENABLE();

    /* TIM1 interrupt Init */
    HAL_NVIC_SetPriority(TIM1_UP_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(TIM1_UP_IRQn);
  /* USER CODE BEGIN TIM1_MspInit 1 */

  /* USER CODE END TIM1_MspInit 1 */
  }
  else if(tim_baseHandle->Instance==TIM3)
  {
  /* USER CODE BEGIN TIM3_MspInit 0 */

  /* USER CODE END TIM3_MspInit 0 */
    /* TIM3 clock enable */
    __HAL_RCC_TIM3_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**TIM3 GPIO Configuration
    PA6     ------> TIM3_CH1
    */
    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* TIM3 interrupt Init */
    HAL_NVIC_SetPriority(TIM3_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(TIM3_IRQn);
  /* USER CODE BEGIN TIM3_MspInit 1 */

  /* USER CODE END TIM3_MspInit 1 */
  }
  else if(tim_baseHandle->Instance==TIM4)
  {
  /* USER CODE BEGIN TIM4_MspInit 0 */

  /* USER CODE END TIM4_MspInit 0 */
    /* TIM4 clock enable */
    __HAL_RCC_TIM4_CLK_ENABLE();

    /* TIM4 interrupt Init */
    HAL_NVIC_SetPriority(TIM4_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(TIM4_IRQn);
  /* USER CODE BEGIN TIM4_MspInit 1 */

  /* USER CODE END TIM4_MspInit 1 */
  }
}
```

**解释:**

*   `void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* tim_baseHandle)`:  HAL 库定时器底层初始化函数（MSP: MCU Specific Package）。  该函数用于配置特定于 MCU 的定时器资源，例如时钟使能、GPIO 配置和中断使能。
*   `GPIO_InitTypeDef GPIO_InitStruct = {0};`: 定义 GPIO 初始化结构体变量，并初始化为 0。
*   `if(tim_baseHandle->Instance==TIM1) { ... }`:  根据定时器实例，执行不同的初始化操作。
*   `__HAL_RCC_TIM1_CLK_ENABLE();`: 使能 TIM1 的时钟。  在使用定时器之前，必须先使能其时钟。
*   `HAL_NVIC_SetPriority(TIM1_UP_IRQn, 5, 0);`: 设置 TIM1 更新中断的优先级。 `TIM1_UP_IRQn` 是 TIM1 更新中断的中断向量。
*   `HAL_NVIC_EnableIRQ(TIM1_UP_IRQn);`: 使能 TIM1 更新中断。
*   对于 TIM3，代码还配置了 PA6 引脚作为输入引脚，用于输入捕获。
*   对于 TIM4，代码使能了 TIM4 的时钟，并使能了 TIM4 的中断。

**作用:**

该函数负责使能定时器的时钟、配置相关的 GPIO 引脚（如果需要）和使能定时器的中断。  它是 HAL 库的一部分，用于将硬件初始化与 HAL 库解耦。

---

```c
void HAL_TIM_PWM_MspInit(TIM_HandleTypeDef* tim_pwmHandle)
{

  if(tim_pwmHandle->Instance==TIM2)
  {
  /* USER CODE BEGIN TIM2_MspInit 0 */

  /* USER CODE END TIM2_MspInit 0 */
    /* TIM2 clock enable */
    __HAL_RCC_TIM2_CLK_ENABLE();
  /* USER CODE BEGIN TIM2_MspInit 1 */

  /* USER CODE END TIM2_MspInit 1 */
  }
}
void HAL_TIM_MspPostInit(TIM_HandleTypeDef* timHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(timHandle->Instance==TIM2)
  {
  /* USER CODE BEGIN TIM2_MspPostInit 0 */

  /* USER CODE END TIM2_MspPostInit 0 */

    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**TIM2 GPIO Configuration
    PB10     ------> TIM2_CH3
    PB11     ------> TIM2_CH4
    */
    GPIO_InitStruct.Pin = HW_ELEC_BPWM_Pin|HW_ELEC_APWM_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    __HAL_AFIO_REMAP_TIM2_PARTIAL_2();

  /* USER CODE BEGIN TIM2_MspPostInit 1 */

  /* USER CODE END TIM2_MspPostInit 1 */
  }

}
```

**解释:**

*   `void HAL_TIM_PWM_MspInit(TIM_HandleTypeDef* tim_pwmHandle)`: HAL 库 PWM 定时器的底层初始化函数。
*   `__HAL_RCC_TIM2_CLK_ENABLE();`: 使能 TIM2 的时钟。
*   `void HAL_TIM_MspPostInit(TIM_HandleTypeDef* timHandle)`: HAL 库定时器 MSP 后期初始化函数。 这个函数在基本的定时器初始化之后被调用，用于执行一些额外的硬件配置，例如 GPIO 的复用功能配置。
*   `__HAL_RCC_GPIOB_CLK_ENABLE();`: 使能 GPIOB 的时钟。
*   `GPIO_InitStruct.Pin = HW_ELEC_BPWM_Pin|HW_ELEC_APWM_Pin;`:  设置要配置的 GPIO 引脚。  `HW_ELEC_BPWM_Pin` 和 `HW_ELEC_APWM_Pin`  是宏定义，分别代表 PB10 和 PB11 引脚。
*   `GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;`: 设置 GPIO 模式为复用推挽输出。
*   `GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;`: 设置 GPIO 速度为高。
*   `HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);`:  初始化 GPIO。
*   `__HAL_AFIO_REMAP_TIM2_PARTIAL_2();`:  将 TIM2 的通道 3 和通道 4  部分重映射到 PB10 和 PB11 引脚。

**作用:**

`HAL_TIM_PWM_MspInit` 函数使能 PWM 定时器的时钟。`HAL_TIM_MspPostInit` 函数配置 TIM2 的通道 3 和通道 4 的输出引脚，并将它们配置为复用推挽输出模式，以便输出 PWM 信号。 此外，代码还执行了 TIM2 的部分重映射，将 PWM 输出映射到特定的 GPIO 引脚。

---

```c
void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* tim_baseHandle)
{

  if(tim_baseHandle->Instance==TIM1)
  {
  /* USER CODE BEGIN TIM1_MspDeInit 0 */

  /* USER CODE END TIM1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM1_CLK_DISABLE();

    /* TIM1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(TIM1_UP_IRQn);
  /* USER CODE BEGIN TIM1_MspDeInit 1 */

  /* USER CODE END TIM1_MspDeInit 1 */
  }
  else if(tim_baseHandle->Instance==TIM3)
  {
  /* USER CODE BEGIN TIM3_MspDeInit 0 */

  /* USER CODE END TIM3_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM3_CLK_DISABLE();

    /**TIM3 GPIO Configuration
    PA6     ------> TIM3_CH1
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_6);

    /* TIM3 interrupt Deinit */
    HAL_NVIC_DisableIRQ(TIM3_IRQn);
  /* USER CODE BEGIN TIM3_MspDeInit 1 */

  /* USER CODE END TIM3_MspDeInit 1 */
  }
  else if(tim_baseHandle->Instance==TIM4)
  {
  /* USER CODE BEGIN TIM4_MspDeInit 0 */

  /* USER CODE END TIM4_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM4_CLK_DISABLE();

    /* TIM4 interrupt Deinit */
    HAL_NVIC_DisableIRQ(TIM4_IRQn);
  /* USER CODE BEGIN TIM4_MspDeInit 1 */

  /* USER CODE END TIM4_MspDeInit 1 */
  }
}

void HAL_TIM_PWM_MspDeInit(TIM_HandleTypeDef* tim_pwmHandle)
{

  if(tim_pwmHandle->Instance==TIM2)
  {
  /* USER CODE BEGIN TIM2_MspDeInit 0 */

  /* USER CODE END TIM2_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM2_CLK_DISABLE();
  /* USER CODE BEGIN TIM2_MspDeInit 1 */

  /* USER CODE END TIM2_MspDeInit 1 */
  }
}
```

**解释:**

*   `void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* tim_baseHandle)`: HAL 库定时器底层反初始化函数。 该函数用于释放定时器资源，例如禁用时钟、取消 GPIO 配置和禁用中断。
*   `__HAL_RCC_TIM1_CLK_DISABLE();`: 禁用 TIM1 的时钟。
*   `HAL_NVIC_DisableIRQ(TIM1_UP_IRQn);`: 禁用 TIM1 更新中断。
*   `HAL_GPIO_DeInit(GPIOA, GPIO_PIN_6);`: 将 PA6 引脚恢复到默认状态。
*   `void HAL_TIM_PWM_MspDeInit(TIM_HandleTypeDef* tim_pwmHandle)`: HAL 库 PWM 定时器的底层反初始化函数。
*   `__HAL_RCC_TIM2_CLK_DISABLE();`: 禁用 TIM2 的时钟。

**作用:**

这些函数负责释放定时器占用的资源，以便在不再需要定时器时，可以节省功耗并避免潜在的冲突。 它们是 HAL 库的一部分，用于将硬件反初始化与 HAL 库解耦。

**总结:**

该代码是 STM32 HAL 库中定时器初始化和反初始化的示例。它展示了如何配置不同的定时器，包括基本定时器、PWM 定时器和输入捕获定时器，并说明了如何使能和禁用定时器的时钟和中断。 这些函数通常由 STM32CubeMX 工具生成，并且可以根据具体的应用需求进行修改。