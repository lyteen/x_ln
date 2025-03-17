Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal.c`

好的，让我们重新开始。我将对提供的 STM32F1xx HAL 库代码进行关键部分的代码提取、中文解释、使用说明和简易示例。

**1. HAL 初始化函数 (`HAL_Init`)**

```c
/**
  * @brief  This function is used to initialize the HAL Library; it must be the first
  *         instruction to be executed in the main program (before to call any other
  *         HAL function), it performs the following:
  *           Configure the Flash prefetch.
  *           Configures the SysTick to generate an interrupt each 1 millisecond,
  *           which is clocked by the HSI (at this stage, the clock is not yet
  *           configured and thus the system is running from the internal HSI at 16 MHz).
  *           Set NVIC Group Priority to 4.
  *           Calls the HAL_MspInit() callback function defined in user file
  *           "stm32f1xx_hal_msp.c" to do the global low level hardware initialization
  *
  * @note   SysTick is used as time base for the HAL_Delay() function, the application
  *         need to ensure that the SysTick time base is always set to 1 millisecond
  *         to have correct HAL operation.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_Init(void)
{
  /* Configure Flash prefetch (如果使能)*/
#if (PREFETCH_ENABLE != 0)
#if defined(STM32F101x6) || defined(STM32F101xB) || defined(STM32F101xE) || defined(STM32F101xG) || \
    defined(STM32F102x6) || defined(STM32F102xB) || \
    defined(STM32F103x6) || defined(STM32F103xB) || defined(STM32F103xE) || defined(STM32F103xG) || \
    defined(STM32F105xC) || defined(STM32F107xC)

  /* Prefetch buffer is not available on value line devices */
  __HAL_FLASH_PREFETCH_BUFFER_ENABLE();
#endif
#endif /* PREFETCH_ENABLE */

  /* Set Interrupt Group Priority 设置中断优先级分组 */
  HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4);

  /* Use systick as time base source and configure 1ms tick (default clock after Reset is HSI) 使用SysTick作为时间基准，配置1ms的tick */
  HAL_InitTick(TICK_INT_PRIORITY);

  /* Init the low level hardware 初始化底层硬件*/
  HAL_MspInit();

  /* Return function status 返回函数状态*/
  return HAL_OK;
}
```

**描述:** `HAL_Init()` 是 HAL 库的初始化函数，必须在 `main` 函数的最开始调用。 它执行以下操作：

*   **配置 Flash 预取:** 提高 Flash 访问速度，如果芯片支持。
*   **设置中断优先级分组:**  将 NVIC 的中断优先级分组设置为 4。
*   **配置 SysTick 定时器:**  配置 SysTick 定时器，使其每 1 毫秒产生一个中断，作为 HAL 库的时间基准。
*   **调用 `HAL_MspInit()`:**  调用用户定义的 `HAL_MspInit()` 函数，进行底层硬件初始化（时钟、GPIO 等）。

**使用方法:**

```c
int main(void)
{
  HAL_Init(); // 首先调用 HAL_Init()
  // ... 其他代码 ...
}

// 用户需要在 stm32f1xx_hal_msp.c 中实现 HAL_MspInit()
void HAL_MspInit(void)
{
  // 在这里初始化时钟、GPIO 等
}
```

**简易示例:**  在 `HAL_MspInit()` 中配置一个 LED 的 GPIO 引脚。

**2. 延时函数 (`HAL_Delay`)**

```c
/**
  * @brief This function provides minimum delay (in milliseconds) based
  *        on variable incremented.
  * @note In the default implementation , SysTick timer is the source of time base.
  *       It is used to generate interrupts at regular time intervals where uwTick
  *       is incremented.
  * @note This function is declared as __weak to be overwritten in case of other
  *       implementations in user file.
  * @param Delay specifies the delay time length, in milliseconds.
  * @retval None
  */
__weak void HAL_Delay(uint32_t Delay)
{
  uint32_t tickstart = HAL_GetTick();
  uint32_t wait = Delay;

  /* Add a freq to guarantee minimum wait */
  if (wait < HAL_MAX_DELAY)
  {
    wait += (uint32_t)(uwTickFreq);
  }

  while ((HAL_GetTick() - tickstart) < wait)
  {
  }
}
```

**描述:** `HAL_Delay()` 函数提供了一个阻塞式延时功能，单位为毫秒。它基于 `SysTick` 定时器实现。

**使用方法:**

```c
int main(void)
{
  HAL_Init();

  // ... 初始化 LED 的 GPIO 引脚 ...

  while (1)
  {
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET); // 点亮 LED
    HAL_Delay(500); // 延时 500 毫秒
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET); // 熄灭 LED
    HAL_Delay(500); // 延时 500 毫秒
  }
}
```

**简易示例:**  使用 `HAL_Delay()` 控制 LED 的闪烁频率。

**3. 获取 Tick 值的函数 (`HAL_GetTick`)**

```c
/**
  * @brief Provides a tick value in millisecond.
  * @note  This function is declared as __weak to be overwritten in case of other
  *       implementations in user file.
  * @retval tick value
  */
__weak uint32_t HAL_GetTick(void)
{
  return uwTick;
}
```

**描述:** `HAL_GetTick()` 函数返回自系统启动以来经过的毫秒数。 `uwTick` 变量在 `SysTick` 中断服务程序中递增。

**使用方法:**  可以用来测量代码的执行时间。

```c
int main(void)
{
  HAL_Init();

  uint32_t start_time = HAL_GetTick();

  // ... 要测量执行时间的代码 ...

  uint32_t end_time = HAL_GetTick();
  uint32_t execution_time = end_time - start_time;

  // execution_time 单位为毫秒
}
```

**简易示例:**  测量一段代码的执行时间。

**4. SysTick 中断处理函数 (`HAL_IncTick`)**

```c
/**
  * @brief This function is called to increment  a global variable "uwTick"
  *        used as application time base.
  * @note In the default implementation, this variable is incremented each 1ms
  *       in SysTick ISR.
  * @note This function is declared as __weak to be overwritten in case of other
  *      implementations in user file.
  * @retval None
  */
__weak void HAL_IncTick(void)
{
  uwTick += uwTickFreq;
}
```

**描述:**  `HAL_IncTick()` 函数在 `SysTick` 中断服务程序 (ISR) 中被调用，用于递增全局变量 `uwTick`。 `uwTick` 作为应用程序的时间基准。

**使用方法:**  无需直接调用，由 `SysTick` 中断自动调用。

**5. MSP 初始化和反初始化函数 (`HAL_MspInit`, `HAL_MspDeInit`)**

```c
/**
  * @brief  Initialize the MSP.
  * @retval None
  */
__weak void HAL_MspInit(void)
{
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes the MSP.
  * @retval None
  */
__weak void HAL_MspDeInit(void)
{
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_MspDeInit could be implemented in the user file
   */
}
```

**描述:**  `HAL_MspInit()` 和 `HAL_MspDeInit()` 是弱函数，需要在用户文件中 (`stm32f1xx_hal_msp.c`) 定义。  它们分别用于初始化和反初始化 MCU 的底层硬件资源，例如时钟配置、GPIO 配置、DMA 配置等。`HAL_MspInit()` 由 `HAL_Init()` 调用。

**使用方法:**  在 `stm32f1xx_hal_msp.c` 文件中实现这两个函数。

```c
// stm32f1xx_hal_msp.c
void HAL_MspInit(void)
{
  // 1. 使能时钟
  __HAL_RCC_GPIOA_CLK_ENABLE();

  // 2. 配置 GPIO
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

void HAL_MspDeInit(void)
{
  // 反初始化 GPIO 和时钟
  HAL_GPIO_DeInit(GPIOA, GPIO_PIN_5);
  __HAL_RCC_GPIOA_CLK_DISABLE();
}
```

**简易示例:**  配置一个 LED 的 GPIO 引脚的时钟和模式。

**总结:**

这些是 `stm32f1xx_hal.c` 文件中一些关键部分的解释和示例。 该文件提供了 STM32F1xx 系列 MCU 的 HAL 库的基础， 允许开发者使用标准化的 API 来初始化硬件并进行操作。 请记住，用户通常需要在 `stm32f1xx_hal_msp.c` 文件中提供底层硬件初始化，以使 HAL 库正常工作。
