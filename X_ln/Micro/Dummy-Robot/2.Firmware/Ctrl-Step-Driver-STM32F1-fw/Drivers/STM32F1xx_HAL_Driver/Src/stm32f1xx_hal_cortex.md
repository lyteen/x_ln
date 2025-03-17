Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_cortex.c`

好的，我们重新开始。 这次我会提供更详细的解释，并以中文进行注释。

**概述**

这个 `stm32f1xx_hal_cortex.c` 文件是 STM32F1 系列微控制器 HAL (Hardware Abstraction Layer) 库的一部分，专门用于 CORTEX 内核的驱动。 它提供了一组函数来管理 CORTEX 内核的功能，如中断控制器 (NVIC)、系统定时器 (SysTick) 和内存保护单元 (MPU)。

**1. 中断配置 (NVIC)**

这部分代码提供了配置 NVIC (Nested Vectored Interrupt Controller) 的函数，用于管理中断。

```c
/**
  * @brief  Sets the priority grouping field (preemption priority and subpriority)
  *         using the required unlock sequence.
  * @param  PriorityGroup: The priority grouping bits length. 
  *         ... (参数描述，省略) ...
  * @retval None
  */
void HAL_NVIC_SetPriorityGrouping(uint32_t PriorityGroup)
{
  /* Check the parameters 检查参数 */
  assert_param(IS_NVIC_PRIORITY_GROUP(PriorityGroup));
  
  /* Set the PRIGROUP[10:8] bits according to the PriorityGroup parameter value
   * 根据 PriorityGroup 参数值设置 PRIGROUP[10:8] 位
   */
  NVIC_SetPriorityGrouping(PriorityGroup);
}
```

**描述:**

*   `HAL_NVIC_SetPriorityGrouping()`:  设置中断优先级分组。 优先级分组决定了抢占优先级和子优先级的位数分配。
*   `PriorityGroup`:  指定优先级分组。  例如，`NVIC_PRIORITYGROUP_0` 表示没有抢占优先级，所有位都用于子优先级。 `NVIC_PRIORITYGROUP_4` 表示所有位都用于抢占优先级，没有子优先级。
*   `assert_param(IS_NVIC_PRIORITY_GROUP(PriorityGroup))`:  一个断言宏，用于检查 `PriorityGroup` 参数是否有效。这是个良好的编程习惯，用于尽早发现错误。
*   `NVIC_SetPriorityGrouping(PriorityGroup)`:  这是一个 CMSIS 函数，用于实际设置 NVIC 的优先级分组。

**用法示例:**

```c
HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4); // 所有中断优先级都用于抢占优先级
```

在这个例子中，所有中断优先级位都被分配给抢占优先级。 这意味着更高抢占优先级的中断始终会抢占较低抢占优先级的中断。

```c
/**
  * @brief  Sets the priority of an interrupt.
  * @param  IRQn: External interrupt number.
  *         ... (参数描述，省略) ...
  * @retval None
  */
void HAL_NVIC_SetPriority(IRQn_Type IRQn, uint32_t PreemptPriority, uint32_t SubPriority)
{ 
  uint32_t prioritygroup = 0x00U;
  
  /* Check the parameters 检查参数*/
  assert_param(IS_NVIC_SUB_PRIORITY(SubPriority));
  assert_param(IS_NVIC_PREEMPTION_PRIORITY(PreemptPriority));
  
  prioritygroup = NVIC_GetPriorityGrouping();
  
  NVIC_SetPriority(IRQn, NVIC_EncodePriority(prioritygroup, PreemptPriority, SubPriority));
}
```

**描述:**

*   `HAL_NVIC_SetPriority()`: 设置特定中断的优先级。
*   `IRQn`:  中断请求号。 定义在 `stm32f10xx.h` (或其他对应的设备头文件) 中，例如 `TIM2_IRQn`、`USART1_IRQn`。
*   `PreemptPriority`: 抢占优先级。  值越小，优先级越高。
*   `SubPriority`: 子优先级。 在抢占优先级相同的情况下，子优先级决定了中断处理的顺序。 值越小，优先级越高。
*   `NVIC_GetPriorityGrouping()`: 获取当前设置的优先级分组。
*   `NVIC_EncodePriority()`:  这是一个 CMSIS 函数，根据优先级分组、抢占优先级和子优先级来编码最终的优先级值。
*   `NVIC_SetPriority()`:  这是一个 CMSIS 函数，用于设置特定中断的优先级。

**用法示例:**

```c
HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4); // 配置优先级分组
HAL_NVIC_SetPriority(TIM2_IRQn, 0, 0);          // 设置 TIM2 中断具有最高的抢占优先级
HAL_NVIC_SetPriority(USART1_IRQn, 1, 0);        // 设置 USART1 中断具有次高的抢占优先级
```

在这个例子中，`TIM2_IRQn` 被赋予最高的抢占优先级 (0)，而 `USART1_IRQn` 被赋予次高的抢占优先级 (1)。 如果两个中断同时发生，TIM2 中断将抢占 USART1 中断的处理。

```c
/**
  * @brief  Enables a device specific interrupt in the NVIC interrupt controller.
  * @note   To configure interrupts priority correctly, the NVIC_PriorityGroupConfig()
  *         function should be called before. 
  * @param  IRQn External interrupt number.
  *         ... (参数描述，省略) ...
  * @retval None
  */
void HAL_NVIC_EnableIRQ(IRQn_Type IRQn)
{
  /* Check the parameters 检查参数*/
  assert_param(IS_NVIC_DEVICE_IRQ(IRQn));

  /* Enable interrupt 使能中断*/
  NVIC_EnableIRQ(IRQn);
}
```

**描述:**

*   `HAL_NVIC_EnableIRQ()`:  使能特定的中断。
*   `IRQn`: 要使能的中断请求号。
*   `NVIC_EnableIRQ()`:  这是一个 CMSIS 函数，用于实际使能中断。

**用法示例:**

```c
HAL_NVIC_EnableIRQ(TIM2_IRQn); // 使能 TIM2 中断
```

在这个例子中，`TIM2_IRQn` 被使能，允许 TIM2 定时器生成中断。

```c
/**
  * @brief  Disables a device specific interrupt in the NVIC interrupt controller.
  * @param  IRQn External interrupt number.
  *         ... (参数描述，省略) ...
  * @retval None
  */
void HAL_NVIC_DisableIRQ(IRQn_Type IRQn)
{
  /* Check the parameters 检查参数*/
  assert_param(IS_NVIC_DEVICE_IRQ(IRQn));

  /* Disable interrupt 禁止中断*/
  NVIC_DisableIRQ(IRQn);
}
```

**描述:**

*   `HAL_NVIC_DisableIRQ()`:  禁止特定的中断。
*   `IRQn`: 要禁止的中断请求号。
*   `NVIC_DisableIRQ()`:  这是一个 CMSIS 函数，用于实际禁止中断。

**用法示例:**

```c
HAL_NVIC_DisableIRQ(TIM2_IRQn); // 禁止 TIM2 中断
```

在这个例子中，`TIM2_IRQn` 被禁止，阻止 TIM2 定时器生成中断。

**2. 系统定时器 (SysTick)**

这部分代码提供了配置 SysTick 定时器的函数，通常用于生成操作系统的心跳或提供简单的延时功能。

```c
/**
  * @brief  Initializes the System Timer and its interrupt, and starts the System Tick Timer.
  *         Counter is in free running mode to generate periodic interrupts.
  * @param  TicksNumb: Specifies the ticks Number of ticks between two interrupts.
  * @retval status:  - 0  Function succeeded.
  *                  - 1  Function failed.
  */
uint32_t HAL_SYSTICK_Config(uint32_t TicksNumb)
{
   return SysTick_Config(TicksNumb);
}
```

**描述:**

*   `HAL_SYSTICK_Config()`:  配置和启动 SysTick 定时器。
*   `TicksNumb`:  指定两个中断之间发生的时钟节拍数。 这个值决定了中断发生的频率。
*   `SysTick_Config()`:  这是一个 CMSIS 函数，用于实际配置 SysTick 定时器。

**用法示例:**

```c
HAL_SYSTICK_Config(SystemCoreClock / 1000); // 每毫秒产生一个中断 (假设 SystemCoreClock 是系统时钟频率)
```

在这个例子中，SysTick 定时器被配置为每毫秒产生一个中断。  `SystemCoreClock` 是系统时钟频率。

```c
/**
  * @brief  Configures the SysTick clock source.
  * @param  CLKSource: specifies the SysTick clock source.
  *         ... (参数描述，省略) ...
  * @retval None
  */
void HAL_SYSTICK_CLKSourceConfig(uint32_t CLKSource)
{
  /* Check the parameters 检查参数*/
  assert_param(IS_SYSTICK_CLK_SOURCE(CLKSource));
  if (CLKSource == SYSTICK_CLKSOURCE_HCLK)
  {
    SysTick->CTRL |= SYSTICK_CLKSOURCE_HCLK;
  }
  else
  {
    SysTick->CTRL &= ~SYSTICK_CLKSOURCE_HCLK;
  }
}
```

**描述:**

*   `HAL_SYSTICK_CLKSourceConfig()`:  配置 SysTick 定时器的时钟源。
*   `CLKSource`:  指定时钟源。  可以是 `SYSTICK_CLKSOURCE_HCLK` (系统时钟) 或 `SYSTICK_CLKSOURCE_HCLK_DIV8` (系统时钟的 1/8)。
*   `SysTick->CTRL`:  SysTick 控制寄存器。

**用法示例:**

```c
HAL_SYSTICK_CLKSourceConfig(SYSTICK_CLKSOURCE_HCLK); // 使用系统时钟作为 SysTick 时钟源
```

在这个例子中，SysTick 定时器使用系统时钟作为其时钟源。

```c
/**
  * @brief  This function handles SYSTICK interrupt request.
  * @retval None
  */
void HAL_SYSTICK_IRQHandler(void)
{
  HAL_SYSTICK_Callback();
}

/**
  * @brief  SYSTICK callback.
  * @retval None
  */
__weak void HAL_SYSTICK_Callback(void)
{
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SYSTICK_Callback could be implemented in the user file
   */
}
```

**描述:**

*   `HAL_SYSTICK_IRQHandler()`:  SysTick 中断处理函数。  当 SysTick 定时器计数到零时，会触发这个中断。
*   `HAL_SYSTICK_Callback()`:  一个弱函数 (weak function)，允许用户在自己的代码中重新定义它。  这是在 SysTick 中断中执行用户代码的地方。 `__weak` 关键字允许你在不修改 HAL 库文件的情况下覆盖这个函数。

**用法示例:**

```c
// 在你的代码中重新定义 HAL_SYSTICK_Callback
void HAL_SYSTICK_Callback(void)
{
  // 在这里编写你的 SysTick 中断处理代码
  // 例如，递增一个全局计数器，或执行其他定时任务
  static uint32_t tick = 0;
  tick++;
  if (tick >= 1000) {
    // 每秒执行一次
    tick = 0;
    //  ...
  }
}
```

在这个例子中，`HAL_SYSTICK_Callback()` 被重新定义，以在每次 SysTick 中断发生时递增一个全局计数器 `tick`。  当 `tick` 达到 1000 时 (假设 SysTick 中断每毫秒发生一次)，执行一些其他的任务。

**3. 内存保护单元 (MPU)**

这部分代码提供了配置 MPU 的函数，用于保护内存区域免受未经授权的访问。 这在需要保护关键数据或代码的嵌入式系统中非常重要。  这部分代码只在定义了 `__MPU_PRESENT` 宏时才会被编译，这意味着你的 STM32F1 设备必须实际支持 MPU 才能使用这些功能。

```c
#if (__MPU_PRESENT == 1U)
/**
  * @brief  Disables the MPU
  * @retval None
  */
void HAL_MPU_Disable(void)
{
  /* Make sure outstanding transfers are done */
  __DMB();

  /* Disable fault exceptions */
  SCB->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk;
  
  /* Disable the MPU and clear the control register*/
  MPU->CTRL = 0U;
}
```

**描述:**

*   `HAL_MPU_Disable()`:  禁用 MPU。
*   `__DMB()`:  数据内存屏障指令。  确保所有未完成的内存传输都已完成。
*   `SCB->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk`:  禁用内存故障异常。
*   `MPU->CTRL = 0U`:  清除 MPU 控制寄存器，禁用 MPU。

```c
/**
  * @brief  Enable the MPU.
  * @param  MPU_Control: Specifies the control mode of the MPU during hard fault, 
  *          NMI, FAULTMASK and privileged access to the default memory 
  *          ... (参数描述，省略) ...
  * @retval None
  */
void HAL_MPU_Enable(uint32_t MPU_Control)
{
  /* Enable the MPU */
  MPU->CTRL = MPU_Control | MPU_CTRL_ENABLE_Msk;
  
  /* Enable fault exceptions */
  SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
  
  /* Ensure MPU setting take effects */
  __DSB();
  __ISB();
}
```

**描述:**

*   `HAL_MPU_Enable()`:  启用 MPU。
*   `MPU_Control`:  指定 MPU 的控制模式。  例如，`MPU_HFNMI_PRIVDEF_NONE` 表示在硬故障、NMI 等情况下不使用 MPU。
*   `MPU->CTRL = MPU_Control | MPU_CTRL_ENABLE_Msk`:  设置 MPU 控制寄存器，启用 MPU。
*   `SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk`:  启用内存故障异常。
*   `__DSB()`:  数据同步屏障指令。  确保 MPU 设置生效。
*   `__ISB()`:  指令同步屏障指令。  刷新流水线，确保新的 MPU 设置被正确应用。

```c
/**
  * @brief  Initializes and configures the Region and the memory to be protected.
  * @param  MPU_Init: Pointer to a MPU_Region_InitTypeDef structure that contains
  *                the initialization and configuration information.
  * @retval None
  */
void HAL_MPU_ConfigRegion(MPU_Region_InitTypeDef *MPU_Init)
{
  /* Check the parameters 检查参数*/
  assert_param(IS_MPU_REGION_NUMBER(MPU_Init->Number));
  assert_param(IS_MPU_REGION_ENABLE(MPU_Init->Enable));

  /* Set the Region number 设置区域编号*/
  MPU->RNR = MPU_Init->Number;

  if ((MPU_Init->Enable) != RESET)
  {
    /* Check the parameters 检查参数*/
    assert_param(IS_MPU_INSTRUCTION_ACCESS(MPU_Init->DisableExec));
    assert_param(IS_MPU_REGION_PERMISSION_ATTRIBUTE(MPU_Init->AccessPermission));
    assert_param(IS_MPU_TEX_LEVEL(MPU_Init->TypeExtField));
    assert_param(IS_MPU_ACCESS_SHAREABLE(MPU_Init->IsShareable));
    assert_param(IS_MPU_ACCESS_CACHEABLE(MPU_Init->IsCacheable));
    assert_param(IS_MPU_ACCESS_BUFFERABLE(MPU_Init->IsBufferable));
    assert_param(IS_MPU_SUB_REGION_DISABLE(MPU_Init->SubRegionDisable));
    assert_param(IS_MPU_REGION_SIZE(MPU_Init->Size));
    
    MPU->RBAR = MPU_Init->BaseAddress;
    MPU->RASR = ((uint32_t)MPU_Init->DisableExec             << MPU_RASR_XN_Pos)   |
                ((uint32_t)MPU_Init->AccessPermission        << MPU_RASR_AP_Pos)   |
                ((uint32_t)MPU_Init->TypeExtField            << MPU_RASR_TEX_Pos)  |
                ((uint32_t)MPU_Init->IsShareable             << MPU_RASR_S_Pos)    |
                ((uint32_t)MPU_Init->IsCacheable             << MPU_RASR_C_Pos)    |
                ((uint32_t)MPU_Init->IsBufferable            << MPU_RASR_B_Pos)    |
                ((uint32_t)MPU_Init->SubRegionDisable        << MPU_RASR_SRD_Pos)  |
                ((uint32_t)MPU_Init->Size                    << MPU_RASR_SIZE_Pos) |
                ((uint32_t)MPU_Init->Enable                  << MPU_RASR_ENABLE_Pos);
  }
  else
  {
    MPU->RBAR = 0x00U;
    MPU->RASR = 0x00U;
  }
}
#endif /* __MPU_PRESENT */
```

**描述:**

*   `HAL_MPU_ConfigRegion()`:  配置 MPU 区域。
*   `MPU_Init`:  指向 `MPU_Region_InitTypeDef` 结构的指针，该结构包含 MPU 区域的配置信息。  这个结构体定义了区域的基地址、大小、权限等。

**MPU_Region_InitTypeDef 的成员通常包括:**

*   `Number`:  区域编号 (0-7)。
*   `Enable`:  是否启用该区域。
*   `BaseAddress`:  区域的起始地址。
*   `Size`:  区域的大小 (32 字节到 4GB)。
*   `DisableExec`:  是否禁止在该区域执行指令。  用于防止代码在数据区域执行。
*   `AccessPermission`:  访问权限 (只读、只写、读写、特权访问等)。
*   `TypeExtField`, `IsShareable`, `IsCacheable`, `IsBufferable`, `SubRegionDisable`:  更高级的属性，用于控制内存类型、共享、缓存和缓冲。

**用法示例:**

```c
#if (__MPU_PRESENT == 1U)
MPU_Region_InitTypeDef MPU_InitStruct;

// 配置 MPU 保护一个 64KB 的内存区域
MPU_InitStruct.Number = MPU_REGION_NUMBER0;        // 区域 0
MPU_InitStruct.Enable = MPU_REGION_ENABLE;          // 启用区域
MPU_InitStruct.BaseAddress = 0x20000000;           // 从 SRAM 的起始地址开始
MPU_InitStruct.Size = MPU_REGION_SIZE_64KB;          // 区域大小为 64KB
MPU_InitStruct.AccessPermission = MPU_REGION_PRIVILEGED_RW; // 特权模式下读写
MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE; // 允许执行指令

HAL_MPU_ConfigRegion(&MPU_InitStruct);           // 配置 MPU 区域
HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);          // 启用 MPU
#endif
```

在这个例子中，MPU 被配置为保护从 SRAM 的起始地址 (0x20000000) 开始的 64KB 内存区域。  只有在特权模式下才能读写这个区域，并且允许在该区域执行指令。`MPU_PRIVILEGED_DEFAULT` 确保在特权模式下 MPU 被启用。

**4. 其他功能**

*   `HAL_NVIC_SystemReset()`:  发起系统复位。
*   `HAL_NVIC_GetPriorityGrouping()`: 获取当前的优先级分组。
*   `HAL_NVIC_GetPriority()`: 获取指定中断的优先级。
*   `HAL_NVIC_SetPendingIRQ()`:  设置指定中断的挂起位。
*   `HAL_NVIC_GetPendingIRQ()`:  获取指定中断的挂起状态。
*   `HAL_NVIC_ClearPendingIRQ()`:  清除指定中断的挂起位。
*   `HAL_NVIC_GetActive()`:  获取指定中断的激活状态。

这些函数提供了对 NVIC 和 SysTick 的更细粒度的控制，例如可以手动设置和清除中断的挂起位，或者查询中断的激活状态。

**总结**

`stm32f1xx_hal_cortex.c` 文件提供了一组关键的函数，用于配置 STM32F1 微控制器的 CORTEX 内核。  通过使用这些函数，你可以管理中断、配置系统定时器和保护内存区域，从而构建更可靠和安全的嵌入式系统。 理解这些函数的工作原理对于开发 STM32F1 应用程序至关重要。 这段代码实现了对 NVIC, SYSTICK, MPU 等 CORTEX 内核组件的底层访问和控制，构成了 HAL 层中非常重要的一部分。
