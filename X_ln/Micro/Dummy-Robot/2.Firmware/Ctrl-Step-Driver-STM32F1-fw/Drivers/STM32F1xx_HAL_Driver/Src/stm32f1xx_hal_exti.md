Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_exti.c`

好的，我们重新开始，这次的回答将更注重代码的功能解释和使用方法，并配合简单的中文描述和演示。

**1. `HAL_EXTI_SetConfigLine(EXTI_HandleTypeDef *hexti, EXTI_ConfigTypeDef *pExtiConfig)`**

```c
HAL_StatusTypeDef HAL_EXTI_SetConfigLine(EXTI_HandleTypeDef *hexti, EXTI_ConfigTypeDef *pExtiConfig)
{
  uint32_t regval;
  uint32_t linepos;
  uint32_t maskline;

  /* Check null pointer */
  if ((hexti == NULL) || (pExtiConfig == NULL))
  {
    return HAL_ERROR;
  }

  /* Check parameters */
  assert_param(IS_EXTI_LINE(pExtiConfig->Line));
  assert_param(IS_EXTI_MODE(pExtiConfig->Mode));

  /* Assign line number to handle */
  hexti->Line = pExtiConfig->Line;

  /* Compute line mask */
  linepos = (pExtiConfig->Line & EXTI_PIN_MASK);
  maskline = (1uL << linepos);

  /* Configure triggers for configurable lines */
  if ((pExtiConfig->Line & EXTI_CONFIG) != 0x00u)
  {
    assert_param(IS_EXTI_TRIGGER(pExtiConfig->Trigger));

    /* Configure rising trigger */
    /* Mask or set line */
    if ((pExtiConfig->Trigger & EXTI_TRIGGER_RISING) != 0x00u)
    {
      EXTI->RTSR |= maskline;
    }
    else
    {
      EXTI->RTSR &= ~maskline;
    }

    /* Configure falling trigger */
    /* Mask or set line */
    if ((pExtiConfig->Trigger & EXTI_TRIGGER_FALLING) != 0x00u)
    {
      EXTI->FTSR |= maskline;
    }
    else
    {
      EXTI->FTSR &= ~maskline;
    }


    /* Configure gpio port selection in case of gpio exti line */
    if ((pExtiConfig->Line & EXTI_GPIO) == EXTI_GPIO)
    {
      assert_param(IS_EXTI_GPIO_PORT(pExtiConfig->GPIOSel));
      assert_param(IS_EXTI_GPIO_PIN(linepos));
      
      regval = AFIO->EXTICR[linepos >> 2u];
      regval &= ~(AFIO_EXTICR1_EXTI0 << (AFIO_EXTICR1_EXTI1_Pos * (linepos & 0x03u)));
      regval |= (pExtiConfig->GPIOSel << (AFIO_EXTICR1_EXTI1_Pos * (linepos & 0x03u)));
      AFIO->EXTICR[linepos >> 2u] = regval;
    }
  }

  /* Configure interrupt mode : read current mode */
  /* Mask or set line */
  if ((pExtiConfig->Mode & EXTI_MODE_INTERRUPT) != 0x00u)
  {
    EXTI->IMR |= maskline;
  }
  else
  {
    EXTI->IMR &= ~maskline;
  }

  /* Configure event mode : read current mode */
  /* Mask or set line */
  if ((pExtiConfig->Mode & EXTI_MODE_EVENT) != 0x00u)
  {
    EXTI->EMR |= maskline;
  }
  else
  {
    EXTI->EMR &= ~maskline;
  }

  return HAL_OK;
}
```

**描述:** 这个函数用于配置指定的EXTI线路。它接收一个EXTI句柄 `hexti` 和一个配置结构体 `pExtiConfig` 作为输入。它设置EXTI线路的模式（中断、事件或两者），触发类型（上升沿、下降沿或两者），以及当EXTI线路连接到GPIO时，选择哪个GPIO端口。

**主要步骤:**

*   **参数检查:** 检查传入的指针是否为空，并使用`assert_param`宏来验证配置参数的有效性。
*   **计算掩码:**  根据线路号计算位掩码，用于操作EXTI寄存器。
*   **配置触发器:** 如果线路是可配置的（`EXTI_CONFIG`），则根据`pExtiConfig->Trigger`设置上升沿触发寄存器（`EXTI->RTSR`）和下降沿触发寄存器（`EXTI->FTSR`）。
*   **GPIO端口选择:**  如果线路与GPIO相关联（`EXTI_GPIO`），则设置AFIO的EXTICR寄存器以选择正确的GPIO端口。这个端口决定了哪个GPIO引脚连接到这个EXTI线路。
*   **中断/事件模式配置:**  根据`pExtiConfig->Mode`设置中断屏蔽寄存器（`EXTI->IMR`）和事件屏蔽寄存器（`EXTI->EMR`）来启用或禁用中断和事件。

**如何使用:**

1.  **初始化EXTI句柄:** 创建一个`EXTI_HandleTypeDef`结构体变量。
2.  **配置EXTI_ConfigTypeDef:**  填充一个`EXTI_ConfigTypeDef`结构体变量，指定要配置的线路，模式，触发器等。
3.  **调用 HAL_EXTI_SetConfigLine():**  调用此函数，传入EXTI句柄和配置结构体。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;
EXTI_ConfigTypeDef    exti_config;

void EXTI_Config(void)
{
    // 1. 初始化 EXTI 句柄 (这里只是简单的赋个 line 值，实际使用还需要其他初始化步骤)
    exti_handle.Line = EXTI_LINE0;

    // 2. 配置 EXTI 结构体
    exti_config.Line    = EXTI_LINE0;
    exti_config.Mode    = EXTI_MODE_INTERRUPT;  // 设置为中断模式
    exti_config.Trigger = EXTI_TRIGGER_RISING;   // 上升沿触发
    exti_config.GPIOSel = EXTI_GPIOA;         // 选择 GPIOA (如果 EXTI0 连接到 GPIOA)

    // 3. 调用配置函数
    HAL_EXTI_SetConfigLine(&exti_handle, &exti_config);

    // 4. 使能 NVIC 中断 (这是配置中断的关键步骤，必须在HAL_EXTI_SetConfigLine之后)
    HAL_NVIC_EnableIRQ(EXTI0_IRQn);
}

// EXTI 中断处理函数 (需要在 stm32f1xx_it.c 中定义)
void EXTI0_IRQHandler(void)
{
    HAL_EXTI_IRQHandler(&exti_handle); // 调用 HAL 库的中断处理函数
}

// EXTI 回调函数 (在 HAL_EXTI_RegisterCallback 中注册)
void EXTI0_Callback(void)
{
    // 在这里处理中断事件，例如翻转一个 LED
    HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);  //假设 GPIOC Pin 13 连接到一个LED
}

int main(void)
{
    // ... 初始化系统时钟，GPIO 等 ...

    EXTI_Config(); // 配置 EXTI

    HAL_EXTI_RegisterCallback(&exti_handle, HAL_EXTI_COMMON_CB_ID, EXTI0_Callback);

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `EXTI_Config()` 函数负责配置EXTI线路。
*   `exti_handle` 用于存储EXTI线路的信息。
*   `exti_config` 用于指定EXTI线路的配置，例如：线路号，模式（中断/事件），触发方式（上升沿/下降沿），以及GPIO的选择。
*   `HAL_EXTI_SetConfigLine()` 函数实际进行配置。
*   `HAL_NVIC_EnableIRQ()` 使能中断控制器(NVIC)中对应的中断。**这是非常重要的一步，否则即使EXTI配置正确，也不会产生中断。**
*   `EXTI0_IRQHandler()` 是中断服务程序，需要在 `stm32f1xx_it.c` 文件中定义。它调用 `HAL_EXTI_IRQHandler()` 来处理中断。
*   `EXTI0_Callback()` 是用户定义的回调函数，在 `HAL_EXTI_RegisterCallback()` 中注册，当中断发生时，此函数会被执行。例如可以翻转一个连接到GPIO的LED，以指示中断发生。

**2. `HAL_EXTI_GetConfigLine(EXTI_HandleTypeDef *hexti, EXTI_ConfigTypeDef *pExtiConfig)`**

```c
HAL_StatusTypeDef HAL_EXTI_GetConfigLine(EXTI_HandleTypeDef *hexti, EXTI_ConfigTypeDef *pExtiConfig)
{
  uint32_t regval;
  uint32_t linepos;
  uint32_t maskline;

  /* Check null pointer */
  if ((hexti == NULL) || (pExtiConfig == NULL))
  {
    return HAL_ERROR;
  }

  /* Check the parameter */
  assert_param(IS_EXTI_LINE(hexti->Line));

  /* Store handle line number to configuration structure */
  pExtiConfig->Line = hexti->Line;

  /* Compute line mask */
  linepos = (pExtiConfig->Line & EXTI_PIN_MASK);
  maskline = (1uL << linepos);

  /* 1] Get core mode : interrupt */

  /* Check if selected line is enable */
  if ((EXTI->IMR & maskline) != 0x00u)
  {
    pExtiConfig->Mode = EXTI_MODE_INTERRUPT;
  }
  else
  {
    pExtiConfig->Mode = EXTI_MODE_NONE;
  }

  /* Get event mode */
  /* Check if selected line is enable */
  if ((EXTI->EMR & maskline) != 0x00u)
  {
    pExtiConfig->Mode |= EXTI_MODE_EVENT;
  }

  /* Get default Trigger and GPIOSel configuration */
  pExtiConfig->Trigger = EXTI_TRIGGER_NONE;
  pExtiConfig->GPIOSel = 0x00u;

  /* 2] Get trigger for configurable lines : rising */
  if ((pExtiConfig->Line & EXTI_CONFIG) != 0x00u)
  {
    /* Check if configuration of selected line is enable */
    if ((EXTI->RTSR & maskline) != 0x00u)
    {
      pExtiConfig->Trigger = EXTI_TRIGGER_RISING;
    }

    /* Get falling configuration */
    /* Check if configuration of selected line is enable */
    if ((EXTI->FTSR & maskline) != 0x00u)
    {
      pExtiConfig->Trigger |= EXTI_TRIGGER_FALLING;
    }

    /* Get Gpio port selection for gpio lines */
    if ((pExtiConfig->Line & EXTI_GPIO) == EXTI_GPIO)
    {
      assert_param(IS_EXTI_GPIO_PIN(linepos));

      regval = AFIO->EXTICR[linepos >> 2u];
      pExtiConfig->GPIOSel = ((regval << (AFIO_EXTICR1_EXTI1_Pos * (3uL - (linepos & 0x03u)))) >> 24);
    }
  }

  return HAL_OK;
}
```

**描述:**  这个函数用于获取指定EXTI线路的当前配置。它接收一个EXTI句柄 `hexti` 和一个指向 `EXTI_ConfigTypeDef` 结构体的指针 `pExtiConfig` 作为输入。函数读取相关的EXTI寄存器，并将当前配置信息填充到 `pExtiConfig` 结构体中。

**主要步骤:**

*   **参数检查:** 检查输入指针是否为空，并使用`assert_param`宏来验证`hexti->Line`的有效性。
*   **计算掩码:** 根据线路号计算位掩码，用于读取EXTI寄存器。
*   **获取模式:** 读取`EXTI->IMR`和`EXTI->EMR`寄存器，确定线路是否配置为中断模式，事件模式或两者。
*   **获取触发器配置:** 读取`EXTI->RTSR`和`EXTI->FTSR`寄存器，确定线路是否配置为上升沿触发，下降沿触发或两者。
*   **获取GPIO端口选择:**  读取`AFIO->EXTICR`寄存器，确定与该EXTI线路关联的GPIO端口。

**如何使用:**

1.  **初始化EXTI句柄:**  创建一个`EXTI_HandleTypeDef`结构体变量，并设置`Line`成员，指定要读取配置的线路。
2.  **创建EXTI_ConfigTypeDef结构体:** 创建一个`EXTI_ConfigTypeDef`结构体变量，用于存储读取到的配置信息。
3.  **调用 HAL_EXTI_GetConfigLine():** 调用此函数，传入EXTI句柄和配置结构体指针。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;
EXTI_ConfigTypeDef    exti_config;

void Read_EXTI_Config(void)
{
    // 1. 初始化 EXTI 句柄
    exti_handle.Line = EXTI_LINE0;

    // 2. 调用获取配置函数
    if (HAL_EXTI_GetConfigLine(&exti_handle, &exti_config) == HAL_OK)
    {
        // 3. 打印配置信息 (这里只是简单的打印，实际使用中可以根据需要进行处理)
        printf("EXTI Line: %lu\r\n", exti_config.Line);
        printf("Mode: %lu\r\n", exti_config.Mode);
        printf("Trigger: %lu\r\n", exti_config.Trigger);
        printf("GPIOSel: %lu\r\n", exti_config.GPIOSel);
    }
    else
    {
        printf("Failed to get EXTI configuration.\r\n");
    }
}

int main(void)
{
    // ... 初始化系统时钟，串口等 ...

    Read_EXTI_Config(); // 读取 EXTI 配置

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `Read_EXTI_Config()` 函数负责读取EXTI线路的配置。
*   首先初始化 `exti_handle.Line`，指定要读取配置的线路。
*   `HAL_EXTI_GetConfigLine()` 函数读取配置，并将结果存储到 `exti_config` 结构体中。
*   然后，可以打印或者使用 `exti_config` 中的信息。

**3. `HAL_EXTI_ClearConfigLine(EXTI_HandleTypeDef *hexti)`**

```c
HAL_StatusTypeDef HAL_EXTI_ClearConfigLine(EXTI_HandleTypeDef *hexti)
{
  uint32_t regval;
  uint32_t linepos;
  uint32_t maskline;

  /* Check null pointer */
  if (hexti == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameter */
  assert_param(IS_EXTI_LINE(hexti->Line));

  /* compute line mask */
  linepos = (hexti->Line & EXTI_PIN_MASK);
  maskline = (1uL << linepos);

  /* 1] Clear interrupt mode */
  EXTI->IMR = (EXTI->IMR & ~maskline);

  /* 2] Clear event mode */
  EXTI->EMR = (EXTI->EMR & ~maskline);

  /* 3] Clear triggers in case of configurable lines */
  if ((hexti->Line & EXTI_CONFIG) != 0x00u)
  {
    EXTI->RTSR = (EXTI->RTSR & ~maskline);
    EXTI->FTSR = (EXTI->FTSR & ~maskline);

    /* Get Gpio port selection for gpio lines */
    if ((hexti->Line & EXTI_GPIO) == EXTI_GPIO)
    {
      assert_param(IS_EXTI_GPIO_PIN(linepos));

      regval = AFIO->EXTICR[linepos >> 2u];
      regval &= ~(AFIO_EXTICR1_EXTI0 << (AFIO_EXTICR1_EXTI1_Pos * (linepos & 0x03u)));
      AFIO->EXTICR[linepos >> 2u] = regval;
    }
  }

  return HAL_OK;
}
```

**描述:** 这个函数用于清除指定EXTI线路的全部配置，包括中断/事件模式、触发器设置和GPIO端口选择。它接收一个EXTI句柄 `hexti` 作为输入。

**主要步骤:**

*   **参数检查:** 检查输入指针是否为空，并使用`assert_param`宏来验证`hexti->Line`的有效性。
*   **计算掩码:** 根据线路号计算位掩码，用于修改EXTI寄存器。
*   **清除中断/事件模式:** 通过与掩码取反后进行与操作，清除`EXTI->IMR`和`EXTI->EMR`寄存器中对应的位，禁用中断和事件模式。
*   **清除触发器配置:** 通过与掩码取反后进行与操作，清除`EXTI->RTSR`和`EXTI->FTSR`寄存器中对应的位，禁用上升沿和下降沿触发。
*   **清除GPIO端口选择:**  如果线路与GPIO相关联（`EXTI_GPIO`），则清除`AFIO->EXTICR`寄存器中对应的位，将GPIO端口选择恢复为默认值。

**如何使用:**

1.  **初始化EXTI句柄:** 创建一个`EXTI_HandleTypeDef`结构体变量，并设置`Line`成员，指定要清除配置的线路。
2.  **调用 HAL_EXTI_ClearConfigLine():** 调用此函数，传入EXTI句柄。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;

void Clear_EXTI_Config(void)
{
    // 1. 初始化 EXTI 句柄
    exti_handle.Line = EXTI_LINE0;

    // 2. 调用清除配置函数
    if (HAL_EXTI_ClearConfigLine(&exti_handle) == HAL_OK)
    {
        printf("EXTI configuration cleared successfully.\r\n");
    }
    else
    {
        printf("Failed to clear EXTI configuration.\r\n");
    }
}

int main(void)
{
    // ... 初始化系统时钟，串口等 ...

    Clear_EXTI_Config(); // 清除 EXTI 配置

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `Clear_EXTI_Config()` 函数负责清除EXTI线路的配置。
*   首先初始化 `exti_handle.Line`，指定要清除配置的线路。
*   `HAL_EXTI_ClearConfigLine()` 函数执行清除操作。

**4. `HAL_EXTI_RegisterCallback(EXTI_HandleTypeDef *hexti, EXTI_CallbackIDTypeDef CallbackID, void (*pPendingCbfn)(void))`**

```c
HAL_StatusTypeDef HAL_EXTI_RegisterCallback(EXTI_HandleTypeDef *hexti, EXTI_CallbackIDTypeDef CallbackID, void (*pPendingCbfn)(void))
{
  HAL_StatusTypeDef status = HAL_OK;

  switch (CallbackID)
  {
    case  HAL_EXTI_COMMON_CB_ID:
      hexti->PendingCallback = pPendingCbfn;
      break;

    default:
      status = HAL_ERROR;
      break;
  }

  return status;
}
```

**描述:**  这个函数用于注册EXTI中断的回调函数。 当EXTI中断发生时，注册的回调函数会被调用。 它接收一个EXTI句柄 `hexti`，一个回调ID `CallbackID` 和一个指向回调函数的指针 `pPendingCbfn` 作为输入。

**主要步骤:**

*   **参数检查:**  检查 `CallbackID` 是否是有效值.
*   **注册回调函数:** 根据 `CallbackID`，将函数指针 `pPendingCbfn` 存储到 `hexti` 句柄的 `PendingCallback` 成员中。

**如何使用:**

1.  **定义回调函数:**  定义一个函数，该函数将在EXTI中断发生时被执行。
2.  **初始化EXTI句柄:** 创建一个`EXTI_HandleTypeDef`结构体变量，并设置`Line`成员，指定要注册回调函数的线路。
3.  **调用 HAL_EXTI_RegisterCallback():**  调用此函数，传入EXTI句柄，回调ID和回调函数指针。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;

// 定义回调函数
void My_EXTI_Callback(void)
{
    // 在这里处理中断事件，例如翻转一个 LED
    HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);  //假设 GPIOC Pin 13 连接到一个LED
}

void Register_EXTI_Callback(void)
{
    // 1. 初始化 EXTI 句柄
    exti_handle.Line = EXTI_LINE0;

    // 2. 注册回调函数
    if (HAL_EXTI_RegisterCallback(&exti_handle, HAL_EXTI_COMMON_CB_ID, My_EXTI_Callback) == HAL_OK)
    {
        printf("EXTI callback registered successfully.\r\n");
    }
    else
    {
        printf("Failed to register EXTI callback.\r\n");
    }
}

int main(void)
{
    // ... 初始化系统时钟，GPIO 等 ...

    Register_EXTI_Callback(); // 注册 EXTI 回调函数

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `Register_EXTI_Callback()` 函数负责注册EXTI中断的回调函数。
*   `My_EXTI_Callback()` 是用户自定义的回调函数，当中断发生时，此函数会被执行。
*   `HAL_EXTI_RegisterCallback()` 函数执行注册操作，将 `My_EXTI_Callback` 函数的指针存储到 `exti_handle` 结构体中。

**5. `HAL_EXTI_GetHandle(EXTI_HandleTypeDef *hexti, uint32_t ExtiLine)`**

```c
HAL_StatusTypeDef HAL_EXTI_GetHandle(EXTI_HandleTypeDef *hexti, uint32_t ExtiLine)
{
  /* Check the parameters */
  assert_param(IS_EXTI_LINE(ExtiLine));

  /* Check null pointer */
  if (hexti == NULL)
  {
    return HAL_ERROR;
  }
  else
  {
    /* Store line number as handle private field */
    hexti->Line = ExtiLine;

    return HAL_OK;
  }
}
```

**描述:** 这个函数用于初始化EXTI句柄，主要是设置句柄中的 `Line` 成员。 它接收一个EXTI句柄 `hexti` 的指针 和一个EXTI线路号 `ExtiLine` 作为输入。

**主要步骤:**

*   **参数检查:** 检查输入指针是否为空，并使用`assert_param`宏来验证`ExtiLine`的有效性。
*   **存储线路号:**  将`ExtiLine` 存储到 `hexti` 句柄的 `Line` 成员中。

**如何使用:**

1.  **创建EXTI句柄:** 创建一个`EXTI_HandleTypeDef`结构体变量。
2.  **调用 HAL_EXTI_GetHandle():** 调用此函数，传入EXTI句柄指针 和 线路号。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;

void Init_EXTI_Handle(void)
{
    // 1. 调用初始化函数
    if (HAL_EXTI_GetHandle(&exti_handle, EXTI_LINE0) == HAL_OK)
    {
        printf("EXTI handle initialized successfully.\r\n");
        printf("EXTI Line: %lu\r\n", exti_handle.Line);
    }
    else
    {
        printf("Failed to initialize EXTI handle.\r\n");
    }
}

int main(void)
{
    // ... 初始化系统时钟，串口等 ...

    Init_EXTI_Handle(); // 初始化 EXTI 句柄

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `Init_EXTI_Handle()` 函数负责初始化EXTI句柄。
*   `HAL_EXTI_GetHandle()` 函数将指定的线路号赋值给 `exti_handle.Line`。

**6. `HAL_EXTI_IRQHandler(EXTI_HandleTypeDef *hexti)`**

```c
void HAL_EXTI_IRQHandler(EXTI_HandleTypeDef *hexti)
{
  uint32_t regval;
  uint32_t maskline;

  /* Compute line mask */
  maskline = (1uL << (hexti->Line & EXTI_PIN_MASK));

  /* Get pending bit  */
  regval = (EXTI->PR & maskline);
  if (regval != 0x00u)
  {
    /* Clear pending bit */
    EXTI->PR = maskline;

    /* Call callback */
    if (hexti->PendingCallback != NULL)
    {
      hexti->PendingCallback();
    }
  }
}
```

**描述:** 这个函数是EXTI中断服务程序 (ISR) 中必须调用的函数。它用于检查中断是否由指定的EXTI线路触发，清除中断标志位，并调用注册的回调函数。它接收一个EXTI句柄 `hexti` 作为输入。

**主要步骤:**

*   **计算掩码:** 根据 `hexti->Line` 计算位掩码。
*   **检查中断挂起标志:**  读取EXTI的挂起寄存器 (`EXTI->PR`)，并与计算出的掩码进行与操作，判断中断是否由该线路触发。
*   **清除中断挂起标志:**  如果中断是由该线路触发的，则将EXTI的挂起寄存器 (`EXTI->PR`) 对应的位置1，清除中断标志位。  **这一步非常重要，否则中断会一直触发。**
*   **调用回调函数:**  如果注册了回调函数 (`hexti->PendingCallback != NULL`)，则调用该回调函数。

**如何使用:**

1.  **在中断服务程序中调用:**  在 `stm32f1xx_it.c` 文件中，找到对应的EXTI中断服务程序，并在其中调用 `HAL_EXTI_IRQHandler()` 函数，并将相应的EXTI句柄作为参数传递给它。

**简单演示:**

```c
// stm32f1xx_it.c 文件

// 假设已经包含了相关的头文件和 extern EXTI_HandleTypeDef exti_handle;

extern EXTI_HandleTypeDef   exti_handle; // 声明在其他地方定义的exti_handle

void EXTI0_IRQHandler(void)
{
  HAL_EXTI_IRQHandler(&exti_handle); // 调用 HAL 库的中断处理函数
}
```

**中文解释:**

*   `EXTI0_IRQHandler()` 是中断服务程序，当EXTI0线路触发中断时，此函数会被调用。
*   `HAL_EXTI_IRQHandler()` 函数检查中断是否由EXTI0线路触发，清除中断标志位，并调用注册的回调函数 (如果注册了)。

**7. `HAL_EXTI_GetPending(EXTI_HandleTypeDef *hexti, uint32_t Edge)`**

```c
uint32_t HAL_EXTI_GetPending(EXTI_HandleTypeDef *hexti, uint32_t Edge)
{
  uint32_t regval;
  uint32_t maskline;
  uint32_t linepos;

  /* Check parameters */
  assert_param(IS_EXTI_LINE(hexti->Line));
  assert_param(IS_EXTI_CONFIG_LINE(hexti->Line));
  assert_param(IS_EXTI_PENDING_EDGE(Edge));

  /* Prevent unused argument compilation warning */
  UNUSED(Edge);

  /* Compute line mask */
  linepos = (hexti->Line & EXTI_PIN_MASK);
  maskline = (1uL << linepos);

  /* return 1 if bit is set else 0 */
  regval = ((EXTI->PR & maskline) >> linepos);
  return regval;
}
```

**描述:** 这个函数用于获取指定EXTI线路的中断挂起状态。它返回1如果中断挂起，否则返回0。 它接收一个EXTI句柄 `hexti` 和一个 `Edge` 参数 (在本例中未使用) 作为输入。

**主要步骤:**

*   **参数检查:** 检查输入参数的有效性。
*   **计算掩码:** 根据 `hexti->Line` 计算位掩码。
*   **读取挂起寄存器:** 读取EXTI的挂起寄存器 (`EXTI->PR`)，并与计算出的掩码进行与操作。
*   **返回挂起状态:** 如果结果非零，则表示中断挂起，返回1；否则返回0。

**如何使用:**

1.  **初始化EXTI句柄:** 创建一个`EXTI_HandleTypeDef`结构体变量，并设置`Line`成员，指定要检查挂起状态的线路。
2.  **调用 HAL_EXTI_GetPending():**  调用此函数，传入EXTI句柄和`Edge`参数 (可以传入 `EXTI_TRIGGER_RISING_FALLING`) 。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;

void Check_EXTI_Pending(void)
{
    // 1. 初始化 EXTI 句柄
    exti_handle.Line = EXTI_LINE0;

    // 2. 检查中断挂起状态
    uint32_t pending = HAL_EXTI_GetPending(&exti_handle, EXTI_TRIGGER_RISING_FALLING);

    if (pending)
    {
        printf("EXTI line 0 interrupt is pending.\r\n");
    }
    else
    {
        printf("EXTI line 0 interrupt is not pending.\r\n");
    }
}

int main(void)
{
    // ... 初始化系统时钟，串口等 ...

    Check_EXTI_Pending(); // 检查 EXTI 中断挂起状态

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `Check_EXTI_Pending()` 函数负责检查EXTI线路的中断挂起状态。
*   `HAL_EXTI_GetPending()` 函数读取EXTI的挂起寄存器，并返回挂起状态。

**8. `HAL_EXTI_ClearPending(EXTI_HandleTypeDef *hexti, uint32_t Edge)`**

```c
void HAL_EXTI_ClearPending(EXTI_HandleTypeDef *hexti, uint32_t Edge)
{
  uint32_t maskline;

  /* Check parameters */
  assert_param(IS_EXTI_LINE(hexti->Line));
  assert_param(IS_EXTI_CONFIG_LINE(hexti->Line));
  assert_param(IS_EXTI_PENDING_EDGE(Edge));

  /* Prevent unused argument compilation warning */
  UNUSED(Edge);

  /* Compute line mask */
  maskline = (1uL << (hexti->Line & EXTI_PIN_MASK));

  /* Clear Pending bit */
  EXTI->PR =  maskline;
}
```

**描述:** 这个函数用于清除指定EXTI线路的中断挂起标志。 它接收一个EXTI句柄 `hexti` 和一个 `Edge` 参数 (在本例中未使用) 作为输入。

**主要步骤:**

*   **参数检查:** 检查输入参数的有效性。
*   **计算掩码:** 根据 `hexti->Line` 计算位掩码。
*   **清除挂起标志:** 将EXTI的挂起寄存器 (`EXTI->PR`) 与计算出的掩码进行或操作 (即向该位写1)，清除中断挂起标志。

**如何使用:**

1.  **初始化EXTI句柄:** 创建一个`EXTI_HandleTypeDef`结构体变量，并设置`Line`成员，指定要清除挂起标志的线路。
2.  **调用 HAL_EXTI_ClearPending():**  调用此函数，传入EXTI句柄和`Edge`参数 (可以传入 `EXTI_TRIGGER_RISING_FALLING`) 。

**简单演示:**

```c
// 假设已经包含了相关的头文件

EXTI_HandleTypeDef   exti_handle;

void Clear_EXTI_Pending(void)
{
    // 1. 初始化 EXTI 句柄
    exti_handle.Line = EXTI_LINE0;

    // 2. 清除中断挂起标志
    HAL_EXTI_ClearPending(&exti_handle, EXTI_TRIGGER_RISING_FALLING);

    printf("EXTI line 0 interrupt pending flag cleared.\r\n");
}

int main(void)
{
    // ... 初始化系统时钟，串口等 ...

    Clear_EXTI_Pending(); // 清除 EXTI 中断挂起标志

    while (1)
    {
        // ... 你的主循环代码 ...
    }
}
```

**中文解释:**

*   `Clear_EXTI_Pending()` 函数负责清除EXTI线路的中断挂起标志。
*   `HAL_EXTI_ClearPending()` 函数将EXTI挂起寄存器中对应的位置1，清除挂起标志。

**9. `HAL_EXTI_GenerateSWI(EXTI_HandleTypeDef *hexti)`**

```c
void HAL_EXTI_GenerateSWI(EXTI_HandleTypeDef *hexti)
{
  