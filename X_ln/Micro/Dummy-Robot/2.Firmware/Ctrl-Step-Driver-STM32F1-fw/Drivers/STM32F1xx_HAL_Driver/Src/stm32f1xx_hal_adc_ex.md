Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_adc_ex.c`

好的，我现在重新开始，对提供的 STM32F1xx HAL 库 ADC 扩展部分代码进行分析，并提供中文解释和示例。

**文件：stm32f1xx_hal_adc_ex.c**

这个文件是 STM32F1 系列单片机 HAL 库中 ADC (模数转换器) 的扩展驱动程序。它提供了一些额外的功能，用于管理 ADC 外设，特别是针对注入组 (injected group) 和多 ADC 模式。

**1. 包含头文件**

```c
#include "stm32f1xx_hal.h"
```

*   **解释:** 包含 HAL 库的主头文件。它包含了访问 HAL 库中其他模块（例如 RCC, DMA）所需的定义和声明。

**2. 定义与常量**

```c
#define ADC_PRECALIBRATION_DELAY_ADCCLOCKCYCLES       2U
#define ADC_CALIBRATION_TIMEOUT          10U
#define ADC_TEMPSENSOR_DELAY_US         10U
```

*   **解释:** 定义了一些 ADC 操作相关的常量：
    *   `ADC_PRECALIBRATION_DELAY_ADCCLOCKCYCLES`: ADC 校准前需要的延时，单位是 ADC 时钟周期。
    *   `ADC_CALIBRATION_TIMEOUT`: ADC 校准的超时时间，单位是毫秒。
    *   `ADC_TEMPSENSOR_DELAY_US`: 温度传感器稳定时间，单位是微秒。

**3. 函数分组**

该文件中的函数被组织成几个组，方便管理和查找：

*   `ADCEx_Exported_Functions_Group1`: 扩展的 I/O 操作函数，例如启动/停止注入组转换，轮询转换完成，获取注入通道结果，启动/停止多模式 DMA 传输，执行 ADC 自校准等。
*   `ADCEx_Exported_Functions_Group2`: 扩展的外设控制函数，例如配置注入组通道，配置多模式。

**4. 函数详解**

**4.1 `HAL_ADCEx_Calibration_Start(ADC_HandleTypeDef* hadc)`**

```c
HAL_StatusTypeDef HAL_ADCEx_Calibration_Start(ADC_HandleTypeDef* hadc)
{
    HAL_StatusTypeDef tmp_hal_status = HAL_OK;
    uint32_t tickstart;
    __IO uint32_t wait_loop_index = 0U;

    /* Check the parameters */
    assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

    /* Process locked */
    __HAL_LOCK(hadc);

    /* 1. Calibration prerequisite:                                             */
    /*    - ADC must be disabled for at least two ADC clock cycles in disable   */
    /*      mode before ADC enable                                              */
    /* Stop potential conversion on going, on regular and injected groups       */
    /* Disable ADC peripheral */
    tmp_hal_status = ADC_ConversionStop_Disable(hadc);

    /* Check if ADC is effectively disabled */
    if (tmp_hal_status == HAL_OK)
    {
        /* Set ADC state */
        ADC_STATE_CLR_SET(hadc->State,
                          HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                          HAL_ADC_STATE_BUSY_INTERNAL);

        /* Hardware prerequisite: delay before starting the calibration.          */
        /*  - Computation of CPU clock cycles corresponding to ADC clock cycles.  */
        /*  - Wait for the expected ADC clock cycles delay */
        wait_loop_index = ((SystemCoreClock
                            / HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_ADC))
                           * ADC_PRECALIBRATION_DELAY_ADCCLOCKCYCLES        );

        while(wait_loop_index != 0U)
        {
            wait_loop_index--;
        }

        /* 2. Enable the ADC peripheral */
        ADC_Enable(hadc);

        /* 3. Resets ADC calibration registers */
        SET_BIT(hadc->Instance->CR2, ADC_CR2_RSTCAL);

        tickstart = HAL_GetTick();

        /* Wait for calibration reset completion */
        while(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_RSTCAL))
        {
            if((HAL_GetTick() - tickstart) > ADC_CALIBRATION_TIMEOUT)
            {
                /* New check to avoid false timeout detection in case of preemption */
                if(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_RSTCAL))
                {
                    /* Update ADC state machine to error */
                    ADC_STATE_CLR_SET(hadc->State,
                                      HAL_ADC_STATE_BUSY_INTERNAL,
                                      HAL_ADC_STATE_ERROR_INTERNAL);

                    /* Process unlocked */
                    __HAL_UNLOCK(hadc);

                    return HAL_ERROR;
                }
            }
        }

        /* 4. Start ADC calibration */
        SET_BIT(hadc->Instance->CR2, ADC_CR2_CAL);

        tickstart = HAL_GetTick();

        /* Wait for calibration completion */
        while(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_CAL))
        {
            if((HAL_GetTick() - tickstart) > ADC_CALIBRATION_TIMEOUT)
            {
                /* New check to avoid false timeout detection in case of preemption */
                if(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_CAL))
                {
                    /* Update ADC state machine to error */
                    ADC_STATE_CLR_SET(hadc->State,
                                      HAL_ADC_STATE_BUSY_INTERNAL,
                                      HAL_ADC_STATE_ERROR_INTERNAL);

                    /* Process unlocked */
                    __HAL_UNLOCK(hadc);

                    return HAL_ERROR;
                }
            }
        }

        /* Set ADC state */
        ADC_STATE_CLR_SET(hadc->State,
                          HAL_ADC_STATE_BUSY_INTERNAL,
                          HAL_ADC_STATE_READY);
    }

    /* Process unlocked */
    __HAL_UNLOCK(hadc);

    /* Return function status */
    return tmp_hal_status;
}
```

*   **描述:**  执行 ADC 自动自校准。
*   **参数:** `hadc`: ADC 句柄。
*   **返回值:** `HAL_StatusTypeDef`: HAL 状态，指示校准是否成功。
*   **流程:**
    1.  **检查参数:** 确保 ADC 句柄有效。
    2.  **锁定 ADC:** 防止多任务冲突。
    3.  **校准前提:** 确保 ADC 处于禁用状态至少两个 ADC 时钟周期。停止潜在的转换并禁用 ADC。
    4.  **延时等待:**  根据时钟频率计算一个延时，确保硬件满足校准的前提条件。
    5.  **使能 ADC:** 开启 ADC 外设。
    6.  **重置校准寄存器:**  将校准相关的寄存器重置为默认值。
    7.  **启动校准:**  设置 `ADC_CR2_CAL` 位启动校准过程。
    8.  **等待校准完成:** 轮询 `ADC_CR2_CAL` 位，直到校准完成或超时。
    9.  **设置 ADC 状态:**  更新 ADC 的状态。
    10. **解锁 ADC:** 释放 ADC。
*   **使用场景:** 在初始化 ADC 之后，启动 ADC 转换之前，需要进行 ADC 校准，以提高转换精度。
*   **示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经配置好了 ADC_HandleTypeDef

// 启动校准
if (HAL_ADCEx_Calibration_Start(&hadc1) != HAL_OK) {
    // 校准失败，处理错误
    printf("ADC 校准失败!\r\n");
} else {
    printf("ADC 校准成功!\r\n");
}
```

**4.2 `HAL_ADCEx_InjectedStart(ADC_HandleTypeDef* hadc)`**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedStart(ADC_HandleTypeDef* hadc)
{
    HAL_StatusTypeDef tmp_hal_status = HAL_OK;

    /* Check the parameters */
    assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

    /* Process locked */
    __HAL_LOCK(hadc);

    /* Enable the ADC peripheral */
    tmp_hal_status = ADC_Enable(hadc);

    /* Start conversion if ADC is effectively enabled */
    if (tmp_hal_status == HAL_OK)
    {
        /* Set ADC state                                                          */
        /* - Clear state bitfield related to injected group conversion results    */
        /* - Set state bitfield related to injected operation                     */
        ADC_STATE_CLR_SET(hadc->State,
                          HAL_ADC_STATE_READY | HAL_ADC_STATE_INJ_EOC,
                          HAL_ADC_STATE_INJ_BUSY);

        /* Case of independent mode or multimode (for devices with several ADCs): */
        /* Set multimode state.                                                   */
        if (ADC_NONMULTIMODE_OR_MULTIMODEMASTER(hadc))
        {
            CLEAR_BIT(hadc->State, HAL_ADC_STATE_MULTIMODE_SLAVE);
        }
        else
        {
            SET_BIT(hadc->State, HAL_ADC_STATE_MULTIMODE_SLAVE);
        }

        /* Check if a regular conversion is ongoing */
        /* Note: On this device, there is no ADC error code fields related to     */
        /*       conversions on group injected only. In case of conversion on     */
        /*       going on group regular, no error code is reset.                  */
        if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_REG_BUSY))
        {
            /* Reset ADC all error code fields */
            ADC_CLEAR_ERRORCODE(hadc);
        }

        /* Process unlocked */
        /* Unlock before starting ADC conversions: in case of potential           */
        /* interruption, to let the process to ADC IRQ Handler.                   */
        __HAL_UNLOCK(hadc);

        /* Clear injected group conversion flag */
        /* (To ensure of no unknown state from potential previous ADC operations) */
        __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_JEOC);

        /* Enable conversion of injected group.                                   */
        /* If software start has been selected, conversion starts immediately.    */
        /* If external trigger has been selected, conversion will start at next   */
        /* trigger event.                                                         */
        /* If automatic injected conversion is enabled, conversion will start     */
        /* after next regular group conversion.                                   */
        /* Case of multimode enabled (for devices with several ADCs): if ADC is   */
        /* slave, ADC is enabled only (conversion is not started). If ADC is      */
        /* master, ADC is enabled and conversion is started.                      */
        if (HAL_IS_BIT_CLR(hadc->Instance->CR1, ADC_CR1_JAUTO))
        {
            if (ADC_IS_SOFTWARE_START_INJECTED(hadc)     &&
                ADC_NONMULTIMODE_OR_MULTIMODEMASTER(hadc)  )
            {
                /* Start ADC conversion on injected group with SW start */
                SET_BIT(hadc->Instance->CR2, (ADC_CR2_JSWSTART | ADC_CR2_JEXTTRIG));
            }
            else
            {
                /* Start ADC conversion on injected group with external trigger */
                SET_BIT(hadc->Instance->CR2, ADC_CR2_JEXTTRIG);
            }
        }
    }
    else
    {
        /* Process unlocked */
        __HAL_UNLOCK(hadc);
    }

    /* Return function status */
    return tmp_hal_status;
}
```

*   **描述:** 启动注入组的 ADC 转换 (不使用中断)。
*   **参数:** `hadc`: ADC 句柄。
*   **返回值:** `HAL_StatusTypeDef`: HAL 状态。
*   **流程:**
    1.  **检查参数:** 确保 ADC 句柄有效。
    2.  **锁定 ADC:** 防止多任务冲突。
    3.  **使能 ADC:** 开启 ADC 外设。
    4.  **设置 ADC 状态:** 更新 ADC 的状态，表示正在进行注入组转换。
    5.  **清除标志:** 清除注入组转换完成标志 `ADC_FLAG_JEOC`。
    6.  **启动转换:**
        *   如果使用软件触发，并且不是多 ADC 从模式，则设置 `ADC_CR2_JSWSTART` 和 `ADC_CR2_JEXTTRIG` 位。
        *   如果使用外部触发，则设置 `ADC_CR2_JEXTTRIG` 位。
*   **使用场景:** 当需要立即启动注入组转换，并且不想使用中断时。
*   **示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经配置好了 ADC_HandleTypeDef

// 启动注入组转换
if (HAL_ADCEx_InjectedStart(&hadc1) != HAL_OK) {
    // 启动失败，处理错误
    printf("启动注入组转换失败!\r\n");
}
```

**4.3 `HAL_ADCEx_InjectedStop(ADC_HandleTypeDef* hadc)`**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedStop(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* Process locked */
  __HAL_LOCK(hadc);

  /* Stop potential conversion and disable ADC peripheral                     */
  /* Conditioned to:                                                          */
  /* - No conversion on the other group (regular group) is intended to        */
  /*   continue (injected and regular groups stop conversion and ADC disable  */
  /*   are common)                                                            */
  /* - In case of auto-injection mode, HAL_ADC_Stop must be used.             */
  if(((hadc->State & HAL_ADC_STATE_REG_BUSY) == RESET)  &&
     HAL_IS_BIT_CLR(hadc->Instance->CR1, ADC_CR1_JAUTO)   )
  {
    /* Stop potential conversion on going, on regular and injected groups */
    /* Disable ADC peripheral */
    tmp_hal_status = ADC_ConversionStop_Disable(hadc);

    /* Check if ADC is effectively disabled */
    if (tmp_hal_status == HAL_OK)
    {
      /* Set ADC state */
      ADC_STATE_CLR_SET(hadc->State,
                        HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                        HAL_ADC_STATE_READY);
    }
  }
  else
  {
    /* Update ADC state machine to error */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_CONFIG);

    tmp_hal_status = HAL_ERROR;
  }

  /* Process unlocked */
  __HAL_UNLOCK(hadc);

  /* Return function status */
  return tmp_hal_status;
}
```

*   **描述:**  停止注入组的 ADC 转换。
*   **参数:** `hadc`: ADC 句柄。
*   **返回值:** `HAL_StatusTypeDef`: HAL 状态。
*   **前提条件:**
    *   没有正在进行常规组的转换。
    *   未启用自动注入模式。如果满足任一条件，应使用 `HAL_ADC_Stop()`。
*   **流程:**
    1.  **检查参数:** 确保 ADC 句柄有效。
    2.  **锁定 ADC:** 防止多任务冲突。
    3.  **停止转换和禁用 ADC:** 调用 `ADC_ConversionStop_Disable()` 来停止任何正在进行的转换并禁用 ADC。
    4.  **设置 ADC 状态:** 更新 ADC 的状态，表示已停止注入组转换。
*   **使用场景:**  当不再需要进行注入组转换时，停止 ADC 以节省功耗。
*   **示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经配置好了 ADC_HandleTypeDef

// 停止注入组转换
if (HAL_ADCEx_InjectedStop(&hadc1) != HAL_OK) {
    // 停止失败，处理错误
    printf("停止注入组转换失败!\r\n");
}
```

**4.4 `HAL_ADCEx_InjectedPollForConversion(ADC_HandleTypeDef* hadc, uint32_t Timeout)`**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedPollForConversion(ADC_HandleTypeDef* hadc, uint32_t Timeout)
{
  uint32_t tickstart;

  /* Variables for polling in case of scan mode enabled and polling for each  */
  /* conversion.                                                              */
  __IO uint32_t Conversion_Timeout_CPU_cycles = 0U;
  uint32_t Conversion_Timeout_CPU_cycles_max = 0U;

  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* Get timeout */
  tickstart = HAL_GetTick();

  /* Polling for end of conversion: differentiation if single/sequence        */
  /* conversion.                                                              */
  /* For injected group, flag JEOC is set only at the end of the sequence,    */
  /* not for each conversion within the sequence.                             */
  /*  - If single conversion for injected group (scan mode disabled or        */
  /*    InjectedNbrOfConversion ==1), flag JEOC is used to determine the      */
  /*    conversion completion.                                                */
  /*  - If sequence conversion for injected group (scan mode enabled and      */
  /*    InjectedNbrOfConversion >=2), flag JEOC is set only at the end of the */
  /*    sequence.                                                             */
  /*    To poll for each conversion, the maximum conversion time is computed  */
  /*    from ADC conversion time (selected sampling time + conversion time of */
  /*    12.5 ADC clock cycles) and APB2/ADC clock prescalers (depending on    */
  /*    settings, conversion time range can be from 28 to 32256 CPU cycles).  */
  /*    As flag JEOC is not set after each conversion, no timeout status can  */
  /*    be set.                                                               */
  if ((hadc->Instance->JSQR & ADC_JSQR_JL) == RESET)
  {
    /* Wait until End of Conversion flag is raised */
    while(HAL_IS_BIT_CLR(hadc->Instance->SR, ADC_FLAG_JEOC))
    {
      /* Check if timeout is disabled (set to infinite wait) */
      if(Timeout != HAL_MAX_DELAY)
      {
        if((Timeout == 0U) || ((HAL_GetTick() - tickstart ) > Timeout))
        {
          /* New check to avoid false timeout detection in case of preemption */
          if(HAL_IS_BIT_CLR(hadc->Instance->SR, ADC_FLAG_JEOC))
          {
            /* Update ADC state machine to timeout */
            SET_BIT(hadc->State, HAL_ADC_STATE_TIMEOUT);

            /* Process unlocked */
            __HAL_UNLOCK(hadc);

            return HAL_TIMEOUT;
          }
        }
      }
    }
  }
  else
  {
    /* Replace polling by wait for maximum conversion time */
    /*  - Computation of CPU clock cycles corresponding to ADC clock cycles   */
    /*    and ADC maximum conversion cycles on all channels.                  */
    /*  - Wait for the expected ADC clock cycles delay                        */
    Conversion_Timeout_CPU_cycles_max = ((SystemCoreClock
                                          / HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_ADC))
                                         * ADC_CONVCYCLES_MAX_RANGE(hadc)                 );

    while(Conversion_Timeout_CPU_cycles < Conversion_Timeout_CPU_cycles_max)
    {
      /* Check if timeout is disabled (set to infinite wait) */
      if(Timeout != HAL_MAX_DELAY)
      {
        if((Timeout == 0)||((HAL_GetTick() - tickstart ) > Timeout))
        {
          /* New check to avoid false timeout detection in case of preemption */
          if(Conversion_Timeout_CPU_cycles < Conversion_Timeout_CPU_cycles_max)
          {
            /* Update ADC state machine to timeout */
            SET_BIT(hadc->State, HAL_ADC_STATE_TIMEOUT);

            /* Process unlocked */
            __HAL_UNLOCK(hadc);

            return HAL_TIMEOUT;
          }
        }
      }
      Conversion_Timeout_CPU_cycles ++;
    }
  }

  /* Clear injected group conversion flag */
  /* Note: On STM32F1 ADC, clear regular conversion flag raised               */
  /* simultaneously.                                                          */
  __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_JSTRT | ADC_FLAG_JEOC | ADC_FLAG_EOC);

  /* Update ADC state machine */
  SET_BIT(hadc->State, HAL_ADC_STATE_INJ_EOC);

  /* Determine whether any further conversion upcoming on group injected      */
  /* by external trigger or by automatic injected conversion                  */
  /* from group regular.                                                      */
  if(ADC_IS_SOFTWARE_START_INJECTED(hadc)                     ||
     (HAL_IS_BIT_CLR(hadc->Instance->CR1, ADC_CR1_JAUTO) &&
     (ADC_IS_SOFTWARE_START_REGULAR(hadc)        &&
      (hadc->Init.ContinuousConvMode == DISABLE)   )        )   )
  {
    /* Set ADC state */
    CLEAR_BIT(hadc->State, HAL_ADC_STATE_INJ_BUSY);

    if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_REG_BUSY))
    {
      SET_BIT(hadc->State, HAL_ADC_STATE_READY);
    }
  }

  /* Return ADC state */
  return HAL_OK;
}
```

*   **描述:** 轮询等待注入组转换完成。
*   **参数:**
    *   `hadc`: ADC 句柄。
    *   `Timeout`: 超时时间，单位是毫秒。
*   **返回值:** `HAL_StatusTypeDef`: HAL 状态。
*   **流程:**
    1.  **检查参数:** 确保 ADC 句柄有效。
    2.  **获取超时起始时间:** 使用 `HAL_GetTick()` 获取当前时间。
    3.  **轮询等待 `ADC_FLAG_JEOC` 标志:** 根据是否启用了扫描模式 (Scan Mode) 和注入转换的数量进行不同的处理。
        *   **单次注入转换或禁用扫描模式:** 等待 `ADC_FLAG_JEOC` 标志被置位。
        *   **多次注入转换且启用扫描模式:** 没有办法在每次转换后轮询等待标志。这段代码使用一个基于最大转换时间的延时来代替轮询。
    4.  **清除标志:** 清除注入组转换相关的标志位 (`ADC_FLAG_JSTRT`, `ADC_FLAG_JEOC`, `ADC_FLAG_EOC`)。
    5.  **更新 ADC 状态:** 设置 `HAL_ADC_STATE_INJ_EOC` 标志。
*   **使用场景:**  在启动注入组转换后，使用轮询方式等待转换完成。
*   **示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经配置好了 ADC_HandleTypeDef

// 启动注入组转换 (假设使用软件触发)
HAL_ADCEx_InjectedStart(&hadc1);

// 轮询等待转换完成
if (HAL_ADCEx_InjectedPollForConversion(&hadc1, 100) != HAL_OK) {
    // 轮询超时或出错
    printf("注入组转换超时或出错!\r\n");
} else {
    // 转换完成
    printf("注入组转换完成!\r\n");
    // 获取转换结果...
}
```

**4.5 `HAL_ADCEx_InjectedStart_IT(ADC_HandleTypeDef* hadc)`**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedStart_IT(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* Process locked */
  __HAL_LOCK(hadc);

  /* Enable the ADC peripheral */
  tmp_hal_status = ADC_Enable(hadc);

  /* Start conversion if ADC is effectively enabled */
  if (tmp_hal_status == HAL_OK)
  {
    /* Set ADC state                                                          */
    /* - Clear state bitfield related to injected group conversion results    */
    /* - Set state bitfield related to injected operation                     */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_READY | HAL_ADC_STATE_INJ_EOC,
                      HAL_ADC_STATE_INJ_BUSY);

    /* Case of independent mode or multimode (for devices with several ADCs): */
    /* Set multimode state.                                                   */
    if (ADC_NONMULTIMODE_OR_MULTIMODEMASTER(hadc))
    {
      CLEAR_BIT(hadc->State, HAL_ADC_STATE_MULTIMODE_SLAVE);
    }
    else
    {
      SET_BIT(hadc->State, HAL_ADC_STATE_MULTIMODE_SLAVE);
    }

    /* Check if a regular conversion is ongoing */
    /* Note: On this device, there is no ADC error code fields related to     */
    /*       conversions on group injected only. In case of conversion on     */
    /*       going on group regular, no error code is reset.                  */
    if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_REG_BUSY))
    {
      /* Reset ADC all error code fields */
      ADC_CLEAR_ERRORCODE(hadc);
    }

    /* Process unlocked */
    /* Unlock before starting ADC conversions: in case of potential           */
    /* interruption, to let the process to ADC IRQ Handler.                   */
    __HAL_UNLOCK(hadc);

    /* Clear injected group conversion flag */
    /* (To ensure of no unknown state from potential previous ADC operations) */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_JEOC);

    /* Enable end of conversion interrupt for injected channels */
    __HAL_ADC_ENABLE_IT(hadc, ADC_IT_JEOC);

    /* Start conversion of injected group if software start has been selected */
    /* and if automatic injected conversion is disabled.                      */
    /* If external trigger has been selected, conversion will start at next   */
    /* trigger event.                                                         */
    /* If automatic injected conversion is enabled, conversion will start     */
    /* after next regular group conversion.                                   */
    if (HAL_IS_BIT_CLR(hadc->Instance->CR1, ADC_CR1_JAUTO))
    {
      if (ADC_IS_SOFTWARE_START_INJECTED(hadc)     &&
          ADC_NONMULTIMODE_OR_MULTIMODEMASTER(hadc)  )
      {
        /* Start ADC conversion on injected group with SW start */
        SET_BIT(hadc->Instance->CR2, (ADC_CR2_JSWSTART | ADC_CR2_JEXTTRIG));
      }
      else
      {
        /* Start ADC conversion on injected group with external trigger */
        SET_BIT(hadc->Instance->CR2, ADC_CR2_JEXTTRIG);
      }
    }
  }
  else
  {
    /* Process unlocked */
    __HAL_UNLOCK(hadc);
  }

  /* Return function status */
  return tmp_hal_status;
}
```

*   **描述:** 启动注入组的 ADC 转换 (使用中断)。
*   **参数:** `hadc`: ADC 句柄。
*   **返回值:** `HAL_StatusTypeDef`: HAL 状态。
*   **流程:**
    1.  **检查参数:** 确保 ADC 句柄有效。
    2.  **锁定 ADC:** 防止多任务冲突。
    3.  **使能 ADC:** 开启 ADC 外设。
    4.  **设置 ADC 状态:** 更新 ADC 的状态，表示正在进行注入组转换。
    5.  **清除标志:** 清除注入组转换完成标志 `ADC_FLAG_JEOC`。
    6.  **使能中断:** 使能注入组转换完成中断 `ADC_IT_JEOC`。
    7.  **启动转换:**
        *   如果使用软件触发，并且不是多 ADC 从模式，则设置 `ADC_CR2_JSWSTART` 和 `ADC_CR2_JEXTTRIG` 位。
        *   如果使用外部触发，则设置 `ADC_CR2_JEXTTRIG` 位。
*   **使用场景:** 当需要启动注入组转换，并且希望在转换完成后通过中断来处理结果时。
*   **示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经配置好了 ADC_HandleTypeDef

// 启动注入组转换 (使用中断)
if (HAL_ADCEx_InjectedStart_IT(&hadc1) != HAL_OK) {
    // 启动失败，处理错误
    printf("启动注入组转换(中断模式)失败!\r\n");
}

// 中断处理函数 (需要在 stm32f1xx_it.c 中定义)
void ADC1_2_IRQHandler(void) { // ADC1 和 ADC2 共享同一个中断向量
  HAL_ADC_IRQHandler(&hadc1); // 调用 HAL 库的中断处理函数
}

// ADC 中断回调函数 (需要在用户代码中定义)
void HAL_ADCEx_InjectedConvCpltCallback(ADC_HandleTypeDef* hadc) {
    // 注入组转换完成，获取转换结果
    uint32_t result = HAL_ADCEx_InjectedGetValue(hadc, ADC_INJECTED_RANK_1);
    printf("注入组转换结果: %lu\r\n", result);
}
```

**4.6 `HAL_ADCEx_InjectedStop_IT(ADC_HandleTypeDef* hadc)`**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedStop_IT(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));

  /* Process locked */
  __HAL_LOCK(hadc);

  /* Stop potential conversion and disable ADC peripheral                     */
  /* Conditioned to:                                                          */
  /* - No conversion on the other group (regular group) is intended to        */
  /*   continue (injected and regular groups stop conversion and ADC disable  */
  /*   are common)                                                            */
  /* - In case of auto-injection mode, HAL_ADC_Stop must be used.             */
  if(((hadc->State & HAL_ADC_STATE_REG_BUSY) == RESET)  &&
     HAL_IS_BIT_CLR(hadc->Instance->CR1, ADC_CR1_JAUTO)   )
  {
    /* Stop potential conversion on going, on regular and injected groups */
    /* Disable ADC peripheral */
    tmp_hal_status = ADC_ConversionStop_Disable(hadc);

    /* Check if ADC is effectively disabled */
    if (tmp_hal_status == HAL_OK)
    {
      /* Disable ADC end of conversion interrupt for injected channels */
      __HAL_ADC_DISABLE_IT(hadc, ADC_IT_JEOC);

      /* Set ADC state */
      ADC_STATE_CLR_SET(hadc->State,
                        HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                        HAL_ADC_STATE_READY);
    }
  }
  else
  {
    /* Update ADC state machine to error */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_CONFIG);

    tmp_hal_status = HAL_ERROR;
  }

  /* Process unlocked */
  __HAL_UNLOCK(hadc);

  /* Return function status */
  return tmp_hal_status;
}
```

*   **描述:** 停止注入组的 ADC 转换 (并禁用中断)。
*   **参数:** `hadc`: ADC 句柄。
*   **返回值:** `HAL_StatusTypeDef`: HAL 状态。
*   **前提条件:**
    *   没有正在进行常规组的转换。
    *   未启用自动注入模式。如果满足任一条件，应使用 `HAL_ADC_Stop()`。
*   **流程:**
    1.  **检查参数:** 确保 ADC 句柄有效。
    2.  **锁定 ADC:** 防止多任务冲突。
    3.  **停止转换和禁用 ADC:** 调用 `ADC_ConversionStop_Disable()` 来停止任何正在进行的转换并禁用 ADC。
    4.  **禁用中断:** 禁用注入组转换完成中断 `ADC_IT_JEOC`。
    5.  **设置 ADC 状态:** 更新 ADC 的状态，表示已停止注入组转换。
*   **使用场景:** 当不再需要进行注入组转换，并且不再需要中断通知时，停止 ADC 并禁用中断以节省功耗。
*   **示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经配置好了 ADC_HandleTypeDef

// 停止注入组转换 (并禁用中断)
if (HAL_ADCEx_InjectedStop_IT(&hadc1) != HAL_OK) {
    // 停止失败，处理错误
    printf("停止注入组转换(中断模式)失败!\r\n");
}
```

**4.7 `HAL_ADCEx_MultiModeStart_DMA(ADC_HandleTypeDef* hadc, uint32_t* pData, uint32_t Length)` (仅适用于部分 F1 系列)**

```c
#if defined (STM32F103x6) || defined (STM32F103xB) || defined (STM32F105xC) || defined (STM32F107xC) || defined (STM32F103xE) || defined (STM32F103xG)
/**
  * @brief  Enables ADC, starts conversion of regular group and transfers result
  *         through DMA.
  *         Multimode must have been previously configured using
  *         HAL_ADCEx_MultiModeConfigChannel() function.
  *         Interruptions enabled