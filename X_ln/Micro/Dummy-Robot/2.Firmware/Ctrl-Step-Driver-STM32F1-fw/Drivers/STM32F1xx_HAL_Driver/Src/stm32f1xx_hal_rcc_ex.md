Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_rcc_ex.c`

好的，我们重新开始。 接下来，我对提供的 C 代码 `stm32f1xx_hal_rcc_ex.c` 的关键部分进行解释，并提供代码片段及其中文描述，以及简短的使用说明和示例。

**1. `HAL_RCCEx_PeriphCLKConfig` 函数**

```c
HAL_StatusTypeDef HAL_RCCEx_PeriphCLKConfig(RCC_PeriphCLKInitTypeDef  *PeriphClkInit)
{
  uint32_t tickstart = 0U, temp_reg = 0U;
#if defined(STM32F105xC) || defined(STM32F107xC)
  uint32_t  pllactive = 0U;
#endif /* STM32F105xC || STM32F107xC */

  /* Check the parameters */
  assert_param(IS_RCC_PERIPHCLOCK(PeriphClkInit->PeriphClockSelection));

  /*------------------------------- RTC/LCD Configuration ------------------------*/
  if ((((PeriphClkInit->PeriphClockSelection) & RCC_PERIPHCLK_RTC) == RCC_PERIPHCLK_RTC))
  {
    // ... (RTC Clock Configuration Logic) ...
  }

  /*------------------------------ ADC clock Configuration ------------------*/
  if (((PeriphClkInit->PeriphClockSelection) & RCC_PERIPHCLK_ADC) == RCC_PERIPHCLK_ADC)
  {
    /* Check the parameters */
    assert_param(IS_RCC_ADCPLLCLK_DIV(PeriphClkInit->AdcClockSelection));

    /* Configure the ADC clock source */
    __HAL_RCC_ADC_CONFIG(PeriphClkInit->AdcClockSelection);
  }

#if defined(STM32F105xC) || defined(STM32F107xC)
  /*------------------------------ I2S2 Configuration ------------------------*/
  if (((PeriphClkInit->PeriphClockSelection) & RCC_PERIPHCLK_I2S2) == RCC_PERIPHCLK_I2S2)
  {
    // ... (I2S2 Clock Configuration Logic) ...
  }

  /*------------------------------ I2S3 Configuration ------------------------*/
  if (((PeriphClkInit->PeriphClockSelection) & RCC_PERIPHCLK_I2S3) == RCC_PERIPHCLK_I2S3)
  {
    // ... (I2S3 Clock Configuration Logic) ...
  }

  /*------------------------------ PLL I2S Configuration ----------------------*/
  // ... (PLLI2S Clock Configuration Logic) ...
#endif /* STM32F105xC || STM32F107xC */

#if defined(STM32F102x6) || defined(STM32F102xB) || defined(STM32F103x6)\
 || defined(STM32F103xB) || defined(STM32F103xE) || defined(STM32F103xG)\
 || defined(STM32F105xC) || defined(STM32F107xC)
  /*------------------------------ USB clock Configuration ------------------*/
  if (((PeriphClkInit->PeriphClockSelection) & RCC_PERIPHCLK_USB) == RCC_PERIPHCLK_USB)
  {
    /* Check the parameters */
    assert_param(IS_RCC_USBPLLCLK_DIV(PeriphClkInit->UsbClockSelection));

    /* Configure the USB clock source */
    __HAL_RCC_USB_CONFIG(PeriphClkInit->UsbClockSelection);
  }
#endif /* STM32F102x6 || STM32F102xB || STM32F103x6 || STM32F103xB || STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */

  return HAL_OK;
}
```

**描述 (描述):**

*   这个函数 `HAL_RCCEx_PeriphCLKConfig` 用于配置扩展外设的时钟，例如 RTC (实时时钟), ADC (模数转换器), I2S (内部集成电路声音), 和 USB。
*   它接收一个 `RCC_PeriphCLKInitTypeDef` 结构体指针作为参数，该结构体包含了需要配置的时钟源和分频系数等信息。
*   函数内部通过一系列条件判断，根据 `PeriphClkInit->PeriphClockSelection` 成员来确定需要配置哪个外设的时钟。
*   对于RTC时钟的配置，需要特别注意备份域的复位问题，因为修改RTC时钟源会导致备份域复位，从而丢失备份寄存器中的数据。
*   函数使用了一些宏，例如 `__HAL_RCC_ADC_CONFIG` 和 `__HAL_RCC_USB_CONFIG`，这些宏实际上是对底层寄存器进行操作，实现时钟的配置。
*   `assert_param` 宏用于检查输入参数的合法性，确保配置的正确性。

**使用说明 (使用说明):**

1.  **填充 `RCC_PeriphCLKInitTypeDef` 结构体:**  首先，你需要创建一个 `RCC_PeriphCLKInitTypeDef` 类型的结构体变量，并根据你的需求填充相应的成员，例如 `RTCClockSelection`、`AdcClockSelection` 等。
2.  **调用 `HAL_RCCEx_PeriphCLKConfig` 函数:**  然后，将该结构体变量的指针作为参数传递给 `HAL_RCCEx_PeriphCLKConfig` 函数，即可完成外设时钟的配置。
3.  **注意备份域:** 配置RTC时钟时，要考虑到备份域复位的影响，如果需要保留备份寄存器中的数据，应该在配置前先读取出来，配置后再写回去。

**示例 (示例):**

```c
RCC_PeriphCLKInitTypeDef  PeriphClkInit;

/* 配置 RTC 时钟源为 LSE */
PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_RTC;
PeriphClkInit.RTCClockSelection = RCC_RTCCLKSOURCE_LSE;

HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit);

/* 配置 ADC 时钟源为 PCLK2 分频 */
PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;

HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit);
```

这段代码首先配置了RTC的时钟源为LSE (Low Speed External)，然后配置了ADC的时钟源为PCLK2 (Peripheral Clock 2) 的 6 分频。

**2. `HAL_RCCEx_GetPeriphCLKConfig` 函数**

```c
void HAL_RCCEx_GetPeriphCLKConfig(RCC_PeriphCLKInitTypeDef  *PeriphClkInit)
{
  uint32_t srcclk = 0U;

  /* Set all possible values for the extended clock type parameter------------*/
  PeriphClkInit->PeriphClockSelection = RCC_PERIPHCLK_RTC;

  /* Get the RTC configuration -----------------------------------------------*/
  srcclk = __HAL_RCC_GET_RTC_SOURCE();
  /* Source clock is LSE or LSI*/
  PeriphClkInit->RTCClockSelection = srcclk;

  /* Get the ADC clock configuration -----------------------------------------*/
  PeriphClkInit->PeriphClockSelection |= RCC_PERIPHCLK_ADC;
  PeriphClkInit->AdcClockSelection = __HAL_RCC_GET_ADC_SOURCE();

#if defined(STM32F105xC) || defined(STM32F107xC)
  /* Get the I2S2 clock configuration -----------------------------------------*/
  PeriphClkInit->PeriphClockSelection |= RCC_PERIPHCLK_I2S2;
  PeriphClkInit->I2s2ClockSelection = __HAL_RCC_GET_I2S2_SOURCE();

  /* Get the I2S3 clock configuration -----------------------------------------*/
  PeriphClkInit->PeriphClockSelection |= RCC_PERIPHCLK_I2S3;
  PeriphClkInit->I2s3ClockSelection = __HAL_RCC_GET_I2S3_SOURCE();

#endif /* STM32F105xC || STM32F107xC */

#if defined(STM32F103xE) || defined(STM32F103xG)
  /* Get the I2S2 clock configuration -----------------------------------------*/
  PeriphClkInit->PeriphClockSelection |= RCC_PERIPHCLK_I2S2;
  PeriphClkInit->I2s2ClockSelection = RCC_I2S2CLKSOURCE_SYSCLK;

  /* Get the I2S3 clock configuration -----------------------------------------*/
  PeriphClkInit->PeriphClockSelection |= RCC_PERIPHCLK_I2S3;
  PeriphClkInit->I2s3ClockSelection = RCC_I2S3CLKSOURCE_SYSCLK;

#endif /* STM32F103xE || STM32F103xG */

#if defined(STM32F102x6) || defined(STM32F102xB) || defined(STM32F103x6)\
 || defined(STM32F103xB) || defined(STM32F103xE) || defined(STM32F103xG)\
 || defined(STM32F105xC) || defined(STM32F107xC)
  /* Get the USB clock configuration -----------------------------------------*/
  PeriphClkInit->PeriphClockSelection |= RCC_PERIPHCLK_USB;
  PeriphClkInit->UsbClockSelection = __HAL_RCC_GET_USB_SOURCE();
#endif /* STM32F102x6 || STM32F102xB || STM32F103x6 || STM32F103xB || STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */
}
```

**描述 (描述):**

*   这个函数 `HAL_RCCEx_GetPeriphCLKConfig` 用于获取当前外设时钟的配置信息。
*   它接收一个 `RCC_PeriphCLKInitTypeDef` 结构体指针作为参数，并将读取到的配置信息填充到该结构体中。
*   函数通过调用 `__HAL_RCC_GET_RTC_SOURCE`、`__HAL_RCC_GET_ADC_SOURCE` 等宏来读取底层寄存器的值，从而获取时钟源和分频系数等信息。
*   函数根据不同的芯片型号，获取不同外设的时钟配置信息。

**使用说明 (使用说明):**

1.  **创建 `RCC_PeriphCLKInitTypeDef` 结构体:**  首先，你需要创建一个 `RCC_PeriphCLKInitTypeDef` 类型的结构体变量。
2.  **调用 `HAL_RCCEx_GetPeriphCLKConfig` 函数:**  然后，将该结构体变量的指针作为参数传递给 `HAL_RCCEx_GetPeriphCLKConfig` 函数。
3.  **读取配置信息:**  函数执行完成后，你可以通过访问该结构体变量的成员来获取外设时钟的配置信息。

**示例 (示例):**

```c
RCC_PeriphCLKInitTypeDef  PeriphClkInit;

HAL_RCCEx_GetPeriphCLKConfig(&PeriphClkInit);

/* 打印 RTC 时钟源 */
if (PeriphClkInit.RTCClockSelection == RCC_RTCCLKSOURCE_LSE) {
  printf("RTC Clock Source: LSE\n");
} else if (PeriphClkInit.RTCClockSelection == RCC_RTCCLKSOURCE_LSI) {
  printf("RTC Clock Source: LSI\n");
} else {
  printf("RTC Clock Source: HSE/128\n");
}

/* 打印 ADC 时钟分频系数 */
if (PeriphClkInit.AdcClockSelection == RCC_ADCPCLK2_DIV2) {
  printf("ADC Clock Prescaler: DIV2\n");
} else {
  printf("ADC Clock Prescaler: DIV6\n");
}
```

这段代码首先获取了当前的外设时钟配置信息，然后打印了RTC的时钟源和ADC的时钟分频系数。

**3. `HAL_RCCEx_GetPeriphCLKFreq` 函数**

```c
uint32_t HAL_RCCEx_GetPeriphCLKFreq(uint32_t PeriphClk)
{
  // ... (Constant Tables and Variable Declarations) ...

  switch (PeriphClk)
  {
#if defined(STM32F102x6) || defined(STM32F102xB) || defined(STM32F103x6)\
 || defined(STM32F103xB) || defined(STM32F103xE) || defined(STM32F103xG)\
 || defined(STM32F105xC) || defined(STM32F107xC)
    case RCC_PERIPHCLK_USB:
    {
      // ... (USB Clock Frequency Calculation Logic) ...
      break;
    }
#endif /* STM32F102x6 || STM32F102xB || STM32F103x6 || STM32F103xB || STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */
#if defined(STM32F103xE) || defined(STM32F103xG) || defined(STM32F105xC) || defined(STM32F107xC)
    case RCC_PERIPHCLK_I2S2:
    {
      // ... (I2S2 Clock Frequency Calculation Logic) ...
      break;
    }
    case RCC_PERIPHCLK_I2S3:
    {
      // ... (I2S3 Clock Frequency Calculation Logic) ...
      break;
    }
#endif /* STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */
    case RCC_PERIPHCLK_RTC:
    {
      // ... (RTC Clock Frequency Calculation Logic) ...
      break;
    }
    case RCC_PERIPHCLK_ADC:
    {
      // ... (ADC Clock Frequency Calculation Logic) ...
      break;
    }
    default:
    {
      break;
    }
  }
  return (frequency);
}
```

**描述 (描述):**

*   这个函数 `HAL_RCCEx_GetPeriphCLKFreq` 用于获取指定外设的时钟频率。
*   它接收一个 `PeriphClk` 参数，该参数指定了需要获取频率的外设，例如 `RCC_PERIPHCLK_RTC`、`RCC_PERIPHCLK_ADC` 等。
*   函数内部使用一个 `switch` 语句，根据 `PeriphClk` 的值来选择相应的频率计算逻辑。
*   对于不同的外设，频率的计算方式不同，需要根据时钟源和分频系数等信息进行计算。
*   函数返回计算出的频率值，单位为 Hz。 如果无法确定频率，则返回 0。

**使用说明 (使用说明):**

1.  **指定外设:**  首先，你需要确定你需要获取哪个外设的时钟频率，并使用相应的宏定义，例如 `RCC_PERIPHCLK_RTC`。
2.  **调用 `HAL_RCCEx_GetPeriphCLKFreq` 函数:**  然后，将该宏定义作为参数传递给 `HAL_RCCEx_GetPeriphCLKFreq` 函数。
3.  **获取频率值:**  函数执行完成后，你可以获取返回值，该返回值即为指定外设的时钟频率。

**示例 (示例):**

```c
uint32_t rtc_frequency = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_RTC);
uint32_t adc_frequency = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_ADC);

printf("RTC Clock Frequency: %lu Hz\n", rtc_frequency);
printf("ADC Clock Frequency: %lu Hz\n", adc_frequency);
```

这段代码首先获取了RTC和ADC的时钟频率，然后打印了这些频率值。

**总结 (总结):**

这三个函数 `HAL_RCCEx_PeriphCLKConfig`、`HAL_RCCEx_GetPeriphCLKConfig` 和 `HAL_RCCEx_GetPeriphCLKFreq` 提供了对 STM32F1 系列单片机扩展外设时钟配置和频率获取的功能。 通过使用这些函数，可以方便地配置和管理外设的时钟，从而保证系统的正常运行。

希望这些解释对您有所帮助！
