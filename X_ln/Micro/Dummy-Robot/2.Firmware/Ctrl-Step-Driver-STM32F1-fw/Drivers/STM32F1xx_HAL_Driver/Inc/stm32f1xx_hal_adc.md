Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_adc.h`

好的，这次我将对代码的关键部分进行解释，生成小代码片段，并附上简短的中文描述。描述中包含代码用途的简要说明和一个简单的演示。

**1. ADC_InitTypeDef 结构体 (ADC_InitTypeDef Structure)**

```c
typedef struct
{
  uint32_t DataAlign;                        /*!< Specifies ADC data alignment to right (MSB on register bit 11 and LSB on register bit 0) (default setting)
                                                  or to left (if regular group: MSB on register bit 15 and LSB on register bit 4, if injected group (MSB kept as signed value due to potential negative value after offset application): MSB on register bit 14 and LSB on register bit 3).
                                                  This parameter can be a value of @ref ADC_Data_align */
  uint32_t ScanConvMode;                     /*!< Configures the sequencer of regular and injected groups.
                                                  This parameter can be associated to parameter 'DiscontinuousConvMode' to have main sequence subdivided in successive parts.
                                                  If disabled: Conversion is performed in single mode (one channel converted, the one defined in rank 1).
                                                               Parameters 'NbrOfConversion' and 'InjectedNbrOfConversion' are discarded (equivalent to set to 1).
                                                  If enabled:  Conversions are performed in sequence mode (multiple ranks defined by 'NbrOfConversion'/'InjectedNbrOfConversion' and each channel rank).
                                                               Scan direction is upward: from rank1 to rank 'n'.
                                                  This parameter can be a value of @ref ADC_Scan_mode
                                                  Note: For regular group, this parameter should be enabled in conversion either by polling (HAL_ADC_Start with Discontinuous mode and NbrOfDiscConversion=1)
                                                        or by DMA (HAL_ADC_Start_DMA), but not by interruption (HAL_ADC_Start_IT): in scan mode, interruption is triggered only on the
                                                        the last conversion of the sequence. All previous conversions would be overwritten by the last one.
                                                        Injected group used with scan mode has not this constraint: each rank has its own result register, no data is overwritten. */
  FunctionalState ContinuousConvMode;         /*!< Specifies whether the conversion is performed in single mode (one conversion) or continuous mode for regular group,
                                                  after the selected trigger occurred (software start or external trigger).
                                                  This parameter can be set to ENABLE or DISABLE. */
  uint32_t NbrOfConversion;                  /*!< Specifies the number of ranks that will be converted within the regular group sequencer.
                                                  To use regular group sequencer and convert several ranks, parameter 'ScanConvMode' must be enabled.
                                                  This parameter must be a number between Min_Data = 1 and Max_Data = 16. */
  FunctionalState  DiscontinuousConvMode;    /*!< Specifies whether the conversions sequence of regular group is performed in Complete-sequence/Discontinuous-sequence (main sequence subdivided in successive parts).
                                                  Discontinuous mode is used only if sequencer is enabled (parameter 'ScanConvMode'). If sequencer is disabled, this parameter is discarded.
                                                  Discontinuous mode can be enabled only if continuous mode is disabled. If continuous mode is enabled, this parameter setting is discarded.
                                                  This parameter can be set to ENABLE or DISABLE. */
  uint32_t NbrOfDiscConversion;              /*!< Specifies the number of discontinuous conversions in which the  main sequence of regular group (parameter NbrOfConversion) will be subdivided.
                                                  If parameter 'DiscontinuousConvMode' is disabled, this parameter is discarded.
                                                  This parameter must be a number between Min_Data = 1 and Max_Data = 8. */
  uint32_t ExternalTrigConv;                 /*!< Selects the external event used to trigger the conversion start of regular group.
                                                  If set to ADC_SOFTWARE_START, external triggers are disabled.
                                                  If set to external trigger source, triggering is on event rising edge.
                                                  This parameter can be a value of @ref ADC_External_trigger_source_Regular */
}ADC_InitTypeDef;
```

   **描述:**  `ADC_InitTypeDef` 结构体用于配置 ADC (模数转换器) 的参数。 它包含了数据对齐方式、扫描模式、连续转换模式、转换通道数、不连续转换模式、不连续转换的数量以及外部触发源等设置。这些参数会影响 ADC 的转换过程。

   **如何使用:**  在使用 HAL 库初始化 ADC 时，需要填充这个结构体。 例如，可以设置 `DataAlign` 为 `ADC_DATAALIGN_RIGHT` 来选择右对齐的数据格式。设置 `ScanConvMode` 为 `ADC_SCAN_ENABLE` 可以开启扫描模式。

   **演示:**
   ```c
   ADC_HandleTypeDef hadc1;
   ADC_InitTypeDef adc_init;

   adc_init.DataAlign = ADC_DATAALIGN_RIGHT;
   adc_init.ScanConvMode = ADC_SCAN_DISABLE;
   adc_init.ContinuousConvMode = DISABLE;
   adc_init.NbrOfConversion = 1;
   adc_init.DiscontinuousConvMode = DISABLE;
   adc_init.NbrOfDiscConversion = 1;
   adc_init.ExternalTrigConv = ADC_SOFTWARE_START;

   hadc1.Instance = ADC1; //选择ADC1外设
   hadc1.Init = adc_init;

   HAL_ADC_Init(&hadc1); // 使用 HAL 库初始化 ADC
   ```
   这段代码创建了一个 `ADC_InitTypeDef` 结构体，并将其成员初始化为一些默认值。然后，它使用 `HAL_ADC_Init` 函数来使用这些配置初始化 ADC1 外设。

**2. ADC_ChannelConfTypeDef 结构体 (ADC_ChannelConfTypeDef Structure)**

```c
typedef struct 
{
  uint32_t Channel;                /*!< Specifies the channel to configure into ADC regular group.
                                        This parameter can be a value of @ref ADC_channels
                                        Note: Depending on devices, some channels may not be available on package pins. Refer to device datasheet for channels availability.
                                        Note: On STM32F1 devices with several ADC: Only ADC1 can access internal measurement channels (VrefInt/TempSensor) 
                                        Note: On STM32F10xx8 and STM32F10xxB devices: A low-amplitude voltage glitch may be generated (on ADC input 0) on the PA0 pin, when the ADC is converting with injection trigger.
                                              It is advised to distribute the analog channels so that Channel 0 is configured as an injected channel.
                                              Refer to errata sheet of these devices for more details. */
  uint32_t Rank;                   /*!< Specifies the rank in the regular group sequencer 
                                        This parameter can be a value of @ref ADC_regular_rank
                                        Note: In case of need to disable a channel or change order of conversion sequencer, rank containing a previous channel setting can be overwritten by the new channel setting (or parameter number of conversions can be adjusted) */
  uint32_t SamplingTime;           /*!< Sampling time value to be set for the selected channel.
                                        Unit: ADC clock cycles
                                        Conversion time is the addition of sampling time and processing time (12.5 ADC clock cycles at ADC resolution 12 bits).
                                        This parameter can be a value of @ref ADC_sampling_times
                                        Caution: This parameter updates the parameter property of the channel, that can be used into regular and/or injected groups.
                                                 If this same channel has been previously configured in the other group (regular/injected), it will be updated to last setting.
                                        Note: In case of usage of internal measurement channels (VrefInt/TempSensor),
                                              sampling time constraints must be respected (sampling time can be adjusted in function of ADC clock frequency and sampling time setting)
                                              Refer to device datasheet for timings values, parameters TS_vrefint, TS_temp (values rough order: 5us to 17.1us min). */
}ADC_ChannelConfTypeDef;
```

   **描述:** `ADC_ChannelConfTypeDef` 结构体用于配置 ADC 通道的参数，例如选择哪个通道，在序列中的优先级 (Rank)，以及采样时间。

   **如何使用:**  在使用 `HAL_ADC_ConfigChannel` 函数配置 ADC 通道时，需要填充这个结构体。 例如，可以设置 `Channel` 为 `ADC_CHANNEL_0` 来选择 ADC 的通道 0。

   **演示:**
   ```c
   ADC_ChannelConfTypeDef channel_config;

   channel_config.Channel = ADC_CHANNEL_0;
   channel_config.Rank = 1;
   channel_config.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;

   HAL_ADC_ConfigChannel(&hadc1, &channel_config); // 配置 ADC 通道
   ```
   这段代码创建了一个 `ADC_ChannelConfTypeDef` 结构体，并将其成员初始化为要配置 ADC 的通道 0，优先级为 1，采样时间为 1.5 个 ADC 时钟周期。 然后，它使用 `HAL_ADC_ConfigChannel` 函数使用这些配置来配置 ADC1 通道。

**3.  HAL_ADC_Start 和 HAL_ADC_GetValue 函数 (HAL_ADC_Start and HAL_ADC_GetValue Functions)**

```c
HAL_StatusTypeDef HAL_ADC_Start(ADC_HandleTypeDef* hadc);
uint32_t          HAL_ADC_GetValue(ADC_HandleTypeDef* hadc);
```

   **描述:** `HAL_ADC_Start` 函数用于启动 ADC 转换， `HAL_ADC_GetValue` 函数用于读取 ADC 转换后的值。

   **如何使用:**  首先调用 `HAL_ADC_Start` 启动转换，然后等待转换完成（可以使用轮询或中断），最后调用 `HAL_ADC_GetValue` 获取转换结果。

   **演示:**
   ```c
   HAL_ADC_Start(&hadc1); // 启动 ADC 转换
   HAL_ADC_PollForConversion(&hadc1, 100); // 等待转换完成 (100ms 超时)

   uint32_t adc_value = HAL_ADC_GetValue(&hadc1); // 获取 ADC 值

   HAL_ADC_Stop(&hadc1); // 停止 ADC
   ```

   这段代码首先启动 ADC 转换，然后轮询等待转换完成。 转换完成后，它读取 ADC 的值并停止 ADC。

**4. 宏 __HAL_ADC_ENABLE 和 __HAL_ADC_DISABLE (Macros __HAL_ADC_ENABLE and __HAL_ADC_DISABLE)**

```c
#define __HAL_ADC_ENABLE(__HANDLE__)                                           \
  (SET_BIT((__HANDLE__)->Instance->CR2, (ADC_CR2_ADON)))
#define __HAL_ADC_DISABLE(__HANDLE__)                                          \
  (CLEAR_BIT((__HANDLE__)->Instance->CR2, (ADC_CR2_ADON)))
```

   **描述:** 这两个宏用于启用和禁用 ADC 外设。 它们通过设置或清除 ADC 的控制寄存器 2 (CR2) 中的 ADON (ADC On) 位来实现。

   **如何使用:** 在初始化和使用 ADC 之前，需要使用 `__HAL_ADC_ENABLE` 启用 ADC。  在使用完毕后，可以调用 `__HAL_ADC_DISABLE` 关闭 ADC。

   **演示:**
   ```c
   __HAL_ADC_ENABLE(&hadc1); // 启用 ADC1

   // ... 执行 ADC 转换 ...

   __HAL_ADC_DISABLE(&hadc1); // 禁用 ADC1
   ```

   这段代码展示了如何使用 `__HAL_ADC_ENABLE` 宏启用 ADC1，并在 ADC 转换完成后使用 `__HAL_ADC_DISABLE` 宏禁用它。

**总结:**

这些代码片段展示了 STM32F1xx HAL 库中 ADC 外设的关键组成部分和常用函数。通过配置 `ADC_InitTypeDef` 和 `ADC_ChannelConfTypeDef` 结构体，可以灵活地控制 ADC 的转换过程。 HAL 库提供的函数和宏简化了 ADC 的初始化、启动、读取和停止操作。 这些示例可以帮助你快速开始在 STM32F1xx 设备上使用 ADC。
