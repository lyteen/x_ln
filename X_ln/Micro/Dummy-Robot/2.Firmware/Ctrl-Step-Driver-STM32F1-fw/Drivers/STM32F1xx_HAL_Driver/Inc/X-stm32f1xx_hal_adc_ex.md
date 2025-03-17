Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_adc_ex.h`

好的，让我们重新开始，并针对STM32F1xx HAL ADC扩展库（`stm32f1xx_hal_adc_ex.h`）提供更细致的讲解和示例。我们将把代码片段分解，并使用中文进行详细描述。

**1. ADC_InjectionConfTypeDef 结构体详解**

```c
typedef struct
{
  uint32_t InjectedChannel;
  uint32_t InjectedRank;
  uint32_t InjectedSamplingTime;
  uint32_t InjectedOffset;
  uint32_t InjectedNbrOfConversion;
  FunctionalState InjectedDiscontinuousConvMode;
  FunctionalState AutoInjectedConv;
  uint32_t ExternalTrigInjecConv;
} ADC_InjectionConfTypeDef;
```

**描述:** 这个结构体定义了注入通道（Injected Channel）的配置。 注入通道允许你在常规ADC转换序列之外插入一个或多个ADC转换，通常用于需要更快响应的关键测量。

*   **`InjectedChannel`**:  指定要配置的ADC通道。  可以是`ADC_channels`枚举中的一个值。  **注意：** 某些通道可能在特定型号的STM32F1上不可用，请参考数据手册。 此外，在具有多个ADC的设备上，只有ADC1可以访问内部测量通道（VrefInt/TempSensor）。

    *中文描述：`InjectedChannel`：指定要使用的ADC输入引脚或内部信号（例如，内部参考电压或温度传感器）。  例如，可以使用`ADC_CHANNEL_0`来选择PA0引脚。*

*   **`InjectedRank`**:  在注入组序列器中的排名。 取值必须是`ADCEx_injected_rank`中的一个。 排名决定了注入通道转换的顺序。

    *中文描述：`InjectedRank`：如果配置多个注入通道，此参数确定每个通道的转换顺序。 例如，`ADC_INJECTED_RANK_1`表示该通道将首先进行转换。*

*   **`InjectedSamplingTime`**:  选择通道的采样时间值。 单位是ADC时钟周期。  可以是`ADC_sampling_times`枚举中的一个值。

    *中文描述：`InjectedSamplingTime`：设置ADC对输入信号进行采样的时间。  更长的采样时间可以提高精度，但会降低转换速度。 确保采样时间足够长，以满足内部参考电压或温度传感器的要求。*

*   **`InjectedOffset`**: 定义从原始转换数据中减去的偏移量（仅适用于注入组中的通道）。  Offset值必须为正数。

    *中文描述：`InjectedOffset`：用于校准ADC读数。 如果已知ADC读数存在偏差，可以使用此参数进行补偿。*

*   **`InjectedNbrOfConversion`**: 指定注入组序列器中将转换的等级数。 要使用注入组序列器并转换多个等级，必须启用'ScanConvMode'参数。  取值范围为1到4。

    *中文描述：`InjectedNbrOfConversion`：确定有多少个注入通道会被转换。 如果只想转换一个注入通道，则设置为1。 如果要转换多个，则设置为相应的数量（最多4个）。 需要启用扫描模式（ScanConvMode）才能转换多个注入通道。*

*   **`InjectedDiscontinuousConvMode`**:  指定注入组的转换序列是在完整序列/不连续序列中执行（主序列细分为连续部分）。仅当启用序列器（参数'ScanConvMode'）时才使用不连续模式。 仅当禁用连续模式时才能启用不连续模式。 可以设置为`ENABLE`或`DISABLE`。

    *中文描述：`InjectedDiscontinuousConvMode`：如果启用，注入转换序列将被分成多个部分。 这可以用于在转换之间执行其他任务。*

*   **`AutoInjectedConv`**: 启用或禁用在常规组转换后自动注入组转换。 可以设置为`ENABLE`或`DISABLE`。

    *中文描述：`AutoInjectedConv`：如果启用，则在完成常规ADC转换后，将自动触发注入转换。 这可用于创建优先级较高的ADC测量。*

*   **`ExternalTrigInjecConv`**:  选择用于触发注入组转换启动的外部事件。 如果设置为`ADC_INJECTED_SOFTWARE_START`，则禁用外部触发器。 如果设置为外部触发源，则在事件上升沿触发。

    *中文描述：`ExternalTrigInjecConv`：选择触发注入转换的外部信号。 可以使用定时器的输出或外部中断。 如果设置为`ADC_INJECTED_SOFTWARE_START`，则使用软件触发。*

**2. HAL_ADCEx_InjectedConfigChannel 函数详解**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedConfigChannel(ADC_HandleTypeDef* hadc, ADC_InjectionConfTypeDef* sConfigInjected);
```

**描述:**  此函数用于配置一个ADC注入通道。

*   **`hadc`**:  ADC句柄。
*   **`sConfigInjected`**:  指向包含注入通道配置信息的`ADC_InjectionConfTypeDef`结构体的指针。

**示例:**

```c
ADC_HandleTypeDef hadc1; // 假设已经初始化了ADC1
ADC_InjectionConfTypeDef sInjConfig;

sInjConfig.InjectedChannel = ADC_CHANNEL_4; // 使用ADC1的通道4（PA4）
sInjConfig.InjectedRank = ADC_INJECTED_RANK_1; // 注入序列中的第一个
sInjConfig.InjectedSamplingTime = ADC_SAMPLETIME_1CYCLE_5; // 1.5个ADC时钟周期
sInjConfig.InjectedOffset = 0; // 没有偏移量
sInjConfig.InjectedNbrOfConversion = 1; // 只转换一个通道
sInjConfig.InjectedDiscontinuousConvMode = DISABLE; // 禁用不连续模式
sInjConfig.AutoInjectedConv = DISABLE; // 禁用自动注入
sInjConfig.ExternalTrigInjecConv = ADC_INJECTED_SOFTWARE_START; // 使用软件触发

HAL_StatusTypeDef status = HAL_ADCEx_InjectedConfigChannel(&hadc1, &sInjConfig);
if (status != HAL_OK) {
  // 处理错误
  printf("注入通道配置失败\r\n");
} else {
  printf("注入通道配置成功\r\n");
}
```

*中文描述：这个例子展示了如何配置ADC1的注入通道4（连接到PA4引脚）。  采样时间设置为1.5个ADC时钟周期，没有偏移量，并且只转换一个注入通道。  不使用外部触发或自动注入，而是使用软件触发。  函数返回`HAL_OK`表示配置成功。*

**3.  HAL_ADCEx_InjectedStart 和 HAL_ADCEx_InjectedStop**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedStart(ADC_HandleTypeDef* hadc);
HAL_StatusTypeDef HAL_ADCEx_InjectedStop(ADC_HandleTypeDef* hadc);
```

**描述:**

*   `HAL_ADCEx_InjectedStart`: 启动注入组的ADC转换（阻塞模式）。
*   `HAL_ADCEx_InjectedStop`:  停止注入组的ADC转换。

**示例:**

```c
// 启动注入转换
HAL_StatusTypeDef start_status = HAL_ADCEx_InjectedStart(&hadc1);
if (start_status != HAL_OK) {
  // 处理启动错误
  printf("注入转换启动失败\r\n");
} else {
    // 等待转换完成
    HAL_StatusTypeDef poll_status = HAL_ADCEx_InjectedPollForConversion(&hadc1, 100); // 100ms 超时
    if(poll_status != HAL_OK){
        printf("注入转换超时或错误\r\n");
    } else {
        // 获取转换值
        uint32_t injectedValue = HAL_ADCEx_InjectedGetValue(&hadc1, ADC_INJECTED_RANK_1);
        printf("注入转换值: %lu\r\n", injectedValue);
    }

  // 停止注入转换
  HAL_StatusTypeDef stop_status = HAL_ADCEx_InjectedStop(&hadc1);
  if (stop_status != HAL_OK) {
    // 处理停止错误
    printf("注入转换停止失败\r\n");
  }
}
```

*中文描述：首先，我们尝试启动注入转换。 如果启动成功，我们使用`HAL_ADCEx_InjectedPollForConversion`函数等待转换完成。 然后，我们使用`HAL_ADCEx_InjectedGetValue`函数获取转换后的值。 最后，我们停止注入转换。 所有的函数调用都会检查返回值，以确保操作成功。*

**4. HAL_ADCEx_InjectedStart_IT 和 HAL_ADCEx_InjectedStop_IT**

```c
HAL_StatusTypeDef HAL_ADCEx_InjectedStart_IT(ADC_HandleTypeDef* hadc);
HAL_StatusTypeDef HAL_ADCEx_InjectedStop_IT(ADC_HandleTypeDef* hadc);
```

**描述:**

*   `HAL_ADCEx_InjectedStart_IT`: 启动注入组的ADC转换（中断模式）。
*   `HAL_ADCEx_InjectedStop_IT`:  停止注入组的ADC转换。

**示例:**

```c
// 启动注入转换（中断模式）
HAL_StatusTypeDef start_status = HAL_ADCEx_InjectedStart_IT(&hadc1);
if (start_status != HAL_OK) {
    // 处理启动错误
    printf("注入转换(中断)启动失败\r\n");
}

// 在HAL_ADCEx_InjectedConvCpltCallback 中处理转换完成
void HAL_ADCEx_InjectedConvCpltCallback(ADC_HandleTypeDef* hadc)
{
    if(hadc->Instance == ADC1){
        uint32_t injectedValue = HAL_ADCEx_InjectedGetValue(hadc, ADC_INJECTED_RANK_1);
        printf("注入转换值(中断): %lu\r\n", injectedValue);

        // 停止转换
        HAL_ADCEx_InjectedStop_IT(hadc); //或者可以继续下一次转换
    }
}
```

*中文描述：这个例子展示了如何在中断模式下使用注入转换。  `HAL_ADCEx_InjectedStart_IT`函数启动转换，转换完成后会触发`HAL_ADCEx_InjectedConvCpltCallback`回调函数。  在这个回调函数中，我们可以获取转换值并停止转换。 需要在NVIC中启用ADC中断。*

**5.  HAL_ADCEx_MultiModeConfigChannel 和 HAL_ADCEx_MultiModeStart_DMA (针对具有多个ADC的设备)**

```c
#if defined (STM32F103x6) || defined (STM32F103xB) || defined (STM32F105xC) || defined (STM32F107xC) || defined (STM32F103xE) || defined (STM32F103xG)
HAL_StatusTypeDef HAL_ADCEx_MultiModeConfigChannel(ADC_HandleTypeDef *hadc, ADC_MultiModeTypeDef *multimode);
HAL_StatusTypeDef HAL_ADCEx_MultiModeStart_DMA(ADC_HandleTypeDef *hadc, uint32_t *pData, uint32_t Length);
HAL_StatusTypeDef HAL_ADCEx_MultiModeStop_DMA(ADC_HandleTypeDef *hadc);
uint32_t HAL_ADCEx_MultiModeGetValue(ADC_HandleTypeDef *hadc);
#endif
```

**描述:**  这些函数用于配置和启动多ADC模式，例如双重ADC同步采样。  这些函数仅在具有多个ADC的STM32F1设备上可用。

*   **`HAL_ADCEx_MultiModeConfigChannel`**: 配置多ADC模式。
*   **`HAL_ADCEx_MultiModeStart_DMA`**: 使用DMA启动多ADC模式。
*   **`HAL_ADCEx_MultiModeStop_DMA`**: 停止多ADC模式的DMA传输。
*   **`HAL_ADCEx_MultiModeGetValue`**: 获取多ADC模式下的转换值。

**示例 (双ADC同步采样):**

```c
#if defined (STM32F103x6) || defined (STM32F103xB) || defined (STM32F105xC) || defined (STM32F107xC) || defined (STM32F103xE) || defined (STM32F103xG)

ADC_HandleTypeDef hadc1, hadc2; // 假设已经初始化了ADC1和ADC2
ADC_MultiModeTypeDef multiModeConfig;
uint32_t ADC_Buffer[100];

// 配置多ADC模式 (假设使用同步模式)
multiModeConfig.Mode = ADC_DUALMODE_REGSIMULT; // 同步规则转换
HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multiModeConfig);

// 启动DMA
HAL_StatusTypeDef dma_status = HAL_ADCEx_MultiModeStart_DMA(&hadc1, ADC_Buffer, 100);
if (dma_status != HAL_OK) {
    printf("多ADC DMA启动失败\r\n");
}

// (转换将在后台通过DMA进行)

// ... 稍后停止DMA
HAL_ADCEx_MultiModeStop_DMA(&hadc1);

// ADC_Buffer现在包含ADC1和ADC2的交替读数

#endif
```

*中文描述：这个例子演示了如何在具有多个ADC的设备上配置双ADC同步采样。  首先，配置`ADC_DUALMODE_REGSIMULT`模式。然后，使用DMA启动转换，并将结果存储在`ADC_Buffer`中。  DMA将自动填充缓冲区，缓冲区中包含ADC1和ADC2交替读取的数据。  最后，停止DMA。需要配置DMA控制器和时钟，并确保两个ADC具有相同的采样时间。还要注意，`hadc1`通常被认为是主ADC，配置在`hadc1`上进行。*

**总结:**

`stm32f1xx_hal_adc_ex.h` 文件提供了对STM32F1系列微控制器的ADC外设的扩展功能，特别是注入通道和多ADC模式。 理解这些函数和结构体，并正确配置参数，可以实现更灵活和高效的ADC测量。 结合具体的硬件和应用需求，可以进一步优化ADC性能。  Remember to consult the STM32F1 reference manual for precise details on specific trigger sources, timings, and limitations for your device. 务必参考STM32F1的参考手册，以了解特定触发源、时序和设备的限制的详细信息。
