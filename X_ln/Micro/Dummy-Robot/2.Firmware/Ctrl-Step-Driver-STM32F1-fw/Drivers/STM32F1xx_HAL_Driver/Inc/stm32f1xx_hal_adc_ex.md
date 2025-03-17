Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_adc_ex.h`

好的，重新开始。我将逐个分解 STM32F1xx HAL 库的 `stm32f1xx_hal_adc_ex.h` 文件，解释关键部分，并提供代码片段和使用示例，用中文描述。

**1. 文件头注释:**

```c
/**
  ******************************************************************************
  * @file    stm32f1xx_hal_adc_ex.h
  * @author  MCD Application Team
  * @brief   Header file of ADC HAL extension module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2016 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
```

**描述:**  这段注释是标准的文件头，说明了该文件的名称、作者、简要描述、版权信息和许可协议。  `stm32f1xx_hal_adc_ex.h`  是 STM32F1 系列单片机 ADC (模数转换器) HAL (硬件抽象层) 扩展模块的头文件。 扩展模块通常包含标准 HAL 驱动之外的额外功能。

**2. 头文件保护:**

```c
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F1xx_HAL_ADC_EX_H
#define __STM32F1xx_HAL_ADC_EX_H

#ifdef __cplusplus
 extern "C" {
#endif
```

**描述:** 这是头文件保护，防止头文件被重复包含。`#ifndef`、`#define` 和 `#endif` 用于创建宏，确保头文件只被包含一次。  `#ifdef __cplusplus extern "C" {`  用于 C++ 环境，确保 C 函数可以被 C++ 代码调用。

**3. 包含头文件:**

```c
/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal_def.h"
```

**描述:**  `#include "stm32f1xx_hal_def.h"`  包含了 HAL 库的定义文件，其中定义了 HAL 库的基础类型、宏和函数。 这是使用 HAL 库所必需的。

**4. 外设组定义:**

```c
/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @addtogroup ADCEx
  * @{
  */
```

**描述:**  `@addtogroup`  是 Doxygen 风格的注释，用于将代码组织成模块和组。 这里定义了 STM32F1xx HAL 驱动组和 ADCEx (ADC 扩展) 组，用于组织 ADC 相关的代码。

**5. 数据类型定义 (ADC_InjectionConfTypeDef):**

```c
/**
  * @brief  ADC Configuration injected Channel structure definition
  */
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
}ADC_InjectionConfTypeDef;
```

**描述:**  `ADC_InjectionConfTypeDef`  结构体定义了注入通道 (injected channel) 的配置。  注入通道是一种特殊的 ADC 转换模式，可以优先于常规转换执行。

*   **`InjectedChannel`:** 选择要配置的 ADC 通道 (@ref ADC_channels)。
*   **`InjectedRank`:**  在注入组序列中的等级 (@ref ADCEx_injected_rank)，决定转换顺序。
*   **`InjectedSamplingTime`:**  通道的采样时间 (@ref ADC_sampling_times)。
*   **`InjectedOffset`:**  从原始转换数据中减去的偏移量。
*   **`InjectedNbrOfConversion`:** 注入组序列中要转换的等级数量 (1-4)。
*   **`InjectedDiscontinuousConvMode`:**  是否启用注入组的不连续转换模式 (`ENABLE` 或 `DISABLE`)。
*   **`AutoInjectedConv`:**  是否在常规转换后自动启动注入组转换 (`ENABLE` 或 `DISABLE`)。
*   **`ExternalTrigInjecConv`:**  选择启动注入组转换的外部事件 (@ref ADCEx_External_trigger_source_Injected)。

**使用示例:**

```c
ADC_InjectionConfTypeDef  sInjConfig;

sInjConfig.InjectedChannel = ADC_CHANNEL_5;
sInjConfig.InjectedRank = ADC_INJECTED_RANK_1;
sInjConfig.InjectedSamplingTime = ADC_SAMPLETIME_239_5CYCLES;
sInjConfig.InjectedOffset = 0;
sInjConfig.InjectedNbrOfConversion = 1;
sInjConfig.InjectedDiscontinuousConvMode = DISABLE;
sInjConfig.AutoInjectedConv = DISABLE;
sInjConfig.ExternalTrigInjecConv = ADC_EXTERNALTRIGINJECCONV_T1_CC4; // 使用定时器1的通道4作为触发源

HAL_ADCEx_InjectedConfigChannel(&hadc1, &sInjConfig);  // hadc1 是你的 ADC 句柄
```

**描述:**  这段代码配置 ADC1 的注入通道 5，将其设置为注入组中的第一个转换，采样时间为 239.5 个时钟周期，无偏移量，注入组只转换一个通道，禁用不连续转换和自动注入，使用定时器1的通道4作为触发源。

**6. 数据类型定义 (ADC_MultiModeTypeDef):**

```c
#if defined (STM32F103x6) || defined (STM32F103xB) || defined (STM32F105xC) || defined (STM32F107xC) || defined (STM32F103xE) || defined (STM32F103xG)
/**
  * @brief  Structure definition of ADC multimode
  * @note   The setting of these parameters with function HAL_ADCEx_MultiModeConfigChannel() is conditioned to ADCs state (both ADCs of the common group).
  *         State of ADCs of the common group must be: disabled.
  */
typedef struct
{
  uint32_t Mode;              /*!< Configures the ADC to operate in independent or multi mode.
                                   This parameter can be a value of @ref ADCEx_Common_mode */
}ADC_MultiModeTypeDef;
#endif /* defined STM32F103x6 || defined STM32F103xB || defined STM32F105xC || defined STM32F107xC || defined STM32F103xE || defined STM32F103xG */
```

**描述:** `ADC_MultiModeTypeDef` 结构体定义了多 ADC 模式的配置。只有部分 STM32F1 器件支持多 ADC 模式。 在多 ADC 模式下，多个 ADC 可以同步工作。

*   **`Mode`:** 配置 ADC 是以独立模式还是多模式运行 (@ref ADCEx_Common_mode)。

**使用示例:**

```c
#ifdef HAL_ADC_MODULE_ENABLED // 确保ADC HAL模块已启用
#if defined (STM32F103x6) || defined (STM32F103xB) || defined (STM32F105xC) || defined (STM32F107xC) || defined (STM32F103xE) || defined (STM32F103xG)
ADC_MultiModeTypeDef multiModeConfig;
multiModeConfig.Mode = ADC_DUALMODE_REGSIMULT; // 配置为规则同步模式
HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multiModeConfig);
#endif
#endif
```

**描述:**  这段代码配置 ADC1 和 ADC2 以规则同步模式运行。 `hadc1` 仍然是你的 ADC1 句柄。  在同步模式下，两个 ADC 会同时进行转换。

**7. 导出常量 (Exported Constants):**

这部分定义了许多宏，用于配置 ADC 的各种参数。

*   **`ADCEx_injected_rank`:** 定义了注入通道在注入组中的排名，例如 `ADC_INJECTED_RANK_1` 到 `ADC_INJECTED_RANK_4`。
*   **`ADCEx_External_trigger_edge_Injected`:** 定义了注入转换的外部触发边沿，例如 `ADC_EXTERNALTRIGINJECCONV_EDGE_NONE` 和 `ADC_EXTERNALTRIGINJECCONV_EDGE_RISING`。
*   **`ADC_External_trigger_source_Regular`:** 定义了规则转换的外部触发源，例如 `ADC_EXTERNALTRIGCONV_T1_CC1` (定时器 1 通道 1) 和 `ADC_SOFTWARE_START` (软件触发)。
*   **`ADCEx_External_trigger_source_Injected`:** 定义了注入转换的外部触发源，例如 `ADC_EXTERNALTRIGINJECCONV_T2_TRGO` (定时器 2 TRGO) 和 `ADC_INJECTED_SOFTWARE_START` (软件触发)。
*   **`ADCEx_Common_mode`:** (如果支持多 ADC 模式) 定义了多 ADC 模式，例如 `ADC_MODE_INDEPENDENT` (独立模式) 和 `ADC_DUALMODE_REGSIMULT` (规则同步模式)。

**8. 宏定义 (Macros):**

这部分定义了一些宏，用于简化代码和提高可读性。  例如，`ADC_CFGR_EXTSEL` 宏用于设置 ADC 的外部触发源。 `ADC_MULTIMODE_IS_ENABLE`宏用于判断当前是否开启多ADC模式等等。

**9. 函数声明 (Exported Functions):**

这部分声明了 HAL 库提供的函数，用于控制 ADC 扩展功能。

*   **`HAL_ADCEx_Calibration_Start()`:** 启动 ADC 校准。
*   **`HAL_ADCEx_InjectedStart()` / `HAL_ADCEx_InjectedStop()`:** 启动/停止注入转换 (阻塞模式)。
*   **`HAL_ADCEx_InjectedPollForConversion()`:**  轮询等待注入转换完成。
*   **`HAL_ADCEx_InjectedStart_IT()` / `HAL_ADCEx_InjectedStop_IT()`:**  启动/停止注入转换 (中断模式)。
*   **`HAL_ADCEx_MultiModeStart_DMA()` / `HAL_ADCEx_MultiModeStop_DMA()`:**  (如果支持多 ADC 模式) 启动/停止多 ADC 模式下的 DMA 转换。
*   **`HAL_ADCEx_InjectedGetValue()`:** 获取注入通道的转换值。
*    **`HAL_ADCEx_MultiModeGetValue()`:** 获取多 ADC 模式下的转换值.
*   **`HAL_ADCEx_InjectedConfigChannel()`:** 配置注入通道。
*   **`HAL_ADCEx_MultiModeConfigChannel()`:**  (如果支持多 ADC 模式) 配置多 ADC 模式。

**综合示例 (使用注入模式):**

```c
#include "stm32f1xx_hal.h"
#include "stm32f1xx_hal_adc.h"
#include "stm32f1xx_hal_adc_ex.h"

ADC_HandleTypeDef hadc1; // ADC句柄

void ADC1_Init(void)
{
    hadc1.Instance = ADC1;  // 选择ADC1
    hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE; // 禁用扫描模式，只转换一个通道
    hadc1.Init.ContinuousConvMode = DISABLE; // 禁用连续转换模式
    hadc1.Init.DiscontinuousConvMode = DISABLE; // 禁用非连续转换模式 (规则组)
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START; // 软件触发规则组转换
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT; // 数据右对齐
    hadc1.Init.NbrOfConversion = 1;            // 规则组只转换一个通道

    if (HAL_ADC_Init(&hadc1) != HAL_OK) {
        // 初始化错误处理
    }
}

void ADC1_InjectedConfig(void)
{
    ADC_InjectionConfTypeDef sInjConfig;

    sInjConfig.InjectedChannel = ADC_CHANNEL_5; // 选择 ADC 通道 5
    sInjConfig.InjectedRank = ADC_INJECTED_RANK_1; // 注入组的第一个转换
    sInjConfig.InjectedSamplingTime = ADC_SAMPLETIME_239_5CYCLES; // 采样时间
    sInjConfig.InjectedOffset = 0; // 无偏移量
    sInjConfig.InjectedNbrOfConversion = 1; // 注入组只转换一个通道
    sInjConfig.InjectedDiscontinuousConvMode = DISABLE; // 禁用不连续转换模式
    sInjConfig.AutoInjectedConv = DISABLE; // 禁用自动注入
    sInjConfig.ExternalTrigInjecConv = ADC_EXTERNALTRIGINJECCONV_T1_CC4; // 使用定时器1的通道4作为触发源，也可以改成软件触发 ADC_INJECTED_SOFTWARE_START

    if (HAL_ADCEx_InjectedConfigChannel(&hadc1, &sInjConfig) != HAL_OK) {
        // 配置错误处理
    }
}


int main(void)
{
    HAL_Init();  // 初始化 HAL 库
    SystemClock_Config(); // 配置系统时钟
    GPIO_Init(); // 初始化GPIO

    ADC1_Init();  // 初始化 ADC1
    ADC1_InjectedConfig(); // 配置注入通道

    // 启动定时器1，配置CH4作为触发源。具体定时器配置代码省略。
    // ... Timer1 configuration ...

    HAL_ADCEx_InjectedStart(&hadc1);  // 启动注入转换

    if (HAL_ADCEx_InjectedPollForConversion(&hadc1, 10) == HAL_OK) { // 等待转换完成，10ms超时
        uint32_t injectedValue = HAL_ADCEx_InjectedGetValue(&hadc1, ADC_INJECTED_RANK_1); // 获取转换值
        // 使用 injectedValue
    }
    HAL_ADCEx_InjectedStop(&hadc1); // 停止转换

    while (1) {
        // ...
    }
}
```

**描述:**

1.  **`ADC1_Init()`:**  初始化 ADC1 的基本参数，例如扫描模式、连续转换模式、触发源等。
2.  **`ADC1_InjectedConfig()`:**  配置 ADC1 的注入通道，包括通道选择、采样时间、偏移量、触发源等。
3.  **`main()`:**
    *   初始化 HAL 库、系统时钟和 GPIO。
    *   调用 `ADC1_Init()` 和 `ADC1_InjectedConfig()` 初始化和配置 ADC1。
    *   配置定时器1，产生PWM信号，使用T1_CC4的上升沿触发注入转换（代码省略）。
    *   `HAL_ADCEx_InjectedStart()` 启动注入转换。
    *   `HAL_ADCEx_InjectedPollForConversion()` 等待转换完成。
    *   `HAL_ADCEx_InjectedGetValue()` 获取转换结果。
    *   `HAL_ADCEx_InjectedStop()` 停止转换。

这个示例展示了如何使用注入模式，通过定时器触发ADC转换，可以实现高优先级的ADC采样。 完整的工程还需要配置GPIO引脚为模拟输入，配置定时器等，这里为了突出ADC部分，做了简化。
