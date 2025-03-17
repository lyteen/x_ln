Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Inc\stm32f1xx_hal_rcc_ex.h`

好的，现在重新开始，我会尽力提供更详细、更易懂的中文解释和使用示例。

**目标:**

对 `stm32f1xx_hal_rcc_ex.h` 文件中的关键部分进行解释，生成每个代码段，并附带中文描述，包括代码的用途和简单的演示。

**开始：**

**1. 头文件保护和包含**

```c
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F1xx_HAL_RCC_EX_H
#define __STM32F1xx_HAL_RCC_EX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal_def.h"
```

**描述：**

*   **头文件保护:** 使用 `#ifndef`, `#define` 和 `#endif` 来防止头文件的重复包含，避免编译错误。
*   **C++ 兼容性:** `extern "C"` 确保 C++ 代码可以正确链接 C 代码。
*   **包含 HAL 定义:** `#include "stm32f1xx_hal_def.h"` 包含了 HAL (Hardware Abstraction Layer，硬件抽象层) 的基本定义，如数据类型和宏定义。

**用途:**

这是所有 HAL 驱动程序的标准起始部分，确保代码的健壮性和兼容性。

**2. RCCEx 外设组和私有常量**

```c
/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @addtogroup RCCEx
  * @{
  */

/** @addtogroup RCCEx_Private_Constants
 * @{
 */

#if defined(STM32F105xC) || defined(STM32F107xC)

/* Alias word address of PLLI2SON bit */
#define PLLI2SON_BITNUMBER           RCC_CR_PLL3ON_Pos
#define RCC_CR_PLLI2SON_BB           ((uint32_t)(PERIPH_BB_BASE + (RCC_CR_OFFSET_BB * 32U) + (PLLI2SON_BITNUMBER * 4U)))
/* Alias word address of PLL2ON bit */
#define PLL2ON_BITNUMBER             RCC_CR_PLL2ON_Pos
#define RCC_CR_PLL2ON_BB             ((uint32_t)(PERIPH_BB_BASE + (RCC_CR_OFFSET_BB * 32U) + (PLL2ON_BITNUMBER * 4U)))

#define PLLI2S_TIMEOUT_VALUE         100U  /* 100 ms */
#define PLL2_TIMEOUT_VALUE           100U  /* 100 ms */

#endif /* STM32F105xC || STM32F107xC */


#define CR_REG_INDEX                 ((uint8_t)1)

/**
  * @}
  */
```

**描述：**

*   **外设分组:** `@addtogroup` 用于在文档中组织代码，方便查找和理解。`STM32F1xx_HAL_Driver` 是 HAL 驱动程序的主组，`RCCEx` 是时钟扩展模块的组。
*   **位带操作 (Bit-banding):**  `PLLI2SON_BITNUMBER`, `RCC_CR_PLLI2SON_BB`, `PLL2ON_BITNUMBER`, `RCC_CR_PLL2ON_BB` 是为支持位带操作而定义的宏。位带操作允许直接修改单个位，而无需读取-修改-写入整个寄存器。 这对于某些 STM32 器件（例如 STM32F105xC 和 STM32F107xC）上的 `PLLI2S` 和 `PLL2` 的控制很有用.
*   **超时值:** `PLLI2S_TIMEOUT_VALUE` 和 `PLL2_TIMEOUT_VALUE` 定义了等待 PLLI2S 和 PLL2 锁定的超时时间，单位为毫秒。
*   **寄存器索引:** `CR_REG_INDEX` 定义了 CR 寄存器的索引，在后续的标志定义中会使用到。

**用途:**

这些定义为更高级别的时钟配置函数提供了基础，并针对特定的 STM32 型号进行优化。

**演示:**

虽然直接演示这些定义比较困难，但理解它们是使用 HAL 时钟函数的基础。 例如，要启用 PLLI2S 时钟，HAL 库可能会使用 `RCC_CR_PLLI2SON_BB` 宏直接设置 `CR` 寄存器中的相应位。

**3. 私有宏定义 (IS_RCC_... 系列)**

```c
/** @addtogroup RCCEx_Private_Macros
 * @{
 */

#if defined(STM32F105xC) || defined(STM32F107xC)
#define IS_RCC_PREDIV1_SOURCE(__SOURCE__) (((__SOURCE__) == RCC_PREDIV1_SOURCE_HSE) || \
                                           ((__SOURCE__) == RCC_PREDIV1_SOURCE_PLL2))
#endif /* STM32F105xC || STM32F107xC */

#if defined(STM32F105xC) || defined(STM32F107xC) || defined(STM32F100xB)\
 || defined(STM32F100xE)
#define IS_RCC_HSE_PREDIV(__DIV__) (((__DIV__) == RCC_HSE_PREDIV_DIV1)  || ((__DIV__) == RCC_HSE_PREDIV_DIV2)  || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV3)  || ((__DIV__) == RCC_HSE_PREDIV_DIV4)  || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV5)  || ((__DIV__) == RCC_HSE_PREDIV_DIV6)  || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV7)  || ((__DIV__) == RCC_HSE_PREDIV_DIV8)  || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV9)  || ((__DIV__) == RCC_HSE_PREDIV_DIV10) || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV11) || ((__DIV__) == RCC_HSE_PREDIV_DIV12) || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV13) || ((__DIV__) == RCC_HSE_PREDIV_DIV14) || \
                                    ((__DIV__) == RCC_HSE_PREDIV_DIV15) || ((__DIV__) == RCC_HSE_PREDIV_DIV16))

#else
#define IS_RCC_HSE_PREDIV(__DIV__) (((__DIV__) == RCC_HSE_PREDIV_DIV1)  || ((__DIV__) == RCC_HSE_PREDIV_DIV2))
#endif /* STM32F105xC || STM32F107xC || STM32F100xB || STM32F100xE */

// ... 其他 IS_RCC_... 宏定义 ...

/**
  * @}
  */
```

**描述:**

*   **输入参数校验:** `IS_RCC_...` 宏定义用于验证函数参数的有效性。  它们接受一个参数 (`__SOURCE__`, `__DIV__`, `__MUL__` 等) ，并检查它是否是允许的值之一。 如果参数有效，则宏返回 true (非零值); 否则返回 false (零值)。
*   **设备特定差异:** `#if defined(...)` 用于处理不同 STM32 型号之间的差异。例如，`IS_RCC_HSE_PREDIV` 宏允许的 HSE 预分频值取决于具体的 STM32 型号。

**用途:**

这些宏在 HAL 库的函数中使用，以确保用户提供的配置是有效的，避免了因无效配置导致系统崩溃。

**演示:**

```c
#include "stm32f1xx_hal.h" // 假设包含了必要的 HAL 头文件

void configure_hse_prediv(uint32_t prediv_value) {
  if (IS_RCC_HSE_PREDIV(prediv_value)) {
    //  执行配置 HSE 预分频的代码， 例如：
    // RCC->CFGR2 |= prediv_value; // 实际代码可能更复杂
    printf("HSE Prediv 设置为有效值\n"); // Valid Value
  } else {
    //  处理无效参数的情况，例如：
    printf("无效的 HSE Prediv 值!\n");  // Invalid value
  }
}

// 使用示例
int main() {
  HAL_Init(); // 初始化 HAL 库（必须先调用）

  configure_hse_prediv(RCC_HSE_PREDIV_DIV8);  // 设置为 DIV8 (仅在某些设备上有效)
  configure_hse_prediv(0x12345678); // 非法的预分频值

  while (1) {}
}
```

**解释:**

1.  `configure_hse_prediv` 函数接受一个 HSE 预分频值作为输入。
2.  `IS_RCC_HSE_PREDIV(prediv_value)` 宏用于验证该值是否有效。
3.  如果预分频值有效，函数会执行相应的配置代码。如果无效，则打印一条错误消息。

**4. RCC 结构体定义和类型定义**

```c
/* Exported types ------------------------------------------------------------*/

/** @defgroup RCCEx_Exported_Types RCCEx Exported Types
  * @{
  */

#if defined(STM32F105xC) || defined(STM32F107xC)
/**
  * @brief  RCC PLL2 configuration structure definition
  */
typedef struct
{
  uint32_t PLL2State;     /*!< The new state of the PLL2.
                              This parameter can be a value of @ref RCCEx_PLL2_Config */

  uint32_t PLL2MUL;         /*!< PLL2MUL: Multiplication factor for PLL2 VCO input clock
                              This parameter must be a value of @ref RCCEx_PLL2_Multiplication_Factor*/

#if defined(STM32F105xC) || defined(STM32F107xC)
  uint32_t HSEPrediv2Value;       /*!<  The Prediv2 factor value.
                                       This parameter can be a value of @ref RCCEx_Prediv2_Factor */

#endif /* STM32F105xC || STM32F107xC */
} RCC_PLL2InitTypeDef;

#endif /* STM32F105xC || STM32F107xC */

/**
  * @brief  RCC Internal/External Oscillator (HSE, HSI, LSE and LSI) configuration structure definition
  */
typedef struct
{
  uint32_t OscillatorType;       /*!< The oscillators to be configured.
                                       This parameter can be a value of @ref RCC_Oscillator_Type */

#if defined(STM32F105xC) || defined(STM32F107xC)
  uint32_t Prediv1Source;       /*!<  The Prediv1 source value.
                                       This parameter can be a value of @ref RCCEx_Prediv1_Source */
#endif /* STM32F105xC || STM32F107xC */

  uint32_t HSEState;              /*!< The new state of the HSE.
                                       This parameter can be a value of @ref RCC_HSE_Config */

  uint32_t HSEPredivValue;       /*!<  The Prediv1 factor value (named PREDIV1 or PLLXTPRE in RM)
                                       This parameter can be a value of @ref RCCEx_Prediv1_Factor */

  uint32_t LSEState;              /*!<  The new state of the LSE.
                                        This parameter can be a value of @ref RCC_LSE_Config */

  uint32_t HSIState;              /*!< The new state of the HSI.
                                       This parameter can be a value of @ref RCC_HSI_Config */

  uint32_t HSICalibrationValue;   /*!< The HSI calibration trimming value (default is RCC_HSICALIBRATION_DEFAULT).
                                       This parameter must be a number between Min_Data = 0x00 and Max_Data = 0x1F */

  uint32_t LSIState;              /*!<  The new state of the LSI.
                                        This parameter can be a value of @ref RCC_LSI_Config */

  RCC_PLLInitTypeDef PLL;         /*!< PLL structure parameters */

#if defined(STM32F105xC) || defined(STM32F107xC)
  RCC_PLL2InitTypeDef PLL2;         /*!< PLL2 structure parameters */
#endif /* STM32F105xC || STM32F107xC */
} RCC_OscInitTypeDef;

#if defined(STM32F105xC) || defined(STM32F107xC)
/**
  * @brief  RCC PLLI2S configuration structure definition
  */
typedef struct
{
  uint32_t PLLI2SMUL;         /*!< PLLI2SMUL: Multiplication factor for PLLI2S VCO input clock
                              This parameter must be a value of @ref RCCEx_PLLI2S_Multiplication_Factor*/

#if defined(STM32F105xC) || defined(STM32F107xC)
  uint32_t HSEPrediv2Value;       /*!<  The Prediv2 factor value.
                                       This parameter can be a value of @ref RCCEx_Prediv2_Factor */

#endif /* STM32F105xC || STM32F107xC */
} RCC_PLLI2SInitTypeDef;
#endif /* STM32F105xC || STM32F107xC */

/**
  * @brief  RCC extended clocks structure definition
  */
typedef struct
{
  uint32_t PeriphClockSelection;      /*!< The Extended Clock to be configured.
                                       This parameter can be a value of @ref RCCEx_Periph_Clock_Selection */

  uint32_t RTCClockSelection;         /*!< specifies the RTC clock source.
                                       This parameter can be a value of @ref RCC_RTC_Clock_Source */

  uint32_t AdcClockSelection;         /*!< ADC clock source
                                       This parameter can be a value of @ref RCCEx_ADC_Prescaler */

#if defined(STM32F103xE) || defined(STM32F103xG) || defined(STM32F105xC)\
 || defined(STM32F107xC)
  uint32_t I2s2ClockSelection;         /*!< I2S2 clock source
                                       This parameter can be a value of @ref RCCEx_I2S2_Clock_Source */

  uint32_t I2s3ClockSelection;         /*!< I2S3 clock source
                                       This parameter can be a value of @ref RCCEx_I2S3_Clock_Source */

#if defined(STM32F105xC) || defined(STM32F107xC)
  RCC_PLLI2SInitTypeDef PLLI2S;  /*!< PLL I2S structure parameters
                                      This parameter will be used only when PLLI2S is selected as Clock Source I2S2 or I2S3 */

#endif /* STM32F105xC || STM32F107xC */
#endif /* STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */

#if defined(STM32F102x6) || defined(STM32F102xB) || defined(STM32F103x6)\
 || defined(STM32F103xB) || defined(STM32F103xE) || defined(STM32F103xG)\
 || defined(STM32F105xC) || defined(STM32F107xC)
  uint32_t UsbClockSelection;         /*!< USB clock source
                                       This parameter can be a value of @ref RCCEx_USB_Prescaler */

#endif /* STM32F102x6 || STM32F102xB || STM32F103x6 || STM32F103xB || STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */
} RCC_PeriphCLKInitTypeDef;

/**
  * @}
  */
```

**描述:**

*   **结构体定义:** 定义了多个结构体 (`RCC_PLL2InitTypeDef`, `RCC_OscInitTypeDef`, `RCC_PLLI2SInitTypeDef`, `RCC_PeriphCLKInitTypeDef`)，用于配置 RCC 相关的各种参数。 这些结构体将相关的时钟配置选项组合在一起，方便用户进行配置。
*   **条件编译:**  `#if defined(...)` 用于根据目标 STM32 型号选择性地包含结构体成员。  例如，`PLL2State` 和 `PLL2MUL` 成员只在支持 PLL2 的 STM32F105xC 和 STM32F107xC 型号上才定义。
*   **注释:** 每个结构体成员都有详细的注释，描述了其用途和取值范围，并链接到相关的宏定义。

**用途:**

这些结构体是 HAL 库配置时钟的关键， 它们被传递给 HAL 函数来设置不同的时钟源、分频系数和外设时钟。

**演示:**

```c
#include "stm32f1xx_hal.h"

void configure_clocks(void) {
  RCC_OscInitTypeDef osc_init = {0};
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};

  // 1. 配置振荡器
  osc_init.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  osc_init.HSEState = RCC_HSE_ON;
  osc_init.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  osc_init.PLL.PLLState = RCC_PLL_ON;
  osc_init.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  osc_init.PLL.PLLMUL = RCC_PLL_MUL9;

  if (HAL_RCC_OscConfig(&osc_init) != HAL_OK) {
    // 处理错误
    printf("RCC 振荡器配置失败!\n"); // RCC Oscillator Config Failed
  }

  // 2. 配置外设时钟 (例如 ADC)
  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  periph_clk_init.AdcClockSelection = RCC_ADCPCLK2_DIV6;  // PCLK2 / 6

  if (HAL_RCCEx_PeriphCLKConfig(&periph_clk_init) != HAL_OK) {
    // 处理错误
    printf("外设时钟配置失败!\n"); // Peripheral Clock Config Failed
  }

  // 3. 选择系统时钟源 (例如 PLL)
  RCC_ClkInitTypeDef clk_init = {0};
  clk_init.ClockType = RCC_CLOCKTYPE_SYSCLK;
  clk_init.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  clk_init.AHBCLKDivider = RCC_SYSCLK_DIV1;
  clk_init.APB1CLKDivider = RCC_HCLK_DIV2;  // PCLK1 = HCLK / 2
  clk_init.APB2CLKDivider = RCC_HCLK_DIV1;  // PCLK2 = HCLK

  if (HAL_RCC_ClockConfig(&clk_init, FLASH_LATENCY_2) != HAL_OK) {
    // 处理错误
    printf("系统时钟配置失败!\n"); // System Clock Config Failed
  }

  printf("时钟配置完成!\n"); // Clock Config Completed
}

int main() {
  HAL_Init();

  configure_clocks();

  while (1) {}
}
```

**解释:**

1.  `configure_clocks` 函数演示了如何使用 `RCC_OscInitTypeDef` 和 `RCC_PeriphCLKInitTypeDef` 结构体来配置时钟。
2.  首先，配置 HSE 振荡器，启用 PLL，设置 PLL 的倍频因子和时钟源。
3.  然后，配置 ADC 的时钟源为 PCLK2 的 1/6.
4.  最后，选择 PLL 作为系统时钟源，并配置 AHB 和 APB 总线的分频系数。
5.  `HAL_RCC_OscConfig` 和 `HAL_RCCEx_PeriphCLKConfig` 函数将这些结构体作为参数，实际配置 RCC 寄存器。
6.  请注意，错误检查是必不可少的。

**5. 导出常量 (RCCEx_Periph_Clock_Selection等)**

```c
/* Exported constants --------------------------------------------------------*/

/** @defgroup RCCEx_Exported_Constants RCCEx Exported Constants
  * @{
  */

/** @defgroup RCCEx_Periph_Clock_Selection Periph Clock Selection
  * @{
  */
#define RCC_PERIPHCLK_RTC           0x00000001U
#define RCC_PERIPHCLK_ADC           0x00000002U
#if defined(STM32F101xE) || defined(STM32F101xG) || defined(STM32F103xE)\
 || defined(STM32F103xG) || defined(STM32F105xC) || defined(STM32F107xC)
#define RCC_PERIPHCLK_I2S2          0x00000004U
#define RCC_PERIPHCLK_I2S3          0x00000008U
#endif /* STM32F101xE || STM32F101xG || STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */
#if defined(STM32F102x6) || defined(STM32F102xB) || defined(STM32F103x6)\
 || defined(STM32F103xB) || defined(STM32F103xE) || defined(STM32F103xG)\
 || defined(STM32F105xC) || defined(STM32F107xC)
#define RCC_PERIPHCLK_USB          0x00000010U
#endif /* STM32F102x6 || STM32F102xB || STM32F103x6 || STM32F103xB || STM32F103xE || STM32F103xG || STM32F105xC || STM32F107xC */

/**
  * @}
  */

/** @defgroup RCCEx_ADC_Prescaler ADC Prescaler
  * @{
  */
#define RCC_ADCPCLK2_DIV2              RCC_CFGR_ADCPRE_DIV2
#define RCC_ADCPCLK2_DIV4              RCC_CFGR_ADCPRE_DIV4
#define RCC_ADCPCLK2_DIV6              RCC_CFGR_ADCPRE_DIV6
#define RCC_ADCPCLK2_DIV8              RCC_CFGR_ADCPRE_DIV8

/**
  * @}
  */

//... 其他导出常量 ...

/**
  * @}
  */
```

**描述:**

*   **预定义值:**  定义了用于配置外设时钟的各种常量。  例如, `RCC_PERIPHCLK_RTC` 是 RTC 时钟的选择位，`RCC_ADCPCLK2_DIV2` 是 ADC 时钟分频系数的宏。
*   **设备特定:**  `#if defined(...)` 用于根据不同的 STM32 器件型号定义不同的常量。例如，`RCC_PERIPHCLK_I2S2` 和 `RCC_PERIPHCLK_USB` 只在支持 I2S 和 USB 外设的器件上才定义。
*   **分级分组:** 使用 `@defgroup` 将常量组织成逻辑组，例如外设时钟选择、ADC 预分频器等。

**用途:**

这些常量用作 `RCC_PeriphCLKInitTypeDef` 结构体的成员的取值，用户通过设置这些值来配置不同的外设时钟。

**演示:**

参考上面的 `configure_clocks` 函数。 `periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_ADC;` 和 `periph_clk_init.AdcClockSelection = RCC_ADCPCLK2_DIV6;`  使用了这些常量。

**6. 宏定义（时钟使能/禁用和复位控制）**

这部分代码定义了大量的宏，用于使能/禁用外设时钟，并控制外设的复位。由于代码量很大，这里只展示一部分作为示例：

```c
/** @defgroup RCCEx_Peripheral_Clock_Enable_Disable Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */

#if defined(STM32F101xE) || defined(STM32F103xE) || defined(STM32F101xG)\
 || defined(STM32F103xG) || defined(STM32F105xC) || defined  (STM32F107xC)\
 || defined  (STM32F100xE)
#define __HAL_RCC_DMA2_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg; \
                                        SET_BIT(RCC->AHBENR, RCC_AHBENR_DMA2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHBENR, RCC_AHBENR_DMA2EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_DMA2_CLK_DISABLE()        (RCC->AHBENR &= ~(RCC_AHBENR_DMA2EN))
#endif /* STM32F101xE || STM32F103xE || STM32F101xG || STM32F103xG || STM32F105xC || STM32F107xC || STM32F100xE */
```

**描述：**

*   **时钟使能/禁用宏:** `__HAL_RCC_DMA2_CLK_ENABLE()` 和 `__HAL_RCC_DMA2_CLK_DISABLE()` 用于使能和禁用 DMA2 的时钟。  使能时钟允许对相应外设的寄存器进行读写操作。
*   **`__IO` 关键字:** `__IO` 表示该变量是易变的 (volatile)，编译器不会对该变量的访问进行优化，每次都从内存中读取。
*   **`SET_BIT` 和 `READ_BIT` 宏:**  `SET_BIT` 用于设置寄存器中的特定位，`READ_BIT` 用于读取寄存器中特定位的值。这些宏简化了寄存器操作。
*   **延迟:**  `tmpreg = READ_BIT(RCC->AHBENR, RCC_AHBENR_DMA2EN); UNUSED(tmpreg);`  这部分代码用于在使能时钟后引入一个短时间的延迟。  这有助于确保时钟稳定后再访问外设。  `UNUSED(tmpreg)` 是为了避免编译器警告 "变量未使用"。
*   **复位控制宏:**  还有类似的宏定义用于控制外设的复位，例如 `__HAL_RCC_ADC1_FORCE_RESET()` 和 `__HAL_RCC_ADC1_RELEASE_RESET()`。

**用途:**

在初始化外设之前，必须先使能其时钟。  在某些情况下，可能需要在外设发生错误时复位它。 这些宏提供了方便的方式来完成这些操作。

**演示:**

```c
#include "stm32f1xx_hal.h"

void initialize_dma(void) {
  // 1. 使能 DMA2 时钟
  __HAL_RCC_DMA2_CLK_ENABLE();

  // 2. 配置 DMA 控制器... (这里省略 DMA 配置的具体代码)
  // DMA_InitTypeDef dma_init;
  // dma_init.Channel = DMA_CHANNEL_4;
  // ...
  // HAL_DMA_Init(&hdma_spi2_rx, &dma_init);

  printf("DMA2 初始化完成!\n"); // DMA2 Initialized
}

void reset_adc(void) {
    __HAL_RCC_ADC1_FORCE_RESET();
    __HAL_RCC_ADC1_RELEASE_RESET();
    printf("ADC1 已复位\n");
}

int main() {
  HAL_Init();

  initialize_dma();
  reset_adc();

  while (1) {}
}
```

**解释:**

1.  `initialize_dma` 函数首先使用 `__HAL_RCC_DMA2_CLK_ENABLE()` 宏使能 DMA2 的时钟。
2.  然后，配置 DMA 控制器的参数 (省略了具体代码)。
3.  `reset_adc()` 函数演示了如何使用`__HAL_RCC_ADC1_FORCE_RESET()` 和 `__HAL_RCC_ADC1_RELEASE_RESET()` 宏复位 ADC1 外设.

**7. 函数声明**

```c
/* Exported functions --------------------------------------------------------*/
/** @addtogroup RCCEx_Exported_Functions
  * @{
  */

/** @addtogroup RCCEx_Exported_Functions_Group1
  * @{
  */

HAL_StatusTypeDef HAL_RCCEx_PeriphCLKConfig(RCC_PeriphCLKInitTypeDef  *PeriphClkInit);
void              HAL_RCCEx_GetPeriphCLKConfig(RCC_PeriphCLKInitTypeDef  *PeriphClkInit);
uint32_t          HAL_RCCEx_GetPeriphCLKFreq(uint32_t PeriphClk);

/**
  * @}
  */

#if defined(STM32F105xC) || defined(STM32F107xC)
/** @addtogroup RCCEx_Exported_Functions_Group2
  * @{
  */
HAL_StatusTypeDef HAL_RCCEx_EnablePLLI2S(RCC_PLLI2SInitTypeDef  *PLLI2SInit);
HAL_StatusTypeDef HAL_RCCEx_DisablePLLI2S(void);

/**
  * @}
  */

/** @addtogroup RCCEx_Exported_Functions_Group3
  * @{
  */
HAL_StatusTypeDef HAL_RCCEx_EnablePLL2(RCC_PLL2InitTypeDef  *PLL2Init);
HAL_StatusTypeDef HAL_RCCEx_DisablePLL2(void);

/**
  * @}
  */
#endif /* STM32F105xC || STM32F107xC */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F1xx_HAL_RCC_EX_H */
```

**描述:**

*   **函数声明:**  声明了 HAL 库中用于配置时钟的各种函数。  这些函数提供了 API 来配置外设时钟， 使能/禁用 PLLI2S 和 PLL2 (如果器件支持)。
*   **`HAL_StatusTypeDef` 返回类型:** 大多数 HAL 函数返回 `HAL_StatusTypeDef` 类型的值，用于指示函数执行的状态 (`HAL_OK` 表示成功，其他值表示错误)。
*   **条件编译:** `#if defined(...)` 用于根据目标 STM32 型号选择性地包含函数声明。 例如，`HAL_RCCEx_EnablePLLI2S` 和 `HAL_RCCEx_EnablePLL2` 函数只在支持 PLLI2S 和 PLL2 的 STM32F105xC 和 STM32F107xC 型号上才声明。

**用途:**

这些函数是用户配置 STM32 时钟系统的主要入口点。

**演示:**

在上面的时钟配置演示中已经使用了 `HAL_RCCEx_PeriphCLKConfig`。  更多函数的使用方法类似， 都是将配置结构体作为参数传递给 HAL 函数。

**总结:**

`stm32f1xx_hal_rcc_ex.h` 文件是 STM32 HAL 库中 RCC 扩展模块的头文件。 它定义了用于配置时钟的各种结构体、宏和函数。  理解这些定义对于正确配置 STM32 的时钟系统至关重要， 这直接影响到外设的正常工作。
希望这个更详细的解释对你有所帮助！ 如果有任何其他问题，请随时提出。
