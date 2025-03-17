Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\system_stm32f1xx.c`

好的，重新开始。这次我会专注于提供更智能的代码改进建议，并用中文进行更详细的解释，同时提供易于理解的示例。

**目标： 改进 `system_stm32f1xx.c` 文件的代码，提供更灵活和可维护的时钟配置方法。**

**1.  更灵活的时钟配置函数 (More Flexible Clock Configuration Function):**

    ```c
    #include "stm32f1xx.h"

    // 定义时钟配置结构体  (Define clock configuration structure)
    typedef struct {
        uint32_t SYSCLKSource;    // 系统时钟源 (System clock source)
        uint32_t PLLMultiplier;   // PLL 倍频因子 (PLL multiplication factor)
        uint32_t AHB_Prescaler;     // AHB 预分频器 (AHB prescaler)
        uint32_t APB1_Prescaler;    // APB1 预分频器 (APB1 prescaler)
        uint32_t APB2_Prescaler;    // APB2 预分频器 (APB2 prescaler)
        uint32_t FlashLatency;      // Flash 访问延迟 (Flash access latency)
    } ClockConfig_TypeDef;

    // 函数： 配置系统时钟 (Function: Configure system clock)
    void SystemClock_Config(ClockConfig_TypeDef *config) {
        // 1. 使能时钟源 (Enable clock source)
        if (config->SYSCLKSource == RCC_CFGR_SW_HSE) {
            RCC->CR |= RCC_CR_HSEON;
            while (!(RCC->CR & RCC_CR_HSERDY)); // 等待 HSE 稳定 (Wait for HSE to stabilize)
        } else if (config->SYSCLKSource == RCC_CFGR_SW_PLL) {
            // 使能 HSE 或 HSI (Enable HSE or HSI - based on PLL configuration)
            // 这里需要根据实际情况进行配置 (Configuration depends on the actual situation)
        } else {
            RCC->CR |= RCC_CR_HSION;
            while (!(RCC->CR & RCC_CR_HSIRDY)); // 等待 HSI 稳定 (Wait for HSI to stabilize)
        }

        // 2. 配置 PLL (Configure PLL)
        if (config->SYSCLKSource == RCC_CFGR_SW_PLL) {
            RCC->CFGR &= ~RCC_CFGR_PLLMULL;
            RCC->CFGR |= config->PLLMultiplier;
            RCC->CR |= RCC_CR_PLLON;
            while (!(RCC->CR & RCC_CR_PLLRDY)); // 等待 PLL 稳定 (Wait for PLL to stabilize)
        }

        // 3. 配置 AHB, APB 预分频器 (Configure AHB, APB prescalers)
        RCC->CFGR &= ~RCC_CFGR_HPRE;
        RCC->CFGR |= config->AHB_Prescaler;

        RCC->CFGR &= ~RCC_CFGR_PPRE1;
        RCC->CFGR |= config->APB1_Prescaler;

        RCC->CFGR &= ~RCC_CFGR_PPRE2;
        RCC->CFGR |= config->APB2_Prescaler;

        // 4. 配置 Flash 访问延迟 (Configure Flash access latency)
        FLASH->ACR &= ~FLASH_ACR_LATENCY;
        FLASH->ACR |= config->FlashLatency;

        // 5. 选择系统时钟源 (Select system clock source)
        RCC->CFGR &= ~RCC_CFGR_SW;
        RCC->CFGR |= config->SYSCLKSource;
        while (((RCC->CFGR & RCC_CFGR_SWS) >> 2) != (config->SYSCLKSource >> 2)); // 等待切换完成 (Wait for the switch to complete)
    }

    // 示例用法 (Example usage)
    void SystemInit(void) {
        ClockConfig_TypeDef myClockConfig;

        // 配置 HSE 作为系统时钟，PLL 倍频为 9， AHB/APB 无分频，Flash 延迟为 2 (Configure HSE as system clock, PLL multiplication factor is 9, AHB/APB no division, Flash delay is 2)
        myClockConfig.SYSCLKSource = RCC_CFGR_SW_HSE;
        myClockConfig.PLLMultiplier = RCC_CFGR_PLLMULL9;
        myClockConfig.AHB_Prescaler = RCC_CFGR_HPRE_DIV1;
        myClockConfig.APB1_Prescaler = RCC_CFGR_PPRE1_DIV1;
        myClockConfig.APB2_Prescaler = RCC_CFGR_PPRE2_DIV1;
        myClockConfig.FlashLatency = FLASH_ACR_LATENCY_2;

        SystemClock_Config(&myClockConfig);

        SystemCoreClockUpdate(); // 更新 SystemCoreClock (Update SystemCoreClock)

    #ifdef USER_VECT_TAB_ADDRESS
      SCB->VTOR = VECT_TAB_BASE_ADDRESS | VECT_TAB_OFFSET;
    #endif
    }

    ```

    **中文描述:**

    *   **更灵活的配置:**  不再使用硬编码的值，而是通过一个结构体 `ClockConfig_TypeDef` 传递时钟配置参数。 这使得在代码中修改时钟设置更加方便和可读。
    *   **错误处理:**  添加了等待时钟源稳定的循环。 确保在继续之前，HSE、HSI 或 PLL 已经准备好。
    *   **可维护性:**  将时钟配置逻辑封装在一个单独的函数 `SystemClock_Config` 中。 这使 `SystemInit` 函数更简洁，并允许在程序的其他地方重复使用时钟配置逻辑。

    **示例:**  在 `SystemInit` 函数中，创建 `ClockConfig_TypeDef` 的实例，并设置所需的时钟参数。 然后，将此配置传递给 `SystemClock_Config` 函数以应用更改。

**2.  条件编译优化 (Conditional Compilation Optimization):**

    ```c
    #include "stm32f1xx.h"

    #ifndef HSE_VALUE
        #define HSE_VALUE    8000000U
    #endif

    #ifndef HSI_VALUE
        #define HSI_VALUE    8000000U
    #endif

    // 其他代码...

    void SystemCoreClockUpdate (void) {
        uint32_t tmp = 0U, pllmull = 0U, pllsource = 0U;

        // ... (原代码) ...

        #if defined(STM32F105xC) || defined(STM32F107xC)
            // ... (特定于 STM32F105xC/107xC 的代码) ...
        #elif defined(STM32F100xB) || defined(STM32F100xE)
            // ... (特定于 STM32F100xB/100xE 的代码) ...
        #else
            // 适用于其他 STM32F1xx 器件的默认代码 (Default code for other STM32F1xx devices)
            pllmull = ( pllmull >> 18U) + 2U;

            if (pllsource == 0x00U) {
                SystemCoreClock = (HSI_VALUE >> 1U) * pllmull;
            } else {
                if ((RCC->CFGR & RCC_CFGR_PLLXTPRE) != (uint32_t)RESET) {
                    SystemCoreClock = (HSE_VALUE >> 1U) * pllmull;
                } else {
                    SystemCoreClock = HSE_VALUE * pllmull;
                }
            }
        #endif

        // ... (原代码) ...
    }
    ```

    **中文描述:**

    *   **更清晰的条件编译:** 使用 `#elif` 指令提供更清晰、更易于维护的条件编译结构。 这使得可以轻松添加或修改针对特定 STM32F1xx 器件的代码。
    *   **默认情况:** 添加了一个 `#else` 分支，用于处理代码中没有明确定义的 STM32F1xx 器件。 这有助于防止意外行为并确保代码更具通用性。
    *   **`HSE_VALUE` 和 `HSI_VALUE` 的默认值:** 使用 `#ifndef` 确保 `HSE_VALUE` 和 `HSI_VALUE` 始终有定义，即使它们没有在其他地方定义。

**3.  更精确的 `SystemCoreClockUpdate` (More Accurate `SystemCoreClockUpdate`):**

    ```c
    void SystemCoreClockUpdate(void) {
      uint32_t tmp = 0U, pllmull = 0U, pllsource = 0U, prediv1factor = 1U; // 初始化 prediv1factor

      tmp = RCC->CFGR & RCC_CFGR_SWS;

      switch (tmp) {
        case 0x00U:  /* HSI used as system clock */
          SystemCoreClock = HSI_VALUE;
          break;
        case 0x04U:  /* HSE used as system clock */
          SystemCoreClock = HSE_VALUE;
          break;
        case 0x08U:  /* PLL used as system clock */
          pllmull = RCC->CFGR & RCC_CFGR_PLLMULL;
          pllsource = RCC->CFGR & RCC_CFGR_PLLSRC;

    #if defined(STM32F100xB) || defined(STM32F100xE)
          prediv1factor = (RCC->CFGR2 & RCC_CFGR2_PREDIV1) + 1U;
    #endif

          pllmull = (pllmull >> 18U) + 2U;

          if (pllsource == 0x00U) {
            SystemCoreClock = (HSI_VALUE >> 1U) * pllmull;
          } else {
    #if defined(STM32F100xB) || defined(STM32F100xE)
            SystemCoreClock = (HSE_VALUE / prediv1factor) * pllmull;
    #else
            if ((RCC->CFGR & RCC_CFGR_PLLXTPRE) != (uint32_t)RESET) {
              SystemCoreClock = (HSE_VALUE >> 1U) * pllmull;
            } else {
              SystemCoreClock = HSE_VALUE * pllmull;
            }
    #endif
          }
          break;
        default:
          SystemCoreClock = HSI_VALUE;
          break;
      }

      const uint8_t ahb_prescaler_table[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 6, 7, 8, 9};
      uint8_t ahb_prescaler_index = (RCC->CFGR & RCC_CFGR_HPRE) >> 4;
      uint32_t ahb_prescaler = (ahb_prescaler_index < 8) ? 1 : (1 << ahb_prescaler_table[ahb_prescaler_index]);

      SystemCoreClock /= ahb_prescaler;
    }
    ```

    **中文描述:**

    *   **初始化 `prediv1factor`:** 在函数开始时，`prediv1factor` 被初始化为 1。 这对于没有定义 `RCC_CFGR2_PREDIV1` 的设备（例如，非 STM32F100xB/E）至关重要，以防止使用未初始化的值。
    *   **更正的 AHB 预分频器计算:** 使用查找表方法改进了 AHB 预分频器的计算。 添加了对表访问的边界检查，确保索引在有效范围内，并使用左移位操作以获得正确的预分频器值。 这样可以确保 HCLK 计算的准确性。
    *   **代码简洁性和可读性:** 精简了代码，使其更易于阅读和理解，同时保持了所有必要的功能。

**总结：**

这些改进主要集中在提高代码的灵活性、可读性、可维护性和准确性。 通过使用结构体传递时钟配置参数，使用条件编译来处理特定于器件的差异，并确保 `SystemCoreClockUpdate` 函数准确计算系统时钟，可以创建一个更可靠和更易于使用的系统时钟配置框架。 示例用法和中文描述使这些更改更易于理解和应用。
