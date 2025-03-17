Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\system_stm32f1xx.c`

好的，我们重新开始。这次我会更详细地解释代码，并提供更清晰的示例。

```c
/**
  ******************************************************************************
  * @file    system_stm32f1xx.c
  * @author  MCD Application Team
  * @brief   CMSIS Cortex-M3 Device Peripheral Access Layer System Source File.
  *
  * 1.  This file provides two functions and one global variable to be called from
  *     user application:
  *      - SystemInit(): Setups the system clock (System clock source, PLL Multiplier
  *                      factors, AHB/APBx prescalers and Flash settings).
  *                      This function is called at startup just after reset and
  *                      before branch to main program. This call is made inside
  *                      the "startup_stm32f1xx_xx.s" file.
  *
  *      - SystemCoreClock variable: Contains the core clock (HCLK), it can be used
  *                                  by the user application to setup the SysTick
  *                                  timer or configure other parameters.
  *
  *      - SystemCoreClockUpdate(): Updates the variable SystemCoreClock and must
  *                                 be called whenever the core clock is changed
  *                                 during program execution.
  *
  * 2. After each device reset the HSI (8 MHz) is used as system clock source.
  *    Then SystemInit() function is called, in "startup_stm32f1xx_xx.s" file, to
  *    configure the system clock before to branch to main program.
  *
  * 4. The default value of HSE crystal is set to 8 MHz (or 25 MHz, depending on
  *    the product used), refer to "HSE_VALUE".
  *    When HSE is used as system clock source, directly or through PLL, and you
  *    are using different crystal you have to adapt the HSE value to your own
  *    configuration.
  *
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/** @addtogroup CMSIS
  * @{
  */

/** @addtogroup stm32f1xx_system
  * @{
  */

/** @addtogroup STM32F1xx_System_Private_Includes
  * @{
  */

#include "stm32f1xx.h"

/**
  * @}
  */

/** @addtogroup STM32F1xx_System_Private_TypesDefinitions
  * @{
  */

/**
  * @}
  */

/** @addtogroup STM32F1xx_System_Private_Defines
  * @{
  */

#if !defined  (HSE_VALUE)
  #define HSE_VALUE               8000000U /*!< Default value of the External oscillator in Hz.
                                                This value can be provided and adapted by the user application. */
#endif /* HSE_VALUE */

#if !defined  (HSI_VALUE)
  #define HSI_VALUE               8000000U /*!< Default value of the Internal oscillator in Hz.
                                                This value can be provided and adapted by the user application. */
#endif /* HSI_VALUE */

/*!< Uncomment the following line if you need to use external SRAM  */
#if defined(STM32F100xE) || defined(STM32F101xE) || defined(STM32F101xG) || defined(STM32F103xE) || defined(STM32F103xG)
/* #define DATA_IN_ExtSRAM */
#endif /* STM32F100xE || STM32F101xE || STM32F101xG || STM32F103xE || STM32F103xG */

/* Note: Following vector table addresses must be defined in line with linker
         configuration. */
/*!< Uncomment the following line if you need to relocate the vector table
     anywhere in Flash or Sram, else the vector table is kept at the automatic
     remap of boot address selected */
/* #define USER_VECT_TAB_ADDRESS */

#if defined(USER_VECT_TAB_ADDRESS)
/*!< Uncomment the following line if you need to relocate your vector Table
     in Sram else user remap will be done in Flash. */
/* #define VECT_TAB_SRAM */
#if defined(VECT_TAB_SRAM)
#define VECT_TAB_BASE_ADDRESS   SRAM_BASE       /*!< Vector Table base address field.
                                                     This value must be a multiple of 0x200. */
#define VECT_TAB_OFFSET         0x00000000U     /*!< Vector Table base offset field.
                                                     This value must be a multiple of 0x200. */
#else
#define VECT_TAB_BASE_ADDRESS   FLASH_BASE      /*!< Vector Table base address field.
                                                     This value must be a multiple of 0x200. */
#define VECT_TAB_OFFSET         0x00000000U     /*!< Vector Table base offset field.
                                                     This value must be a multiple of 0x200. */
#endif /* VECT_TAB_SRAM */
#endif /* USER_VECT_TAB_ADDRESS */

/******************************************************************************/

/**
  * @}
  */

/** @addtogroup STM32F1xx_System_Private_Macros
  * @{
  */

/**
  * @}
  */

/** @addtogroup STM32F1xx_System_Private_Variables
  * @{
  */

  /* This variable is updated in three ways:
      1) by calling CMSIS function SystemCoreClockUpdate()
      2) by calling HAL API function HAL_RCC_GetHCLKFreq()
      3) each time HAL_RCC_ClockConfig() is called to configure the system clock frequency
         Note: If you use this function to configure the system clock; then there
               is no need to call the 2 first functions listed above, since SystemCoreClock
               variable is updated automatically.
  */
uint32_t SystemCoreClock = 16000000; // 系统时钟，默认值为 16MHz
const uint8_t AHBPrescTable[16U] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 6, 7, 8, 9}; // AHB 预分频系数表
const uint8_t APBPrescTable[8U] =  {0, 0, 0, 0, 1, 2, 3, 4}; // APB 预分频系数表

/**
  * @}
  */

/** @addtogroup STM32F1xx_System_Private_FunctionPrototypes
  * @{
  */

#if defined(STM32F100xE) || defined(STM32F101xE) || defined(STM32F101xG) || defined(STM32F103xE) || defined(STM32F103xG)
#ifdef DATA_IN_ExtSRAM
  static void SystemInit_ExtMemCtl(void);
#endif /* DATA_IN_ExtSRAM */
#endif /* STM32F100xE || STM32F101xE || STM32F101xG || STM32F103xE || STM32F103xG */

/**
  * @}
  */

/** @addtogroup STM32F1xx_System_Private_Functions
  * @{
  */

/**
  * @brief  Setup the microcontroller system
  *         Initialize the Embedded Flash Interface, the PLL and update the
  *         SystemCoreClock variable.
  * @note   This function should be used only after reset.
  * @param  None
  * @retval None
  */
void SystemInit (void)
{
#if defined(STM32F100xE) || defined(STM32F101xE) || defined(STM32F101xG) || defined(STM32F103xE) || defined(STM32F103xG)
  #ifdef DATA_IN_ExtSRAM
    SystemInit_ExtMemCtl();
  #endif /* DATA_IN_ExtSRAM */
#endif

  /* Configure the Vector Table location -------------------------------------*/
#if defined(USER_VECT_TAB_ADDRESS)
  SCB->VTOR = VECT_TAB_BASE_ADDRESS | VECT_TAB_OFFSET; /* Vector Table Relocation in Internal SRAM. */
#endif /* USER_VECT_TAB_ADDRESS */
}

/**
  * @brief  Update SystemCoreClock variable according to Clock Register Values.
  *         The SystemCoreClock variable contains the core clock (HCLK), it can
  *         be used by the user application to setup the SysTick timer or configure
  *         other parameters.
  *
  * @note   Each time the core clock (HCLK) changes, this function must be called
  *         to update SystemCoreClock variable value. Otherwise, any configuration
  *         based on this variable will be incorrect.
  *
  * @note   - The system frequency computed by this function is not the real
  *           frequency in the chip. It is calculated based on the predefined
  *           constant and the selected clock source:
  *
  *           - If SYSCLK source is HSI, SystemCoreClock will contain the HSI_VALUE(*)
  *
  *           - If SYSCLK source is HSE, SystemCoreClock will contain the HSE_VALUE(**)
  *
  *           - If SYSCLK source is PLL, SystemCoreClock will contain the HSE_VALUE(**)
  *             or HSI_VALUE(*) multiplied by the PLL factors.
  *
  *         (*) HSI_VALUE is a constant defined in stm32f1xx.h file (default value
  *             8 MHz) but the real value may vary depending on the variations
  *             in voltage and temperature.
  *
  *         (**) HSE_VALUE is a constant defined in stm32f1xx.h file (default value
  *              8 MHz or 25 MHz, depending on the product used), user has to ensure
  *              that HSE_VALUE is same as the real frequency of the crystal used.
  *              Otherwise, this function may have wrong result.
  *
  *         - The result of this function could be not correct when using fractional
  *           value for HSE crystal.
  * @param  None
  * @retval None
  */
void SystemCoreClockUpdate (void)
{
  uint32_t tmp = 0U, pllmull = 0U, pllsource = 0U; // 临时变量，PLL 倍频因子，PLL 时钟源

#if defined(STM32F105xC) || defined(STM32F107xC)
  uint32_t prediv1source = 0U, prediv1factor = 0U, prediv2factor = 0U, pll2mull = 0U;
#endif /* STM32F105xC */

#if defined(STM32F100xB) || defined(STM32F100xE)
  uint32_t prediv1factor = 0U;
#endif /* STM32F100xB or STM32F100xE */

  /* Get SYSCLK source -------------------------------------------------------*/
  tmp = RCC->CFGR & RCC_CFGR_SWS; // 读取时钟配置寄存器 CFGR 的 SWS 位，获取系统时钟源

  switch (tmp)
  {
    case 0x00U:  /* HSI used as system clock */ // HSI 作为系统时钟源
      SystemCoreClock = HSI_VALUE; // 系统时钟等于 HSI 频率
      break;
    case 0x04U:  /* HSE used as system clock */ // HSE 作为系统时钟源
      SystemCoreClock = HSE_VALUE; // 系统时钟等于 HSE 频率
      break;
    case 0x08U:  /* PLL used as system clock */ // PLL 作为系统时钟源

      /* Get PLL clock source and multiplication factor ----------------------*/
      pllmull = RCC->CFGR & RCC_CFGR_PLLMULL; // 读取 PLL 倍频因子
      pllsource = RCC->CFGR & RCC_CFGR_PLLSRC; // 读取 PLL 时钟源

#if !defined(STM32F105xC) && !defined(STM32F107xC)
      pllmull = ( pllmull >> 18U) + 2U; // 计算 PLL 倍频因子 (PLLMULL 位的值 + 2)

      if (pllsource == 0x00U)
      {
        /* HSI oscillator clock divided by 2 selected as PLL clock entry */ // HSI/2 作为 PLL 时钟输入
        SystemCoreClock = (HSI_VALUE >> 1U) * pllmull; // 系统时钟 = (HSI / 2) * PLLMULL
      }
      else
      {
 #if defined(STM32F100xB) || defined(STM32F100xE)
       prediv1factor = (RCC->CFGR2 & RCC_CFGR2_PREDIV1) + 1U;
       /* HSE oscillator clock selected as PREDIV1 clock entry */
       SystemCoreClock = (HSE_VALUE / prediv1factor) * pllmull;
 #else
        /* HSE selected as PLL clock entry */ // HSE 直接作为 PLL 时钟输入
        if ((RCC->CFGR & RCC_CFGR_PLLXTPRE) != (uint32_t)RESET)
        {/* HSE oscillator clock divided by 2 */ // 如果 PLLXTPRE 位被设置，HSE 频率除以 2
          SystemCoreClock = (HSE_VALUE >> 1U) * pllmull; // 系统时钟 = (HSE / 2) * PLLMULL
        }
        else
        {
          SystemCoreClock = HSE_VALUE * pllmull; // 系统时钟 = HSE * PLLMULL
        }
 #endif
      }
#else
      pllmull = pllmull >> 18U;

      if (pllmull != 0x0DU)
      {
         pllmull += 2U;
      }
      else
      { /* PLL multiplication factor = PLL input clock * 6.5 */
        pllmull = 13U / 2U;
      }

      if (pllsource == 0x00U)
      {
        /* HSI oscillator clock divided by 2 selected as PLL clock entry */
        SystemCoreClock = (HSI_VALUE >> 1U) * pllmull;
      }
      else
      {/* PREDIV1 selected as PLL clock entry */

        /* Get PREDIV1 clock source and division factor */
        prediv1source = RCC->CFGR2 & RCC_CFGR2_PREDIV1SRC;
        prediv1factor = (RCC->CFGR2 & RCC_CFGR2_PREDIV1) + 1U;

        if (prediv1source == 0U)
        {
          /* HSE oscillator clock selected as PREDIV1 clock entry */
          SystemCoreClock = (HSE_VALUE / prediv1factor) * pllmull;
        }
        else
        {/* PLL2 clock selected as PREDIV1 clock entry */

          /* Get PREDIV2 division factor and PLL2 multiplication factor */
          prediv2factor = ((RCC->CFGR2 & RCC_CFGR2_PREDIV2) >> 4U) + 1U;
          pll2mull = ((RCC->CFGR2 & RCC_CFGR2_PLL2MUL) >> 8U) + 2U;
          SystemCoreClock = (((HSE_VALUE / prediv2factor) * pll2mull) / prediv1factor) * pllmull;
        }
      }
#endif /* STM32F105xC */
      break;

    default:
      SystemCoreClock = HSI_VALUE; // 默认使用 HSI
      break;
  }

  /* Compute HCLK clock frequency ----------------*/
  /* Get HCLK prescaler */
  tmp = AHBPrescTable[((RCC->CFGR & RCC_CFGR_HPRE) >> 4U)]; // 获取 AHB 预分频系数
  /* HCLK clock frequency */
  SystemCoreClock >>= tmp; // 系统时钟 = 系统时钟 / AHB 预分频系数
}

#if defined(STM32F100xE) || defined(STM32F101xE) || defined(STM32F101xG) || defined(STM32F103xE) || defined(STM32F103xG)
/**
  * @brief  Setup the external memory controller. Called in startup_stm32f1xx.s
  *          before jump to __main
  * @param  None
  * @retval None
  */
#ifdef DATA_IN_ExtSRAM
/**
  * @brief  Setup the external memory controller.
  *         Called in startup_stm32f1xx_xx.s/.c before jump to main.
  *         This function configures the external SRAM mounted on STM3210E-EVAL
  *         board (STM32 High density devices). This SRAM will be used as program
  *         data memory (including heap and stack).
  * @param  None
  * @retval None
  */
void SystemInit_ExtMemCtl(void)
{
  __IO uint32_t tmpreg; // 临时变量，使用 volatile 确保编译器不会优化掉读取操作
  /*!< FSMC Bank1 NOR/SRAM3 is used for the STM3210E-EVAL, if another Bank is
    required, then adjust the Register Addresses */

  /* Enable FSMC clock */ // 使能 FSMC 时钟
  RCC->AHBENR = 0x00000114U; // 设置 RCC->AHBENR 寄存器，使能 FSMC 时钟

  /* Delay after an RCC peripheral clock enabling */
  tmpreg = READ_BIT(RCC->AHBENR, RCC_AHBENR_FSMCEN); // 读取 FSMCEN 位，确保时钟使能完成

  /* Enable GPIOD, GPIOE, GPIOF and GPIOG clocks */ // 使能 GPIOD, GPIOE, GPIOF 和 GPIOG 时钟
  RCC->APB2ENR = 0x000001E0U; // 设置 RCC->APB2ENR 寄存器，使能 GPIO 时钟

  /* Delay after an RCC peripheral clock enabling */
  tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_IOPDEN); // 读取 IOPDEN 位，确保时钟使能完成

  (void)(tmpreg); // 消除编译器警告

/* ---------------  SRAM Data lines, NOE and NWE configuration ---------------*/
/*----------------  SRAM Address lines configuration -------------------------*/
/*----------------  NOE and NWE configuration --------------------------------*/
/*----------------  NE3 configuration ----------------------------------------*/
/*----------------  NBL0, NBL1 configuration ---------------------------------*/

  GPIOD->CRL = 0x44BB44BBU; // 配置 GPIOD 的 CRL 寄存器
  GPIOD->CRH = 0xBBBBBBBBU; // 配置 GPIOD 的 CRH 寄存器

  GPIOE->CRL = 0xB44444BBU; // 配置 GPIOE 的 CRL 寄存器
  GPIOE->CRH = 0xBBBBBBBBU; // 配置 GPIOE 的 CRH 寄存器

  GPIOF->CRL = 0x44BBBBBBU; // 配置 GPIOF 的 CRL 寄存器
  GPIOF->CRH = 0xBBBB4444U; // 配置 GPIOF 的 CRH 寄存器

  GPIOG->CRL = 0x44BBBBBBU; // 配置 GPIOG 的 CRL 寄存器
  GPIOG->CRH = 0x444B4B44U; // 配置 GPIOG 的 CRH 寄存器

/*----------------  FSMC Configuration ---------------------------------------*/
/*----------------  Enable FSMC Bank1_SRAM Bank ------------------------------*/

  FSMC_Bank1->BTCR[4U] = 0x00001091U; // 配置 FSMC_Bank1 的 BTCR4 寄存器
  FSMC_Bank1->BTCR[5U] = 0x00110212U; // 配置 FSMC_Bank1 的 BTCR5 寄存器
}
#endif /* DATA_IN_ExtSRAM */
#endif /* STM32F100xE || STM32F101xE || STM32F101xG || STM32F103xE || STM32F103xG */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */
/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
```

**代码关键部分的解释 (中文):**

1.  **`#include "stm32f1xx.h"`**: 包含 STM32F1xx 系列芯片的头文件，定义了所有外设的寄存器地址和位定义。 这相当于 Python 中的 `import` 语句，它允许您访问 STM32 芯片的硬件资源。

2.  **`#define HSE_VALUE 8000000U` 和 `#define HSI_VALUE 8000000U`**: 定义了外部高速晶振 (HSE) 和内部高速振荡器 (HSI) 的频率值。 这些值是配置时钟系统的重要参数。 如果您使用不同的晶振，您需要根据实际情况修改 `HSE_VALUE`。

3.  **`SystemCoreClock` 变量**:  一个全局变量，存储了当前系统时钟 (HCLK) 的频率。  很多外设的配置，比如 SysTick 定时器，都需要使用这个值。

4.  **`AHBPrescTable` 和 `APBPrescTable` 数组**: 存储了 AHB 和 APB 总线预分频系数的查找表。 通过读取时钟配置寄存器中的相应位，并使用这些表，可以计算出 AHB 和 APB 总线的时钟频率。

5.  **`SystemInit()` 函数**:  这个函数在系统启动时被调用，主要完成以下工作：
    *   **`SystemInit_ExtMemCtl()`**: 如果定义了 `DATA_IN_ExtSRAM`，则调用这个函数来初始化外部 SRAM。 这通常用于将外部 SRAM 用作程序的堆和栈空间。
    *   **`SCB->VTOR = VECT_TAB_BASE_ADDRESS | VECT_TAB_OFFSET;`**: 配置向量表的位置。 向量表包含了中断向量的地址，告诉 CPU 在发生中断时跳转到哪里执行中断服务程序。

6.  **`SystemCoreClockUpdate()` 函数**:  这个函数根据时钟配置寄存器的值，更新 `SystemCoreClock` 变量。  每次修改时钟配置后，都应该调用这个函数，以确保 `SystemCoreClock` 的值是正确的。  它的实现过程就是根据不同的时钟源 (HSI, HSE, PLL) 和分频系数，计算出 HCLK 的频率。

7.  **时钟源选择和 PLL 配置**:  `SystemCoreClockUpdate()` 函数的核心部分是根据 `RCC->CFGR` 寄存器中的值，判断当前使用的是哪个时钟源，以及 PLL 的配置。  它会根据这些信息，计算出系统时钟频率。  PLL 的作用是将输入的时钟信号倍频，以获得更高的系统时钟频率。

8.  **`SystemInit_ExtMemCtl()` 函数**:  这个函数用于初始化外部 SRAM 控制器 (FSMC)。  它首先使能 FSMC 和 GPIO 的时钟，然后配置 GPIO 引脚的功能 (作为 FSMC 的数据线、地址线、控制线)，最后配置 FSMC 的控制寄存器，设置 SRAM 的时序参数和访问模式。

**代码如何使用以及简单 Demo:**

这个文件是 STM32 固件库的一部分，通常不需要用户手动修改。  `SystemInit()` 函数由启动文件 (startup\*.s) 调用，在 `main()` 函数之前执行。  `SystemCoreClockUpdate()` 函数可以在任何时候调用，以更新系统时钟频率。

**一个简单的 Demo:**

```c
#include "stm32f1xx.h"

void SysTick_Handler(void) {
  // 每隔一段时间翻转 LED 的状态
  GPIOA->ODR ^= (1 << 5); // 假设 PA5 连接 LED
}

int main(void) {
  SystemInit(); // 初始化系统

  // 使能 GPIOA 时钟
  RCC->APB2ENR |= RCC_APB2ENR_IOPAEN;

  // 配置 PA5 为输出模式
  GPIOA->CRL &= ~(GPIO_CRL_MODE5 | GPIO_CRL_CNF5);
  GPIOA->CRL |= GPIO_CRL_MODE5_1; // 输出模式，最大速度 2MHz

  // 配置 SysTick 定时器，每秒触发一次中断
  SysTick_Config(SystemCoreClock / 1); // 1 秒中断一次

  while (1) {
    // 空循环，等待中断
  }
}
```

**Demo 解释:**

1.  **`SystemInit()`**: 在 `main()` 函数的开始处调用，初始化时钟和其他系统资源。

2.  **GPIO 配置**:  代码配置 GPIOA 的第 5 个引脚 (PA5) 为输出模式，并连接一个 LED。

3.  **SysTick 定时器**:  `SysTick_Config(SystemCoreClock / 1);` 这行代码配置 SysTick 定时器，使其每秒产生一次中断。  `SystemCoreClock` 变量提供了系统时钟频率，用于计算 SysTick 的重载值。

4.  **中断服务程序**:  `SysTick_Handler()` 函数是 SysTick 定时器的中断服务程序。 每次 SysTick 计数器减到 0 时，这个函数会被调用。  在这个例子中，它翻转 PA5 的状态，从而使 LED 闪烁。

**总结:**

`system_stm32f1xx.c` 文件是 STM32 时钟系统配置的核心。  理解这个文件的作用，可以更好地掌握 STM32 的时钟配置，并能更有效地使用各种外设。希望这次的解释更加详细和清晰。