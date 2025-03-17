Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_gpio_ex.c`

OK, 了解，重新开始。  我会分解代码，提供更详细的解释，并包含一个简单的示例。

```c
/**
  ******************************************************************************
  * @file    stm32f1xx_hal_gpio_ex.c
  * @author  MCD Application Team
  * @brief   GPIO Extension HAL module driver.
  *         This file provides firmware functions to manage the following
  *          functionalities of the General Purpose Input/Output (GPIO) extension peripheral.
  *           + Extended features functions
  *
  @verbatim
  ==============================================================================
                    ##### GPIO Peripheral extension features #####
  ==============================================================================
  [..] GPIO module on STM32F1 family, manage also the AFIO register:
       (+) Possibility to use the EVENTOUT Cortex feature

                     ##### How to use this driver #####
  ==============================================================================
  [..] This driver provides functions to use EVENTOUT Cortex feature
    (#) Configure EVENTOUT Cortex feature using the function HAL_GPIOEx_ConfigEventout()
    (#) Activate EVENTOUT Cortex feature using the HAL_GPIOEx_EnableEventout()
    (#) Deactivate EVENTOUT Cortex feature using the HAL_GPIOEx_DisableEventout()

  @endverbatim
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

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @defgroup GPIOEx GPIOEx
  * @brief GPIO HAL module driver
  * @{
  */

#ifdef HAL_GPIO_MODULE_ENABLED

/** @defgroup GPIOEx_Exported_Functions GPIOEx Exported Functions
  * @{
  */

/** @defgroup GPIOEx_Exported_Functions_Group1 Extended features functions
 *  @brief    Extended features functions
 *
@verbatim
  ==============================================================================
                 ##### Extended features functions #####
  ==============================================================================
    [..]  This section provides functions allowing to:
    (+) Configure EVENTOUT Cortex feature using the function HAL_GPIOEx_ConfigEventout()
    (+) Activate EVENTOUT Cortex feature using the HAL_GPIOEx_EnableEventout()
    (+) Deactivate EVENTOUT Cortex feature using the HAL_GPIOEx_DisableEventout()

@endverbatim
  * @{
  */

/**
  * @brief  Configures the port and pin on which the EVENTOUT Cortex signal will be connected.
  * @param  GPIO_PortSource Select the port used to output the Cortex EVENTOUT signal.
  *   This parameter can be a value of @ref GPIOEx_EVENTOUT_PORT.
  * @param  GPIO_PinSource Select the pin used to output the Cortex EVENTOUT signal.
  *   This parameter can be a value of @ref GPIOEx_EVENTOUT_PIN.
  * @retval None
  */
void HAL_GPIOEx_ConfigEventout(uint32_t GPIO_PortSource, uint32_t GPIO_PinSource)
{
  /* Verify the parameters */
  assert_param(IS_AFIO_EVENTOUT_PORT(GPIO_PortSource));
  assert_param(IS_AFIO_EVENTOUT_PIN(GPIO_PinSource));

  /* Apply the new configuration */
  MODIFY_REG(AFIO->EVCR, (AFIO_EVCR_PORT) | (AFIO_EVCR_PIN), (GPIO_PortSource) | (GPIO_PinSource));
}

/**
  * @brief  Enables the Event Output.
  * @retval None
  */
void HAL_GPIOEx_EnableEventout(void)
{
  SET_BIT(AFIO->EVCR, AFIO_EVCR_EVOE);
}

/**
  * @brief  Disables the Event Output.
  * @retval None
  */
void HAL_GPIOEx_DisableEventout(void)
{
  CLEAR_BIT(AFIO->EVCR, AFIO_EVCR_EVOE);
}

/**
  * @}
  */

/**
  * @}
  */

#endif /* HAL_GPIO_MODULE_ENABLED */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
```

**代码分解和解释:**

1.  **`#include "stm32f1xx_hal.h"`**: 包含 STM32F1xx HAL 库的头文件。 这是使用任何 HAL 库函数的必要步骤。

    *   `#include "stm32f1xx_hal.h"`: 包含STM32F1 HAL库的头文件，必须包含此文件才能使用HAL库提供的函数。

2.  **`HAL_GPIOEx_ConfigEventout(uint32_t GPIO_PortSource, uint32_t GPIO_PinSource)`**

    *   **功能:** 配置 Cortex-M3 的 EVENTOUT 信号的输出端口和引脚。EVENTOUT  允许将特定的内部事件（例如中断、定时器溢出）路由到一个 GPIO 引脚，用于外部触发或其他用途。
    *   **参数:**
        *   `GPIO_PortSource`:  指定要使用的 GPIO 端口。 这是 `AFIO_EVCR_PORT` 中定义的一个宏，例如 `AFIO_EVCR_PORTA`。
        *   `GPIO_PinSource`:  指定要使用的 GPIO 引脚。 这是 `AFIO_EVCR_PIN` 中定义的一个宏，例如 `AFIO_EVCR_PIN5`。
    *   **实现:**
        *   `assert_param()`:  这是一个断言宏，用于检查输入参数的有效性。如果参数无效，程序将停止执行。`IS_AFIO_EVENTOUT_PORT` 和 `IS_AFIO_EVENTOUT_PIN` 是检查端口和引脚值是否有效的宏。
        *   `MODIFY_REG()`:  这是一个宏，用于修改寄存器的特定位。  它读取 `AFIO->EVCR` 寄存器的值，清除 `AFIO_EVCR_PORT` 和 `AFIO_EVCR_PIN` 掩码对应的位，然后设置 `GPIO_PortSource` 和 `GPIO_PinSource` 对应的值。

    ```c
    void HAL_GPIOEx_ConfigEventout(uint32_t GPIO_PortSource, uint32_t GPIO_PinSource)
    {
      /* Verify the parameters */
      assert_param(IS_AFIO_EVENTOUT_PORT(GPIO_PortSource)); // 确保端口参数有效
      assert_param(IS_AFIO_EVENTOUT_PIN(GPIO_PinSource));   // 确保引脚参数有效

      /* Apply the new configuration */
      MODIFY_REG(AFIO->EVCR, (AFIO_EVCR_PORT) | (AFIO_EVCR_PIN), (GPIO_PortSource) | (GPIO_PinSource));
      // 修改AFIO_EVCR寄存器，配置EVENTOUT的端口和引脚
    }
    ```
    *这段代码用于配置Cortex-M3内核的EVENTOUT信号所连接的端口和引脚。首先，它使用assert_param宏来验证传入的GPIO_PortSource和GPIO_PinSource参数的有效性，确保它们是有效的AFIO EVENTOUT端口和引脚。然后，它使用MODIFY_REG宏来修改AFIO->EVCR寄存器，以应用新的配置。*

3.  **`HAL_GPIOEx_EnableEventout(void)`**

    *   **功能:** 启用 EVENTOUT 信号。
    *   **实现:**  设置 `AFIO->EVCR` 寄存器中的 `AFIO_EVCR_EVOE` 位。  `EVOE` 代表 "Event Output Enable"。

    ```c
    void HAL_GPIOEx_EnableEventout(void)
    {
      SET_BIT(AFIO->EVCR, AFIO_EVCR_EVOE); // 使能EVENTOUT功能
    }
    ```
    *该函数用于启用EVENTOUT功能。它通过设置AFIO->EVCR寄存器中的AFIO_EVCR_EVOE位来实现。当该位被设置时，EVENTOUT信号将被激活。*

4.  **`HAL_GPIOEx_DisableEventout(void)`**

    *   **功能:** 禁用 EVENTOUT 信号。
    *   **实现:** 清除 `AFIO->EVCR` 寄存器中的 `AFIO_EVCR_EVOE` 位。

    ```c
    void HAL_GPIOEx_DisableEventout(void)
    {
      CLEAR_BIT(AFIO->EVCR, AFIO_EVCR_EVOE); // 禁用EVENTOUT功能
    }
    ```
    *该函数用于禁用EVENTOUT功能。它通过清除AFIO->EVCR寄存器中的AFIO_EVCR_EVOE位来实现。当该位被清除时，EVENTOUT信号将被禁用。*

**简单示例:**

```c
#include "stm32f1xx_hal.h"

void Error_Handler(void); // 错误处理函数，如果初始化失败会调用

int main(void) {
  HAL_Init(); // 初始化 HAL 库

  // 假设你已经配置了时钟系统等等

  // 1. 使能 GPIO 端口的时钟 (例如，GPIOA)
  __HAL_RCC_GPIOA_CLK_ENABLE();

  // 2. 配置 GPIO 引脚为复用推挽输出
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_8; // 例如，GPIOA Pin 8
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // 3. 配置 EVENTOUT  (使用 PA8)
  HAL_GPIOEx_ConfigEventout(AFIO_EVENTOUT_PORT_GPIOA, AFIO_EVENTOUT_PIN_8);

  // 4. 使能 EVENTOUT
  HAL_GPIOEx_EnableEventout();

  // 现在，Cortex-M3 内核的 EVENTOUT 信号将输出到 PA8 引脚。
  // 你需要配置内核来产生 EVENTOUT 信号。  这通常涉及设置一个
  // 定时器或其他外设来触发一个事件。

  while (1) {
    // 你的主循环
  }
}

void Error_Handler(void) {
  // 错误处理逻辑
  while(1);
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
```

**示例解释:**

1.  **`HAL_Init()`**: 初始化 HAL 库。
2.  **`__HAL_RCC_GPIOA_CLK_ENABLE()`**: 使能 GPIOA 端口的时钟。  在使用 GPIO 之前，必须先使能其时钟。
3.  **`GPIO_InitTypeDef`**:  配置 GPIOA Pin 8 为复用推挽输出。  因为 EVENTOUT  使用复用功能，所以需要将引脚配置为复用模式。`GPIO_MODE_AF_PP` 表示复用推挽输出。
4.  **`HAL_GPIOEx_ConfigEventout()`**:  将 EVENTOUT 信号配置到 GPIOA Pin 8。  `AFIO_EVENTOUT_PORT_GPIOA` 和 `AFIO_EVENTOUT_PIN_8` 是定义好的宏。
5.  **`HAL_GPIOEx_EnableEventout()`**: 启用 EVENTOUT。
6.  **关键：**  这个例子仅仅是配置了 GPIO。  要真正让 PA8 输出信号，你需要在 Cortex-M3 内核中配置一个事件源。  例如，你可以配置一个定时器中断，然后将该中断配置为触发 EVENTOUT 信号。  这部分配置通常涉及 NVIC (Nested Vectored Interrupt Controller) 和 SYSTICK 定时器。  这部分代码会因你想要触发的事件而异。

**注意事项:**

*   **`AFIO->EVCR` 寄存器:**  这个寄存器位于 AFIO (Alternate Function I/O) 模块中，用于配置和控制 EVENTOUT 功能。
*   **`assert_param()`**:  在调试时，`assert_param()` 宏可以帮助你发现参数错误。  在发布版本中，可以禁用这些断言以提高性能。
*   **时钟配置:** 确保系统时钟配置正确，并且已经使能了相关的 GPIO 端口的时钟。
*   **事件源配置:** EVENTOUT 需要一个事件源才能工作。 你必须配置 NVIC、定时器或其他外设来产生一个事件，该事件将触发 EVENTOUT 信号。 具体配置取决于你的应用需求。

**中文总结:**

这段代码提供了一个 STM32F103 系列微控制器的 GPIO 扩展 HAL 驱动程序，用于配置和使用 Cortex-M3 内核的 EVENTOUT 功能。 EVENTOUT 允许将内部事件（例如中断或定时器溢出）路由到 GPIO 引脚。

代码包含以下几个关键函数：

*   **`HAL_GPIOEx_ConfigEventout()`**:  配置 EVENTOUT 信号输出的 GPIO 端口和引脚。
*   **`HAL_GPIOEx_EnableEventout()`**: 启用 EVENTOUT 功能。
*   **`HAL_GPIOEx_DisableEventout()`**: 禁用 EVENTOUT 功能。

示例代码展示了如何使用这些函数来配置 GPIO 引脚并启用 EVENTOUT。  **请务必注意，仅仅配置 GPIO 引脚是不够的。你还需要配置 Cortex-M3 内核来生成 EVENTOUT 信号。** 这通常涉及配置 NVIC 和定时器等外设。 成功的 EVENTOUT 配置需要对 STM32F103 的硬件架构和 HAL 库有深入的了解。

希望这个更详细的解释和示例能帮助你理解这段代码。
