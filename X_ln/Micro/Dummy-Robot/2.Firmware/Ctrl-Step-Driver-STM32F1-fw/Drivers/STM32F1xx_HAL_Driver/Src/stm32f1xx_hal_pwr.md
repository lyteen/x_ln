Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_pwr.c`

好的，重新开始，针对 STM32F1xx HAL 库中 `stm32f1xx_hal_pwr.c` 文件的关键部分进行解释，并提供代码片段和中文描述，以及使用说明和简单示例。

**总体介绍:**

`stm32f1xx_hal_pwr.c` 文件是 STM32F1 系列微控制器的 HAL (Hardware Abstraction Layer) 库的一部分，它提供了对电源控制器 (PWR) 外设的驱动函数。  PWR 外设允许配置低功耗模式，电压监测，以及备份域访问控制等功能。  该文件定义了初始化、配置和控制 PWR 外设的函数。

**1.  头文件包含 (Includes):**

```c
#include "stm32f1xx_hal.h"
```

* **描述:**  包含 STM32F1xx HAL 库的头文件。 这个头文件定义了所有 HAL 库函数的原型、数据结构和宏定义。
* **用途:**  在使用任何 HAL 库函数之前，必须包含相应的头文件。

**2.  宏定义 (Macros):**

定义了一些宏，用于位操作，以及地址的定义

```c
#define PVD_MODE_IT               0x00010000U
#define PVD_MODE_EVT              0x00020000U
#define PVD_RISING_EDGE           0x00000001U
#define PVD_FALLING_EDGE          0x00000002U
```

*   **描述:** 定义PVD中断、事件、上升沿、下降沿的宏。
*   **用途:** 使用宏定义可以增加代码的可读性以及可维护性

```c
#define PWR_OFFSET               (PWR_BASE - PERIPH_BASE)
#define PWR_CR_OFFSET            0x00U
#define PWR_CSR_OFFSET           0x04U
#define PWR_CR_OFFSET_BB         (PWR_OFFSET + PWR_CR_OFFSET)
#define PWR_CSR_OFFSET_BB        (PWR_OFFSET + PWR_CSR_OFFSET)
```

*   **描述:** 定义PWR各个寄存器地址偏移的宏。
*   **用途:**  方便对寄存器进行操作。

```c
#define LPSDSR_BIT_NUMBER        PWR_CR_LPDS_Pos
#define CR_LPSDSR_BB             ((uint32_t)(PERIPH_BB_BASE + (PWR_CR_OFFSET_BB * 32U) + (LPSDSR_BIT_NUMBER * 4U)))

#define DBP_BIT_NUMBER            PWR_CR_DBP_Pos
#define CR_DBP_BB                ((uint32_t)(PERIPH_BB_BASE + (PWR_CR_OFFSET_BB * 32U) + (DBP_BIT_NUMBER * 4U)))

#define PVDE_BIT_NUMBER           PWR_CR_PVDE_Pos
#define CR_PVDE_BB               ((uint32_t)(PERIPH_BB_BASE + (PWR_CR_OFFSET_BB * 32U) + (PVDE_BIT_NUMBER * 4U)))

#define CSR_EWUP_BB(VAL)         ((uint32_t)(PERIPH_BB_BASE + (PWR_CSR_OFFSET_BB * 32U) + (POSITION_VAL(VAL) * 4U)))
```

*   **描述:** 定义位带操作的宏。
*   **用途:**  方便进行位带操作。

**3.  取消初始化函数 (HAL_PWR_DeInit):**

```c
void HAL_PWR_DeInit(void)
{
  __HAL_RCC_PWR_FORCE_RESET();
  __HAL_RCC_PWR_RELEASE_RESET();
}
```

*   **描述:** 将 PWR 外设的寄存器重置为默认值。首先强制复位 PWR 外设，然后释放复位。
*   **用途:** 在重新配置 PWR 外设之前，通常需要调用此函数以确保外设处于已知状态。

**4.  备份域访问控制 (HAL_PWR_EnableBkUpAccess, HAL_PWR_DisableBkUpAccess):**

```c
void HAL_PWR_EnableBkUpAccess(void)
{
  /* Enable access to RTC and backup registers */
  *(__IO uint32_t *) CR_DBP_BB = (uint32_t)ENABLE;
}

void HAL_PWR_DisableBkUpAccess(void)
{
  /* Disable access to RTC and backup registers */
  *(__IO uint32_t *) CR_DBP_BB = (uint32_t)DISABLE;
}
```

*   **描述:** `HAL_PWR_EnableBkUpAccess` 允许访问备份域（RTC 寄存器，RTC 备份数据寄存器）。 `HAL_PWR_DisableBkUpAccess` 禁止访问备份域。
*   **用途:** 备份域在系统复位或进入低功耗模式时保持其内容。 必须启用备份域访问才能修改 RTC 寄存器或备份数据寄存器。

    **示例:**

    ```c
    // 启用 PWR 时钟 (必须首先执行)
    __HAL_RCC_PWR_CLK_ENABLE();

    // 允许访问备份域
    HAL_PWR_EnableBkUpAccess();

    // ... 在这里修改 RTC 寄存器或备份数据寄存器 ...

    // (可选) 禁止访问备份域
    //HAL_PWR_DisableBkUpAccess();
    ```

**5.  电源电压检测器 (PVD) 配置 (HAL_PWR_ConfigPVD, HAL_PWR_EnablePVD, HAL_PWR_DisablePVD):**

```c
void HAL_PWR_ConfigPVD(PWR_PVDTypeDef *sConfigPVD)
{
  /* Check the parameters */
  assert_param(IS_PWR_PVD_LEVEL(sConfigPVD->PVDLevel));
  assert_param(IS_PWR_PVD_MODE(sConfigPVD->Mode));

  /* Set PLS[7:5] bits according to PVDLevel value */
  MODIFY_REG(PWR->CR, PWR_CR_PLS, sConfigPVD->PVDLevel);
  
  /* Clear any previous config. Keep it clear if no event or IT mode is selected */
  __HAL_PWR_PVD_EXTI_DISABLE_EVENT();
  __HAL_PWR_PVD_EXTI_DISABLE_IT();
  __HAL_PWR_PVD_EXTI_DISABLE_FALLING_EDGE(); 
  __HAL_PWR_PVD_EXTI_DISABLE_RISING_EDGE();

  /* Configure interrupt mode */
  if((sConfigPVD->Mode & PVD_MODE_IT) == PVD_MODE_IT)
  {
    __HAL_PWR_PVD_EXTI_ENABLE_IT();
  }
  
  /* Configure event mode */
  if((sConfigPVD->Mode & PVD_MODE_EVT) == PVD_MODE_EVT)
  {
    __HAL_PWR_PVD_EXTI_ENABLE_EVENT();
  }
  
  /* Configure the edge */
  if((sConfigPVD->Mode & PVD_RISING_EDGE) == PVD_RISING_EDGE)
  {
    __HAL_PWR_PVD_EXTI_ENABLE_RISING_EDGE();
  }
  
  if((sConfigPVD->Mode & PVD_FALLING_EDGE) == PVD_FALLING_EDGE)
  {
    __HAL_PWR_PVD_EXTI_ENABLE_FALLING_EDGE();
  }
}

void HAL_PWR_EnablePVD(void)
{
  /* Enable the power voltage detector */
  *(__IO uint32_t *) CR_PVDE_BB = (uint32_t)ENABLE;
}

void HAL_PWR_DisablePVD(void)
{
  /* Disable the power voltage detector */
  *(__IO uint32_t *) CR_PVDE_BB = (uint32_t)DISABLE;
}
```

*   **描述:**  `HAL_PWR_ConfigPVD` 配置 PVD 的电压阈值和中断/事件模式。 `HAL_PWR_EnablePVD` 启用 PVD。 `HAL_PWR_DisablePVD` 禁用 PVD。
*   **用途:**  PVD 用于监视电源电压 (VDD)。 如果 VDD 低于配置的阈值，则可以生成中断或事件。

    **示例:**

    ```c
    PWR_PVDTypeDef pvd_config;

    // 配置 PVD
    pvd_config.PVDLevel = PWR_PVDLEVEL_2V2;  // 设置阈值为 2.2V
    pvd_config.Mode = PWR_PVD_MODE_IT_RISING; // 中断模式，上升沿触发
    HAL_PWR_ConfigPVD(&pvd_config);

    // 启用 PVD
    HAL_PWR_EnablePVD();

    // ... 在中断处理程序中处理 PVD 中断 (EXTI16_IRQHandler) ...
    ```

**6.  唤醒引脚配置 (HAL_PWR_EnableWakeUpPin, HAL_PWR_DisableWakeUpPin):**

```c
void HAL_PWR_EnableWakeUpPin(uint32_t WakeUpPinx)
{
  /* Check the parameter */
  assert_param(IS_PWR_WAKEUP_PIN(WakeUpPinx));
  /* Enable the EWUPx pin */
  *(__IO uint32_t *) CSR_EWUP_BB(WakeUpPinx) = (uint32_t)ENABLE;
}

void HAL_PWR_DisableWakeUpPin(uint32_t WakeUpPinx)
{
  /* Check the parameter */
  assert_param(IS_PWR_WAKEUP_PIN(WakeUpPinx));
  /* Disable the EWUPx pin */
  *(__IO uint32_t *) CSR_EWUP_BB(WakeUpPinx) = (uint32_t)DISABLE;
}
```

*   **描述:** `HAL_PWR_EnableWakeUpPin` 启用指定的唤醒引脚。 `HAL_PWR_DisableWakeUpPin` 禁用指定的唤醒引脚。
*   **用途:** 唤醒引脚用于从待机模式唤醒系统。

    **示例:**

    ```c
    // 启用唤醒引脚 1 (PA0)
    HAL_PWR_EnableWakeUpPin(PWR_WAKEUP_PIN1);

    // ... 进入待机模式 ...
    ```

**7.  低功耗模式进入 (HAL_PWR_EnterSLEEPMode, HAL_PWR_EnterSTOPMode, HAL_PWR_EnterSTANDBYMode):**

```c
void HAL_PWR_EnterSLEEPMode(uint32_t Regulator, uint8_t SLEEPEntry)
{
  /* Check the parameters */
  /* No check on Regulator because parameter not used in SLEEP mode */
  /* Prevent unused argument(s) compilation warning */
  UNUSED(Regulator);

  assert_param(IS_PWR_SLEEP_ENTRY(SLEEPEntry));

  /* Clear SLEEPDEEP bit of Cortex System Control Register */
  CLEAR_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SLEEPDEEP_Msk));

  /* Select SLEEP mode entry -------------------------------------------------*/
  if(SLEEPEntry == PWR_SLEEPENTRY_WFI)
  {
    /* Request Wait For Interrupt */
    __WFI();
  }
  else
  {
    /* Request Wait For Event */
    __SEV();
    __WFE();
    __WFE();
  }
}
```

```c
void HAL_PWR_EnterSTOPMode(uint32_t Regulator, uint8_t STOPEntry)
{
  /* Check the parameters */
  assert_param(IS_PWR_REGULATOR(Regulator));
  assert_param(IS_PWR_STOP_ENTRY(STOPEntry));

  /* Clear PDDS bit in PWR register to specify entering in STOP mode when CPU enter in Deepsleep */ 
  CLEAR_BIT(PWR->CR,  PWR_CR_PDDS);

  /* Select the voltage regulator mode by setting LPDS bit in PWR register according to Regulator parameter value */
  MODIFY_REG(PWR->CR, PWR_CR_LPDS, Regulator);

  /* Set SLEEPDEEP bit of Cortex System Control Register */
  SET_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SLEEPDEEP_Msk));

  /* Select Stop mode entry --------------------------------------------------*/
  if(STOPEntry == PWR_STOPENTRY_WFI)
  {
    /* Request Wait For Interrupt */
    __WFI();
  }
  else
  {
    /* Request Wait For Event */
    __SEV();
    PWR_OverloadWfe(); /* WFE redefine locally */
    PWR_OverloadWfe(); /* WFE redefine locally */
  }
  /* Reset SLEEPDEEP bit of Cortex System Control Register */
  CLEAR_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SLEEPDEEP_Msk));
}
```

```c
void HAL_PWR_EnterSTANDBYMode(void)
{
  /* Select Standby mode */
  SET_BIT(PWR->CR, PWR_CR_PDDS);

  /* Set SLEEPDEEP bit of Cortex System Control Register */
  SET_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SLEEPDEEP_Msk));

  /* This option is used to ensure that store operations are completed */
#if defined ( __CC_ARM)
  __force_stores();
#endif
  /* Request Wait For Interrupt */
  __WFI();
}
```

*   **描述:**  这些函数分别用于进入睡眠模式、停止模式和待机模式。
    *   `HAL_PWR_EnterSLEEPMode`：CPU 关闭时钟，外设继续运行。
    *   `HAL_PWR_EnterSTOPMode`：所有时钟停止，SRAM 和寄存器内容保留。
    *   `HAL_PWR_EnterSTANDBYMode`：1.8V 域断电，SRAM 和寄存器内容丢失（备份域除外）。
*   **用途:**  用于降低功耗。

    **示例:**

    ```c
    // 进入睡眠模式 (WFI)
    HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);

    // 进入停止模式 (低功耗稳压器，WFI)
    HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);

    // 进入待机模式
    HAL_PWR_EnterSTANDBYMode();
    ```

**8. Sleep-On-Exit和SEVONPEND配置**

```c
void HAL_PWR_EnableSleepOnExit(void)
{
  /* Set SLEEPONEXIT bit of Cortex System Control Register */
  SET_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SLEEPONEXIT_Msk));
}

void HAL_PWR_DisableSleepOnExit(void)
{
  /* Clear SLEEPONEXIT bit of Cortex System Control Register */
  CLEAR_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SLEEPONEXIT_Msk));
}

void HAL_PWR_EnableSEVOnPend(void)
{
  /* Set SEVONPEND bit of Cortex System Control Register */
  SET_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SEVONPEND_Msk));
}

void HAL_PWR_DisableSEVOnPend(void)
{
  /* Clear SEVONPEND bit of Cortex System Control Register */
  CLEAR_BIT(SCB->SCR, ((uint32_t)SCB_SCR_SEVONPEND_Msk));
}
```

*   **描述:**
    *   `HAL_PWR_EnableSleepOnExit`: 使能从中断返回后进入睡眠模式。
    *   `HAL_PWR_DisableSleepOnExit`: 关闭从中断返回后进入睡眠模式。
    *    `HAL_PWR_EnableSEVOnPend`: 使能pending中断唤醒WFE.
    *   `HAL_PWR_DisableSEVOnPend`: 关闭pending中断唤醒WFE.
*   **用途:** 可以实现更细粒度的功耗管理。

**9.  PVD 中断处理 (HAL_PWR_PVD_IRQHandler, HAL_PWR_PVDCallback):**

```c
void HAL_PWR_PVD_IRQHandler(void)
{
  /* Check PWR exti flag */
  if(__HAL_PWR_PVD_EXTI_GET_FLAG() != RESET)
  {
    /* PWR PVD interrupt user callback */
    HAL_PWR_PVDCallback();

    /* Clear PWR Exti pending bit */
    __HAL_PWR_PVD_EXTI_CLEAR_FLAG();
  }
}

__weak void HAL_PWR_PVDCallback(void)
{
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_PWR_PVDCallback could be implemented in the user file
   */
}
```

*   **描述:** `HAL_PWR_PVD_IRQHandler` 是 PVD 中断处理程序。 它检查 PVD EXTI 标志，如果已设置，则调用 `HAL_PWR_PVDCallback` 用户回调函数。 `HAL_PWR_PVDCallback` 是一个弱函数，用户可以重新定义它以处理 PVD 中断。
*   **用途:**  用于在电源电压低于阈值时执行特定操作。

    **示例:**

    ```c
    // 在 stm32f1xx_it.c 中定义 PVD 中断处理程序
    void EXTI16_IRQHandler(void)
    {
      HAL_PWR_PVD_IRQHandler();
    }

    // 在 main.c 或其他用户文件中重新定义 HAL_PWR_PVDCallback
    void HAL_PWR_PVDCallback(void)
    {
      // 电源电压过低时的处理逻辑
      // 例如: 关闭外设，进入低功耗模式，等等
      HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET); // 关闭 LED
    }
    ```

**注意事项:**

*   在使用任何 PWR HAL 库函数之前，必须启用 PWR 外设的时钟：`__HAL_RCC_PWR_CLK_ENABLE()`。
*   低功耗模式的选择取决于具体的应用需求。 待机模式提供最低的功耗，但会丢失 SRAM 和寄存器内容。 停止模式提供较低的功耗，并保留 SRAM 和寄存器内容。 睡眠模式提供最低的唤醒延迟，但功耗相对较高。

希望以上详细的解释和示例能够帮助你理解 STM32F1xx HAL 库中 `stm32f1xx_hal_pwr.c` 文件的内容。
