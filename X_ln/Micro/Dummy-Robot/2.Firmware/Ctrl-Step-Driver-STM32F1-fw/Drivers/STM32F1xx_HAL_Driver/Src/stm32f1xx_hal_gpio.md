Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\STM32F1xx_HAL_Driver\Src\stm32f1xx_hal_gpio.c`

Okay, let's analyze the STM32F1xx GPIO HAL driver code, breaking it down into key parts with explanations and Chinese translations.  I'll also provide simple usage examples in Chinese.

**1. Header Inclusion and Group Definitions:**

```c
#include "stm32f1xx_hal.h"

/** @addtogroup STM32F1xx_HAL_Driver
  * @{
  */

/** @defgroup GPIO GPIO
  * @brief GPIO HAL module driver
  * @{
  */

#ifdef HAL_GPIO_MODULE_ENABLED
```

**Explanation:**

*   `#include "stm32f1xx_hal.h"`:  Includes the main HAL header file, providing definitions for data types, structures, and other HAL functions.
*   `/** @addtogroup STM32F1xx_HAL_Driver ... */`: These are Doxygen comments used for documentation generation.  They define the overall HAL driver group.
*   `/** @defgroup GPIO GPIO ... */`: Defines the specific GPIO driver group within the HAL.
*   `#ifdef HAL_GPIO_MODULE_ENABLED`:  Conditional compilation. The code within this `#ifdef` block is only compiled if the `HAL_GPIO_MODULE_ENABLED` macro is defined (typically in `stm32f1xx_hal_conf.h`).  This allows you to enable or disable the GPIO driver module to save code space if it's not needed.

**Chinese Translation:**

*   `#include "stm32f1xx_hal.h"`:  包含 HAL 库的主头文件，提供了数据类型、结构体和其他 HAL 函数的定义。
*   `/** @addtogroup STM32F1xx_HAL_Driver ... */`: 这些是 Doxygen 注释，用于生成文档。它们定义了整个 HAL 驱动组。
*   `/** @defgroup GPIO GPIO ... */`: 定义了 HAL 中特定的 GPIO 驱动组。
*   `#ifdef HAL_GPIO_MODULE_ENABLED`:  条件编译。只有定义了 `HAL_GPIO_MODULE_ENABLED` 宏（通常在 `stm32f1xx_hal_conf.h` 中）时，才会编译此 `#ifdef` 块中的代码。这允许您启用或禁用 GPIO 驱动模块，以节省代码空间（如果不需要）。

**2. Private Definitions (Macros):**

```c
#define GPIO_MODE             0x00000003u
#define EXTI_MODE             0x10000000u
#define GPIO_MODE_IT          0x00010000u
#define GPIO_MODE_EVT         0x00020000u
#define RISING_EDGE           0x00100000u
#define FALLING_EDGE          0x00200000u
#define GPIO_OUTPUT_TYPE      0x00000010u

#define GPIO_NUMBER           16u

/* Definitions for bit manipulation of CRL and CRH register */
#define  GPIO_CR_MODE_INPUT         0x00000000u /*!< 00: Input mode (reset state)  */
#define  GPIO_CR_CNF_ANALOG         0x00000000u /*!< 00: Analog mode  */
#define  GPIO_CR_CNF_INPUT_FLOATING 0x00000004u /*!< 01: Floating input (reset state)  */
#define  GPIO_CR_CNF_INPUT_PU_PD    0x00000008u /*!< 10: Input with pull-up / pull-down  */
#define  GPIO_CR_CNF_GP_OUTPUT_PP   0x00000000u /*!< 00: General purpose output push-pull  */
#define  GPIO_CR_CNF_GP_OUTPUT_OD   0x00000004u /*!< 01: General purpose output Open-drain  */
#define  GPIO_CR_CNF_AF_OUTPUT_PP   0x00000008u /*!< 10: Alternate function output Push-pull  */
#define  GPIO_CR_CNF_AF_OUTPUT_OD   0x0000000Cu /*!< 11: Alternate function output Open-drain  */
```

**Explanation:**

*   These `#define` macros define constants used internally by the GPIO driver.  They represent bit masks and values for configuring GPIO pins, especially related to the `GPIO_InitTypeDef` structure.
*   `GPIO_MODE`, `EXTI_MODE`, `GPIO_MODE_IT`, etc.:  These define bit flags to indicate the GPIO mode (input, output, alternate function, interrupt, event).  These flags are used within the `GPIO_InitTypeDef` structure's `Mode` member.
*   `RISING_EDGE`, `FALLING_EDGE`: Define the trigger edge for external interrupts/events.
*   `GPIO_NUMBER`: Specifies the number of GPIO pins per port (16 in this case).
*   `GPIO_CR_MODE_INPUT`, `GPIO_CR_CNF_ANALOG`, etc.: These are very important! They define the bit values for configuring the `MODE` and `CNF` (Configuration) bits within the GPIO control registers (`CRL` and `CRH`).  These macros directly map to the hardware's register settings.  For example:
    *   `GPIO_CR_MODE_INPUT` is 0, meaning the mode bits are set to 00 for input.
    *   `GPIO_CR_CNF_INPUT_FLOATING` is 4, meaning the configuration bits are set to 01 for floating input.
    *   `GPIO_CR_CNF_GP_OUTPUT_PP` is 0, meaning the configuration bits are set to 00 for general-purpose output, push-pull.

**Chinese Translation:**

*   这些 `#define` 宏定义了 GPIO 驱动程序内部使用的常量。它们代表了用于配置 GPIO 引脚的位掩码和值，特别是与 `GPIO_InitTypeDef` 结构体相关的。
*   `GPIO_MODE`, `EXTI_MODE`, `GPIO_MODE_IT` 等：这些定义了指示 GPIO 模式（输入、输出、复用功能、中断、事件）的位标志。这些标志在 `GPIO_InitTypeDef` 结构体的 `Mode` 成员中使用。
*   `RISING_EDGE`, `FALLING_EDGE`: 定义了外部中断/事件的触发边沿。
*   `GPIO_NUMBER`: 指定每个端口的 GPIO 引脚数（本例中为 16）。
*   `GPIO_CR_MODE_INPUT`, `GPIO_CR_CNF_ANALOG` 等：这些非常重要！它们定义了用于配置 GPIO 控制寄存器 (`CRL` 和 `CRH`) 中的 `MODE` 和 `CNF`（配置）位的位值。这些宏直接映射到硬件的寄存器设置。例如：
    *   `GPIO_CR_MODE_INPUT` 为 0，表示模式位设置为 00（输入）。
    *   `GPIO_CR_CNF_INPUT_FLOATING` 为 4，表示配置位设置为 01（浮空输入）。
    *   `GPIO_CR_CNF_GP_OUTPUT_PP` 为 0，表示配置位设置为 00（通用输出，推挽）。

**3. `HAL_GPIO_Init()` Function:**

```c
void HAL_GPIO_Init(GPIO_TypeDef  *GPIOx, GPIO_InitTypeDef *GPIO_Init)
{
  uint32_t position = 0x00u;
  uint32_t ioposition;
  uint32_t iocurrent;
  uint32_t temp;
  uint32_t config = 0x00u;
  __IO uint32_t *configregister; /* Store the address of CRL or CRH register based on pin number */
  uint32_t registeroffset;       /* offset used during computation of CNF and MODE bits placement inside CRL or CRH register */

  /* Check the parameters */
  assert_param(IS_GPIO_ALL_INSTANCE(GPIOx));
  assert_param(IS_GPIO_PIN(GPIO_Init->Pin));
  assert_param(IS_GPIO_MODE(GPIO_Init->Mode));

  /* Configure the port pins */
  while (((GPIO_Init->Pin) >> position) != 0x00u)
  {
    /* Get the IO position */
    ioposition = (0x01uL << position);

    /* Get the current IO position */
    iocurrent = (uint32_t)(GPIO_Init->Pin) & ioposition;

    if (iocurrent == ioposition)
    {
      /* Check the Alternate function parameters */
      assert_param(IS_GPIO_AF_INSTANCE(GPIOx));

      /* Based on the required mode, filling config variable with MODEy[1:0] and CNFy[3:2] corresponding bits */
      switch (GPIO_Init->Mode)
      {
        /* If we are configuring the pin in OUTPUT push-pull mode */
        case GPIO_MODE_OUTPUT_PP:
          /* Check the GPIO speed parameter */
          assert_param(IS_GPIO_SPEED(GPIO_Init->Speed));
          config = GPIO_Init->Speed + GPIO_CR_CNF_GP_OUTPUT_PP;
          break;

        /* If we are configuring the pin in OUTPUT open-drain mode */
        case GPIO_MODE_OUTPUT_OD:
          /* Check the GPIO speed parameter */
          assert_param(IS_GPIO_SPEED(GPIO_Init->Speed));
          config = GPIO_Init->Speed + GPIO_CR_CNF_GP_OUTPUT_OD;
          break;

        /* If we are configuring the pin in ALTERNATE FUNCTION push-pull mode */
        case GPIO_MODE_AF_PP:
          /* Check the GPIO speed parameter */
          assert_param(IS_GPIO_SPEED(GPIO_Init->Speed));
          config = GPIO_Init->Speed + GPIO_CR_CNF_AF_OUTPUT_PP;
          break;

        /* If we are configuring the pin in ALTERNATE FUNCTION open-drain mode */
        case GPIO_MODE_AF_OD:
          /* Check the GPIO speed parameter */
          assert_param(IS_GPIO_SPEED(GPIO_Init->Speed));
          config = GPIO_Init->Speed + GPIO_CR_CNF_AF_OUTPUT_OD;
          break;

        /* If we are configuring the pin in INPUT (also applicable to EVENT and IT mode) */
        case GPIO_MODE_INPUT:
        case GPIO_MODE_IT_RISING:
        case GPIO_MODE_IT_FALLING:
        case GPIO_MODE_IT_RISING_FALLING:
        case GPIO_MODE_EVT_RISING:
        case GPIO_MODE_EVT_FALLING:
        case GPIO_MODE_EVT_RISING_FALLING:
          /* Check the GPIO pull parameter */
          assert_param(IS_GPIO_PULL(GPIO_Init->Pull));
          if (GPIO_Init->Pull == GPIO_NOPULL)
          {
            config = GPIO_CR_MODE_INPUT + GPIO_CR_CNF_INPUT_FLOATING;
          }
          else if (GPIO_Init->Pull == GPIO_PULLUP)
          {
            config = GPIO_CR_MODE_INPUT + GPIO_CR_CNF_INPUT_PU_PD;

            /* Set the corresponding ODR bit */
            GPIOx->BSRR = ioposition;
          }
          else /* GPIO_PULLDOWN */
          {
            config = GPIO_CR_MODE_INPUT + GPIO_CR_CNF_INPUT_PU_PD;

            /* Reset the corresponding ODR bit */
            GPIOx->BRR = ioposition;
          }
          break;

        /* If we are configuring the pin in INPUT analog mode */
        case GPIO_MODE_ANALOG:
          config = GPIO_CR_MODE_INPUT + GPIO_CR_CNF_ANALOG;
          break;

        /* Parameters are checked with assert_param */
        default:
          break;
      }

      /* Check if the current bit belongs to first half or last half of the pin count number
       in order to address CRH or CRL register*/
      configregister = (iocurrent < GPIO_PIN_8) ? &GPIOx->CRL     : &GPIOx->CRH;
      registeroffset = (iocurrent < GPIO_PIN_8) ? (position << 2u) : ((position - 8u) << 2u);

      /* Apply the new configuration of the pin to the register */
      MODIFY_REG((*configregister), ((GPIO_CRL_MODE0 | GPIO_CRL_CNF0) << registeroffset), (config << registeroffset));

      /*--------------------- EXTI Mode Configuration ------------------------*/
      /* Configure the External Interrupt or event for the current IO */
      if ((GPIO_Init->Mode & EXTI_MODE) == EXTI_MODE)
      {
        /* Enable AFIO Clock */
        __HAL_RCC_AFIO_CLK_ENABLE();
        temp = AFIO->EXTICR[position >> 2u];
        CLEAR_BIT(temp, (0x0Fu) << (4u * (position & 0x03u)));
        SET_BIT(temp, (GPIO_GET_INDEX(GPIOx)) << (4u * (position & 0x03u)));
        AFIO->EXTICR[position >> 2u] = temp;


        /* Configure the interrupt mask */
        if ((GPIO_Init->Mode & GPIO_MODE_IT) == GPIO_MODE_IT)
        {
          SET_BIT(EXTI->IMR, iocurrent);
        }
        else
        {
          CLEAR_BIT(EXTI->IMR, iocurrent);
        }

        /* Configure the event mask */
        if ((GPIO_Init->Mode & GPIO_MODE_EVT) == GPIO_MODE_EVT)
        {
          SET_BIT(EXTI->EMR, iocurrent);
        }
        else
        {
          CLEAR_BIT(EXTI->EMR, iocurrent);
        }

        /* Enable or disable the rising trigger */
        if ((GPIO_Init->Mode & RISING_EDGE) == RISING_EDGE)
        {
          SET_BIT(EXTI->RTSR, iocurrent);
        }
        else
        {
          CLEAR_BIT(EXTI->RTSR, iocurrent);
        }

        /* Enable or disable the falling trigger */
        if ((GPIO_Init->Mode & FALLING_EDGE) == FALLING_EDGE)
        {
          SET_BIT(EXTI->FTSR, iocurrent);
        }
        else
        {
          CLEAR_BIT(EXTI->FTSR, iocurrent);
        }
      }
    }

	position++;
  }
}
```

**Explanation:**

*   This function initializes a GPIO pin according to the parameters specified in the `GPIO_InitTypeDef` structure.  It's the core function for configuring GPIOs.
*   `GPIO_TypeDef *GPIOx`:  Pointer to the GPIO port (e.g., `GPIOA`, `GPIOB`).
*   `GPIO_InitTypeDef *GPIO_Init`: Pointer to the structure containing the configuration parameters.
*   **Parameter Checks:** `assert_param()` is used extensively to validate the input parameters (e.g., the GPIO port, the pin number, and the mode). This is important for debugging.
*   **Pin Iteration:** The `while` loop iterates through each pin specified in the `GPIO_Init->Pin` bitmask. This allows you to initialize multiple pins with the same configuration.
*   **Mode Configuration:** The `switch` statement handles different GPIO modes (output, input, alternate function, analog).
*   **Output Configuration:**  For output modes (push-pull, open-drain), it sets the speed and output type.
*   **Input Configuration:** For input modes, it configures the pull-up/pull-down resistors.  The code directly sets/resets the Output Data Register (`ODR`) bits to enable pull-up or pull-down.
*   **Register Selection (CRL/CRH):** The code determines whether to use the `CRL` (Control Register Low) or `CRH` (Control Register High) based on the pin number. Pins 0-7 are configured using `CRL`, and pins 8-15 are configured using `CRH`.
*   **`MODIFY_REG()`:** This macro is crucial! It's used to modify specific bits in the GPIO control registers (`CRL` and `CRH`) without affecting other bits.  It reads the register, clears the relevant bits, sets the new bits, and writes the value back to the register.
*   **EXTI Configuration:** If the GPIO pin is configured for external interrupt/event, the code configures the AFIO (Alternate Function I/O) registers and the EXTI (External Interrupt/Event Controller) registers to enable interrupts or events on the specified pin and trigger edge (rising, falling, or both).

**Chinese Translation:**

*   此函数根据 `GPIO_InitTypeDef` 结构体中指定的参数初始化 GPIO 引脚。 它是配置 GPIO 的核心函数。
*   `GPIO_TypeDef *GPIOx`: 指向 GPIO 端口的指针（例如，`GPIOA`、`GPIOB`）。
*   `GPIO_InitTypeDef *GPIO_Init`: 指向包含配置参数的结构体的指针。
*   **参数检查：** 广泛使用 `assert_param()` 来验证输入参数（例如，GPIO 端口、引脚号和模式）。 这对于调试很重要。
*   **引脚迭代：** `while` 循环遍历 `GPIO_Init->Pin` 位掩码中指定的每个引脚。 这允许您使用相同的配置初始化多个引脚。
*   **模式配置：** `switch` 语句处理不同的 GPIO 模式（输出、输入、复用功能、模拟）。
*   **输出配置：** 对于输出模式（推挽、开漏），它设置速度和输出类型。
*   **输入配置：** 对于输入模式，它配置上拉/下拉电阻。 代码直接设置/复位输出数据寄存器 (`ODR`) 位以启用上拉或下拉。
*   **寄存器选择 (CRL/CRH)：** 代码根据引脚号确定是使用 `CRL`（控制寄存器低）还是 `CRH`（控制寄存器高）。 引脚 0-7 使用 `CRL` 配置，引脚 8-15 使用 `CRH` 配置。
*   **`MODIFY_REG()`：** 这个宏非常重要！ 它用于修改 GPIO 控制寄存器 (`CRL` 和 `CRH`) 中的特定位，而不影响其他位。 它读取寄存器，清除相关位，设置新位，然后将值写回寄存器。
*   **EXTI 配置：** 如果 GPIO 引脚配置为外部中断/事件，则代码配置 AFIO（复用功能 I/O）寄存器和 EXTI（外部中断/事件控制器）寄存器，以在指定的引脚和触发边沿（上升沿、下降沿或两者）上启用中断或事件。

**4. `HAL_GPIO_DeInit()` Function:**

```c
void HAL_GPIO_DeInit(GPIO_TypeDef  *GPIOx, uint32_t GPIO_Pin)
{
  uint32_t position = 0x00u;
  uint32_t iocurrent;
  uint32_t tmp;
  __IO uint32_t *configregister; /* Store the address of CRL or CRH register based on pin number */
  uint32_t registeroffset;

  /* Check the parameters */
  assert_param(IS_GPIO_ALL_INSTANCE(GPIOx));
  assert_param(IS_GPIO_PIN(GPIO_Pin));

  /* Configure the port pins */
  while ((GPIO_Pin >> position) != 0u)
  {
    /* Get current io position */
    iocurrent = (GPIO_Pin) & (1uL << position);

    if (iocurrent)
    {
      /*------------------------- EXTI Mode Configuration --------------------*/
      /* Clear the External Interrupt or Event for the current IO */

      tmp = AFIO->EXTICR[position >> 2u];
      tmp &= 0x0FuL << (4u * (position & 0x03u));
      if (tmp == (GPIO_GET_INDEX(GPIOx) << (4u * (position & 0x03u))))
      {
        tmp = 0x0FuL << (4u * (position & 0x03u));
        CLEAR_BIT(AFIO->EXTICR[position >> 2u], tmp);

        /* Clear EXTI line configuration */
        CLEAR_BIT(EXTI->IMR, (uint32_t)iocurrent);
        CLEAR_BIT(EXTI->EMR, (uint32_t)iocurrent);

        /* Clear Rising Falling edge configuration */
        CLEAR_BIT(EXTI->RTSR, (uint32_t)iocurrent);
        CLEAR_BIT(EXTI->FTSR, (uint32_t)iocurrent);
      }
      /*------------------------- GPIO Mode Configuration --------------------*/
      /* Check if the current bit belongs to first half or last half of the pin count number
       in order to address CRH or CRL register */
      configregister = (iocurrent < GPIO_PIN_8) ? &GPIOx->CRL     : &GPIOx->CRH;
      registeroffset = (iocurrent < GPIO_PIN_8) ? (position << 2u) : ((position - 8u) << 2u);

      /* CRL/CRH default value is floating input(0x04) shifted to correct position */
      MODIFY_REG(*configregister, ((GPIO_CRL_MODE0 | GPIO_CRL_CNF0) << registeroffset), GPIO_CRL_CNF0_0 << registeroffset);

      /* ODR default value is 0 */
      CLEAR_BIT(GPIOx->ODR, iocurrent);
    }

    position++;
  }
}
```

**Explanation:**

*   This function de-initializes a GPIO pin, setting it back to its default state (input floating).
*   `GPIO_TypeDef *GPIOx`: Pointer to the GPIO port.
*   `uint32_t GPIO_Pin`: Bitmask of the pins to de-initialize.
*   **EXTI Clearing:** The code first clears any EXTI configuration associated with the pin.  It checks if the AFIO EXTICR register points to the current GPIO port before clearing the EXTI settings. This prevents accidentally clearing EXTI configurations for other GPIO ports that might be using the same EXTI line.
*   **Mode Reset:**  It then resets the `CRL` or `CRH` register to its default state (input floating).
*   **ODR Clearing:**  The Output Data Register (`ODR`) bit is cleared to ensure the pin is not driving any output.

**Chinese Translation:**

*   此函数取消初始化 GPIO 引脚，将其设置回其默认状态（浮空输入）。
*   `GPIO_TypeDef *GPIOx`: 指向 GPIO 端口的指针。
*   `uint32_t GPIO_Pin`: 要取消初始化的引脚的位掩码。
*   **EXTI 清除：** 代码首先清除与引脚关联的任何 EXTI 配置。 在清除 EXTI 设置之前，它会检查 AFIO EXTICR 寄存器是否指向当前 GPIO 端口。 这样可以防止意外清除可能正在使用同一 EXTI 线的其他 GPIO 端口的 EXTI 配置。
*   **模式复位：** 然后，它将 `CRL` 或 `CRH` 寄存器重置为其默认状态（浮空输入）。
*   **ODR 清除：** 清除输出数据寄存器 (`ODR`) 位，以确保引脚不驱动任何输出。

**5. IO Operation Functions (`HAL_GPIO_ReadPin()`, `HAL_GPIO_WritePin()`, `HAL_GPIO_TogglePin()`):**

```c
GPIO_PinState HAL_GPIO_ReadPin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
  GPIO_PinState bitstatus;

  /* Check the parameters */
  assert_param(IS_GPIO_PIN(GPIO_Pin));

  if ((GPIOx->IDR & GPIO_Pin) != (uint32_t)GPIO_PIN_RESET)
  {
    bitstatus = GPIO_PIN_SET;
  }
  else
  {
    bitstatus = GPIO_PIN_RESET;
  }
  return bitstatus;
}

void HAL_GPIO_WritePin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, GPIO_PinState PinState)
{
  /* Check the parameters */
  assert_param(IS_GPIO_PIN(GPIO_Pin));
  assert_param(IS_GPIO_PIN_ACTION(PinState));

  if (PinState != GPIO_PIN_RESET)
  {
    GPIOx->BSRR = GPIO_Pin;
  }
  else
  {
    GPIOx->BSRR = (uint32_t)GPIO_Pin << 16u;
  }
}

void HAL_GPIO_TogglePin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
  uint32_t odr;

  /* Check the parameters */
  assert_param(IS_GPIO_PIN(GPIO_Pin));

  /* get current Ouput Data Register value */
  odr = GPIOx->ODR;

  /* Set selected pins that were at low level, and reset ones that were high */
  GPIOx->BSRR = ((odr & GPIO_Pin) << GPIO_NUMBER) | (~odr & GPIO_Pin);
}
```

**Explanation:**

*   `HAL_GPIO_ReadPin()`: Reads the state of a GPIO input pin.  It reads the Input Data Register (`IDR`).
*   `HAL_GPIO_WritePin()`: Sets or clears a GPIO output pin. It uses the Bit Set/Reset Register (`BSRR`) for atomic writes. Writing `GPIO_Pin` to `BSRR` sets the pin, and writing `GPIO_Pin << 16` resets it. Using `BSRR` avoids race conditions when multiple pins on the same port are being controlled.
*   `HAL_GPIO_TogglePin()`: Toggles the state of a GPIO pin.  It reads the current state from the Output Data Register (`ODR`), then uses `BSRR` to set the pins that were low and reset the pins that were high.

**Chinese Translation:**

*   `HAL_GPIO_ReadPin()`：读取 GPIO 输入引脚的状态。 它读取输入数据寄存器 (`IDR`)。
*   `HAL_GPIO_WritePin()`：设置或清除 GPIO 输出引脚。 它使用位设置/复位寄存器 (`BSRR`) 进行原子写入。 将 `GPIO_Pin` 写入 `BSRR` 会设置引脚，将 `GPIO_Pin << 16` 写入会复位引脚。 当控制同一端口上的多个引脚时，使用 `BSRR` 可以避免竞争条件。
*   `HAL_GPIO_TogglePin()`：切换 GPIO 引脚的状态。 它从输出数据寄存器 (`ODR`) 读取当前状态，然后使用 `BSRR` 设置低电平的引脚，并复位高电平的引脚。

**6. `HAL_GPIO_LockPin()` Function:**

```c
HAL_StatusTypeDef HAL_GPIO_LockPin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
  __IO uint32_t tmp = GPIO_LCKR_LCKK;

  /* Check the parameters */
  assert_param(IS_GPIO_LOCK_INSTANCE(GPIOx));
  assert_param(IS_GPIO_PIN(GPIO_Pin));

  /* Apply lock key write sequence */
  SET_BIT(tmp, GPIO_Pin);
  /* Set LCKx bit(s): LCKK='1' + LCK[15-0] */
  GPIOx->LCKR = tmp;
  /* Reset LCKx bit(s): LCKK='0' + LCK[15-0] */
  GPIOx->LCKR = GPIO_Pin;
  /* Set LCKx bit(s): LCKK='1' + LCK[15-0] */
  GPIOx->LCKR = tmp;
  /* Read LCKK register. This read is mandatory to complete key lock sequence */
  tmp = GPIOx->LCKR;

  /* read again in order to confirm lock is active */
  if ((uint32_t)(GPIOx->LCKR & GPIO_LCKR_LCKK))
  {
    return HAL_OK;
  }
  else
  {
    return HAL_ERROR;
  }
}
```

**Explanation:**

*   This function locks the configuration of the specified GPIO pins. Once locked, the pin configuration cannot be changed until the next reset. This is useful for safety-critical applications.
*   The locking mechanism involves writing a specific sequence to the Lock Configuration Register (`LCKR`).  The sequence is:
    1.  Write `1` to the LCKK bit (Lock Key bit) and the pin bits to be locked.
    2.  Write `0` to the LCKK bit and the pin bits.
    3.  Write `1` to the LCKK bit and the pin bits again.
    4.  Read the LCKR register.
*   The LCKK bit must then be read again to confirm the lock is active.
*   The specific sequence is crucial; any deviation will not lock the pins.

**Chinese Translation:**

*   此函数锁定指定 GPIO 引脚的配置。 锁定后，在下次复位之前，无法更改引脚配置。 这对于安全关键型应用很有用。
*   锁定机制涉及将特定序列写入锁定配置寄存器 (`LCKR`)。 该序列是：
    1.  将 `1` 写入 LCKK 位（锁定密钥位）和要锁定的引脚位。
    2.  将 `0` 写入 LCKK 位和引脚位。
    3.  再次将 `1` 写入 LCKK 位和引脚位。
    4.  读取 LCKR 寄存器。
*   然后必须再次读取 LCKK 位，以确认锁定已激活。
*   特定序列至关重要； 任何偏差都不会锁定引脚。

**7. Interrupt Handling (`HAL_GPIO_EXTI_IRQHandler()`, `HAL_GPIO_EXTI_Callback()`):**

```c
void HAL_GPIO_EXTI_IRQHandler(uint16_t GPIO_Pin)
{
  /* EXTI line interrupt detected */
  if (__HAL_GPIO_EXTI_GET_IT(GPIO_Pin) != 0x00u)
  {
    __HAL_GPIO_EXTI_CLEAR_IT(GPIO_Pin);
    HAL_GPIO_EXTI_Callback(GPIO_Pin);
  }
}

__weak void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(GPIO_Pin);
  /* NOTE: This function Should not be modified, when the callback is needed,
           the HAL_GPIO_EXTI_Callback could be implemented in the user file
   */
}
```

**Explanation:**

*   `HAL_GPIO_EXTI_IRQHandler()`: This is the interrupt handler function that is called when an EXTI interrupt occurs on one of the GPIO pins.  It checks if the interrupt flag is set for the specific pin, clears the interrupt flag, and then calls the user-defined callback function `HAL_GPIO_EXTI_Callback()`.
*   `HAL_GPIO_EXTI_Callback()`: This is a *weak* function.  A weak function is a function that can be overridden by the user in their own code. The default implementation is empty.  You *must* define your own `HAL_GPIO_EXTI_Callback()` function in your application code to handle the interrupt.

**Chinese Translation:**

*   `HAL_GPIO_EXTI_IRQHandler()`：这是中断处理函数，当 GPIO 引脚之一上发生 EXTI 中断时调用该函数。 它检查是否为特定引脚设置了中断标志，清除中断标志，然后调用用户定义的回调函数 `HAL_GPIO_EXTI_Callback()`。
*   `HAL_GPIO_EXTI_Callback()`：这是一个*弱*函数。 弱函数是可以由用户在其自己的代码中覆盖的函数。 默认实现为空。 您*必须*在应用程序代码中定义自己的 `HAL_GPIO_EXTI_Callback()` 函数来处理中断。

**Simple Usage Example (Chinese):**

```c
#include "stm32f1xx_hal.h"

GPIO_InitTypeDef GPIO_InitStruct;

void Error_Handler(void);

int main(void) {
  HAL_Init(); // 初始化 HAL 库

  // 1. 使能 GPIO 时钟 (例如, GPIOA)
  __HAL_RCC_GPIOA_CLK_ENABLE();

  // 2. 配置 GPIO 引脚 (例如, PA5 作为输出)
  GPIO_InitStruct.Pin = GPIO_PIN_5;       // 使用 PA5 引脚
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP; // 推挽输出模式
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;  // 低速
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);  // 初始化 GPIOA 的 PA5 引脚

  // 3. 无限循环，控制 PA5 引脚
  while (1) {
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET); // 设置 PA5 为高电平
    HAL_Delay(500);                                    // 延时 500 毫秒
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET); // 设置 PA5 为低电平
    HAL_Delay(500);                                    // 延时 500 毫秒
  }
}

void Error_Handler(void) {
  /* 用户可以添加错误处理代码 */
  while (1) {
    // 例如，点亮一个错误指示灯
  }
}
```

**Explanation of the Example:**

1.  **`HAL_Init()`**: Initializes the HAL library.  This is essential before using any HAL functions.
2.  **`__HAL_RCC_GPIOA_CLK_ENABLE()`**: Enables the clock for GPIO port A.  You *must* enable the clock for the GPIO port you want to use, or the GPIO pins will not function correctly.  The clock enables the power to the GPIO peripheral.
3.  **`GPIO_InitStruct`**: A `GPIO_InitTypeDef` structure is created to hold the configuration settings for the GPIO pin.
4.  **`GPIO_InitStruct.Pin = GPIO_PIN_5`**:  Specifies that we want to configure pin 5 of GPIO port A.
5.  **`GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP`**: Sets the pin to output mode, push-pull.
6.  **`GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW`**: Sets the output speed to low.  This is important for reducing noise and power consumption.
7.  **`HAL_GPIO_Init(GPIOA, &GPIO_InitStruct)`**: Calls the `HAL_GPIO_Init()` function to apply the configuration to GPIO port A, pin 5.
8.  **`HAL