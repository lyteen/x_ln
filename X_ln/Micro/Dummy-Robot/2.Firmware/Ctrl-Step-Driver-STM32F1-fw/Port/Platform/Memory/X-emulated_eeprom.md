Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Port\Platform\Memory\emulated_eeprom.h`

好的，我们重新开始。请注意，以下代码示例假设你已经配置好了 STM32 的 HAL 库，并且已经了解了如何使用 Flash memory 模拟 EEPROM。为了简化代码，我将专注于提供 `eeprom_read_byte` 和 `eeprom_write_byte` 的实现，并针对 STM32F103 和 STM32MP1xx 系列芯片提供不同的实现方式。

**1. STM32F103 系列芯片 EEPROM 模拟实现:**

```c
#include "stm32f103xb.h"
#include "stm32f1xx_hal.h"

#define EEPROM_START_ADDRESS  0x0800FC00  // Flash memory 地址，根据实际情况修改
#define EEPROM_END_ADDRESS    0x0800FFFF  // Flash memory 结束地址，根据实际情况修改
#define EEPROM_SIZE           (EEPROM_END_ADDRESS - EEPROM_START_ADDRESS + 1) // EEPROM 大小

// 读取 EEPROM 中的一个字节
uint8_t eeprom_read_byte(uint32_t pos) {
    if (pos >= EEPROM_SIZE) {
        return 0xFF; // 错误：越界，返回默认值
    }
    uint32_t address = EEPROM_START_ADDRESS + pos;
    return *(__IO uint8_t *)address;
}

// 将一个字节写入 EEPROM
void eeprom_write_byte(uint32_t pos, uint8_t value) {
    if (pos >= EEPROM_SIZE) {
        return; // 错误：越界，直接返回
    }

    // 1. 解锁 Flash
    HAL_FLASH_Unlock();

    // 2. 擦除包含该字节的 Flash 页
    uint32_t page_error = 0;
    FLASH_EraseInitTypeDef EraseInitStruct;
    EraseInitStruct.TypeErase   = FLASH_TYPEERASE_PAGES;
    EraseInitStruct.PageAddress = EEPROM_START_ADDRESS; // 假设整个 EEPROM 区域都在一页内
    EraseInitStruct.NbPages     = 1;                      // 擦除一页

    if (HAL_FLASHEx_Erase(&EraseInitStruct, &page_error) != HAL_OK) {
        // 擦除出错，处理错误
        HAL_FLASH_Lock(); // 记得锁定 Flash
        return;
    }

    // 3. 将数据写入 Flash
    uint32_t address = EEPROM_START_ADDRESS + pos;
    if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_BYTE, address, value) != HAL_OK) {
        // 写入出错，处理错误
        HAL_FLASH_Lock(); // 记得锁定 Flash
        return;
    }

    // 4. 锁定 Flash
    HAL_FLASH_Lock();
}
```

**描述 (中文):**

这段代码模拟了 STM32F103 系列芯片的 EEPROM 功能。

*   `EEPROM_START_ADDRESS` 和 `EEPROM_END_ADDRESS` 定义了 Flash 存储器中用于模拟 EEPROM 的起始和结束地址。你需要根据你的具体应用和 Flash 的可用空间修改这些值。
*   `eeprom_read_byte(uint32_t pos)` 函数从指定的地址 `pos` 读取一个字节。如果地址越界，函数返回 0xFF。
*   `eeprom_write_byte(uint32_t pos, uint8_t value)` 函数将一个字节写入到指定的地址 `pos`。  写入过程包括解锁 Flash，擦除包含该字节的 Flash 页，将数据写入 Flash，然后锁定 Flash。  这个过程非常重要，因为 Flash memory 必须先擦除才能写入。擦除操作是以页为单位进行的，所以即使你只想修改一个字节，你也需要擦除整个页。

**简易 Demo (中文):**

```c
int main(void) {
    HAL_Init(); // 初始化 HAL 库

    // 设置需要写入的数据
    uint32_t address = 0; // 从 EEPROM 起始位置开始
    uint8_t data = 0x5A;

    // 写入数据到 EEPROM
    eeprom_write_byte(address, data);

    // 从 EEPROM 中读取数据
    uint8_t read_data = eeprom_read_byte(address);

    // 验证数据
    if (read_data == data) {
        // 数据写入和读取成功
        // 可以点亮 LED 或者通过串口输出信息
    } else {
        // 数据写入或读取失败
        // 处理错误
    }

    while (1) {
        // 循环
    }
}
```

**2. STM32MP1xx 系列芯片 EEPROM 模拟实现 (使用 RETRAM):**

```c
#include "stm32mp1xx_hal.h"

#define EEPROM_RETRAM_START_ADDRESS (0x00000400UL)
#define EEPROM_RETRAM_MODE_SIZE ((uint32_t)(4*1024))
#define E2END (EEPROM_RETRAM_MODE_SIZE - 1)


// 读取 EEPROM 中的一个字节
uint8_t eeprom_read_byte(uint32_t pos) {
    if (pos > E2END) {
        return 0xFF; // 错误：越界，返回默认值
    }
    uint32_t address = EEPROM_RETRAM_START_ADDRESS + pos;
    return *(__IO uint8_t *)address;
}

// 将一个字节写入 EEPROM
void eeprom_write_byte(uint32_t pos, uint8_t value) {
    if (pos > E2END) {
        return; // 错误：越界，直接返回
    }
    uint32_t address = EEPROM_RETRAM_START_ADDRESS + pos;
    *(__IO uint8_t *)address = value;
}
```

**描述 (中文):**

这段代码模拟了 STM32MP1xx 系列芯片的 EEPROM 功能，使用了 RETRAM (Retention RAM)  来存储数据。 RETRAM 是一种低功耗 SRAM，当芯片进入 Standby 模式并且 VBAT 供电时，数据可以被保存。

*   `EEPROM_RETRAM_START_ADDRESS` 定义了 RETRAM 中用于模拟 EEPROM 的起始地址。 请务必确保这个地址不与你的其他变量或代码冲突。
*   `EEPROM_RETRAM_MODE_SIZE` 定义了 RETRAM 模拟 EEPROM 的大小。
*   `eeprom_read_byte(uint32_t pos)` 函数从 RETRAM 中指定的地址 `pos` 读取一个字节。
*   `eeprom_write_byte(uint32_t pos, uint8_t value)` 函数将一个字节写入到 RETRAM 中指定的地址 `pos`。 由于 RETRAM 可以直接读写，所以写入操作非常简单，不需要擦除。

**简易 Demo (中文):**

```c
int main(void) {
    HAL_Init(); // 初始化 HAL 库

    // 设置需要写入的数据
    uint32_t address = 0; // 从 RETRAM 起始位置开始
    uint8_t data = 0xA5;

    // 写入数据到 RETRAM
    eeprom_write_byte(address, data);

    // 从 RETRAM 中读取数据
    uint8_t read_data = eeprom_read_byte(address);

    // 验证数据
    if (read_data == data) {
        // 数据写入和读取成功
        // 可以点亮 LED 或者通过串口输出信息
    } else {
        // 数据写入或读取失败
        // 处理错误
    }

    while (1) {
        // 循环
    }
}
```

**重要注意事项 (中文):**

*   **STM32F103:**  模拟 EEPROM 使用 Flash memory。Flash memory 的写入次数有限制 (通常是 10,000 次)。频繁的写入会缩短 Flash 的寿命。 尽量避免频繁地写入同一个地址。可以使用 wear leveling 算法来分散写入。
*   **STM32MP1xx:**  模拟 EEPROM 使用 RETRAM。RETRAM 在 VBAT 供电的情况下可以保存数据。 但请注意，在没有 VBAT 供电的情况下，RETRAM 中的数据会丢失。  与 Flash memory 不同，RETRAM 可以直接读写，无需擦除，因此写入速度更快，寿命更长。但是，RETRAM 的容量通常比 Flash memory 小。
*   **错误处理:**  在 `eeprom_write_byte` 函数中，需要处理 Flash 写入错误的情况。 例如，如果 Flash 写入失败，可以尝试重新写入，或者通过串口输出错误信息。
*   **中断:** 在进行 Flash 写入操作时，应该禁用中断，以防止数据损坏。
*   **地址对齐:**  在某些情况下，Flash 写入需要地址对齐。  例如，某些 Flash memory 只能以字 (word) 为单位写入。  如果需要写入的数据不是字对齐的，需要进行额外的处理。

在实际应用中，你需要根据你的具体需求选择合适的 EEPROM 模拟方案，并进行充分的测试。
