Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/3rdParty\u8g2\cpp\pgmspace.h`

好的，我们重新开始。

这段C++代码定义了用于在Arduino环境中使用程序空间（Program Space）的宏和类型定义。程序空间通常指的是闪存，用于存储程序代码和只读数据。在资源有限的嵌入式系统中，有效地使用程序空间非常重要。

以下是对代码的逐段解释，并附带一些使用示例，示例会侧重于嵌入式环境中，尤其是在 Arduino 中常见的应用场景。

**1. 类型定义**

```c++
typedef void prog_void;
typedef char prog_char;
typedef unsigned char prog_uchar;
typedef char prog_int8_t;
typedef unsigned char prog_uint8_t;
typedef short prog_int16_t;
typedef unsigned short prog_uint16_t;
typedef long prog_int32_t;
typedef unsigned long prog_uint32_t;
```

*   **描述:**  这些 `typedef` 语句定义了与程序空间相关的类型。 使用这些类型可以明确地指示变量或指针指向程序空间中的数据。例如，`prog_char` 表示指向程序空间中字符的指针。
*   **为什么重要:**  在某些架构（如AVR，用于Arduino）上，从闪存读取数据需要特殊的指令。使用这些类型可以让编译器知道数据位于闪存中，并生成正确的代码。
*   **中文解释:** 这些是类型的别名，`prog_char` 其实就是 `char` 类型，但是编译器会知道，这个 `char` 变量是存储在 Flash 里面的。

**2. 宏定义**

```c++
#define PROGMEM
#define PGM_P         const char *
#define PGM_VOID_P    const void *
#define PSTR(s)       (s)
#define _SFR_BYTE(n)  (n)
```

*   **`PROGMEM`:**  这是一个空宏。  在AVR Arduino中，通常用于指示变量应存储在闪存中。  这个宏在 Raspberry Pi 核心中可能没有实际效果，但保留它是为了兼容现有的Arduino代码。
*   **`PGM_P` 和 `PGM_VOID_P`:**  这些宏定义了指向程序空间中字符和 `void` 数据的常量指针类型。  它们等效于 `const char*` 和 `const void*`。
*   **`PSTR(s)`:**  这个宏接受一个字符串字面量 `s`，并将其标记为存储在闪存中。在AVR Arduino中，这会将字符串字面量存储在闪存中，而不是SRAM。
*   **`_SFR_BYTE(n)`:**  这个宏用于访问特殊功能寄存器（SFR）。 在 Raspberry Pi 核心中，它只是简单地返回 `n`。在AVR Arduino中，它会被编译器识别并用于生成访问SFR的特殊指令。
*   **中文解释:**
    *   `PROGMEM`: 一个空宏，为了兼容现有的 Arduino 代码。
    *   `PGM_P`: `const char*` 的别名，用于指向 Flash 中的字符串。
    *   `PGM_VOID_P`: `const void*` 的别名，用于指向 Flash 中的任意数据。
    *   `PSTR(s)`:  告诉编译器，这个字符串 `s` 应该存储到 Flash 里面。
    *   `_SFR_BYTE(n)`: 用于访问特殊功能寄存器 (SFR)，可以忽略。

**3. 读取程序空间数据的宏**

```c++
#define pgm_read_byte(addr)   (*(const unsigned char *)(addr))
#define pgm_read_word(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(const unsigned short *)(_addr); \
})
#define pgm_read_dword(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(const unsigned long *)(_addr); \
})
#define pgm_read_float(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(const float *)(_addr); \
})
#define pgm_read_ptr(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(void * const *)(_addr); \
})

#define pgm_get_far_address(x) ((uint32_t)(&(x)))

#define pgm_read_byte_near(addr)  pgm_read_byte(addr)
#define pgm_read_word_near(addr)  pgm_read_word(addr)
#define pgm_read_dword_near(addr) pgm_read_dword(addr)
#define pgm_read_float_near(addr) pgm_read_float(addr)
#define pgm_read_ptr_near(addr)   pgm_read_ptr(addr)
#define pgm_read_byte_far(addr)   pgm_read_byte(addr)
#define pgm_read_word_far(addr)   pgm_read_word(addr)
#define pgm_read_dword_far(addr)  pgm_read_dword(addr)
#define pgm_read_float_far(addr)  pgm_read_float(addr)
#define pgm_read_ptr_far(addr)    pgm_read_ptr(addr)
```

*   **描述:** 这些宏定义了用于从程序空间读取数据的函数。 它们接受一个地址 `addr` 作为参数，并返回该地址处的值。 这些宏使用类型转换来确保读取的数据类型正确。 `near` 和 `far` 变体是为了兼容不同的内存模型而存在的，但在 Raspberry Pi 核心中，它们实际上是相同的。
*   **为什么重要:** 在AVR Arduino中，从闪存读取数据需要使用特殊的指令，例如 `pgm_read_byte()`。 这些宏封装了这些指令，使代码更具可读性和可移植性。
*   **`pgm_get_far_address(x)`:** 获取变量 `x` 在程序空间中的地址。
*   **中文解释:**
    *   `pgm_read_byte(addr)`: 从地址 `addr` 读取一个字节。
    *   `pgm_read_word(addr)`: 从地址 `addr` 读取一个字（2个字节）。
    *   `pgm_read_dword(addr)`: 从地址 `addr` 读取一个双字（4个字节）。
    *   `pgm_read_float(addr)`: 从地址 `addr` 读取一个 `float`。
    *   `pgm_read_ptr(addr)`: 从地址 `addr` 读取一个指针。
    *   `pgm_get_far_address(x)`: 获取变量 `x` 在 Flash 中的地址。
    *   `pgm_read_..._near(addr)` 和 `pgm_read_..._far(addr)`: `near` 和 `far` 在这里没有区别，只是为了兼容性。

**4. 字符串操作宏**

```c++
#define memcmp_P      memcmp
#define memccpy_P     memccpy
#define memmem_P      memmem
#define memcpy_P      memcpy
#define strcpy_P      strcpy
#define strncpy_P     strncpy
#define strcat_P      strcat
#define strncat_P     strncat
#define strcmp_P      strcmp
#define strncmp_P     strncmp
#define strcasecmp_P  strcasecmp
#define strncasecmp_P strncasecmp
#define strlen_P      strlen
#define strnlen_P     strnlen
#define strstr_P      strstr
#define printf_P      printf
#define sprintf_P     sprintf
#define snprintf_P    snprintf
#define vsnprintf_P   vsnprintf
```

*   **描述:** 这些宏定义了与标准C库函数对应的函数，用于操作存储在程序空间中的字符串。  例如，`strcpy_P` 用于将程序空间中的字符串复制到SRAM中。
*   **为什么重要:** 在AVR Arduino中，直接使用标准C库函数操作程序空间中的字符串可能会导致问题，因为标准C库函数通常假设字符串存储在SRAM中。使用这些宏可以确保字符串操作正确执行。
*   **中文解释:** 这些宏将标准的字符串操作函数（例如 `strcpy`, `strlen`）重命名为 `strcpy_P`, `strlen_P`。  `_P` 后缀表示这些函数用于操作存储在 Flash 中的字符串。

**示例代码**

下面是一个简单的示例，演示如何在Arduino环境中使用这些宏定义：

```c++
#include <stdio.h>
#include <string.h>

#ifndef PGMSPACE_INCLUDE
#define PGMSPACE_INCLUDE

typedef void prog_void;
typedef char prog_char;
typedef unsigned char prog_uchar;
typedef char prog_int8_t;
typedef unsigned char prog_uint8_t;
typedef short prog_int16_t;
typedef unsigned short prog_uint16_t;
typedef long prog_int32_t;
typedef unsigned long prog_uint32_t;

#define PROGMEM
#define PGM_P         const char *
#define PGM_VOID_P    const void *
#define PSTR(s)       (s)
#define _SFR_BYTE(n)  (n)

#define pgm_read_byte(addr)   (*(const unsigned char *)(addr))
#define pgm_read_word(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(const unsigned short *)(_addr); \
})
#define pgm_read_dword(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(const unsigned long *)(_addr); \
})
#define pgm_read_float(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(const float *)(addr); \
})
#define pgm_read_ptr(addr) ({ \
  typeof(addr) _addr = (addr); \
  *(void * const *)(_addr); \
})

#define pgm_get_far_address(x) ((uint32_t)(&(x)))

#define pgm_read_byte_near(addr)  pgm_read_byte(addr)
#define pgm_read_word_near(addr)  pgm_read_word(addr)
#define pgm_read_dword_near(addr) pgm_read_dword(addr)
#define pgm_read_float_near(addr) pgm_read_float(addr)
#define pgm_read_ptr_near(addr)   pgm_read_ptr(addr)
#define pgm_read_byte_far(addr)   pgm_read_byte(addr)
#define pgm_read_word_far(addr)   pgm_read_word(addr)
#define pgm_read_dword_far(addr)  pgm_read_dword(addr)
#define pgm_read_float_far(addr)  pgm_read_float(addr)
#define pgm_read_ptr_far(addr)    pgm_read_ptr(addr)

#define memcmp_P      memcmp
#define memccpy_P     memccpy
#define memmem_P      memmem
#define memcpy_P      memcpy
#define strcpy_P      strcpy
#define strncpy_P     strncpy
#define strcat_P      strcat
#define strncat_P     strncat
#define strcmp_P      strcmp
#define strncmp_P     strncmp
#define strcasecmp_P  strcasecmp
#define strncasecmp_P strncasecmp
#define strlen_P      strlen
#define strnlen_P     strnlen
#define strstr_P      strstr
#define printf_P      printf
#define sprintf_P     sprintf
#define snprintf_P    snprintf
#define vsnprintf_P   vsnprintf

#endif

const char string_PROGMEM[] PROGMEM = "Hello from Flash!"; // 将字符串存储在闪存中

void setup() {
  Serial.begin(9600);
  while (!Serial); // 等待串口连接

  char buffer[32]; // 用于存储从闪存读取的字符串

  // 将闪存中的字符串复制到SRAM中
  strcpy_P(buffer, string_PROGMEM);

  // 通过串口打印字符串
  Serial.println(buffer);

  // 获取字符串的长度
  int len = strlen_P(string_PROGMEM);
  Serial.print("String length: ");
  Serial.println(len);

  // 从 Flash 中读取单个字符
  char first_char = pgm_read_byte(&string_PROGMEM[0]);
    Serial.print("First char: ");
  Serial.println(first_char);

}

void loop() {
  // 不做任何事情
}
```

*   **描述:**  此代码首先将一个字符串 "Hello from Flash!" 存储在闪存中，使用 `PROGMEM` 属性。 然后，在 `setup()` 函数中，它将字符串从闪存复制到SRAM中，并通过串口打印出来。它还演示了如何使用 `strlen_P` 获取字符串的长度，以及如何使用 `pgm_read_byte` 从 Flash 中读取单个字符。
*   **中文解释:**
    1.  **`const char string_PROGMEM[] PROGMEM = "Hello from Flash!";`**: 定义一个字符串 `string_PROGMEM`，并使用 `PROGMEM` 关键字将其存储到 Flash 里面。  `const` 关键字表示这个字符串是只读的。
    2.  **`char buffer[32];`**: 定义一个缓冲区 `buffer`，用于存储从 Flash 读取到的字符串。  这个缓冲区存储在 SRAM 里面。
    3.  **`strcpy_P(buffer, string_PROGMEM);`**: 使用 `strcpy_P` 函数，将 Flash 中的 `string_PROGMEM` 复制到 SRAM 中的 `buffer` 里面。
    4.  **`Serial.println(buffer);`**: 通过串口打印 `buffer` 中的内容。
    5.  **`int len = strlen_P(string_PROGMEM);`**: 使用 `strlen_P` 函数获取 Flash 中 `string_PROGMEM` 的长度。
    6.   **`char first_char = pgm_read_byte(&string_PROGMEM[0]);`**: 从 Flash 中读取 `string_PROGMEM` 的第一个字符。

**总结**

这段代码定义了一组用于在Arduino环境中使用程序空间的宏和类型定义。这些宏和类型定义可以帮助开发者更有效地使用闪存，并编写更具可读性和可移植性的代码。虽然部分功能在 Raspberry Pi 核心中可能没有实际作用，但保留它们是为了兼容现有的Arduino代码。 理解这些宏和类型定义对于开发需要在资源受限的嵌入式系统中运行的Arduino程序至关重要。  在 Raspberry Pi 核心中，这些宏可能主要为了代码兼容性而存在。 如果直接在 Raspberry Pi 上进行开发，可能不需要像在 AVR Arduino 中那样特别关注程序空间。
