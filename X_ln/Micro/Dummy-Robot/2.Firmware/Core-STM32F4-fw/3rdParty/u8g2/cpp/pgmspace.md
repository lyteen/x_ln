Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/3rdParty\u8g2\cpp\pgmspace.h`

好的，重新开始。

This code defines a set of macros and typedefs for accessing data stored in program memory (PROGMEM) on AVR microcontrollers, commonly used with Arduino.  PROGMEM is a special memory region that is used to store constants that would otherwise take up valuable SRAM.  This is especially important on microcontrollers with limited RAM, such as the Arduino Uno. This header file makes it easier to work with data in PROGMEM by providing convenient functions and type definitions.

Here's a breakdown of the key parts:

**1. Type Definitions (类型定义)**

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

*   **Description (描述):** These `typedef` statements define new names (aliases) for existing data types.  The `prog_` prefix indicates that these types are intended to be used when working with data stored in program memory.  For example, `prog_char` is a character stored in program memory. These are important because the compiler treats pointers to data in PROGMEM differently than pointers to data in SRAM.
*   **Use Case (用法):** When declaring variables that point to data in PROGMEM, use these types.
*   **Example (例子):**
    ```c++
    const char myString[] PROGMEM = "Hello, world!";
    prog_char *p = myString; // Declare a pointer to a character in PROGMEM
    ```

**2. PROGMEM Macro (PROGMEM 宏)**

```c++
#define PROGMEM
```

*   **Description (描述):** This macro is used to tell the compiler to store a variable in program memory (flash memory) instead of SRAM.  On AVR microcontrollers, SRAM is a limited resource, so storing constant data in PROGMEM frees up SRAM for other variables.
*   **Use Case (用法):**  Place `PROGMEM` after the data type in a variable declaration.
*   **Example (例子):**
    ```c++
    const char message[] PROGMEM = "This is stored in PROGMEM";
    ```

**3. PGM_P and PGM_VOID_P Macros (PGM_P 和 PGM_VOID_P 宏)**

```c++
#define PGM_P         const char *
#define PGM_VOID_P    const void *
```

*   **Description (描述):** These macros define types for pointers to character strings and generic data in PROGMEM, respectively. They're essentially creating a more descriptive alias for `const char*` and `const void*`.
*   **Use Case (用法):**  Use these types when declaring pointers that will point to strings or other data stored in PROGMEM.
*   **Example (例子):**
    ```c++
    const char myString[] PROGMEM = "My string in PROGMEM";
    PGM_P stringPointer = myString; // Pointer to a string in PROGMEM
    ```

**4. PSTR Macro (PSTR 宏)**

```c++
#define PSTR(s)       (s)
```

*   **Description (描述):**  This macro is a shorthand way to define a string literal that is automatically stored in program memory. It's a convenience macro, especially useful when passing strings directly to functions that expect PGM_P pointers.

*   **Use Case (用法):** Enclose a string literal within `PSTR()` to store it in PROGMEM.

*   **Example (例子):**

    ```c++
    void printStringFromPROGMEM(PGM_P str) {
      // ... code to read and print the string from PROGMEM ...
    }

    void setup() {
      Serial.begin(9600);
      printStringFromPROGMEM(PSTR("Hello from PROGMEM!"));
    }
    ```

**5. _SFR_BYTE Macro (_SFR_BYTE 宏)**

```c++
#define _SFR_BYTE(n)  (n)
```

*   **Description (描述):**  This macro is likely related to accessing Special Function Registers (SFRs) in the microcontroller.  However, in this implementation, it's simply defined as passing the argument `n` through unchanged.  This suggests it's either a placeholder or its functionality is handled elsewhere in the environment.  It's often used for accessing hardware registers.
*   **Use Case (用法):** Typically used for interacting directly with microcontroller hardware.  Without more context, it's hard to say how this is intended to be used in this specific library.
*   **Example (例子):** The specific use would depend on the intended hardware interaction.

**6. pgm_read Functions (pgm_read 函数)**

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
```

*   **Description (描述):** These macros provide a way to read data of different types (byte, word, dword, float, pointer) from program memory.  They take an address as input and return the value stored at that address.  These macros are crucial for accessing data declared with `PROGMEM`.  The `typeof(addr) _addr = (addr);` part is important for type safety. It creates a temporary variable `_addr` of the same type as `addr` to avoid potential issues with macro expansion.
*   **Use Case (用法):** Use these functions to retrieve data from variables stored in PROGMEM.
*   **Example (例子):**
    ```c++
    const unsigned short values[] PROGMEM = {10, 20, 30, 40};

    void setup() {
      Serial.begin(9600);
      unsigned short value = pgm_read_word(&values[1]); // Read the second value (20)
      Serial.println(value);
    }
    ```

**7. pgm_get_far_address (pgm_get_far_address)**

```c++
#define pgm_get_far_address(x) ((uint32_t)(&(x)))
```

*   **Description (描述):** This macro returns the 32-bit address of a variable `x`. This is important because AVR microcontrollers have different memory addressing schemes.
*   **Use Case (用法):**  It is used when you need the absolute address of a variable located in PROGMEM, especially when working with functions or libraries that require 32-bit addresses.
*   **Example (例子):**
    ```c++
    const char myString[] PROGMEM = "My string";
    uint32_t address = pgm_get_far_address(myString);
    // Now you can use 'address' with functions requiring a 32-bit address.
    ```

**8. pgm_read_near and pgm_read_far (pgm_read_near 和 pgm_read_far)**

```c++
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

*   **Description (描述):** These macros are likely included for compatibility reasons or to provide a distinction between near and far pointers, which might be relevant on some AVR microcontrollers with larger memory spaces. However, in this specific implementation, both `_near` and `_far` versions simply call the base `pgm_read` functions.  This suggests that the memory model being used doesn't require explicit near/far pointer handling.
*   **Use Case (用法):**  In practice, you can use either the `_near` or `_far` versions, or the base `pgm_read` functions, and they will all behave the same way.
*   **Example (例子):**
    ```c++
    const int myValues[] PROGMEM = {1, 2, 3};
    int valueNear = pgm_read_word_near(&myValues[0]);
    int valueFar = pgm_read_word_far(&myValues[1]);

    Serial.print("Near: "); Serial.println(valueNear); // Prints 1
    Serial.print("Far: "); Serial.println(valueFar);   // Prints 2
    ```

**9. String and Memory Functions (字符串和内存函数)**

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

*   **Description (描述):**  These macros provide aliases for standard C string and memory manipulation functions (like `strcpy`, `strlen`, `memcpy`, etc.).  The `_P` suffix indicates that these versions are intended to be used with strings or data stored in PROGMEM. **However**, in this specific implementation, they are simply aliased to the standard C functions.  This is problematic and potentially incorrect! Standard C functions are designed to work with data in SRAM, *not* PROGMEM.  To correctly work with PROGMEM strings, you need to use special versions of these functions that are designed to read from PROGMEM (e.g., `strcpy_P` should use `pgm_read_byte` internally to fetch characters from PROGMEM).
*   **Use Case (用法):** These functions *should* be used to manipulate strings and data stored in PROGMEM. However, because of the incorrect aliasing, they will not work correctly.  You would need to provide your own implementations or use a proper PROGMEM-aware library.
*   **Example (例子):**  This example will likely fail to produce the correct result.
    ```c++
    #include <string.h> // For strlen

    const char myString[] PROGMEM = "Hello";
    char buffer[10];

    void setup() {
      Serial.begin(9600);
      // Incorrect! This will likely copy garbage data.
      strcpy_P(buffer, myString);
      Serial.println(buffer);
    }
    ```

**Important Considerations and Corrections:**

*   **Incorrect String/Memory Function Aliasing:**  The biggest issue with this code is the aliasing of `*_P` functions to the standard C library functions.  This is fundamentally wrong. When working with strings in PROGMEM, you *must* use functions that are specifically designed to read from PROGMEM (using `pgm_read_byte`, etc.).  The standard C functions assume data is in SRAM and will not be able to access PROGMEM data correctly.  This will lead to unpredictable behavior, crashes, or garbage data.

*   **Correct PROGMEM String Handling:** To correctly handle strings in PROGMEM, you need to use the `pgm_read_byte` function to read characters one at a time and implement your own versions of string functions or use a library that provides proper PROGMEM support.

*   **Missing PROGMEM-Aware String Functions:**  A proper PROGMEM library should include functions like `strcpy_P`, `strlen_P`, etc., that are specifically written to read data from program memory.  These functions would use `pgm_read_byte` (or `pgm_read_word` etc., as appropriate) internally to access the string data.

**Corrected Example (with a Basic `strlen_P` Implementation):**

```c++
#include <Arduino.h> // For Serial

const char myString[] PROGMEM = "Hello, PROGMEM!";

// Correct implementation of strlen_P
size_t strlen_P(const char *str) {
  size_t len = 0;
  while (pgm_read_byte(str + len) != 0) {
    len++;
  }
  return len;
}

void setup() {
  Serial.begin(9600);
  delay(100); // Allow serial to initialize

  size_t length = strlen_P(myString);
  Serial.print("Length of string in PROGMEM: ");
  Serial.println(length);

  // Example of reading and printing characters individually
  Serial.print("String: ");
  for (size_t i = 0; i < length; i++) {
    Serial.print((char)pgm_read_byte(myString + i));
  }
  Serial.println();
}

void loop() {
  // put your main code here, to run repeatedly:
}
```

**Explanation of the Corrected Example:**

1.  **`strlen_P` Implementation:**  The corrected example includes a proper implementation of `strlen_P`.  It uses `pgm_read_byte` to read characters from PROGMEM one by one until it encounters a null terminator (0).

2.  **Reading and Printing Characters:**  The example also demonstrates how to read and print individual characters from PROGMEM using `pgm_read_byte`.  This is the fundamental way to access data stored in PROGMEM.

3. **`Arduino.h` include:** The example now includes Arduino.h because the `Serial` object is defined there and used for printing to the serial monitor.

**In summary,** this header file provides tools for working with PROGMEM on AVR microcontrollers. The `pgm_read` functions are crucial for accessing data in PROGMEM.  However, the string/memory function aliasing is incorrect and needs to be addressed by either using a proper PROGMEM-aware library or implementing your own PROGMEM-aware versions of those functions. The provided `strlen_P` example demonstrates the correct approach.
