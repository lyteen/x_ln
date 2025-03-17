Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/3rdParty\u8g2\cpp\WString.h`

好的，我们重新开始。 这次我将更详细地解释代码，并将代码分解成更小的片段，并附上中文描述和示例。

```c++
/*
 WString.h - String library for Wiring & Arduino
 ...mostly rewritten by Paul Stoffregen...
 Copyright (c) 2009-10 Hernando Barragan.  All right reserved.
 Copyright 2011, Paul Stoffregen, paul@pjrc.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef String_class_h
#define String_class_h
#ifdef __cplusplus

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

// An inherited class for holding the result of a concatenation.  These
// result objects are assumed to be writable by subsequent concatenations.
class StringSumHelper;

// an abstract class used as a means to proide a unique pointer type
// but really has no body
class __FlashStringHelper;
#define FPSTR(pstr_pointer) (reinterpret_cast<const __FlashStringHelper *>(pstr_pointer))
#define F(string_literal) (FPSTR(PSTR(string_literal)))
```

**代码片段解释:**

*   `#ifndef String_class_h`, `#define String_class_h`, `#ifdef __cplusplus`:  这些是头文件保护，确保 `String_class.h` 只被包含一次。`__cplusplus`宏用于判断是否是c++环境。
*   `#include <stdlib.h>`, `#include <string.h>`, `#include <ctype.h>`, `#include <stdint.h>`: 包含标准C库的头文件，提供内存分配、字符串操作、字符类型判断和整数类型定义等功能。
*   `class StringSumHelper;`: 声明 `StringSumHelper` 类，用于字符串连接的辅助类。
*   `class __FlashStringHelper;`: 声明 `__FlashStringHelper` 类，用于处理存储在Flash中的字符串。在Arduino中，Flash存储器用于存储程序代码和常量数据，包括字符串字面量，使用`__FlashStringHelper`可以节省RAM。
*   `#define FPSTR(pstr_pointer) (reinterpret_cast<const __FlashStringHelper *>(pstr_pointer))`:  `FPSTR`宏将指针强制转换为指向`__FlashStringHelper`的指针。
*   `#define F(string_literal) (FPSTR(PSTR(string_literal)))`: `F` 宏用于将字符串字面量包装成 `__FlashStringHelper` 类型，告诉编译器将该字符串存储在 Flash 存储器中，而不是 RAM 中。`PSTR`宏也是Arduino提供的，用于将字符串字面量放置在Flash存储器中。

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);
  String myString = F("Hello from Flash!"); // 将字符串存储在 Flash 中
  Serial.println(myString);
}

void loop() {
  // do nothing
}
```

这段代码将 "Hello from Flash!" 字符串存储在 Flash 存储器中，而不是 RAM 中，从而节省了 RAM 空间。

```c++
// The string class
class String {
        // use a function pointer to allow for "if (s)" without the
        // complications of an operator bool(). for more information, see:
        // http://www.artima.com/cppsource/safebool.html
        typedef void (String::*StringIfHelperType)() const;
        void StringIfHelper() const {
        }

    public:
        // constructors
        // creates a copy of the initial value.
        // if the initial value is null or invalid, or if memory allocation
        // fails, the string will be marked as invalid (i.e. "if (s)" will
        // be false).
        String(const char *cstr = "");
        String(const String &str);
        String(const __FlashStringHelper *str);
#ifdef __GXX_EXPERIMENTAL_CXX0X__
        String(String &&rval);
        String(StringSumHelper &&rval);
#endif
        explicit String(char c);
        explicit String(unsigned char, unsigned char base = 10);
        explicit String(int, unsigned char base = 10);
        explicit String(unsigned int, unsigned char base = 10);
        explicit String(long, unsigned char base = 10);
        explicit String(unsigned long, unsigned char base = 10);
        explicit String(float, unsigned char decimalPlaces = 2);
        explicit String(double, unsigned char decimalPlaces = 2);
        ~String(void);
```

**代码片段解释:**

*   `class String { ... }`: 定义 `String` 类，这是核心的字符串类。
*   `typedef void (String::*StringIfHelperType)() const;`: 定义一个函数指针类型 `StringIfHelperType`，指向 `String` 类的 `const` 成员函数，该函数没有参数，返回 `void`。
*   `void StringIfHelper() const { }`:  定义一个空的成员函数 `StringIfHelper`，用于实现 "safe bool" 惯用法。这个技巧允许像 `if (myString)` 这样使用 `String` 对象进行条件判断，而避免隐式类型转换可能带来的问题。
*   `String(const char *cstr = "");`: 构造函数，从 C 风格字符串创建 `String` 对象。默认参数 "" 允许创建一个空字符串。
*   `String(const String &str);`: 构造函数，从另一个 `String` 对象创建 `String` 对象 (拷贝构造函数)。
*   `String(const __FlashStringHelper *str);`: 构造函数，从 Flash 存储器中的字符串创建 `String` 对象。
*   `#ifdef __GXX_EXPERIMENTAL_CXX0X__ ... #endif`:  条件编译，如果编译器支持 C++11 的移动语义，则定义移动构造函数。
*   `explicit String(char c);`: 构造函数，从单个字符创建 `String` 对象。`explicit` 关键字防止隐式类型转换。
*   `explicit String(unsigned char, unsigned char base = 10);`: 构造函数，从无符号字符创建 `String` 对象，并可以指定进制。
*   `explicit String(int, unsigned char base = 10);`: 构造函数，从整数创建 `String` 对象，并可以指定进制。
*   `explicit String(unsigned int, unsigned char base = 10);`: 构造函数，从无符号整数创建 `String` 对象，并可以指定进制。
*   `explicit String(long, unsigned char base = 10);`: 构造函数，从长整数创建 `String` 对象，并可以指定进制。
*   `explicit String(unsigned long, unsigned char base = 10);`: 构造函数，从无符号长整数创建 `String` 对象，并可以指定进制。
*   `explicit String(float, unsigned char decimalPlaces = 2);`: 构造函数，从浮点数创建 `String` 对象，并可以指定小数位数。
*   `explicit String(double, unsigned char decimalPlaces = 2);`: 构造函数，从双精度浮点数创建 `String` 对象，并可以指定小数位数。
*   `~String(void);`: 析构函数，释放 `String` 对象占用的内存。

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);

  String str1 = "Hello"; // 从 C 风格字符串创建
  String str2(str1);    // 拷贝构造
  String str3 = F("World"); // 从 Flash 字符串创建
  String str4('!');      // 从字符创建
  String str5 = String(123, DEC); // 从整数创建，十进制
  String str6 = String(45.67, 2);  // 从浮点数创建，保留两位小数

  Serial.println(str1); // 输出 Hello
  Serial.println(str2); // 输出 Hello
  Serial.println(str3); // 输出 World
  Serial.println(str4); // 输出 !
  Serial.println(str5); // 输出 123
  Serial.println(str6); // 输出 45.67
}

void loop() {
  // do nothing
}
```

```c++
        // memory management
        // return true on success, false on failure (in which case, the string
        // is left unchanged).  reserve(0), if successful, will validate an
        // invalid string (i.e., "if (s)" will be true afterwards)
        unsigned char reserve(unsigned int size);
        inline unsigned int length(void) const {
            if(buffer()) {
                return len();
            } else {
                return 0;
            }
        }
        inline void clear(void) {
            setLen(0);
        }
        inline bool isEmpty(void) const {
            return length() == 0;
        }
```

**代码片段解释:**

*   `unsigned char reserve(unsigned int size);`:  `reserve` 函数用于预先分配 `String` 对象所需的内存空间。如果分配成功，返回 `true`，否则返回 `false`。`reserve(0)` 可以用于验证一个无效的字符串（即，将 `String` 对象标记为有效，如果分配成功）。
*   `inline unsigned int length(void) const { ... }`:  `length` 函数返回 `String` 对象中字符串的长度（不包括 null 终止符）。如果字符串无效（`buffer()` 返回 `NULL`），则返回 0。`inline` 关键字建议编译器将该函数内联展开，以提高性能。
*   `inline void clear(void) { ... }`:  `clear` 函数清空 `String` 对象，将其长度设置为 0。
*   `inline bool isEmpty(void) const { ... }`:  `isEmpty` 函数检查 `String` 对象是否为空，如果长度为 0，则返回 `true`，否则返回 `false`。

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);

  String myString = "Hello";
  Serial.print("Length: ");
  Serial.println(myString.length()); // 输出 Length: 5

  myString.clear();
  Serial.print("Is empty: ");
  Serial.println(myString.isEmpty()); // 输出 Is empty: 1 (true)

  if (myString.reserve(10)) {
    Serial.println("Memory reserved successfully.");
  } else {
    Serial.println("Memory reservation failed.");
  }
}

void loop() {
  // do nothing
}
```

```c++
        // creates a copy of the assigned value.  if the value is null or
        // invalid, or if the memory allocation fails, the string will be
        // marked as invalid ("if (s)" will be false).
        String & operator =(const String &rhs);
        String & operator =(const char *cstr);
        String & operator = (const __FlashStringHelper *str);
#ifdef __GXX_EXPERIMENTAL_CXX0X__
        String & operator =(String &&rval);
        String & operator =(StringSumHelper &&rval);
#endif

        // concatenate (works w/ built-in types)

        // returns true on success, false on failure (in which case, the string
        // is left unchanged).  if the argument is null or invalid, the
        // concatenation is considered unsuccessful.
        unsigned char concat(const String &str);
        unsigned char concat(const char *cstr);
        unsigned char concat(char c);
        unsigned char concat(unsigned char c);
        unsigned char concat(int num);
        unsigned char concat(unsigned int num);
        unsigned char concat(long num);
        unsigned char concat(unsigned long num);
        unsigned char concat(float num);
        unsigned char concat(double num);
        unsigned char concat(const __FlashStringHelper * str);

        // if there's not enough memory for the concatenated value, the string
        // will be left unchanged (but this isn't signalled in any way)
        String & operator +=(const String &rhs) {
            concat(rhs);
            return (*this);
        }
        String & operator +=(const char *cstr) {
            concat(cstr);
            return (*this);
        }
        String & operator +=(char c) {
            concat(c);
            return (*this);
        }
        String & operator +=(unsigned char num) {
            concat(num);
            return (*this);
        }
        String & operator +=(int num) {
            concat(num);
            return (*this);
        }
        String & operator +=(unsigned int num) {
            concat(num);
            return (*this);
        }
        String & operator +=(long num) {
            concat(num);
            return (*this);
        }
        String & operator +=(unsigned long num) {
            concat(num);
            return (*this);
        }
        String & operator +=(float num) {
            concat(num);
            return (*this);
        }
        String & operator +=(double num) {
            concat(num);
            return (*this);
        }
        String & operator += (const __FlashStringHelper *str){
            concat(str);
            return (*this);
        }
```

**代码片段解释:**

*   `String & operator =(const String &rhs);`: 赋值运算符，将一个 `String` 对象赋值给另一个 `String` 对象 (拷贝赋值)。
*   `String & operator =(const char *cstr);`: 赋值运算符，将 C 风格字符串赋值给 `String` 对象。
*   `String & operator = (const __FlashStringHelper *str);`: 赋值运算符，将 Flash 存储器中的字符串赋值给 `String` 对象。
*   `#ifdef __GXX_EXPERIMENTAL_CXX0X__ ... #endif`:  条件编译，如果编译器支持 C++11 的移动语义，则定义移动赋值运算符。
*   `unsigned char concat(const String &str);`:  `concat` 函数将一个 `String` 对象连接到当前 `String` 对象的末尾。如果连接成功，返回 `true`，否则返回 `false`。
*   `unsigned char concat(const char *cstr);`:  `concat` 函数将 C 风格字符串连接到当前 `String` 对象的末尾。如果连接成功，返回 `true`，否则返回 `false`。
*   `unsigned char concat(char c);`:  `concat` 函数将一个字符连接到当前 `String` 对象的末尾。如果连接成功，返回 `true`，否则返回 `false`。
*   `unsigned char concat(unsigned char c);`: `concat` 函数将一个无符号字符连接到当前 `String` 对象的末尾。
*   `unsigned char concat(int num);`: `concat` 函数将一个整数连接到当前 `String` 对象的末尾。
*   `unsigned char concat(unsigned int num);`: `concat` 函数将一个无符号整数连接到当前 `String` 对象的末尾。
*   `unsigned char concat(long num);`: `concat` 函数将一个长整数连接到当前 `String` 对象的末尾。
*   `unsigned char concat(unsigned long num);`: `concat` 函数将一个无符号长整数连接到当前 `String` 对象的末尾。
*   `unsigned char concat(float num);`: `concat` 函数将一个浮点数连接到当前 `String` 对象的末尾。
*   `unsigned char concat(double num);`: `concat` 函数将一个双精度浮点数连接到当前 `String` 对象的末尾。
*   `unsigned char concat(const __FlashStringHelper * str);`:  `concat` 函数将 Flash 存储器中的字符串连接到当前 `String` 对象的末尾。如果连接成功，返回 `true`，否则返回 `false`。
*   `String & operator +=(const String &rhs) { ... }`:  `+=` 运算符重载，用于将一个 `String` 对象连接到当前 `String` 对象的末尾。
*   `String & operator +=(const char *cstr) { ... }`:  `+=` 运算符重载，用于将 C 风格字符串连接到当前 `String` 对象的末尾。
*   `String & operator +=(char c) { ... }`:  `+=` 运算符重载，用于将一个字符连接到当前 `String` 对象的末尾。
*   `String & operator +=(unsigned char num) { ... }`:  `+=` 运算符重载，用于将一个无符号字符连接到当前 `String` 对象的末尾。
*   `String & operator +=(int num) { ... }`:  `+=` 运算符重载，用于将一个整数连接到当前 `String` 对象的末尾。
*   `String & operator +=(unsigned int num) { ... }`:  `+=` 运算符重载，用于将一个无符号整数连接到当前 `String` 对象的末尾。
*   `String & operator +=(long num) { ... }`:  `+=` 运算符重载，用于将一个长整数连接到当前 `String` 对象的末尾。
*   `String & operator +=(unsigned long num) { ... }`:  `+=` 运算符重载，用于将一个无符号长整数连接到当前 `String` 对象的末尾。
*   `String & operator +=(float num) { ... }`:  `+=` 运算符重载，用于将一个浮点数连接到当前 `String` 对象的末尾。
*   `String & operator +=(double num) { ... }`:  `+=` 运算符重载，用于将一个双精度浮点数连接到当前 `String` 对象的末尾。
*   `String & operator += (const __FlashStringHelper *str){ ... }`:  `+=` 运算符重载，用于将 Flash 存储器中的字符串连接到当前 `String` 对象的末尾。

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);

  String str1 = "Hello";
  String str2 = " World";
  String str3;

  str3 = str1; // 赋值
  Serial.println(str3); // 输出 Hello

  str3 += str2; // 连接
  Serial.println(str3); // 输出 Hello World

  str3 += '!'; // 连接字符
  Serial.println(str3); // 输出 Hello World!

  str3 += 123; // 连接整数
  Serial.println(str3); // 输出 Hello World!123

  str3 += F(" from Flash"); // 连接 Flash 字符串
  Serial.println(str3); // 输出 Hello World!123 from Flash
}

void loop() {
  // do nothing
}
```

```c++
        friend StringSumHelper & operator +(const StringSumHelper &lhs, const String &rhs);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, const char *cstr);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, char c);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, unsigned char num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, int num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, unsigned int num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, long num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, unsigned long num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, float num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, double num);
        friend StringSumHelper & operator +(const StringSumHelper &lhs, const __FlashStringHelper *rhs);
```

**代码片段解释:**

*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, const String &rhs);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和 `String` 对象。`friend` 关键字允许该函数访问 `String` 类的私有成员。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, const char *cstr);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和 C 风格字符串。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, char c);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和字符。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, unsigned char num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和无符号字符。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, int num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和整数。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, unsigned int num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和无符号整数。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, long num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和长整数。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, unsigned long num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和无符号长整数。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, float num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和浮点数。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, double num);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和双精度浮点数。
*   `friend StringSumHelper & operator +(const StringSumHelper &lhs, const __FlashStringHelper *rhs);`:  `+` 运算符重载，用于连接 `StringSumHelper` 对象和 Flash 存储器中的字符串。

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);

  String str1 = "Hello";
  StringSumHelper helper(str1);

  String str2 = helper + " World" + "!"; // 使用 + 运算符连接
  Serial.println(str2); // 输出 Hello World!

  String str3 = helper + 123 + F(" from Flash"); // 连接整数和 Flash 字符串
  Serial.println(str3); // 输出 Hello123 from Flash
}

void loop() {
  // do nothing
}
```

```c++
        // comparison (only works w/ Strings and "strings")
        operator StringIfHelperType() const {
            return buffer() ? &String::StringIfHelper : 0;
        }
        int compareTo(const String &s) const;
        unsigned char equals(const String &s) const;
        unsigned char equals(const char *cstr) const;
        unsigned char operator ==(const String &rhs) const {
            return equals(rhs);
        }
        unsigned char operator ==(const char *cstr) const {
            return equals(cstr);
        }
        unsigned char operator !=(const String &rhs) const {
            return !equals(rhs);
        }
        unsigned char operator !=(const char *cstr) const {
            return !equals(cstr);
        }
        unsigned char operator <(const String &rhs) const;
        unsigned char operator >(const String &rhs) const;
        unsigned char operator <=(const String &rhs) const;
        unsigned char operator >=(const String &rhs) const;
        unsigned char equalsIgnoreCase(const String &s) const;
        unsigned char equalsConstantTime(const String &s) const;
        unsigned char startsWith(const String &prefix) const;
        unsigned char startsWith(const char *prefix) const {
            return this->startsWith(String(prefix));
        }
        unsigned char startsWith(const __FlashStringHelper *prefix) const {
            return this->startsWith(String(prefix));
        }
        unsigned char startsWith(const String &prefix, unsigned int offset) const;
        unsigned char endsWith(const String &suffix) const;
        unsigned char endsWith(const char *suffix) const {
            return this->endsWith(String(suffix));
        }
        unsigned char endsWith(const __FlashStringHelper * suffix) const {
            return this->endsWith(String(suffix));
        }
```

**代码片段解释:**

*   `operator StringIfHelperType() const { ... }`: 重载类型转换运算符，将 `String` 对象转换为 `StringIfHelperType` 函数指针。用于实现 "safe bool" 惯用法，允许像 `if (myString)` 这样使用 `String` 对象进行条件判断。
*   `int compareTo(const String &s) const;`:  `compareTo` 函数比较当前 `String` 对象和另一个 `String` 对象。返回值为负数表示当前对象小于参数对象，返回值为正数表示当前对象大于参数对象，返回值为 0 表示两个对象相等。
*   `unsigned char equals(const String &s) const;`:  `equals` 函数比较当前 `String` 对象和另一个 `String` 对象是否相等。返回 `true` 表示相等，返回 `false` 表示不相等。
*   `unsigned char equals(const char *cstr) const;`:  `equals` 函数比较当前 `String` 对象和 C 风格字符串是否相等。返回 `true` 表示相等，返回 `false` 表示不相等。
*   `unsigned char operator ==(const String &rhs) const { ... }`:  `==` 运算符重载，用于比较两个 `String` 对象是否相等。
*   `unsigned char operator ==(const char *cstr) const { ... }`:  `==` 运算符重载，用于比较 `String` 对象和 C 风格字符串是否相等。
*   `unsigned char operator !=(const String &rhs) const { ... }`:  `!=` 运算符重载，用于比较两个 `String` 对象是否不相等。
*   `unsigned char operator !=(const char *cstr) const { ... }`:  `!=` 运算符重载，用于比较 `String` 对象和 C 风格字符串是否不相等。
*   `unsigned char operator <(const String &rhs) const;`:  `<` 运算符重载，用于比较两个 `String` 对象的大小。
*   `unsigned char operator >(const String &rhs) const;`:  `>` 运算符重载，用于比较两个 `String` 对象的大小。
*   `unsigned char operator <=(const String &rhs) const;`:  `<=` 运算符重载，用于比较两个 `String` 对象的大小。
*   `unsigned char operator >=(const String &rhs) const;`:  `>=` 运算符重载，用于比较两个 `String` 对象的大小。
*   `unsigned char equalsIgnoreCase(const String &s) const;`:  `equalsIgnoreCase` 函数比较当前 `String` 对象和另一个 `String` 对象是否相等，忽略大小写。
*   `unsigned char equalsConstantTime(const String &s) const;`: `equalsConstantTime` 函数以恒定时间比较字符串，防止时序攻击。
*   `unsigned char startsWith(const String &prefix) const;`:  `startsWith` 函数检查当前 `String` 对象是否以指定的前缀开头。
*   `unsigned char startsWith(const char *prefix) const { ... }`:  `startsWith` 函数检查当前 `String` 对象是否以指定的前缀开头 (C 风格字符串)。
*   `unsigned char startsWith(const __FlashStringHelper *prefix) const { ... }`:  `startsWith` 函数检查当前 `String` 对象是否以指定的前缀开头 (Flash 存储器中的字符串)。
*   `unsigned char startsWith(const String &prefix, unsigned int offset) const;`:  `startsWith` 函数检查当前 `String` 对象从指定偏移量开始是否以指定的前缀开头。
*   `unsigned char endsWith(const String &suffix) const;`:  `endsWith` 函数检查当前 `String` 对象是否以指定的后缀结尾。
*   `unsigned char endsWith(const char *suffix) const { ... }`:  `endsWith` 函数检查当前 `String` 对象是否以指定的后缀结尾 (C 风格字符串)。
*   `unsigned char endsWith(const __FlashStringHelper * suffix) const { ... }`:  `endsWith` 函数检查当前 `String` 对象是否以指定的后缀结尾 (Flash 存储器中的字符串)。

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);

  String str1 = "Hello";
  String str2 = "World";
  String str3 = "hello";

  if (str1 == "Hello") {
    Serial.println("str1 equals Hello");
  }

  if (str1 != str2) {
    Serial.println("str1 is not equal to str2");
  }

  if (str1.equalsIgnoreCase(str3)) {
    Serial.println("str1 equals str3, ignoring case");
  }

  if (str1.startsWith("He")) {
    Serial.println("str1 starts with He");
  }

  if (str2.endsWith("ld")) {
    Serial.println("str2 ends with ld");
  }

  int comparison = str1.compareTo(str2);
  if (comparison < 0) {
    Serial.println("str1 is less than str2");
  } else if (comparison > 0) {
    Serial.println("str1 is greater than str2");
  } else {
    Serial.println("str1 is equal to str2");
  }
}

void loop() {
  // do nothing
}
```

```c++
        // character access
        char charAt(unsigned int index) const;
        void setCharAt(unsigned int index, char c);
        char operator [](unsigned int index) const;
        char& operator [](unsigned int index);
        void getBytes(unsigned char *buf, unsigned int bufsize, unsigned int index = 0) const;
        void toCharArray(char *buf, unsigned int bufsize, unsigned int index = 0) const {
            getBytes((unsigned char *) buf, bufsize, index);
        }
        const char* c_str() const { return buffer(); }
        char* begin() { return wbuffer(); }
        char* end() { return wbuffer() + length(); }
        const char* begin() const { return c_str(); }
        const char* end() const { return c_str() + length(); }
```

**代码片段解释:**

*   `char charAt(unsigned int index) const;`:  `charAt` 函数返回指定索引处的字符。
*   `void setCharAt(unsigned int index, char c);`:  `setCharAt` 函数设置指定索引处的字符。
*   `char operator [](unsigned int index) const;`:  `[]` 运算符重载，用于访问指定索引处的字符 (只读)。
*   `char& operator [](unsigned int index);`:  `[]` 运算符重载，用于访问指定索引处的字符 (可写)。
*   `void getBytes(unsigned char *buf, unsigned int bufsize, unsigned int index = 0) const;`:  `getBytes` 函数将 `String` 对象的内容复制到指定的缓冲区。
*   `void toCharArray(char *buf, unsigned int bufsize, unsigned int index = 0) const { ... }`:  `toCharArray` 函数将 `String` 对象的内容复制到指定的字符数组。
*   `const char* c_str() const { return buffer(); }`:  `c_str` 函数返回指向 `String` 对象内容的 C 风格字符串的指针。
* `char* begin() { return wbuffer(); }`: 返回指向字符缓冲区的可修改起始指针，用于迭代等操作.
* `char* end() { return wbuffer() + length(); }`: 返回指向字符缓冲区结尾的可修改指针，用于迭代等操作.
* `const char* begin() const { return c_str(); }`: 返回指向常量字符缓冲区的起始指针，用于常量迭代等操作.
* `const char* end() const { return c_str() + length(); }`: 返回指向常量字符缓冲区结尾的指针，用于常量迭代等操作.

**用法示例:**

```c++
void setup() {
  Serial.begin(9600);

  String str = "Hello";

  Serial.print("Character at index 0: ");
  Serial.println(str.charAt(0)); // 输出 Character at index 0: H

  str.setCharAt(0, 'J');
  Serial.println(str); // 输出 Jello

  Serial.print("Character at index 1: ");
  Serial.println(str[1]); // 输出 Character at index 1: e

  str[1] = 'a';
  Serial.println(str); // 输出 Jallo

  char buffer[10];
  str.toCharArray(buffer, sizeof(buffer));
  Serial.print("Character array: ");
  Serial.println(buffer); // 输出 Character array: Jallo

  Serial.print("C-style string: ");
  Serial.println(str.c_str()); // 输出 C-style string: Jallo

  Serial.print("Iterating through the string: ");
  for (auto it = str.begin(); it != str.end(); ++it) {
      Serial.print(*it); // output Jallo
  }
  Serial.println();
}

void loop() {
  // do nothing
}
```

```c++
        // search
        int indexOf(char ch) const;
        int indexOf(char ch, unsigned int fromIndex) const;
        int indexOf(const String &str) const;
        int indexOf(const String &str, unsigned int fromIndex) const;
        int lastIndexOf(char ch) const;
        int lastIndexOf(char ch, unsigned int fromIndex) const;
        int lastIndexOf(const String &str) const;
        int lastIndexOf(const String &str, unsigned int fromIndex) const;
        String substring(unsigned int beginIndex) const {
            return substring(beginIndex, len());
        }
        ;
        String substring(unsigned int beginIndex, unsigned int endIndex) const;

        // modification
        void replace(char find, char replace);
        void replace(const String &find, const String &replace);
        void replace(const char *find, const String &replace) {
            this->replace(String(find), replace);
        }
        void replace(const __FlashStringHelper *find, const String &replace) {
            this->replace(String(find), replace);
        }
        void replace(const char *find, const char *replace) {
            this->replace(String(find), String(replace));
        }
        void replace(const __FlashStringHelper *find, const char *replace