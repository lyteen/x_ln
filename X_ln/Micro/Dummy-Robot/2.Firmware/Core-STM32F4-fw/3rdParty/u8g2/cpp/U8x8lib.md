Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/3rdParty\u8g2\cpp\U8x8lib.h`

好的，我们重新开始。 这次我会更详细地解释代码，并包含示例用法和中文解释。

**文件: `U8X8LIB.HH`**

```c++
#ifndef U8X8LIB_HH
#define U8X8LIB_HH

#include <Print.h>  // 包含Arduino的Print类，U8X8类继承自它，用于打印输出
#include "u8x8.h"    // 包含底层u8x8库的头文件，该库处理与OLED显示器的实际通信

#define U8X8_HAVE_HW_I2C  // 定义是否使用硬件I2C，如果你的Arduino板支持硬件I2C，就应该定义它

/* Exported variables --------------------------------------------------------*/
// 这两个extern "C"声明是为了让C++代码可以调用C代码，因为u8x8库是用C语言编写的
extern "C" uint8_t u8x8_byte_stm32_hw_i2c(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, void *arg_ptr);  // STM32硬件I2C字节传输函数
extern "C" uint8_t u8x8_stm32_gpio_and_delay(u8x8_t *u8x8, uint8_t msg, uint8_t arg_int, U8X8_UNUSED void *arg_ptr); // STM32 GPIO和延时函数

// U8X8类，提供高级接口来控制OLED显示器
class U8X8 : public Print
{
protected:
    u8x8_t u8x8;  // 底层u8x8库的结构体实例，包含了显示器的配置信息

public:
    uint8_t tx, ty; // 当前文本光标的位置 (x, y)

    // 构造函数，初始化光标位置到原点
    U8X8(void)
    { home(); }

    // 返回底层u8x8结构体的指针
    u8x8_t *getU8x8(void)
    { return &u8x8; }

    // 格式化字符串输出，类似于printf
    void sendF(const char *fmt, ...)
    {
        va_list va;
        va_start(va, fmt);
        u8x8_cad_vsendf(&u8x8, fmt, va); // 调用底层函数进行格式化输出
        va_end(va);
    }

    // 获取总线时钟频率
    uint32_t getBusClock(void)
    { return u8x8.bus_clock; }

    // 设置总线时钟频率
    void setBusClock(uint32_t clock_speed)
    { u8x8.bus_clock = clock_speed; }

    // 设置I2C设备地址
    void setI2CAddress(uint8_t adr)
    { u8x8_SetI2CAddress(&u8x8, adr); }

    // 获取显示器的列数
    uint8_t getCols(void)
    { return u8x8_GetCols(&u8x8); }

    // 获取显示器的行数
    uint8_t getRows(void)
    { return u8x8_GetRows(&u8x8); }

    // 绘制一个tile (8x8像素的块)
    void drawTile(uint8_t x, uint8_t y, uint8_t cnt, uint8_t *tile_ptr)
    {
        u8x8_DrawTile(&u8x8, x, y, cnt, tile_ptr);
    }

    // 初始化显示器
    void initDisplay(void)
    {
        u8x8_InitDisplay(&u8x8);
    }

    // 清空显示器
    void clearDisplay(void)
    {
        u8x8_ClearDisplay(&u8x8);
    }

    // 填充整个显示器
    void fillDisplay(void)
    {
        u8x8_FillDisplay(&u8x8);
    }

    // 设置省电模式
    void setPowerSave(uint8_t is_enable)
    {
        u8x8_SetPowerSave(&u8x8, is_enable);
    }

    // 初始化显示器并设置一些默认值
    bool begin(void)
    {
        initDisplay();
        clearDisplay();
        setPowerSave(0);
        return 1;
    }

    // 设置翻转模式
    void setFlipMode(uint8_t mode)
    {
        u8x8_SetFlipMode(&u8x8, mode);
    }

    // 刷新显示 (仅对某些显示器有效，例如SSD1606)
    void refreshDisplay(void)
    {            // Dec 16: Only required for SSD1606
        u8x8_RefreshDisplay(&u8x8);
    }

    // 清除指定行
    void clearLine(uint8_t line)
    {
        u8x8_ClearLine(&u8x8, line);
    }

    // 设置对比度
    void setContrast(uint8_t value)
    {
        u8x8_SetContrast(&u8x8, value);
    }

    // 设置反显字体
    void setInverseFont(uint8_t value)
    {
        u8x8_SetInverseFont(&u8x8, value);
    }

    // 设置字体
    void setFont(const uint8_t *font_8x8)
    {
        u8x8_SetFont(&u8x8, font_8x8);
    }

    // 绘制一个字形
    void drawGlyph(uint8_t x, uint8_t y, uint8_t encoding)
    {
        u8x8_DrawGlyph(&u8x8, x, y, encoding);
    }

    // 绘制一个2x2大小的字形
    void draw2x2Glyph(uint8_t x, uint8_t y, uint8_t encoding)
    {
        u8x8_Draw2x2Glyph(&u8x8, x, y, encoding);
    }

    // 绘制一个1x2大小的字形
    void draw1x2Glyph(uint8_t x, uint8_t y, uint8_t encoding)
    {
        u8x8_Draw1x2Glyph(&u8x8, x, y, encoding);
    }

    // 绘制一个字符串
    void drawString(uint8_t x, uint8_t y, const char *s)
    {
        u8x8_DrawString(&u8x8, x, y, s);
    }

    // 绘制UTF-8编码的字符串
    void drawUTF8(uint8_t x, uint8_t y, const char *s)
    {
        u8x8_DrawUTF8(&u8x8, x, y, s);
    }

    // 绘制一个2x2大小的字符串
    void draw2x2String(uint8_t x, uint8_t y, const char *s)
    {
        u8x8_Draw2x2String(&u8x8, x, y, s);
    }

    // 绘制一个1x2大小的字符串
    void draw1x2String(uint8_t x, uint8_t y, const char *s)
    {
        u8x8_Draw1x2String(&u8x8, x, y, s);
    }

    // 绘制一个2x2大小的UTF-8字符串
    void draw2x2UTF8(uint8_t x, uint8_t y, const char *s)
    {
        u8x8_Draw2x2UTF8(&u8x8, x, y, s);
    }

    // 绘制一个1x2大小的UTF-8字符串
    void draw1x2UTF8(uint8_t x, uint8_t y, const char *s)
    {
        u8x8_Draw1x2UTF8(&u8x8, x, y, s);
    }

    // 获取UTF-8字符串的长度
    uint8_t getUTF8Len(const char *s)
    {
        return u8x8_GetUTF8Len(&u8x8, s);
    }

    // 实现Print类的write方法，用于打印单个字符
    size_t write(uint8_t v); // 函数定义在.cpp文件中

    // 实现Print类的write方法，用于打印字符串
    size_t write(const uint8_t *buffer, size_t size)
    {
        size_t cnt = 0;
        while (size > 0)
        {
            cnt += write(*buffer++);
            size--;
        }
        return cnt;
    }

    // 设置反显模式
    void inverse(void)
    { setInverseFont(1); }

    // 取消反显模式
    void noInverse(void)
    { setInverseFont(0); }

    // 获取菜单事件 (例如按键事件)
    uint8_t getMenuEvent(void)
    { return u8x8_GetMenuEvent(&u8x8); }

    // 用户界面选择列表
    uint8_t userInterfaceSelectionList(const char *title, uint8_t start_pos, const char *sl)
    {
        return u8x8_UserInterfaceSelectionList(&u8x8, title, start_pos, sl);
    }

    // 用户界面消息框
    uint8_t userInterfaceMessage(const char *title1, const char *title2, const char *title3, const char *buttons)
    {
        return u8x8_UserInterfaceMessage(&u8x8, title1, title2, title3, buttons);
    }

    // 用户界面输入值
    uint8_t
    userInterfaceInputValue(const char *title, const char *pre, uint8_t *value, uint8_t lo, uint8_t hi, uint8_t digits,
                            const char *post)
    {
        return u8x8_UserInterfaceInputValue(&u8x8, title, pre, value, lo, hi, digits, post);
    }

    // LiquidCrystal兼容函数，设置光标到原点
    void home(void)
    {
        tx = 0;
        ty = 0;
    }

    // LiquidCrystal兼容函数，清空显示器并设置光标到原点
    void clear(void)
    {
        clearDisplay();
        home();
    }

    // LiquidCrystal兼容函数，关闭显示
    void noDisplay(void)
    { u8x8_SetPowerSave(&u8x8, 1); }

    // LiquidCrystal兼容函数，打开显示
    void display(void)
    { u8x8_SetPowerSave(&u8x8, 0); }

    // LiquidCrystal兼容函数，设置光标位置
    void setCursor(uint8_t x, uint8_t y)
    {
        tx = x;
        ty = y;
    }

    // 绘制log信息
    void drawLog(uint8_t x, uint8_t y, class U8X8LOG &u8x8log);

};

// U8X8LOG类，用于在显示器上显示log信息
class U8X8LOG : public Print
{

public:
    u8log_t u8log; // 底层u8log库的结构体实例

    // 构造函数，不执行任何操作，需要在begin()中初始化
    U8X8LOG(void)
    {}

    // 初始化U8X8LOG，并连接到U8X8对象
    bool begin(class U8X8 &u8x8, uint8_t width, uint8_t height, uint8_t *buf)
    {
        u8log_Init(&u8log, width, height, buf);
        u8log_SetCallback(&u8log, u8log_u8x8_cb, u8x8.getU8x8()); // 设置回调函数，用于将log信息绘制到U8X8显示器上
        return true;
    }

    // 初始化U8X8LOG， disconnected版本，需要手动刷新
    bool begin(uint8_t width, uint8_t height, uint8_t *buf)
    {
        u8log_Init(&u8log, width, height, buf);
        return true;
    }

    // 设置行高偏移
    void setLineHeightOffset(int8_t line_height_offset)
    {
        u8log_SetLineHeightOffset(&u8log, line_height_offset);
    }

    // 设置是否在每个字符后重绘行
    void setRedrawMode(uint8_t is_redraw_line_for_each_char)
    {
        u8log_SetRedrawMode(&u8log, is_redraw_line_for_each_char);
    }

    // 实现Print类的write方法，用于写入单个字符到log
    size_t write(uint8_t v)
    {
        u8log_WriteChar(&u8log, v);
        return 1;
    }

    // 实现Print类的write方法，用于写入字符串到log
    size_t write(const uint8_t *buffer, size_t size)
    {
        size_t cnt = 0;
        while (size > 0)
        {
            cnt += write(*buffer++);
            size--;
        }
        return cnt;
    }

    // 写入字符串到log
    void writeString(const char *s)
    { u8log_WriteString(&u8log, s); }

    // 写入单个字符到log
    void writeChar(uint8_t c)
    { u8log_WriteChar(&u8log, c); }

    // 写入8位16进制数到log
    void writeHex8(uint8_t b)
    { u8log_WriteHex8(&u8log, b); }

    // 写入16位16进制数到log
    void writeHex16(uint16_t v)
    { u8log_WriteHex16(&u8log, v); }

    // 写入32位16进制数到log
    void writeHex32(uint32_t v)
    { u8log_WriteHex32(&u8log, v); }

    // 写入8位10进制数到log
    void writeDec8(uint8_t v, uint8_t d)
    { u8log_WriteDec8(&u8log, v, d); }

    // 写入16位10进制数到log
    void writeDec16(uint8_t v, uint8_t d)
    { u8log_WriteDec16(&u8log, v, d); }
};


/* u8log_u8x8.c */
// U8X8类的drawLog方法的内联实现
inline void U8X8::drawLog(uint8_t x, uint8_t y, class U8X8LOG &u8x8log)
{
    u8x8_DrawLog(&u8x8, x, y, &(u8x8log.u8log));
}


#endif /* _U8X8LIB_HH */
```

**代码解释:**

*   **`#ifndef U8X8LIB_HH`, `#define U8X8LIB_HH`, `#endif`**:  这是头文件保护，防止头文件被重复包含。  (头文件保护，避免重复包含)
*   **`#include <Print.h>`**: 包含Arduino的 `Print` 类，使得 `U8X8` 和 `U8X8LOG` 类可以像 `Serial` 一样使用 `print()` 和 `println()` 函数。(包含Arduino的打印类，U8X8类和U8X8LOG类可以使用print和println函数)
*   **`#include "u8x8.h"`**: 包含底层 `u8x8` 库的头文件。 这是核心库，负责与OLED显示器的通信。(包含底层u8x8库的头文件，控制OLED显示器)
*   **`#define U8X8_HAVE_HW_I2C`**:  如果你的Arduino板支持硬件I2C，你应该定义这个宏。这会告诉库使用硬件I2C接口，通常比软件I2C快。(定义是否使用硬件I2C，如果你的Arduino板支持硬件I2C，就应该定义它)
*   **`extern "C"`**:  `u8x8` 库是用C语言编写的，所以我们需要使用 `extern "C"` 来告诉C++编译器使用C语言的链接方式。 (使用C语言链接方式)
*   **`class U8X8 : public Print`**:  `U8X8` 类继承自 `Print` 类，所以你可以使用 `print()` 和 `println()` 函数在OLED显示器上打印文本。(U8X8类继承自Print类，可以使用print和println函数)
*   **`u8x8_t u8x8`**:  `U8X8` 类包含一个 `u8x8_t` 类型的成员变量。 这是底层 `u8x8` 库的结构体，包含了显示器的配置信息。(底层u8x8库的结构体实例，包含了显示器的配置信息)
*   **`tx`, `ty`**:  这两个变量表示当前文本光标的位置。(当前文本光标的位置)
*   **`sendF(const char *fmt, ...)`**:  这个函数类似于 `printf()` 函数，可以格式化字符串并将其输出到OLED显示器。(格式化字符串输出，类似于printf)
*   **`begin()`**:  初始化显示器。你需要在 `setup()` 函数中调用这个函数。(初始化显示器)
*   **`clear()`**: 清空显示器。(清空显示器)
*   **`setFont()`**: 设置字体。`u8x8` 库支持多种字体。(设置字体)
*   **`drawString()`**:  在指定位置绘制一个字符串。(在指定位置绘制一个字符串)
*   **`write(uint8_t v)`**:  实现 `Print` 类的 `write()` 方法。 这个方法将单个字符输出到OLED显示器。  这个函数的实现通常在`.cpp` 文件中。(实现Print类的write方法，用于打印单个字符)
*   **`class U8X8LOG : public Print`**:  `U8X8LOG` 类用于在OLED显示器上显示log信息。(U8X8LOG类，用于在显示器上显示log信息)
*   **`u8log_t u8log`**:  `U8X8LOG` 类包含一个 `u8log_t` 类型的成员变量。 这是底层 `u8log` 库的结构体，用于管理log信息。(底层u8log库的结构体实例)
*   **`U8X8::drawLog()`**:  这个函数用于在U8X8显示器上绘制U8X8LOG信息。(在U8X8显示器上绘制U8X8LOG信息)

**示例用法:**

```c++
#include <Arduino.h>
#include <U8x8lib.h>  // 包含U8X8库的头文件

// 根据你的OLED显示器和Arduino板选择合适的构造函数
U8X8 u8x8(/*display_rotation=*/U8X8_R0, /*rst=*/16, /*scl=*/15, /*sda=*/4); // Example for ESP32

void setup() {
  Serial.begin(115200);
  u8x8.begin();          // 初始化U8X8库
  u8x8.setFont(u8x8_font_chroma48medium8_r1); // 设置字体
  u8x8.clearDisplay();   // 清空显示器
}

void loop() {
  u8x8.setCursor(0, 0);    // 设置光标位置到 (0, 0)
  u8x8.print("Hello, OLED!"); // 打印字符串
  delay(1000);
  u8x8.clearDisplay();
  u8x8.setCursor(0,0);
  u8x8.print("Count: ");
  u8x8.print(millis()/1000); // 显示运行时间
  delay(1000);
}
```

**中文解释:**

这段代码演示了如何使用 `U8X8` 库在OLED显示器上显示 "Hello, OLED!" 和一个计数器。  `setup()` 函数初始化串口通信和 `U8X8` 库，并设置字体和清空显示器。 `loop()` 函数每隔一秒钟在OLED显示器的第一行打印 "Hello, OLED!"，然后清空显示器，显示计数器.

**更详细的示例:**

```c++
#include <Arduino.h>
#include <U8x8lib.h>

// 根据你的OLED显示器和Arduino板选择合适的构造函数
U8X8 u8x8(/*display_rotation=*/U8X8_R0, /*rst=*/16, /*scl=*/15, /*sda=*/4); // Example for ESP32

U8X8LOG u8x8log; // 创建一个U8X8LOG对象
uint8_t log_buffer[32]; // 为U8X8LOG分配一个缓冲区

void setup() {
  Serial.begin(115200);
  u8x8.begin();
  u8x8.setFont(u8x8_font_chroma48medium8_r1);
  u8x8.clearDisplay();

  u8x8log.begin(u8x8, 16, 4, log_buffer); // 初始化U8X8LOG，16列，4行
  u8x8log.println("Starting...");
  Serial.println("Starting..."); //同时打印到串口
}

void loop() {
  u8x8.clearDisplay(); // 清除屏幕
  u8x8.setCursor(0, 0);
  u8x8.print("Millis: ");
  u8x8.print(millis());

  u8x8.drawLog(0, 4, u8x8log); // 将log信息绘制到屏幕上，从第0列，第4行开始

  u8x8log.println(millis());  // 写入log信息，也会打印到串口
  Serial.println(millis());

  delay(1000); // 暂停1秒
}
```

**中文解释:**

这段代码演示了如何使用 `U8X8LOG` 类在OLED显示器上显示log信息。  `setup()` 函数初始化串口通信、`U8X8` 库和 `U8X8LOG` 库。  `loop()` 函数首先清除屏幕，然后打印当前的 `millis()` 值到屏幕的第一行。  接下来，它将 `U8X8LOG` 中的log信息绘制到屏幕上，从第0列，第4行开始。  最后，它将当前的 `millis()` 值写入 `U8X8LOG` 中，并暂停1秒。

希望这些解释和示例能够帮助你理解 `U8X8LIB.HH` 文件中的代码!
