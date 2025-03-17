Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/3rdParty\u8g2\u8g2_bitmap.c`

好的，没问题，我们重新开始，我对代码中的关键部分进行解释，并附上中文描述和简单的使用示例。

**1. `u8g2_SetBitmapMode` 函数**

```c
void u8g2_SetBitmapMode(u8g2_t *u8g2, uint8_t is_transparent) {
  u8g2->bitmap_transparency = is_transparent;
}
```

*   **描述:** 这个函数用来设置位图绘制模式。`is_transparent` 参数决定了位图是否采用透明模式。如果 `is_transparent` 为 1，则位图中的 0 值像素将不被绘制，相当于透明效果。如果为 0，则 0 值像素将使用背景色绘制。

    **中文描述:** 此函数用于设置位图的绘制模式。 `is_transparent` 参数决定是否启用透明模式。 如果 `is_transparent` 为 1，则位图中值为 0 的像素将不会被绘制，实现透明效果。 如果为 0，则值为 0 的像素将使用背景色绘制。

*   **如何使用:**

    ```c
    u8g2_t u8g2;
    u8g2_Setup_ssd1306_i2c_128x64_noname_f(&u8g2, U8G2_R0, u8x8_byte_sw_i2c, u8x8_gpio_and_delay_arduino);  // 初始化 u8g2 结构体
    u8g2_InitDisplay(&u8g2); // 初始化显示
    u8g2_SetPowerSave(&u8g2, 0); // 打开显示

    u8g2_SetBitmapMode(&u8g2, 1); // 设置为透明模式
    // 或者
    u8g2_SetBitmapMode(&u8g2, 0); // 设置为不透明模式
    ```

*   **简单示例:** 想象一下，你有一个黑底白色图案的位图。如果设置为透明模式，那么黑色的部分将不会显示，只显示白色的图案。如果设置为不透明模式，那么黑色的背景也会显示出来。

    **中文描述:** 想象一下，你有一张黑色背景，白色图案的位图。如果设置为透明模式，黑色部分将不显示，只显示白色图案。如果设置为不透明模式，黑色背景也会显示。

**2. `u8g2_DrawHorizontalBitmap` 函数**

```c
void u8g2_DrawHorizontalBitmap(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t len, const uint8_t *b)
{
  uint8_t mask;
  uint8_t color = u8g2->draw_color;
  uint8_t ncolor = (color == 0 ? 1 : 0);

#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+len, y+1) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  mask = 128;
  while(len > 0)
  {
    if ( *b & mask ) {
      u8g2->draw_color = color;
      u8g2_DrawHVLine(u8g2, x, y, 1, 0);
    } else if ( u8g2->bitmap_transparency == 0 ) {
      u8g2->draw_color = ncolor;
      u8g2_DrawHVLine(u8g2, x, y, 1, 0);
    }

    x++;
    mask >>= 1;
    if ( mask == 0 )
    {
      mask = 128;
      b++;
    }
    len--;
  }
  u8g2->draw_color = color;
}
```

*   **描述:** 此函数在指定位置 `(x, y)` 绘制水平位图。`len` 参数是位图的像素长度。`b` 是指向位图数据的指针。函数会根据 `u8g2->bitmap_transparency` 的设置来决定是否绘制位图中的 0 值像素。

    **中文描述:** 此函数在指定位置 `(x, y)` 绘制水平位图。`len` 参数是位图的像素长度。`b` 是指向位图数据的指针。函数根据 `u8g2->bitmap_transparency` 的设置来决定是否绘制位图中值为 0 的像素。

*   **如何使用:**

    ```c
    u8g2_t u8g2;
    u8g2_Setup_ssd1306_i2c_128x64_noname_f(&u8g2, U8G2_R0, u8x8_byte_sw_i2c, u8x8_gpio_and_delay_arduino);
    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);

    uint8_t bitmap_data[] = {0b11111111, 0b00000000, 0b11111111}; // 示例位图数据
    u8g2_DrawHorizontalBitmap(&u8g2, 10, 20, 24, bitmap_data); // 在 (10, 20) 位置绘制位图，长度为 24 像素
    ```

*   **简单示例:**  `bitmap_data` 中存储了位图数据。每个字节代表 8 个像素。`0b11111111` 表示 8 个像素都设置为 1（例如白色），`0b00000000` 表示 8 个像素都设置为 0 （例如黑色）。此函数会根据这些数据在屏幕上绘制一条水平线。

    **中文描述:**  `bitmap_data` 存储了位图数据。每个字节代表 8 个像素。`0b11111111` 表示 8 个像素都设置为 1（例如白色），`0b00000000` 表示 8 个像素都设置为 0 （例如黑色）。 此函数会根据这些数据在屏幕上绘制一条水平线。

**3. `u8g2_DrawBitmap` 函数**

```c
void u8g2_DrawBitmap(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t cnt, u8g2_uint_t h, const uint8_t *bitmap)
{
  u8g2_uint_t w;
  w = cnt;
  w *= 8;
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  while( h > 0 )
  {
    u8g2_DrawHorizontalBitmap(u8g2, x, y, w, bitmap);
    bitmap += cnt;
    y++;
    h--;
  }
}
```

*   **描述:**  此函数是一个 u8glib 兼容的位图绘制函数，它在指定位置 `(x, y)` 绘制一个矩形位图。`cnt` 参数是位图的宽度（以字节为单位），`h` 是位图的高度（以像素为单位），`bitmap` 是指向位图数据的指针。

    **中文描述:**  此函数是一个 u8glib 兼容的位图绘制函数，它在指定位置 `(x, y)` 绘制一个矩形位图。`cnt` 参数是位图的宽度（以字节为单位），`h` 是位图的高度（以像素为单位），`bitmap` 是指向位图数据的指针。

*   **如何使用:**

    ```c
    u8g2_t u8g2;
    u8g2_Setup_ssd1306_i2c_128x64_noname_f(&u8g2, U8G2_R0, u8x8_byte_sw_i2c, u8x8_gpio_and_delay_arduino);
    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);

    uint8_t bitmap_data[] = {
      0b11111111, 0b00000000,
      0b00000000, 0b11111111
    }; // 2x2 像素的位图数据
    u8g2_DrawBitmap(&u8g2, 10, 20, 1, 2, bitmap_data); // 在 (10, 20) 位置绘制 2x2 像素的位图
    ```

*   **简单示例:**  在这个例子中，`bitmap_data` 存储了一个 2x2 像素的位图。`cnt` 为 1，表示宽度为 1 字节 (8 像素，但实际只用了前2个)，`h` 为 2，表示高度为 2 像素。`u8g2_DrawBitmap` 函数会将这个 2x2 的位图绘制到屏幕上。

    **中文描述:**  在这个例子中，`bitmap_data` 存储了一个 2x2 像素的位图。`cnt` 为 1，表示宽度为 1 字节 (8 像素, 但实际只用了前2个)，`h` 为 2，表示高度为 2 像素。`u8g2_DrawBitmap` 函数会将这个 2x2 的位图绘制到屏幕上。

**4. `u8g2_DrawHXBM` 和 `u8g2_DrawXBM` 函数**

```c
void u8g2_DrawHXBM(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t len, const uint8_t *b)
{
  uint8_t mask;
  uint8_t color = u8g2->draw_color;
  uint8_t ncolor = (color == 0 ? 1 : 0);
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+len, y+1) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  mask = 1;
  while(len > 0) {
    if ( *b & mask ) {
      u8g2->draw_color = color;
      u8g2_DrawHVLine(u8g2, x, y, 1, 0);
    } else if ( u8g2->bitmap_transparency == 0 ) {
      u8g2->draw_color = ncolor;
      u8g2_DrawHVLine(u8g2, x, y, 1, 0);
    }
    x++;
    mask <<= 1;
    if ( mask == 0 )
    {
      mask = 1;
      b++;
    }
    len--;
  }
  u8g2->draw_color = color;
}


void u8g2_DrawXBM(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h, const uint8_t *bitmap)
{
  u8g2_uint_t blen;
  blen = w;
  blen += 7;
  blen >>= 3;
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  while( h > 0 )
  {
    u8g2_DrawHXBM(u8g2, x, y, w, bitmap);
    bitmap += blen;
    y++;
    h--;
  }
}
```

*   **描述:**  `u8g2_DrawHXBM` 函数类似于 `u8g2_DrawHorizontalBitmap`，但它处理 XBM 格式的位图数据。  XBM 格式的位图数据中，每个字节的位是从右向左排列的 (LSB first)。`u8g2_DrawXBM` 函数用于绘制整个 XBM 格式的位图。

    **中文描述:** `u8g2_DrawHXBM` 函数类似于 `u8g2_DrawHorizontalBitmap`，但它处理 XBM 格式的位图数据。 XBM 格式的位图数据中，每个字节的位是从右向左排列的 (LSB first)。`u8g2_DrawXBM` 函数用于绘制整个 XBM 格式的位图。

*   **如何使用:**

    ```c
    u8g2_t u8g2;
    u8g2_Setup_ssd1306_i2c_128x64_noname_f(&u8g2, U8G2_R0, u8x8_byte_sw_i2c, u8x8_gpio_and_delay_arduino);
    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);

    uint8_t xbm_data[] = {
      0b10000000, 0b00000001,
      0b00000001, 0b10000000
    }; // 2x2 像素的 XBM 位图数据
    u8g2_DrawXBM(&u8g2, 10, 20, 2, 2, xbm_data); // 在 (10, 20) 位置绘制 2x2 像素的 XBM 位图
    ```

*   **简单示例:**  `xbm_data` 存储了一个 2x2 像素的 XBM 格式的位图数据。注意，位图数据的排列方式与 `u8g2_DrawBitmap` 不同。`u8g2_DrawXBM` 函数会将这个 2x2 的位图绘制到屏幕上。

     **中文描述:** `xbm_data` 存储了一个 2x2 像素的 XBM 格式的位图数据。 注意，位图数据的排列方式与 `u8g2_DrawBitmap` 不同。 `u8g2_DrawXBM` 函数会将这个 2x2 的位图绘制到屏幕上。

**5. `u8g2_DrawHXBMP` 和 `u8g2_DrawXBMP` 函数**

```c
void u8g2_DrawHXBMP(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t len, const uint8_t *b)
{
  uint8_t mask;
  uint8_t color = u8g2->draw_color;
  uint8_t ncolor = (color == 0 ? 1 : 0);
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+len, y+1) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  mask = 1;
  while(len > 0)
  {
    if( u8x8_pgm_read(b) & mask ) {
      u8g2->draw_color = color;
      u8g2_DrawHVLine(u8g2, x, y, 1, 0);
    } else if( u8g2->bitmap_transparency == 0 ) {
      u8g2->draw_color = ncolor;
      u8g2_DrawHVLine(u8g2, x, y, 1, 0);
    }
   
    x++;
    mask <<= 1;
    if ( mask == 0 )
    {
      mask = 1;
      b++;
    }
    len--;
  }
  u8g2->draw_color = color;
}


void u8g2_DrawXBMP(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h, const uint8_t *bitmap)
{
  u8g2_uint_t blen;
  blen = w;
  blen += 7;
  blen >>= 3;
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  while( h > 0 )
  {
    u8g2_DrawHXBMP(u8g2, x, y, w, bitmap);
    bitmap += blen;
    y++;
    h--;
  }
}
```

*   **描述:**  这两个函数与 `u8g2_DrawHXBM` 和 `u8g2_DrawXBM` 类似，但它们用于绘制存储在程序存储器 (通常是 Flash) 中的 XBM 格式的位图数据。`u8x8_pgm_read(b)` 用于从程序存储器读取数据。

    **中文描述:** 这两个函数与 `u8g2_DrawHXBM` 和 `u8g2_DrawXBM` 类似，但它们用于绘制存储在程序存储器 (通常是 Flash) 中的 XBM 格式的位图数据。 `u8x8_pgm_read(b)` 用于从程序存储器读取数据。

*   **如何使用:** 需要将位图数据存储在程序存储器中，例如使用 `PROGMEM` 关键字 (在 Arduino 环境中)。

    ```c
    #include <avr/pgmspace.h> // Arduino 环境需要包含此头文件

    u8g2_t u8g2;
    u8g2_Setup_ssd1306_i2c_128x64_noname_f(&u8g2, U8G2_R0, u8x8_byte_sw_i2c, u8x8_gpio_and_delay_arduino);
    u8g2_InitDisplay(&u8g2);
    u8g2_SetPowerSave(&u8g2, 0);

    const uint8_t xbm_data[] PROGMEM = {
      0b10000000, 0b00000001,
      0b00000001, 0b10000000
    }; // 2x2 像素的 XBM 位图数据，存储在程序存储器中

    u8g2_DrawXBMP(&u8g2, 10, 20, 2, 2, xbm_data); // 在 (10, 20) 位置绘制 2x2 像素的 XBM 位图
    ```

*   **简单示例:**  此示例与 `u8g2_DrawXBM` 示例类似，但位图数据 `xbm_data` 存储在程序存储器中。这在嵌入式系统中很常见，可以节省 RAM 空间。

     **中文描述:** 此示例与 `u8g2_DrawXBM` 示例类似，但位图数据 `xbm_data` 存储在程序存储器中。 这在嵌入式系统中很常见，可以节省 RAM 空间。

希望这些解释和示例能够帮助你理解 `u8g2` 库中的位图绘制函数。  这些函数提供了灵活的方式来在 OLED 屏幕上显示图像和图标，并且支持透明模式和不同的位图格式。
