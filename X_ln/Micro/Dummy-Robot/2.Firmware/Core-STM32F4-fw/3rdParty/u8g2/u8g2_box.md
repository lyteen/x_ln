Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Core-STM32F4-fw/3rdParty\u8g2\u8g2_box.c`

好的，我们重新开始，这次我们将对 C 语言的 `u8g2_box.c` 文件进行解释。

```c
/*

  u8g2_box.c

  Universal 8bit Graphics Library (https://github.com/olikraus/u8g2/)

  Copyright (c) 2016, olikraus@gmail.com
  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, 
  are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list 
    of conditions and the following disclaimer.
    
  * Redistributions in binary form must reproduce the above copyright notice, this 
    list of conditions and the following disclaimer in the documentation and/or other 
    materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND 
  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

*/

#include "u8g2.h"

/*
  draw a filled box
  restriction: does not work for w = 0 or h = 0
*/
void u8g2_DrawBox(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h)
{
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  while( h != 0 )
  { 
    u8g2_DrawHVLine(u8g2, x, y, w, 0);
    y++;    
    h--;
  }
}


/*
  draw a frame (empty box)
  restriction: does not work for w = 0 or h = 0
*/
void u8g2_DrawFrame(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h)
{
  u8g2_uint_t xtmp = x;
  
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  u8g2_DrawHVLine(u8g2, x, y, w, 0);
  if (h >= 2) {
    h-=2;
    y++;
    if (h > 0) {
      u8g2_DrawHVLine(u8g2, x, y, h, 1);
      x+=w;
      x--;
      u8g2_DrawHVLine(u8g2, x, y, h, 1);
      y+=h;
    }
    u8g2_DrawHVLine(u8g2, xtmp, y, w, 0);
  }
}




void u8g2_DrawRBox(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h, u8g2_uint_t r)
{
  u8g2_uint_t xl, yu;
  u8g2_uint_t yl, xr;

#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */

  xl = x;
  xl += r;
  yu = y;
  yu += r;
 
  xr = x;
  xr += w;
  xr -= r;
  xr -= 1;
  
  yl = y;
  yl += h;
  yl -= r; 
  yl -= 1;

  u8g2_DrawDisc(u8g2, xl, yu, r, U8G2_DRAW_UPPER_LEFT);
  u8g2_DrawDisc(u8g2, xr, yu, r, U8G2_DRAW_UPPER_RIGHT);
  u8g2_DrawDisc(u8g2, xl, yl, r, U8G2_DRAW_LOWER_LEFT);
  u8g2_DrawDisc(u8g2, xr, yl, r, U8G2_DRAW_LOWER_RIGHT);

  {
    u8g2_uint_t ww, hh;

    ww = w;
    ww -= r;
    ww -= r;
    xl++;
    yu++;
    
    if ( ww >= 3 )
    {
      ww -= 2;
      u8g2_DrawBox(u8g2, xl, y, ww, r+1);
      u8g2_DrawBox(u8g2, xl, yl, ww, r+1);
    }
    
    hh = h;
    hh -= r;
    hh -= r;
    //h--;
    if ( hh >= 3 )
    {
      hh -= 2;
      u8g2_DrawBox(u8g2, x, yu, w, hh);
    }
  }
}


void u8g2_DrawRFrame(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h, u8g2_uint_t r)
{
  u8g2_uint_t xl, yu;

#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */

  xl = x;
  xl += r;
  yu = y;
  yu += r;
 
  {
    u8g2_uint_t yl, xr;
      
    xr = x;
    xr += w;
    xr -= r;
    xr -= 1;
    
    yl = y;
    yl += h;
    yl -= r; 
    yl -= 1;

    u8g2_DrawCircle(u8g2, xl, yu, r, U8G2_DRAW_UPPER_LEFT);
    u8g2_DrawCircle(u8g2, xr, yu, r, U8G2_DRAW_UPPER_RIGHT);
    u8g2_DrawCircle(u8g2, xl, yl, r, U8G2_DRAW_LOWER_LEFT);
    u8g2_DrawCircle(u8g2, xr, yl, r, U8G2_DRAW_LOWER_RIGHT);
  }

  {
    u8g2_uint_t ww, hh;

    ww = w;
    ww -= r;
    ww -= r;
    hh = h;
    hh -= r;
    hh -= r;
    
    xl++;
    yu++;
    
    if ( ww >= 3 )
    {
      ww -= 2;
      h--;
      u8g2_DrawHLine(u8g2, xl, y, ww);
      u8g2_DrawHLine(u8g2, xl, y+h, ww);
    }
    
    if ( hh >= 3 )
    {
      hh -= 2;
      w--;
      u8g2_DrawVLine(u8g2, x, yu, hh);
      u8g2_DrawVLine(u8g2, x+w, yu, hh);
    }
  }
}
```

**1. 包含头文件 (Include Header):**

```c
#include "u8g2.h"
```

**描述:** 这一行代码包含了 `u8g2.h` 头文件。该头文件包含了 u8g2 图形库的函数声明、数据结构定义以及一些常量定义，这些对于使用 u8g2 库至关重要。
**用途:** 这是使用 u8g2 库的必要步骤，确保可以使用库提供的函数。

**2. 绘制填充矩形 (Draw Filled Box):**

```c
void u8g2_DrawBox(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h)
{
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  while( h != 0 )
  { 
    u8g2_DrawHVLine(u8g2, x, y, w, 0);
    y++;    
    h--;
  }
}
```

**描述:**  `u8g2_DrawBox` 函数用于在 u8g2 显示屏上绘制一个填充的矩形。它接受 u8g2 结构体指针、矩形左上角的 x 和 y 坐标、矩形的宽度 w 和高度 h 作为参数。该函数内部通过循环调用 `u8g2_DrawHVLine` 函数来逐行绘制水平线，从而填充矩形。`U8G2_WITH_INTERSECTION`宏定义决定是否开启裁剪功能。
**用途:**  该函数常用于绘制图形界面元素，例如按钮、进度条等。
**示例:**

```c
// 假设 u8g2 是一个已经初始化的 u8g2_t 结构体指针
u8g2_DrawBox(u8g2, 10, 20, 50, 30); // 在 (10, 20) 绘制一个 50x30 的填充矩形
```

**3. 绘制空心矩形 (Draw Frame):**

```c
void u8g2_DrawFrame(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h)
{
  u8g2_uint_t xtmp = x;
  
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
  
  u8g2_DrawHVLine(u8g2, x, y, w, 0);
  if (h >= 2) {
    h-=2;
    y++;
    if (h > 0) {
      u8g2_DrawHVLine(u8g2, x, y, h, 1);
      x+=w;
      x--;
      u8g2_DrawHVLine(u8g2, x, y, h, 1);
      y+=h;
    }
    u8g2_DrawHVLine(u8g2, xtmp, y, w, 0);
  }
}
```

**描述:**  `u8g2_DrawFrame` 函数用于在 u8g2 显示屏上绘制一个空心矩形（边框）。它同样接受 u8g2 结构体指针、矩形左上角的 x 和 y 坐标、矩形的宽度 w 和高度 h 作为参数。该函数通过绘制矩形的四条边来实现空心矩形。使用了`u8g2_DrawHVLine` 来画横线和竖线。`U8G2_WITH_INTERSECTION`宏定义决定是否开启裁剪功能。
**用途:**  该函数常用于绘制窗口、选中框等。
**示例:**

```c
// 假设 u8g2 是一个已经初始化的 u8g2_t 结构体指针
u8g2_DrawFrame(u8g2, 5, 10, 40, 25); // 在 (5, 10) 绘制一个 40x25 的空心矩形
```

**4. 绘制圆角填充矩形 (Draw Rounded Filled Box):**

```c
void u8g2_DrawRBox(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h, u8g2_uint_t r)
{
  u8g2_uint_t xl, yu;
  u8g2_uint_t yl, xr;

#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */

  xl = x;
  xl += r;
  yu = y;
  yu += r;
 
  xr = x;
  xr += w;
  xr -= r;
  xr -= 1;
  
  yl = y;
  yl += h;
  yl -= r; 
  yl -= 1;

  u8g2_DrawDisc(u8g2, xl, yu, r, U8G2_DRAW_UPPER_LEFT);
  u8g2_DrawDisc(u8g2, xr, yu, r, U8G2_DRAW_UPPER_RIGHT);
  u8g2_DrawDisc(u8g2, xl, yl, r, U8G2_DRAW_LOWER_LEFT);
  u8g2_DrawDisc(u8g2, xr, yl, r, U8G2_DRAW_LOWER_RIGHT);

  {
    u8g2_uint_t ww, hh;

    ww = w;
    ww -= r;
    ww -= r;
    xl++;
    yu++;
    
    if ( ww >= 3 )
    {
      ww -= 2;
      u8g2_DrawBox(u8g2, xl, y, ww, r+1);
      u8g2_DrawBox(u8g2, xl, yl, ww, r+1);
    }
    
    hh = h;
    hh -= r;
    hh -= r;
    //h--;
    if ( hh >= 3 )
    {
      hh -= 2;
      u8g2_DrawBox(u8g2, x, yu, w, hh);
    }
  }
}
```

**描述:** `u8g2_DrawRBox` 函数用于绘制一个圆角填充矩形。它接受 u8g2 结构体指针、矩形左上角的 x 和 y 坐标、矩形的宽度 w 和高度 h 以及圆角的半径 r 作为参数。该函数首先绘制四个角的圆弧，然后绘制连接这些圆弧的矩形部分。使用了`u8g2_DrawDisc` 画圆， 使用了`u8g2_DrawBox` 画矩形， `U8G2_WITH_INTERSECTION`宏定义决定是否开启裁剪功能。
**用途:**  该函数常用于绘制美观的按钮、对话框等。
**示例:**

```c
// 假设 u8g2 是一个已经初始化的 u8g2_t 结构体指针
u8g2_DrawRBox(u8g2, 15, 25, 60, 40, 8); // 在 (15, 25) 绘制一个 60x40 圆角半径为 8 的填充矩形
```

**5. 绘制圆角空心矩形 (Draw Rounded Frame):**

```c
void u8g2_DrawRFrame(u8g2_t *u8g2, u8g2_uint_t x, u8g2_uint_t y, u8g2_uint_t w, u8g2_uint_t h, u8g2_uint_t r)
{
  u8g2_uint_t xl, yu;

#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */

  xl = x;
  xl += r;
  yu = y;
  yu += r;
 
  {
    u8g2_uint_t yl, xr;
      
    xr = x;
    xr += w;
    xr -= r;
    xr -= 1;
    
    yl = y;
    yl += h;
    yl -= r; 
    yl -= 1;

    u8g2_DrawCircle(u8g2, xl, yu, r, U8G2_DRAW_UPPER_LEFT);
    u8g2_DrawCircle(u8g2, xr, yu, r, U8G2_DRAW_UPPER_RIGHT);
    u8g2_DrawCircle(u8g2, xl, yl, r, U8G2_DRAW_LOWER_LEFT);
    u8g2_DrawCircle(u8g2, xr, yl, r, U8G2_DRAW_LOWER_RIGHT);
  }

  {
    u8g2_uint_t ww, hh;

    ww = w;
    ww -= r;
    ww -= r;
    hh = h;
    hh -= r;
    hh -= r;
    
    xl++;
    yu++;
    
    if ( ww >= 3 )
    {
      ww -= 2;
      h--;
      u8g2_DrawHLine(u8g2, xl, y, ww);
      u8g2_DrawHLine(u8g2, xl, y+h, ww);
    }
    
    if ( hh >= 3 )
    {
      hh -= 2;
      w--;
      u8g2_DrawVLine(u8g2, x, yu, hh);
      u8g2_DrawVLine(u8g2, x+w, yu, hh);
    }
  }
}
```

**描述:**  `u8g2_DrawRFrame` 函数用于绘制一个圆角空心矩形（圆角边框）。它接受的参数与 `u8g2_DrawRBox` 类似。该函数首先绘制四个角的圆弧，然后绘制连接这些圆弧的直线部分。使用了 `u8g2_DrawCircle` 画圆弧，使用 `u8g2_DrawHLine` 和 `u8g2_DrawVLine` 画横线和竖线。`U8G2_WITH_INTERSECTION`宏定义决定是否开启裁剪功能。
**用途:**  该函数常用于绘制美观的窗口、输入框等。
**示例:**

```c
// 假设 u8g2 是一个已经初始化的 u8g2_t 结构体指针
u8g2_DrawRFrame(u8g2, 20, 30, 70, 50, 10); // 在 (20, 30) 绘制一个 70x50 圆角半径为 10 的空心矩形
```

**6. 裁剪判断 (`U8G2_WITH_INTERSECTION` and `u8g2_IsIntersection`)**

```c
#ifdef U8G2_WITH_INTERSECTION
  if ( u8g2_IsIntersection(u8g2, x, y, x+w, y+h) == 0 ) 
    return;
#endif /* U8G2_WITH_INTERSECTION */
```

**描述:**  这些代码片段用于在绘制之前进行裁剪判断。  `U8G2_WITH_INTERSECTION` 是一个宏，如果定义了它，则会启用裁剪功能。  `u8g2_IsIntersection` 函数用于判断要绘制的矩形是否与当前绘图区域相交。 如果不相交，则直接返回，不进行绘制，从而避免了绘制到屏幕外面的区域，提高效率。

**总而言之:**

`u8g2_box.c` 文件提供了一系列函数，用于在 u8g2 显示屏上绘制各种矩形，包括填充矩形、空心矩形、圆角填充矩形和圆角空心矩形。这些函数通过调用 u8g2 库中的其他绘图函数来实现，并可以根据需要进行裁剪。这些函数可以用于构建各种图形用户界面。

希望这次的解释更清楚！
