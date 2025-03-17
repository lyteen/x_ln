Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\core_cm1.h`

好的，我们重新开始。这次我将把 CMSIS 头文件的重要部分分解成更小的代码片段，附带中文描述和简单的用例。

**1. I/O 访问限定符 (I/O Access Qualifiers)**

```c
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

// 以下定义应该用于结构体成员
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */

```

**描述 (描述):**

*   这些宏定义用于指定对外设寄存器的访问权限。`volatile` 关键字告诉编译器，这个变量的值可能会在编译器无法预料的情况下发生改变，因此每次都应该直接从内存读取或写入，而不是使用缓存的值。
*   `__I` 表示只读 (Read-Only)。
*   `__O` 表示只写 (Write-Only)。
*   `__IO` 表示可读可写 (Read/Write)。
*   `__IM`、`__OM`、`__IOM` 用于结构体成员，作用类似。

**简单示例 (简单示例):**

```c
typedef struct {
  __IO uint32_t DATA;    // 数据寄存器，可读写
  __I  uint32_t STATUS;  // 状态寄存器，只读
  __O  uint32_t CONTROL; // 控制寄存器，只写
} MyPeripheral_TypeDef;

MyPeripheral_TypeDef *MyPeripheral = (MyPeripheral_TypeDef *)0x40000000; // 假设外设地址

void myFunction(void) {
  MyPeripheral->CONTROL = 0x01; // 写控制寄存器
  uint32_t status = MyPeripheral->STATUS; // 读状态寄存器
  MyPeripheral->DATA = 123; // 写数据寄存器
  uint32_t dataRead = MyPeripheral->DATA; //读数据寄存器
}
```

**中文解释 (中文解释):**

这段代码定义了一个名为 `MyPeripheral_TypeDef` 的结构体，用于表示一个外设的寄存器。`DATA` 寄存器是可读写的，`STATUS` 寄存器是只读的，`CONTROL` 寄存器是只写的。在 `myFunction` 函数中，我们通过指向外设基地址的指针来访问这些寄存器。

---

**2. 应用程序状态寄存器 (APSR):**

```c
typedef union
{
  struct
  {
    uint32_t _reserved0:28;              /*!< bit:  0..27  Reserved */
    uint32_t V:1;                        /*!< bit:     28  Overflow condition code flag */
    uint32_t C:1;                        /*!< bit:     29  Carry condition code flag */
    uint32_t Z:1;                        /*!< bit:     30  Zero condition code flag */
    uint32_t N:1;                        /*!< bit:     31  Negative condition code flag */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} APSR_Type;

/* APSR Register Definitions */
#define APSR_N_Pos                         31U                                            /*!< APSR: N Position */
#define APSR_N_Msk                         (1UL << APSR_N_Pos)                            /*!< APSR: N Mask */

#define APSR_Z_Pos                         30U                                            /*!< APSR: Z Position */
#define APSR_Z_Msk                         (1UL << APSR_Z_Pos)                            /*!< APSR: Z Mask */

#define APSR_C_Pos                         29U                                            /*!< APSR: C Position */
#define APSR_C_Msk                         (1UL << APSR_C_Pos)                            /*!< APSR: C Mask */

#define APSR_V_Pos                         28U                                            /*!< APSR: V Position */
#define APSR_V_Msk                         (1UL << APSR_V_Pos)                            /*!< APSR: V Mask */
```

**描述 (描述):**

*   `APSR_Type` 是一个联合体 (union)，允许你以两种方式访问应用程序状态寄存器：作为一个 32 位的字 (`w`) 或者作为一个包含多个位域的结构体 (`b`)。
*   这些位域表示不同的条件标志：
    *   `N`: 负数标志 (Negative flag)
    *   `Z`: 零标志 (Zero flag)
    *   `C`: 进位标志 (Carry flag)
    *   `V`: 溢出标志 (Overflow flag)
*   宏定义 `APSR_N_Pos`、`APSR_N_Msk` 等用于方便地访问和操作这些标志位。

**简单示例 (简单示例):**

```c
APSR_Type apsr_value;

// 假设我们从某个地方读取了 APSR 的值
uint32_t raw_apsr = __get_PSR(); // 获取程序状态寄存器的值（这个函数是平台相关的）
apsr_value.w = raw_apsr;

if (apsr_value.b.Z) {
  // 如果零标志被设置，执行某些操作
  printf("结果为零!\n");
}

if (apsr_value.b.C) {
    //如果进位标志被设置，执行某些操作
    printf("发生进位!\n");
}

// 清除进位标志 (不推荐直接修改 APSR，这里只是为了演示)
// apsr_value.b.C = 0; //直接修改可能会导致问题，推荐使用特定的指令
// __set_PSR(apsr_value.w);

```

**中文解释 (中文解释):**

这段代码演示了如何使用 `APSR_Type` 联合体来读取和访问应用程序状态寄存器的各个标志位。首先，我们从硬件读取 APSR 的原始值，然后将其存储到 `apsr_value` 联合体中。 接着，我们可以使用 `apsr_value.b.Z` 来检查零标志是否被设置。  请注意，直接修改 APSR 寄存器通常是不推荐的，因为它可能会影响程序的正确执行。应该使用特定的指令或函数来修改状态标志。

---

**3. 中断向量表 (Interrupt Vector Table)**

```c
__STATIC_INLINE void __NVIC_SetVector(IRQn_Type IRQn, uint32_t vector)
{
  uint32_t *vectors = (uint32_t *)0x0U;
  vectors[(int32_t)IRQn + NVIC_USER_IRQ_OFFSET] = vector;
}

__STATIC_INLINE uint32_t __NVIC_GetVector(IRQn_Type IRQn)
{
  uint32_t *vectors = (uint32_t *)0x0U;
  return vectors[(int32_t)IRQn + NVIC_USER_IRQ_OFFSET];
}
```

**描述:**

*   `__NVIC_SetVector` 用于设置中断向量表中特定中断的地址。 中断向量表通常位于 SRAM 的起始地址 `0x0U`。
*   `__NVIC_GetVector` 用于获取中断向量表中特定中断的地址。
*   `IRQn` 是中断号，`vector` 是中断处理函数的地址。
*   `NVIC_USER_IRQ_OFFSET` 是一个偏移量，用于将中断号映射到向量表中的正确位置。

**简单示例:**

```c
void MyInterruptHandler(void) {
  // 中断处理代码
  printf("中断发生了!\n");
}

int main(void) {
  // 设置中断向量
  __NVIC_SetVector(EXTI0_IRQn, (uint32_t)MyInterruptHandler);

  // ... 其他初始化代码 ...

  // 启动中断
  NVIC_EnableIRQ(EXTI0_IRQn);

  while (1) {
    // 主循环
  }
}
```

**中文解释:**

*   这段代码演示了如何使用 `__NVIC_SetVector` 函数来设置中断向量表。 首先，我们定义一个名为 `MyInterruptHandler` 的中断处理函数。 然后，我们使用 `__NVIC_SetVector` 函数将 `EXTI0_IRQn` 中断向量设置为 `MyInterruptHandler` 函数的地址。
*   `EXTI0_IRQn` 是一个定义在特定于设备的头文件中的常量，表示外部中断 0。
*   `NVIC_EnableIRQ(EXTI0_IRQn)` 启用外部中断0.
*   当外部中断 0 发生时，处理器将跳转到 `MyInterruptHandler` 函数执行。

---
希望这个更细致的分解和中文解释对您有所帮助！ 如果您有任何其他问题，请随时提出。
