Lan: `h` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Drivers\CMSIS\Include\core_cm4.h`

好的，我们重新开始，目标是针对 `core_cm4.h` 文件中的重要组成部分进行更详细的解读，并且提供示例代码，加上中文描述。

**1. IO 类型限定符 (IO Type Qualifiers)**

```c
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

/* following defines should be used for structure members */
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */
```

**描述:** 这些宏定义用于指定对外设寄存器的访问权限。`volatile` 关键字确保编译器每次都从内存中读取变量，而不是使用缓存值。

*   `__I`: 只读 (Read Only)。
*   `__O`: 只写 (Write Only)。
*   `__IO`: 读写 (Read/Write)。
*   `__IM`, `__OM`, `__IOM`: 用于结构体成员，分别对应只读、只写和读写权限。

**示例:**

```c
typedef struct {
  __IO uint32_t CR;   // 控制寄存器 (Control Register), 可读写
  __I  uint32_t SR;   // 状态寄存器 (Status Register), 只读
  __O  uint32_t DR;   // 数据寄存器 (Data Register), 只写
} MyPeripheral_TypeDef;

MyPeripheral_TypeDef *MyPeripheral = (MyPeripheral_TypeDef *)0x40000000; // 假设外设地址

int main() {
  MyPeripheral->CR = 0x01;  // 正确：写入控制寄存器
  uint32_t status = MyPeripheral->SR; // 正确：读取状态寄存器
  // MyPeripheral->SR = 0x02;  // 错误：不能写入状态寄存器（只读）
}
```

**中文描述:**  这些宏定义就像给外设寄存器加上了访问权限标签。`__I` 表示这个寄存器是“只读”的，你的程序只能读取它的值，不能修改。`__O` 表示这个寄存器是“只写”的，你的程序只能向它写入数据，不能读取它的当前值。`__IO` 表示这个寄存器是“读写”的，你的程序既可以读取它的值，也可以修改它。`volatile` 关键字告诉编译器，这个寄存器的值可能会随时改变，不要进行优化，每次都必须从实际的硬件地址读取。 这样做可以确保你在操作硬件时，程序行为和你预期的一致。

---

**2. NVIC (Nested Vectored Interrupt Controller) 结构体**

```c
typedef struct
{
  __IOM uint32_t ISER[8U];               /*!< Offset: 0x000 (R/W)  Interrupt Set Enable Register */
        uint32_t RESERVED0[24U];
  __IOM uint32_t ICER[8U];               /*!< Offset: 0x080 (R/W)  Interrupt Clear Enable Register */
        uint32_t RSERVED1[24U];
  __IOM uint32_t ISPR[8U];               /*!< Offset: 0x100 (R/W)  Interrupt Set Pending Register */
        uint32_t RESERVED2[24U];
  __IOM uint32_t ICPR[8U];               /*!< Offset: 0x180 (R/W)  Interrupt Clear Pending Register */
        uint32_t RESERVED3[24U];
  __IOM uint32_t IABR[8U];               /*!< Offset: 0x200 (R/W)  Interrupt Active bit Register */
        uint32_t RESERVED4[56U];
  __IOM uint8_t  IP[240U];               /*!< Offset: 0x300 (R/W)  Interrupt Priority Register (8Bit wide) */
        uint32_t RESERVED5[644U];
  __OM  uint32_t STIR;                   /*!< Offset: 0xE00 ( /W)  Software Trigger Interrupt Register */
}  NVIC_Type;
```

**描述:** 这个结构体定义了 NVIC 的寄存器布局。 NVIC 用于管理中断，允许处理器响应外部事件。

*   `ISER`: 中断使能寄存器 (Interrupt Set Enable Register)。用于使能中断。
*   `ICER`: 中断清除使能寄存器 (Interrupt Clear Enable Register)。用于禁用中断。
*   `ISPR`: 中断设置挂起寄存器 (Interrupt Set Pending Register)。用于手动设置中断为挂起状态。
*   `ICPR`: 中断清除挂起寄存器 (Interrupt Clear Pending Register)。用于清除中断的挂起状态。
*   `IABR`: 中断活动位寄存器 (Interrupt Active Bit Register)。指示当前正在处理的中断。
*   `IP`: 中断优先级寄存器 (Interrupt Priority Register)。设置中断的优先级。
*   `STIR`: 软件触发中断寄存器 (Software Trigger Interrupt Register)。用于软件触发中断。

**示例:**

```c
#define NVIC                ((NVIC_Type      *)     NVIC_BASE     )   /*!< NVIC configuration struct */
#define EXTI0_IRQn 6 //Example IRQ number for external interrupt 0

int main() {
  NVIC->ISER[0] = (1 << EXTI0_IRQn); // 使能 EXTI0 中断
  NVIC->IP[EXTI0_IRQn] = 0x80;       // 设置 EXTI0 中断优先级为 0x80 (示例值)
  NVIC->STIR = EXTI0_IRQn;             // 软件触发 EXTI0 中断
}

void EXTI0_IRQHandler(void) {
  // 中断处理程序
  // ...
}
```

**中文描述:** NVIC 就像一个交通调度员，管理着各种中断事件。 `ISER` 相当于给某个交通路口开启信号灯，允许车辆通过（使能中断）。 `ICER` 相当于关闭信号灯，禁止车辆通过（禁用中断）。 `ISPR` 相当于手动按下一个路口的按钮，让交通信号灯立即切换到允许方向（设置中断挂起）。 `ICPR` 相当于取消手动按钮，让交通信号灯恢复正常（清除中断挂起）。 `IP` 相当于给不同的交通路口分配优先级，让更重要的路口优先放行（设置中断优先级）。 `STIR` 相当于让一个程序员用代码模拟一个外部事件发生，触发一个特定的中断（软件触发中断）。 `EXTI0_IRQHandler` 是真正处理中断的函数，当EXTI0中断发生，程序就会跳转到这个函数执行。

---

**3. SCB (System Control Block) 结构体**

```c
typedef struct
{
  __IM  uint32_t CPUID;                  /*!< Offset: 0x000 (R/ )  CPUID Base Register */
  __IOM uint32_t ICSR;                   /*!< Offset: 0x004 (R/W)  Interrupt Control and State Register */
  __IOM uint32_t VTOR;                   /*!< Offset: 0x008 (R/W)  Vector Table Offset Register */
  __IOM uint32_t AIRCR;                  /*!< Offset: 0x00C (R/W)  Application Interrupt and Reset Control Register */
  __IOM uint32_t SCR;                    /*!< Offset: 0x010 (R/W)  System Control Register */
  __IOM uint32_t CCR;                    /*!< Offset: 0x014 (R/W)  Configuration Control Register */
  __IOM uint8_t  SHP[12U];               /*!< Offset: 0x018 (R/W)  System Handlers Priority Registers (4-7, 8-11, 12-15) */
  __IOM uint32_t SHCSR;                  /*!< Offset: 0x024 (R/W)  System Handler Control and State Register */
  __IOM uint32_t CFSR;                   /*!< Offset: 0x028 (R/W)  Configurable Fault Status Register */
  __IOM uint32_t HFSR;                   /*!< Offset: 0x02C (R/W)  HardFault Status Register */
  __IOM uint32_t DFSR;                   /*!< Offset: 0x030 (R/W)  Debug Fault Status Register */
  __IOM uint32_t MMFAR;                  /*!< Offset: 0x034 (R/W)  MemManage Fault Address Register */
  __IOM uint32_t BFAR;                   /*!< Offset: 0x038 (R/W)  BusFault Address Register */
  __IOM uint32_t AFSR;                   /*!< Offset: 0x03C (R/W)  Auxiliary Fault Status Register */
  __IM  uint32_t PFR[2U];                /*!< Offset: 0x040 (R/ )  Processor Feature Register */
  __IM  uint32_t DFR;                    /*!< Offset: 0x048 (R/ )  Debug Feature Register */
  __IM  uint32_t ADR;                    /*!< Offset: 0x04C (R/ )  Auxiliary Feature Register */
  __IM  uint32_t MMFR[4U];               /*!< Offset: 0x050 (R/ )  Memory Model Feature Register */
  __IM  uint32_t ISAR[5U];               /*!< Offset: 0x060 (R/ )  Instruction Set Attributes Register */
        uint32_t RESERVED0[5U];
  __IOM uint32_t CPACR;                  /*!< Offset: 0x088 (R/W)  Coprocessor Access Control Register */
} SCB_Type;
```

**描述:** SCB 包含了系统控制和配置相关的寄存器。

*   `CPUID`:  CPU ID 寄存器，只读，包含 CPU 的信息 (架构、厂商等)。
*   `ICSR`:  中断控制和状态寄存器，用于控制中断状态。
*   `VTOR`:  向量表偏移量寄存器，用于设置中断向量表的起始地址。
*   `AIRCR`:  应用中断和复位控制寄存器，用于控制系统复位和中断优先级分组。
*   `SCR`: 系统控制寄存器，用于控制睡眠模式等。
*   `CCR`: 配置控制寄存器， 用于配置一些系统行为，比如对齐错误的处理。
*   `SHP`: 系统处理程序优先级寄存器，用于设置系统异常的优先级。
*   `SHCSR`: 系统处理程序控制和状态寄存器，用于使能或禁用系统异常。
*   `CFSR`: 可配置故障状态寄存器， 包含了内存管理，总线错误，用法的状态信息
*   `HFSR`: 硬故障状态寄存器， 包含硬故障的状态信息。
*   `DFSR`: 调试故障状态寄存器， 包含调试相关的错误信息
*   `MMFAR`: 内存管理错误地址寄存器， 如果发生内存管理错误，那么错误的地址将会储存在这个寄存器里面
*   `BFAR`: 总线错误地址寄存器， 如果发生总线错误， 那么错误的地址将会储存在这个寄存器里面
*   `AFSR`: 辅助故障状态寄存器， 一些芯片厂商会用这个寄存器存储其他的状态信息
*    `CPACR`:  协处理器访问控制寄存器，用于控制对协处理器（例如FPU）的访问权限。

**示例:**

```c
#define SCB                 ((SCB_Type       *)     SCB_BASE      )   /*!< SCB configuration struct */

int main() {
  // 设置向量表地址到 SRAM (假设 SRAM 起始地址为 0x20000000)
  SCB->VTOR = 0x20000000;

  // 设置中断优先级分组 (3 bits for preemption priority, 1 bit for subpriority)
  SCB->AIRCR = (0x5FA << 16) | (0x05 << 8); // VECTKEY | PRIGROUP
  //启用除以零错误陷阱
  SCB->CCR |= SCB_CCR_DIV_0_TRP_Msk;
}

```

**中文描述:** SCB 就像一个中央控制面板，负责管理整个系统的核心功能。 `CPUID` 寄存器提供处理器的身份证信息。 `VTOR` 寄存器告诉处理器到哪里去寻找中断处理函数的列表 (向量表)。 `AIRCR` 寄存器用于配置系统复位的方式和中断的优先级分组。 `SCR` 寄存器可以控制处理器进入睡眠状态时的行为。通过修改这些寄存器的值，你可以调整系统的运行模式和响应外部事件的方式。`CPACR`可以控制对FPU之类的硬件的访问权限。

接下来， 我们继续讲解 `SysTick` 和 `ITM`，并提供更多示例。
