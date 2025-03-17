Lan: `c` From`Micro/Dummy-Robot/2.Firmware/Ctrl-Step-Driver-STM32F1-fw\Core\Src\syscalls.c`

好的，我们重新开始。

这个 `syscalls.c` 文件是针对嵌入式系统（特别是基于 STM32 的系统）的一个关键组件。它实现了 Newlib C 库所需的底层系统调用，Newlib 是一个为嵌入式环境设计的 C 标准库。 由于嵌入式系统通常没有像桌面操作系统那样的完整操作系统，因此需要提供这些系统调用的自定义实现。

**代码总览:**

此文件包含标准 C 库函数所需的各种系统调用的存根实现。 重要的是要注意，这些实现中的许多实现都返回错误或不执行任何有意义的操作，因为它们针对的是资源受限且没有完整操作系统的嵌入式环境。 这允许使用标准 C 库函数，而无需完整的操作系统支持。

现在，让我们分解代码的关键部分并提供解释和示例：

**1. 头文件包含:**

```c
#include <sys/stat.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
```

*   **描述:**  这些头文件包含标准 C 库中定义的各种函数和数据结构的声明。例如，`<stdio.h>` 提供诸如 `printf` 和 `scanf` 等函数，而 `<stdlib.h>` 提供诸如 `malloc` 和 `free` 等函数。 `<errno.h>` 定义了 `errno` 变量，该变量用于指示系统调用中的错误。
*   **用途:**  提供 C 标准库函数的声明和定义，使代码可以使用这些函数。

**2. 变量:**

```c
//#undef errno
extern int errno;
extern int __io_putchar(int ch) __attribute__((weak));
extern int __io_getchar(void) __attribute__((weak));

register char * stack_ptr asm("sp");

char *__env[1] = { 0 };
char **environ = __env;
```

*   **`errno`**:  一个全局变量，用于存储系统调用返回的错误代码。 当系统调用失败时，它会设置 `errno` 以指示错误类型。
*   **`__io_putchar` 和 `__io_getchar`**:  这些是弱链接的函数，用于字符的输入和输出。`__attribute__((weak))` 属性允许用户提供自己的实现，如果用户未提供，则使用默认的空实现。它们通常由特定于硬件的驱动程序提供。
*   **`stack_ptr`**:  用于跟踪当前堆栈指针的寄存器变量。 这主要用于防止 `_sbrk` 函数中的堆栈溢出。
*   **`environ`**:  指向环境变量数组的指针。 在此代码中，它被初始化为一个空数组，因为嵌入式系统通常没有环境变量的概念。

**3. `initialise_monitor_handles()`:**

```c
void initialise_monitor_handles()
{
}
```

*   **描述:**  这个函数通常用于初始化监视器（例如，用于调试）。 在这个最小的实现中，它不执行任何操作。
*   **用途:**  可以扩展此函数以执行任何特定于调试器的初始化。

**4. `_getpid()`:**

```c
int _getpid(void)
{
    return 1;
}
```

*   **描述:**  返回当前进程 ID。 在单进程嵌入式系统中，通常返回一个固定值（此处为 1）。
*   **用途:**  某些 C 库函数可能会使用进程 ID，即使在单进程系统中。

**5. `_kill()` 和 `_exit()`:**

```c
int _kill(int pid, int sig)
{
    errno = EINVAL;
    return -1;
}

void _exit (int status)
{
    _kill(status, -1);
    while (1) {}		/* Make sure we hang here */
}
```

*   **`_kill()`**:  向进程发送信号。 在此实现中，始终返回错误，因为不支持信号。
*   **`_exit()`**:  终止当前进程。 在此实现中，它首先尝试使用无效信号调用 `_kill()`，然后进入无限循环，有效地停止执行。
*   **用途:**  `_exit` 用于干净地退出程序，而 `_kill` (如果实现) 用于发送信号到进程。

**6. `_read()` 和 `_write()`:**

```c
__attribute__((weak)) int _read(int file, char *ptr, int len)
{
    int DataIdx;

    for (DataIdx = 0; DataIdx < len; DataIdx++)
    {
        *ptr++ = __io_getchar();
    }

    return len;
}

__attribute__((weak)) int _write(int file, char *ptr, int len)
{
    int DataIdx;

    for (DataIdx = 0; DataIdx < len; DataIdx++)
    {
        __io_putchar(*ptr++);
    }
    return len;
}
```

*   **`_read()`**: 从文件描述符读取数据。 在此实现中，它从 `__io_getchar()` 读取 `len` 个字符并将它们存储在缓冲区 `ptr` 中。 `__io_getchar()` 应该由用户提供（通常使用 UART），以便从串行端口获取输入。
*   **`_write()`**: 将数据写入文件描述符。 在此实现中，它将 `len` 个字符从缓冲区 `ptr` 写入到 `__io_putchar()`。 `__io_putchar()` 应该由用户提供（通常使用 UART），以便将输出发送到串行端口。
*   **用途:**  这些函数提供基本的输入/输出功能。 `_read` 用于从输入设备读取数据，而 `_write` 用于将数据写入输出设备。  `__attribute__((weak))` 表示你可以重新定义这些函数以适应你的硬件。

**7. `_sbrk()`:**

```c
caddr_t _sbrk(int incr)
{
    extern char end asm("end");
    static char *heap_end;
    char *prev_heap_end;

    if (heap_end == 0)
        heap_end = &end;

    prev_heap_end = heap_end;
    if (heap_end + incr > stack_ptr)
    {
//		write(1, "Heap and stack collision\n", 25);
//		abort();
        errno = ENOMEM;
        return (caddr_t) -1;
    }

    heap_end += incr;

    return (caddr_t) prev_heap_end;
}
```

*   **描述:**  `_sbrk()`（“增加程序中断”）用于分配动态内存。 它通过增加堆的大小来实现。
*   **工作原理:**
    *   `end`: 这是一个链接器符号，表示程序的数据段的末尾。 堆从这里开始。
    *   `heap_end`: 一个静态变量，用于跟踪堆的当前末尾。
    *   该函数首先检查 `heap_end` 是否已初始化。 如果没有，则将其设置为 `end`。
    *   然后，它检查增加堆的大小是否会导致堆栈溢出。 如果是，则返回错误。
    *   否则，它会增加 `heap_end` 并返回堆的前一个末尾。
*   **用途:**  `malloc()` 和 `free()` 等函数使用 `_sbrk()` 来分配和释放内存。

**8. 其他系统调用存根:**

```c
int _open(char *path, int flags, ...)
{
    /* Pretend like we always fail */
    return -1;
}

int _wait(int *status)
{
    errno = ECHILD;
    return -1;
}

int _unlink(char *name)
{
    errno = ENOENT;
    return -1;
}

int _times(struct tms *buf)
{
    return -1;
}

int _stat(char *file, struct stat *st)
{
    st->st_mode = S_IFCHR;
    return 0;
}

int _link(char *old, char *new)
{
    errno = EMLINK;
    return -1;
}

int _fork(void)
{
    errno = EAGAIN;
    return -1;
}

int _execve(char *name, char **argv, char **env)
{
    errno = ENOMEM;
    return -1;
}
```

*   **描述:**  这些是其他系统调用的存根实现。 它们通常返回错误，因为这些操作在典型的嵌入式系统中不受支持。
*   **用途:**  提供 C 库函数所需的最小系统调用集，即使它们未完全实现。  这允许代码编译而无需所有操作系统功能。

**代码用法:**

1.  **编译:**  此文件与你的 C 代码一起编译，以创建嵌入式应用程序。
2.  **链接:**  在链接过程中，链接器将解析对标准 C 库函数的调用，并将其链接到此文件中提供的系统调用实现。
3.  **执行:**  当你的程序调用 C 库函数（例如 `printf`、`malloc` 等）时，这些函数最终会调用此文件中定义的系统调用。

**简单示例：**

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Hello, embedded world!\n");
    int *ptr = (int*) malloc(sizeof(int));
    if (ptr != NULL) {
        *ptr = 10;
        printf("Value: %d\n", *ptr);
        free(ptr);
    } else {
        printf("Memory allocation failed!\n");
    }
    return 0;
}

// 假设你提供了 __io_putchar() 的实现，例如：
int __io_putchar(int ch) {
    // 发送字符 'ch' 到 UART (例如)
    // 具体的代码取决于你的硬件
    // 例如:
    // HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch; // 返回发送的字符
}

// 假设你提供了 __io_getchar() 的实现，例如：
int __io_getchar(void) {
    // 从 UART 接收一个字符 (例如)
    // 具体的代码取决于你的硬件
    // 例如:
    // uint8_t ch;
    // HAL_UART_Receive(&huart1, &ch, 1, HAL_MAX_DELAY);
    // return ch;
    return 0; //返回接收到的字符，此处为了简单直接返回0
}
```

在这个例子中：

*   `printf` 调用使用 `_write` 系统调用来将文本输出到控制台（通过 `__io_putchar`）。
*   `malloc` 调用使用 `_sbrk` 系统调用来分配内存。
*   `free` 调用 (尽管没有实际释放内存，因为没有释放内存的实现，在某些实现中会将`_sbrk`的指针减少)。

**总结:**

`syscalls.c` 文件是使标准 C 库能够在没有完整操作系统的嵌入式系统上使用的桥梁。 它提供了 C 库函数所需的最小系统调用集。 大多数实现都是存根，有些需要针对特定硬件进行自定义，特别是 `_read`、`_write` 以及与动态内存分配相关的函数。理解这个文件对于嵌入式开发至关重要，特别是当你需要使用标准 C 库函数时。
