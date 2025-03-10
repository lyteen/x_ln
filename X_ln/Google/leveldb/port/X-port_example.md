Lan: `h` From`Google/leveldb\port\port_example.h`

OK, let's start over with a practical example of a `port.h` implementation, focusing on providing a basic (but functional) implementation.  This example will be tailored for a hypothetical POSIX-like system (Linux, macOS, etc.). I will also provide a simple explanation in Chinese for each part.

```c++
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// This file contains the specification, but not the implementations,
// of the types/operations/etc. that should be defined by a platform
// specific port_<platform>.h file.  Use this file as a reference for
// how to port this package to a new platform.

#ifndef STORAGE_LEVELDB_PORT_PORT_POSIX_H_ // Changed to POSIX
#define STORAGE_LEVELDB_PORT_PORT_POSIX_H_

#include <pthread.h>  // For threading primitives
#include <stddef.h>   // For size_t
#include <stdint.h>  // For uint32_t
#include <string>     // For std::string
#include <snappy.h>   // For Snappy compression
#include <zstd.h>     // For Zstd compression (you'll need to install zstd)
#include <stdlib.h> // For malloc/free
#include <stdio.h>  // For printf (for debugging)
#include <atomic> // For Atomic counter for crc

#include "port/thread_annotations.h"  // Already included
#include <immintrin.h> // Include for crc32c intrinsic

namespace leveldb {
namespace port {

// ------------------ Threading -------------------

// A Mutex represents an exclusive lock.
class LOCKABLE Mutex {
 private:
  pthread_mutex_t mu_;

 public:
  Mutex() {
    pthread_mutex_init(&mu_, nullptr);
  }
  ~Mutex() {
    pthread_mutex_destroy(&mu_);
  }

  // Lock the mutex.  Waits until other lockers have exited.
  // Will deadlock if the mutex is already locked by this thread.
  void Lock() EXCLUSIVE_LOCK_FUNCTION() {
    pthread_mutex_lock(&mu_);
  }

  // Unlock the mutex.
  // REQUIRES: This mutex was locked by this thread.
  void Unlock() UNLOCK_FUNCTION() {
    pthread_mutex_unlock(&mu_);
  }

  // Optionally crash if this thread does not hold this mutex.
  // The implementation must be fast, especially if NDEBUG is
  // defined.  The implementation is allowed to skip all checks.
  void AssertHeld() ASSERT_EXCLUSIVE_LOCK() {
#ifndef NDEBUG
    // Very basic check.  Not foolproof, but better than nothing.
    // In a real port, you'd use pthread_mutex_trylock and check the result.
     int result = pthread_mutex_trylock(&mu_);
     if (result == 0) {
       pthread_mutex_unlock(&mu_); // Restore the mutex.
     } else {
       // Assume the mutex is held if trylock fails. This can be incorrect in cases other than "already locked", but it's an OK approximation for debugging.
       return;
     }

    fprintf(stderr, "Mutex::AssertHeld failed.  Mutex not held by current thread.\n");
    abort(); // Crash.
#endif
  }
};

// Condition Variable
class CondVar {
 private:
  pthread_cond_t cond_;
  Mutex* mu_; // Associated mutex

 public:
  explicit CondVar(Mutex* mu) : mu_(mu) {
    pthread_cond_init(&cond_, nullptr);
  }
  ~CondVar() {
    pthread_cond_destroy(&cond_);
  }

  // Atomically release *mu and block on this condition variable until
  // either a call to SignalAll(), or a call to Signal() that picks
  // this thread to wakeup.
  // REQUIRES: this thread holds *mu
  void Wait() {
    pthread_cond_wait(&cond_, &mu_->mu_);
  }

  // If there are some threads waiting, wake up at least one of them.
  void Signal() {
    pthread_cond_signal(&cond_);
  }

  // Wake up all waiting threads.
  void SignalAll() {
    pthread_cond_broadcast(&cond_);
  }
};

// ------------------ Compression -------------------

bool Snappy_Compress(const char* input, size_t input_length,
                     std::string* output) {
  output->resize(snappy::MaxCompressedLength(input_length));
  size_t compressed_length;
  snappy::RawCompress(input, input_length, &((*output)[0]), &compressed_length);
  output->resize(compressed_length);
  return true;
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
  return snappy::GetUncompressedLength(input, length, result);
}

bool Snappy_Uncompress(const char* input_data, size_t input_length,
                       char* output) {
  return snappy::RawUncompress(input_data, input_length, output);
}

bool Zstd_Compress(int level, const char* input, size_t input_length,
                   std::string* output) {
  size_t const max_compressed_size = ZSTD_compressBound(input_length);
  output->resize(max_compressed_size);
  size_t compressed_size = ZSTD_compress(&(*output)[0], max_compressed_size,
                                         input, input_length, level);
  if (ZSTD_isError(compressed_size)) {
    return false; // Or handle the error more gracefully
  }
  output->resize(compressed_size);
  return true;
}

bool Zstd_GetUncompressedLength(const char* input, size_t length,
                                size_t* result) {
  unsigned long long const decompressed_size = ZSTD_getFrameContentSize(input, length);
  if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
    return false;
  }
  *result = static_cast<size_t>(decompressed_size);
  return true;
}

bool Zstd_Uncompress(const char* input_data, size_t input_length, char* output) {
  size_t const decompressed_size = ZSTD_getFrameContentSize(input_data, input_length);
  if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
    return false;
  }

  size_t const result = ZSTD_decompress(output, decompressed_size, input_data, input_length);
  if (ZSTD_isError(result)) {
    return false;
  }
  return true;
}

// ------------------ Miscellaneous -------------------

bool GetHeapProfile(void (*func)(void*, const char*, int), void* arg) {
  // Heap profiling is often platform-specific and requires tools like gperftools.
  // This is a placeholder. You would need to use the appropriate API
  // for your system.
  // Example using gperftools (if available):
  // #ifdef HAVE_GPERFTOOLS
  //   HeapProfilerStart("leveldb_heap_profile");
  //   // ... profiling logic ...
  //   HeapProfilerStop();
  // #endif
  return false; // Indicate that heap profiling is not supported in this example.
}


uint32_t AcceleratedCRC32C(uint32_t crc, const char* buf, size_t size) {
  crc = ~crc;
  for (size_t i = 0; i < size; ++i) {
    crc = _mm_crc32_u8(crc, buf[i]);
  }
  return ~crc;
}


}  // namespace port
}  // namespace leveldb

#endif  // STORAGE_LEVELDB_PORT_PORT_POSIX_H_
```

**Explanation (中文解释):**

*   **`#ifndef STORAGE_LEVELDB_PORT_PORT_POSIX_H_ ... #define STORAGE_LEVELDB_PORT_PORT_POSIX_H_`**:  防止头文件被重复包含，这是C++编程中的标准做法。(Prevent multiple inclusions of the header file, a standard practice in C++ programming.)
*   **`#include <pthread.h>`**: 包含 POSIX 线程库的头文件，用于互斥锁和条件变量。(Includes the POSIX threads library header file for mutexes and condition variables.)
*   **`#include <stddef.h>`**, **`#include <stdint.h>`**, **`#include <string>`**:  包含标准 C++ 头文件，用于 `size_t`、`uint32_t` 和 `std::string`。(Includes standard C++ header files for `size_t`, `uint32_t`, and `std::string`.)
*   **`#include <snappy.h>`**:  包含 Snappy 压缩库的头文件。确保已经安装了 Snappy 库。(Includes the Snappy compression library header file.  Make sure you have the Snappy library installed.)
*   **`#include <zstd.h>`**: 包含 Zstd 压缩库的头文件。 需要安装zstd库。(Includes the Zstd compression library header file.  You need to install the zstd library.)
*   **`#include "port/thread_annotations.h"`**: 包含用于线程安全注解的头文件，用于帮助进行静态分析。(Includes the header file for thread-safety annotations, which helps with static analysis.)

*   **`namespace leveldb { namespace port { ... } }`**:  将所有代码放在 `leveldb::port` 命名空间中，避免命名冲突。(Puts all the code in the `leveldb::port` namespace to avoid naming conflicts.)

*   **`class Mutex`**:
    *   `pthread_mutex_t mu_`:  互斥锁的实际存储。(The actual storage for the mutex.)
    *   `Mutex()`, `~Mutex()`: 构造函数和析构函数，用于初始化和销毁互斥锁。(Constructor and destructor to initialize and destroy the mutex.)
    *   `Lock()`, `Unlock()`:  加锁和解锁互斥锁。(Lock and unlock the mutex.)
    *   `AssertHeld()`:  (仅在调试模式下) 检查当前线程是否持有锁。这个简单实现尝试锁定互斥锁，如果成功，则立即解锁；如果失败，则假设锁被持有（可能不完全准确，但对于调试目的来说足够了）。( (Debug mode only) Checks if the current thread holds the lock. This simple implementation attempts to lock the mutex, and if successful, unlocks it immediately; if it fails, it assumes the lock is held (may not be perfectly accurate, but sufficient for debugging purposes).)

*   **`class CondVar`**:
    *   `pthread_cond_t cond_`: 条件变量的实际存储。(The actual storage for the condition variable.)
    *   `Mutex* mu_`:  与条件变量关联的互斥锁。(The mutex associated with the condition variable.)
    *   `CondVar(Mutex* mu)`, `~CondVar()`: 构造函数和析构函数，用于初始化和销毁条件变量。(Constructor and destructor to initialize and destroy the condition variable.)
    *   `Wait()`, `Signal()`, `SignalAll()`:  等待、发出信号和发出所有信号。(Wait, signal, and signal all.)

*   **`Snappy_Compress`, `Snappy_GetUncompressedLength`, `Snappy_Uncompress`**:  使用 Snappy 库进行压缩和解压缩。(Uses the Snappy library for compression and decompression.)

*   **`Zstd_Compress`, `Zstd_GetUncompressedLength`, `Zstd_Uncompress`**:  使用 Zstd 库进行压缩和解压缩。(Uses the Zstd library for compression and decompression.)  Requires the Zstd library to be installed.
*   **`GetHeapProfile`**:  一个占位符，表示堆分析功能。通常需要特定于平台的工具（例如 gperftools）。(A placeholder for heap profiling functionality. This usually requires platform-specific tools (e.g., gperftools).)

*   **`AcceleratedCRC32C`**: Implements accelerated CRC32C using Intel's `_mm_crc32_u8` intrinsic. Requires a CPU that supports the SSE4.2 instruction set.

**如何使用 (How to Use):**

1.  **Install Snappy and Zstd:**  Make sure you have the Snappy and Zstd libraries installed on your system.  On Debian/Ubuntu: `sudo apt-get install libsnappy-dev libzstd-dev`. On macOS with Homebrew: `brew install snappy zstd`.
2.  **Compile:**  Compile your LevelDB code, including this `port_posix.h` file. You might need to add `-lsnappy` and `-lzstd` to your linker flags.
3.  **Test:**  Write some test code that uses the mutexes, condition variables, and compression functions to ensure they are working correctly.
4.  **Heap Profiling:**  If you want to implement heap profiling, you'll need to integrate with a platform-specific profiling tool (like gperftools) and modify the `GetHeapProfile` function accordingly.
5. **CRC32C**: Check that your CPU supports the SSE4.2 instruction set. Most modern CPUs support it.

**Example Test Code (示例测试代码):**

```c++
#include "port/port_posix.h"
#include <iostream>
#include <thread>

int main() {
  leveldb::port::Mutex mutex;
  leveldb::port::CondVar cond(&mutex);
  bool data_ready = false;

  std::thread t1([&]() {
    mutex.Lock();
    while (!data_ready) {
      std::cout << "Thread 1 waiting...\n";
      cond.Wait();
    }
    std::cout << "Thread 1 received signal.\n";
    mutex.Unlock();
  });

  std::thread t2([&]() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    mutex.Lock();
    data_ready = true;
    std::cout << "Thread 2 sending signal.\n";
    cond.Signal();
    mutex.Unlock();
  });

  t1.join();
  t2.join();

  std::string original_data = "This is some data to compress.";
  std::string compressed_data;
  leveldb::port::Snappy_Compress(original_data.data(), original_data.size(), &compressed_data);
  std::cout << "Original size: " << original_data.size() << ", Compressed size: " << compressed_data.size() << "\n";

  return 0;
}
```

**Important Considerations (重要注意事项):**

*   **Error Handling:** The compression functions, and other functions, should have more robust error handling.  This example uses basic error checking.  In a production environment, you'd want to provide more detailed error information.
*   **Platform-Specific Code:** This example is tailored for POSIX-like systems.  For Windows, you would need to use Windows threading primitives and potentially different compression libraries.
*   **Optimization:** The implementations provided are basic.  You can further optimize them for performance if needed.

This comprehensive example provides a solid starting point for implementing a `port.h` file for LevelDB on a POSIX-compatible system.  Remember to adapt it to your specific platform and requirements.