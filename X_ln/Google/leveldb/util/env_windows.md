Lan: `cc` From`Google/leveldb\util\env_windows.cc`

好的，这次我会更详细地解释代码的关键部分，并为每个小代码段添加中文描述，并提供使用方式和简单的示例。

**1. 预处理和头文件包含 (Preprocessing and Header Includes):**

```c++
// Copyright (c) 2018 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// Prevent Windows headers from defining min/max macros and instead
// use STL.
#ifndef NOMINMAX
#define NOMINMAX
#endif  // ifndef NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "leveldb/env.h"
#include "leveldb/slice.h"
#include "port/port.h"
#include "port/thread_annotations.h"
#include "util/env_windows_test_helper.h"
#include "util/logging.h"
#include "util/mutexlock.h"
#include "util/windows_logger.h"
```

**描述:**  这段代码主要进行一些预处理操作和包含必要的头文件。

*   `#ifndef NOMINMAX ... #endif`:  防止 Windows 头文件定义 `min` 和 `max` 宏，避免与 C++ 标准库中的 `std::min` 和 `std::max` 函数冲突。
*   `#include <windows.h>`:  包含 Windows API 头文件，用于访问 Windows 系统调用。
*   `#include <algorithm>`, `#include <atomic>` 等: 包含 C++ 标准库的头文件，提供各种常用的功能，例如算法、原子操作、时间处理、线程同步等。
*   `#include "leveldb/env.h"` 等: 包含 LevelDB 项目的头文件，定义了 LevelDB 的环境接口、切片（Slice）类、日志接口等。

**用途:** 这是代码的基础部分，确保可以使用所需的库和函数。

**2. 常量定义 (Constant Definitions):**

```c++
namespace leveldb {

namespace {

constexpr const size_t kWritableFileBufferSize = 65536;

// Up to 1000 mmaps for 64-bit binaries; none for 32-bit.
constexpr int kDefaultMmapLimit = (sizeof(void*) >= 8) ? 1000 : 0;

// Can be set by by EnvWindowsTestHelper::SetReadOnlyMMapLimit().
int g_mmap_limit = kDefaultMmapLimit;
```

**描述:** 这段代码定义了一些常量，用于配置 LevelDB 在 Windows 环境下的行为。

*   `kWritableFileBufferSize`: 可写文件的缓冲区大小，设置为 65536 字节。 这是写操作在写入磁盘之前在内存中缓冲的数据量。
*   `kDefaultMmapLimit`:  mmap (内存映射) 的默认限制。在 64 位系统上，允许最多 1000 个 mmap，而在 32 位系统上，不允许使用 mmap (设置为 0)。 使用mmap可以提高读取文件的性能。
*   `g_mmap_limit`:  全局变量，存储 mmap 的限制。 它可以被 `EnvWindowsTestHelper::SetReadOnlyMMapLimit()` 函数修改，用于测试目的。

**用途:** 这些常量影响 LevelDB 的性能和资源使用。 例如，较大的 `kWritableFileBufferSize` 可以提高写性能，但也会增加内存使用。`g_mmap_limit`  控制同时使用的内存映射文件数量，防止资源耗尽。

**3. 错误处理函数 (Error Handling Functions):**

```c++
std::string GetWindowsErrorMessage(DWORD error_code) {
  std::string message;
  char* error_text = nullptr;
  // Use MBCS version of FormatMessage to match return value.
  size_t error_text_size = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<char*>(&error_text), 0, nullptr);
  if (!error_text) {
    return message;
  }
  message.assign(error_text, error_text_size);
  ::LocalFree(error_text);
  return message;
}

Status WindowsError(const std::string& context, DWORD error_code) {
  if (error_code == ERROR_FILE_NOT_FOUND || error_code == ERROR_PATH_NOT_FOUND)
    return Status::NotFound(context, GetWindowsErrorMessage(error_code));
  return Status::IOError(context, GetWindowsErrorMessage(error_code));
}
```

**描述:**  这两个函数用于处理 Windows API 的错误。

*   `GetWindowsErrorMessage`:  将 Windows 错误代码 (DWORD) 转换为可读的字符串消息。  它使用 `FormatMessageA` 函数来检索系统提供的错误描述。
*   `WindowsError`:  根据 Windows 错误代码创建一个 LevelDB `Status` 对象。 如果错误代码是 `ERROR_FILE_NOT_FOUND` 或 `ERROR_PATH_NOT_FOUND`，则返回 `Status::NotFound`，否则返回 `Status::IOError`。

**用途:** LevelDB 使用 `Status` 对象来报告操作的结果。 这些函数允许 LevelDB 将 Windows API 错误转换为 LevelDB 错误，以便更好地进行错误处理。

**4.  ScopedHandle 类 (ScopedHandle Class):**

```c++
class ScopedHandle {
 public:
  ScopedHandle(HANDLE handle) : handle_(handle) {}
  ScopedHandle(const ScopedHandle&) = delete;
  ScopedHandle(ScopedHandle&& other) noexcept : handle_(other.Release()) {}
  ~ScopedHandle() { Close(); }

  ScopedHandle& operator=(const ScopedHandle&) = delete;

  ScopedHandle& operator=(ScopedHandle&& rhs) noexcept {
    if (this != &rhs) handle_ = rhs.Release();
    return *this;
  }

  bool Close() {
    if (!is_valid()) {
      return true;
    }
    HANDLE h = handle_;
    handle_ = INVALID_HANDLE_VALUE;
    return ::CloseHandle(h);
  }

  bool is_valid() const {
    return handle_ != INVALID_HANDLE_VALUE && handle_ != nullptr;
  }

  HANDLE get() const { return handle_; }

  HANDLE Release() {
    HANDLE h = handle_;
    handle_ = INVALID_HANDLE_VALUE;
    return h;
  }

 private:
  HANDLE handle_;
};
```

**描述:**  `ScopedHandle` 类是一个 RAII (Resource Acquisition Is Initialization) 风格的包装器，用于管理 Windows API 中的 HANDLE 对象。  它负责在对象销毁时自动关闭 HANDLE，防止资源泄漏。

*   构造函数:  获取一个 `HANDLE` 对象的所有权。
*   析构函数:  在对象销毁时调用 `Close()` 函数来关闭 `HANDLE`。
*   `Close()`:  关闭 `HANDLE` 对象。
*   `is_valid()`:  检查 `HANDLE` 是否有效。
*   `get()`:  返回原始的 `HANDLE` 对象。
* `Release()`: 释放句柄的所有权，防止 `ScopedHandle` 关闭它.

**用途:** 简化了 Windows API 中 HANDLE 对象的资源管理，减少了手动调用 `CloseHandle()` 导致错误的可能性。

**示例:**

```c++
ScopedHandle file_handle(CreateFileA("test.txt", GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
if (file_handle.is_valid()) {
  // 使用 file_handle 进行文件操作
  // ...
} // file_handle 在这里自动关闭
```

**5. Limiter 类 (Limiter Class):**

```c++
class Limiter {
 public:
  // Limit maximum number of resources to |max_acquires|.
  Limiter(int max_acquires)
      :
#if !defined(NDEBUG)
        max_acquires_(max_acquires),
#endif  // !defined(NDEBUG)
        acquires_allowed_(max_acquires) {
    assert(max_acquires >= 0);
  }

  Limiter(const Limiter&) = delete;
  Limiter operator=(const Limiter&) = delete;

  // If another resource is available, acquire it and return true.
  // Else return false.
  bool Acquire() {
    int old_acquires_allowed =
        acquires_allowed_.fetch_sub(1, std::memory_order_relaxed);

    if (old_acquires_allowed > 0) return true;

    acquires_allowed_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  // Release a resource acquired by a previous call to Acquire() that returned
  // true.
  void Release() {
    int old_acquires_allowed =
        acquires_allowed_.fetch_add(1, std::memory_order_relaxed);

    // Silence compiler warnings about unused arguments when NDEBUG is defined.
    (void)old_acquires_allowed;
    // If the check below fails, Release() was called more times than acquire.
    assert(old_acquires_allowed < max_acquires_);
  }

 private:
#if !defined(NDEBUG)
  // Catches an excessive number of Release() calls.
  const int max_acquires_;
#endif  // !defined(NDEBUG)

  // The number of available resources.
  //
  // This is a counter and is not tied to the invariants of any other class, so
  // it can be operated on safely using std::memory_order_relaxed.
  std::atomic<int> acquires_allowed_;
};
```

**描述:**  `Limiter` 类用于限制资源的并发使用，例如文件描述符或 mmap 的数量。 它使用原子操作来确保线程安全。

*   构造函数:  初始化 `acquires_allowed_` 为 `max_acquires`。
*   `Acquire()`:  尝试获取一个资源。 如果还有可用的资源，则返回 `true`，否则返回 `false`。 它使用 `fetch_sub` 原子操作来减少 `acquires_allowed_` 的值。
*   `Release()`:  释放一个资源，增加 `acquires_allowed_` 的值。 它使用 `fetch_add` 原子操作。

**用途:**  防止程序耗尽资源，例如打开过多的文件或创建过多的 mmap，导致系统崩溃或性能下降。

**示例:**

```c++
Limiter mmap_limiter(100); // 允许最多 100 个 mmap
if (mmap_limiter.Acquire()) {
  // 创建 mmap
  char* mmap_base = (char*)MapViewOfFile(...);
  if (mmap_base) {
    // 使用 mmap
    // ...
    UnmapViewOfFile(mmap_base);
    mmap_limiter.Release(); // 释放 mmap
  } else {
    mmap_limiter.Release(); // 释放失败，也要释放
  }
} else {
  // 无法创建 mmap，因为已经达到限制
  // ...
}
```

**6. 文件操作类 (File Operation Classes):**

*   `WindowsSequentialFile`: 用于顺序读取文件。
*   `WindowsRandomAccessFile`: 用于随机读取文件。
*   `WindowsMmapReadableFile`:  用于使用内存映射读取文件。
*   `WindowsWritableFile`: 用于写入文件。

这些类都实现了 LevelDB 的相应接口 (例如 `SequentialFile`, `RandomAccessFile`, `WritableFile`)，并使用 Windows API 来执行实际的文件操作。

**示例 (WindowsWritableFile):**

```c++
class WindowsWritableFile : public WritableFile {
 public:
  WindowsWritableFile(std::string filename, ScopedHandle handle)
      : pos_(0), handle_(std::move(handle)), filename_(std::move(filename)) {}

  ~WindowsWritableFile() override = default;

  Status Append(const Slice& data) override {
    // ... (写入数据到缓冲区或直接写入文件) ...
  }

  Status Close() override {
    // ... (刷新缓冲区并关闭文件) ...
  }

  Status Flush() override {
    // ... (刷新缓冲区) ...
  }

  Status Sync() override {
    // ... (刷新缓冲区并同步文件到磁盘) ...
  }

 private:
  Status FlushBuffer() {
    // ... (将缓冲区中的数据写入文件) ...
  }

  Status WriteUnbuffered(const char* data, size_t size) {
    // ... (直接写入数据到文件) ...
  }

  // buf_[0, pos_-1] contains data to be written to handle_.
  char buf_[kWritableFileBufferSize];
  size_t pos_;

  ScopedHandle handle_;
  const std::string filename_;
};
```

**描述:**  `WindowsWritableFile` 类实现了 `WritableFile` 接口，用于写入数据到文件。 它使用一个缓冲区 `buf_` 来提高写入性能。

*   `Append()`:  将数据追加到文件。 如果缓冲区已满，则先刷新缓冲区，然后再写入数据。
*   `Close()`:  刷新缓冲区，并关闭文件。
*   `Flush()`:  刷新缓冲区，将数据写入磁盘，但不同步元数据。
*   `Sync()`:  刷新缓冲区，并将数据和元数据同步到磁盘。
*   `handle_`:  一个 `ScopedHandle` 对象，用于管理文件句柄。
*   `filename_`:  文件名。

**用途:**  允许 LevelDB 写入数据到文件，并提供缓冲和同步等功能。

**7. 文件锁类 (File Lock Classes):**

*   `WindowsFileLock`:  用于锁定文件，防止其他进程同时访问该文件。

**示例:**

```c++
class WindowsFileLock : public FileLock {
 public:
  WindowsFileLock(ScopedHandle handle, std::string filename)
      : handle_(std::move(handle)), filename_(std::move(filename)) {}

  const ScopedHandle& handle() const { return handle_; }
  const std::string& filename() const { return filename_; }

 private:
  const ScopedHandle handle_;
  const std::string filename_;
};
```

**描述:**  `WindowsFileLock` 类实现了 `FileLock` 接口，用于锁定文件。  它使用 Windows API 的 `LockFile` 和 `UnlockFile` 函数来实现文件锁定。

*   `handle_`:  一个 `ScopedHandle` 对象，用于管理文件句柄。
*   `filename_`:  文件名。

**用途:**  确保数据的一致性，防止多个进程同时修改同一个文件。

**8. WindowsEnv 类 (WindowsEnv Class):**

```c++
class WindowsEnv : public Env {
 public:
  WindowsEnv();
  ~WindowsEnv() override {
    static const char msg[] =
        "WindowsEnv singleton destroyed. Unsupported behavior!\n";
    std::fwrite(msg, 1, sizeof(msg), stderr);
    std::abort();
  }

  Status NewSequentialFile(const std::string& filename,
                           SequentialFile** result) override {
    // ... (创建 WindowsSequentialFile 对象) ...
  }

  Status NewRandomAccessFile(const std::string& filename,
                             RandomAccessFile** result) override {
    // ... (创建 WindowsRandomAccessFile 或 WindowsMmapReadableFile 对象) ...
  }

  Status NewWritableFile(const std::string& filename,
                         WritableFile** result) override {
    // ... (创建 WindowsWritableFile 对象) ...
  }

  Status NewAppendableFile(const std::string& filename,
                           WritableFile** result) override {
    // ... (创建 WindowsWritableFile 对象，以追加模式打开) ...
  }

  bool FileExists(const std::string& filename) override {
    // ... (检查文件是否存在) ...
  }

  Status GetChildren(const std::string& directory_path,
                     std::vector<std::string>* result) override {
    // ... (获取目录下的所有文件和目录) ...
  }

  Status RemoveFile(const std::string& filename) override {
    // ... (删除文件) ...
  }

  Status CreateDir(const std::string& dirname) override {
    // ... (创建目录) ...
  }

  Status RemoveDir(const std::string& dirname) override {
    // ... (删除目录) ...
  }

  Status GetFileSize(const std::string& filename, uint64_t* size) override {
    // ... (获取文件大小) ...
  }

  Status RenameFile(const std::string& from, const std::string& to) override {
    // ... (重命名文件) ...
  }

  Status LockFile(const std::string& filename, FileLock** lock) override {
    // ... (锁定文件) ...
  }

  Status UnlockFile(FileLock* lock) override {
    // ... (解锁文件) ...
  }

  void Schedule(void (*background_work_function)(void* background_work_arg),
                void* background_work_arg) override;

  void StartThread(void (*thread_main)(void* thread_main_arg),
                   void* thread_main_arg) override {
    std::thread new_thread(thread_main, thread_main_arg);
    new_thread.detach();
  }

  Status GetTestDirectory(std::string* result) override {
    // ... (获取测试目录) ...
  }

  Status NewLogger(const std::string& filename, Logger** result) override {
    // ... (创建日志对象) ...
  }

  uint64_t NowMicros() override {
    // ... (获取当前时间，以微秒为单位) ...
  }

  void SleepForMicroseconds(int micros) override {
    std::this_thread::sleep_for(std::chrono::microseconds(micros));
  }

 private:
  void BackgroundThreadMain();

  static void BackgroundThreadEntryPoint(WindowsEnv* env) {
    env->BackgroundThreadMain();
  }

  // Stores the work item data in a Schedule() call.
  //
  // Instances are constructed on the thread calling Schedule() and used on the
  // background thread.
  //
  // This structure is thread-safe because it is immutable.
  struct BackgroundWorkItem {
    explicit BackgroundWorkItem(void (*function)(void* arg), void* arg)
        : function(function), arg(arg) {}

    void (*const function)(void*);
    void* const arg;
  };

  port::Mutex background_work_mutex_;
  port::CondVar background_work_cv_ GUARDED_BY(background_work_mutex_);
  bool started_background_thread_ GUARDED_BY(background_work_mutex_);

  std::queue<BackgroundWorkItem> background_work_queue_
      GUARDED_BY(background_work_mutex_);

  Limiter mmap_limiter_;  // Thread-safe.
};
```

**描述:**  `WindowsEnv` 类是 LevelDB 在 Windows 平台上的环境实现。 它实现了 `Env` 接口，提供了文件系统操作、线程管理、时间获取等功能。

*   `NewSequentialFile`, `NewRandomAccessFile`, `NewWritableFile`, `NewAppendableFile`:  创建相应的文件对象。
*   `FileExists`, `GetChildren`, `RemoveFile`, `CreateDir`, `RemoveDir`, `GetFileSize`, `RenameFile`:  执行文件系统操作。
*   `LockFile`, `UnlockFile`:  锁定和解锁文件。
*   `Schedule`, `StartThread`:  管理线程。
*   `GetTestDirectory`, `NewLogger`:  提供测试和日志功能。
*   `NowMicros`, `SleepForMicroseconds`:  获取当前时间和睡眠。
*   `mmap_limiter_`:  用于限制 mmap 的数量。
*   `background_work_mutex_`, `background_work_cv_`, `background_work_queue_`:  用于管理后台工作线程。

**用途:**  `WindowsEnv` 类是 LevelDB 与 Windows 操作系统之间的桥梁，它提供了 LevelDB 运行所需的所有环境功能。

**9. SingletonEnv 模板类 (SingletonEnv Template Class):**

```c++
template <typename EnvType>
class SingletonEnv {
 public:
  SingletonEnv() {
#if !defined(NDEBUG)
    env_initialized_.store(true, std::memory_order_relaxed);
#endif  // !defined(NDEBUG)
    static_assert(sizeof(env_storage_) >= sizeof(EnvType),
                  "env_storage_ will not fit the Env");
    static_assert(std::is_standard_layout_v<SingletonEnv<EnvType>>);
    static_assert(
        offsetof(SingletonEnv<EnvType>, env_storage_) % alignof(EnvType) == 0,
        "env_storage_ does not meet the Env's alignment needs");
    static_assert(alignof(SingletonEnv<EnvType>) % alignof(EnvType) == 0,
                  "env_storage_ does not meet the Env's alignment needs");
    new (env_storage_) EnvType();
  }
  ~SingletonEnv() = default;

  SingletonEnv(const SingletonEnv&) = delete;
  SingletonEnv& operator=(const SingletonEnv&) = delete;

  Env* env() { return reinterpret_cast<Env*>(&env_storage_); }

  static void AssertEnvNotInitialized() {
#if !defined(NDEBUG)
    assert(!env_initialized_.load(std::memory_order_relaxed));
#endif  // !defined(NDEBUG)
  }

 private:
  alignas(EnvType) char env_storage_[sizeof(EnvType)];
#if !defined(NDEBUG)
  static std::atomic<bool> env_initialized_;
#endif  // !defined(NDEBUG)
};

using WindowsDefaultEnv = SingletonEnv<WindowsEnv>;
```

**描述:** `SingletonEnv` 类是一个模板类，用于创建单例 `Env` 对象。 单例模式确保在应用程序中只有一个 `Env` 实例。 这样可以避免资源冲突和配置问题。

*   构造函数:  在内部存储 `env_storage_` 中创建一个 `EnvType` 对象。
*   `env()`:  返回指向 `Env` 对象的指针。
*   `AssertEnvNotInitialized()`:  用于断言 `Env` 对象尚未初始化。

**用途:**  确保 LevelDB 使用唯一的 `Env` 对象。

**10. Env::Default() 函数 (Env::Default() Function):**

```c++
Env* Env::Default() {
  static WindowsDefaultEnv env_container;
  return env_container.env();
}
```

**描述:**  `Env::Default()` 函数返回 LevelDB 的默认 `Env` 对象。  它使用 `WindowsDefaultEnv` 单例类来创建和管理 `Env` 对象。

**用途:**  允许应用程序获取 LevelDB 的默认环境，以便执行文件系统操作、线程管理等。

**总结:**

这段代码是 LevelDB 在 Windows 平台上的环境实现。它提供了文件系统操作、线程管理、时间获取等功能。 它使用 Windows API 来执行实际的操作，并使用 RAII 和单例模式来管理资源和确保数据的一致性。 这些代码使得 LevelDB 能够在 Windows 平台上运行，并提供与平台无关的接口。
