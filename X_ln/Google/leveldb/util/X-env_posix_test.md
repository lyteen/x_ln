Lan: `cc` From`Google/leveldb\util\env_posix_test.cc`

好的，我将重新开始，并提供更细致的代码分解和中文解释。我将逐步生成代码片段，并附带描述。

**目标：改进 LevelDB 的 EnvPosixTest 测试套件，重点关注 `close-on-exec` 标志的正确设置。**

在多线程或多进程环境中，确保文件描述符不被意外地传递给子进程至关重要。`close-on-exec` 标志可以防止这种情况。

**1.  基础头文件和宏定义:**

```c++
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <sys/resource.h> // 用于获取文件描述符限制
#include <sys/wait.h>      // 用于进程等待
#include <unistd.h>       // POSIX 系统调用，如 fork, exec, dup2, close

#include <cstdio>          // 标准 C 输入/输出
#include <cstdlib>          // 标准 C 库函数，如 atoi, exit
#include <cstring>          // 字符串操作
#include <string>           // C++ 字符串类
#include <unordered_set>   // 无序集合
#include <vector>           // 动态数组

#include "gtest/gtest.h"        // Google Test 框架
#include "leveldb/env.h"         // LevelDB 环境抽象
#include "port/port.h"           // 平台相关定义
#include "util/env_posix_test_helper.h" // POSIX 环境测试辅助函数
#include "util/testutil.h"       // 测试工具函数

#if HAVE_O_CLOEXEC // 检查系统是否支持 O_CLOEXEC 标志

namespace {
```

**描述:**

*   引入必要的头文件，这些头文件提供了进行文件操作、进程管理和测试所需的函数和类。
*   `#if HAVE_O_CLOEXEC`  是一个预处理器指令，用于检查系统是否支持 `O_CLOEXEC` 标志。  `O_CLOEXEC` 是一个在 `open()` 系统调用中使用的标志，用于设置新打开的文件描述符在执行 `exec()` 系统调用时自动关闭。 如果系统支持此标志，则会编译后面的代码。
*   `namespace {`  定义一个匿名命名空间，其中的变量和函数只在当前文件中可见，防止命名冲突。

**2.  辅助进程的退出码定义:**

```c++
// Exit codes for the helper process spawned by TestCloseOnExec* tests.
// Useful for debugging test failures.
constexpr int kTextCloseOnExecHelperExecFailedCode = 61;
constexpr int kTextCloseOnExecHelperDup2FailedCode = 62;
constexpr int kTextCloseOnExecHelperFoundOpenFdCode = 63;
```

**描述:**

*   定义了几个常量，用于表示辅助进程的不同退出状态码。
*   如果辅助进程执行 `exec()` 失败，将返回 `kTextCloseOnExecHelperExecFailedCode`。
*   如果 `dup2()` 调用失败，返回 `kTextCloseOnExecHelperDup2FailedCode`。
*   如果辅助进程发现不应该打开的文件描述符仍然是打开状态，返回 `kTextCloseOnExecHelperFoundOpenFdCode`。
*   这些退出码有助于在测试失败时进行调试，快速定位问题原因。

**3.  全局 `argv[0]` 存储:**

```c++
// Global set by main() and read in TestCloseOnExec.
//
// The argv[0] value is stored in a std::vector instead of a std::string because
// std::string does not return a mutable pointer to its buffer until C++17.
//
// The vector stores the string pointed to by argv[0], plus the trailing null.
std::vector<char>* GetArgvZero() {
  static std::vector<char> program_name;
  return &program_name;
}
```

**描述:**

*   `GetArgvZero()` 函数用于获取并存储程序的可执行文件路径 (即 `argv[0]`)。
*   它使用 `std::vector<char>` 而不是 `std::string`，因为在 C++17 之前，`std::string` 不提供直接获取可修改的内部缓冲区的接口。 `execv()` 函数需要一个可修改的 `char*` 数组作为参数。
*   `static std::vector<char> program_name;`  确保只初始化一次，并在程序的整个生命周期内保持存在。

**4.  命令行开关定义:**

```c++
// Command-line switch used to run this test as the CloseOnExecSwitch helper.
static const char kTestCloseOnExecSwitch[] = "--test-close-on-exec-helper";
```

**描述:**

*   `kTestCloseOnExecSwitch`  定义一个常量字符串，作为命令行参数，用于指示程序以辅助进程的身份运行。

**5.  辅助进程的主函数 `TestCloseOnExecHelperMain`:**

```c++
// Executed in a separate process by TestCloseOnExec* tests.
//
// main() delegates to this function when the test executable is launched with
// a special command-line switch. TestCloseOnExec* tests fork()+exec() the test
// executable and pass the special command-line switch.
//

// main() delegates to this function when the test executable is launched with
// a special command-line switch. TestCloseOnExec* tests fork()+exec() the test
// executable and pass the special command-line switch.
//
// When main() delegates to this function, the process probes whether a given
// file descriptor is open, and communicates the result via its exit code.
int TestCloseOnExecHelperMain(char* pid_arg) {
  int fd = std::atoi(pid_arg);
  // When given the same file descriptor twice, dup2() returns -1 if the
  // file descriptor is closed, or the given file descriptor if it is open.
  if (::dup2(fd, fd) == fd) {
    std::fprintf(stderr, "Unexpected open fd %d\n", fd);
    return kTextCloseOnExecHelperFoundOpenFdCode;
  }
  // Double-check that dup2() is saying the file descriptor is closed.
  if (errno != EBADF) {
    std::fprintf(stderr, "Unexpected errno after calling dup2 on fd %d: %s\n",
                 fd, std::strerror(errno));
    return kTextCloseOnExecHelperDup2FailedCode;
  }
  return 0;
}
```

**描述:**

*   `TestCloseOnExecHelperMain`  函数是辅助进程的入口点。
*   它接收一个字符串参数 `pid_arg`，该参数表示要检查的文件描述符的整数值。
*   使用 `dup2(fd, fd)` 来检查文件描述符 `fd` 是否打开。 `dup2` 函数尝试复制文件描述符 `fd` 到自身。 如果 `fd` 是一个有效的文件描述符，`dup2` 会成功返回 `fd`。 如果 `fd` 无效（例如，已经被关闭），`dup2` 会返回 -1，并且 `errno` 会被设置为 `EBADF` (Bad file descriptor)。
*   如果 `dup2(fd, fd)` 返回 `fd`，说明文件描述符意外地打开了，函数会输出错误信息并返回 `kTextCloseOnExecHelperFoundOpenFdCode`。
*   如果 `dup2(fd, fd)` 返回 -1，并且 `errno` 不是 `EBADF`，说明发生了其他错误，函数会输出错误信息并返回 `kTextCloseOnExecHelperDup2FailedCode`。
*   如果 `dup2(fd, fd)` 返回 -1，并且 `errno` 是 `EBADF`，说明文件描述符已正确关闭，函数返回 0。

**6. 获取最大文件描述符的函数 `GetMaxFileDescriptor`:**

```c++
// File descriptors are small non-negative integers.
//
// Returns void so the implementation can use ASSERT_EQ.
void GetMaxFileDescriptor(int* result_fd) {
  // Get the maximum file descriptor number.
  ::rlimit fd_rlimit;
  ASSERT_EQ(0, ::getrlimit(RLIMIT_NOFILE, &fd_rlimit));
  *result_fd = fd_rlimit.rlim_cur;
}
```

**描述:**

*   `GetMaxFileDescriptor`  函数用于获取当前进程允许的最大文件描述符数量。
*   它使用 `getrlimit(RLIMIT_NOFILE, &fd_rlimit)`  来获取文件描述符限制。  `RLIMIT_NOFILE` 指定了每个进程可以打开的最大文件描述符数量。
*   `fd_rlimit.rlim_cur`  表示当前的软限制。
*   `ASSERT_EQ(0, ::getrlimit(RLIMIT_NOFILE, &fd_rlimit));`  断言 `getrlimit` 调用成功。 如果调用失败，测试会立即停止。
*   将获取的最大文件描述符数量存储在 `result_fd` 指针指向的变量中。

**7. 获取所有打开文件描述符的函数 `GetOpenFileDescriptors`:**

```c++
// Iterates through all possible FDs and returns the currently open ones.
//
// Returns void so the implementation can use ASSERT_EQ.
void GetOpenFileDescriptors(std::unordered_set<int>* open_fds) {
  int max_fd = 0;
  GetMaxFileDescriptor(&max_fd);

  for (int fd = 0; fd < max_fd; ++fd) {
    if (::dup2(fd, fd) != fd) {
      // When given the same file descriptor twice, dup2() returns -1 if the
      // file descriptor is closed, or the given file descriptor if it is open.
      //
      // Double-check that dup2() is saying the fd is closed.
      ASSERT_EQ(EBADF, errno)
          << "dup2() should set errno to EBADF on closed file descriptors";
      continue;
    }
    open_fds->insert(fd);
  }
}
```

**描述:**

*   `GetOpenFileDescriptors` 函数用于检测当前进程中所有打开的文件描述符。
*   它首先调用 `GetMaxFileDescriptor` 获取最大文件描述符数量。
*   然后，它遍历从 0 到 `max_fd` 的所有可能的文件描述符。
*   对于每个文件描述符，它使用 `dup2(fd, fd)`  来检查文件描述符是否有效。
*   如果 `dup2` 返回 -1，并且 `errno` 是 `EBADF`，说明文件描述符已关闭，循环继续。
*   如果 `dup2` 返回 `fd`，说明文件描述符是打开的，将其插入到 `open_fds` 集合中。
*   `ASSERT_EQ(EBADF, errno) << "dup2() should set errno to EBADF on closed file descriptors";`  是一个断言，用于确保当文件描述符关闭时，`errno` 被正确地设置为 `EBADF`。

**8. 获取新打开的文件描述符的函数 `GetNewlyOpenedFileDescriptor`:**

```c++
// Finds an FD open since a previous call to GetOpenFileDescriptors().
//
// |baseline_open_fds| is the result of a previous GetOpenFileDescriptors()
// call. Assumes that exactly one FD was opened since that call.
//
// Returns void so the implementation can use ASSERT_EQ.
void GetNewlyOpenedFileDescriptor(
    const std::unordered_set<int>& baseline_open_fds, int* result_fd) {
  std::unordered_set<int> open_fds;
  GetOpenFileDescriptors(&open_fds);
  for (int fd : baseline_open_fds) {
    ASSERT_EQ(1, open_fds.count(fd))
        << "Previously opened file descriptor was closed during test setup";
    open_fds.erase(fd);
  }
  ASSERT_EQ(1, open_fds.size())
      << "Expected exactly one newly opened file descriptor during test setup";
  *result_fd = *open_fds.begin();
}
```

**描述:**

*   `GetNewlyOpenedFileDescriptor`  函数用于查找在调用 `GetOpenFileDescriptors` 函数之后新打开的文件描述符。
*   它接收一个 `baseline_open_fds` 集合，该集合包含之前打开的文件描述符。
*   它再次调用 `GetOpenFileDescriptors` 获取当前所有打开的文件描述符。
*   它从当前打开的文件描述符集合中移除 `baseline_open_fds` 中包含的文件描述符。
*   `ASSERT_EQ(1, open_fds.size()) << "Expected exactly one newly opened file descriptor during test setup";` 断言只有一个新的文件描述符被打开。
*   将新的文件描述符存储在 `result_fd` 指针指向的变量中。

**9. 检查 `close-on-exec` 是否生效的函数 `CheckCloseOnExecDoesNotLeakFDs`:**

```c++
// Check that a fork()+exec()-ed child process does not have an extra open FD.
void CheckCloseOnExecDoesNotLeakFDs(
    const std::unordered_set<int>& baseline_open_fds) {
  // Prepare the argument list for the child process.
  // execv() wants mutable buffers.
  char switch_buffer[sizeof(kTestCloseOnExecSwitch)];
  std::memcpy(switch_buffer, kTestCloseOnExecSwitch,
              sizeof(kTestCloseOnExecSwitch));

  int probed_fd;
  GetNewlyOpenedFileDescriptor(baseline_open_fds, &probed_fd);
  std::string fd_string = std::to_string(probed_fd);
  std::vector<char> fd_buffer(fd_string.begin(), fd_string.end());
  fd_buffer.emplace_back('\0');

  // The helper process is launched with the command below.
  //      env_posix_tests --test-close-on-exec-helper 3
  char* child_argv[] = {GetArgvZero()->data(), switch_buffer, fd_buffer.data(),
                        nullptr};

  constexpr int kForkInChildProcessReturnValue = 0;
  int child_pid = fork();
  if (child_pid == kForkInChildProcessReturnValue) {
    ::execv(child_argv[0], child_argv);
    std::fprintf(stderr, "Error spawning child process: %s\n", strerror(errno));
    std::exit(kTextCloseOnExecHelperExecFailedCode);
  }

  int child_status = 0;
  ASSERT_EQ(child_pid, ::waitpid(child_pid, &child_status, 0));
  ASSERT_TRUE(WIFEXITED(child_status))
      << "The helper process did not exit with an exit code";
  ASSERT_EQ(0, WEXITSTATUS(child_status))
      << "The helper process encountered an error";
}
```

**描述:**

*   `CheckCloseOnExecDoesNotLeakFDs`  函数用于检查通过 `fork()` 和 `exec()` 创建的子进程是否意外地继承了不应该继承的文件描述符。
*   它接收一个 `baseline_open_fds` 集合，该集合包含在打开新文件描述符之前打开的所有文件描述符。
*   它首先调用 `GetNewlyOpenedFileDescriptor`  来获取新打开的文件描述符 (即，理论上应该被设置了 `close-on-exec` 标志的文件描述符)。
*   它创建一个命令行参数列表 `child_argv`，用于传递给 `execv` 函数。  这个参数列表包括程序的可执行文件路径、`kTestCloseOnExecSwitch` 开关和新打开的文件描述符的字符串表示。
*   然后，它使用 `fork()` 创建一个子进程。
*   在子进程中，它使用 `execv()`  执行自身，但传递了 `kTestCloseOnExecSwitch` 开关，这将使程序以辅助进程的身份运行，并执行 `TestCloseOnExecHelperMain` 函数。
*   在父进程中，它使用 `waitpid()` 等待子进程完成。
*   `ASSERT_TRUE(WIFEXITED(child_status))`  断言子进程正常退出。
*   `ASSERT_EQ(0, WEXITSTATUS(child_status))`  断言子进程的退出码为 0，这意味着子进程没有检测到任何错误 (即，新打开的文件描述符在子进程中没有打开)。

**10. LevelDB 命名空间和测试类定义:**

```c++
}  // namespace

#endif  // HAVE_O_CLOEXEC

namespace leveldb {

static const int kReadOnlyFileLimit = 4;
static const int kMMapLimit = 4;

class EnvPosixTest : public testing::Test {
 public:
  static void SetFileLimits(int read_only_file_limit, int mmap_limit) {
    EnvPosixTestHelper::SetReadOnlyFDLimit(read_only_file_limit);
    EnvPosixTestHelper::SetReadOnlyMMapLimit(mmap_limit);
  }

  EnvPosixTest() : env_(Env::Default()) {}

  Env* env_;
};
```

**描述:**

*   `namespace leveldb {`  定义 LevelDB 的命名空间，包含所有 LevelDB 相关的类和函数。
*   `kReadOnlyFileLimit` 和 `kMMapLimit` 定义了只读文件和内存映射文件的限制数量，这些限制用于测试文件打开的行为。
*   `EnvPosixTest`  类是一个 Google Test 测试类，用于测试 POSIX 环境下的 LevelDB 功能。
*   `SetFileLimits`  是一个静态方法，用于设置只读文件和内存映射文件的限制。  它调用 `EnvPosixTestHelper` 中的函数来设置这些限制。
*   构造函数 `EnvPosixTest()` 初始化 `env_` 成员变量，该变量是 LevelDB 环境的抽象。

**11. 测试用例：`TestOpenOnRead`**

```c++
TEST_F(EnvPosixTest, TestOpenOnRead) {
  // Write some test data to a single file that will be opened |n| times.
  std::string test_dir;
  ASSERT_LEVELDB_OK(env_->GetTestDirectory(&test_dir));
  std::string test_file = test_dir + "/open_on_read.txt";

  FILE* f = std::fopen(test_file.c_str(), "we");
  ASSERT_TRUE(f != nullptr);
  const char kFileData[] = "abcdefghijklmnopqrstuvwxyz";
  fputs(kFileData, f);
  std::fclose(f);

  // Open test file some number above the sum of the two limits to force
  // open-on-read behavior of POSIX Env leveldb::RandomAccessFile.
  const int kNumFiles = kReadOnlyFileLimit + kMMapLimit + 5;
  leveldb::RandomAccessFile* files[kNumFiles] = {0};
  for (int i = 0; i < kNumFiles; i++) {
    ASSERT_LEVELDB_OK(env_->NewRandomAccessFile(test_file, &files[i]));
  }
  char scratch;
  Slice read_result;
  for (int i = 0; i < kNumFiles; i++) {
    ASSERT_LEVELDB_OK(files[i]->Read(i, 1, &read_result, &scratch));
    ASSERT_EQ(kFileData[i], read_result[0]);
  }
  for (int i = 0; i < kNumFiles; i++) {
    delete files[i];
  }
  ASSERT_LEVELDB_OK(env_->RemoveFile(test_file));
}
```

**描述:**

*   `TEST_F(EnvPosixTest, TestOpenOnRead)`  是一个 Google Test 测试用例，用于测试 `Env::NewRandomAccessFile`  在打开大量文件时的行为。
*   该测试首先创建一个测试文件 `open_on_read.txt`，并向其中写入一些数据。
*   然后，它尝试打开该文件 `kNumFiles`  次，其中 `kNumFiles`  大于 `kReadOnlyFileLimit + kMMapLimit`。 这会强制 `Env::NewRandomAccessFile` 使用 "open-on-read" 模式，即在达到文件描述符限制时，关闭一些之前打开的文件描述符，以便打开新的文件。
*   对于每个打开的文件，它读取一个字节的数据，并验证读取的数据是否正确。
*   最后，它关闭所有打开的文件，并删除测试文件。

**12. 测试用例：`TestCloseOnExecSequentialFile` 等等 (针对不同文件类型):**

```c++
#if HAVE_O_CLOEXEC

TEST_F(EnvPosixTest, TestCloseOnExecSequentialFile) {
  std::unordered_set<int> open_fds;
  GetOpenFileDescriptors(&open_fds);

  std::string test_dir;
  ASSERT_LEVELDB_OK(env_->GetTestDirectory(&test_dir));
  std::string file_path = test_dir + "/close_on_exec_sequential.txt";
  ASSERT_LEVELDB_OK(WriteStringToFile(env_, "0123456789", file_path));

  leveldb::SequentialFile* file = nullptr;
  ASSERT_LEVELDB_OK(env_->NewSequentialFile(file_path, &file));
  CheckCloseOnExecDoesNotLeakFDs(open_fds);
  delete file;

  ASSERT_LEVELDB_OK(env_->RemoveFile(file_path));
}

// Similar tests for RandomAccessFile, WritableFile, AppendableFile, LockFile, Logger

#endif  // HAVE_O_CLOEXEC
```

**描述:**

*   这些测试用例（`TestCloseOnExecSequentialFile`, `TestCloseOnExecRandomAccessFile`, `TestCloseOnExecWritableFile`, `TestCloseOnExecAppendableFile`, `TestCloseOnExecLockFile`, `TestCloseOnExecLogger`）  都遵循相同的模式，用于测试不同类型的 LevelDB 文件对象是否正确设置了 `close-on-exec` 标志。
*   它们首先获取当前所有打开的文件描述符。
*   然后，创建一个测试文件，并使用相应的 `Env::New...File`  函数打开该文件。
*   调用 `CheckCloseOnExecDoesNotLeakFDs` 函数，该函数会 `fork()` 一个子进程，并在子进程中检查新打开的文件描述符是否意外地被继承。
*   最后，关闭文件并删除测试文件。
*   `#if HAVE_O_CLOEXEC`  确保这些测试只在支持 `O_CLOEXEC` 标志的系统上运行。

**13.  主函数 `main`:**

```c++
}  // namespace leveldb

int main(int argc, char** argv) {
#if HAVE_O_CLOEXEC
  // Check if we're invoked as a helper program, or as the test suite.
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], kTestCloseOnExecSwitch)) {
      return TestCloseOnExecHelperMain(argv[i + 1]);
    }
  }

  // Save argv[0] early, because googletest may modify argv.
  GetArgvZero()->assign(argv[0], argv[0] + std::strlen(argv[0]) + 1);
#endif  // HAVE_O_CLOEXEC

  // All tests currently run with the same read-only file limits.
  leveldb::EnvPosixTest::SetFileLimits(leveldb::kReadOnlyFileLimit,
                                       leveldb::kMMapLimit);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

**描述:**

*   `main` 函数是程序的入口点。
*   它首先检查程序是否以辅助进程的身份运行。 如果是，它调用 `TestCloseOnExecHelperMain` 函数，并将文件描述符作为参数传递。
*   如果程序不是以辅助进程的身份运行，它将保存 `argv[0]` 的值，因为 Google Test 可能会修改 `argv`。
*   然后，它设置只读文件和内存映射文件的限制。
*   最后，它初始化 Google Test 框架，并运行所有测试用例。
*   `#if HAVE_O_CLOEXEC` 确保只有在支持 `O_CLOEXEC` 标志的系统上才会执行辅助进程相关的代码。

**总结:**

这个代码库提供了一个全面的测试套件，用于验证 LevelDB 在 POSIX 环境下文件操作的正确性，特别是 `close-on-exec` 标志的设置。  它使用辅助进程来模拟 `fork()` 和 `exec()` 行为，并检查文件描述符是否被意外地继承。  该代码结构清晰，注释详尽，易于理解和维护。

这段代码充分考虑了多进程环境下的文件描述符管理，并通过测试确保了程序的健壮性和安全性。
