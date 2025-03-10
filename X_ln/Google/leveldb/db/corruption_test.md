Lan: `cc` From`Google/leveldb\db\corruption_test.cc`

Okay, restarting. I will provide a detailed explanation of the C++ LevelDB corruption test code, breaking it down into smaller, understandable segments with Chinese descriptions.  I'll also include explanations of how each section is used and a simple usage demo where relevant.

```c++
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <sys/types.h>

#include "gtest/gtest.h"
#include "db/db_impl.h"
#include "db/filename.h"
#include "db/log_format.h"
#include "db/version_set.h"
#include "leveldb/cache.h"
#include "leveldb/db.h"
#include "leveldb/table.h"
#include "leveldb/write_batch.h"
#include "util/logging.h"
#include "util/testutil.h"

namespace leveldb {

static const int kValueSize = 1000;

// -------------------------------------------------------------------
// CorruptionTest Class Definition  损坏测试类定义
// -------------------------------------------------------------------

class CorruptionTest : public testing::Test {
 public:
  CorruptionTest()
      : db_(nullptr),
        dbname_("/memenv/corruption_test"),
        tiny_cache_(NewLRUCache(100)) {
    options_.env = &env_;
    options_.block_cache = tiny_cache_;
    DestroyDB(dbname_, options_);

    options_.create_if_missing = true;
    Reopen();
    options_.create_if_missing = false;
  }

  ~CorruptionTest() {
    delete db_;
    delete tiny_cache_;
  }

  Status TryReopen() {
    delete db_;
    db_ = nullptr;
    return DB::Open(options_, dbname_, &db_);
  }

  void Reopen() { ASSERT_LEVELDB_OK(TryReopen()); }

  void RepairDB() {
    delete db_;
    db_ = nullptr;
    ASSERT_LEVELDB_OK(::leveldb::RepairDB(dbname_, options_));
  }

  void Build(int n) {
    std::string key_space, value_space;
    WriteBatch batch;
    for (int i = 0; i < n; i++) {
      // if ((i % 100) == 0) std::fprintf(stderr, "@ %d of %d\n", i, n);
      Slice key = Key(i, &key_space);
      batch.Clear();
      batch.Put(key, Value(i, &value_space));
      WriteOptions options;
      // Corrupt() doesn't work without this sync on windows; stat reports 0 for
      // the file size.
      if (i == n - 1) {
        options.sync = true;
      }
      ASSERT_LEVELDB_OK(db_->Write(options, &batch));
    }
  }

  void Check(int min_expected, int max_expected) {
    int next_expected = 0;
    int missed = 0;
    int bad_keys = 0;
    int bad_values = 0;
    int correct = 0;
    std::string value_space;
    Iterator* iter = db_->NewIterator(ReadOptions());
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      uint64_t key;
      Slice in(iter->key());
      if (in == "" || in == "~") {
        // Ignore boundary keys.
        continue;
      }
      if (!ConsumeDecimalNumber(&in, &key) || !in.empty() ||
          key < next_expected) {
        bad_keys++;
        continue;
      }
      missed += (key - next_expected);
      next_expected = key + 1;
      if (iter->value() != Value(key, &value_space)) {
        bad_values++;
      } else {
        correct++;
      }
    }
    delete iter;

    std::fprintf(
        stderr,
        "expected=%d..%d; got=%d; bad_keys=%d; bad_values=%d; missed=%d\n",
        min_expected, max_expected, correct, bad_keys, bad_values, missed);
    ASSERT_LE(min_expected, correct);
    ASSERT_GE(max_expected, correct);
  }

  void Corrupt(FileType filetype, int offset, int bytes_to_corrupt) {
    // Pick file to corrupt
    std::vector<std::string> filenames;
    ASSERT_LEVELDB_OK(env_.target()->GetChildren(dbname_, &filenames));
    uint64_t number;
    FileType type;
    std::string fname;
    int picked_number = -1;
    for (size_t i = 0; i < filenames.size(); i++) {
      if (ParseFileName(filenames[i], &number, &type) && type == filetype &&
          int(number) > picked_number) {  // Pick latest file
        fname = dbname_ + "/" + filenames[i];
        picked_number = number;
      }
    }
    ASSERT_TRUE(!fname.empty()) << filetype;

    uint64_t file_size;
    ASSERT_LEVELDB_OK(env_.target()->GetFileSize(fname, &file_size));

    if (offset < 0) {
      // Relative to end of file; make it absolute
      if (-offset > file_size) {
        offset = 0;
      } else {
        offset = file_size + offset;
      }
    }
    if (offset > file_size) {
      offset = file_size;
    }
    if (offset + bytes_to_corrupt > file_size) {
      bytes_to_corrupt = file_size - offset;
    }

    // Do it
    std::string contents;
    Status s = ReadFileToString(env_.target(), fname, &contents);
    ASSERT_TRUE(s.ok()) << s.ToString();
    for (int i = 0; i < bytes_to_corrupt; i++) {
      contents[i + offset] ^= 0x80;
    }
    s = WriteStringToFile(env_.target(), contents, fname);
    ASSERT_TRUE(s.ok()) << s.ToString();
  }

  int Property(const std::string& name) {
    std::string property;
    int result;
    if (db_->GetProperty(name, &property) &&
        sscanf(property.c_str(), "%d", &result) == 1) {
      return result;
    } else {
      return -1;
    }
  }

  // Return the ith key
  Slice Key(int i, std::string* storage) {
    char buf[100];
    std::snprintf(buf, sizeof(buf), "%016d", i);
    storage->assign(buf, strlen(buf));
    return Slice(*storage);
  }

  // Return the value to associate with the specified key
  Slice Value(int k, std::string* storage) {
    Random r(k);
    return test::RandomString(&r, kValueSize, storage);
  }

  test::ErrorEnv env_;
  Options options_;
  DB* db_;

 private:
  std::string dbname_;
  Cache* tiny_cache_;
};
```

**解释:**

*   **`CorruptionTest` 类:**  这个类继承自 `testing::Test`，用于组织和执行 LevelDB 的损坏测试。
    *   **构造函数 (`CorruptionTest()`):**
        *   初始化数据库指针 `db_` 为 `nullptr`。
        *   设置数据库名称 `dbname_` 为 `"/memenv/corruption_test"`。`memenv` 是一个内存中的文件系统，方便测试，避免对真实磁盘的读写。
        *   创建一个小的 LRU 缓存 `tiny_cache_`，大小为 100。
        *   设置 `options_.env` 为 `&env_`，使用自定义的 `ErrorEnv`，用于模拟错误。
        *   设置 `options_.block_cache` 为 `tiny_cache_`。
        *   销毁已存在的数据库（如果存在）。
        *   使用 `options_.create_if_missing = true` 创建一个新的数据库。
        *   调用 `Reopen()` 打开数据库。
        *   设置 `options_.create_if_missing = false`，防止意外创建数据库。
        *   **析构函数 (`~CorruptionTest()`):**
        *   释放 `db_` 和 `tiny_cache_` 占用的内存。
    *   **`TryReopen()`:** 尝试重新打开数据库。如果成功，返回 `Status::OK()`，否则返回错误状态。
    *   **`Reopen()`:** 重新打开数据库，如果失败则断言失败（使用 `ASSERT_LEVELDB_OK`）。  这确保了测试在数据库无法正常打开时立即停止。
    *   **`RepairDB()`:** 尝试修复数据库。如果成功，返回 `Status::OK()`，否则返回错误状态。
    *   **`Build(int n)`:** 构建数据库，插入 `n` 个键值对。
        *   生成 key 使用 `Key(i, &key_space)`
        *   生成 value 使用 `Value(i, &value_space)`
        *   使用 `WriteBatch` 批量写入，提高效率。
        *  如果 i == n - 1 同步写入，用于防止 windows 上的`Corrupt()` 函数无法工作.
    *   **`Check(int min_expected, int max_expected)`:** 检查数据库中的键值对是否正确。
        *   使用 `Iterator` 遍历数据库。
        *   检查 key 是否是期望的数字，value 是否与 key 对应。
        *   统计正确、错误和缺失的键值对数量。
        *   使用 `ASSERT_LE` 和 `ASSERT_GE` 断言正确的键值对数量在期望的范围内。
    *   **`Corrupt(FileType filetype, int offset, int bytes_to_corrupt)`:** 损坏指定类型的文件。
        *   获取指定类型的文件名列表。
        *   选择最新的文件进行损坏。
        *   读取文件内容。
        *   在指定偏移量处，将指定数量的字节进行异或操作，从而损坏数据。
        *   将损坏后的数据写回文件。
    *   **`Property(const std::string& name)`:** 获取数据库的属性值。
    *   **`Key(int i, std::string* storage)`:**  生成 key。
    *   **`Value(int k, std::string* storage)`:** 生成 value。
    *   **成员变量:**
        *   `env_`: `test::ErrorEnv` 类型的对象，用于模拟错误环境。
        *   `options_`: `Options` 类型的对象，用于配置数据库。
        *   `db_`: `DB*` 类型的指针，指向数据库对象。
        *   `dbname_`: `std::string` 类型的对象，存储数据库名称。
        *   `tiny_cache_`: `Cache*` 类型的指针，指向缓存对象。

**使用示例:**

```c++
CorruptionTest test; // 创建CorruptionTest对象，会自动创建和初始化数据库
test.Build(100);     // 插入100个键值对
test.Check(100, 100); // 检查数据是否正确
test.Corrupt(kLogFile, 19, 1); // 损坏日志文件
test.Reopen();       // 重新打开数据库，看是否能恢复
test.Check(36, 36);  // 检查数据，看有多少丢失了
```

**解释（中文）:**

*   **`CorruptionTest` 类:** 这个类是用来测试 LevelDB 在遇到数据损坏时行为的。它继承了 `testing::Test`，这是一个 Google Test 框架提供的基类，用于组织测试用例。
    *   **构造函数 (`CorruptionTest()`):**  构造函数会做以下事情：
        *   初始化数据库连接，指定数据库的名称（`dbname_`），并创建一个小的缓存（`tiny_cache_`）来提高性能。
        *   配置数据库的选项（`options_`），例如设置错误环境（`env_`）和缓存。
        *   删除之前可能存在的数据库，然后创建一个新的数据库。这是为了确保每次测试都在干净的环境中进行。
    *   **析构函数 (`~CorruptionTest()`):**  析构函数负责清理资源，例如关闭数据库连接和释放缓存。
    *   **`TryReopen()`:**  尝试重新打开数据库。这个函数会删除当前的数据库连接，然后尝试用相同的选项重新打开数据库。如果打开失败，它会返回一个错误状态。
    *   **`Reopen()`:**  重新打开数据库，但如果打开失败，会直接终止测试。
    *   **`RepairDB()`:**  尝试修复损坏的数据库。LevelDB 提供了一个工具来尝试修复损坏的数据库，这个函数就是调用这个工具。
    *   **`Build(int n)`:**  向数据库中写入 `n` 个键值对。这个函数会生成一些测试数据，然后将它们写入数据库。
    *   **`Check(int min_expected, int max_expected)`:**  检查数据库中的数据是否符合预期。这个函数会遍历数据库，检查每个键值对是否正确，并统计错误和缺失的数据。
    *   **`Corrupt(FileType filetype, int offset, int bytes_to_corrupt)`:**  故意损坏数据库文件。这个函数会找到指定类型的文件（例如日志文件或数据表文件），然后在指定的偏移量处修改一些字节，从而模拟数据损坏。
    *   **`Property(const std::string& name)`:**  获取数据库的属性值。LevelDB 允许你查询一些数据库的内部属性，例如数据表的大小、缓存的使用情况等等。
    *   **`Key(int i, std::string* storage)`:**  生成测试用的键。
    *   **`Value(int k, std::string* storage)`:**  生成测试用的值。

**2. Test Cases (测试用例):**

```c++
TEST_F(CorruptionTest, Recovery) {
  Build(100);
  Check(100, 100);
  Corrupt(kLogFile, 19, 1);  // WriteBatch tag for first record
  Corrupt(kLogFile, log::kBlockSize + 1000, 1);  // Somewhere in second block
  Reopen();

  // The 64 records in the first two log blocks are completely lost.
  Check(36, 36);
}

TEST_F(CorruptionTest, RecoverWriteError) {
  env_.writable_file_error_ = true;
  Status s = TryReopen();
  ASSERT_TRUE(!s.ok());
}

TEST_F(CorruptionTest, NewFileErrorDuringWrite) {
  // Do enough writing to force minor compaction
  env_.writable_file_error_ = true;
  const int num = 3 + (Options().write_buffer_size / kValueSize);
  std::string value_storage;
  Status s;
  for (int i = 0; s.ok() && i < num; i++) {
    WriteBatch batch;
    batch.Put("a", Value(100, &value_storage));
    s = db_->Write(WriteOptions(), &batch);
  }
  ASSERT_TRUE(!s.ok());
  ASSERT_GE(env_.num_writable_file_errors_, 1);
  env_.writable_file_error_ = false;
  Reopen();
}

TEST_F(CorruptionTest, TableFile) {
  Build(100);
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  dbi->TEST_CompactRange(0, nullptr, nullptr);
  dbi->TEST_CompactRange(1, nullptr, nullptr);

  Corrupt(kTableFile, 100, 1);
  Check(90, 99);
}

TEST_F(CorruptionTest, TableFileRepair) {
  options_.block_size = 2 * kValueSize;  // Limit scope of corruption
  options_.paranoid_checks = true;
  Reopen();
  Build(100);
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  dbi->TEST_CompactRange(0, nullptr, nullptr);
  dbi->TEST_CompactRange(1, nullptr, nullptr);

  Corrupt(kTableFile, 100, 1);
  RepairDB();
  Reopen();
  Check(95, 99);
}

TEST_F(CorruptionTest, TableFileIndexData) {
  Build(10000);  // Enough to build multiple Tables
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();

  Corrupt(kTableFile, -2000, 500);
  Reopen();
  Check(5000, 9999);
}

TEST_F(CorruptionTest, MissingDescriptor) {
  Build(1000);
  RepairDB();
  Reopen();
  Check(1000, 1000);
}

TEST_F(CorruptionTest, SequenceNumberRecovery) {
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "v1"));
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "v2"));
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "v3"));
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "v4"));
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "v5"));
  RepairDB();
  Reopen();
  std::string v;
  ASSERT_LEVELDB_OK(db_->Get(ReadOptions(), "foo", &v));
  ASSERT_EQ("v5", v);
  // Write something.  If sequence number was not recovered properly,
  // it will be hidden by an earlier write.
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "v6"));
  ASSERT_LEVELDB_OK(db_->Get(ReadOptions(), "foo", &v));
  ASSERT_EQ("v6", v);
  Reopen();
  ASSERT_LEVELDB_OK(db_->Get(ReadOptions(), "foo", &v));
  ASSERT_EQ("v6", v);
}

TEST_F(CorruptionTest, CorruptedDescriptor) {
  ASSERT_LEVELDB_OK(db_->Put(WriteOptions(), "foo", "hello"));
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  dbi->TEST_CompactRange(0, nullptr, nullptr);

  Corrupt(kDescriptorFile, 0, 1000);
  Status s = TryReopen();
  ASSERT_TRUE(!s.ok());

  RepairDB();
  Reopen();
  std::string v;
  ASSERT_LEVELDB_OK(db_->Get(ReadOptions(), "foo", &v));
  ASSERT_EQ("hello", v);
}

TEST_F(CorruptionTest, CompactionInputError) {
  Build(10);
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  const int last = config::kMaxMemCompactLevel;
  ASSERT_EQ(1, Property("leveldb.num-files-at-level" + NumberToString(last)));

  Corrupt(kTableFile, 100, 1);
  Check(5, 9);

  // Force compactions by writing lots of values
  Build(10000);
  Check(10000, 10000);
}

TEST_F(CorruptionTest, CompactionInputErrorParanoid) {
  options_.paranoid_checks = true;
  options_.write_buffer_size = 512 << 10;
  Reopen();
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);

  // Make multiple inputs so we need to compact.
  for (int i = 0; i < 2; i++) {
    Build(10);
    dbi->TEST_CompactMemTable();
    Corrupt(kTableFile, 100, 1);
    env_.SleepForMicroseconds(100000);
  }
  dbi->CompactRange(nullptr, nullptr);

  // Write must fail because of corrupted table
  std::string tmp1, tmp2;
  Status s = db_->Put(WriteOptions(), Key(5, &tmp1), Value(5, &tmp2));
  ASSERT_TRUE(!s.ok()) << "write did not fail in corrupted paranoid db";
}

TEST_F(CorruptionTest, UnrelatedKeys) {
  Build(10);
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  Corrupt(kTableFile, 100, 1);

  std::string tmp1, tmp2;
  ASSERT_LEVELDB_OK(
      db_->Put(WriteOptions(), Key(1000, &tmp1), Value(1000, &tmp2)));
  std::string v;
  ASSERT_LEVELDB_OK(db_->Get(ReadOptions(), Key(1000, &tmp1), &v));
  ASSERT_EQ(Value(1000, &tmp2).ToString(), v);
  dbi->TEST_CompactMemTable();
  ASSERT_LEVELDB_OK(db_->Get(ReadOptions(), Key(1000, &tmp1), &v));
  ASSERT_EQ(Value(1000, &tmp2).ToString(), v);
}

}  // namespace leveldb
```

**解释:**

*   **`TEST_F(CorruptionTest, TestName)`:** 这是 Google Test 框架的宏，用于定义一个测试用例。 `CorruptionTest` 是测试类的名称， `TestName` 是测试用例的名称。

Let's break down a few of these test cases in more detail:

*   **`TEST_F(CorruptionTest, Recovery)`:** 测试日志文件损坏后的恢复能力。
    1.  `Build(100)`:  写入 100 条记录。
    2.  `Check(100, 100)`: 确认所有记录都写入成功。
    3.  `Corrupt(kLogFile, 19, 1)`: 损坏日志文件的第 19 个字节，这会破坏第一个 `WriteBatch` 的标签。
    4.  `Corrupt(kLogFile, log::kBlockSize + 1000, 1)`: 损坏日志文件中第二个块的某个位置。
    5.  `Reopen()`: 重新打开数据库，触发恢复过程。
    6.  `Check(36, 36)`: 验证恢复后的数据。由于损坏，可能会丢失一些记录。

*   **`TEST_F(CorruptionTest, RecoverWriteError)`:** 测试在写入过程中遇到错误时的恢复能力。
    1.  `env_.writable_file_error_ = true`: 设置 `ErrorEnv`，使其在写入文件时返回错误。
    2.  `TryReopen()`: 尝试重新打开数据库。  由于 `ErrorEnv` 的设置，打开应该会失败。
    3.  `ASSERT_TRUE(!s.ok())`: 验证打开操作是否失败。

*   **`TEST_F(CorruptionTest, TableFile)`:** 测试数据表文件（SSTable）损坏后的情况。
    1.  `Build(100)`: 写入 100 条记录。
    2.  `DBImpl* dbi = reinterpret_cast<DBImpl*>(db_)`:  获取对 `DBImpl` 对象的访问权限。 `DBImpl` 是 `DB` 的具体实现类，可以访问一些内部函数.
    3.  `dbi->TEST_CompactMemTable()`: 将内存表（MemTable）压缩到 SSTable 文件。
    4.  `dbi->TEST_CompactRange(0, nullptr, nullptr)`: 手动触发 level 0 的压缩.
    5.  `dbi->TEST_CompactRange(1, nullptr, nullptr)`: 手动触发 level 1 的压缩.
    6.  `Corrupt(kTableFile, 100, 1)`: 损坏数据表文件的第 100 个字节。
    7.  `Check(90, 99)`: 验证损坏后的数据。

*   **`TEST_F(CorruptionTest, SequenceNumberRecovery)`:** 这是一个更复杂的测试，用于验证 LevelDB 在恢复期间是否正确处理了序列号。
    1.  写入五个不同的值到同一个key "foo"。
    2.  `RepairDB()` 尝试修复数据库。
    3.  `Reopen()` 重新打开数据库。
    4.  `ASSERT_EQ("v5", v)` 确保正确恢复了 "foo" 的最后一个值。
    5. 验证即使在恢复之后，写入新值也能正确工作，并且不会被旧的数据覆盖。

**General Usage Pattern:**

1.  **Build Data:** Use `Build(n)` to populate the database with initial data.
2.  **Introduce Corruption:** Use `Corrupt(filetype, offset, bytes_to_corrupt)` to simulate data corruption in a specific file.
3.  **Attempt Recovery:** Use `Reopen()` or `RepairDB()` to attempt to recover from the corruption.
4.  **Check Results:** Use `Check(min_expected, max_expected)` to verify the state of the database after the recovery attempt.  Assertions (`ASSERT_TRUE`, `ASSERT_EQ`, `ASSERT_LE`, `ASSERT_GE`) are used to check for expected behavior.

**中文总结:**

这些测试用例的目标是模拟各种数据损坏的情况，例如日志文件损坏、数据表文件损坏、描述文件丢失等等。 通过这些测试，可以验证 LevelDB 在遇到这些问题时是否能够正确地恢复数据，以及是否能够防止数据丢失。 `TEST_F` 是 Google Test 的一部分，允许你创建针对特定类（在这个例子中是 `CorruptionTest`）的测试用例。

希望这个解释更清楚！ 如果你有任何其他问题，请随时提问。
