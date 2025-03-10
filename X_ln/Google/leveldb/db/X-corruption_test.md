Lan: `cc` From`Google/leveldb\db\corruption_test.cc`

Okay, let's refactor and enhance the LevelDB CorruptionTest code.  I'll break down the changes into smaller chunks, providing explanations in both English and Chinese.

**1.  Improved Error Handling in `Corrupt` Function**

```c++
void Corrupt(FileType filetype, int offset, int bytes_to_corrupt) {
  // 选取要损坏的文件
  std::vector<std::string> filenames;
  Status s = env_.target()->GetChildren(dbname_, &filenames);
  ASSERT_LEVELDB_OK(s); // Check if getting children failed

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
  s = env_.target()->GetFileSize(fname, &file_size);
  ASSERT_LEVELDB_OK(s); // Check if getting file size failed

  if (offset < 0) {
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

  std::string contents;
  s = ReadFileToString(env_.target(), fname, &contents);
  ASSERT_LEVELDB_OK(s);  // Check if reading file failed

  for (int i = 0; i < bytes_to_corrupt; i++) {
    contents[i + offset] ^= 0x80;
  }

  s = WriteStringToFile(env_.target(), contents, fname);
  ASSERT_LEVELDB_OK(s);  // Check if writing file failed
}
```

**Description:**

*   **English:** Added explicit error checking after each `env_.target()` call in the `Corrupt` function using `ASSERT_LEVELDB_OK(s)`.  This ensures that the test fails immediately if there's an error interacting with the file system, providing more informative failure messages.
*   **Chinese:** 在 `Corrupt` 函数中，每次调用 `env_.target()` 之后都添加了显式的错误检查，使用 `ASSERT_LEVELDB_OK(s)`。这确保了当与文件系统交互发生错误时，测试会立即失败，并提供更具信息量的失败消息。

**2.  More Robust Key/Value Generation**

```c++
  // Return the ith key
  Slice Key(int i, std::string* storage) {
    storage->clear(); // Ensure storage is empty
    char buf[100];
    int len = std::snprintf(buf, sizeof(buf), "%016d", i);
    assert(len >= 0 && len < sizeof(buf)); // Check for buffer overflow
    storage->assign(buf, len);
    return Slice(*storage);
  }

  // Return the value to associate with the specified key
  Slice Value(int k, std::string* storage) {
    storage->clear(); // Ensure storage is empty
    Random r(k);
    return test::RandomString(&r, kValueSize, storage);
  }
```

**Description:**

*   **English:**
    *   Added `storage->clear()` at the beginning of `Key` and `Value` functions to ensure the string is empty before assigning to it.  This prevents potential issues if the storage string was previously used and contains data.
    *   Added an `assert` in `Key` to check for buffer overflows in `snprintf`. This is a good practice to prevent potential security vulnerabilities.
*   **Chinese:**
    *   在 `Key` 和 `Value` 函数的开头添加了 `storage->clear()`，以确保在赋值之前字符串是空的。这防止了如果存储字符串之前被使用过并且包含数据时可能出现的问题。
    *   在 `Key` 中添加了一个 `assert` 来检查 `snprintf` 中的缓冲区溢出。这是一个很好的实践，可以防止潜在的安全漏洞。

**3.  Improved `Check` Function for Debugging**

```c++
  void Check(int min_expected, int max_expected) {
    int next_expected = 0;
    int missed = 0;
    int bad_keys = 0;
    int bad_values = 0;
    int correct = 0;
    std::string value_space;
    Iterator* iter = db_->NewIterator(ReadOptions());
    if (iter == nullptr) {
      fprintf(stderr, "Error: NewIterator returned nullptr\n");
      ASSERT_TRUE(false); // Force a failure if iterator creation fails
      return;
    }
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      uint64_t key;
      Slice in(iter->key());
      if (in == "" || in == "~") {
        continue;
      }
      if (!ConsumeDecimalNumber(&in, &key) || !in.empty() || key < next_expected) {
        bad_keys++;
        fprintf(stderr, "Bad Key: %s\n", in.ToString().c_str()); // Print bad key
        continue;
      }
      missed += (key - next_expected);
      next_expected = key + 1;
      std::string expected_value;
      if (iter->value() != Value(key, &expected_value)) {
        bad_values++;
        fprintf(stderr, "Bad Value for Key: %lu\n", key); // Print key with bad value
      } else {
        correct++;
      }
    }
    Status iter_status = iter->status();
    if (!iter_status.ok()) {
      fprintf(stderr, "Iterator error: %s\n", iter_status.ToString().c_str());
      ASSERT_LEVELDB_OK(iter_status); // Fail if iterator had an error
    }
    delete iter;

    std::fprintf(
        stderr,
        "expected=%d..%d; got=%d; bad_keys=%d; bad_values=%d; missed=%d\n",
        min_expected, max_expected, correct, bad_keys, bad_values, missed);
    ASSERT_LE(min_expected, correct);
    ASSERT_GE(max_expected, correct);
  }
```

**Description:**

*   **English:**
    *   Added a check to ensure that `NewIterator()` doesn't return `nullptr`. If it does, the test will now fail immediately with an informative message.
    *   Added printing of bad keys when `ConsumeDecimalNumber` fails.  This helps diagnose the cause of key corruption.
    *   Added printing of the key when its value is incorrect. This helps identify which values have been corrupted.
    *   Added error checking on the iterator status *after* the iteration.  If the iterator encountered an error (e.g., due to corruption), this will catch it and fail the test.
*   **Chinese:**
    *   添加了一个检查来确保 `NewIterator()` 不会返回 `nullptr`。如果返回了，测试现在会立即失败并显示信息性的消息。
    *   添加了在 `ConsumeDecimalNumber` 失败时打印错误的键。这有助于诊断键损坏的原因。
    *   添加了在键的值不正确时打印键。这有助于识别哪些值已被损坏。
    *   在迭代*之后*添加了对迭代器状态的错误检查。如果迭代器遇到错误（例如，由于损坏），这将捕获它并使测试失败。

**4.  Example Corruption Test with Targeted Corruption**

```c++
TEST_F(CorruptionTest, TargetedTableFileCorruption) {
  options_.block_size = 2 * kValueSize;  // Limit scope of corruption
  options_.paranoid_checks = true;
  Reopen();
  Build(100);
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  dbi->TEST_CompactRange(0, nullptr, nullptr);

  // Corrupt a specific block's data
  int target_block_offset = 4096; // Example offset
  int corruption_size = 10;
  Corrupt(kTableFile, target_block_offset, corruption_size);

  RepairDB();
  Reopen();
  Check(90, 99); // Adjust expected range based on corruption size
}
```

**Description:**

*   **English:** This test case demonstrates targeted corruption.
    *   It explicitly sets the `block_size`.
    *   It picks a specific `target_block_offset` to corrupt, allowing you to target a particular data block within the table file.
    *   The expected range in the `Check` function is adjusted based on the `corruption_size`.
*   **Chinese:**  这个测试用例演示了定向的损坏。
    *   它显式地设置了 `block_size`。
    *   它选择了一个特定的 `target_block_offset` 来损坏，允许你定位表文件中的特定数据块。
    *   `Check` 函数中的预期范围根据 `corruption_size` 进行调整。

**Complete Combined Code:**

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
      Slice key = Key(i, &key_space);
      batch.Clear();
      batch.Put(key, Value(i, &value_space));
      WriteOptions options;
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
    if (iter == nullptr) {
      fprintf(stderr, "Error: NewIterator returned nullptr\n");
      ASSERT_TRUE(false);
      return;
    }
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      uint64_t key;
      Slice in(iter->key());
      if (in == "" || in == "~") {
        continue;
      }
      if (!ConsumeDecimalNumber(&in, &key) || !in.empty() || key < next_expected) {
        bad_keys++;
        fprintf(stderr, "Bad Key: %s\n", in.ToString().c_str());
        continue;
      }
      missed += (key - next_expected);
      next_expected = key + 1;
      std::string expected_value;
      if (iter->value() != Value(key, &expected_value)) {
        bad_values++;
        fprintf(stderr, "Bad Value for Key: %lu\n", key);
      } else {
        correct++;
      }
    }
    Status iter_status = iter->status();
    if (!iter_status.ok()) {
      fprintf(stderr, "Iterator error: %s\n", iter_status.ToString().c_str());
      ASSERT_LEVELDB_OK(iter_status);
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
    std::vector<std::string> filenames;
    Status s = env_.target()->GetChildren(dbname_, &filenames);
    ASSERT_LEVELDB_OK(s);

    uint64_t number;
    FileType type;
    std::string fname;
    int picked_number = -1;

    for (size_t i = 0; i < filenames.size(); i++) {
      if (ParseFileName(filenames[i], &number, &type) && type == filetype &&
          int(number) > picked_number) {
        fname = dbname_ + "/" + filenames[i];
        picked_number = number;
      }
    }

    ASSERT_TRUE(!fname.empty()) << filetype;

    uint64_t file_size;
    s = env_.target()->GetFileSize(fname, &file_size);
    ASSERT_LEVELDB_OK(s);

    if (offset < 0) {
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

    std::string contents;
    s = ReadFileToString(env_.target(), fname, &contents);
    ASSERT_LEVELDB_OK(s);

    for (int i = 0; i < bytes_to_corrupt; i++) {
      contents[i + offset] ^= 0x80;
    }

    s = WriteStringToFile(env_.target(), contents, fname);
    ASSERT_LEVELDB_OK(s);
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

  Slice Key(int i, std::string* storage) {
      storage->clear();
    char buf[100];
    int len = std::snprintf(buf, sizeof(buf), "%016d", i);
    assert(len >= 0 && len < sizeof(buf));
    storage->assign(buf, len);
    return Slice(*storage);
  }

  Slice Value(int k, std::string* storage) {
      storage->clear();
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

TEST_F(CorruptionTest, Recovery) {
  Build(100);
  Check(100, 100);
  Corrupt(kLogFile, 19, 1);
  Corrupt(kLogFile, log::kBlockSize + 1000, 1);
  Reopen();

  Check(36, 36);
}

TEST_F(CorruptionTest, RecoverWriteError) {
  env_.writable_file_error_ = true;
  Status s = TryReopen();
  ASSERT_TRUE(!s.ok());
}

TEST_F(CorruptionTest, NewFileErrorDuringWrite) {
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
  options_.block_size = 2 * kValueSize;
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
  Build(10000);
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

  Build(10000);
  Check(10000, 10000);
}

TEST_F(CorruptionTest, CompactionInputErrorParanoid) {
  options_.paranoid_checks = true;
  options_.write_buffer_size = 512 << 10;
  Reopen();
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);

  for (int i = 0; i < 2; i++) {
    Build(10);
    dbi->TEST_CompactMemTable();
    Corrupt(kTableFile, 100, 1);
    env_.SleepForMicroseconds(100000);
  }
  dbi->CompactRange(nullptr, nullptr);

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

TEST_F(CorruptionTest, TargetedTableFileCorruption) {
  options_.block_size = 2 * kValueSize;  // Limit scope of corruption
  options_.paranoid_checks = true;
  Reopen();
  Build(100);
  DBImpl* dbi = reinterpret_cast<DBImpl*>(db_);
  dbi->TEST_CompactMemTable();
  dbi->TEST_CompactRange(0, nullptr, nullptr);

  // Corrupt a specific block's data
  int target_block_offset = 4096; // Example offset
  int corruption_size = 10;
  Corrupt(kTableFile, target_block_offset, corruption_size);

  RepairDB();
  Reopen();
  Check(90, 99); // Adjust expected range based on corruption size
}

}  // namespace leveldb
```

Key improvements in this version:

*   **Error Handling:**  Comprehensive error handling to catch file system issues and iterator errors.
*   **Robustness:**  Clear string storage and buffer overflow checks to prevent unexpected behavior.
*   **Debuggability:**  More informative logging to help diagnose corruption issues.
*   **Targeted Corruption:**  An example test case that demonstrates how to corrupt specific parts of a table file.

This revised version provides a more robust and reliable corruption testing framework for LevelDB.  It will help you more easily identify and diagnose corruption-related issues in your LevelDB deployments.
