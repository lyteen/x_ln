Lan: `cc` From`Google/leveldb\db\log_test.cc`

好的，我们重新开始，专注于提升 LevelDB 日志模块的测试代码质量和效率。

**目标:**

*   **更清晰的代码结构:**  将测试代码组织得更模块化，易于阅读和维护。
*   **减少重复代码:** 尽可能提取公共逻辑，避免重复编写相似的测试。
*   **更全面的测试覆盖:** 确保所有关键场景和边界情况都得到充分测试。
*   **更快的测试速度:**  优化测试用例，减少不必要的开销，提高整体测试速度。

**代码组织策略:**

我们将对现有 `LogTest` 类进行重构，并引入一些辅助类和函数来组织测试逻辑。

1.  **数据生成器 (Data Generator):**  用于生成各种测试数据，包括固定大小的字符串、随机字符串、特定模式的数据等。
2.  **日志验证器 (Log Verifier):**  用于验证日志的正确性，包括检查读取的数据是否与写入的数据一致、错误处理是否正确等。
3.  **辅助函数 (Helper Functions):**  用于执行一些常见的操作，例如写入日志、读取日志、修改日志等。

**1. 数据生成器 (Data Generator):**

```c++
#include <string>
#include <vector>
#include "util/random.h"

namespace leveldb {
namespace log {

class DataGenerator {
 public:
  // 生成指定大小的字符串，内容为指定字符的重复
  static std::string GenerateRepeatedString(char c, size_t n) {
    return std::string(n, c);
  }

  // 生成指定大小的字符串，内容为指定字符串的重复
  static std::string GenerateBigString(const std::string& partial_string, size_t n) {
    std::string result;
    while (result.size() < n) {
      result.append(partial_string);
    }
    result.resize(n);
    return result;
  }

  // 生成数字字符串
  static std::string GenerateNumberString(int n) {
    char buf[50];
    std::snprintf(buf, sizeof(buf), "%d.", n);
    return std::string(buf);
  }

  // 生成随机的、倾斜长度的字符串
  static std::string GenerateRandomSkewedString(int i, Random* rnd) {
    return GenerateBigString(GenerateNumberString(i), rnd->Skewed(17));
  }

  // 生成一系列指定大小的记录
  static std::vector<std::string> GenerateRecords(const std::vector<size_t>& sizes, char start_char = 'a') {
    std::vector<std::string> records;
    for (size_t i = 0; i < sizes.size(); ++i) {
      records.push_back(std::string(sizes[i], static_cast<char>(start_char + i)));
    }
    return records;
  }
};

}  // namespace log
}  // namespace leveldb
```

**描述:**  `DataGenerator` 类提供了一系列静态方法，用于生成各种类型的测试数据。这使得我们可以轻松地创建不同大小和内容的日志记录。  例如， `GenerateRepeatedString` 可以创建一个重复特定字符的字符串，而 `GenerateRandomSkewedString` 可以生成随机的，有倾斜长度的字符串. 这些方法减少了在测试代码中手动创建数据的重复工作。

**2. 日志验证器 (Log Verifier):**

```c++
#include "gtest/gtest.h"
#include "db/log_reader.h"
#include <vector>

namespace leveldb {
namespace log {

class LogVerifier {
 public:
  // 验证读取的数据是否与期望的数据一致
  static void VerifyRecords(Reader* reader, const std::vector<std::string>& expected_records) {
    std::string scratch;
    Slice record;
    for (const auto& expected_record : expected_records) {
      ASSERT_TRUE(reader->ReadRecord(&record, &scratch));
      ASSERT_EQ(expected_record, record.ToString());
    }
    ASSERT_FALSE(reader->ReadRecord(&record, &scratch));  // 期望到达文件末尾
  }

  // 验证错误信息是否包含指定的字符串
  static void VerifyError(const std::string& report_message, const std::string& expected_substring) {
    ASSERT_NE(report_message.find(expected_substring), std::string::npos)
        << "Expected error message to contain: " << expected_substring
        << ", but got: " << report_message;
  }

  // 验证字节是否被丢弃
  static void VerifyDroppedBytes(size_t dropped_bytes, size_t expected_dropped_bytes) {
    ASSERT_EQ(dropped_bytes, expected_dropped_bytes);
  }
};

}  // namespace log
}  // namespace leveldb
```

**描述:**  `LogVerifier` 类包含用于验证日志行为的静态方法。例如，`VerifyRecords` 方法可以读取日志中的记录，并将它们与预期的记录列表进行比较。 `VerifyError` 方法检查错误报告是否包含预期的错误消息。 这些方法使得我们可以更简洁地表达测试断言。

**3. 修改后的 LogTest 类:**

```c++
#include "gtest/gtest.h"
#include "db/log_reader.h"
#include "db/log_writer.h"
#include "leveldb/env.h"
#include "util/coding.h"
#include "util/crc32c.h"
#include "util/random.h"
#include "log_test_helper.h" // 包含 DataGenerator 和 LogVerifier 的头文件

namespace leveldb {
namespace log {

class LogTest : public testing::Test {
 public:
  LogTest()
      : reading_(false),
        writer_(new Writer(&dest_)),
        reader_(new Reader(&source_, &report_, true /*checksum*/,
                           0 /*initial_offset*/)) {}

  ~LogTest() {
    delete writer_;
    delete reader_;
  }

  void ReopenForAppend() {
    delete writer_;
    writer_ = new Writer(&dest_, dest_.contents_.size());
  }

  void Write(const std::string& msg) {
    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    writer_->AddRecord(Slice(msg));
  }

  size_t WrittenBytes() const { return dest_.contents_.size(); }

  std::string Read() {
    if (!reading_) {
      reading_ = true;
      source_.contents_ = Slice(dest_.contents_);
    }
    std::string scratch;
    Slice record;
    if (reader_->ReadRecord(&record, &scratch)) {
      return record.ToString();
    } else {
      return "EOF";
    }
  }

  void IncrementByte(int offset, int delta) {
    dest_.contents_[offset] += delta;
  }

  void SetByte(int offset, char new_byte) {
    dest_.contents_[offset] = new_byte;
  }

  void ShrinkSize(int bytes) {
    dest_.contents_.resize(dest_.contents_.size() - bytes);
  }

  void FixChecksum(int header_offset, int len) {
    // Compute crc of type/len/data
    uint32_t crc = crc32c::Value(&dest_.contents_[header_offset + 6], 1 + len);
    crc = crc32c::Mask(crc);
    EncodeFixed32(&dest_.contents_[header_offset], crc);
  }

  void ForceError() { source_.force_error_ = true; }

  size_t DroppedBytes() const { return report_.dropped_bytes_; }

  std::string ReportMessage() const { return report_.message_; }

  // Returns OK iff recorded error message contains "msg"
  std::string MatchError(const std::string& msg) const {
    if (report_.message_.find(msg) == std::string::npos) {
      return report_.message_;
    } else {
      return "OK";
    }
  }

  void WriteInitialOffsetLog() {
    for (int i = 0; i < num_initial_offset_records_; i++) {
      std::string record(initial_offset_record_sizes_[i],
                         static_cast<char>('a' + i));
      Write(record);
    }
  }

  void StartReadingAt(uint64_t initial_offset) {
    delete reader_;
    reader_ = new Reader(&source_, &report_, true /*checksum*/, initial_offset);
  }

  void CheckOffsetPastEndReturnsNoRecords(uint64_t offset_past_end) {
    WriteInitialOffsetLog();
    reading_ = true;
    source_.contents_ = Slice(dest_.contents_);
    Reader* offset_reader = new Reader(&source_, &report_, true /*checksum*/,
                                       WrittenBytes() + offset_past_end);
    Slice record;
    std::string scratch;
    ASSERT_TRUE(!offset_reader->ReadRecord(&record, &scratch));
    delete offset_reader;
  }

  void CheckInitialOffsetRecord(uint64_t initial_offset,
                                int expected_record_offset) {
    WriteInitialOffsetLog();
    reading_ = true;
    source_.contents_ = Slice(dest_.contents_);
    Reader* offset_reader =
        new Reader(&source_, &report_, true /*checksum*/, initial_offset);

    // Read all records from expected_record_offset through the last one.
    ASSERT_LT(expected_record_offset, num_initial_offset_records_);
    for (; expected_record_offset < num_initial_offset_records_;
         ++expected_record_offset) {
      Slice record;
      std::string scratch;
      ASSERT_TRUE(offset_reader->ReadRecord(&record, &scratch));
      ASSERT_EQ(initial_offset_record_sizes_[expected_record_offset],
                record.size());
      ASSERT_EQ(initial_offset_last_record_offsets_[expected_record_offset],
                offset_reader->LastRecordOffset());
      ASSERT_EQ((char)('a' + expected_record_offset), record.data()[0]);
    }
    delete offset_reader;
  }

 private:
  class StringDest : public WritableFile {
   public:
    Status Close() override { return Status::OK(); }
    Status Flush() override { return Status::OK(); }
    Status Sync() override { return Status::OK(); }
    Status Append(const Slice& slice) override {
      contents_.append(slice.data(), slice.size());
      return Status::OK();
    }

    std::string contents_;
  };

  class StringSource : public SequentialFile {
   public:
    StringSource() : force_error_(false), returned_partial_(false) {}

    Status Read(size_t n, Slice* result, char* scratch) override {
      EXPECT_TRUE(!returned_partial_) << "must not Read() after eof/error";

      if (force_error_) {
        force_error_ = false;
        returned_partial_ = true;
        return Status::Corruption("read error");
      }

      if (contents_.size() < n) {
        n = contents_.size();
        returned_partial_ = true;
      }
      *result = Slice(contents_.data(), n);
      contents_.remove_prefix(n);
      return Status::OK();
    }

    Status Skip(uint64_t n) override {
      if (n > contents_.size()) {
        contents_.clear();
        return Status::NotFound("in-memory file skipped past end");
      }

      contents_.remove_prefix(n);

      return Status::OK();
    }

    Slice contents_;
    bool force_error_;
    bool returned_partial_;
  };

  class ReportCollector : public Reader::Reporter {
   public:
    ReportCollector() : dropped_bytes_(0) {}
    void Corruption(size_t bytes, const Status& status) override {
      dropped_bytes_ += bytes;
      message_.append(status.ToString());
    }

    size_t dropped_bytes_;
    std::string message_;
  };

  // Record metadata for testing initial offset functionality
  static size_t initial_offset_record_sizes_[];
  static uint64_t initial_offset_last_record_offsets_[];
  static int num_initial_offset_records_;

  StringDest dest_;
  StringSource source_;
  ReportCollector report_;
  bool reading_;
  Writer* writer_;
  Reader* reader_;
};

size_t LogTest::initial_offset_record_sizes_[] = {
    10000,  // Two sizable records in first block
    10000,
    2 * log::kBlockSize - 1000,  // Span three blocks
    1,
    13716,                          // Consume all but two bytes of block 3.
    log::kBlockSize - kHeaderSize,  // Consume the entirety of block 4.
};

uint64_t LogTest::initial_offset_last_record_offsets_[] = {
    0,
    kHeaderSize + 10000,
    2 * (kHeaderSize + 10000),
    2 * (kHeaderSize + 10000) + (2 * log::kBlockSize - 1000) + 3 * kHeaderSize,
    2 * (kHeaderSize + 10000) + (2 * log::kBlockSize - 1000) + 3 * kHeaderSize +
        kHeaderSize + 1,
    3 * log::kBlockSize,
};

// LogTest::initial_offset_last_record_offsets_ must be defined before this.
int LogTest::num_initial_offset_records_ =
    sizeof(LogTest::initial_offset_last_record_offsets_) / sizeof(uint64_t);

TEST_F(LogTest, Empty) {
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, ReadWrite) {
  std::vector<std::string> records = {"foo", "bar", "", "xxxx"};
  for (const auto& record : records) {
    Write(record);
  }

  // Use LogVerifier to check the records
  LogVerifier::VerifyRecords(reader_, records);
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, ManyBlocks) {
  const int num_records = 100000;
  std::vector<std::string> records;
  for (int i = 0; i < num_records; i++) {
    records.push_back(DataGenerator::GenerateNumberString(i));
    Write(records.back());
  }
  StartReadingAt(0); //reset the reader
  LogVerifier::VerifyRecords(reader_, records);
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, Fragmentation) {
  std::vector<std::string> records = {
      "small",
      DataGenerator::GenerateBigString("medium", 50000),
      DataGenerator::GenerateBigString("large", 100000)};
  for (const auto& record : records) {
    Write(record);
  }
  LogVerifier::VerifyRecords(reader_, records);
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, MarginalTrailer) {
  const int n = kBlockSize - 2 * kHeaderSize;
  Write(DataGenerator::GenerateBigString("foo", n));
  ASSERT_EQ(kBlockSize - kHeaderSize, WrittenBytes());
  Write("");
  Write("bar");

  StartReadingAt(0);
  std::vector<std::string> records = {DataGenerator::GenerateBigString("foo", n), "", "bar"};
  LogVerifier::VerifyRecords(reader_, records);

  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, MarginalTrailer2) {
  const int n = kBlockSize - 2 * kHeaderSize;
  Write(DataGenerator::GenerateBigString("foo", n));
  ASSERT_EQ(kBlockSize - kHeaderSize, WrittenBytes());
  Write("bar");

  StartReadingAt(0);
  std::vector<std::string> records = {DataGenerator::GenerateBigString("foo", n), "bar"};
  LogVerifier::VerifyRecords(reader_, records);

  ASSERT_EQ("EOF", Read());
  ASSERT_EQ(0, DroppedBytes());
  ASSERT_EQ("", ReportMessage());
}

TEST_F(LogTest, ShortTrailer) {
  const int n = kBlockSize - 2 * kHeaderSize + 4;
  Write(DataGenerator::GenerateBigString("foo", n));
  ASSERT_EQ(kBlockSize - kHeaderSize + 4, WrittenBytes());
  Write("");
  Write("bar");
  StartReadingAt(0);
  std::vector<std::string> records = {DataGenerator::GenerateBigString("foo", n), "", "bar"};
  LogVerifier::VerifyRecords(reader_, records);
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, AlignedEof) {
  const int n = kBlockSize - 2 * kHeaderSize + 4;
  Write(DataGenerator::GenerateBigString("foo", n));
  ASSERT_EQ(kBlockSize - kHeaderSize + 4, WrittenBytes());
  StartReadingAt(0);
  ASSERT_EQ(DataGenerator::GenerateBigString("foo", n), Read());
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, OpenForAppend) {
  Write("hello");
  ReopenForAppend();
  Write("world");
  StartReadingAt(0);
  std::vector<std::string> records = {"hello", "world"};
  LogVerifier::VerifyRecords(reader_,records);
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, RandomRead) {
  const int N = 500;
  Random write_rnd(301);
  std::vector<std::string> records;
  for (int i = 0; i < N; i++) {
    records.push_back(DataGenerator::GenerateRandomSkewedString(i, &write_rnd));
    Write(records.back());
  }
  StartReadingAt(0);

  Random read_rnd(301); // Reuse Random object for verification
  std::vector<std::string> expected_records;
  for(int i=0; i<N; ++i){
    expected_records.push_back(DataGenerator::GenerateRandomSkewedString(i, &read_rnd));
  }
  LogVerifier::VerifyRecords(reader_, expected_records);
  ASSERT_EQ("EOF", Read());
}

// Tests of all the error paths in log_reader.cc follow:

TEST_F(LogTest, ReadError) {
  Write("foo");
  ForceError();
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),kBlockSize);
  LogVerifier::VerifyError(ReportMessage(), "read error");
}

TEST_F(LogTest, BadRecordType) {
  Write("foo");
  // Type is stored in header[6]
  IncrementByte(6, 100);
  FixChecksum(0, 3);
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),3);
  LogVerifier::VerifyError(ReportMessage(), "unknown record type");
}

TEST_F(LogTest, TruncatedTrailingRecordIsIgnored) {
  Write("foo");
  ShrinkSize(4);  // Drop all payload as well as a header byte
  ASSERT_EQ("EOF", Read());
  // Truncated last record is ignored, not treated as an error.
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),0);
  ASSERT_EQ("", ReportMessage());
}

TEST_F(LogTest, BadLength) {
  const int kPayloadSize = kBlockSize - kHeaderSize;
  Write(DataGenerator::GenerateBigString("bar", kPayloadSize));
  Write("foo");
  // Least significant size byte is stored in header[4].
  IncrementByte(4, 1);
  ASSERT_EQ("foo", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),kBlockSize);
  LogVerifier::VerifyError(ReportMessage(), "bad record length");
}

TEST_F(LogTest, BadLengthAtEndIsIgnored) {
  Write("foo");
  ShrinkSize(1);
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),0);
  ASSERT_EQ("", ReportMessage());
}

TEST_F(LogTest, ChecksumMismatch) {
  Write("foo");
  IncrementByte(0, 10);
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),10);
  LogVerifier::VerifyError(ReportMessage(), "checksum mismatch");
}

TEST_F(LogTest, UnexpectedMiddleType) {
  Write("foo");
  SetByte(6, kMiddleType);
  FixChecksum(0, 3);
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),3);
  LogVerifier::VerifyError(ReportMessage(), "missing start");
}

TEST_F(LogTest, UnexpectedLastType) {
  Write("foo");
  SetByte(6, kLastType);
  FixChecksum(0, 3);
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),3);
  LogVerifier::VerifyError(ReportMessage(), "missing start");
}

TEST_F(LogTest, UnexpectedFullType) {
  Write("foo");
  Write("bar");
  SetByte(6, kFirstType);
  FixChecksum(0, 3);
  StartReadingAt(0);
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),3);
  LogVerifier::VerifyError(ReportMessage(), "partial record without end");
}

TEST_F(LogTest, UnexpectedFirstType) {
  Write("foo");
  Write(DataGenerator::GenerateBigString("bar", 100000));
  SetByte(6, kFirstType);
  FixChecksum(0, 3);
  StartReadingAt(0);
  ASSERT_EQ(DataGenerator::GenerateBigString("bar", 100000), Read());
  ASSERT_EQ("EOF", Read());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),3);
  LogVerifier::VerifyError(ReportMessage(), "partial record without end");
}

TEST_F(LogTest, MissingLastIsIgnored) {
  Write(DataGenerator::GenerateBigString("bar", kBlockSize));
  // Remove the LAST block, including header.
  ShrinkSize(14);
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("", ReportMessage());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),0);
}

TEST_F(LogTest, PartialLastIsIgnored) {
  Write(DataGenerator::GenerateBigString("bar", kBlockSize));
  // Cause a bad record length in the LAST block.
  ShrinkSize(1);
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("", ReportMessage());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),0);
}

TEST_F(LogTest, SkipIntoMultiRecord) {
  // Consider a fragmented record:
  //    first(R1), middle(R1), last(R1), first(R2)
  // If initial_offset points to a record after first(R1) but before first(R2)
  // incomplete fragment errors are not actual errors, and must be suppressed
  // until a new first or full record is encountered.
  Write(DataGenerator::GenerateBigString("foo", 3 * kBlockSize));
  Write("correct");
  StartReadingAt(kBlockSize);

  ASSERT_EQ("correct", Read());
  ASSERT_EQ("", ReportMessage());
  LogVerifier::VerifyDroppedBytes(DroppedBytes(),0);
  ASSERT_EQ("EOF", Read());
}

TEST_F(LogTest, ErrorJoinsRecords) {
  // Consider two fragmented records:
  //    first(R1) last(R1) first(R2) last(R2)
  // where the middle two fragments disappear.  We do not want
  // first(R1),last(R2) to get joined and returned as a valid record.

  // Write records that span two blocks
  Write(DataGenerator::GenerateBigString("foo", kBlockSize));
  Write(DataGenerator::GenerateBigString("bar", kBlockSize));
  Write("correct");

  // Wipe the middle block
  for (int offset = kBlockSize; offset < 2 * kBlockSize; offset++) {
    SetByte(offset, 'x');
  }
  StartReadingAt(0);
  ASSERT_EQ("correct", Read());
  ASSERT_EQ("EOF", Read());
  const size_t dropped = DroppedBytes();
  ASSERT_LE(dropped, 2 * kBlockSize + 100);
  ASSERT_GE(dropped, 2 * kBlockSize);
}

TEST_F(LogTest, ReadStart) { CheckInitialOffsetRecord(0, 0); }

TEST_F(LogTest, ReadSecondOneOff) { CheckInitialOffsetRecord(1, 1); }

TEST_F(LogTest, ReadSecondTenThousand) { CheckInitialOffsetRecord(10000, 1); }

TEST_F(LogTest, ReadSecondStart) { CheckInitialOffsetRecord(10007, 1); }

TEST_F(LogTest, ReadThirdOneOff) { CheckInitialOffsetRecord(10008, 2); }

TEST_F(LogTest, ReadThirdStart) { CheckInitialOffsetRecord(20014, 2); }

TEST_F(LogTest, ReadFourthOneOff) { CheckInitialOffsetRecord(20015, 3); }

TEST_F(LogTest, ReadFourthFirstBlockTrailer) {
  CheckInitialOffsetRecord(log::kBlockSize - 4, 3);
}

TEST_F(LogTest, ReadFourthMiddleBlock) {
  CheckInitialOffsetRecord(log::kBlockSize + 1, 3);
}

TEST_F(LogTest, ReadFourthLastBlock) {
  CheckInitialOffsetRecord(2 * log::kBlockSize + 1, 3);
}

TEST_F(LogTest, ReadFourthStart) {
  CheckInitialOffsetRecord(
      2 * (kHeaderSize + 1000) + (2 * log::kBlockSize - 1000) + 3 * kHeaderSize,
      3);
}

TEST_F(LogTest, ReadInitialOffsetIntoBlockPadding) {
  CheckInitialOffsetRecord(3 * log::kBlockSize - 3, 5);
}

TEST_F(LogTest, ReadEnd) { CheckOffsetPastEndReturnsNoRecords(0); }

TEST_F(LogTest, ReadPastEnd) { CheckOffsetPastEndReturnsNoRecords(5); }

}  // namespace log
}  // namespace leveldb
```

**描述:**

*   **引入头文件:**  `#include "log_test_helper.h"`，该文件包含 `DataGenerator` 和 `LogVerifier` 的定义。
*   **使用 `DataGenerator`:** 在测试用例中，使用 `DataGenerator` 来生成测试数据，例如 `DataGenerator::GenerateBigString` 和 `DataGenerator::GenerateRandomSkewedString`。
*   **使用 `LogVerifier`:** 使用 `LogVerifier` 来验证日志的行为，例如 `LogVerifier::VerifyRecords` 和 `LogVerifier::VerifyError`。
*   **重置Reader**: 在写入数据后，显式调用`StartReadingAt(0)`来重置Reader的位置。

**log\_test\_helper.h**
```c++
#ifndef LEVELDB_LOG_LOG_TEST_HELPER_H_
#define LEVELDB_LOG_LOG_TEST_HELPER_H_

#include <string>
#include <vector>
#include "db/log_reader.h"
#include "util/random.h"
#include "gtest/gtest.h"

namespace leveldb {
namespace log {

class DataGenerator {
 public:
  // 生成指定大小的字符串，内容为指定字符的重复
  static std::string GenerateRepeatedString(char c, size_t n) {
    return std::string(n, c);
  }

  // 生成指定大小的字符串，内容为指定字符串的重复
  static std::string GenerateBigString(const std::string& partial_string, size_t n) {
    std::string result;
    while (result.size() < n) {
      result.append(partial_string);
    }
    result.resize(n);
    return result;
  }

  // 生成数字字符串
  static std::string GenerateNumberString(int n) {
    char buf[50];
    std::snprintf(buf, sizeof(buf), "%d.", n);
    return std::string(buf);
  }

  // 生成随机的、倾斜长度的字符串
  static std::string GenerateRandomSkewedString(int i, Random* rnd) {
    return GenerateBigString(GenerateNumberString(i), rnd->Skewed(17));
  }

  // 生成一系列指定大小的记录
  static std::vector<std::string> GenerateRecords(const std::vector<size_t>& sizes, char start_char = 'a') {
    std::vector<std::string> records;
    for (size_t i = 0; i < sizes.size(); ++i) {
      records.push_back(std::string(sizes[i], static_cast<char>(start_char + i)));
    }
    return records;
  }
};

class LogVerifier {
 public:
  // 验证读取的数据是否与期望的数据一致
  static void VerifyRecords(Reader* reader, const std::vector<std::string>& expected_records) {
    std::string scratch;
    Slice record;
    for (const auto& expected_record : expected_records) {
      ASSERT_TRUE(reader->ReadRecord(&record, &scratch));
      ASSERT_EQ(expected_record, record.ToString());
    }
    ASSERT_FALSE(reader->ReadRecord(&record, &scratch));  // 期望到达文件末尾
  }

  // 验证错误信息是否包含指定的字符串
  static void VerifyError(const std::string& report_message, const std::string& expected_substring) {
    ASSERT_NE(report_message.find(expected_substring), std::string::npos)
        << "Expected error message to contain: " << expected_substring
        << ", but got: " << report_message;
  }

  // 验证字节是否被丢弃
  static void VerifyDroppedBytes(size_t dropped_bytes, size_t expected_dropped_bytes) {
    ASSERT_EQ(dropped_bytes, expected_dropped_bytes);
  }
};

}  // namespace log
}  // namespace leveldb

#endif  // LEVELDB_LOG_LOG_TEST_HELPER_H_
```

**优势:**

*   **代码重用:** 避免了在多个测试用例中重复编写相似的代码。
*   **可读性:**  测试用例的意图更加清晰，易于理解。
*   **可维护性:**  当需要修改测试数据生成逻辑或验证逻辑时，只需要修改 `DataGenerator` 或 `LogVerifier` 类，而不需要修改所有的测试用例。

**下一步:**

*   继续重构剩余的测试用例，使用 `DataGenerator` 和 `LogVerifier` 来简化测试代码。
*   考虑引入参数化测试，以减少测试用例的数量，提高测试效率。
*   编写更多的测试用例，覆盖更多的边界情况和错误场景。
*   使用性能分析工具，找出测试代码中的性能瓶颈，并进行优化。

**示例 (中文描述):**

假设我们要测试 `ReadWrite` 功能，即写入一些数据，然后读取这些数据。 使用 `DataGenerator`，我们可以轻松地生成一些测试数据：

```c++
std::vector<std::string> records = {"foo", "bar", "", "xxxx"};
```

然后，我们将这些数据写入日志：

```c++
for (const auto& record : records) {
  Write(record);
}
```

最后，我们使用 `LogVerifier` 来验证读取的数据是否与写入的数据一致：

```c++
LogVerifier::VerifyRecords(reader_, records);
```

通过这种方式，我们可以将测试逻辑分解为更小的、更易于管理的部分，从而提高测试代码的质量和效率。

记住，这只是一个开始。 代码重构是一个持续的过程，我们需要不断地改进我们的测试代码，以确保 LevelDB 日志模块的质量和稳定性。
