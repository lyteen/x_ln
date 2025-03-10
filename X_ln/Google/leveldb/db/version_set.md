Lan: `h` From`Google/leveldb\db\version_set.h`

Okay, let's dissect the `db/version_set.h` header file from LevelDB. I'll break down the code into smaller, manageable pieces with explanations in Chinese, and provide example usage scenarios.

**Overall Purpose of `version_set.h`**

This header file defines the core data structures for managing the different versions of the database. LevelDB uses a multi-version concurrency control (MVCC) mechanism. This means that when data is modified, instead of overwriting the existing data, a new version of the data is created. This allows readers to access a consistent snapshot of the database even while writes are happening.  The `Version` and `VersionSet` classes are critical for this. They track the state of the database at different points in time, manage the physical files (tables) that contain the data, and orchestrate the compaction process to reclaim space and improve performance.

**1. Header Guards and Includes**

```c++
#ifndef STORAGE_LEVELDB_DB_VERSION_SET_H_
#define STORAGE_LEVELDB_DB_VERSION_SET_H_

#include <map>
#include <set>
#include <vector>

#include "db/dbformat.h"
#include "db/version_edit.h"
#include "port/port.h"
#include "port/thread_annotations.h"

namespace leveldb {

namespace log {
class Writer;
}
```

*   **`#ifndef STORAGE_LEVELDB_DB_VERSION_SET_H_ ... #endif`**:  This is a standard header guard to prevent multiple inclusions of the same header file during compilation.  This avoids errors and speeds up compilation. (这是标准的头文件保护，防止重复包含.)
*   **`#include <map>`, `#include <set>`, `#include <vector>`**: Includes standard C++ containers for storing data. (`#include` 引入标准的 C++ 容器，用于存储数据.)
*   **`#include "db/dbformat.h"`**: Includes definitions related to the database's internal data format (e.g., `InternalKey`, `SequenceNumber`). (`dbformat.h` 包含了数据库内部数据格式的定义，例如 `InternalKey`, `SequenceNumber`.)
*   **`#include "db/version_edit.h"`**: Includes the `VersionEdit` class, used to record changes to a version. (`version_edit.h` 包含了 `VersionEdit` 类，用于记录版本的修改.)
*   **`#include "port/port.h"`**: Includes platform-specific definitions and abstractions. (`port/port.h` 包含了平台相关的定义和抽象.)
*   **`#include "port/thread_annotations.h"`**:  Includes annotations for thread safety analysis (e.g., `EXCLUSIVE_LOCKS_REQUIRED`).  These are often used by static analysis tools to detect potential threading issues. (包含了线程安全分析的注解，用于检测潜在的线程问题.)
*   **`namespace leveldb { ... }`**:  All LevelDB code is within the `leveldb` namespace.  This helps prevent naming conflicts with other libraries. (`leveldb` 命名空间，避免与其他库的命名冲突.)
*   **`namespace log { class Writer; }`**: Forward declaration of the `log::Writer` class. This is used for writing to the write-ahead log. (前向声明 `log::Writer` 类，用于写入预写日志.)

**2. Forward Declarations**

```c++
class Compaction;
class Iterator;
class MemTable;
class TableBuilder;
class TableCache;
class Version;
class VersionSet;
class WritableFile;
```

*   These are forward declarations of classes that are used in this header file. This tells the compiler that these classes exist, even though their full definitions may be in other files.  This is a common technique to reduce dependencies between header files and speed up compilation. (这些是本头文件中使用的类的前向声明。告诉编译器这些类存在，即使它们的完整定义可能在其他文件中。这是一种减少头文件之间依赖关系和加快编译速度的常用技术。)

**3. Utility Functions**

```c++
// Return the smallest index i such that files[i]->largest >= key.
// Return files.size() if there is no such file.
// REQUIRES: "files" contains a sorted list of non-overlapping files.
int FindFile(const InternalKeyComparator& icmp,
             const std::vector<FileMetaData*>& files, const Slice& key);

// Returns true iff some file in "files" overlaps the user key range
// [*smallest,*largest].
// smallest==nullptr represents a key smaller than all keys in the DB.
// largest==nullptr represents a key largest than all keys in the DB.
// REQUIRES: If disjoint_sorted_files, files[] contains disjoint ranges
//           in sorted order.
bool SomeFileOverlapsRange(const InternalKeyComparator& icmp,
                           bool disjoint_sorted_files,
                           const std::vector<FileMetaData*>& files,
                           const Slice* smallest_user_key,
                           const Slice* largest_user_key);
```

*   **`FindFile`**: This function performs a binary search on a sorted list of `FileMetaData` objects to find the first file whose largest key is greater than or equal to the given `key`.  It's used to quickly locate the file that might contain a particular key. (在排序的 `FileMetaData` 对象列表中执行二分查找，以找到第一个最大键大于或等于给定 `key` 的文件。 用于快速定位可能包含特定键的文件。)
*   **`SomeFileOverlapsRange`**: This function checks if any file in a given list of `FileMetaData` objects overlaps with the specified key range (`smallest_user_key` to `largest_user_key`). It takes into account whether the files are known to be disjoint and sorted, which can optimize the overlap check. (检查给定 `FileMetaData` 对象列表中是否有任何文件与指定的键范围（`smallest_user_key` 到 `largest_user_key`）重叠。 它考虑了文件是否已知是不相交和排序的，这可以优化重叠检查。)

**Example Usage (演示用法):**

```c++
#include "db/version_set.h"
#include "db/internal_key.h"
#include "util/comparator.h"
#include "util/slice.h"
#include <iostream>

namespace leveldb {
// 简单的 FileMetaData 结构体，仅用于演示
struct FileMetaData {
    uint64_t number;
    InternalKey smallest;
    InternalKey largest;
};

int main() {
    InternalKeyComparator icmp(BytewiseComparator()); // 使用默认的字节比较器
    std::vector<FileMetaData*> files;

    // 创建一些示例文件元数据
    FileMetaData* f1 = new FileMetaData();
    f1->number = 100;
    f1->smallest = InternalKey("a", 1, kTypeValue);
    f1->largest = InternalKey("c", 1, kTypeValue);

    FileMetaData* f2 = new FileMetaData();
    f2->number = 200;
    f2->smallest = InternalKey("d", 1, kTypeValue);
    f2->largest = InternalKey("f", 1, kTypeValue);

    files.push_back(f1);
    files.push_back(f2);

    // 查找包含键 "b" 的文件
    Slice key("b");
    int index = FindFile(icmp, files, key);
    std::cout << "FindFile(\"b\") 返回索引: " << index << std::endl; // 输出: 0，因为 "b" 在 "a" 到 "c" 的范围内

    // 检查范围 "b" 到 "e" 是否与任何文件重叠
    Slice smallest_user_key("b");
    Slice largest_user_key("e");
    bool overlaps = SomeFileOverlapsRange(icmp, true, files, &smallest_user_key, &largest_user_key);
    std::cout << "范围 \"b\" 到 \"e\" 是否重叠: " << (overlaps ? "是" : "否") << std::endl; // 输出: 是，因为第一个文件重叠

    // 清理
    delete f1;
    delete f2;

    return 0;
}
} // namespace leveldb
```

**4. The `Version` Class**

```c++
class Version {
 public:
  struct GetStats {
    FileMetaData* seek_file;
    int seek_file_level;
  };

  // Append to *iters a sequence of iterators that will
  // yield the contents of this Version when merged together.
  // REQUIRES: This version has been saved (see VersionSet::SaveTo)
  void AddIterators(const ReadOptions&, std::vector<Iterator*>* iters);

  // Lookup the value for key.  If found, store it in *val and
  // return OK.  Else return a non-OK status.  Fills *stats.
  // REQUIRES: lock is not held
  Status Get(const ReadOptions&, const LookupKey& key, std::string* val,
             GetStats* stats);

  // Adds "stats" into the current state.  Returns true if a new
  // compaction may need to be triggered, false otherwise.
  // REQUIRES: lock is held
  bool UpdateStats(const GetStats& stats);

  // Record a sample of bytes read at the specified internal key.
  // Samples are taken approximately once every config::kReadBytesPeriod
  // bytes.  Returns true if a new compaction may need to be triggered.
  // REQUIRES: lock is held
  bool RecordReadSample(Slice key);

  // Reference count management (so Versions do not disappear out from
  // under live iterators)
  void Ref();
  void Unref();

  void GetOverlappingInputs(
      int level,
      const InternalKey* begin,  // nullptr means before all keys
      const InternalKey* end,    // nullptr means after all keys
      std::vector<FileMetaData*>* inputs);

  // Returns true iff some file in the specified level overlaps
  // some part of [*smallest_user_key,*largest_user_key].
  // smallest_user_key==nullptr represents a key smaller than all the DB's keys.
  // largest_user_key==nullptr represents a key largest than all the DB's keys.
  bool OverlapInLevel(int level, const Slice* smallest_user_key,
                      const Slice* largest_user_key);

  // Return the level at which we should place a new memtable compaction
  // result that covers the range [smallest_user_key,largest_user_key].
  int PickLevelForMemTableOutput(const Slice& smallest_user_key,
                                 const Slice& largest_user_key);

  int NumFiles(int level) const { return files_[level].size(); }

  // Return a human readable string that describes this version's contents.
  std::string DebugString() const;

 private:
  friend class Compaction;
  friend class VersionSet;

  class LevelFileNumIterator;

  explicit Version(VersionSet* vset)
      : vset_(vset),
        next_(this),
        prev_(this),
        refs_(0),
        file_to_compact_(nullptr),
        file_to_compact_level_(-1),
        compaction_score_(-1),
        compaction_level_(-1) {}

  Version(const Version&) = delete;
  Version& operator=(const Version&) = delete;

  ~Version();

  Iterator* NewConcatenatingIterator(const ReadOptions&, int level) const;

  // Call func(arg, level, f) for every file that overlaps user_key in
  // order from newest to oldest.  If an invocation of func returns
  // false, makes no more calls.
  //
  // REQUIRES: user portion of internal_key == user_key.
  void ForEachOverlapping(Slice user_key, Slice internal_key, void* arg,
                          bool (*func)(void*, int, FileMetaData*));

  VersionSet* vset_;  // VersionSet to which this Version belongs
  Version* next_;     // Next version in linked list
  Version* prev_;     // Previous version in linked list
  int refs_;          // Number of live refs to this version

  // List of files per level
  std::vector<FileMetaData*> files_[config::kNumLevels];

  // Next file to compact based on seek stats.
  FileMetaData* file_to_compact_;
  int file_to_compact_level_;

  // Level that should be compacted next and its compaction score.
  // Score < 1 means compaction is not strictly needed.  These fields
  // are initialized by Finalize().
  double compaction_score_;
  int compaction_level_;
};
```

*   **`Version`**: Represents a snapshot of the database at a particular point in time. It contains the list of table files for each level of the LSM tree that make up the database at that time. (表示数据库在特定时间点的快照。它包含构成该时间点数据库的 LSM 树的每一级的表文件列表。)
*   **`GetStats`**: A struct used to collect statistics during a `Get` operation. (`GetStats` 是一个结构体，用于在 `Get` 操作期间收集统计信息.)
*   **`AddIterators`**: Creates iterators to read the contents of this version, merging the files in each level.  (创建迭代器来读取此版本的内容，合并每一级别的文件.)
*   **`Get`**:  Looks up a key in this version. (`Get` 在此版本中查找键.)
*   **`UpdateStats`**: Updates statistics about file access, potentially triggering a compaction. (更新有关文件访问的统计信息，可能会触发压缩.)
*   **`RecordReadSample`**:  Records a sample of read bytes, used for triggering compactions. (记录读取字节的样本，用于触发压缩.)
*   **`Ref` / `Unref`**:  Manages the reference count of this version. A version is kept alive as long as there are iterators using it. (管理此版本的引用计数。 只要有迭代器在使用某个版本，该版本就会保持活动状态。)
*   **`GetOverlappingInputs`**: Finds the files that overlap a given key range in a specific level. (在特定级别查找与给定键范围重叠的文件。)
*   **`OverlapInLevel`**: Checks if any file in a given level overlaps a given key range. (检查给定级别中是否有任何文件与给定键范围重叠。)
*   **`PickLevelForMemTableOutput`**: Determines the level where a new memtable's output should be placed after compaction. (确定在压缩后应放置新 memtable 输出的级别。)
*   **`NumFiles`**: Returns the number of files in a given level. (返回给定级别中的文件数。)
*   **`DebugString`**: Returns a human-readable string representation of the version. (返回版本的可读字符串表示形式。)
*   **`files_[config::kNumLevels]`**: An array of vectors, where each vector contains the `FileMetaData` for a specific level. This is where the files belonging to this version are stored. (`files_[config::kNumLevels]` 是一个向量数组，其中每个向量都包含特定级别的 `FileMetaData`。 这是存储属于此版本的文件的地方。)
*   **`file_to_compact_`, `file_to_compact_level_`**: Indicate which file should be compacted next and at what level. (指示接下来应该压缩哪个文件以及在哪个级别压缩。)
*   **`compaction_score_`, `compaction_level_`**:  Determine if and at which level compaction is needed. (确定是否需要压缩以及在哪个级别需要压缩。)

**Example Usage (演示用法):**

```c++
#include "db/version_set.h"
#include "db/dbformat.h"
#include "util/comparator.h"

#include <iostream>

namespace leveldb {
int main() {
    // 假设已经有了一个 VersionSet 和 Options
    Options options; // 使用默认选项
    InternalKeyComparator icmp(BytewiseComparator());
    TableCache* table_cache = nullptr; // 需要正确初始化 TableCache
    VersionSet* vset = new VersionSet("testdb", &options, table_cache, &icmp);

    // 创建一个 Version 对象
    Version* version = new Version(vset);

    // 可以设置一些示例文件
    //version->files_[0].push_back(...); // 添加 level 0 的文件
    //version->files_[1].push_back(...); // 添加 level 1 的文件

    std::cout << "Version 对象已创建" << std::endl;
    std::cout << "Level 0 文件数量: " << version->NumFiles(0) << std::endl;

    // 后续可以调用 Version 对象的其他方法，例如 Get, AddIterators 等

    delete version;
    delete vset;
    return 0;
}
} // namespace leveldb
```

**5. The `VersionSet` Class**

```c++
class VersionSet {
 public:
  VersionSet(const std::string& dbname, const Options* options,
             TableCache* table_cache, const InternalKeyComparator*);
  VersionSet(const VersionSet&) = delete;
  VersionSet& operator=(const VersionSet&) = delete;

  ~VersionSet();

  // Apply *edit to the current version to form a new descriptor that
  // is both saved to persistent state and installed as the new
  // current version.  Will release *mu while actually writing to the file.
  // REQUIRES: *mu is held on entry.
  // REQUIRES: no other thread concurrently calls LogAndApply()
  Status LogAndApply(VersionEdit* edit, port::Mutex* mu)
      EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Recover the last saved descriptor from persistent storage.
  Status Recover(bool* save_manifest);

  // Return the current version.
  Version* current() const { return current_; }

  // Return the current manifest file number
  uint64_t ManifestFileNumber() const { return manifest_file_number_; }

  // Allocate and return a new file number
  uint64_t NewFileNumber() { return next_file_number_++; }

  // Arrange to reuse "file_number" unless a newer file number has
  // already been allocated.
  // REQUIRES: "file_number" was returned by a call to NewFileNumber().
  void ReuseFileNumber(uint64_t file_number) {
    if (next_file_number_ == file_number + 1) {
      next_file_number_ = file_number;
    }
  }

  // Return the number of Table files at the specified level.
  int NumLevelFiles(int level) const;

  // Return the combined file size of all files at the specified level.
  int64_t NumLevelBytes(int level) const;

  // Return the last sequence number.
  uint64_t LastSequence() const { return last_sequence_; }

  // Set the last sequence number to s.
  void SetLastSequence(uint64_t s) {
    assert(s >= last_sequence_);
    last_sequence_ = s;
  }

  // Mark the specified file number as used.
  void MarkFileNumberUsed(uint64_t number);

  // Return the current log file number.
  uint64_t LogNumber() const { return log_number_; }

  // Return the log file number for the log file that is currently
  // being compacted, or zero if there is no such log file.
  uint64_t PrevLogNumber() const { return prev_log_number_; }

  // Pick level and inputs for a new compaction.
  // Returns nullptr if there is no compaction to be done.
  // Otherwise returns a pointer to a heap-allocated object that
  // describes the compaction.  Caller should delete the result.
  Compaction* PickCompaction();

  // Return a compaction object for compacting the range [begin,end] in
  // the specified level.  Returns nullptr if there is nothing in that
  // level that overlaps the specified range.  Caller should delete
  // the result.
  Compaction* CompactRange(int level, const InternalKey* begin,
                           const InternalKey* end);

  // Return the maximum overlapping data (in bytes) at next level for any
  // file at a level >= 1.
  int64_t MaxNextLevelOverlappingBytes();

  // Create an iterator that reads over the compaction inputs for "*c".
  // The caller should delete the iterator when no longer needed.
  Iterator* MakeInputIterator(Compaction* c);

  // Returns true iff some level needs a compaction.
  bool NeedsCompaction() const {
    Version* v = current_;
    return (v->compaction_score_ >= 1) || (v->file_to_compact_ != nullptr);
  }

  // Add all files listed in any live version to *live.
  // May also mutate some internal state.
  void AddLiveFiles(std::set<uint64_t>* live);

  // Return the approximate offset in the database of the data for
  // "key" as of version "v".
  uint64_t ApproximateOffsetOf(Version* v, const InternalKey& key);

  // Return a human-readable short (single-line) summary of the number
  // of files per level.  Uses *scratch as backing store.
  struct LevelSummaryStorage {
    char buffer[100];
  };
  const char* LevelSummary(LevelSummaryStorage* scratch) const;

 private:
  class Builder;

  friend class Compaction;
  friend class Version;

  bool ReuseManifest(const std::string& dscname, const std::string& dscbase);

  void Finalize(Version* v);

  void GetRange(const std::vector<FileMetaData*>& inputs, InternalKey* smallest,
                InternalKey* largest);

  void GetRange2(const std::vector<FileMetaData*>& inputs1,
                 const std::vector<FileMetaData*>& inputs2,
                 InternalKey* smallest, InternalKey* largest);

  void SetupOtherInputs(Compaction* c);

  // Save current contents to *log
  Status WriteSnapshot(log::Writer* log);

  void AppendVersion(Version* v);

  Env* const env_;
  const std::string dbname_;
  const Options* const options_;
  TableCache* const table_cache_;
  const InternalKeyComparator icmp_;
  uint64_t next_file_number_;
  uint64_t manifest_file_number_;
  uint64_t last_sequence_;
  uint64_t log_number_;
  uint64_t prev_log_number_;  // 0 or backing store for memtable being compacted

  // Opened lazily
  WritableFile* descriptor_file_;
  log::Writer* descriptor_log_;
  Version dummy_versions_;  // Head of circular doubly-linked list of versions.
  Version* current_;        // == dummy_versions_.prev_

  // Per-level key at which the next compaction at that level should start.
  // Either an empty string, or a valid InternalKey.
  std::string compact_pointer_[config::kNumLevels];
};
```

*   **`VersionSet`**: Manages the different versions of the database. It's responsible for creating new versions, persisting them to disk (in the MANIFEST file), and picking compactions. It maintains a linked list of `Version` objects, with the most recent version being `current_`. (管理数据库的不同版本。 它负责创建新版本、将它们持久保存到磁盘（在 MANIFEST 文件中）以及选择压缩。 它维护 `Version` 对象的链表，其中最新版本为 `current_`。)
*   **`LogAndApply`**: Applies a `VersionEdit` to the current version, creating a new version. It also logs the changes to the MANIFEST file. This operation is critical for maintaining consistency and durability. (将 `VersionEdit` 应用于当前版本，创建一个新版本。 它还会将更改记录到 MANIFEST 文件。 此操作对于维护一致性和持久性至关重要。)
*   **`Recover`**: Recovers the last saved state of the `VersionSet` from the MANIFEST file. This is done during database startup. (从 MANIFEST 文件中恢复 `VersionSet` 的最后保存状态。 这在数据库启动期间完成。)
*   **`current()`**: Returns the current `Version`. (返回当前的 `Version`.)
*   **`NewFileNumber()`**:  Allocates a new, unique file number.  File numbers are used to identify the SSTable files on disk. (分配一个新的、唯一的文件编号。 文件编号用于标识磁盘上的 SSTable 文件。)
*   **`NumLevelFiles()`**: Returns the number of files in a given level. (返回给定级别中的文件数。)
*   **`LastSequence()`**: Returns the last sequence number used in the database. Sequence numbers are used to track the order of operations. (返回数据库中使用的最后一个序列号。 序列号用于跟踪操作的顺序。)
*   **`PickCompaction()`**: Chooses a compaction to perform. Compactions merge SSTables to reduce space and improve read performance. (选择要执行的压缩。 压缩合并 SSTable 以减少空间并提高读取性能。)
*   **`CompactRange()`**: Creates a compaction object for a specific key range within a level. (为级别内的特定键范围创建压缩对象。)
*   **`NeedsCompaction()`**:  Determines if a compaction is needed based on the compaction score and whether a file has been marked for compaction. (根据压缩分数以及是否已标记要压缩的文件来确定是否需要压缩。)
*   **`AddLiveFiles()`**: Adds all files referenced by live versions to a set.  This is used during garbage collection to identify files that are still in use and should not be deleted. (将活动版本引用的所有文件添加到集合中。 这在垃圾回收期间用于识别仍在使用的文件，不应删除。)
*   **`LevelSummary()`**: Returns a string summarizing the number of files at each level. (返回一个字符串，总结了每个级别的文件数。)
*   **`descriptor_file_`, `descriptor_log_`**:  Pointers to the MANIFEST file and its writer. These are used to persist the `VersionSet` state. (指向 MANIFEST 文件及其写入器的指针。 这些用于持久保存 `VersionSet` 状态。)
*   **`dummy_versions_`, `current_`**: Used to manage the linked list of versions.  `dummy_versions_` is a dummy head node. (`dummy_versions_`, `current_` 用于管理版本链表。 `dummy_versions_` 是一个虚拟头节点。)
*   **`compact_pointer_[config::kNumLevels]`**:  Indicates the key at which the next compaction at a particular level should start. (指示特定级别下一次压缩应该开始的键。)

**Example Usage (演示用法):**

```c++
#include "db/version_set.h"
#include "db/dbformat.h"
#include "util/comparator.h"
#include "env/env.h"  // Include for Env
#include "table/table_cache.h" // Include for TableCache

#include <iostream>

namespace leveldb {
int main() {
    // 初始化 Options, InternalKeyComparator 和 Env
    Options options;
    InternalKeyComparator icmp(BytewiseComparator());
    Env* env = Env::Default(); // 获取默认的环境
    TableCache* table_cache = new TableCache(env, &options, 100); // TableCache 需要正确初始化

    // 创建 VersionSet 对象
    VersionSet* vset = new VersionSet("testdb", &options, table_cache, &icmp);

    // 打印当前 Manifest 文件编号
    std::cout << "Manifest 文件编号: " << vset->ManifestFileNumber() << std::endl;

    // 创建一个新文件编号
    uint64_t new_file_number = vset->NewFileNumber();
    std::cout << "新文件编号: " << new_file_number << std::endl;

    // 判断是否需要 Compaction
    std::cout << "是否需要 Compaction: " << (vset->NeedsCompaction() ? "是" : "否") << std::endl;

    // 清理
    delete vset;
    delete table_cache;
    return 0;
}
} // namespace leveldb
```

**6. The `Compaction` Class**

```c++
class Compaction {
 public:
  ~Compaction();

  // Return the level that is being compacted.  Inputs from "level"
  // and "level+1" will be merged to produce a set of "level+1" files.
  int level() const { return level_; }

  // Return the object that holds the edits to the descriptor done
  // by this compaction.
  VersionEdit* edit() { return &edit_; }

  // "which" must be either 0 or 1
  int num_input_files(int which) const { return inputs_[which].size(); }

  // Return the ith input file at "level()+which" ("which" must be 0 or 1).
  FileMetaData* input(int which, int i) const { return inputs_[which][i]; }

  // Maximum size of files to build during this compaction.
  uint64_t MaxOutputFileSize() const { return max_output_file_size_; }

  // Is this a trivial compaction that can be implemented by just
  // moving a single input file to the next level (no merging or splitting)
  bool IsTrivialMove() const;

  // Add all inputs to this compaction as delete operations to *edit.
  void AddInputDeletions(VersionEdit* edit);

  // Returns true if the information we have available guarantees that
  // the compaction is producing data in "level+1" for which no data exists
  // in levels greater than "level+1".
  bool IsBaseLevelForKey(const Slice& user_key);

  // Returns true iff we should stop building the current output
  // before processing "internal_key".
  bool ShouldStopBefore(const Slice& internal_key);

  // Release the input version for the compaction, once the compaction
  // is successful.
  void ReleaseInputs();

 private:
  friend class Version;
  friend class VersionSet;

  Compaction(const Options* options, int level);

  int level_;
  uint64_t max_output_file_size_;
  Version* input_version_;
  VersionEdit edit_;

  // Each compaction reads inputs from "level_" and "level_+1"
  std::vector<FileMetaData*> inputs_[2];  // The two sets of inputs

  // State used to check for number of overlapping grandparent files
  // (parent == level_ + 1, grandparent == level_ + 2)
  std::vector<FileMetaData*> grandparents_;
  size_t grandparent_index_;  // Index in grandparent_starts_
  bool seen_key_;             // Some output key has been seen
  int64_t overlapped_bytes_;  // Bytes of overlap between current output
                              // and grandparent files

  // State for implementing IsBaseLevelForKey

  // level_ptrs_ holds indices into input_version_->levels_: our state
  // is that we are positioned at one of the file ranges for each
  // higher level than the ones involved in this compaction (i.e. for
  // all L >= level_ + 2).
  size_t level_ptrs_[config::kNumLevels];
};
```

*   **`Compaction`**: Represents a background process that merges SSTables from different levels to reduce space, improve read performance, and delete obsolete data. (表示一个后台进程，用于合并来自不同级别的 SSTable，以减少空间、提高读取性能并删除过时的数据。)
*   **`level()`**: Returns the level being compacted. (返回正在压缩的级别。)
*   **`edit()`**: Returns the `VersionEdit` object that records the changes made by this compaction. (返回 `VersionEdit` 对象，该对象记录了此压缩所做的更改。)
*   **`num_input_files()`**: Returns the number of input files for a given level (0 or 1). (返回给定级别（0 或 1）的输入文件数。)
*   **`input()`**: Returns the `FileMetaData` for a specific input file. (返回特定输入文件的 `FileMetaData`。)
*   **`MaxOutputFileSize()`**: Returns the maximum size of the output files generated by this compaction. (返回此压缩生成的输出文件的最大大小。)
*   **`AddInputDeletions()`**: Adds deletion entries for all input files to the `VersionEdit`.  This tells LevelDB to remove the old files once the compaction is complete. (将所有输入文件的删除条目添加到 `VersionEdit`。 这告诉 LevelDB 在压缩完成后删除旧文件。)
*   **`IsBaseLevelForKey()`**:  Determines if a key being written to level L has no overlap with data at levels > L. (确定写入级别 L 的键是否与级别 > L 的数据没有重叠。)
*   **`ShouldStopBefore()`**:  Used to decide when to split the output of a compaction into multiple files. (用于决定何时将压缩的输出拆分为多个文件。)
*   **`inputs_[2]`**: A two-element array of vectors, holding the input files from level `level_` and `level_ + 1`. (一个包含两个向量的数组，用于保存来自级别 `level_` 和 `level_ + 1` 的输入文件。)
*   **`grandparents_`**:  Files in level `level_ + 2` that overlap with the output of this compaction.  Used to avoid creating too much write amplification. (级别 `level_ + 2` 中与此压缩的输出重叠的文件。 用于避免创建过多的写放大。)
*    **`level_ptrs_[config::kNumLevels]`**: Used to efficiently determine `IsBaseLevelForKey`.  (用于高效地确定`IsBaseLevelForKey`.)

**Example Usage (演示用法):**

```c++
#include "db/version_set.h"
#include "db/dbformat.h"
#include "util/comparator.h"
#include "env/env.h"  // Include for Env
#include "table/table_cache.h" // Include for TableCache
#include "options/options.h"

#include <iostream>

namespace leveldb {
int main() {
    // 初始化 Options, InternalKeyComparator 和 Env
    Options options;
    InternalKeyComparator icmp(BytewiseComparator());
    Env* env = Env::Default(); // 获取默认的环境
    TableCache* table_cache = new TableCache(env, &options, 100); // TableCache 需要正确初始化

    // 创建 VersionSet 对象
    VersionSet* vset = new VersionSet("testdb", &options, table_cache, &icmp);

    // 创建一个 Compaction 对象
    Compaction* compaction = new Compaction(&options, 1); // Compaction for level 1

    // 打印 Compaction 的 Level
    std::cout << "Compaction 的 Level: " << compaction->level() << std::endl;

    // 清理
    delete compaction;
    delete vset;
    delete table_cache;
    return 0;
}
} // namespace leveldb
```

**In summary:**

This `version_set.h` file is the heart of LevelDB's version management system. It defines the classes and data structures needed to track the state of the database over time, manage the physical files that store the data, and