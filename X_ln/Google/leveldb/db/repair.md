Lan: `cc` From`Google/leveldb\db\repair.cc`

Okay, let's break down this LevelDB repair utility. This code is designed to recover a LevelDB database from a potentially corrupted state. It analyzes the existing files in the database directory, reconstructs the database's metadata, and creates a new, consistent database descriptor.  I'll provide explanations and code snippets with Chinese comments, focusing on the key aspects of the repair process.

**Overall Goal:** The `RepairDB` function attempts to salvage as much data as possible from a damaged LevelDB database. It reads log files and SSTables, extracts metadata, and creates a new, valid database state.

**1. `Repairer` Class: Core Logic**

The `Repairer` class encapsulates the entire repair process. Let's look at its key components:

*   **Constructor (`Repairer(const std::string& dbname, const Options& options)`):**

```c++
Repairer(const std::string& dbname, const Options& options)
    : dbname_(dbname),
      env_(options.env),
      icmp_(options.comparator),
      ipolicy_(options.filter_policy),
      options_(SanitizeOptions(dbname, &icmp_, &ipolicy_, options)),
      owns_info_log_(options_.info_log != options.info_log),
      owns_cache_(options_.block_cache != options.block_cache),
      next_file_number_(1) {
  // TableCache can be small since we expect each table to be opened once.
  table_cache_ = new TableCache(dbname_, options_, 10);
}
```

```cpp
// Repairer 类的构造函数。
// 作用：初始化 Repairer 对象，包括数据库名称、环境、比较器、过滤器策略、选项等。
// 初始化 TableCache，用于缓存 table 的元数据，加快访问速度。
Repairer::Repairer(const std::string& dbname, const Options& options)
    : dbname_(dbname), // 数据库名称
      env_(options.env), // 环境（文件系统等）
      icmp_(options.comparator), // 内部 key 比较器
      ipolicy_(options.filter_policy), // 过滤器策略
      options_(SanitizeOptions(dbname, &icmp_, &ipolicy_, options)), // 清理后的选项
      owns_info_log_(options_.info_log != options.info_log), // 是否拥有 info_log 的所有权
      owns_cache_(options_.block_cache != options.block_cache), // 是否拥有 block_cache 的所有权
      next_file_number_(1) { // 下一个文件编号，从 1 开始
  // TableCache can be small since we expect each table to be opened once.
  table_cache_ = new TableCache(dbname_, options_, 10); // 初始化 TableCache
}
```

*   **Destructor (`~Repairer()`):** Cleans up allocated resources like the table cache and info log.

```c++
~Repairer() {
  delete table_cache_;
  if (owns_info_log_) {
    delete options_.info_log;
  }
  if (owns_cache_) {
    delete options_.block_cache;
  }
}
```

```cpp
// Repairer 类的析构函数。
// 作用：释放 Repairer 对象占用的资源，包括 TableCache、info_log 和 block_cache。
Repairer::~Repairer() {
  delete table_cache_; // 释放 TableCache
  if (owns_info_log_) {
    delete options_.info_log; // 释放 info_log
  }
  if (owns_cache_) {
    delete options_.block_cache; // 释放 block_cache
  }
}
```

*   **`Run()` Method:** Orchestrates the repair process.

```c++
Status Run() {
  Status status = FindFiles(); // 1. 查找所有相关文件
  if (status.ok()) {
    ConvertLogFilesToTables(); // 2. 将日志文件转换为表文件
    ExtractMetaData(); // 3. 从表文件中提取元数据
    status = WriteDescriptor(); // 4. 写入新的描述符文件
  }
  if (status.ok()) {
    unsigned long long bytes = 0;
    for (size_t i = 0; i < tables_.size(); i++) {
      bytes += tables_[i].meta.file_size;
    }
    Log(options_.info_log,
        "**** Repaired leveldb %s; "
        "recovered %d files; %llu bytes. "
        "Some data may have been lost. "
        "****",
        dbname_.c_str(), static_cast<int>(tables_.size()), bytes);
  }
  return status;
}
```

```cpp
// Run 方法：执行修复过程的主流程。
// 1. 查找数据库目录中的所有相关文件（manifest 文件、log 文件、table 文件）。
// 2. 将 log 文件转换为 table 文件。
// 3. 扫描 table 文件，提取元数据（最小 key、最大 key、最大序列号）。
// 4. 将提取的元数据写入新的 descriptor 文件，完成修复。
Status Repairer::Run() {
  Status status = FindFiles(); // 1. 查找文件
  if (status.ok()) {
    ConvertLogFilesToTables(); // 2. 转换 log 文件
    ExtractMetaData(); // 3. 提取元数据
    status = WriteDescriptor(); // 4. 写入描述符文件
  }
  if (status.ok()) {
    unsigned long long bytes = 0;
    for (size_t i = 0; i < tables_.size(); i++) {
      bytes += tables_[i].meta.file_size;
    }
    Log(options_.info_log,
        "**** Repaired leveldb %s; "
        "recovered %d files; %llu bytes. "
        "Some data may have been lost. "
        "****",
        dbname_.c_str(), static_cast<int>(tables_.size()), bytes);
  }
  return status;
}
```

**2. `FindFiles()`: Identifying Database Files**

This method scans the database directory and identifies the different types of files present: manifest files, log files, and table files.

```c++
Status FindFiles() {
  std::vector<std::string> filenames;
  Status status = env_->GetChildren(dbname_, &filenames); // 获取数据库目录下所有文件名
  if (!status.ok()) {
    return status;
  }
  if (filenames.empty()) {
    return Status::IOError(dbname_, "repair found no files");
  }

  uint64_t number;
  FileType type;
  for (size_t i = 0; i < filenames.size(); i++) {
    if (ParseFileName(filenames[i], &number, &type)) {  // 解析文件名，提取文件编号和文件类型
      if (type == kDescriptorFile) {
        manifests_.push_back(filenames[i]); // 记录 manifest 文件
      } else {
        if (number + 1 > next_file_number_) {
          next_file_number_ = number + 1; // 更新下一个文件编号
        }
        if (type == kLogFile) {
          logs_.push_back(number);  // 记录 log 文件编号
        } else if (type == kTableFile) {
          table_numbers_.push_back(number); // 记录 table 文件编号
        } else {
          // Ignore other files
        }
      }
    }
  }
  return status;
}
```

```cpp
// FindFiles 方法：查找数据库目录中的所有相关文件。
// 1. 使用 env_->GetChildren() 获取数据库目录下所有文件名。
// 2. 遍历文件名列表，使用 ParseFileName() 解析文件名，提取文件编号和文件类型。
// 3. 根据文件类型，将文件名添加到相应的列表中（manifests_、logs_、table_numbers_）。
Status Repairer::FindFiles() {
  std::vector<std::string> filenames;
  Status status = env_->GetChildren(dbname_, &filenames); // 1. 获取所有文件名
  if (!status.ok()) {
    return status;
  }
  if (filenames.empty()) {
    return Status::IOError(dbname_, "repair found no files");
  }

  uint64_t number;
  FileType type;
  for (size_t i = 0; i < filenames.size(); i++) {
    if (ParseFileName(filenames[i], &number, &type)) {  // 2. 解析文件名
      if (type == kDescriptorFile) {
        manifests_.push_back(filenames[i]); // 记录 manifest 文件
      } else {
        if (number + 1 > next_file_number_) {
          next_file_number_ = number + 1; // 更新下一个文件编号
        }
        if (type == kLogFile) {
          logs_.push_back(number);  // 记录 log 文件编号
        } else if (type == kTableFile) {
          table_numbers_.push_back(number); // 记录 table 文件编号
        } else {
          // Ignore other files
        }
      }
    }
  }
  return status;
}
```

**3. `ConvertLogFilesToTables()`: Recovering from Logs**

LevelDB uses log files to record recent changes.  This method converts these log files into SSTables. This ensures that all data, even data that hasn't been compacted into tables yet, is included in the recovered database.

```c++
void ConvertLogFilesToTables() {
  for (size_t i = 0; i < logs_.size(); i++) {
    std::string logname = LogFileName(dbname_, logs_[i]); // 构建 log 文件名
    Status status = ConvertLogToTable(logs_[i]); // 将 log 文件转换为 table 文件
    if (!status.ok()) {
      Log(options_.info_log, "Log #%llu: ignoring conversion error: %s",
          (unsigned long long)logs_[i], status.ToString().c_str());
    }
    ArchiveFile(logname); // 归档 log 文件
  }
}
```

```cpp
// ConvertLogFilesToTables 方法：将 log 文件转换为 table 文件。
// 1. 遍历 logs_ 列表，获取 log 文件编号。
// 2. 使用 LogFileName() 构建 log 文件名。
// 3. 调用 ConvertLogToTable() 将 log 文件转换为 table 文件。
// 4. 调用 ArchiveFile() 归档 log 文件。
void Repairer::ConvertLogFilesToTables() {
  for (size_t i = 0; i < logs_.size(); i++) {
    std::string logname = LogFileName(dbname_, logs_[i]); // 1. 构建文件名
    Status status = ConvertLogToTable(logs_[i]); // 2. 转换 log 文件
    if (!status.ok()) {
      Log(options_.info_log, "Log #%llu: ignoring conversion error: %s",
          (unsigned long long)logs_[i], status.ToString().c_str());
    }
    ArchiveFile(logname); // 3. 归档文件
  }
}
```

The `ConvertLogToTable()` method does the actual conversion.  It reads the log file, replays the write operations into a `MemTable`, and then builds an SSTable from the `MemTable`.

```c++
Status ConvertLogToTable(uint64_t log) {
  struct LogReporter : public log::Reader::Reporter { // 定义一个 LogReporter，用于处理 log 文件读取过程中的错误
    Env* env;
    Logger* info_log;
    uint64_t lognum;
    void Corruption(size_t bytes, const Status& s) override { // 覆盖 Corruption 方法，用于记录错误信息
      Log(info_log, "Log #%llu: dropping %d bytes; %s",
          (unsigned long long)lognum, static_cast<int>(bytes),
          s.ToString().c_str());
    }
  };

  std::string logname = LogFileName(dbname_, log); // 构建 log 文件名
  SequentialFile* lfile;
  Status status = env_->NewSequentialFile(logname, &lfile); // 打开 log 文件
  if (!status.ok()) {
    return status;
  }

  LogReporter reporter;
  reporter.env = env_;
  reporter.info_log = options_.info_log;
  reporter.lognum = log;
  log::Reader reader(lfile, &reporter, false /*do not checksum*/,
                      0 /*initial_offset*/); // 创建 log reader

  std::string scratch;
  Slice record;
  WriteBatch batch;
  MemTable* mem = new MemTable(icmp_); // 创建 MemTable
  mem->Ref();
  int counter = 0;
  while (reader.ReadRecord(&record, &scratch)) { // 循环读取 log 记录
    if (record.size() < 12) {
      reporter.Corruption(record.size(),
                          Status::Corruption("log record too small"));
      continue;
    }
    WriteBatchInternal::SetContents(&batch, record);
    status = WriteBatchInternal::InsertInto(&batch, mem); // 将记录插入 MemTable
    if (status.ok()) {
      counter += WriteBatchInternal::Count(&batch);
    } else {
      Log(options_.info_log, "Log #%llu: ignoring %s",
          (unsigned long long)log, status.ToString().c_str());
      status = Status::OK();  // Keep going with rest of file
    }
  }
  delete lfile;

  FileMetaData meta;
  meta.number = next_file_number_++;
  Iterator* iter = mem->NewIterator();
  status = BuildTable(dbname_, env_, options_, table_cache_, iter, &meta); // 构建 table
  delete iter;
  mem->Unref();
  mem = nullptr;
  if (status.ok()) {
    if (meta.file_size > 0) {
      table_numbers_.push_back(meta.number);
    }
  }
  Log(options_.info_log, "Log #%llu: %d ops saved to Table #%llu %s",
      (unsigned long long)log, counter, (unsigned long long)meta.number,
      status.ToString().c_str());
  return status;
}
```

```cpp
// ConvertLogToTable 方法：将指定的 log 文件转换为 table 文件。
// 1. 打开 log 文件，创建 log reader。
// 2. 创建 MemTable，用于存储 log 记录。
// 3. 循环读取 log 记录，将记录插入 MemTable。
// 4. 使用 MemTable 构建 table 文件。
// 5. 记录 table 文件编号，归档 log 文件。
Status Repairer::ConvertLogToTable(uint64_t log) {
  struct LogReporter : public log::Reader::Reporter { // 定义一个 LogReporter，用于处理读取 log 过程中的错误
    Env* env;
    Logger* info_log;
    uint64_t lognum;
    void Corruption(size_t bytes, const Status& s) override { // 覆盖 Corruption 方法，记录错误
      Log(info_log, "Log #%llu: dropping %d bytes; %s",
          (unsigned long long)lognum, static_cast<int>(bytes),
          s.ToString().c_str());
    }
  };

  // Open the log file
  std::string logname = LogFileName(dbname_, log); // 1. 构建文件名
  SequentialFile* lfile;
  Status status = env_->NewSequentialFile(logname, &lfile); // 打开文件
  if (!status.ok()) {
    return status;
  }

  // Create the log reader.
  LogReporter reporter;
  reporter.env = env_;
  reporter.info_log = options_.info_log;
  reporter.lognum = log;
  // We intentionally make log::Reader do checksumming so that
  // corruptions cause entire commits to be skipped instead of
  // propagating bad information (like overly large sequence
  // numbers).
  log::Reader reader(lfile, &reporter, false /*do not checksum*/,
                       0 /*initial_offset*/); // 创建 log reader

  // Read all the records and add to a memtable
  std::string scratch;
  Slice record;
  WriteBatch batch;
  MemTable* mem = new MemTable(icmp_); // 2. 创建 MemTable
  mem->Ref();
  int counter = 0;
  while (reader.ReadRecord(&record, &scratch)) { // 3. 循环读取记录
    if (record.size() < 12) {
      reporter.Corruption(record.size(),
                          Status::Corruption("log record too small"));
      continue;
    }
    WriteBatchInternal::SetContents(&batch, record);
    status = WriteBatchInternal::InsertInto(&batch, mem); // 插入 MemTable
    if (status.ok()) {
      counter += WriteBatchInternal::Count(&batch);
    } else {
      Log(options_.info_log, "Log #%llu: ignoring %s",
          (unsigned long long)log, status.ToString().c_str());
      status = Status::OK();  // Keep going with rest of file
    }
  }
  delete lfile;

  // Do not record a version edit for this conversion to a Table
  // since ExtractMetaData() will also generate edits.
  FileMetaData meta;
  meta.number = next_file_number_++;
  Iterator* iter = mem->NewIterator();
  status = BuildTable(dbname_, env_, options_, table_cache_, iter, &meta); // 4. 构建 table
  delete iter;
  mem->Unref();
  mem = nullptr;
  if (status.ok()) {
    if (meta.file_size > 0) {
      table_numbers_.push_back(meta.number);
    }
  }
  Log(options_.info_log, "Log #%llu: %d ops saved to Table #%llu %s",
      (unsigned long long)log, counter, (unsigned long long)meta.number,
      status.ToString().c_str());
  return status;
}
```

**4. `ExtractMetaData()`: Gathering Table Information**

This method scans each SSTable to extract metadata: the smallest and largest keys within the table, and the largest sequence number. This information is crucial for rebuilding the database's index.

```c++
void ExtractMetaData() {
  for (size_t i = 0; i < table_numbers_.size(); i++) {
    ScanTable(table_numbers_[i]); // 扫描 table 文件
  }
}
```

```cpp
// ExtractMetaData 方法：从 table 文件中提取元数据。
// 1. 遍历 table_numbers_ 列表，获取 table 文件编号。
// 2. 调用 ScanTable() 扫描 table 文件，提取元数据。
void Repairer::ExtractMetaData() {
  for (size_t i = 0; i < table_numbers_.size(); i++) {
    ScanTable(table_numbers_[i]); // 1. 扫描 table 文件
  }
}
```

`ScanTable()` is the workhorse of this process:

```c++
void ScanTable(uint64_t number) {
  TableInfo t;
  t.meta.number = number; // table 文件编号
  std::string fname = TableFileName(dbname_, number); // 构建 table 文件名
  Status status = env_->GetFileSize(fname, &t.meta.file_size); // 获取文件大小
  if (!status.ok()) {
    // Try alternate file name.
    fname = SSTTableFileName(dbname_, number);
    Status s2 = env_->GetFileSize(fname, &t.meta.file_size);
    if (s2.ok()) {
      status = Status::OK();
    }
  }
  if (!status.ok()) {
    ArchiveFile(TableFileName(dbname_, number));
    ArchiveFile(SSTTableFileName(dbname_, number));
    Log(options_.info_log, "Table #%llu: dropped: %s",
        (unsigned long long)t.meta.number, status.ToString().c_str());
    return;
  }

  int counter = 0;
  Iterator* iter = NewTableIterator(t.meta); // 创建 table 迭代器
  bool empty = true;
  ParsedInternalKey parsed;
  t.max_sequence = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) { // 遍历 table 文件
    Slice key = iter->key();
    if (!ParseInternalKey(key, &parsed)) {
      Log(options_.info_log, "Table #%llu: unparsable key %s",
          (unsigned long long)t.meta.number, EscapeString(key).c_str());
      continue;
    }

    counter++;
    if (empty) {
      empty = false;
      t.meta.smallest.DecodeFrom(key); // 记录最小 key
    }
    t.meta.largest.DecodeFrom(key);  // 记录最大 key
    if (parsed.sequence > t.max_sequence) {
      t.max_sequence = parsed.sequence; // 记录最大序列号
    }
  }
  if (!iter->status().ok()) {
    status = iter->status();
  }
  delete iter;
  Log(options_.info_log, "Table #%llu: %d entries %s",
      (unsigned long long)t.meta.number, counter, status.ToString().c_str());

  if (status.ok()) {
    tables_.push_back(t); // 记录 table 信息
  } else {
    RepairTable(fname, t);  // RepairTable archives input file.
  }
}
```

```cpp
// ScanTable 方法：扫描指定的 table 文件，提取元数据。
// 1. 构建 table 文件名，获取文件大小。
// 2. 创建 table 迭代器，用于遍历 table 文件。
// 3. 遍历 table 文件，提取最小 key、最大 key 和最大序列号。
// 4. 记录 table 信息，或者修复 table 文件。
void Repairer::ScanTable(uint64_t number) {
  TableInfo t;
  t.meta.number = number; // 1. 设置 table 文件编号
  std::string fname = TableFileName(dbname_, number); // 构建文件名
  Status status = env_->GetFileSize(fname, &t.meta.file_size); // 获取文件大小
  if (!status.ok()) {
    // Try alternate file name.
    fname = SSTTableFileName(dbname_, number);
    Status s2 = env_->GetFileSize(fname, &t.meta.file_size);
    if (s2.ok()) {
      status = Status::OK();
    }
  }
  if (!status.ok()) {
    ArchiveFile(TableFileName(dbname_, number));
    ArchiveFile(SSTTableFileName(dbname_, number));
    Log(options_.info_log, "Table #%llu: dropped: %s",
        (unsigned long long)t.meta.number, status.ToString().c_str());
    return;
  }

  // Extract metadata by scanning through table.
  int counter = 0;
  Iterator* iter = NewTableIterator(t.meta); // 2. 创建 table 迭代器
  bool empty = true;
  ParsedInternalKey parsed;
  t.max_sequence = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) { // 3. 遍历 table 文件
    Slice key = iter->key();
    if (!ParseInternalKey(key, &parsed)) {
      Log(options_.info_log, "Table #%llu: unparsable key %s",
          (unsigned long long)t.meta.number, EscapeString(key).c_str());
      continue;
    }

    counter++;
    if (empty) {
      empty = false;
      t.meta.smallest.DecodeFrom(key); // 记录最小 key
    }
    t.meta.largest.DecodeFrom(key);  // 记录最大 key
    if (parsed.sequence > t.max_sequence) {
      t.max_sequence = parsed.sequence; // 记录最大序列号
    }
  }
  if (!iter->status().ok()) {
    status = iter->status();
  }
  delete iter;
  Log(options_.info_log, "Table #%llu: %d entries %s",
      (unsigned long long)t.meta.number, counter, status.ToString().c_str());

  if (status.ok()) {
    tables_.push_back(t); // 4. 记录 table 信息
  } else {
    RepairTable(fname, t);  // RepairTable archives input file.
  }
}
```

If `ScanTable` encounters an error, it calls `RepairTable` to attempt to fix the SSTable.

**5. `RepairTable()`: Fixing Corrupted Tables**

This method attempts to salvage data from corrupted SSTables by copying their contents to a new table.

```c++
void RepairTable(const std::string& src, TableInfo t) {
  // Create builder.
  std::string copy = TableFileName(dbname_, next_file_number_++);
  WritableFile* file;
  Status s = env_->NewWritableFile(copy, &file); // 创建一个新的 table 文件
  if (!s.ok()) {
    return;
  }
  TableBuilder* builder = new TableBuilder(options_, file); // 创建 TableBuilder

  // Copy data.
  Iterator* iter = NewTableIterator(t.meta);
  int counter = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {  // 遍历 table 文件
    builder->Add(iter->key(), iter->value()); // 将数据添加到新的 table 文件
    counter++;
  }
  delete iter;

  ArchiveFile(src); // 归档原始 table 文件
  if (counter == 0) {
    builder->Abandon();  // Nothing to save
  } else {
    s = builder->Finish(); // 完成 table 构建
    if (s.ok()) {
      t.meta.file_size = builder->FileSize(); // 更新文件大小
    }
  }
  delete builder;
  builder = nullptr;

  if (s.ok()) {
    s = file->Close();
  }
  delete file;
  file = nullptr;

  if (counter > 0 && s.ok()) {
    std::string orig = TableFileName(dbname_, t.meta.number);
    s = env_->RenameFile(copy, orig); // 重命名新的 table 文件，覆盖原始文件
    if (s.ok()) {
      Log(options_.info_log, "Table #%llu: %d entries repaired",
          (unsigned long long)t.meta.number, counter);
      tables_.push_back(t); // 记录 table 信息
    }
  }
  if (!s.ok()) {
    env_->RemoveFile(copy); // 移除新的 table 文件
  }
}
```

```cpp
// RepairTable 方法：修复指定的 table 文件。
// 1. 创建一个新的 table 文件，用于存储修复后的数据。
// 2. 创建 TableBuilder，用于构建新的 table 文件。
// 3. 遍历原始 table 文件，将数据添加到新的 table 文件。
// 4. 归档原始 table 文件。
// 5. 重命名新的 table 文件，覆盖原始文件。
void Repairer::RepairTable(const std::string& src, TableInfo t) {
  // We will copy src contents to a new table and then rename the
  // new table over the source.

  // Create builder.
  std::string copy = TableFileName(dbname_, next_file_number_++); // 1. 构建新的文件名
  WritableFile* file;
  Status s = env_->NewWritableFile(copy, &file); // 创建新的文件
  if (!s.ok()) {
    return;
  }
  TableBuilder* builder = new TableBuilder(options_, file); // 创建 TableBuilder

  // Copy data.
  Iterator* iter = NewTableIterator(t.meta);
  int counter = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {  // 3. 遍历原始文件
    builder->Add(iter->key(), iter->value()); // 添加到新的文件
    counter++;
  }
  delete iter;

  ArchiveFile(src); // 4. 归档原始文件
  if (counter == 0) {
    builder->Abandon();  // Nothing to save
  } else {
    s = builder->Finish(); // 完成构建
    if (s.ok()) {
      t.meta.file_size = builder->FileSize(); // 更新文件大小
    }
  }
  delete builder;
  builder = nullptr;

  if (s.ok()) {
    s = file->Close();
  }
  delete file;
  file = nullptr;

  if (counter > 0 && s.ok()) {
    std::string orig = TableFileName(dbname_, t.meta.number);
    s = env_->RenameFile(copy, orig); // 5. 重命名，覆盖原始文件
    if (s.ok()) {
      Log(options_.info_log, "Table #%llu: %d entries repaired",
          (unsigned long long)t.meta.number, counter);
      tables_.push_back(t); // 记录 table 信息
    }
  }
  if (!s.ok()) {
    env_->RemoveFile(copy); // 移除新的文件
  }
}
```

**6. `WriteDescriptor()`: Creating the New Manifest**

Finally, the `WriteDescriptor` method creates a new database descriptor file based on the extracted metadata. The descriptor file describes the current state of the database, including the active SSTables.

```c++
Status WriteDescriptor() {
  std::string tmp = TempFileName(dbname_, 1);
  WritableFile* file;
  Status status = env_->NewWritableFile(tmp, &file); // 创建一个临时文件
  if (!status.ok()) {
    return status;
  }

  SequenceNumber max_sequence = 0;
  for (size_t i = 0; i < tables_.size(); i++) {
    if (max_sequence < tables_[i].max_sequence) {
      max_sequence = tables_[i].max_sequence; // 找到最大的序列号
    }
  }

  edit_.SetComparatorName(icmp_.user_comparator()->Name());
  edit_.SetLogNumber(0);
  edit_.SetNextFile(next_file_number_);
  edit_.SetLastSequence(max_sequence);

  for (size_t i = 0; i < tables_.size(); i++) {
    const TableInfo& t = tables_[i];
    edit_.AddFile(0, t.meta.number, t.meta.file_size, t.meta.smallest,
                  t.meta.largest); // 添加 table 文件信息到 VersionEdit
  }

  {
    log::Writer log(file);
    std::string record;
    edit_.EncodeTo(&record); // 将 VersionEdit 编码为字符串
    status = log.AddRecord(record); // 将字符串写入 log 文件
  }
  if (status.ok()) {
    status = file->Close();
  }
  delete file;
  file = nullptr;

  if (!status.ok()) {
    env_->RemoveFile(tmp);
  } else {
    // Discard older manifests
    for (size_t i = 0; i < manifests_.size(); i++) {
      ArchiveFile(dbname_ + "/" + manifests_[i]);
    }

    status = env_->RenameFile(tmp, DescriptorFileName(dbname_, 1)); // 重命名临时文件，覆盖原始描述符文件
    if (status.ok()) {
      status = SetCurrentFile(env_, dbname_, 1); // 设置 CURRENT 文件
    } else {
      env_->RemoveFile(tmp);
    }
  }
  return status;
}
```

```cpp
// WriteDescriptor 方法：写入新的描述符文件。
// 1. 创建一个临时文件，用于存储新的描述符文件内容。
// 2. 找到最大的序列号。
// 3. 构建 VersionEdit 对象，记录比较器名称、log 文件编号、下一个文件编号和最大的序列号。
// 4. 遍历 table 文件信息，将 table 文件信息添加到 VersionEdit 对象。
// 5. 将 VersionEdit 对象编码为字符串，写入临时文件。
// 6. 归档旧的 manifest 文件。
// 7. 重命名临时文件，覆盖原始描述符文件。
Status Repairer::WriteDescriptor() {
  std::string tmp = TempFileName(dbname_, 1); // 1. 构建临时文件名
  WritableFile* file;
  Status status = env_->NewWritableFile(tmp, &file); // 创建临时文件
  if (!status.ok()) {
    return status;
  }

  // We recovered the contents of the descriptor from the other files we find.
  // - log number is set to zero
  // - next-file-number is set to 1 + largest file number we found
  // - last-sequence-number is set to largest sequence# found across
  //   all tables (see 2c)
  // - compaction pointers are cleared
  // - every table file is added at level 0

  SequenceNumber max_sequence = 0; // 2. 找到最大的序列号
  for (size_t i = 0; i < tables_.size(); i++) {
    if (max_sequence < tables_[i].max_sequence) {
      max_sequence = tables_[i].max_sequence;
    }
  }

  edit_.SetComparatorName(icmp_.user_comparator()->Name()); // 3. 设置比较器名称
  edit_.SetLogNumber(0); // 设置 log 文件编号为 0
  edit_.SetNextFile(next_file_number_); // 设置下一个文件编号
  edit_.SetLastSequence(max_sequence); // 设置最大的序列号

  for (size_t i = 0; i < tables_.size(); i++) { // 4. 遍历 table 文件信息
    // TODO(opt): separate out into multiple levels
    const TableInfo& t = tables_[i];
    edit_.AddFile(0, t.meta.number, t.meta.file_size, t.meta.smallest,
                  t.meta.largest); // 添加 table 文件信息
  }

  // std::fprintf(stderr,
  //              "NewDescriptor:\n%s\n", edit_.DebugString().c_str());
  {
    log::Writer log(file);
    std::string record;
    edit_.EncodeTo(&record); // 5. 编码为字符串
    status = log.AddRecord(record); // 写入临时文件
  }
  if (status.ok())