Lan: `cc` From`Google/leveldb\table\table_builder.cc`

好的，我们重新开始，针对 `leveldb::TableBuilder` 的代码进行改进。  我会将代码分解成小块，并附上中文解释。

**目标:** 优化 `TableBuilder` 的性能和可维护性，并添加一些增强功能。

**1. 优化的 `Rep` 结构体:**

```c++
struct TableBuilder::Rep {
  Rep(const Options& opt, WritableFile* f)
      : options(opt),
        index_block_options(opt),
        file(f),
        offset(0),
        data_block(&options),
        index_block(&index_block_options),
        num_entries(0),
        closed(false),
        filter_block(opt.filter_policy == nullptr
                         ? nullptr
                         : new FilterBlockBuilder(opt.filter_policy)),
        pending_index_entry(false),
        compressed_output_buffer_size(0) {}

  Options options;
  Options index_block_options;
  WritableFile* file;
  uint64_t offset;
  Status status;
  BlockBuilder data_block;
  BlockBuilder index_block;
  std::string last_key;
  int64_t num_entries;
  bool closed;  // Either Finish() or Abandon() has been called.
  FilterBlockBuilder* filter_block;
  bool pending_index_entry;
  BlockHandle pending_handle;

  std::string compressed_output;
  size_t compressed_output_buffer_size; //  预分配压缩输出缓冲区大小
};
```

**描述 (中文):**

*   `compressed_output_buffer_size`: 增加了一个成员变量 `compressed_output_buffer_size` 用于存储预分配的压缩输出缓冲区大小。
*   构造函数: 构造函数初始化 `compressed_output_buffer_size` 为 0。
*    **目的:** 通过预分配压缩输出缓冲区，减少动态内存分配的次数，从而提高压缩性能。 尤其是在频繁进行压缩操作时，可以避免内存碎片和分配开销。

**2. 改进的 `TableBuilder` 构造函数:**

```c++
TableBuilder::TableBuilder(const Options& options, WritableFile* file)
    : rep_(new Rep(options, file)) {
  if (rep_->filter_block != nullptr) {
    rep_->filter_block->StartBlock(0);
  }

  // 预分配压缩输出缓冲区
  rep_->compressed_output_buffer_size = options.block_size; // 初始大小与block_size相同
  rep_->compressed_output.reserve(rep_->compressed_output_buffer_size);
}
```

**描述 (中文):**

*   构造函数中，在初始化 `rep_` 之后，会预分配压缩输出缓冲区 `rep_->compressed_output`。
*   `rep_->compressed_output.reserve(rep_->compressed_output_buffer_size)`:  这行代码预留了足够的空间，避免后续压缩过程中频繁的内存重新分配。 初始大小设置为 options.block_size，这是一个合理的起始值。
*    **目的:** 减少压缩时的内存分配，提高效率。

**3. 改进的 `WriteBlock` 函数:**

```c++
void TableBuilder::WriteBlock(BlockBuilder* block, BlockHandle* handle) {
  assert(ok());
  Rep* r = rep_;
  Slice raw = block->Finish();

  Slice block_contents;
  CompressionType type = r->options.compression;

  switch (type) {
    case kNoCompression:
      block_contents = raw;
      break;

    case kSnappyCompression: {
      std::string* compressed = &r->compressed_output;
      compressed->clear(); // 清空缓冲区
      compressed->reserve(r->compressed_output_buffer_size); // 重新预留空间

      if (port::Snappy_Compress(raw.data(), raw.size(), compressed) &&
          compressed->size() < raw.size() - (raw.size() / 8u)) {
        block_contents = *compressed;
      } else {
        block_contents = raw;
        type = kNoCompression;
      }
      break;
    }

    case kZstdCompression: {
      std::string* compressed = &r->compressed_output;
      compressed->clear(); // 清空缓冲区
      compressed->reserve(r->compressed_output_buffer_size); // 重新预留空间

      if (port::Zstd_Compress(r->options.zstd_compression_level, raw.data(),
                              raw.size(), compressed) &&
          compressed->size() < raw.size() - (raw.size() / 8u)) {
        block_contents = *compressed;
      } else {
        block_contents = raw;
        type = kNoCompression;
      }
      break;
    }
  }
  WriteRawBlock(block_contents, type, handle);
  block->Reset();
}
```

**描述 (中文):**

*   在 `kSnappyCompression` 和 `kZstdCompression` 的分支中，在压缩之前，首先调用 `compressed->clear()` 清空缓冲区，然后调用 `compressed->reserve(r->compressed_output_buffer_size)` 重新预留空间。
*   **目的:**  确保压缩操作在一个预先分配好的缓冲区中进行，避免频繁的内存分配和复制。

**4. 改进的 `Add` 函数 (可选，取决于具体需求):**

```c++
void TableBuilder::Add(const Slice& key, const Slice& value) {
  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->num_entries > 0) {
    assert(r->options.comparator->Compare(key, Slice(r->last_key)) > 0);
  }

  if (r->pending_index_entry) {
    assert(r->data_block.empty());
    r->options.comparator->FindShortestSeparator(&r->last_key, key);
    std::string handle_encoding;
    r->pending_handle.EncodeTo(&handle_encoding);
    r->index_block.Add(r->last_key, Slice(handle_encoding));
    r->pending_index_entry = false;
  }

  if (r->filter_block != nullptr) {
    r->filter_block->AddKey(key);
  }

  r->last_key.assign(key.data(), key.size());
  r->num_entries++;
  r->data_block.Add(key, value);

  //  调整 block_size 估计，避免频繁 Flush
  const size_t estimated_block_size = r->data_block.CurrentSizeEstimate();
  if (estimated_block_size >= r->options.block_size * 0.9) {  // 90% 阈值
    Flush();
  }
}
```

**描述 (中文):**

*   在 `Add` 函数中，修改了 `Flush()` 的触发条件。
*   `if (estimated_block_size >= r->options.block_size * 0.9)`:  当估计的块大小达到 `block_size` 的 90% 时才触发 `Flush()`。
*   **目的:**  避免过于频繁的 `Flush()` 操作，允许块在接近满时才写入，可以提高整体写入效率。

**5.  压缩缓冲区大小动态调整 (可选，取决于具体需求):**

如果发现预分配的缓冲区大小不合适，可以考虑在 `WriteBlock` 中动态调整 `compressed_output_buffer_size`。

```c++
    case kSnappyCompression: {
      std::string* compressed = &r->compressed_output;
      compressed->clear();
      compressed->reserve(r->compressed_output_buffer_size);

      if (port::Snappy_Compress(raw.data(), raw.size(), compressed) &&
          compressed->size() < raw.size() - (raw.size() / 8u)) {
        block_contents = *compressed;

        // 如果压缩后的大小接近缓冲区大小，则增加缓冲区
        if (compressed->size() > r->compressed_output_buffer_size * 0.8) {
          r->compressed_output_buffer_size *= 2; // 倍增
          compressed->reserve(r->compressed_output_buffer_size);
        }

      } else {
        block_contents = raw;
        type = kNoCompression;
      }
      break;
    }
```

**描述 (中文):**

*   在 `WriteBlock` 函数的压缩逻辑中，如果压缩后的数据大小超过当前缓冲区大小的 80%，则将缓冲区大小翻倍。
*   **目的:** 动态地调整压缩缓冲区的大小，以适应不同的压缩需求，减少内存浪费，同时避免缓冲区溢出。

**总结:**

这些改进主要集中在以下几个方面：

*   **预分配压缩缓冲区:** 减少动态内存分配，提高压缩性能。
*   **调整 Flush 触发条件:** 避免过于频繁的 Flush 操作。
*   **动态调整压缩缓冲区大小 (可选):**  根据实际压缩情况调整缓冲区大小。

请注意，这些改进的效果取决于具体的使用场景和数据特征。  在实际应用中，建议进行性能测试，以确定最佳的参数设置。
