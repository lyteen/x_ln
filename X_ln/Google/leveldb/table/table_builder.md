Lan: `cc` From`Google/leveldb\table\table_builder.cc`

好的，我们重新开始。 这次我们将 LevelDB 的 `TableBuilder` 代码分解成小块，并用中文解释每个部分。

**1. `TableBuilder::Rep` 结构体**

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
        pending_index_entry(false) {
    index_block_options.block_restart_interval = 1;
  }

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

  // We do not emit the index entry for a block until we have seen the
  // first key for the next data block.  This allows us to use shorter
  // keys in the index block.
  bool pending_index_entry;
  BlockHandle pending_handle;  // Handle to add to index block

  std::string compressed_output;
};
```

**描述:** `Rep` 结构体是 `TableBuilder` 的内部实现细节的容器。 它包含了构建 Table 所需的所有状态信息。

*   `Options options`:  存储 table 的配置选项，比如 comparator, compression 类型, block 大小等等. （存储table的配置选项，如comparator，压缩类型，block大小等。）
*   `WritableFile* file`: 指向用于写入 table 内容的文件。（指向用于写入table内容的文件。）
*   `uint64_t offset`:  当前文件写入的偏移量。（当前文件写入的偏移量。）
*   `Status status`:  存储构建过程中遇到的错误状态。（存储构建过程中遇到的错误状态。）
*   `BlockBuilder data_block`:  用于构建数据块。（用于构建数据块。）
*   `BlockBuilder index_block`:  用于构建索引块。（用于构建索引块。）
*   `std::string last_key`:  上一个添加的 key.  (上一个添加的key.)
*   `int64_t num_entries`:  添加到 table 的 entry 数量.  (添加到table的entry数量.)
*   `bool closed`: 标志 table 是否已经完成或者放弃。 (标志table是否已经完成或者放弃.)
*   `FilterBlockBuilder* filter_block`: 用于构建 filter block，加速查找。 （用于构建filter block，加速查找。）
*   `bool pending_index_entry`:  标志是否有一个待处理的索引 entry。 (标志是否有一个待处理的索引entry。)
*   `BlockHandle pending_handle`:  待处理的 block 的 handle。(待处理的block的handle。)
*   `std::string compressed_output`:  用于存储压缩后的block数据。 (用于存储压缩后的block数据。)

**用途:**  `Rep` 结构体将 `TableBuilder` 的内部状态封装起来，使得 `TableBuilder` 的接口更加清晰和易于使用。

**2. `TableBuilder` 构造函数和析构函数**

```c++
TableBuilder::TableBuilder(const Options& options, WritableFile* file)
    : rep_(new Rep(options, file)) {
  if (rep_->filter_block != nullptr) {
    rep_->filter_block->StartBlock(0);
  }
}

TableBuilder::~TableBuilder() {
  assert(rep_->closed);  // Catch errors where caller forgot to call Finish()
  delete rep_->filter_block;
  delete rep_;
}
```

**描述:** 构造函数初始化 `Rep` 结构体，并启动 filter block (如果启用了 filter)。 析构函数检查 `Finish()` 是否被调用，并释放 `Rep` 结构体分配的内存。

*   构造函数 (`TableBuilder`)：
    *   创建一个 `Rep` 结构体实例，存储配置选项和文件指针。
    *   如果配置了 `filter_policy`，则启动 `filter_block` 的构建。
*   析构函数 (`~TableBuilder`)：
    *   断言 `Finish()` 或 `Abandon()` 已经被调用，确保资源被正确清理。
    *   删除 `filter_block` 和 `Rep` 结构体。

**用途:** 构造函数负责初始化构建环境，析构函数负责清理资源，保证内存安全。

**3. `TableBuilder::ChangeOptions` 函数**

```c++
Status TableBuilder::ChangeOptions(const Options& options) {
  if (options.comparator != rep_->options.comparator) {
    return Status::InvalidArgument("changing comparator while building table");
  }
  rep_->options = options;
  rep_->index_block_options = options;
  rep_->index_block_options.block_restart_interval = 1;
  return Status::OK();
}
```

**描述:**  允许在构建 table 的过程中修改某些选项。  但是，不允许修改 comparator。

*   `options.comparator != rep_->options.comparator`: 检查新的 comparator 是否与现有的 comparator 相同。如果不同，则返回一个 `InvalidArgument` 错误。
*   `rep_->options = options;`: 更新 `Rep` 结构体中的 `options`。
*   `rep_->index_block_options = options;`:  更新索引块的选项。
*    `rep_->index_block_options.block_restart_interval = 1;`: 重置索引块的restart interval。

**用途:** 允许动态调整 table 的配置，但为了保证数据一致性，禁止修改 comparator。

**4. `TableBuilder::Add` 函数**

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

  const size_t estimated_block_size = r->data_block.CurrentSizeEstimate();
  if (estimated_block_size >= r->options.block_size) {
    Flush();
  }
}
```

**描述:**  添加一个 key-value 对到 table 中。

*   `assert(!r->closed);`:  确保 table 还没有被关闭。
*   `if (!ok()) return;`: 检查之前的操作是否成功。
*   `assert(r->options.comparator->Compare(key, Slice(r->last_key)) > 0);`:  确保 key 是递增的。
*   如果存在待处理的索引 entry (`r->pending_index_entry`)：
    *   调用 `FindShortestSeparator` 找到一个分隔符，该分隔符大于等于上一个 key，小于当前 key。
    *   将上一个 block 的 handle 编码后添加到索引块。
    *   重置 `pending_index_entry` 标志。
*   如果启用了 filter，则将 key 添加到 filter block。
*   更新 `last_key` 和 `num_entries`。
*   将 key-value 对添加到数据块。
*   如果数据块的大小超过了 `block_size`，则刷新数据块。

**用途:**  `Add` 函数是构建 table 的核心函数，它将 key-value 对添加到 table 中，并负责维护索引和 filter。

**5. `TableBuilder::Flush` 函数**

```c++
void TableBuilder::Flush() {
  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->data_block.empty()) return;
  assert(!r->pending_index_entry);
  WriteBlock(&r->data_block, &r->pending_handle);
  if (ok()) {
    r->pending_index_entry = true;
    r->status = r->file->Flush();
  }
  if (r->filter_block != nullptr) {
    r->filter_block->StartBlock(r->offset);
  }
}
```

**描述:**  将当前数据块写入文件。

*   `assert(!r->closed);`:  确保 table 还没有被关闭。
*   `if (!ok()) return;`:  检查之前的操作是否成功。
*   `if (r->data_block.empty()) return;`:  如果数据块为空，则直接返回。
*   `assert(!r->pending_index_entry);`:  确保没有待处理的索引 entry。
*   调用 `WriteBlock` 将数据块写入文件，并获取 block handle。
*   设置 `pending_index_entry` 标志，表示需要添加索引 entry。
*   调用 `file->Flush()` 将数据写入磁盘。
*   如果启用了 filter，则启动一个新的 filter block。

**用途:**  `Flush` 函数负责将内存中的数据块写入文件，保证数据的持久性。

**6. `TableBuilder::WriteBlock` 函数**

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
  r->compressed_output.clear();
  block->Reset();
}
```

**描述:**  将一个 block 写入文件，并进行压缩。

*   `assert(ok());`:  确保之前的操作是否成功。
*   `Slice raw = block->Finish();`:  完成 block 的构建，获取原始数据。
*   根据配置的压缩类型 (`r->options.compression`) 进行压缩：
    *   `kNoCompression`:  不进行压缩。
    *   `kSnappyCompression`:  使用 Snappy 压缩算法。
    *   `kZstdCompression`: 使用 Zstd 压缩算法。
*   调用 `WriteRawBlock` 将压缩后的数据写入文件。
*   清空 `compressed_output` 缓冲区。
*   重置 `BlockBuilder`。

**用途:**  `WriteBlock` 函数负责将数据块写入文件，并根据配置进行压缩，以减少磁盘空间占用。

**7. `TableBuilder::WriteRawBlock` 函数**

```c++
void TableBuilder::WriteRawBlock(const Slice& block_contents,
                                 CompressionType type, BlockHandle* handle) {
  Rep* r = rep_;
  handle->set_offset(r->offset);
  handle->set_size(block_contents.size());
  r->status = r->file->Append(block_contents);
  if (r->status.ok()) {
    char trailer[kBlockTrailerSize];
    trailer[0] = type;
    uint32_t crc = crc32c::Value(block_contents.data(), block_contents.size());
    crc = crc32c::Extend(crc, trailer, 1);  // Extend crc to cover block type
    EncodeFixed32(trailer + 1, crc32c::Mask(crc));
    r->status = r->file->Append(Slice(trailer, kBlockTrailerSize));
    if (r->status.ok()) {
      r->offset += block_contents.size() + kBlockTrailerSize;
    }
  }
}
```

**描述:**  将原始 block 数据写入文件。

*   `handle->set_offset(r->offset);`: 设置 block handle 的 offset。
*   `handle->set_size(block_contents.size());`: 设置 block handle 的 size。
*   `r->status = r->file->Append(block_contents);`: 将 block 内容写入文件。
*   如果写入成功：
    *   构建一个 trailer，包含压缩类型和 CRC32 校验值。
    *   将 trailer 写入文件。
    *   更新文件 offset。

**用途:**  `WriteRawBlock` 函数负责将 block 的内容和元数据写入文件，保证数据的完整性和可验证性。

**8. `TableBuilder::Finish` 函数**

```c++
Status TableBuilder::Finish() {
  Rep* r = rep_;
  Flush();
  assert(!r->closed);
  r->closed = true;

  BlockHandle filter_block_handle, metaindex_block_handle, index_block_handle;

  // Write filter block
  if (ok() && r->filter_block != nullptr) {
    WriteRawBlock(r->filter_block->Finish(), kNoCompression,
                  &filter_block_handle);
  }

  // Write metaindex block
  if (ok()) {
    BlockBuilder meta_index_block(&r->options);
    if (r->filter_block != nullptr) {
      std::string key = "filter.";
      key.append(r->options.filter_policy->Name());
      std::string handle_encoding;
      filter_block_handle.EncodeTo(&handle_encoding);
      meta_index_block.Add(key, handle_encoding);
    }

    WriteBlock(&meta_index_block, &metaindex_block_handle);
  }

  // Write index block
  if (ok()) {
    if (r->pending_index_entry) {
      r->options.comparator->FindShortSuccessor(&r->last_key);
      std::string handle_encoding;
      r->pending_handle.EncodeTo(&handle_encoding);
      r->index_block.Add(r->last_key, Slice(handle_encoding));
      r->pending_index_entry = false;
    }
    WriteBlock(&r->index_block, &index_block_handle);
  }

  // Write footer
  if (ok()) {
    Footer footer;
    footer.set_metaindex_handle(metaindex_block_handle);
    footer.set_index_handle(index_block_handle);
    std::string footer_encoding;
    footer.EncodeTo(&footer_encoding);
    r->status = r->file->Append(footer_encoding);
    if (r->status.ok()) {
      r->offset += footer_encoding.size();
    }
  }
  return r->status;
}
```

**描述:**  完成 table 的构建，包括写入 filter block, metaindex block, index block 和 footer。

*   `Flush();`:  刷新剩余的数据块。
*   `assert(!r->closed);`:  确保 table 还没有被关闭。
*   `r->closed = true;`:  标记 table 已经关闭。
*   依次写入 filter block, metaindex block, index block 和 footer。
    *   Filter Block: 写入 Bloom Filter 数据，用于快速判断 key 是否存在于 table 中。
    *   Metaindex Block:  写入指向 filter block 和其他 meta block 的指针。
    *   Index Block: 写入指向 data block 的指针，用于快速查找 key 所在的 data block。
    *   Footer:  写入指向 metaindex block 和 index block 的指针，以及一些元数据。

**用途:**  `Finish` 函数是构建 table 的最后一步，它将所有的数据和元数据写入文件，完成 table 的构建。

**9. `TableBuilder::Abandon` 函数**

```c++
void TableBuilder::Abandon() {
  Rep* r = rep_;
  assert(!r->closed);
  r->closed = true;
}
```

**描述:**  放弃 table 的构建。

*   `assert(!r->closed);`:  确保 table 还没有被关闭。
*   `r->closed = true;`:  标记 table 已经关闭。

**用途:**  `Abandon` 函数用于在发生错误时放弃 table 的构建，释放资源。

**10. `TableBuilder::NumEntries` 和 `TableBuilder::FileSize` 函数**

```c++
uint64_t TableBuilder::NumEntries() const { return rep_->num_entries; }

uint64_t TableBuilder::FileSize() const { return rep_->offset; }
```

**描述:**  返回 table 中 entry 的数量和文件大小。

*   `NumEntries()`: 返回添加到 table 中的 key-value 对的数量。
*   `FileSize()`: 返回 table 文件的大小。

**用途:**  这两个函数用于获取 table 的统计信息。

总而言之，这段 `TableBuilder` 代码是 LevelDB 中构建 SSTable 的核心组件。 它负责将 key-value 数据写入文件，并维护索引和 filter，以提高查找效率。理解这段代码对于理解 LevelDB 的存储机制至关重要。
