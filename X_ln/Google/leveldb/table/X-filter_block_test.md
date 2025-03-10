Lan: `cc` From`Google/leveldb\table\filter_block_test.cc`

好的，我们重新开始。 这次我会更注重代码的模块化和可读性，并提供更详细的中文描述。

**1.  更灵活的哈希过滤器 (Flexible Hash Filter):**

```c++
#include "leveldb/filter_policy.h"
#include "util/coding.h"
#include "util/hash.h"

#include <cstdint>
#include <string>
#include <vector>

namespace leveldb {

class FlexibleHashFilter : public FilterPolicy {
 public:
  explicit FlexibleHashFilter(size_t hash_count = 1) : hash_count_(hash_count) {}

  const char* Name() const override { return "FlexibleHashFilter"; }

  void CreateFilter(const Slice* keys, int n, std::string* dst) const override {
    // 为每个键生成多个哈希值，增加过滤器的准确性
    for (int i = 0; i < n; ++i) {
      for (size_t j = 0; j < hash_count_; ++j) {
        uint32_t h = Hash(keys[i].data(), keys[i].size(), static_cast<uint32_t>(j)); // 使用不同的种子
        PutFixed32(dst, h);
      }
    }
  }

  bool KeyMayMatch(const Slice& key, const Slice& filter) const override {
    // 检查键的任何哈希值是否在过滤器中
    for (size_t j = 0; j < hash_count_; ++j) {
      uint32_t h = Hash(key.data(), key.size(), static_cast<uint32_t>(j));
      for (size_t i = 0; i + 4 <= filter.size(); i += 4) {
        if (h == DecodeFixed32(filter.data() + i)) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  size_t hash_count_; // 每个键的哈希值数量
};

}  // namespace leveldb
```

**描述:**

这个 `FlexibleHashFilter` 类允许你指定每个键生成多少个哈希值 (`hash_count_`)。 增加哈希值的数量可以提高过滤器的准确性（降低误判率），但也会增加过滤器的大小。  `CreateFilter` 方法现在为每个键生成 `hash_count_` 个哈希值，并将其添加到过滤器中。  `KeyMayMatch` 方法会检查键的 *任何* 哈希值是否出现在过滤器中。 使用不同的种子（`j`）来生成不同的哈希值。

**用法示例 (Usage Example):**

```c++
#include "leveldb/filter_policy.h"
#include "leveldb/options.h" // For Options

int main() {
  leveldb::FlexibleHashFilter* policy = new leveldb::FlexibleHashFilter(3); // 每个键 3 个哈希值
  leveldb::Options options;
  options.filter_policy = policy; // 设置选项
  // ... (继续使用选项创建和操作 LevelDB) ...
  delete policy; // 记得释放内存
  return 0;
}
```

**中文描述:**

这段代码定义了一个更灵活的哈希过滤器，允许用户指定为每个键生成多少个哈希值。  哈希值的数量越多，过滤器的准确性越高，但同时过滤器也会更大。 `CreateFilter` 函数现在为每个键生成指定数量的哈希值，并存储到过滤器数据中。 `KeyMayMatch` 函数检查键的任何一个哈希值是否存在于过滤器数据中，只要有一个匹配就返回 `true`。  通过使用不同的种子来生成不同的哈希值，确保了哈希值的多样性。

---

**2.  改进的过滤器块构建器 (Improved Filter Block Builder):**

```c++
#include "table/filter_block.h"

#include "leveldb/filter_policy.h"
#include "util/coding.h"
#include "util/logging.h"

#include <algorithm>
#include <vector>

namespace leveldb {

FilterBlockBuilder::FilterBlockBuilder(const FilterPolicy* policy)
    : policy_(policy), start_(nullptr), block_offset_(0), filters_(), tmp_keys_(),
      filter_offsets_() {}

void FilterBlockBuilder::StartBlock(uint64_t block_offset) {
  // 新的数据块开始，保存偏移量
  uint64_t filter_index = block_offset / kFilterBase;
  while (filter_index > filter_offsets_.size()) {
    GenerateFilter();
  }
  block_offset_ = block_offset;
}

void FilterBlockBuilder::AddKey(const Slice& key) {
  // 添加键到临时的键列表中
  tmp_keys_.push_back(key);
}

Slice FilterBlockBuilder::Finish() {
  // 完成构建过程，生成剩余的过滤器
  if (!tmp_keys_.empty()) {
    GenerateFilter();
  }

  // 添加所有过滤器的偏移量
  const uint32_t array_offset = filters_.size();
  for (size_t i = 0; i < filter_offsets_.size(); ++i) {
    PutFixed32(&filters_, filter_offsets_[i]);
  }

  // 添加偏移量数组的起始位置
  PutFixed32(&filters_, array_offset);
  filters_.push_back(kFilterBaseLg);  // 添加 base_lg

  return Slice(filters_);
}

void FilterBlockBuilder::GenerateFilter() {
  // 使用策略生成过滤器
  const size_t num_keys = tmp_keys_.size();
  if (num_keys == 0) {
    // 如果没有键，添加一个空的偏移量
    filter_offsets_.push_back(filters_.size());
    return;
  }

  filter_offsets_.push_back(filters_.size());
  policy_->CreateFilter(&tmp_keys_[0], static_cast<int>(num_keys), &filters_);

  tmp_keys_.clear(); // 清空键列表
}

FilterBlockBuilder::~FilterBlockBuilder() = default;

}  // namespace leveldb
```

**描述:**

这个 `FilterBlockBuilder` 类用于构建过滤器块。

**主要改进:**

*   **Clear Key List Immediately:** `tmp_keys_` 会在生成过滤器后立即清空。
*   **Explicit Destructor:** 添加了显式的析构函数，虽然在这里是默认的，但可以为以后的扩展提供便利。

**中文描述:**

`FilterBlockBuilder` 类负责构建 LevelDB 的过滤器块。  `StartBlock` 函数用于标记一个新的数据块的开始，并计算该数据块对应的过滤器索引。  如果需要，它会生成之前的过滤器。  `AddKey` 函数将键添加到临时的键列表中，等待生成过滤器。  `Finish` 函数完成构建过程，生成剩余的过滤器，并将所有过滤器的偏移量以及偏移量数组的起始位置添加到 `filters_` 中。  `GenerateFilter` 函数使用指定的 `FilterPolicy` 来生成过滤器，并将生成的过滤器数据追加到 `filters_` 中。  在生成过滤器之后，`tmp_keys_` 立即被清空。

---

**3.  改进的过滤器块读取器 (Improved Filter Block Reader):**

```c++
#include "table/filter_block.h"

#include "leveldb/filter_policy.h"
#include "util/coding.h"

namespace leveldb {

FilterBlockReader::FilterBlockReader(const FilterPolicy* policy, const Slice& contents)
    : policy_(policy), data_(nullptr), filter_offset_(nullptr), num_filters_(0) {
  size_t n = contents.size();
  if (n < 5) return;  // 最小长度：base_lg (1) + 偏移数组指针 (4)

  const uint32_t array_offset = DecodeFixed32(contents.data() + n - 5);
  if (array_offset > n - 5) return; // 偏移量超出范围

  const uint8_t base_lg = contents.data()[n - 1];
  if (base_lg > 30) return; // base_lg 的合理性检查

  // 初始化
  data_ = contents.data();
  filter_offset_ = data_ + array_offset;
  num_filters_ = (n - 5 - array_offset) / 4;
}

bool FilterBlockReader::KeyMayMatch(uint64_t block_offset, const Slice& key) {
  uint64_t index = block_offset >> kFilterBaseLg;
  if (index < num_filters_) {
    uint32_t filter_offset = DecodeFixed32(filter_offset_ + index * 4);
    uint32_t next_offset = DecodeFixed32(filter_offset_ + (index + 1) * 4);
    return policy_->KeyMayMatch(key, Slice(data_ + filter_offset, next_offset - filter_offset));
  }
  return true;  // 假设匹配，避免误判
}

}  // namespace leveldb
```

**描述:**

这个 `FilterBlockReader` 类用于读取和查询过滤器块。

**主要改进:**

*   **Range Checks (范围检查):** 增加了对 `array_offset` 和 `base_lg` 的范围检查，防止越界访问。
*   **Clearer Variable Names (更清晰的变量名):** 使用更具描述性的变量名，例如 `array_offset` 和 `base_lg`，提高代码可读性。
*   **Early Exit (提前退出):** 如果输入内容太短，或者偏移量超出范围，则立即返回。

**中文描述:**

`FilterBlockReader` 类用于读取 LevelDB 的过滤器块并判断键是否可能存在。 构造函数接收 `FilterPolicy` 和过滤器块的内容。  它首先进行一系列安全检查，包括检查内容长度是否足够，偏移量是否在范围内，以及 `base_lg` 的值是否合理。  然后，它初始化 `data_`、`filter_offset_` 和 `num_filters_` 成员变量。  `KeyMayMatch` 函数根据给定的块偏移量计算过滤器索引，并从过滤器块中读取相应的过滤器数据。  然后，它调用 `FilterPolicy` 的 `KeyMayMatch` 函数来判断键是否可能存在。如果索引超出范围，为了避免误判，会返回 `true`，表示假设匹配。

---

**4.  更新后的测试 (Updated Tests):**

```c++
#include "table/filter_block.h"

#include "gtest/gtest.h"
#include "leveldb/filter_policy.h"
#include "util/coding.h"
#include "util/hash.h"
#include "util/logging.h"
#include "util/testutil.h"

namespace leveldb {

// 替换为 FlexibleHashFilter
class FilterBlockTest : public testing::Test {
 public:
  FlexibleHashFilter policy_; // 使用默认的 hash_count = 1
};

TEST_F(FilterBlockTest, EmptyBuilder) {
  FilterBlockBuilder builder(&policy_);
  Slice block = builder.Finish();
  ASSERT_EQ("\\x00\\x00\\x00\\x00\\x0b", EscapeString(block));
  FilterBlockReader reader(&policy_, block);
  ASSERT_TRUE(reader.KeyMayMatch(0, "foo"));
  ASSERT_TRUE(reader.KeyMayMatch(100000, "foo"));
}

TEST_F(FilterBlockTest, SingleChunk) {
  FilterBlockBuilder builder(&policy_);
  builder.StartBlock(100);
  builder.AddKey("foo");
  builder.AddKey("bar");
  builder.AddKey("box");
  builder.StartBlock(200);
  builder.AddKey("box");
  builder.StartBlock(300);
  builder.AddKey("hello");
  Slice block = builder.Finish();
  FilterBlockReader reader(&policy_, block);
  ASSERT_TRUE(reader.KeyMayMatch(100, "foo"));
  ASSERT_TRUE(reader.KeyMayMatch(100, "bar"));
  ASSERT_TRUE(reader.KeyMayMatch(100, "box"));
  ASSERT_TRUE(reader.KeyMayMatch(100, "hello"));
  ASSERT_TRUE(reader.KeyMayMatch(100, "foo"));
  ASSERT_TRUE(!reader.KeyMayMatch(100, "missing"));
  ASSERT_TRUE(!reader.KeyMayMatch(100, "other"));
}

TEST_F(FilterBlockTest, MultiChunk) {
  FilterBlockBuilder builder(&policy_);

  // First filter
  builder.StartBlock(0);
  builder.AddKey("foo");
  builder.StartBlock(2000);
  builder.AddKey("bar");

  // Second filter
  builder.StartBlock(3100);
  builder.AddKey("box");

  // Third filter is empty

  // Last filter
  builder.StartBlock(9000);
  builder.AddKey("box");
  builder.AddKey("hello");

  Slice block = builder.Finish();
  FilterBlockReader reader(&policy_, block);

  // Check first filter
  ASSERT_TRUE(reader.KeyMayMatch(0, "foo"));
  ASSERT_TRUE(reader.KeyMayMatch(2000, "bar"));
  ASSERT_TRUE(!reader.KeyMayMatch(0, "box"));
  ASSERT_TRUE(!reader.KeyMayMatch(0, "hello"));

  // Check second filter
  ASSERT_TRUE(reader.KeyMayMatch(3100, "box"));
  ASSERT_TRUE(!reader.KeyMayMatch(3100, "foo"));
  ASSERT_TRUE(!reader.KeyMayMatch(3100, "bar"));
  ASSERT_TRUE(!reader.KeyMayMatch(3100, "hello"));

  // Check third filter (empty)
  ASSERT_TRUE(!reader.KeyMayMatch(4100, "foo"));
  ASSERT_TRUE(!reader.KeyMayMatch(4100, "bar"));
  ASSERT_TRUE(!reader.KeyMayMatch(4100, "box"));
  ASSERT_TRUE(!reader.KeyMayMatch(4100, "hello"));

  // Check last filter
  ASSERT_TRUE(reader.KeyMayMatch(9000, "box"));
  ASSERT_TRUE(reader.KeyMayMatch(9000, "hello"));
  ASSERT_TRUE(!reader.KeyMayMatch(9000, "foo"));
  ASSERT_TRUE(!reader.KeyMayMatch(9000, "bar"));
}

// 添加一个测试，使用多个哈希值
TEST_F(FilterBlockTest, MultiHash) {
  FlexibleHashFilter multi_hash_policy(3); // 每个键 3 个哈希值
  FilterBlockBuilder builder(&multi_hash_policy);
  builder.StartBlock(100);
  builder.AddKey("foo");
  Slice block = builder.Finish();
  FilterBlockReader reader(&multi_hash_policy, block);
  ASSERT_TRUE(reader.KeyMayMatch(100, "foo")); // 应该匹配
  ASSERT_TRUE(!reader.KeyMayMatch(100, "bar")); // 不应该匹配
}


}  // namespace leveldb
```

**描述:**

这些测试用例验证了 `FilterBlockBuilder` 和 `FilterBlockReader` 的功能。  `FilterBlockTest` 类现在使用 `FlexibleHashFilter` 作为默认的过滤器策略。  添加了一个新的测试用例 `MultiHash`，用于验证使用多个哈希值的过滤器是否正常工作。

**中文描述:**

这些测试用例用于验证过滤器块构建器和读取器的功能是否正确。 `FilterBlockTest` 类现在使用 `FlexibleHashFilter`，并使用默认的哈希值数量。  `EmptyBuilder` 测试用例验证了空构建器的行为。 `SingleChunk` 和 `MultiChunk` 测试用例验证了单个和多个数据块的过滤器构建和读取。  `MultiHash` 测试用例特别测试了当使用多个哈希值时，过滤器是否能够正确地匹配和排除键。

**总结:**

我试图提供更模块化、更可读、更健壮的代码。 我添加了详细的中文描述，解释了每个组件的功能和改进之处。 希望这些改进对你有所帮助！ 记住，在实际使用时，要根据你的应用需求选择合适的哈希值数量，并在性能和准确性之间进行权衡。
