Lan: `cc` From`Google/leveldb\util\bloom.cc`

Okay, I will restart and provide a detailed explanation of the C++ LevelDB Bloom Filter implementation, including code snippets with Chinese explanations.

```cpp
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "leveldb/filter_policy.h" // 引入filter policy的头文件

#include "leveldb/slice.h" // 引入slice，用于存储键值对等数据
#include "util/hash.h" // 引入hash函数

namespace leveldb { // 使用leveldb命名空间

namespace { // 匿名命名空间，使得内部的定义只在本文件可见

// 计算Bloom Filter的哈希值
static uint32_t BloomHash(const Slice& key) {
  return Hash(key.data(), key.size(), 0xbc9f1d34); // 使用util/hash.h中定义的Hash函数
}

// BloomFilterPolicy类，实现了FilterPolicy接口
class BloomFilterPolicy : public FilterPolicy {
 public:
  // 构造函数，bits_per_key表示每个key使用的bit数
  explicit BloomFilterPolicy(int bits_per_key) : bits_per_key_(bits_per_key) {
    // 调整探针的数量，以减少探测成本
    k_ = static_cast<size_t>(bits_per_key * 0.69);  // 0.69 =~ ln(2)  最佳的hash function数量通常是ln(2)*m/n，其中m是bloom filter的bit数，n是key的个数
    if (k_ < 1) k_ = 1; // 保证至少有一个探针
    if (k_ > 30) k_ = 30; // 限制探针数量上限，避免过高的计算成本
  }

  // 返回过滤器的名字
  const char* Name() const override { return "leveldb.BuiltinBloomFilter2"; }

  // 创建Bloom Filter，将keys中的key加入到过滤器中
  void CreateFilter(const Slice* keys, int n, std::string* dst) const override {
    // 计算Bloom Filter需要的bit数和byte数
    size_t bits = n * bits_per_key_; // 总共的位数

    // 对于小的n，可能会导致较高的误判率，因此强制指定最小的Bloom Filter长度
    if (bits < 64) bits = 64; // 至少64位

    size_t bytes = (bits + 7) / 8; // 转换为字节数，向上取整
    bits = bytes * 8; // 保证bit数为8的整数倍

    // 分配Bloom Filter的内存空间
    const size_t init_size = dst->size();
    dst->resize(init_size + bytes, 0); // 初始化为0
    dst->push_back(static_cast<char>(k_));  // 在filter的末尾存储探针的数量，以便后续使用
    char* array = &(*dst)[init_size]; // 指向filter的起始位置

    // 将key加入到Bloom Filter中
    for (int i = 0; i < n; i++) {
      // 使用双重哈希生成一系列的哈希值
      // 参考文献：[Kirsch,Mitzenmacher 2006]
      uint32_t h = BloomHash(keys[i]); // 计算哈希值
      const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits, 为了生成不同的哈希值
      for (size_t j = 0; j < k_; j++) {
        const uint32_t bitpos = h % bits; // 计算bit位的位置
        array[bitpos / 8] |= (1 << (bitpos % 8)); // 将相应的bit位设置为1
        h += delta; // 加上delta，生成下一个哈希值
      }
    }
  }

  // 判断key是否可能在Bloom Filter中
  bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override {
    const size_t len = bloom_filter.size();
    if (len < 2) return false; // filter长度小于2，肯定不存在（至少包含1个byte和k）

    const char* array = bloom_filter.data(); // filter的数据指针
    const size_t bits = (len - 1) * 8; // bloom filter的总位数

    // 使用编码后的k，以便读取使用不同参数创建的过滤器
    const size_t k = array[len - 1]; // 读取探针的数量
    if (k > 30) {
      // 保留给未来使用，如果k > 30，则认为是匹配的
      return true;
    }

    // 检查key是否在Bloom Filter中
    uint32_t h = BloomHash(key); // 计算哈希值
    const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits， 为了生成不同的哈希值
    for (size_t j = 0; j < k; j++) {
      const uint32_t bitpos = h % bits; // 计算bit位的位置
      if ((array[bitpos / 8] & (1 << (bitpos % 8))) == 0) return false; // 如果对应的bit位为0，则肯定不存在
      h += delta; // 加上delta，生成下一个哈希值
    }
    return true; // 可能存在
  }

 private:
  size_t bits_per_key_; // 每个key使用的bit数
  size_t k_; // 探针的数量
};
}  // namespace

// 创建BloomFilterPolicy的工厂函数
const FilterPolicy* NewBloomFilterPolicy(int bits_per_key) {
  return new BloomFilterPolicy(bits_per_key); // 返回一个BloomFilterPolicy的实例
}

}  // namespace leveldb
```

**Code Breakdown and Explanations (代码分解和解释):**

1.  **`#include` directives (包含头文件):**
    ```cpp
    #include "leveldb/filter_policy.h" // 引入filter policy的头文件，定义了FilterPolicy的接口
    #include "leveldb/slice.h" // 引入slice，用于存储键值对等数据，Slice是LevelDB中用于存储字符串的类，避免了拷贝
    #include "util/hash.h" // 引入hash函数，提供哈希计算功能
    ```

    *   These lines include necessary header files for the Bloom filter implementation.  They provide definitions for `FilterPolicy`, `Slice`, and hashing functions.
    *   这些行包含了Bloom Filter实现所需的头文件。它们为`FilterPolicy`、`Slice`和哈希函数提供了定义。

2.  **`namespace leveldb` (命名空间):**
    ```cpp
    namespace leveldb {
    ...
    }  // namespace leveldb
    ```

    *   This encapsulates the Bloom filter implementation within the `leveldb` namespace to avoid naming conflicts.
    *   这将Bloom Filter的实现封装在`leveldb`命名空间中，以避免命名冲突。

3.  **Anonymous Namespace (匿名命名空间):**
    ```cpp
    namespace {
    ...
    }  // namespace
    ```

    *   The anonymous namespace makes the enclosed definitions (like `BloomHash` and `BloomFilterPolicy`) only visible within this compilation unit (i.e., this `.cc` file). This helps prevent linking errors if other files define symbols with the same names.
    *   匿名命名空间使封闭的定义（如`BloomHash`和`BloomFilterPolicy`）仅在此编译单元（即此`.cc`文件）中可见。 这有助于防止其他文件定义具有相同名称的符号时出现链接错误。

4.  **`BloomHash` function (哈希函数):**
    ```cpp
    static uint32_t BloomHash(const Slice& key) {
      return Hash(key.data(), key.size(), 0xbc9f1d34);
    }
    ```

    *   This function calculates a hash value for a given `Slice` (key). It uses the `Hash` function from `util/hash.h`, seeding it with `0xbc9f1d34`.
    *   此函数计算给定`Slice`（键）的哈希值。 它使用`util/hash.h`中的`Hash`函数，并使用`0xbc9f1d34`作为种子。

5.  **`BloomFilterPolicy` class (Bloom Filter策略类):**
    ```cpp
    class BloomFilterPolicy : public FilterPolicy {
     public:
      explicit BloomFilterPolicy(int bits_per_key) : bits_per_key_(bits_per_key) {
        // We intentionally round down to reduce probing cost a little bit
        k_ = static_cast<size_t>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
        if (k_ < 1) k_ = 1;
        if (k_ > 30) k_ = 30;
      }

      const char* Name() const override { return "leveldb.BuiltinBloomFilter2"; }

      void CreateFilter(const Slice* keys, int n, std::string* dst) const override { ... }

      bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override { ... }

     private:
      size_t bits_per_key_;
      size_t k_;
    };
    ```

    *   This class implements the `FilterPolicy` interface and provides the core logic for creating and using Bloom filters.
    *   It stores the number of bits per key (`bits_per_key_`) and the number of hash functions to use (`k_`).
    *   `k_` is calculated based on `bits_per_key` to optimize the false positive rate.  The formula `ln(2) * m / n` where `m` is the number of bits and `n` is the number of keys, provides the optimal number of hash functions,  and 0.69 is an approximation of ln(2).
    *   此类实现了`FilterPolicy`接口，并提供了创建和使用Bloom Filter的核心逻辑。
    *   它存储每个键的位数（`bits_per_key_`）和要使用的哈希函数的数量（`k_`）。
    *   `k_`基于`bits_per_key`计算，以优化误报率。 公式`ln(2) * m / n`，其中`m`是位数，`n`是键的数量，提供了最佳的哈希函数数量，0.69是ln(2)的近似值。

6.  **`BloomFilterPolicy` Constructor (构造函数):**
    ```cpp
     explicit BloomFilterPolicy(int bits_per_key) : bits_per_key_(bits_per_key) {
        // We intentionally round down to reduce probing cost a little bit
        k_ = static_cast<size_t>(bits_per_key * 0.69);  // 0.69 =~ ln(2)
        if (k_ < 1) k_ = 1;
        if (k_ > 30) k_ = 30;
      }
    ```
    *   This constructor takes the number of bits per key as input and calculates the number of hash functions to use. It limits the number of hash functions between 1 and 30 to avoid excessive computation.
    *   此构造函数将每个键的位数作为输入，并计算要使用的哈希函数的数量。 它将哈希函数的数量限制在1到30之间，以避免过多的计算。

7.  **`Name` method (名称方法):**
    ```cpp
    const char* Name() const override { return "leveldb.BuiltinBloomFilter2"; }
    ```

    *   This method returns a string identifying the filter policy.
    *   此方法返回一个字符串，用于标识过滤器策略。

8.  **`CreateFilter` method (创建过滤器方法):**
    ```cpp
    void CreateFilter(const Slice* keys, int n, std::string* dst) const override {
      // Compute bloom filter size (in both bits and bytes)
      size_t bits = n * bits_per_key_;

      // For small n, we can see a very high false positive rate.  Fix it
      // by enforcing a minimum bloom filter length.
      if (bits < 64) bits = 64;

      size_t bytes = (bits + 7) / 8;
      bits = bytes * 8;

      const size_t init_size = dst->size();
      dst->resize(init_size + bytes, 0);
      dst->push_back(static_cast<char>(k_));  // Remember # of probes in filter
      char* array = &(*dst)[init_size];
      for (int i = 0; i < n; i++) {
        // Use double-hashing to generate a sequence of hash values.
        // See analysis in [Kirsch,Mitzenmacher 2006].
        uint32_t h = BloomHash(keys[i]);
        const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
        for (size_t j = 0; j < k_; j++) {
          const uint32_t bitpos = h % bits;
          array[bitpos / 8] |= (1 << (bitpos % 8));
          h += delta;
        }
      }
    }
    ```

    *   This method creates the Bloom filter.
    *   It calculates the required number of bits and bytes for the filter.
    *   It initializes the filter with zeros and sets the bits corresponding to the hash values of each key.
    *   It uses a double-hashing technique to generate multiple hash values for each key.
    *   The number of hash functions (`k_`) determines the number of bits to set for each key.
    *   此方法创建Bloom Filter。
    *   它计算过滤器所需的位数和字节数。
    *   它使用零初始化过滤器，并设置与每个键的哈希值相对应的位。
    *   它使用双重哈希技术为每个键生成多个哈希值。
    *   哈希函数的数量（`k_`）决定了为每个键设置的位数。

9.  **`KeyMayMatch` method (键可能匹配方法):**
    ```cpp
    bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override {
      const size_t len = bloom_filter.size();
      if (len < 2) return false;

      const char* array = bloom_filter.data();
      const size_t bits = (len - 1) * 8;

      // Use the encoded k so that we can read filters generated by
      // bloom filters created using different parameters.
      const size_t k = array[len - 1];
      if (k > 30) {
        // Reserved for potentially new encodings for short bloom filters.
        // Consider it a match.
        return true;
      }

      uint32_t h = BloomHash(key);
      const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
      for (size_t j = 0; j < k; j++) {
        const uint32_t bitpos = h % bits;
        if ((array[bitpos / 8] & (1 << (bitpos % 8))) == 0) return false;
        h += delta;
      }
      return true;
    }
    ```

    *   This method checks if a given key *might* be present in the Bloom filter.  It's crucial to understand "might" because Bloom filters can have false positives.
    *   It calculates the hash values for the key and checks if the corresponding bits in the filter are set.
    *   If all the corresponding bits are set, the method returns `true` (the key *might* be present).
    *   If any of the corresponding bits are not set, the method returns `false` (the key is definitely not present).
    *   此方法检查给定的键*可能*是否存在于Bloom Filter中。 重要的是要理解“可能”，因为Bloom Filter可能存在误报。
    *   它计算键的哈希值，并检查过滤器中相应的位是否已设置。
    *   如果所有相应的位都已设置，则该方法返回`true`（该键*可能*存在）。
    *   如果任何相应的位未设置，则该方法返回`false`（该键肯定不存在）。

10. **`NewBloomFilterPolicy` function (创建Bloom Filter策略函数):**
    ```cpp
    const FilterPolicy* NewBloomFilterPolicy(int bits_per_key) {
      return new BloomFilterPolicy(bits_per_key);
    }
    ```

    *   This function is a factory function that creates a new `BloomFilterPolicy` object with the specified number of bits per key.
    *   This allows users of the LevelDB library to easily create Bloom filters with different configurations.
    *   此函数是一个工厂函数，用于创建一个新的`BloomFilterPolicy`对象，并指定每个键的位数。
    *   这允许LevelDB库的用户轻松创建具有不同配置的Bloom Filter。

**How it's used (如何使用):**

The Bloom filter policy is used by LevelDB to reduce the number of disk reads required to find a key.  When LevelDB needs to find a key, it first checks the Bloom filter. If the Bloom filter indicates that the key might be present, then LevelDB will read the corresponding data from disk. If the Bloom filter indicates that the key is definitely not present, then LevelDB can avoid reading from disk.

Bloom Filter策略被LevelDB用来减少查找键所需的磁盘读取次数。 当LevelDB需要查找一个键时，它首先检查Bloom Filter。 如果Bloom Filter指示该键可能存在，那么LevelDB将从磁盘读取相应的数据。 如果Bloom Filter指示该键肯定不存在，那么LevelDB可以避免从磁盘读取。

**Simple Demo (简单演示):**

```cpp
#include <iostream>
#include <string>
#include "leveldb/filter_policy.h"
#include "leveldb/slice.h"

int main() {
  // Create a Bloom filter policy with 10 bits per key.
  const leveldb::FilterPolicy* policy = leveldb::NewBloomFilterPolicy(10);

  // Create a set of keys.
  std::vector<leveldb::Slice> keys;
  keys.push_back(leveldb::Slice("key1"));
  keys.push_back(leveldb::Slice("key2"));
  keys.push_back(leveldb::Slice("key3"));

  // Create the Bloom filter.
  std::string bloom_filter;
  policy->CreateFilter(keys.data(), keys.size(), &bloom_filter);

  // Test if keys are present in the Bloom filter.
  leveldb::Slice test_key1("key1");
  leveldb::Slice test_key4("key4");

  if (policy->KeyMayMatch(test_key1, leveldb::Slice(bloom_filter))) {
    std::cout << "key1 可能存在" << std::endl;
  } else {
    std::cout << "key1 肯定不存在" << std::endl;
  }

  if (policy->KeyMayMatch(test_key4, leveldb::Slice(bloom_filter))) {
    std::cout << "key4 可能存在" << std::endl; // 可能会误判
  } else {
    std::cout << "key4 肯定不存在" << std::endl;
  }

  delete policy;  // Important to prevent memory leaks. 记得释放内存
  return 0;
}
```

This demo creates a Bloom filter, adds three keys to it, and then checks if two keys (one that was added and one that was not) might be present.  Remember to compile this code with the LevelDB library linked.  Also, pay attention to memory management, especially deleting the `policy` object to avoid memory leaks.
这个演示程序创建了一个Bloom Filter，向其中添加了三个键，然后检查两个键（一个已添加，一个未添加）是否可能存在。 记住在编译此代码时链接LevelDB库。 另外，请注意内存管理，尤其是删除`policy`对象以避免内存泄漏。
