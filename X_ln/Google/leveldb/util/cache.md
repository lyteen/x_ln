Lan: `cc` From`Google/leveldb\util\cache.cc`

好的，这次我会更清晰地解释代码，并提供更详细的中文注释和演示。

**1. `LRUHandle` 结构体 (LRUHandle Structure)**

```c++
struct LRUHandle {
  void* value; // 缓存的值的指针，可以是任何类型的数据. (Pointer to the cached value, can be any data type.)
  void (*deleter)(const Slice&, void* value); // 用于释放缓存值的函数指针. (Function pointer used to release the cached value.)
  LRUHandle* next_hash; // 哈希链表中的下一个节点指针，用于解决哈希冲突. (Pointer to the next node in the hash chain, used to resolve hash collisions.)
  LRUHandle* next; // LRU链表中的下一个节点指针. (Pointer to the next node in the LRU list.)
  LRUHandle* prev; // LRU链表中的前一个节点指针. (Pointer to the previous node in the LRU list.)
  size_t charge;  // 缓存项的开销，例如缓存值的大小. (The cost of the cache entry, such as the size of the cached value.)
  size_t key_length; // 键的长度. (The length of the key.)
  bool in_cache;     // 指示该缓存项是否在缓存中. (Indicates whether the cache entry is in the cache.)
  uint32_t refs;     // 引用计数，包括缓存本身的引用. (Reference count, including the reference from the cache itself.)
  uint32_t hash;     // 键的哈希值，用于快速查找和分片. (Hash value of the key, used for fast lookups and sharding.)
  char key_data[1];  // 键的数据，可变长度数组的起始位置. (The key data, the starting position of a variable-length array.)

  Slice key() const {
    // next is only equal to this if the LRU handle is the list head of an
    // empty list. List heads never have meaningful keys.
    assert(next != this);

    return Slice(key_data, key_length);
  }
};
```

**描述:**  `LRUHandle` 是缓存中存储的每个条目的表示。它存储键、值、哈希、引用计数等元数据。 LRU 链表使用双向链表实现，以便有效地进行最近最少使用 (LRU) 的管理。

**如何使用:**  `LRUHandle` 对象由 `LRUCache` 类创建和管理。 当一个键值对被插入缓存时，会创建一个新的 `LRUHandle` 对象。

**2. `HandleTable` 类 (HandleTable Class)**

```c++
class HandleTable {
 public:
  HandleTable() : length_(0), elems_(0), list_(nullptr) { Resize(); } //构造函数，初始化长度、元素数量和哈希表. (Constructor, initializes length, number of elements, and hash table.)
  ~HandleTable() { delete[] list_; } //析构函数，释放哈希表内存. (Destructor, releases hash table memory.)

  LRUHandle* Lookup(const Slice& key, uint32_t hash) { //查找键对应的缓存项. (Lookup the cache entry corresponding to the key.)
    return *FindPointer(key, hash);
  }

  LRUHandle* Insert(LRUHandle* h) { //插入新的缓存项，如果键已存在，则替换旧的缓存项. (Insert a new cache entry. If the key already exists, replace the old cache entry.)
    LRUHandle** ptr = FindPointer(h->key(), h->hash);
    LRUHandle* old = *ptr;
    h->next_hash = (old == nullptr ? nullptr : old->next_hash);
    *ptr = h;
    if (old == nullptr) {
      ++elems_;
      if (elems_ > length_) {
        // Since each cache entry is fairly large, we aim for a small
        // average linked list length (<= 1).
        Resize();
      }
    }
    return old;
  }

  LRUHandle* Remove(const Slice& key, uint32_t hash) { //移除键对应的缓存项. (Remove the cache entry corresponding to the key.)
    LRUHandle** ptr = FindPointer(key, hash);
    LRUHandle* result = *ptr;
    if (result != nullptr) {
      *ptr = result->next_hash;
      --elems_;
    }
    return result;
  }

 private:
  // The table consists of an array of buckets where each bucket is
  // a linked list of cache entries that hash into the bucket.
  uint32_t length_; //哈希表的长度，桶的数量. (The length of the hash table, the number of buckets.)
  uint32_t elems_; //哈希表中元素的数量. (The number of elements in the hash table.)
  LRUHandle** list_; //哈希表，一个指向LRUHandle指针数组的指针. (The hash table, a pointer to an array of LRUHandle pointers.)

  // Return a pointer to slot that points to a cache entry that
  // matches key/hash.  If there is no such cache entry, return a
  // pointer to the trailing slot in the corresponding linked list.
  LRUHandle** FindPointer(const Slice& key, uint32_t hash) { //查找键对应的缓存项的指针的指针. (Finds the pointer to the pointer to the cache entry corresponding to the key.)
    LRUHandle** ptr = &list_[hash & (length_ - 1)];
    while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
      ptr = &(*ptr)->next_hash;
    }
    return ptr;
  }

  void Resize() { //调整哈希表的大小，以保持较低的冲突率. (Resize the hash table to maintain a low collision rate.)
    uint32_t new_length = 4;
    while (new_length < elems_) {
      new_length *= 2;
    }
    LRUHandle** new_list = new LRUHandle*[new_length];
    memset(new_list, 0, sizeof(new_list[0]) * new_length);
    uint32_t count = 0;
    for (uint32_t i = 0; i < length_; i++) {
      LRUHandle* h = list_[i];
      while (h != nullptr) {
        LRUHandle* next = h->next_hash;
        uint32_t hash = h->hash;
        LRUHandle** ptr = &new_list[hash & (new_length - 1)];
        h->next_hash = *ptr;
        *ptr = h;
        h = next;
        count++;
      }
    }
    assert(elems_ == count);
    delete[] list_;
    list_ = new_list;
    length_ = new_length;
  }
};
```

**描述:**  `HandleTable` 是一个简单的哈希表，用于存储和查找 `LRUHandle` 对象。 它使用链式哈希来解决冲突。 `Resize` 方法用于动态调整哈希表的大小，以保持性能。

**如何使用:**  `HandleTable` 对象由 `LRUCache` 类使用，用于存储和检索缓存条目。它提供 `Lookup`、`Insert` 和 `Remove` 方法来操作哈希表。

**3. `LRUCache` 类 (LRUCache Class)**

```c++
class LRUCache {
 public:
  LRUCache(); //构造函数，初始化LRU链表和使用量. (Constructor, initializes LRU list and usage.)
  ~LRUCache(); //析构函数，释放LRU链表中的所有缓存项. (Destructor, releases all cache entries in the LRU list.)

  // Separate from constructor so caller can easily make an array of LRUCache
  void SetCapacity(size_t capacity) { capacity_ = capacity; } //设置缓存的容量. (Sets the capacity of the cache.)

  // Like Cache methods, but with an extra "hash" parameter.
  Cache::Handle* Insert(const Slice& key, uint32_t hash, void* value,
                        size_t charge,
                        void (*deleter)(const Slice& key, void* value)); //插入新的缓存项. (Insert a new cache entry.)
  Cache::Handle* Lookup(const Slice& key, uint32_t hash); //查找键对应的缓存项. (Lookup the cache entry corresponding to the key.)
  void Release(Cache::Handle* handle); //释放缓存句柄，减少引用计数. (Release the cache handle, decrement the reference count.)
  void Erase(const Slice& key, uint32_t hash); //移除键对应的缓存项. (Remove the cache entry corresponding to the key.)
  void Prune(); //清除所有未使用的缓存项. (Prune all unused cache entries.)
  size_t TotalCharge() const { //获取当前缓存的使用量. (Get the current usage of the cache.)
    MutexLock l(&mutex_);
    return usage_;
  }

 private:
  void LRU_Remove(LRUHandle* e); //从LRU链表中移除一个缓存项. (Remove a cache entry from the LRU list.)
  void LRU_Append(LRUHandle* list, LRUHandle* e); //将一个缓存项添加到LRU链表的末尾. (Append a cache entry to the end of the LRU list.)
  void Ref(LRUHandle* e); //增加缓存项的引用计数. (Increment the reference count of the cache entry.)
  void Unref(LRUHandle* e); //减少缓存项的引用计数，如果引用计数为0，则释放缓存项. (Decrement the reference count of the cache entry. If the reference count is 0, release the cache entry.)
  bool FinishErase(LRUHandle* e) EXCLUSIVE_LOCKS_REQUIRED(mutex_); //完成移除操作，释放缓存项. (Complete the removal operation and release the cache entry.)

  // Initialized before use.
  size_t capacity_; //缓存的容量. (The capacity of the cache.)

  // mutex_ protects the following state.
  mutable port::Mutex mutex_; //互斥锁，保护缓存的内部状态. (Mutex, protects the internal state of the cache.)
  size_t usage_ GUARDED_BY(mutex_); //当前缓存的使用量. (The current usage of the cache.)

  // Dummy head of LRU list.
  // lru.prev is newest entry, lru.next is oldest entry.
  // Entries have refs==1 and in_cache==true.
  LRUHandle lru_ GUARDED_BY(mutex_); //LRU链表的虚拟头节点. (Virtual head node of the LRU list.)

  // Dummy head of in-use list.
  // Entries are in use by clients, and have refs >= 2 and in_cache==true.
  LRUHandle in_use_ GUARDED_BY(mutex_); //正在使用的缓存项链表的虚拟头节点. (Virtual head node of the list of cache entries in use.)

  HandleTable table_ GUARDED_BY(mutex_); //用于存储缓存项的哈希表. (Hash table used to store cache entries.)
};
```

**描述:**  `LRUCache` 是 LRU 缓存的核心实现。 它使用双向链表 (`lru_` 和 `in_use_`) 和哈希表 (`table_`) 来管理缓存条目。  `Insert`、`Lookup`、`Release` 和 `Erase` 方法提供对缓存的基本操作。  `Ref` 和 `Unref` 方法管理缓存条目的引用计数。  互斥锁 (`mutex_`) 用于保护缓存的并发访问。

**如何使用:**  `LRUCache` 对象是 `ShardedLRUCache` 的构建块。  它代表缓存的一个单独的分片。

**4. `ShardedLRUCache` 类 (ShardedLRUCache Class)**

```c++
static const int kNumShardBits = 4; //分片数量的比特数. (Number of bits for the number of shards.)
static const int kNumShards = 1 << kNumShardBits; //分片的数量. (The number of shards.)

class ShardedLRUCache : public Cache {
 private:
  LRUCache shard_[kNumShards]; //LRUCache分片数组. (Array of LRUCache shards.)
  port::Mutex id_mutex_; //互斥锁，保护ID生成. (Mutex, protects ID generation.)
  uint64_t last_id_; //最后一个生成的ID. (The last generated ID.)

  static inline uint32_t HashSlice(const Slice& s) { //计算Slice的哈希值. (Calculates the hash value of the Slice.)
    return Hash(s.data(), s.size(), 0);
  }

  static uint32_t Shard(uint32_t hash) { return hash >> (32 - kNumShardBits); } //根据哈希值计算分片ID. (Calculates the shard ID based on the hash value.)

 public:
  explicit ShardedLRUCache(size_t capacity) : last_id_(0) { //构造函数，创建LRUCache分片并设置容量. (Constructor, creates LRUCache shards and sets the capacity.)
    const size_t per_shard = (capacity + (kNumShards - 1)) / kNumShards;
    for (int s = 0; s < kNumShards; s++) {
      shard_[s].SetCapacity(per_shard);
    }
  }
  ~ShardedLRUCache() override {} //析构函数. (Destructor.)
  Handle* Insert(const Slice& key, void* value, size_t charge,
                 void (*deleter)(const Slice& key, void* value)) override { //插入新的缓存项. (Insert a new cache entry.)
    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Insert(key, hash, value, charge, deleter);
  }
  Handle* Lookup(const Slice& key) override { //查找键对应的缓存项. (Lookup the cache entry corresponding to the key.)
    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Lookup(key, hash);
  }
  void Release(Handle* handle) override { //释放缓存句柄. (Release the cache handle.)
    LRUHandle* h = reinterpret_cast<LRUHandle*>(handle);
    shard_[Shard(h->hash)].Release(handle);
  }
  void Erase(const Slice& key) override { //移除键对应的缓存项. (Remove the cache entry corresponding to the key.)
    const uint32_t hash = HashSlice(key);
    shard_[Shard(hash)].Erase(key, hash);
  }
  void* Value(Handle* handle) override { //获取缓存句柄对应的值. (Get the value corresponding to the cache handle.)
    return reinterpret_cast<LRUHandle*>(handle)->value;
  }
  uint64_t NewId() override { //生成新的ID. (Generate a new ID.)
    MutexLock l(&id_mutex_);
    return ++(last_id_);
  }
  void Prune() override { //清除所有分片中的未使用缓存项. (Prune all unused cache entries in all shards.)
    for (int s = 0; s < kNumShards; s++) {
      shard_[s].Prune();
    }
  }
  size_t TotalCharge() const override { //获取所有分片的总使用量. (Get the total usage of all shards.)
    size_t total = 0;
    for (int s = 0; s < kNumShards; s++) {
      total += shard_[s].TotalCharge();
    }
    return total;
  }
};
```

**描述:**  `ShardedLRUCache` 是 `Cache` 接口的实现，它将缓存分成多个 `LRUCache` 分片。 这允许并发访问缓存的不同部分，从而提高性能。  `Shard` 方法用于根据键的哈希值确定哪个分片负责存储该键。

**如何使用:**  `ShardedLRUCache` 对象通过调用 `NewLRUCache` 函数创建。  然后，可以使用 `Insert`、`Lookup`、`Release` 和 `Erase` 方法来操作缓存。

**5. `NewLRUCache` 函数 (NewLRUCache Function)**

```c++
Cache* NewLRUCache(size_t capacity) { return new ShardedLRUCache(capacity); }
```

**描述:**  `NewLRUCache` 是一个工厂函数，它创建一个新的 `ShardedLRUCache` 对象。

**如何使用:**  这是创建 LRU 缓存对象的推荐方法。
**总结:**

此代码实现了一个分片的 LRU 缓存，可在 LevelDB 中使用。 它使用哈希表和双向链表来实现高效的缓存操作。 分片允许并发访问缓存的不同部分，从而提高性能。

希望这些解释更清晰。
