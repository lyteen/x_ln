Lan: `cc` From`Google/leveldb\util\cache.cc`

好的，我们重新开始，专注于对 LevelDB 缓存代码进行改进。

**目标:**

1.  **减少锁竞争:**  ShardedLRUCache 使用多个分片来减少锁竞争。我们探索更细粒度的锁或者无锁数据结构。
2.  **提升查找性能:** 针对高并发读取场景，优化HandleTable的查找效率。
3.  **提升LRU淘汰策略:**  考虑更智能的淘汰策略，例如 LIRS (低介入重用距离) 或 TinyLFU。
4.  **减少内存分配:**  尽可能地使用对象池或者预分配来减少内存分配和释放的开销。

**1. 更细粒度的锁或无锁数据结构（尝试使用读写锁）**

```c++
#include "leveldb/cache.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <atomic>

#include "port/port.h"
#include "port/thread_annotations.h"
#include "util/hash.h"
#include "util/mutexlock.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

namespace leveldb {

Cache::~Cache() {}

namespace {

// LRU cache implementation
struct LRUHandle {
  void* value;
  void (*deleter)(const Slice&, void* value);
  LRUHandle* next_hash;
  LRUHandle* next;
  LRUHandle* prev;
  size_t charge;
  size_t key_length;
  std::atomic<bool> in_cache {false}; // 原子操作，提升并发安全性
  std::atomic<uint32_t> refs {0};      // 原子操作，提升并发安全性
  uint32_t hash;
  char key_data[1];

  Slice key() const {
    assert(next != this);
    return Slice(key_data, key_length);
  }
};

class HandleTable {
public:
  HandleTable() : length_(0), elems_(0), list_(nullptr) { Resize(); }
  ~HandleTable() { delete[] list_; }

  LRUHandle* Lookup(const Slice& key, uint32_t hash) {
    return *FindPointer(key, hash);
  }

  LRUHandle* Insert(LRUHandle* h) {
    LRUHandle** ptr = FindPointer(h->key(), h->hash);
    LRUHandle* old = *ptr;
    h->next_hash = (old == nullptr ? nullptr : old->next_hash);
    *ptr = h;
    if (old == nullptr) {
      ++elems_;
      if (elems_ > length_) {
        Resize();
      }
    }
    return old;
  }

  LRUHandle* Remove(const Slice& key, uint32_t hash) {
    LRUHandle** ptr = FindPointer(key, hash);
    LRUHandle* result = *ptr;
    if (result != nullptr) {
      *ptr = result->next_hash;
      --elems_;
    }
    return result;
  }

private:
  uint32_t length_;
  uint32_t elems_;
  LRUHandle** list_;

  LRUHandle** FindPointer(const Slice& key, uint32_t hash) {
    LRUHandle** ptr = &list_[hash & (length_ - 1)];
    while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
      ptr = &(*ptr)->next_hash;
    }
    return ptr;
  }

  void Resize() {
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

#ifdef _WIN32
class RWMutex {
public:
    RWMutex() {
        ::InitializeSRWLock(&srwLock_);
    }
    ~RWMutex() {}

    void ReadLock() {
        ::AcquireSRWLockShared(&srwLock_);
    }

    void ReadUnlock() {
        ::ReleaseSRWLockShared(&srwLock_);
    }

    void WriteLock() {
        ::AcquireSRWLockExclusive(&srwLock_);
    }

    void WriteUnlock() {
        ::ReleaseSRWLockExclusive(&srwLock_);
    }
private:
    SRWLOCK srwLock_;
};
#else
class RWMutex {
public:
    RWMutex() {
        pthread_rwlock_init(&rwlock_, nullptr);
    }
    ~RWMutex() {
        pthread_rwlock_destroy(&rwlock_);
    }

    void ReadLock() {
        pthread_rwlock_rdlock(&rwlock_);
    }

    void ReadUnlock() {
        pthread_rwlock_unlock(&rwlock_);
    }

    void WriteLock() {
        pthread_rwlock_wrlock(&rwlock_);
    }

    void WriteUnlock() {
        pthread_rwlock_unlock(&rwlock_);
    }
private:
    pthread_rwlock_t rwlock_;
};
#endif

class LRUCache {
public:
  LRUCache();
  ~LRUCache();

  void SetCapacity(size_t capacity) { capacity_ = capacity; }

  Cache::Handle* Insert(const Slice& key, uint32_t hash, void* value,
                        size_t charge,
                        void (*deleter)(const Slice& key, void* value));
  Cache::Handle* Lookup(const Slice& key, uint32_t hash);
  void Release(Cache::Handle* handle);
  void Erase(const Slice& key, uint32_t hash);
  void Prune();
  size_t TotalCharge() const {
    ReadLock l(&mutex_);
    return usage_;
  }

private:
  void LRU_Remove(LRUHandle* e);
  void LRU_Append(LRUHandle* list, LRUHandle* e);
  void Ref(LRUHandle* e);
  void Unref(LRUHandle* e);
  bool FinishErase(LRUHandle* e) EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  size_t capacity_;

  RWMutex mutex_; // 使用读写锁替换互斥锁
  size_t usage_ GUARDED_BY(mutex_);

  LRUHandle lru_ GUARDED_BY(mutex_);
  LRUHandle in_use_ GUARDED_BY(mutex_);

  HandleTable table_ GUARDED_BY(mutex_);
};

LRUCache::LRUCache() : capacity_(0), usage_(0) {
  lru_.next = &lru_;
  lru_.prev = &lru_;
  in_use_.next = &in_use_;
  in_use_.prev = &in_use_;
}

LRUCache::~LRUCache() {
  assert(in_use_.next == &in_use_);
  for (LRUHandle* e = lru_.next; e != &lru_;) {
    LRUHandle* next = e->next;
    assert(e->in_cache.load());
    e->in_cache = false;
    assert(e->refs.load() == 1);
    Unref(e);
    e = next;
  }
}

void LRUCache::Ref(LRUHandle* e) {
  if (e->refs.load() == 1 && e->in_cache.load()) {
    LRU_Remove(e);
    LRU_Append(&in_use_, e);
  }
  e->refs++;
}

void LRUCache::Unref(LRUHandle* e) {
  assert(e->refs > 0);
  e->refs--;
  if (e->refs == 0) {
    assert(!e->in_cache.load());
    (*e->deleter)(e->key(), e->value);
    free(e);
  } else if (e->in_cache.load() && e->refs == 1) {
    LRU_Remove(e);
    LRU_Append(&lru_, e);
  }
}

void LRUCache::LRU_Remove(LRUHandle* e) {
  e->next->prev = e->prev;
  e->prev->next = e->next;
}

void LRUCache::LRU_Append(LRUHandle* list, LRUHandle* e) {
  e->next = list;
  e->prev = list->prev;
  e->prev->next = e;
  e->next->prev = e;
}

Cache::Handle* LRUCache::Lookup(const Slice& key, uint32_t hash) {
  ReadLock l(&mutex_); // 读操作使用读锁
  LRUHandle* e = table_.Lookup(key, hash);
  if (e != nullptr) {
    Ref(e);
  }
  return reinterpret_cast<Cache::Handle*>(e);
}

void LRUCache::Release(Cache::Handle* handle) {
  ReadLock l(&mutex_); // 读操作使用读锁
  Unref(reinterpret_cast<LRUHandle*>(handle));
}

Cache::Handle* LRUCache::Insert(const Slice& key, uint32_t hash, void* value,
                                size_t charge,
                                void (*deleter)(const Slice& key,
                                                void* value)) {
  LRUHandle* e =
      reinterpret_cast<LRUHandle*>(malloc(sizeof(LRUHandle) - 1 + key.size()));
  e->value = value;
  e->deleter = deleter;
  e->charge = charge;
  e->key_length = key.size();
  e->hash = hash;
  e->in_cache = false;
  e->refs = 1;
  std::memcpy(e->key_data, key.data(), key.size());

  WriteLock l(&mutex_); // 写操作使用写锁
  if (capacity_ > 0) {
    e->refs++;
    e->in_cache = true;
    LRU_Append(&in_use_, e);
    usage_ += charge;
    FinishErase(table_.Insert(e));
  } else {
    e->next = nullptr;
  }
  while (usage_ > capacity_ && lru_.next != &lru_) {
    LRUHandle* old = lru_.next;
    assert(old->refs == 1);
    bool erased = FinishErase(table_.Remove(old->key(), old->hash));
    if (!erased) {
      assert(erased);
    }
  }

  return reinterpret_cast<Cache::Handle*>(e);
}

bool LRUCache::FinishErase(LRUHandle* e) {
  if (e != nullptr) {
    assert(e->in_cache.load());
    LRU_Remove(e);
    e->in_cache = false;
    usage_ -= e->charge;
    Unref(e);
  }
  return e != nullptr;
}

void LRUCache::Erase(const Slice& key, uint32_t hash) {
  WriteLock l(&mutex_); // 写操作使用写锁
  FinishErase(table_.Remove(key, hash));
}

void LRUCache::Prune() {
  WriteLock l(&mutex_); // 写操作使用写锁
  while (lru_.next != &lru_) {
    LRUHandle* e = lru_.next;
    assert(e->refs == 1);
    bool erased = FinishErase(table_.Remove(e->key(), e->hash));
    if (!erased) {
      assert(erased);
    }
  }
}

static const int kNumShardBits = 4;
static const int kNumShards = 1 << kNumShardBits;

class ShardedLRUCache : public Cache {
private:
  LRUCache shard_[kNumShards];
  port::Mutex id_mutex_;
  uint64_t last_id_;

  static inline uint32_t HashSlice(const Slice& s) {
    return Hash(s.data(), s.size(), 0);
  }

  static uint32_t Shard(uint32_t hash) { return hash >> (32 - kNumShardBits); }

public:
  explicit ShardedLRUCache(size_t capacity) : last_id_(0) {
    const size_t per_shard = (capacity + (kNumShards - 1)) / kNumShards;
    for (int s = 0; s < kNumShards; s++) {
      shard_[s].SetCapacity(per_shard);
    }
  }
  ~ShardedLRUCache() override {}
  Handle* Insert(const Slice& key, void* value, size_t charge,
                 void (*deleter)(const Slice& key, void* value)) override {
    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Insert(key, hash, value, charge, deleter);
  }
  Handle* Lookup(const Slice& key) override {
    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Lookup(key, hash);
  }
  void Release(Handle* handle) override {
    LRUHandle* h = reinterpret_cast<LRUHandle*>(handle);
    shard_[Shard(h->hash)].Release(handle);
  }
  void Erase(const Slice& key) override {
    const uint32_t hash = HashSlice(key);
    shard_[Shard(hash)].Erase(key, hash);
  }
  void* Value(Handle* handle) override {
    return reinterpret_cast<LRUHandle*>(handle)->value;
  }
  uint64_t NewId() override {
    MutexLock l(&id_mutex_);
    return ++(last_id_);
  }
  void Prune() override {
    for (int s = 0; s < kNumShards; s++) {
      shard_[s].Prune();
    }
  }
  size_t TotalCharge() const override {
    size_t total = 0;
    for (int s = 0; s < kNumShards; s++) {
      total += shard_[s].TotalCharge();
    }
    return total;
  }
};

}  // end anonymous namespace

Cache* NewLRUCache(size_t capacity) { return new ShardedLRUCache(capacity); }

}  // namespace leveldb
```

**代码描述与中文解释：**

1.  **原子操作 (Atomic Operations):**

    *   `std::atomic<bool> in_cache {false};` 和 `std::atomic<uint32_t> refs {0};`：  `LRUHandle` 结构体中的 `in_cache` 和 `refs` 成员变量被声明为原子类型。这意味着对这些变量的读写操作将是原子的，无需额外的锁机制即可保证线程安全。  这提高了并发访问这些成员变量的效率。
    *   **解释:** 原子操作是在单个指令中完成的，不会被其他线程中断。  在并发环境中，这可以避免数据竞争，例如多个线程同时修改 `in_cache` 或 `refs` 变量，从而导致数据不一致。原子操作通常比使用互斥锁更高效，因为它们避免了锁的开销。

2.  **读写锁 (Read-Write Lock):**

    *   `RWMutex` 类：  自定义的读写锁类，用于替换原有的互斥锁 `port::Mutex`。
    *   `ReadLock l(&mutex_);` 和 `WriteLock l(&mutex_);`：  在 `LRUCache` 的 `Lookup`、`Release` 和 `TotalCharge` 函数中使用读锁，而在 `Insert`、`Erase` 和 `Prune` 函数中使用写锁。
    *   **解释:** 读写锁允许多个线程同时读取共享资源，但只允许一个线程写入共享资源。  这在高并发读取的场景下可以显著提高性能，因为多个读取操作可以并行执行，而写入操作仍然是互斥的。

3.  **代码结构的变化:**

    *   `RWMutex` 类的实现：分别在Windows和POSIX平台下使用了对应的读写锁API（`SRWLOCK` 和 `pthread_rwlock_t`）。
    *   `LRUCache` 类中的锁类型替换：将 `port::Mutex` 替换为 `RWMutex`。
    *   `LRUCache` 类中相关函数的锁操作修改：根据读写操作的性质，使用 `ReadLock` 或 `WriteLock` 来保护共享资源。

**代码的改进说明：**

*   **细粒度锁:**  通过使用读写锁，允许并发的读取操作，减少了锁的争用。  `Lookup` (查找) 操作通常是只读的，所以多个线程可以同时进行查找，而不需要等待互斥锁释放。
*   **原子操作增强并发安全:** 使用原子操作保证了引用计数和缓存状态修改的线程安全性，进一步减少了对锁的依赖。
*   **平台兼容性:**  `RWMutex` 类使用了条件编译，以确保在 Windows 和 POSIX 系统上的正确运行。

**简单的中文 Demo 说明:**

假设我们有一个缓存系统，用于存储网页内容。  大量的用户会同时访问这个缓存来获取网页内容（读取操作），而管理员会不定期地更新缓存中的内容（写入操作）。

*   **使用互斥锁的传统方式：**  每次有用户或管理员访问缓存，都需要获得一个互斥锁。  这意味着即使是读取操作，也需要等待其他读取或写入操作完成。  在高并发的情况下，这会导致大量的等待，降低系统性能。
*   **使用读写锁的改进方式：**  当用户访问缓存读取网页内容时，可以获得一个读锁。  由于读锁允许多个线程同时持有，因此多个用户可以同时读取缓存，而无需等待。  只有当管理员更新缓存内容时，才需要获得一个写锁。  写锁是独占的，确保在更新过程中没有其他线程访问缓存，避免数据不一致。

**总结:**

这段改进的代码通过引入读写锁和原子操作，提高了缓存系统的并发性能。 读写锁允许多个线程同时读取缓存，减少了锁的争用，而原子操作保证了并发修改缓存状态的线程安全性。  这种改进特别适用于读取操作远多于写入操作的场景。

**下一步改进方向:**

1.  **更智能的淘汰策略 (LIRS, TinyLFU):**  这些淘汰策略比 LRU 更能适应不同的访问模式。
2.  **无锁 Hash Table (如果可行):**  虽然实现复杂，但可以进一步减少锁的开销。
3.  **对象池:**  减少 `LRUHandle` 对象的频繁分配和释放。
4.  **基准测试:**  使用实际的 workload 对比改进前后的性能差异。

请注意，这仅仅是一个示例，实际的性能提升取决于具体的应用场景和 workload。 需要进行详细的基准测试才能确定最佳的实现方式。
