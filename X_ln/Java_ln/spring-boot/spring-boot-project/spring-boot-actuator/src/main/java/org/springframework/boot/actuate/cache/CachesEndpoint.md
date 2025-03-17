Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\cache\CachesEndpoint.java`

好的，重新开始，我会更详细地解释代码的关键部分，并附上中文注释，以及代码的使用方法和简单的示例。

**1. `CachesEndpoint` 类 (缓存端点类)**

```java
package org.springframework.boot.actuate.cache;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

import org.springframework.boot.actuate.endpoint.OperationResponseBody;
import org.springframework.boot.actuate.endpoint.annotation.DeleteOperation;
import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.boot.actuate.endpoint.annotation.Selector;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.lang.Nullable;

/**
 * {@link Endpoint @Endpoint} to expose available {@link Cache caches}.
 * 用于暴露可用 {@link Cache 缓存}的 {@link Endpoint @Endpoint}。
 *
 * @author Johannes Edmeier
 * @author Stephane Nicoll
 * @since 2.1.0
 */
@Endpoint(id = "caches") // 定义端点ID为 "caches"，用于访问该端点
public class CachesEndpoint {

	private final Map<String, CacheManager> cacheManagers; // 存储缓存管理器，Key是缓存管理器的名字，Value是CacheManager实例

	/**
	 * Create a new endpoint with the {@link CacheManager} instances to use.
	 * 创建一个新的端点，使用提供的 {@link CacheManager} 实例。
	 * @param cacheManagers the cache managers to use, indexed by name 使用的缓存管理器，以名称为索引
	 */
	public CachesEndpoint(Map<String, CacheManager> cacheManagers) {
		this.cacheManagers = new LinkedHashMap<>(cacheManagers); // 使用 LinkedHashMap 保持插入顺序
	}

	/**
	 * Return a {@link CachesDescriptor} of all available {@link Cache caches}.
	 * 返回一个包含所有可用 {@link Cache 缓存}的 {@link CachesDescriptor}。
	 * @return a caches reports 缓存报告
	 */
	@ReadOperation // 定义一个读取操作，当发送GET请求到 /actuator/caches 时会调用此方法
	public CachesDescriptor caches() {
		Map<String, Map<String, CacheDescriptor>> descriptors = new LinkedHashMap<>(); // 用于存储所有缓存的描述信息
		getCacheEntries(matchAll(), matchAll()).forEach((entry) -> { // 获取所有缓存条目，并遍历
			String cacheName = entry.getName(); // 获取缓存名称
			String cacheManager = entry.getCacheManager(); // 获取缓存管理器名称
			Map<String, CacheDescriptor> cacheManagerDescriptors = descriptors.computeIfAbsent(cacheManager,
					(key) -> new LinkedHashMap<>()); // 如果 descriptors 中不存在该缓存管理器的条目，则创建一个新的 LinkedHashMap
			cacheManagerDescriptors.put(cacheName, new CacheDescriptor(entry.getTarget())); // 将缓存名称和描述信息添加到缓存管理器的描述信息中
		});
		Map<String, CacheManagerDescriptor> cacheManagerDescriptors = new LinkedHashMap<>(); // 用于存储所有缓存管理器的描述信息
		descriptors.forEach((name, entries) -> cacheManagerDescriptors.put(name, new CacheManagerDescriptor(entries))); // 遍历 descriptors，并将缓存管理器的名称和描述信息添加到 cacheManagerDescriptors 中
		return new CachesDescriptor(cacheManagerDescriptors); // 返回所有缓存管理器的描述信息
	}

	/**
	 * Return a {@link CacheDescriptor} for the specified cache.
	 * 返回指定缓存的 {@link CacheDescriptor}。
	 * @param cache the name of the cache 缓存的名称
	 * @param cacheManager the name of the cacheManager (can be {@code null} 缓存管理器的名称 (可以为 {@code null})
	 * @return the descriptor of the cache or {@code null} if no such cache exists
	 * 缓存的描述信息，如果不存在则返回 {@code null}
	 * @throws NonUniqueCacheException if more than one cache with that name exists and no
	 * {@code cacheManager} was provided to identify a unique candidate
	 * 如果存在多个具有相同名称的缓存，并且没有提供 {@code cacheManager} 来标识唯一的候选者，则抛出 NonUniqueCacheException
	 */
	@ReadOperation // 定义一个读取操作，当发送GET请求到 /actuator/caches/{cache} 时会调用此方法
	public CacheEntryDescriptor cache(@Selector String cache, @Nullable String cacheManager) {
		return extractUniqueCacheEntry(cache, getCacheEntries((name) -> name.equals(cache), isNameMatch(cacheManager))); // 获取指定缓存的条目，并提取唯一的缓存条目
	}

	/**
	 * Clear all the available {@link Cache caches}.
	 * 清除所有可用的 {@link Cache 缓存}。
	 */
	@DeleteOperation // 定义一个删除操作，当发送DELETE请求到 /actuator/caches 时会调用此方法
	public void clearCaches() {
		getCacheEntries(matchAll(), matchAll()).forEach(this::clearCache); // 获取所有缓存条目，并清除每个缓存
	}

	/**
	 * Clear the specific {@link Cache}.
	 * 清除指定的 {@link Cache}。
	 * @param cache the name of the cache 缓存的名称
	 * @param cacheManager the name of the cacheManager (can be {@code null} to match all)
	 * 缓存管理器的名称 (可以为 {@code null} 以匹配所有)
	 * @return {@code true} if the cache was cleared or {@code false} if no such cache
	 * exists
	 * 如果缓存被清除则返回 {@code true}，如果不存在则返回 {@code false}
	 * @throws NonUniqueCacheException if more than one cache with that name exists and no
	 * {@code cacheManager} was provided to identify a unique candidate
	 * 如果存在多个具有相同名称的缓存，并且没有提供 {@code cacheManager} 来标识唯一的候选者，则抛出 NonUniqueCacheException
	 */
	@DeleteOperation // 定义一个删除操作，当发送DELETE请求到 /actuator/caches/{cache} 时会调用此方法
	public boolean clearCache(@Selector String cache, @Nullable String cacheManager) {
		CacheEntryDescriptor entry = extractUniqueCacheEntry(cache,
				getCacheEntries((name) -> name.equals(cache), isNameMatch(cacheManager))); // 获取指定缓存的条目，并提取唯一的缓存条目
		return (entry != null && clearCache(entry)); // 如果缓存条目存在，则清除缓存并返回 true，否则返回 false
	}

	private List<CacheEntryDescriptor> getCacheEntries(Predicate<String> cacheNamePredicate,
			Predicate<String> cacheManagerNamePredicate) {
		return this.cacheManagers.keySet()
			.stream()
			.filter(cacheManagerNamePredicate) // 过滤缓存管理器名称
			.flatMap((cacheManagerName) -> getCacheEntries(cacheManagerName, cacheNamePredicate).stream()) // 获取缓存管理器下的所有缓存条目
			.toList(); // 转换为 List
	}

	private List<CacheEntryDescriptor> getCacheEntries(String cacheManagerName, Predicate<String> cacheNamePredicate) {
		CacheManager cacheManager = this.cacheManagers.get(cacheManagerName); // 获取缓存管理器
		return cacheManager.getCacheNames()
			.stream()
			.filter(cacheNamePredicate) // 过滤缓存名称
			.map(cacheManager::getCache) // 获取缓存
			.filter(Objects::nonNull) // 过滤掉 null 值
			.map((cache) -> new CacheEntryDescriptor(cache, cacheManagerName)) // 创建缓存条目描述符
			.toList(); // 转换为 List
	}

	private CacheEntryDescriptor extractUniqueCacheEntry(String cache, List<CacheEntryDescriptor> entries) {
		if (entries.size() > 1) { // 如果找到多个缓存
			throw new NonUniqueCacheException(cache,
					entries.stream().map(CacheEntryDescriptor::getCacheManager).distinct().toList()); // 抛出异常，说明缓存名称不唯一
		}
		return (!entries.isEmpty() ? entries.get(0) : null); // 如果找到缓存，则返回第一个，否则返回 null
	}

	private boolean clearCache(CacheEntryDescriptor entry) {
		String cacheName = entry.getName(); // 获取缓存名称
		String cacheManager = entry.getCacheManager(); // 获取缓存管理器名称
		Cache cache = this.cacheManagers.get(cacheManager).getCache(cacheName); // 获取缓存
		if (cache != null) { // 如果缓存存在
			cache.clear(); // 清除缓存
			return true; // 返回 true
		}
		return false; // 返回 false
	}

	private Predicate<String> isNameMatch(String name) {
		return (name != null) ? ((requested) -> requested.equals(name)) : matchAll(); // 如果 name 不为 null，则返回一个 Predicate，用于判断请求的名称是否与 name 相等，否则返回 matchAll()
	}

	private Predicate<String> matchAll() {
		return (name) -> true; // 返回一个 Predicate，用于匹配所有名称
	}

	/**
	 * Description of the caches.
	 * 缓存的描述信息。
	 */
	public static final class CachesDescriptor implements OperationResponseBody {

		private final Map<String, CacheManagerDescriptor> cacheManagers;

		public CachesDescriptor(Map<String, CacheManagerDescriptor> cacheManagers) {
			this.cacheManagers = cacheManagers;
		}

		public Map<String, CacheManagerDescriptor> getCacheManagers() {
			return this.cacheManagers;
		}

	}

	/**
	 * Description of a {@link CacheManager}.
	 * {@link CacheManager} 的描述信息。
	 */
	public static final class CacheManagerDescriptor {

		private final Map<String, CacheDescriptor> caches;

		public CacheManagerDescriptor(Map<String, CacheDescriptor> caches) {
			this.caches = caches;
		}

		public Map<String, CacheDescriptor> getCaches() {
			return this.caches;
		}

	}

	/**
	 * Description of a {@link Cache}.
	 * {@link Cache} 的描述信息。
	 */
	public static class CacheDescriptor implements OperationResponseBody {

		private final String target;

		public CacheDescriptor(String target) {
			this.target = target;
		}

		/**
		 * Return the fully qualified name of the native cache.
		 * 返回原生缓存的完全限定名称。
		 * @return the fully qualified name of the native cache 原生缓存的完全限定名称
		 */
		public String getTarget() {
			return this.target;
		}

	}

	/**
	 * Description of a {@link Cache} entry.
	 * {@link Cache} 条目的描述信息。
	 */
	public static final class CacheEntryDescriptor extends CacheDescriptor {

		private final String name;

		private final String cacheManager;

		public CacheEntryDescriptor(Cache cache, String cacheManager) {
			super(cache.getNativeCache().getClass().getName()); // 调用父类的构造方法，设置 target 为原生缓存的类名
			this.name = cache.getName(); // 设置缓存名称
			this.cacheManager = cacheManager; // 设置缓存管理器名称
		}

		public String getName() {
			return this.name;
		}

		public String getCacheManager() {
			return this.cacheManager;
		}

	}

}
```

**描述:**

*   **`@Endpoint(id = "caches")`**: 这是一个Spring Boot Actuator端点，用于暴露应用程序的缓存信息。 `id` 属性定义了访问此端点的URL路径，例如 `/actuator/caches`。
*   **`cacheManagers`**:  这是一个`Map`，存储了应用程序中所有的`CacheManager`实例。 `Key`是`CacheManager`的名称，`Value`是`CacheManager`的实例。
*   **`@ReadOperation`**:  此注解标记的方法可以处理HTTP GET请求，用于读取缓存的信息。例如，`caches()`方法返回所有缓存管理器的描述信息，`cache(String cache, @Nullable String cacheManager)` 方法返回特定缓存的描述信息。
*   **`@DeleteOperation`**: 此注解标记的方法可以处理HTTP DELETE请求，用于清除缓存。例如，`clearCaches()`方法清除所有缓存，`clearCache(String cache, @Nullable String cacheManager)`方法清除特定缓存。
*   **`@Selector`**:  此注解用于从URL路径中提取参数。例如，在`cache(String cache, @Nullable String cacheManager)`和`clearCache(String cache, @Nullable String cacheManager)`方法中，`@Selector String cache`  用于提取缓存的名称。
*   **描述类 (Descriptor Classes)**:  `CachesDescriptor`, `CacheManagerDescriptor`, `CacheDescriptor`, `CacheEntryDescriptor` 这些类用于封装缓存的信息，以便于以结构化的方式返回给客户端。

**如何使用:**

1.  **添加依赖**:  确保你的Spring Boot项目中添加了Spring Boot Actuator和Cache相关的依赖。
2.  **配置缓存**:  配置Spring Boot的缓存，例如使用`@EnableCaching`注解，并配置合适的`CacheManager` (例如，`ConcurrentMapCacheManager`, `RedisCacheManager`等)。
3.  **访问端点**:  启动应用程序后，可以通过HTTP请求访问该端点来获取和管理缓存信息。

**简单的示例 (Simple Example):**

假设你有一个名为`myCacheManager`的`CacheManager`，并且在这个`CacheManager`下有一个名为`myCache`的缓存。

*   **获取所有缓存信息**:  发送GET请求到 `/actuator/caches`  会返回一个JSON，包含`myCacheManager`和`myCache`的描述信息。
*   **获取特定缓存信息**:  发送GET请求到 `/actuator/caches/myCache?cacheManager=myCacheManager`  会返回`myCache`的描述信息。
*   **清除所有缓存**:  发送DELETE请求到 `/actuator/caches`  会清除所有缓存管理器下的所有缓存。
*   **清除特定缓存**:  发送DELETE请求到 `/actuator/caches/myCache?cacheManager=myCacheManager` 会清除`myCache`缓存。

**2. 辅助类 (Helper Classes):**

*   **`NonUniqueCacheException`**:  当存在多个具有相同名称的缓存时，会抛出此异常。
*   **`CachesDescriptor`, `CacheManagerDescriptor`, `CacheDescriptor`, `CacheEntryDescriptor`**: 这些类是数据传输对象 (DTO)，用于封装缓存信息并将其返回给客户端。它们提供了清晰的结构来表示缓存层次结构和属性。

**总结 (Summary):**

`CachesEndpoint`提供了一种方便的方式来监控和管理Spring Boot应用程序的缓存。通过Actuator端点，你可以轻松地查看缓存信息，清除缓存，并诊断缓存相关的问题。
