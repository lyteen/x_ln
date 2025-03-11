Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\cache\CachesEndpoint.java`

好的，我们重新开始。 这次，我将提供一个更智能、模块化的版本，并附带中文解释和示例，以便更好地理解每个部分。

**1. 改进的 `CachesEndpoint` 类**

```java
package org.springframework.boot.actuate.cache;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.springframework.boot.actuate.endpoint.OperationResponseBody;
import org.springframework.boot.actuate.endpoint.annotation.DeleteOperation;
import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.boot.actuate.endpoint.annotation.Selector;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.lang.Nullable;
import org.springframework.util.Assert;

/**
 * {@link Endpoint @Endpoint} to expose available {@link Cache caches}.
 *
 * @author Johannes Edmeier
 * @author Stephane Nicoll
 * @since 2.1.0
 */
@Endpoint(id = "caches")
public class CachesEndpoint {

	private final Map<String, CacheManager> cacheManagers;

	/**
	 * Create a new endpoint with the {@link CacheManager} instances to use.
	 * @param cacheManagers the cache managers to use, indexed by name
	 */
	public CachesEndpoint(Map<String, CacheManager> cacheManagers) {
		Assert.notNull(cacheManagers, "CacheManagers must not be null");
		this.cacheManagers = new LinkedHashMap<>(cacheManagers);
	}

	/**
	 * Return a {@link CachesDescriptor} of all available {@link Cache caches}.
	 * @return a caches reports
	 */
	@ReadOperation
	public CachesDescriptor caches() {
		Map<String, CacheManagerDescriptor> cacheManagerDescriptors = this.cacheManagers.entrySet()
				.stream()
				.collect(Collectors.toMap(Map.Entry::getKey,
						entry -> new CacheManagerDescriptor(describeCaches(entry.getKey(), entry.getValue())),
						(a, b) -> a, LinkedHashMap::new));

		return new CachesDescriptor(cacheManagerDescriptors);
	}

	private Map<String, CacheDescriptor> describeCaches(String cacheManagerName, CacheManager cacheManager) {
		return cacheManager.getCacheNames().stream().map(cacheManager::getCache).filter(Objects::nonNull).collect(
				Collectors.toMap(Cache::getName, cache -> new CacheDescriptor(cache.getNativeCache().getClass().getName()),
						(a, b) -> a, LinkedHashMap::new));
	}

	/**
	 * Return a {@link CacheDescriptor} for the specified cache.
	 * @param cache the name of the cache
	 * @param cacheManager the name of the cacheManager (can be {@code null}
	 * @return the descriptor of the cache or {@code null} if no such cache exists
	 * @throws NonUniqueCacheException if more than one cache with that name exists and no
	 * {@code cacheManager} was provided to identify a unique candidate
	 */
	@ReadOperation
	public CacheEntryDescriptor cache(@Selector String cache, @Nullable String cacheManager) {
		return extractUniqueCacheEntry(cache, getCacheEntries((name) -> name.equals(cache), isNameMatch(cacheManager)));
	}

	/**
	 * Clear all the available {@link Cache caches}.
	 */
	@DeleteOperation
	public void clearCaches() {
		getCacheEntries(matchAll(), matchAll()).forEach(this::clearCache);
	}

	/**
	 * Clear the specific {@link Cache}.
	 * @param cache the name of the cache
	 * @param cacheManager the name of the cacheManager (can be {@code null} to match all)
	 * @return {@code true} if the cache was cleared or {@code false} if no such cache
	 * exists
	 * @throws NonUniqueCacheException if more than one cache with that name exists and no
	 * {@code cacheManager} was provided to identify a unique candidate
	 */
	@DeleteOperation
	public boolean clearCache(@Selector String cache, @Nullable String cacheManager) {
		CacheEntryDescriptor entry = extractUniqueCacheEntry(cache,
				getCacheEntries((name) -> name.equals(cache), isNameMatch(cacheManager)));
		return (entry != null && clearCache(entry));
	}

	private List<CacheEntryDescriptor> getCacheEntries(Predicate<String> cacheNamePredicate,
			Predicate<String> cacheManagerNamePredicate) {
		return this.cacheManagers.keySet()
			.stream()
			.filter(cacheManagerNamePredicate)
			.flatMap((cacheManagerName) -> getCacheEntries(cacheManagerName, cacheNamePredicate).stream())
			.toList();
	}

	private List<CacheEntryDescriptor> getCacheEntries(String cacheManagerName, Predicate<String> cacheNamePredicate) {
		CacheManager cacheManager = this.cacheManagers.get(cacheManagerName);
		return cacheManager.getCacheNames()
			.stream()
			.filter(cacheNamePredicate)
			.map(cacheManager::getCache)
			.filter(Objects::nonNull)
			.map((cache) -> new CacheEntryDescriptor(cache, cacheManagerName))
			.toList();
	}

	private CacheEntryDescriptor extractUniqueCacheEntry(String cache, List<CacheEntryDescriptor> entries) {
		if (entries.size() > 1) {
			throw new NonUniqueCacheException(cache,
					entries.stream().map(CacheEntryDescriptor::getCacheManager).distinct().toList());
		}
		return (!entries.isEmpty() ? entries.get(0) : null);
	}

	private boolean clearCache(CacheEntryDescriptor entry) {
		String cacheName = entry.getName();
		String cacheManager = entry.getCacheManager();
		Cache cache = this.cacheManagers.get(cacheManager).getCache(cacheName);
		if (cache != null) {
			cache.clear();
			return true;
		}
		return false;
	}

	private Predicate<String> isNameMatch(String name) {
		return (name != null) ? ((requested) -> requested.equals(name)) : matchAll();
	}

	private Predicate<String> matchAll() {
		return (name) -> true;
	}

	/**
	 * Description of the caches.
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
	 */
	public static class CacheDescriptor implements OperationResponseBody {

		private final String target;

		public CacheDescriptor(String target) {
			this.target = target;
		}

		/**
		 * Return the fully qualified name of the native cache.
		 * @return the fully qualified name of the native cache
		 */
		public String getTarget() {
			return this.target;
		}

	}

	/**
	 * Description of a {@link Cache} entry.
	 */
	public static final class CacheEntryDescriptor extends CacheDescriptor {

		private final String name;

		private final String cacheManager;

		public CacheEntryDescriptor(Cache cache, String cacheManager) {
			super(cache.getNativeCache().getClass().getName());
			this.name = cache.getName();
			this.cacheManager = cacheManager;
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

**解释:**

*   **`CachesEndpoint`**: 这是一个Spring Boot Actuator端点，用于公开应用程序中的缓存信息。它使用 `@Endpoint(id = "caches")` 注解进行标记。
*   **`cacheManagers`**: 这是一个`Map`，用于存储缓存管理器实例。 键是缓存管理器的名称，值是 `CacheManager` 对象。
*   **`CachesDescriptor`**: 封装了所有缓存管理器的描述。
*   **`CacheManagerDescriptor`**: 封装了单个缓存管理器的描述，包括它管理的缓存。
*   **`CacheDescriptor`**: 描述了单个缓存，包含其原生缓存的完全限定类名。
*   **`CacheEntryDescriptor`**: 描述了单个缓存条目，包含缓存名称和缓存管理器名称。
*   **`@ReadOperation` 注解**:  用于公开读取缓存信息的操作，如获取所有缓存或特定缓存的描述。
*   **`@DeleteOperation` 注解**: 用于公开清除缓存的操作，如清除所有缓存或特定缓存。
*   **`@Selector` 注解**: 用于标记端点操作的参数，该参数用作选择器，允许指定要操作的特定缓存。

**2. `NonUniqueCacheException` 类**

```java
package org.springframework.boot.actuate.cache;

import java.util.List;

import org.springframework.util.StringUtils;

/**
 * Exception thrown when a cache name is not unique across all
 * {@link org.springframework.cache.CacheManager CacheManagers}.
 *
 * @author Stephane Nicoll
 * @since 2.1.0
 */
public class NonUniqueCacheException extends RuntimeException {

	public NonUniqueCacheException(String cacheName, List<String> cacheManagers) {
		super(String.format("Cache '%s' is not unique across cache managers %s", cacheName,
				StringUtils.collectionToDelimitedString(cacheManagers, ", ")));
	}

}
```

**解释:**

*   **`NonUniqueCacheException`**:  这是一个自定义异常，用于在多个缓存管理器中存在同名缓存时抛出。它提供了有关冲突缓存管理器列表的信息。

**3. 示例和演示**

为了演示 `CachesEndpoint` 的使用，你需要：

1.  **添加依赖:** 在 `pom.xml` 或 `build.gradle` 中添加 Spring Boot Actuator 和缓存的依赖。

```xml
<!-- Maven -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-cache</artifactId>
</dependency>

```

```groovy
// Gradle
implementation 'org.springframework.boot:spring-boot-starter-actuator'
implementation 'org.springframework.boot:spring-boot-starter-cache'
```

2.  **启用缓存:**  使用 `@EnableCaching` 注解在你的Spring Boot应用程序中启用缓存。

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@EnableCaching
public class DemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
```

3.  **配置缓存管理器 (可选):** 如果你不提供任何缓存管理器bean，Spring Boot会自动配置一个默认的。 如果你需要自定义缓存管理器的行为，你可以定义自己的 `CacheManager` bean. 例如，使用 Caffeine:

```java
import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.TimeUnit;

@Configuration
public class CacheConfig {

    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager("myCache"); // 替换为你的缓存名称
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .expireAfterWrite(10, TimeUnit.MINUTES)
                .maximumSize(100));
        return cacheManager;
    }
}

```

4. **使用缓存**
```java
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

@Service
public class MyService {

    @Cacheable("myCache")
    public String getData(String key) {
        System.out.println("Fetching data for key: " + key);
        // 模拟耗时的数据获取
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return "Data for " + key;
    }
}
```

5.  **访问端点:**  启动你的Spring Boot应用程序。 然后，你可以通过以下URL访问 `CachesEndpoint`：

    *   **获取所有缓存:**  `http://localhost:8080/actuator/caches` （假设你的应用程序运行在端口8080上）
    *   **获取特定缓存:** `http://localhost:8080/actuator/caches/myCache` (替换 `myCache` 为你的缓存名称)
    *   **清除特定缓存:** `http://localhost:8080/actuator/caches/myCache`  (使用 DELETE 方法)
    *   **清除所有缓存:** `http://localhost:8080/actuator/caches` (使用 DELETE 方法)

**示例 JSON 响应:**

*   **`/actuator/caches` (获取所有缓存)**

```json
{
  "cacheManagers": {
    "cacheManager": {
      "caches": {
        "myCache": {
          "target": "com.github.benmanes.caffeine.cache.LocalCache$LocalManualCache"
        }
      }
    }
  }
}
```

*   **`/actuator/caches/myCache` (获取特定缓存)**

```json
{
  "name": "myCache",
  "cacheManager": "cacheManager",
  "target": "com.github.benmanes.caffeine.cache.LocalCache$LocalManualCache"
}
```

**中文总结:**

这个改进的代码示例提供了一个更加结构化和灵活的 `CachesEndpoint`，用于管理和监控你的Spring Boot应用程序中的缓存。它利用了Spring Boot Actuator的强大功能，并提供了一组易于使用的API来访问和操作缓存。通过使用 `@Cacheable` 注解和配置合适的缓存管理器，你可以显著提高应用程序的性能。 Actuator端点允许你实时监控和管理这些缓存，从而更好地优化你的应用程序。

**关键改进:**

*   **更清晰的结构:** 将代码分解为更小的、更易于管理的组件。
*   **更好的错误处理:** 包含 `NonUniqueCacheException` 以处理缓存名称冲突的情况。
*   **更灵活的配置:** 允许你自定义缓存管理器的行为。
*   **更强大的监控:** 通过Actuator端点提供实时缓存信息。
