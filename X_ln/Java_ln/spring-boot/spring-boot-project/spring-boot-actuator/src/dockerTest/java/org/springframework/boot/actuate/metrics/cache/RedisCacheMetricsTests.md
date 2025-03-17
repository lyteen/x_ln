Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\dockerTest\java\org\springframework\boot\actuate\metrics\cache\RedisCacheMetricsTests.java`

Okay, let's break down this Spring Boot code snippet, explaining the key parts and providing smaller code examples with descriptions in Chinese. This code is focused on testing and verifying the metrics exposed for a Redis cache in a Spring Boot application.

**Overall Purpose:** This code tests whether the cache statistics of a Redis cache (hits, misses, puts, etc.) are correctly exposed as metrics using Micrometer.  It uses Spring Boot's testing framework along with Testcontainers to run a Redis instance in a Docker container for integration testing.

**1. Setup and Dependencies:**

```java
import java.util.UUID;
import java.util.function.BiConsumer;

import com.redis.testcontainers.RedisContainer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.autoconfigure.cache.CacheAutoConfiguration;
import org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration;
import org.springframework.boot.test.context.assertj.AssertableApplicationContext;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.test.context.runner.ContextConsumer;
import org.springframework.boot.testsupport.container.TestImage;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.cache.RedisCache;
import org.springframework.data.redis.cache.RedisCacheManager;

import static org.assertj.core.api.Assertions.assertThat;
```

**描述:**  这是一系列的 `import` 语句。它们导入了各种类，这些类用于：

*   **测试:**  `org.junit.jupiter.api.Test` (JUnit 测试), `org.assertj.core.api.Assertions.assertThat` (断言).
*   **Redis:** `com.redis.testcontainers.RedisContainer` (Testcontainers Redis 集成), `org.springframework.data.redis.cache.*` (Spring Data Redis 缓存).
*   **Spring Boot:**  `org.springframework.boot.autoconfigure.*` (自动配置), `org.springframework.boot.test.context.*` (Spring Boot 测试支持).
*   **Metrics:** `io.micrometer.core.instrument.*` (Micrometer 度量).
*   **Testcontainers:** `org.testcontainers.junit.jupiter.*` (Testcontainers annotations).
*   **其他:** `java.util.*`, `java.util.function.*`.

**2. Testcontainers Setup:**

```java
@Testcontainers(disabledWithoutDocker = true)
class RedisCacheMetricsTests {

	@Container
	static final RedisContainer redis = TestImage.container(RedisContainer.class);

	private static final Tags TAGS = Tags.of("app", "test").and("cache", "test");

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(RedisAutoConfiguration.class, CacheAutoConfiguration.class))
		.withUserConfiguration(CachingConfiguration.class)
		.withPropertyValues("spring.data.redis.host=" + redis.getHost(),
				"spring.data.redis.port=" + redis.getFirstMappedPort(), "spring.cache.type=redis",
				"spring.cache.redis.enable-statistics=true");

   // ... rest of the class

}
```

**描述:**

*   `@Testcontainers(disabledWithoutDocker = true)`:  这是一个 JUnit 注解。 它启用了 Testcontainers 集成。如果 Docker 不可用，则测试将被禁用。
*   `@Container static final RedisContainer redis = TestImage.container(RedisContainer.class);`:  创建一个 Redis 容器，使用 Testcontainers 启动一个真正的 Redis 实例。`@Container` 注解指示 Testcontainers 管理此容器的生命周期。
*   `private static final Tags TAGS = Tags.of("app", "test").and("cache", "test");`: 定义了一组 Micrometer tags，用于标识指标（metrics）。
*   `private final ApplicationContextRunner contextRunner = ...`: `ApplicationContextRunner` 是一个 Spring Boot 测试实用程序，它允许你以编程方式配置和启动 Spring 应用程序上下文以进行测试。这里，它被配置为：
    *   自动配置 Redis 和缓存。
    *   加载 `CachingConfiguration` (启用缓存).
    *   设置 Redis 连接属性 (主机和端口)使用testcontainers动态获取.
    *   设置缓存类型为 Redis (`spring.cache.type=redis`).
    *   启用 Redis 缓存统计信息 (`spring.cache.redis.enable-statistics=true`). 这个配置非常重要，因为Redis Cache 默认不开启统计，我们需要开启统计才能获取metrics。

**3. Testing Cache Statistics Exposure:**

```java
	@Test
	void cacheStatisticsAreExposed() {
		this.contextRunner.run(withCacheMetrics((cache, meterRegistry) -> {
			assertThat(meterRegistry.find("cache.size").tags(TAGS).functionCounter()).isNull();
			assertThat(meterRegistry.find("cache.gets").tags(TAGS.and("result", "hit")).functionCounter()).isNotNull();
			assertThat(meterRegistry.find("cache.gets").tags(TAGS.and("result", "miss")).functionCounter()).isNotNull();
			assertThat(meterRegistry.find("cache.gets").tags(TAGS.and("result", "pending")).functionCounter())
				.isNotNull();
			assertThat(meterRegistry.find("cache.evictions").tags(TAGS).functionCounter()).isNull();
			assertThat(meterRegistry.find("cache.puts").tags(TAGS).functionCounter()).isNotNull();
			assertThat(meterRegistry.find("cache.removals").tags(TAGS).functionCounter()).isNotNull();
			assertThat(meterRegistry.find("cache.lock.duration").tags(TAGS).timeGauge()).isNotNull();
		}));
	}
```

**描述:**

*   `@Test void cacheStatisticsAreExposed() { ... }`:  这是一个 JUnit 测试方法。
*   `contextRunner.run(withCacheMetrics((cache, meterRegistry) -> { ... }));`: 启动 Spring 应用程序上下文，并执行 `withCacheMetrics` 方法提供的回调函数。`withCacheMetrics` 方法接收一个 `RedisCache` 实例和一个 `MeterRegistry` 实例。
*   `assertThat(meterRegistry.find("cache.size").tags(TAGS).functionCounter()).isNull();`:  断言 `cache.size` 指标不存在 (对于 Redis 缓存，通常不提供大小).
*   `assertThat(meterRegistry.find("cache.gets").tags(TAGS.and("result", "hit")).functionCounter()).isNotNull();`: 断言存在带有 `result=hit` 标签的 `cache.gets` 指标 (表示缓存命中).
*   `assertThat(meterRegistry.find("cache.gets").tags(TAGS.and("result", "miss")).functionCounter()).isNotNull();`: 断言存在带有 `result=miss` 标签的 `cache.gets` 指标 (表示缓存未命中).
*   `assertThat(meterRegistry.find("cache.gets").tags(TAGS.and("result", "pending")).functionCounter()).isNotNull();`: 断言存在带有 `result=pending` 标签的 `cache.gets` 指标 (表示缓存等待).
*   `assertThat(meterRegistry.find("cache.evictions").tags(TAGS).functionCounter()).isNull();`: 断言 `cache.evictions` 指标不存在 (对于 Redis 缓存，驱逐通常由 Redis 本身处理，而不是缓存管理器).
*   `assertThat(meterRegistry.find("cache.puts").tags(TAGS).functionCounter()).isNotNull();`: 断言存在 `cache.puts` 指标 (表示缓存放入).
*   `assertThat(meterRegistry.find("cache.removals").tags(TAGS).functionCounter()).isNotNull();`: 断言存在 `cache.removals` 指标 (表示缓存删除).
*  `assertThat(meterRegistry.find("cache.lock.duration").tags(TAGS).timeGauge()).isNotNull();`: 断言存在 `cache.lock.duration` 指标 (表示缓存锁获取时间).

**4. Testing Cache Hits and Misses:**

```java
	@Test
	void cacheHitsAreExposed() {
		this.contextRunner.run(withCacheMetrics((cache, meterRegistry) -> {
			String key = UUID.randomUUID().toString();
			cache.put(key, "test");

			cache.get(key);
			cache.get(key);
			assertThat(meterRegistry.get("cache.gets").tags(TAGS.and("result", "hit")).functionCounter().count())
				.isEqualTo(2.0d);
		}));
	}

	@Test
	void cacheMissesAreExposed() {
		this.contextRunner.run(withCacheMetrics((cache, meterRegistry) -> {
			String key = UUID.randomUUID().toString();
			cache.get(key);
			cache.get(key);
			cache.get(key);
			assertThat(meterRegistry.get("cache.gets").tags(TAGS.and("result", "miss")).functionCounter().count())
				.isEqualTo(3.0d);
		}));
	}
```

**描述:**

*   `@Test void cacheHitsAreExposed() { ... }`:  测试缓存命中是否正确记录。
    *   向缓存中放入一个键值对。
    *   执行两次 `cache.get(key)` (命中缓存).
    *   断言 `cache.gets` 指标 (带有 `result=hit` 标签) 的计数为 2.0.

*   `@Test void cacheMissesAreExposed() { ... }`:  测试缓存未命中是否正确记录。
    *   执行三次 `cache.get(key)` (未命中缓存，因为键不存在).
    *   断言 `cache.gets` 指标 (带有 `result=miss` 标签) 的计数为 3.0.

**5.  Verifying Metrics Against Cache Statistics:**

```java
	@Test
	void cacheMetricsMatchCacheStatistics() {
		this.contextRunner.run((context) -> {
			RedisCache cache = getTestCache(context);
			RedisCacheMetrics cacheMetrics = new RedisCacheMetrics(cache, TAGS);
			assertThat(cacheMetrics.hitCount()).isEqualTo(cache.getStatistics().getHits());
			assertThat(cacheMetrics.missCount()).isEqualTo(cache.getStatistics().getMisses());
			assertThat(cacheMetrics.putCount()).isEqualTo(cache.getStatistics().getPuts());
			assertThat(cacheMetrics.size()).isNull();
			assertThat(cacheMetrics.evictionCount()).isNull();
		});
	}
```

**描述:**

*   `@Test void cacheMetricsMatchCacheStatistics() { ... }`:  测试 Micrometer 指标是否与 Redis 缓存提供的统计信息一致。
*   `RedisCache cache = getTestCache(context);`: 获取测试用的 `RedisCache` 实例。
*   `RedisCacheMetrics cacheMetrics = new RedisCacheMetrics(cache, TAGS);`:  创建一个 `RedisCacheMetrics` 实例，该实例负责将缓存统计信息公开为 Micrometer 指标。
*   `assertThat(cacheMetrics.hitCount()).isEqualTo(cache.getStatistics().getHits());`: 断言 `RedisCacheMetrics` 报告的命中计数与 `RedisCache` 提供的统计信息中的命中计数相同。
*   `assertThat(cacheMetrics.missCount()).isEqualTo(cache.getStatistics().getMisses());`: 断言未命中计数相同。
*   `assertThat(cacheMetrics.putCount()).isEqualTo(cache.getStatistics().getPuts());`: 断言放入计数相同。
*   `assertThat(cacheMetrics.size()).isNull();`: 断言大小为空。
*   `assertThat(cacheMetrics.evictionCount()).isNull();`: 断言驱逐计数为空。

**6. Helper Methods:**

```java
	private ContextConsumer<AssertableApplicationContext> withCacheMetrics(
			BiConsumer<RedisCache, MeterRegistry> stats) {
		return (context) -> {
			RedisCache cache = getTestCache(context);
			SimpleMeterRegistry meterRegistry = new SimpleMeterRegistry();
			new RedisCacheMetrics(cache, Tags.of("app", "test")).bindTo(meterRegistry);
			stats.accept(cache, meterRegistry);
		};
	}

	private RedisCache getTestCache(AssertableApplicationContext context) {
		assertThat(context).hasSingleBean(RedisCacheManager.class);
		RedisCacheManager cacheManager = context.getBean(RedisCacheManager.class);
		RedisCache cache = (RedisCache) cacheManager.getCache("test");
		assertThat(cache).isNotNull();
		return cache;
	}
```

**描述:**

*   `private ContextConsumer<AssertableApplicationContext> withCacheMetrics(...)`: 这是一个辅助方法，用于简化测试代码。它接收一个 `BiConsumer`，该 `BiConsumer` 接收 `RedisCache` 和 `MeterRegistry` 实例。它负责：
    *   从 Spring 应用程序上下文中获取 `RedisCache` 实例。
    *   创建一个 `SimpleMeterRegistry` (一个简单的内存中的 Micrometer 注册表).
    *   将 `RedisCacheMetrics` 绑定到 `MeterRegistry` (这将公开缓存统计信息作为指标).
    *   执行提供的 `BiConsumer`，将 `RedisCache` 和 `MeterRegistry` 传递给它，允许测试代码访问缓存和指标注册表。

*   `private RedisCache getTestCache(AssertableApplicationContext context)`:  这是一个辅助方法，用于从 Spring 应用程序上下文中获取 `RedisCache` 实例。

**7. Caching Configuration:**

```java
	@Configuration(proxyBeanMethods = false)
	@EnableCaching
	static class CachingConfiguration {

	}
```

**描述:**

*   `@Configuration @EnableCaching static class CachingConfiguration { ... }`:  这是一个 Spring 配置类。
*   `@EnableCaching`:  启用 Spring 的缓存抽象。

**总结:**

这段代码展示了如何使用 Spring Boot 测试框架、Testcontainers 和 Micrometer 来测试 Redis 缓存的指标暴露。 它确保缓存统计信息已正确公开为指标，并且这些指标与缓存的实际行为一致。 重要的是，要确保 `spring.cache.redis.enable-statistics` 设置为 `true`，否则 Redis 缓存统计信息将不会被收集，并且测试将会失败。
