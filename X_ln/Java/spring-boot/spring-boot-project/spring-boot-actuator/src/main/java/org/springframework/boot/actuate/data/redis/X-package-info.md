Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\data\redis\package-info.java`

好的，我们重新开始。这次我将提供更简洁的代码示例，并附上更详细的中文注释和简单的演示说明。

**1. HealthIndicator 改进版 (HealthIndicator Enhanced):**

```java
package org.springframework.boot.actuate.data.redis;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisConnectionUtils;
import org.springframework.data.redis.core.RedisTemplate;

/**
 * {@link org.springframework.boot.actuate.health.HealthIndicator} for Redis.
 *
 * @author Christian Dupuis
 * @author Stephane Nicoll
 * @since 1.1.0
 */
public class RedisHealthIndicator extends AbstractHealthIndicator {

	private final RedisConnectionFactory redisConnectionFactory;

    private final RedisTemplate<?, ?> redisTemplate;

	public RedisHealthIndicator(RedisConnectionFactory redisConnectionFactory, RedisTemplate<?, ?> redisTemplate) {
		super("Redis health check failed"); // 设置默认的失败信息 (Set the default failure message)
		this.redisConnectionFactory = redisConnectionFactory;
        this.redisTemplate = redisTemplate;
	}

	@Override
	protected void doHealthCheck(Health.Builder builder) throws Exception {
		try {
			RedisConnectionUtils.getConnection(this.redisConnectionFactory).ping();
            redisTemplate.opsForValue().set("health_check_key", "health_check_value");
            String value = (String) redisTemplate.opsForValue().get("health_check_key");
            if (!"health_check_value".equals(value)) {
                builder.down().withDetail("message", "Redis get operation failed");
                return;
            }
			builder.up(); // Redis 服务正常 (Redis service is up)
		}
		catch (Exception ex) {
			builder.down(ex); // Redis 服务异常 (Redis service is down)
		}
	}

}
```

**描述:** 这个 `RedisHealthIndicator` 检查 Redis 连接的健康状况。

*   **改进:**
    *   除了`ping()` 方法之外，增加了使用`RedisTemplate` 进行简单的set和get 操作，确认Redis 服务可用性。
    *   如果 Redis 连接出现问题，它会设置 `builder.down(ex)`，并包含异常信息，方便排查问题。

**如何使用:**  将此 `RedisHealthIndicator` 添加到 Spring Boot Actuator 的配置中，Actuator 将自动调用它来检查 Redis 的健康状况。

---

**2. RedisInfoContributor 改进版 (RedisInfoContributor Enhanced):**

```java
package org.springframework.boot.actuate.data.redis;

import java.util.Map;

import org.springframework.boot.actuate.info.Info;
import org.springframework.boot.actuate.info.InfoContributor;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.RedisConnectionUtils;

/**
 * An {@link InfoContributor} that exposes Redis information.
 *
 * @author Stephane Nicoll
 * @since 2.0.0
 */
public class RedisInfoContributor implements InfoContributor {

	private final RedisConnectionFactory redisConnectionFactory;

	public RedisInfoContributor(RedisConnectionFactory redisConnectionFactory) {
		this.redisConnectionFactory = redisConnectionFactory;
	}

	@Override
	public void contribute(Info.Builder builder) {
		try {
			RedisConnection connection = RedisConnectionUtils.getConnection(this.redisConnectionFactory);
			try {
				Map<String, String> info = connection.info();
				if (info != null && !info.isEmpty()) {
					builder.withDetail("redis", info); // 添加 Redis 信息 (Add Redis information)
				} else {
                    builder.withDetail("redis", "No info available"); //添加Redis信息不可用
                }
			}
			finally {
				RedisConnectionUtils.releaseConnection(connection, this.redisConnectionFactory);
			}
		}
		catch (Exception ex) {
			builder.withDetail("redis", "Unable to retrieve info: " + ex.getMessage()); // 无法获取 Redis 信息 (Unable to retrieve Redis information)
		}
	}

}
```

**描述:** 这个 `RedisInfoContributor` 提供关于 Redis 实例的信息。

*   **改进:**
    *  增加判断 `connection.info()` 返回是否为空，避免空指针。
    *   它捕获异常，并在无法获取 Redis 信息时提供更详细的错误消息。

**如何使用:** 将此 `RedisInfoContributor` 添加到 Spring Boot Actuator 的配置中，Actuator 将自动调用它来获取 Redis 信息并将其包含在 `/info` 端点中。

**简易演示:**

1.  **添加依赖:**  确保你的 `pom.xml` (Maven) 或 `build.gradle` (Gradle) 文件中包含 Spring Boot Actuator 和 Spring Data Redis 依赖。

2.  **配置 Redis:**  在你的 `application.properties` 或 `application.yml` 文件中配置 Redis 连接信息 (host, port, password)。

3.  **运行应用程序:**  运行你的 Spring Boot 应用程序。

4.  **访问端点:**  访问 `/actuator/health` 端点来查看 Redis 的健康状况，访问 `/actuator/info` 端点来查看 Redis 的信息。

通过以上步骤，你可以在 Spring Boot Actuator 中集成和使用这些组件来监控和管理 Redis 实例。  我希望这次的回答更符合你的要求!
