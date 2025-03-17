Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\data\redis\RedisReactiveHealthIndicator.java`

Okay, let's try this again, focusing on providing a "smarter" version of the code with more robust error handling, better connection management, and more detailed health information. I'll also break down the code into smaller, explained chunks and provide a Chinese translation for the descriptions.

**Improvements:**

*   **Robust Error Handling:** Handles connection errors and Redis command errors more gracefully.
*   **Connection Pooling (Optional):**  If you are *not* using Spring Data Redis's connection pooling, adds explicit closing to prevent resource leaks.  This is less relevant with modern connection pooling.
*   **Detailed Health Information:**  Includes Redis version and role (master/slave) in the health details.
*   **Timeout:** Adds a timeout to the health check operation to prevent indefinite blocking.
*   **More Comments:**  Added more comments for better understanding.

```java
/*
 * Copyright 2012-2023 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.boot.actuate.data.redis;

import java.time.Duration;
import java.util.Properties;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import org.springframework.boot.actuate.health.AbstractReactiveHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.ReactiveHealthIndicator;
import org.springframework.data.redis.RedisConnectionFailureException;
import org.springframework.data.redis.connection.ClusterInfo;
import org.springframework.data.redis.connection.ReactiveRedisClusterConnection;
import org.springframework.data.redis.connection.ReactiveRedisConnection;
import org.springframework.data.redis.connection.ReactiveRedisConnectionFactory;
import org.springframework.util.StringUtils;

/**
 * A {@link ReactiveHealthIndicator} for Redis.
 *
 * @author Stephane Nicoll
 * @author Mark Paluch
 * @author Artsiom Yudovin
 * @author Scott Frederick
 * @since 2.0.0
 */
public class RedisReactiveHealthIndicator extends AbstractReactiveHealthIndicator {

	private static final Log logger = LogFactory.getLog(RedisReactiveHealthIndicator.class);

	private final ReactiveRedisConnectionFactory connectionFactory;

	private final Duration timeout = Duration.ofSeconds(5); // Health check timeout

	public RedisReactiveHealthIndicator(ReactiveRedisConnectionFactory connectionFactory) {
		super("Redis health check failed");  // Default error message
		this.connectionFactory = connectionFactory;
	}

	@Override
	protected Mono<Health> doHealthCheck(Health.Builder builder) {
		return getConnection()
				.flatMap(connection -> doHealthCheck(builder, connection))
				.timeout(this.timeout) // Add a timeout
				.onErrorResume(ex -> {
					logger.warn("Redis health check failed", ex);
					return Mono.just(builder.down(ex).build()); // Handle connection errors and other exceptions.
				});
	}

	private Mono<ReactiveRedisConnection> getConnection() {
		return Mono.fromSupplier(this.connectionFactory::getReactiveConnection)
				.subscribeOn(Schedulers.boundedElastic()) // Use a dedicated scheduler
				.onErrorResume(ex -> {
					logger.error("Failed to obtain Redis connection", ex);
					return Mono.error(new RedisConnectionFailureException("Failed to obtain Redis connection", ex));
				});
	}

	private Mono<Health> doHealthCheck(Health.Builder builder, ReactiveRedisConnection connection) {
		return getHealth(builder, connection)
				.onErrorResume(ex -> {
					logger.warn("Redis command execution failed", ex);
					return Mono.just(builder.down(ex).build()); // Handle Redis command errors
				})
				.flatMap(health -> connection.closeLater().thenReturn(health));
	}

	private Mono<Health> getHealth(Health.Builder builder, ReactiveRedisConnection connection) {
		if (connection instanceof ReactiveRedisClusterConnection clusterConnection) {
			return clusterConnection.clusterGetClusterInfo().map(info -> fromClusterInfo(builder, info));
		}
		return connection.serverCommands().info("server")
				.map(info -> up(builder, info));
	}

	private Health up(Health.Builder builder, Properties info) {
		RedisHealth.up(builder, info);  // Use existing logic

		// Add Redis version and role information
		String redisVersion = info.getProperty("redis_version");
		if (StringUtils.hasText(redisVersion)) {
			builder.withDetail("version", redisVersion);
		}

		String role = info.getProperty("role"); // Available since Redis 2.8.12
		if (StringUtils.hasText(role)) {
			builder.withDetail("role", role);  // e.g., master, slave
		}

		return builder.build();
	}

	private Health fromClusterInfo(Health.Builder builder, ClusterInfo clusterInfo) {
		return RedisHealth.fromClusterInfo(builder, clusterInfo).build(); // Use existing logic
	}

}
```

**Code Breakdown & Chinese Translation:**

1.  **Imports (导入):** Standard imports for reactive programming, Redis, and health indicators.

    ```java
    import java.time.Duration;
    import java.util.Properties;

    import org.apache.commons.logging.Log;
    import org.apache.commons.logging.LogFactory;
    import reactor.core.publisher.Mono;
    import reactor.core.scheduler.Schedulers;

    import org.springframework.boot.actuate.health.AbstractReactiveHealthIndicator;
    import org.springframework.boot.actuate.health.Health;
    import org.springframework.boot.actuate.health.ReactiveHealthIndicator;
    import org.springframework.data.redis.RedisConnectionFailureException;
    import org.springframework.data.redis.connection.ClusterInfo;
    import org.springframework.data.redis.connection.ReactiveRedisClusterConnection;
    import org.springframework.data.redis.connection.ReactiveRedisConnection;
    import org.springframework.data.redis.connection.ReactiveRedisConnectionFactory;
    import org.springframework.util.StringUtils;
    ```

    *   *中文: 导入必要的类，包括反应式编程的类、Redis相关的类和健康指示器相关的类。*

2.  **Class Definition (类定义):** Defines the `RedisReactiveHealthIndicator` class, extending `AbstractReactiveHealthIndicator`.

    ```java
    public class RedisReactiveHealthIndicator extends AbstractReactiveHealthIndicator {
    ```

    *   *中文: 定义 `RedisReactiveHealthIndicator` 类，它继承自 `AbstractReactiveHealthIndicator`。*

3.  **Logger and Connection Factory (日志记录器和连接工厂):**  Defines a logger for error logging and stores the `ReactiveRedisConnectionFactory`.

    ```java
    private static final Log logger = LogFactory.getLog(RedisReactiveHealthIndicator.class);

    private final ReactiveRedisConnectionFactory connectionFactory;

    private final Duration timeout = Duration.ofSeconds(5); // Health check timeout
    ```

    *   *中文: 定义一个日志记录器用于记录错误，并存储 `ReactiveRedisConnectionFactory`。还定义了一个超时时间，以防止健康检查无限期阻塞。*

4.  **Constructor (构造函数):** Initializes the class with the connection factory.

    ```java
    public RedisReactiveHealthIndicator(ReactiveRedisConnectionFactory connectionFactory) {
        super("Redis health check failed");  // Default error message
        this.connectionFactory = connectionFactory;
    }
    ```

    *   *中文: 构造函数，使用连接工厂初始化类。*

5.  **`doHealthCheck` Method (`doHealthCheck` 方法):**  The core health check logic.  It obtains a connection, performs the health check, and handles errors.  It also sets a timeout.

    ```java
    @Override
    protected Mono<Health> doHealthCheck(Health.Builder builder) {
        return getConnection()
                .flatMap(connection -> doHealthCheck(builder, connection))
                .timeout(this.timeout) // Add a timeout
                .onErrorResume(ex -> {
                    logger.warn("Redis health check failed", ex);
                    return Mono.just(builder.down(ex).build()); // Handle connection errors and other exceptions.
                });
    }
    ```

    *   *中文: 核心健康检查逻辑。它获取连接、执行健康检查并处理错误。还设置了一个超时时间。*

6.  **`getConnection` Method (`getConnection` 方法):**  Obtains a `ReactiveRedisConnection` from the factory, handling connection failures.

    ```java
    private Mono<ReactiveRedisConnection> getConnection() {
        return Mono.fromSupplier(this.connectionFactory::getReactiveConnection)
                .subscribeOn(Schedulers.boundedElastic()) // Use a dedicated scheduler
                .onErrorResume(ex -> {
                    logger.error("Failed to obtain Redis connection", ex);
                    return Mono.error(new RedisConnectionFailureException("Failed to obtain Redis connection", ex));
                });
    }
    ```

    *   *中文: 从工厂获取 `ReactiveRedisConnection`，并处理连接失败的情况。使用 `Schedulers.boundedElastic()` 确保连接获取不会阻塞主线程。*

7.  **`doHealthCheck` (Overloaded) Method (重载的 `doHealthCheck` 方法):** Performs the actual Redis health check, handling command execution errors.

    ```java
    private Mono<Health> doHealthCheck(Health.Builder builder, ReactiveRedisConnection connection) {
        return getHealth(builder, connection)
                .onErrorResume(ex -> {
                    logger.warn("Redis command execution failed", ex);
                    return Mono.just(builder.down(ex).build()); // Handle Redis command errors
                })
                .flatMap(health -> connection.closeLater().thenReturn(health));
    }
    ```

    *   *中文: 执行实际的 Redis 健康检查，并处理命令执行错误。`connection.closeLater()` 确保连接在操作完成后被释放。*

8.  **`getHealth` Method (`getHealth` 方法):**  Determines if the connection is a cluster connection and performs the appropriate health check.

    ```java
    private Mono<Health> getHealth(Health.Builder builder, ReactiveRedisConnection connection) {
        if (connection instanceof ReactiveRedisClusterConnection clusterConnection) {
            return clusterConnection.clusterGetClusterInfo().map(info -> fromClusterInfo(builder, info));
        }
        return connection.serverCommands().info("server")
                .map(info -> up(builder, info));
    }
    ```

    *   *中文: 确定连接是否为集群连接，并执行相应的健康检查。*

9.  **`up` Method (`up` 方法):**  Builds the `Health` object with Redis server information.

    ```java
    private Health up(Health.Builder builder, Properties info) {
        RedisHealth.up(builder, info);  // Use existing logic

        // Add Redis version and role information
        String redisVersion = info.getProperty("redis_version");
        if (StringUtils.hasText(redisVersion)) {
            builder.withDetail("version", redisVersion);
        }

        String role = info.getProperty("role"); // Available since Redis 2.8.12
        if (StringUtils.hasText(role)) {
            builder.withDetail("role", role);  // e.g., master, slave
        }

        return builder.build();
    }
    ```

    *   *中文: 构建包含 Redis 服务器信息的 `Health` 对象。添加了 Redis 版本和角色信息。*

10. **`fromClusterInfo` Method (`fromClusterInfo` 方法):**  Handles cluster-specific health information.

    ```java
    private Health fromClusterInfo(Health.Builder builder, ClusterInfo clusterInfo) {
        return RedisHealth.fromClusterInfo(builder, clusterInfo).build(); // Use existing logic
    }
    ```

    *   *中文: 处理集群特定的健康信息。*

**How to Use:**

This health indicator integrates directly with Spring Boot's Actuator. You don't need to call it directly. Spring Boot will automatically discover and use it when the Actuator is enabled.  Ensure you have `spring-boot-starter-actuator` and `spring-boot-starter-data-redis` dependencies in your `pom.xml` or `build.gradle`.

**Example `application.properties`:**

```properties
management.endpoints.web.exposure.include=health,info
management.health.redis.enabled=true
# If you need to configure Redis connection details:
spring.redis.host=localhost
spring.redis.port=6379
```

This version provides more robust error handling, detailed information, and prevents resource leaks. Remember to adjust the `timeout` value based on your environment.  The use of connection pooling (handled by Spring Data Redis) is generally recommended for production environments.

