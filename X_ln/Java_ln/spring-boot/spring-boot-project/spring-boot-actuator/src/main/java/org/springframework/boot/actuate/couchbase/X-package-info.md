Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\couchbase\package-info.java`

好的，重新开始。

**目的:** 提供 Spring Boot Actuator 对 Couchbase 的支持相关的代码示例，并附上中文描述。由于 `org.springframework.boot.actuate.couchbase` 本身就是一个 package，这意味着它包含多个类。 我将提供几个关键的 Actuator 组件，并解释它们的作用。

**1. `CouchbaseHealthIndicator` (Couchbase 健康指示器):**

```java
package org.springframework.boot.actuate.couchbase;

import com.couchbase.client.java.Bucket;
import com.couchbase.client.java.Cluster;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;

import java.time.Duration;

/**
 * {@link org.springframework.boot.actuate.health.HealthIndicator} for Couchbase.
 *
 * @author Eddú Meléndez
 * @author Stephane Nicoll
 * @since 2.0.0
 */
public class CouchbaseHealthIndicator extends AbstractHealthIndicator {

	private final Cluster couchbaseCluster;
	private final String bucketName;
	private final Duration timeout = Duration.ofSeconds(3);

	public CouchbaseHealthIndicator(Cluster couchbaseCluster, String bucketName) {
		super("Couchbase health check failed");
		this.couchbaseCluster = couchbaseCluster;
		this.bucketName = bucketName;
	}

	@Override
	protected void doHealthCheck(Health.Builder builder) throws Exception {
		try {
			Bucket bucket = this.couchbaseCluster.bucket(this.bucketName);
			bucket.exists(); // 简单的操作检查连接
			builder.up().withDetail("bucket", this.bucketName);
		}
		catch (Exception ex) {
			builder.down(ex);
		}
	}

}
```

**描述 (中文):** `CouchbaseHealthIndicator` 是一个健康指示器，用于检查 Couchbase 集群的健康状况。 它尝试连接到指定的 Bucket，如果连接成功，则认为 Couchbase 是健康的。 如果连接失败，则会返回一个包含错误信息的 down 状态。

**解释:**

*   `extends AbstractHealthIndicator`:  继承自 `AbstractHealthIndicator`，这是 Spring Boot Actuator 提供的一个方便的基类，用于创建自定义的健康指示器。
*   `couchbaseCluster`:  Couchbase 集群的实例。
*   `bucketName`:  要连接的 Bucket 的名称。
*   `doHealthCheck()`:  执行实际的健康检查逻辑。 在这里，它获取指定的 Bucket，并调用 `exists()` 方法来检查 Bucket 是否存在并可访问。
*   `builder.up()`:  如果健康检查成功，则将 Health 状态设置为 "up"，并添加 Bucket 名称作为详细信息。
*   `builder.down(ex)`:  如果健康检查失败，则将 Health 状态设置为 "down"，并添加异常信息。

**2. `CouchbaseHealthContributorAutoConfiguration` (Couchbase 健康贡献者自动配置):**

```java
package org.springframework.boot.actuate.couchbase;

import com.couchbase.client.java.Cluster;

import org.springframework.boot.actuate.autoconfigure.health.ConditionalOnEnabledHealthIndicator;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.couchbase.CouchbaseProperties;
import org.springframework.context.annotation.Bean;

/**
 * {@link EnableAutoConfiguration Auto-configuration} for {@link CouchbaseHealthIndicator}.
 *
 * @author Eddú Meléndez
 * @author Stephane Nicoll
 * @since 2.0.0
 */
@AutoConfiguration
@ConditionalOnBean({ Cluster.class, CouchbaseProperties.class })
@ConditionalOnEnabledHealthIndicator("couchbase")
public class CouchbaseHealthContributorAutoConfiguration {

	@Bean
	public CouchbaseHealthIndicator couchbaseHealthIndicator(Cluster couchbaseCluster,
			CouchbaseProperties couchbaseProperties) {
		return new CouchbaseHealthIndicator(couchbaseCluster, couchbaseProperties.getBucketName());
	}

}
```

**描述 (中文):**  `CouchbaseHealthContributorAutoConfiguration` 是一个自动配置类，用于配置 `CouchbaseHealthIndicator`。 它根据应用程序的配置来创建和注册 Couchbase 健康指示器。

**解释:**

*   `@AutoConfiguration`: 表明这是一个自动配置类，Spring Boot 会自动处理它。
*   `@ConditionalOnBean`:  只有当应用程序上下文中存在 `Cluster` 和 `CouchbaseProperties` Bean 时，才会应用此自动配置。 这确保了只有在配置了 Couchbase 连接时，才会创建健康指示器。
*   `@ConditionalOnEnabledHealthIndicator("couchbase")`: 只有当启用了名为 "couchbase" 的健康指示器时，才会应用此自动配置。 允许通过 `management.health.couchbase.enabled=false` 在 `application.properties` 或 `application.yml` 中禁用此健康指示器。
*   `@Bean`:  创建一个 `CouchbaseHealthIndicator` Bean，并将其添加到应用程序上下文中。  它使用 `Cluster` 和 `CouchbaseProperties` Bean 来配置健康指示器。

**简单演示 (中文):**

想象一下，你正在开发一个使用 Couchbase 的 Spring Boot 应用程序。

1.  **添加依赖:** 首先，你需要添加 Spring Boot Actuator 和 Couchbase 的依赖到你的 `pom.xml` 或 `build.gradle` 文件中。
2.  **配置 Couchbase:** 在 `application.properties` 或 `application.yml` 中配置 Couchbase 连接信息，例如服务器地址、端口、用户名和密码。  还需要配置 Bucket 名称。
3.  **启用健康检查:** (默认情况下已启用) 确保 `management.endpoint.health.enabled` 和 `management.health.couchbase.enabled` 都设置为 `true` (或者未显式设置，因为默认值为 `true`)。
4.  **访问健康端点:** 启动应用程序后，你可以访问 `/actuator/health` 端点 (默认情况下) 来查看应用程序的健康状况。  你应该会看到一个包含 Couchbase 健康信息的条目。

例如，返回的 JSON 可能如下所示：

```json
{
  "status": "UP",
  "components": {
    "couchbase": {
      "status": "UP",
      "details": {
        "bucket": "my-bucket"
      }
    }
  }
}
```

如果 Couchbase 连接失败，则 `status` 将为 "DOWN"，并且 `details` 中会包含错误信息。

**总结:**

Spring Boot Actuator 提供了方便的工具来监控你的应用程序。 `CouchbaseHealthIndicator` 和 `CouchbaseHealthContributorAutoConfiguration` 使得集成 Couchbase 健康检查变得非常容易。  通过简单的配置，你就可以在 `/actuator/health` 端点中获得 Couchbase 的健康信息。这对于监控和诊断 Couchbase 相关的问题非常有帮助。
