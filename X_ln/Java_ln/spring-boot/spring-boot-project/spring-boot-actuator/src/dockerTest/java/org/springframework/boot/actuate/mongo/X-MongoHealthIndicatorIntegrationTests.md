Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\dockerTest\java\org\springframework\boot\actuate\mongo\MongoHealthIndicatorIntegrationTests.java`

好的，让我们重新开始，并提供带有中文描述和示例的更智能的改进版本。

**1. 改进的 `MongoHealthIndicator` 类:**

```java
package org.springframework.boot.actuate.mongo;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import com.mongodb.MongoClientSettings.Builder;
import com.mongodb.ServerApi;
import com.mongodb.ServerApiVersion;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.springframework.boot.actuate.data.mongo.MongoHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.util.StringUtils;

import java.util.Map;

/**
 * 增强的 MongoDB 健康指示器，提供更详细的信息。
 * Enhanced MongoDB health indicator providing more detailed information.
 */
public class EnhancedMongoHealthIndicator implements HealthIndicator {

    private final MongoTemplate mongoTemplate;

    private final String databaseName;

    public EnhancedMongoHealthIndicator(MongoTemplate mongoTemplate) {
        this(mongoTemplate, null);
    }

    public EnhancedMongoHealthIndicator(MongoTemplate mongoTemplate, String databaseName) {
        this.mongoTemplate = mongoTemplate;
        this.databaseName = StringUtils.hasText(databaseName) ? databaseName : mongoTemplate.getDb().getName();
    }

    @Override
    public Health health() {
        try {
            Map<String, Object> info = this.mongoTemplate.db.runCommand("{ serverStatus: 1 }").toDocument();
            // 获取服务器状态信息，例如版本、连接数等。Get server status information such as version, connections, etc.
            String version = (String) info.get("version");
            Number connections = ((Map<?, ?>) info.get("connections")).get("current");

            // 构建健康信息。Build health information.
            return Health.up()
                    .withDetail("version", version)
                    .withDetail("database", this.databaseName)
                    .withDetail("connections", connections)
                    .build();

        } catch (Exception ex) {
            // 如果发生异常，则指示服务DOWN，并包含错误信息。If an exception occurs, indicate the service is DOWN and include error information.
            return Health.down(ex).withDetail("database", this.databaseName).build();
        }
    }
}
```

**描述:** 这个改进的 `EnhancedMongoHealthIndicator` 不仅仅检查 MongoDB 连接是否正常，还会获取 MongoDB 服务器的状态信息，例如版本号和当前连接数。  这些信息会添加到 `Health` 对象中，提供更丰富的监控数据。

*   **更详细的信息:**  包含了 MongoDB 版本和连接数。
*   **数据库名称:**  可以指定要检查的数据库名称。

**2.  修改后的集成测试类:**

```java
package org.springframework.boot.actuate.mongo;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import com.mongodb.MongoClientSettings.Builder;
import com.mongodb.ServerApi;
import com.mongodb.ServerApiVersion;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.MongoDBContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.testsupport.container.TestImage;
import org.springframework.data.mongodb.core.MongoTemplate;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link EnhancedMongoHealthIndicator} 的集成测试.
 * Integration tests for {@link EnhancedMongoHealthIndicator}.
 */
@Testcontainers(disabledWithoutDocker = true)
class EnhancedMongoHealthIndicatorIntegrationTests {

    @Container
    static MongoDBContainer mongo = TestImage.container(MongoDBContainer.class);

    @Test
    void standardApi() {
        Health health = mongoHealth();
        assertThat(health.getStatus()).isEqualTo(Status.UP);
        assertThat(health.getDetails()).containsKey("version");
        assertThat(health.getDetails()).containsKey("database");
        assertThat(health.getDetails()).containsKey("connections");
    }

    @Test
    void strictV1Api() {
        Health health = mongoHealth(ServerApi.builder().strict(true).version(ServerApiVersion.V1).build());
        assertThat(health.getStatus()).isEqualTo(Status.UP);
        assertThat(health.getDetails()).containsKey("version");
        assertThat(health.getDetails()).containsKey("database");
        assertThat(health.getDetails()).containsKey("connections");
    }

    private Health mongoHealth() {
        return mongoHealth(null);
    }

    private Health mongoHealth(ServerApi serverApi) {
        Builder settingsBuilder = MongoClientSettings.builder()
                .applyConnectionString(new ConnectionString(mongo.getConnectionString()));
        if (serverApi != null) {
            settingsBuilder.serverApi(serverApi);
        }
        MongoClientSettings settings = settingsBuilder.build();
        MongoClient mongoClient = MongoClients.create(settings);
        MongoTemplate mongoTemplate = new MongoTemplate(mongoClient, "testdb"); // 使用指定的数据库
        EnhancedMongoHealthIndicator healthIndicator = new EnhancedMongoHealthIndicator(mongoTemplate, "testdb"); // 指定数据库名称
        return healthIndicator.health();  // 使用 health() 方法
    }

}
```

**描述:**

*   **测试 `Health` 对象的细节:** 集成测试现在会检查 `Health` 对象是否包含 "version", "database" 和 "connections" 字段，确保新的健康指示器返回了期望的详细信息。
*   **使用指定的数据库:**  在创建 `MongoTemplate` 和 `EnhancedMongoHealthIndicator` 时，显式地指定了数据库名称 "testdb"。这确保了健康检查针对的是正确的数据库。
*   **使用 `health()` 方法:**  现在调用的是 `healthIndicator.health()` 方法，而不是 `getHealth(true)`。  `health()` 方法是 `HealthIndicator` 接口的标准方法。

**3.  演示:**

假设你的 Spring Boot 应用连接到了一个 MongoDB 数据库。  当调用 `/actuator/health` 端点时，你将会看到如下类似的 JSON 响应：

```json
{
  "status": "UP",
  "details": {
    "version": "5.0.5",
    "database": "testdb",
    "connections": 12
  }
}
```

这个响应表明 MongoDB 连接正常，并且提供了 MongoDB 服务器的版本号 (5.0.5)，数据库名称 (testdb) 以及当前连接数 (12)。  如果 MongoDB 服务不可用，`status` 将会是 "DOWN"，`details` 包含错误信息。

**总结:**

这些改进提供了一个更健壮和信息丰富的 MongoDB 健康指示器，它可以帮助你更好地监控 MongoDB 服务的运行状况。  测试也经过了更新，以确保新的健康指示器按预期工作。  希望这些改进能够满足你的需求!
