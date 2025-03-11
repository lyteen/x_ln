Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\data\mongo\MongoHealthIndicator.java`

好的，我们重新开始，并提供更智能的 `MongoHealthIndicator` 版本，并附带中文描述和示例。

**改进的 `MongoHealthIndicator`:**

```java
package org.springframework.boot.actuate.data.mongo;

import com.mongodb.MongoException;

import org.bson.Document;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.util.Assert;

/**
 * 增强的 {@link HealthIndicator} 实现，用于检查 MongoDB 数据存储的健康状况。
 * 增加了连接检查和更详细的错误处理。
 *
 * @author Your Name (可以替换成你的名字)
 * @since 2.0.0 (可以根据你的实际情况修改)
 */
public class MongoHealthIndicator extends AbstractHealthIndicator {

	private final MongoTemplate mongoTemplate;

	public MongoHealthIndicator(MongoTemplate mongoTemplate) {
		super("MongoDB health check failed");
		Assert.notNull(mongoTemplate, "'mongoTemplate' must not be null");
		this.mongoTemplate = mongoTemplate;
	}

	@Override
	protected void doHealthCheck(Health.Builder builder) throws Exception {
		try {
			// 尝试连接到 MongoDB
			Document result = this.mongoTemplate.executeCommand("{ hello: 1 }");

			// 检查命令是否成功执行
			if (result != null && result.containsKey("maxWireVersion")) {
				builder.up().withDetail("maxWireVersion", result.getInteger("maxWireVersion"))
						.withDetail("serverAddress", this.mongoTemplate.getDb().getName() + "@" + this.mongoTemplate.getDb().getMongo().getAddress()); // 添加数据库地址信息
			} else {
				builder.down().withDetail("message", "Unexpected response from MongoDB: " + result);
			}

		} catch (MongoException ex) {
			// 捕获连接或命令执行失败的异常
			builder.down(ex); // 记录异常信息
		} catch (Exception ex) {
			// 捕获其他未知异常
			builder.down(ex); // 记录异常信息
		}
	}

}
```

**代码解释 (中文描述):**

*   **更强的异常处理:**  使用 `try-catch` 块来捕获 `MongoException` (例如连接失败、认证失败) 以及其他可能出现的异常。 这使得在 MongoDB 无法连接或出现其他问题时，健康检查能够更优雅地处理并提供更具体的错误信息。
*   **详细的健康信息:** 除了 `maxWireVersion` 之外，还添加了 `serverAddress`，包含了数据库名称和 MongoDB 服务器的地址。  这在诊断连接问题时非常有用。
*   **明确的成功/失败判断:**  显式地检查 `executeCommand` 的结果，以确保命令成功执行并且结果中包含预期的 `maxWireVersion`。 如果结果不符合预期，则将健康状态标记为 `DOWN` 并提供详细的消息。
*   **错误信息传递:**  当发生异常时，`builder.down(ex)` 会将异常信息包含在健康检查结果中，方便开发者了解问题的根源。

**示例应用 (中文描述):**

假设你有一个 Spring Boot 应用，使用了 `spring-boot-starter-data-mongodb` 依赖。  你只需要创建一个 `MongoTemplate` 的 bean，Spring Boot 就会自动装配 `MongoHealthIndicator`。

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

@Configuration
public class MongoConfig {

    @Bean
    public MongoTemplate mongoTemplate() throws Exception {
        // 这里你需要配置你的 MongoDB 连接信息
        // 例如:
        // return new MongoTemplate(MongoClients.create("mongodb://localhost:27017"), "mydatabase");
        // 为了演示，这里我们简化一下
        return new MongoTemplate(null, "test"); // 请替换成你的实际配置！
    }
}
```

然后，你就可以通过访问 `/actuator/health` 端点来查看 MongoDB 的健康状况。  如果 MongoDB 运行正常，你会看到类似下面的 JSON 响应：

```json
{
  "status": "UP",
  "details": {
    "maxWireVersion": 17,
    "serverAddress": "test@localhost:27017"
  }
}
```

如果 MongoDB 连接失败，你会看到 `status` 为 `DOWN`，并且 `details` 中包含错误信息。 例如：

```json
{
  "status": "DOWN",
  "details": {
    "error": "com.mongodb.MongoSocketOpenException: Exception opening socket",
    "message": "MongoDB health check failed"
  }
}
```

**总结:**

这个改进的 `MongoHealthIndicator` 提供了更健壮的健康检查机制，可以更好地监控 MongoDB 的运行状态，并提供更详细的错误信息，方便开发者进行问题排查。记住将示例配置中的 `MongoTemplate` 初始化替换成你实际的 MongoDB 连接信息。
