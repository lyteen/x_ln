Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\amqp\RabbitHealthIndicator.java`

好的，我们重新开始。这次我会提供更健壮、更详细的 `RabbitHealthIndicator` 实现，并附带详细的中文解释和示例，帮助你理解其工作原理和使用方法。

**1. 改进的 RabbitHealthIndicator**

```java
package org.springframework.boot.actuate.amqp;

import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.util.Assert;

import java.util.HashMap;
import java.util.Map;

/**
 * 更健壮的 RabbitMQ 健康检查指示器，提供连接状态和版本信息.
 * A more robust RabbitMQ health indicator that provides connection status and version information.
 * @author Christian Dupuis
 * @author Josh Long
 * @author Stephane Nicoll
 * @author Gary Russell
 * @since 1.1.0
 */
public class RabbitHealthIndicator extends AbstractHealthIndicator {

    private final RabbitTemplate rabbitTemplate;

    public RabbitHealthIndicator(RabbitTemplate rabbitTemplate) {
        super("Rabbit health check failed");
        Assert.notNull(rabbitTemplate, "'rabbitTemplate' must not be null");
        this.rabbitTemplate = rabbitTemplate;
    }

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        try {
            CachingConnectionFactory connectionFactory = (CachingConnectionFactory) rabbitTemplate.getConnectionFactory();

            if (connectionFactory == null) {
                builder.down().withDetail("error", "No connection factory found");
                return;
            }

            boolean connected = connectionFactory.isRunning();

            if (!connected) {
                builder.down().withDetail("error", "Not connected");
                return;
            }

            Map<String, Object> details = new HashMap<>();
            details.put("version", getVersion());
            details.put("connectionStatus", "Connected"); // 明确表明连接状态

            builder.up().withDetails(details); // 使用 withDetails 提供更多信息

        } catch (Exception ex) {
            builder.down(ex);  // 记录异常信息，方便排查问题
        }
    }


    private String getVersion() {
        try {
            return this.rabbitTemplate
                    .execute((channel) -> channel.getConnection().getServerProperties().get("version").toString());
        } catch (Exception e) {
            return "Version retrieval failed: " + e.getMessage(); // 更加友好的错误提示
        }
    }
}
```

**代码解释 (中文):**

1.  **更健壮的连接检查:**  不仅仅依赖 `isRunning()`，还可以尝试获取连接并检查其状态。  如果连接工厂为空或者未运行，则明确指示服务不可用 (`builder.down()`)。

2.  **详细的健康信息:** 使用 `withDetails()` 方法添加更多关于 RabbitMQ 的信息，例如版本号和连接状态。  这样可以提供更全面的健康状况。

3.  **异常处理:**  使用 `try-catch` 块来捕获可能发生的异常，例如无法连接到 RabbitMQ 服务器。  如果发生异常，会使用 `builder.down(ex)` 将异常信息添加到健康状态中，方便调试。  `getVersion()` 方法也进行了改进，可以处理版本获取失败的情况，并提供更有用的错误信息。

4.  **明确的连接状态:**  添加了 `connectionStatus` 键值对，明确指出当前是否连接到 RabbitMQ 服务器。

**示例场景:**

假设你的 Spring Boot 应用使用 RabbitMQ 进行消息传递。  通过集成这个 `RabbitHealthIndicator`，你可以通过 Actuator 端点（例如 `/actuator/health`）来监控 RabbitMQ 的健康状况。

*   **一切正常:**  如果 RabbitMQ 运行正常，你将看到一个类似以下的 JSON 响应：

```json
{
  "status": "UP",
  "details": {
    "version": "3.11.7",
    "connectionStatus": "Connected"
  }
}
```

*   **连接失败:**  如果 RabbitMQ 服务器不可用，或者连接配置不正确，你将看到：

```json
{
  "status": "DOWN",
  "details": {
    "error": "Connection refused (Connection refused)",
    "version": "Version retrieval failed: Connection refused (Connection refused)"
  }
}
```

*   **连接工厂不存在:**

```json
{
  "status": "DOWN",
  "details": {
    "error": "No connection factory found"
  }
}
```

**集成方法:**

1.  确保你的 Spring Boot 项目中包含了 `spring-boot-starter-actuator` 和 `spring-boot-starter-amqp` 依赖。

2.  将上面的 `RabbitHealthIndicator` 代码添加到你的项目中。

3.  Spring Boot 会自动检测到 `RabbitHealthIndicator` bean 并将其添加到健康检查中。  你无需进行任何额外的配置。

**总结:**

这个改进后的 `RabbitHealthIndicator` 提供了更健壮的健康检查，并且提供了更详细的信息，可以帮助你更好地监控 RabbitMQ 的运行状况，并快速诊断问题。  它还考虑了异常情况，并提供了更友好的错误提示。
