Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\cassandra\CassandraDriverHealthIndicator.java`

好的，我们重新开始。这次我会提供更智能、更完善的代码，并附带更详细的中文描述和演示。

**目标：**  改进 `CassandraDriverHealthIndicator` 类，使其能提供更详细的 Cassandra 集群健康信息，并更健壮地处理各种异常情况。

**1. 改进后的 `CassandraDriverHealthIndicator`:**

```java
package org.springframework.boot.actuate.cassandra;

import com.datastax.oss.driver.api.core.CqlSession;
import com.datastax.oss.driver.api.core.metadata.Node;
import com.datastax.oss.driver.api.core.metadata.NodeState;
import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;
import org.springframework.util.Assert;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * A more comprehensive {@link org.springframework.boot.actuate.health.HealthIndicator}
 * for Cassandra, providing details about node status and cluster health.
 */
public class CassandraDriverHealthIndicator extends AbstractHealthIndicator {

    private final CqlSession session;

    public CassandraDriverHealthIndicator(CqlSession session) {
        super("Cassandra health check failed");
        Assert.notNull(session, "CqlSession must not be null");
        this.session = session;
    }

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        try {
            Map<String, Object> details = new HashMap<>();
            Collection<Node> nodes = this.session.getMetadata().getNodes().values();

            // Check if ANY node is UP.  If none are UP, the cluster is considered DOWN.
            Optional<Node> anyNodeUp = nodes.stream().filter(node -> node.getState() == NodeState.UP).findAny();

            if (anyNodeUp.isPresent()) {
                builder.status(Status.UP);

                // Add detailed information about each node.
                for (Node node : nodes) {
                    Map<String, Object> nodeDetails = new HashMap<>();
                    nodeDetails.put("address", node.getEndPoint().resolve().toString()); // Use resolve() to get IP address
                    nodeDetails.put("state", node.getState().name());
                    nodeDetails.put("cassandraVersion", node.getCassandraVersion().toString());
                    nodeDetails.put("datacenter", node.getDatacenter());
                    nodeDetails.put("rack", node.getRack());
                    details.put(node.getHostId().toString(), nodeDetails);
                }

                anyNodeUp.map(Node::getCassandraVersion).ifPresent(version -> builder.withDetail("clusterVersion", version.toString()));
                builder.withDetails(details); // Add all node details.

            } else {
                builder.status(Status.DOWN).withDetail("message", "No Cassandra nodes are currently UP.");
            }

        } catch (Exception e) {
            builder.down().withException(e); // Capture any exceptions during health check.
        }
    }
}
```

**代码改进说明 (中文):**

*   **更详细的健康信息:**  现在，不仅会检查是否有节点处于 UP 状态，还会收集每个节点的详细信息，例如地址、状态、Cassandra 版本、数据中心和机架。这些信息以 `Map` 的形式添加到 `Health.Builder` 中。
*   **更强的异常处理:**  使用 `try-catch` 块捕获在健康检查过程中可能发生的任何异常，并将异常信息添加到 `Health.Builder` 中，使诊断问题更容易。
*   **更精确的地址:** 使用`node.getEndPoint().resolve().toString()` 而不是`node.getEndPoint().toString()` 以确保获得节点的IP地址，而不是主机名（在某些环境中主机名可能无法解析）。
*   **清晰的 DOWN 状态信息:**  如果没有任何 Cassandra 节点处于 UP 状态，则会添加一条包含 "No Cassandra nodes are currently UP." 消息的详细信息，说明 Cassandra 集群无法正常工作的原因。
*   **注释:** 代码中添加了更多注释，解释每个部分的功能。
*   **使用`withDetails`方法:** 将所有的节点信息添加到`Health.Builder`中，而不是只添加单个节点的信息。

**2. 演示 (Demo):**

因为这是一个 Spring Boot Actuator 的健康指示器，所以演示需要在 Spring Boot 环境中进行。你需要一个 Cassandra 数据库运行，并且你的 Spring Boot 应用配置为连接到该数据库。

**步骤：**

1.  **添加依赖:**  确保你的 `pom.xml` 或 `build.gradle` 文件中包含以下依赖项：

    ```xml
    <!-- Maven -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-cassandra</artifactId>
    </dependency>
    <!-- 确保你也有 Cassandra Driver 的依赖 -->
    <dependency>
        <groupId>com.datastax.oss</groupId>
        <artifactId>java-driver-core</artifactId>
        <version>最新版本</version> <!-- 请替换为实际的版本号 -->
    </dependency>
    ```

    ```gradle
    // Gradle
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
    implementation 'org.springframework.boot:spring-boot-starter-data-cassandra'
    implementation group: 'com.datastax.oss', name: 'java-driver-core', version: '最新版本' // 替换为实际版本号
    ```

2.  **配置 Cassandra:**  在你的 `application.properties` 或 `application.yml` 文件中配置 Cassandra 连接信息：

    ```properties
    spring.data.cassandra.keyspace-name=your_keyspace
    spring.data.cassandra.contact-points=127.0.0.1
    spring.data.cassandra.port=9042
    #spring.data.cassandra.local-datacenter=datacenter1  // 如果需要，取消注释
    ```

3.  **创建 Bean:** 创建一个 `CassandraDriverHealthIndicator` 的 Bean：

    ```java
    import com.datastax.oss.driver.api.core.CqlSession;
    import org.springframework.context.annotation.Bean;
    import org.springframework.context.annotation.Configuration;
    import org.springframework.boot.actuate.cassandra.CassandraDriverHealthIndicator;

    @Configuration
    public class CassandraHealthConfig {

        @Bean
        public CassandraDriverHealthIndicator cassandraDriverHealthIndicator(CqlSession session) {
            return new CassandraDriverHealthIndicator(session);
        }
    }
    ```

4.  **访问 Actuator 端点:**  运行你的 Spring Boot 应用，然后访问 `/actuator/health` 端点 (通常在 `http://localhost:8080/actuator/health`)。 你应该看到类似以下的 JSON 响应：

    ```json
    {
      "status": "UP",
      "details": {
        "cassandra": {
          "status": "UP",
          "details": {
            "clusterVersion": "4.0.0",
            "08a3b12c-6397-484c-a537-6508a40687e4": {
              "address": "127.0.0.1:9042",
              "state": "UP",
              "cassandraVersion": "4.0.0",
              "datacenter": "datacenter1",
              "rack": "rack1"
            }
          }
        }
      }
    }
    ```

    如果 Cassandra 集群不可用，你将会看到 `"status": "DOWN"` 和一个描述问题的 `message` 字段。

**中文总结:**

这个改进后的 `CassandraDriverHealthIndicator` 提供了更全面、更健壮的 Cassandra 集群健康检查。 通过提供每个节点的详细信息和更强的异常处理，你可以更容易地诊断和解决 Cassandra 集群中的问题。  演示部分展示了如何在 Spring Boot 应用中使用这个健康指示器，并通过访问 `/actuator/health` 端点来查看健康信息。

希望这个更详细的解释和演示能满足你的需求!  如果还有其他问题，请随时提出。
