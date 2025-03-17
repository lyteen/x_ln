Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\cassandra\CassandraDriverReactiveHealthIndicator.java`

好的，让我们重新开始。 我会提供一个更智能的 `CassandraDriverReactiveHealthIndicator` 实现，并在描述中包含代码片段和中文解释，同时提供一个简单的 demo。

**改进后的 CassandraDriverReactiveHealthIndicator**

```java
/*
 * Copyright 2012-2025 the original author or authors.
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

package org.springframework.boot.actuate.cassandra;

import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import com.datastax.oss.driver.api.core.CqlSession;
import com.datastax.oss.driver.api.core.metadata.Node;
import com.datastax.oss.driver.api.core.metadata.NodeState;
import reactor.core.publisher.Mono;

import org.springframework.boot.actuate.health.AbstractReactiveHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.ReactiveHealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.util.Assert;

/**
 * Simple implementation of a {@link ReactiveHealthIndicator} returning status information
 * for Cassandra data stores.
 *
 * @author Alexandre Dutra
 * @author Tomasz Lelek
 * @since 2.4.0
 */
public class CassandraDriverReactiveHealthIndicator extends AbstractReactiveHealthIndicator {

	private final CqlSession session;

	/**
	 * Create a new {@link CassandraDriverReactiveHealthIndicator} instance.
	 * @param session the {@link CqlSession}.
	 */
	public CassandraDriverReactiveHealthIndicator(CqlSession session) {
		super("Cassandra health check failed");
		Assert.notNull(session, "'session' must not be null");
		this.session = session;
	}

	@Override
	protected Mono<Health> doHealthCheck(Health.Builder builder) {
		return Mono.fromSupplier(() -> {
			Collection<Node> nodes = this.session.getMetadata().getNodes().values();

			// 统计节点状态，并添加到 Health Detail 中
			Map<NodeState, Long> nodeStateCounts = nodes.stream()
					.collect(Collectors.groupingBy(Node::getState, Collectors.counting()));
			builder.withDetail("nodeStates", nodeStateCounts);

			// 找出至少一个 UP 状态的节点
			Optional<Node> nodeUp = nodes.stream().filter((node) -> node.getState() == NodeState.UP).findAny();

			// 设置 Health 状态
			builder.status(nodeUp.isPresent() ? Status.UP : Status.DOWN);

			// 添加 Cassandra 版本信息 (如果找到 UP 状态的节点)
			nodeUp.map(Node::getCassandraVersion).ifPresent((version) -> builder.withDetail("version", version));

			// 添加 Keyspace 信息 (可选，需要执行 CQL 查询)
			try {
				// 执行一个简单的 CQL 查询来验证连接和权限
				session.execute("SELECT keyspace_name FROM system_schema.keyspaces LIMIT 1");
				builder.withDetail("keyspaceAccess", "OK");
			}
			catch (Exception e) {
				builder.withDetail("keyspaceAccess", "ERROR: " + e.getMessage());
				builder.down(e); // 如果查询失败，将 Health 状态设置为 DOWN
			}

			return builder.build();
		});
	}

}
```

**代码描述 (中文):**

*   **节点状态统计:**  我们不再仅仅检查是否有 UP 状态的节点，而是统计所有节点的状态，并将每个状态的节点数量添加到 Health Detail 中。这提供了更详细的集群状态信息。
    ```java
    Map<NodeState, Long> nodeStateCounts = nodes.stream()
    		.collect(Collectors.groupingBy(Node::getState, Collectors.counting()));
    builder.withDetail("nodeStates", nodeStateCounts);
    ```
    这段代码使用 Java 8 的 Stream API 对节点列表进行分组统计，统计每个 `NodeState` 的节点数量，并将结果存储在一个 `Map` 中。  然后，将此 `Map` 作为 `nodeStates` 详细信息添加到 Health Builder 中。

*   **Keyspace 访问验证:**  添加了一个简单的 CQL 查询来验证是否可以连接到 Cassandra 集群并访问 Keyspace。如果查询失败，会捕获异常并将错误信息添加到 Health Detail 中，并且将 Health 状态设置为 `DOWN`。 这可以帮助诊断权限问题或连接问题。
    ```java
    try {
    	// 执行一个简单的 CQL 查询来验证连接和权限
    	session.execute("SELECT keyspace_name FROM system_schema.keyspaces LIMIT 1");
    	builder.withDetail("keyspaceAccess", "OK");
    }
    catch (Exception e) {
    	builder.withDetail("keyspaceAccess", "ERROR: " + e.getMessage());
    	builder.down(e); // 如果查询失败，将 Health 状态设置为 DOWN
    }
    ```
    这段代码尝试执行一个简单的 `SELECT` 查询，来验证与 Cassandra 集群的连接是否正常，以及当前用户是否具有访问 `system_schema.keyspaces` 的权限。 如果查询成功，则在 Health Detail 中添加 `keyspaceAccess: OK`。 如果查询失败（例如，连接错误，权限不足），则捕获异常，在 Health Detail 中添加错误信息，并将 Health 状态设置为 `DOWN`。

*   **异常处理:**  如果 CQL 查询失败，会捕获异常并使用 `builder.down(e)` 将 Health 状态设置为 DOWN，以便更准确地反映 Cassandra 集群的健康状况。

**简单的 Demo (中文):**

1.  **引入依赖:**  确保你的 Spring Boot 项目中已经引入了 Cassandra Driver 的依赖。

2.  **配置 Cassandra:**  在 `application.properties` 或 `application.yml` 中配置 Cassandra 连接信息。  例如:
    ```yaml
    spring.data.cassandra.keyspace-name=my_keyspace
    spring.data.cassandra.contact-points=127.0.0.1
    spring.data.cassandra.port=9042
    ```

3.  **注入 CqlSession:**  在你的 Spring Bean 中注入 `CqlSession`。 Spring Boot 会自动配置它。

4.  **使用 Health Indicator:**  Spring Boot Actuator 会自动发现并使用 `CassandraDriverReactiveHealthIndicator`。  你可以通过访问 `/actuator/health` 端点来查看 Cassandra 的健康状况。

**期望的 Health Endpoint 响应 (中文):**

当 Cassandra 集群正常运行时，你可能会看到如下类似的 Health Endpoint 响应:

```json
{
  "status": "UP",
  "details": {
    "nodeStates": {
      "UP": 3  // 3个节点处于 UP 状态
    },
    "version": "4.0.0",
    "keyspaceAccess": "OK"
  }
}
```

如果 Cassandra 集群出现问题，例如无法连接或 Keyspace 访问失败，你可能会看到:

```json
{
  "status": "DOWN",
  "details": {
    "nodeStates": {
      "DOWN": 1, // 1个节点处于 DOWN 状态
      "UP": 2    // 2个节点处于 UP 状态
    },
    "keyspaceAccess": "ERROR: User myuser does not have permission to access keyspace system_schema"
  }
}
```

这个改进后的 `CassandraDriverReactiveHealthIndicator` 提供了更详细和准确的 Cassandra 集群健康状况信息，帮助你更好地监控和管理你的 Cassandra 集群。 统计节点状态和验证 Keyspace 访问是关键的改进点。
