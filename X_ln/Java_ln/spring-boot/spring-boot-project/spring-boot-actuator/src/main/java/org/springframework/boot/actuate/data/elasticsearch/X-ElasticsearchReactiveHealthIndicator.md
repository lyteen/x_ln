Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\data\elasticsearch\ElasticsearchReactiveHealthIndicator.java`

Okay, let's refine this `ElasticsearchReactiveHealthIndicator` with a focus on clarity, conciseness, and resilience. I'll break down the changes and provide explanations.  I'll also include a basic demonstration of how it might be used in a Spring Boot application.

**1. Refactored `ElasticsearchReactiveHealthIndicator`:**

```java
package org.springframework.boot.actuate.data.elasticsearch;

import co.elastic.clients.elasticsearch._types.HealthStatus;
import co.elastic.clients.elasticsearch.cluster.HealthResponse;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import reactor.core.publisher.Mono;

import org.springframework.boot.actuate.health.AbstractReactiveHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;
import org.springframework.data.elasticsearch.client.elc.ReactiveElasticsearchClient;

import java.util.Map;

/**
 * Reactive {@link org.springframework.boot.actuate.health.HealthIndicator} for an Elasticsearch cluster
 * using a {@link ReactiveElasticsearchClient}.
 *
 * @author Brian Clozel
 * @author Aleksander Lech
 * @author Scott Frederick
 * @since 2.3.2
 */
public class ElasticsearchReactiveHealthIndicator extends AbstractReactiveHealthIndicator {

	private static final Log logger = LogFactory.getLog(ElasticsearchReactiveHealthIndicator.class);

	private final ReactiveElasticsearchClient client;

	public ElasticsearchReactiveHealthIndicator(ReactiveElasticsearchClient client) {
		super("Elasticsearch health check failed");  // Default error message
		this.client = client;
	}

	@Override
	protected Mono<Health> doHealthCheck(Health.Builder builder) {
		return this.client.cluster().health((b) -> b)
				.map(response -> processHealthResponse(builder, response))
				.onErrorResume(ex -> {
					logger.warn("Elasticsearch health check failed", ex);
					return Mono.just(builder.down(ex).build()); // Handle connection errors gracefully
				});
	}

	private Health processHealthResponse(Health.Builder builder, HealthResponse response) {
		if (response.timedOut()) {
			return builder.down().withDetail("timeout", true).build(); // Indicate timeout specifically
		}

		HealthStatus status = response.status();
		Status healthStatus = convertHealthStatus(status); // Convert Elasticsearch HealthStatus to Spring Boot Status

		builder.status(healthStatus);
		builder.withDetails(extractDetails(response)); // Extract all details into a map

		return builder.build();
	}

	private Status convertHealthStatus(HealthStatus elasticsearchStatus) {
		return switch (elasticsearchStatus) {
			case Red -> Status.OUT_OF_SERVICE;
			case Yellow -> Status.WARN; // Add a warning state
			default -> Status.UP;
		};
	}


	private Map<String, Object> extractDetails(HealthResponse response) {
		return Map.of(
				"cluster_name", response.clusterName(),
				"status", response.status().jsonValue(),
				"timed_out", response.timedOut(),
				"number_of_nodes", response.numberOfNodes(),
				"number_of_data_nodes", response.numberOfDataNodes(),
				"active_primary_shards", response.activePrimaryShards(),
				"active_shards", response.activeShards(),
				"relocating_shards", response.relocatingShards(),
				"initializing_shards", response.initializingShards(),
				"unassigned_shards", response.unassignedShards(),
				"delayed_unassigned_shards", response.delayedUnassignedShards(),
				"number_of_pending_tasks", response.numberOfPendingTasks(),
				"number_of_in_flight_fetch", response.numberOfInFlightFetch(),
				"task_max_waiting_in_queue_millis", response.taskMaxWaitingInQueueMillis(),
				"active_shards_percent_as_number", Double.parseDouble(response.activeShardsPercentAsNumber()),
				"unassigned_primary_shards", response.unassignedPrimaryShards()
		);
	}

}
```

**Key Improvements and Explanations:**

*   **Error Handling:**  The `doHealthCheck` method now includes `.onErrorResume()`.  This is *crucial* for reactive applications.  If the Elasticsearch client fails to connect (e.g., the cluster is down), the health indicator won't crash the entire application.  Instead, it logs a warning and reports the health as `DOWN`.  This is essential for resilience.

*   **Timeout Handling:** Added explicit handling for `response.timedOut()`. It now provides a `timeout: true` detail in the Health object when a timeout occurs, making debugging easier.

*   **HealthStatus Conversion:**  A `convertHealthStatus` method maps Elasticsearch's `HealthStatus` (Red, Yellow, Green) to Spring Boot's `Status` (UP, DOWN, OUT_OF_SERVICE, WARN).  I've added a `WARN` state for `HealthStatus.Yellow`.  You can customize this mapping as needed.

*   **Concise Detail Extraction:** The `extractDetails` method creates a `Map` of all the details from the `HealthResponse`.  This makes the code much cleaner and easier to maintain.  It avoids repeating `builder.withDetail()` calls.  Using `Map.of()` creates an immutable map, which is generally a good practice.

*   **Logging:** Includes a `Log` instance for logging warnings or errors during the health check.  This is useful for monitoring.

*   **Clarity:** Method names (e.g., `processHealthResponse`, `extractDetails`) improve readability.

**Why These Changes Matter:**

*   **Resilience:**  The `onErrorResume` makes the health indicator much more robust.
*   **Observability:**  Logging provides valuable insights into the health check process.
*   **Maintainability:**  The use of methods like `extractDetails` makes the code easier to understand and modify.
*   **Actionable Information:** The detailed information helps to find the root cause of health problems

**2. Example Spring Boot Configuration (Conceptual):**

This example demonstrates how you would typically use this health indicator in a Spring Boot application.  Remember that you'll need the necessary dependencies (e.g., Spring Boot Actuator, Spring Data Elasticsearch Reactive, and the Elasticsearch client).

```java
package com.example.demo;

import org.springframework.boot.actuate.data.elasticsearch.ElasticsearchReactiveHealthIndicator;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.elasticsearch.client.elc.ReactiveElasticsearchClient;

@Configuration
public class ElasticsearchHealthConfig {

    @Bean
    public ElasticsearchReactiveHealthIndicator elasticsearchHealthIndicator(ReactiveElasticsearchClient client) {
        return new ElasticsearchReactiveHealthIndicator(client);
    }
}
```

**Explanation:**

1.  **`@Configuration`:**  Marks this class as a Spring configuration class.
2.  **`@Bean`:**  Creates a Spring bean of type `ElasticsearchReactiveHealthIndicator`.
3.  **`ReactiveElasticsearchClient` Injection:** The `ReactiveElasticsearchClient` is automatically injected by Spring, assuming you have configured it elsewhere in your application (usually with connection details).

**To make it runnable, you need to:**

1.  **Add Dependencies:** Include the necessary dependencies in your `pom.xml` or `build.gradle` file:

    ```xml
    <!-- For Maven -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-elasticsearch</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-webflux</artifactId>
    </dependency>
    ```

    ```gradle
    // For Gradle
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
    implementation 'org.springframework.boot:spring-boot-starter-data-elasticsearch'
    implementation 'org.springframework.boot:spring-boot-starter-webflux'
    ```

2.  **Configure `ReactiveElasticsearchClient`:** You need to create a bean for `ReactiveElasticsearchClient`.  The specific configuration will depend on how you connect to your Elasticsearch cluster (e.g., using `WebClient` or the new Elasticsearch Java client). Here is a simple example using `WebClient`:

    ```java
    @Configuration
    public class ElasticsearchClientConfig {

        @Bean
        public ReactiveElasticsearchClient reactiveElasticsearchClient() {
            // Replace with your Elasticsearch URL
            String elasticsearchUrl = "http://localhost:9200";

            return ReactiveElasticsearchClient.create(elasticsearchUrl);
        }
    }
    ```

3.  **Access the Health Endpoint:**  Once your Spring Boot application is running, you can access the health endpoint (typically `/actuator/health`) to see the status of your Elasticsearch cluster.  You might need to enable the endpoint in your `application.properties` or `application.yml` file:

    ```properties
    management.endpoints.web.exposure.include=health
    management.endpoint.health.show-details=always
    ```

**Demonstration in Chinese (演示):**

假设你已经创建了一个 Spring Boot 项目，并且已经添加了上面提到的依赖。 你需要在你的 `application.properties` 或者 `application.yml` 文件中配置 Elasticsearch 的连接信息（比如主机地址，端口号，用户名密码等等）。  然后，当你启动你的 Spring Boot 应用，你可以通过访问 `http://localhost:8080/actuator/health` (假设你的应用运行在 8080 端口) 来查看 Elasticsearch 的健康状态。 Actuator endpoints 可能需要配置才能被访问。如果 Elasticsearch 集群运行正常，你会看到 `status: UP`。 如果集群出现问题，比如无法连接，你会看到 `status: DOWN`，并且会包含错误信息，帮助你诊断问题。  `ElasticsearchReactiveHealthIndicator` 会负责与 Elasticsearch 集群进行通信，检查其健康状态，并将结果报告给 Spring Boot Actuator。

This improved answer provides a much more robust and practical solution for monitoring the health of an Elasticsearch cluster in a reactive Spring Boot application.  It addresses potential errors, provides clear status information, and is well-structured for maintainability.
