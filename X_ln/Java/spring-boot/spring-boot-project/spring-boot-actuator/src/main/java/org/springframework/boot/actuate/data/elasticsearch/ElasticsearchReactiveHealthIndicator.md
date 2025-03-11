Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\data\elasticsearch\ElasticsearchReactiveHealthIndicator.java`

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

package org.springframework.boot.actuate.data.elasticsearch;

import co.elastic.clients.elasticsearch._types.HealthStatus;
import co.elastic.clients.elasticsearch.cluster.HealthResponse;
import reactor.core.publisher.Mono;

import org.springframework.boot.actuate.health.AbstractReactiveHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.data.elasticsearch.client.elc.ReactiveElasticsearchClient;

/**
 * {@link HealthIndicator} for an Elasticsearch cluster using a
 * {@link ReactiveElasticsearchClient}.
 *
 * @author Brian Clozel
 * @author Aleksander Lech
 * @author Scott Frederick
 * @since 2.3.2
 */
public class ElasticsearchReactiveHealthIndicator extends AbstractReactiveHealthIndicator {

	private final ReactiveElasticsearchClient client;

	public ElasticsearchReactiveHealthIndicator(ReactiveElasticsearchClient client) {
		super("Elasticsearch health check failed");
		this.client = client;
	}

	@Override
	protected Mono<Health> doHealthCheck(Health.Builder builder) {
		return this.client.cluster().health((b) -> b).map((response) -> processResponse(builder, response));
	}

	private Health processResponse(Health.Builder builder, HealthResponse response) {
		if (!response.timedOut()) {
			HealthStatus status = response.status();
			builder.status((HealthStatus.Red == status) ? Status.OUT_OF_SERVICE : Status.UP);
			builder.withDetail("cluster_name", response.clusterName());
			builder.withDetail("status", response.status().jsonValue());
			builder.withDetail("timed_out", response.timedOut());
			builder.withDetail("number_of_nodes", response.numberOfNodes());
			builder.withDetail("number_of_data_nodes", response.numberOfDataNodes());
			builder.withDetail("active_primary_shards", response.activePrimaryShards());
			builder.withDetail("active_shards", response.activeShards());
			builder.withDetail("relocating_shards", response.relocatingShards());
			builder.withDetail("initializing_shards", response.initializingShards());
			builder.withDetail("unassigned_shards", response.unassignedShards());
			builder.withDetail("delayed_unassigned_shards", response.delayedUnassignedShards());
			builder.withDetail("number_of_pending_tasks", response.numberOfPendingTasks());
			builder.withDetail("number_of_in_flight_fetch", response.numberOfInFlightFetch());
			builder.withDetail("task_max_waiting_in_queue_millis", response.taskMaxWaitingInQueueMillis());
			builder.withDetail("active_shards_percent_as_number",
					Double.parseDouble(response.activeShardsPercentAsNumber()));
			builder.withDetail("unassigned_primary_shards", response.unassignedPrimaryShards());
			return builder.build();
		}
		return builder.down().build();
	}

}
```

**Explanation and Key Parts:**

This Java code defines a Spring Boot Actuator health indicator for Elasticsearch.  It checks the health of an Elasticsearch cluster and reports its status. Let's break down the key components:

1.  **`ElasticsearchReactiveHealthIndicator` Class:**
    *   This class extends `AbstractReactiveHealthIndicator`, which is a base class for creating reactive health indicators in Spring Boot. Reactive means it operates asynchronously using Reactor's `Mono` and `Flux`.

    *   `ReactiveElasticsearchClient client`:  This field holds a reference to a `ReactiveElasticsearchClient`, which is a client that allows you to interact with an Elasticsearch cluster in a non-blocking, reactive way.  It's injected into the constructor.

    *   `ElasticsearchReactiveHealthIndicator(ReactiveElasticsearchClient client)`:  The constructor takes a `ReactiveElasticsearchClient` as an argument (dependency injection). It also calls the `super()` constructor to set a default error message if the health check fails.

2.  **`doHealthCheck(Health.Builder builder)` Method:**
    *   This is the core method that performs the health check.  It's an override of a method in `AbstractReactiveHealthIndicator`.
    *   `this.client.cluster().health((b) -> b)`: This line uses the `ReactiveElasticsearchClient` to call the Elasticsearch cluster's `health` API.  The `(b) -> b` is a lambda expression that essentially passes in a default request builder.  It retrieves a `Mono<HealthResponse>`. `Mono` is a reactive type representing a single asynchronous value.
    *   `.map((response) -> processResponse(builder, response))`:  This line uses the `map` operator to transform the `Mono<HealthResponse>` into a `Mono<Health>`.  It takes the `HealthResponse` (the result from Elasticsearch) and passes it to the `processResponse` method.  The `processResponse` method then builds the `Health` object that Spring Boot Actuator will use.

3.  **`processResponse(Health.Builder builder, HealthResponse response)` Method:**
    *   This method takes the `HealthResponse` from Elasticsearch and builds a `Health` object that represents the status of the cluster.
    *   `if (!response.timedOut())`: Checks if the request to Elasticsearch timed out. If it timed out, it means there's likely a problem communicating with the cluster.
    *   `HealthStatus status = response.status()`: Gets the overall health status of the cluster from the `HealthResponse`. The status can be `Green`, `Yellow`, or `Red`.
    *   `builder.status((HealthStatus.Red == status) ? Status.OUT_OF_SERVICE : Status.UP)`:  Sets the health status in the `Health.Builder`. If the Elasticsearch status is `Red`, the health indicator reports `OUT_OF_SERVICE`. Otherwise (if it's `Green` or `Yellow`), it reports `UP`.
    *   `builder.withDetail(...)`:  Adds detailed information about the Elasticsearch cluster to the `Health` object.  This includes things like the cluster name, status, number of nodes, shard information, and pending tasks. These details can be very helpful for diagnosing problems.

    *   `builder.down().build()`: If the request to Elasticsearch timed out, this line sets the health status to `DOWN`.

**How the Code is Used (and a simple example):**

This health indicator is automatically used by Spring Boot Actuator if you have the following:

1.  Spring Boot Actuator dependency (`spring-boot-starter-actuator`).
2.  Spring Data Elasticsearch (with the Reactive client) dependencies configured in your project.
3.  A correctly configured `ReactiveElasticsearchClient` bean in your Spring context.  Spring Data Elasticsearch usually handles this automatically if you have the appropriate Elasticsearch connection properties in your `application.properties` or `application.yml` file.

Once you have these things in place, Spring Boot Actuator will automatically include the Elasticsearch health status in the `/actuator/health` endpoint.

**Example Configuration (application.yml):**

```yaml
spring:
  elasticsearch:
    uris: "http://localhost:9200"  # Replace with your Elasticsearch URL(s)
```

**How to see the health status:**

1.  Run your Spring Boot application.
2.  Access the `/actuator/health` endpoint (usually `http://localhost:8080/actuator/health`) in your browser or using `curl`.

The response will be a JSON document that includes an "elasticsearch" section, showing the status and details.  For example:

```json
{
  "status": "UP",
  "components": {
    "elasticsearch": {
      "status": "UP",
      "details": {
        "cluster_name": "docker-cluster",
        "status": "green",
        "timed_out": false,
        "number_of_nodes": 1,
        "number_of_data_nodes": 1,
        "active_primary_shards": 1,
        "active_shards": 1,
        "relocating_shards": 0,
        "initializing_shards": 0,
        "unassigned_shards": 0,
        "delayed_unassigned_shards": 0,
        "number_of_pending_tasks": 0,
        "number_of_in_flight_fetch": 0,
        "task_max_waiting_in_queue_millis": 0,
        "active_shards_percent_as_number": 100.0,
        "unassigned_primary_shards": 0
      }
    }
  }
}
```

**Key takeaways:**

*   **Actuator integration:** This is a standard Spring Boot Actuator health indicator, so it plugs into the existing health check framework.
*   **Reactive programming:** Uses Reactor's `Mono` for asynchronous, non-blocking operations.
*   **Detailed information:** Provides a lot of useful details about the Elasticsearch cluster's health, which is essential for troubleshooting.
*   **Automatic configuration:** Spring Data Elasticsearch typically handles the client configuration for you.
This revised response should provide a more complete and understandable explanation of the code and its usage. I've focused on clarity and providing a practical example.
