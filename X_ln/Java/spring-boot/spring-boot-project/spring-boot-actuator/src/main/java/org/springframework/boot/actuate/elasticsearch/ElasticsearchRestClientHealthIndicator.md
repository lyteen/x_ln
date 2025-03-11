Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\elasticsearch\ElasticsearchRestClientHealthIndicator.java`

Okay, I will restart and provide a breakdown of the Java code you provided.

```java
/*
 * Copyright 2012-2022 the original author or authors.
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

package org.springframework.boot.actuate.elasticsearch;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import org.apache.http.HttpStatus;
import org.apache.http.StatusLine;
import org.elasticsearch.client.Request;
import org.elasticsearch.client.Response;
import org.elasticsearch.client.RestClient;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.json.JsonParser;
import org.springframework.boot.json.JsonParserFactory;
import org.springframework.util.StreamUtils;

/**
 * {@link HealthIndicator} for an Elasticsearch cluster using a {@link RestClient}.
 *
 * @author Artsiom Yudovin
 * @author Brian Clozel
 * @author Filip Hrisafov
 * @since 2.7.0
 */
public class ElasticsearchRestClientHealthIndicator extends AbstractHealthIndicator {

	private static final String RED_STATUS = "red";

	private final RestClient client;

	private final JsonParser jsonParser;

	public ElasticsearchRestClientHealthIndicator(RestClient client) {
		super("Elasticsearch health check failed");
		this.client = client;
		this.jsonParser = JsonParserFactory.getJsonParser();
	}

	@Override
	protected void doHealthCheck(Health.Builder builder) throws Exception {
		Response response = this.client.performRequest(new Request("GET", "/_cluster/health/"));
		StatusLine statusLine = response.getStatusLine();
		if (statusLine.getStatusCode() != HttpStatus.SC_OK) {
			builder.down();
			builder.withDetail("statusCode", statusLine.getStatusCode());
			builder.withDetail("reasonPhrase", statusLine.getReasonPhrase());
			return;
		}
		try (InputStream inputStream = response.getEntity().getContent()) {
			doHealthCheck(builder, StreamUtils.copyToString(inputStream, StandardCharsets.UTF_8));
		}
	}

	private void doHealthCheck(Health.Builder builder, String json) {
		Map<String, Object> response = this.jsonParser.parseMap(json);
		String status = (String) response.get("status");
		if (RED_STATUS.equals(status)) {
			builder.outOfService();
		}
		else {
			builder.up();
		}
		builder.withDetails(response);
	}

}
```

**代码解释 (Code Explanation):**

This Java code defines a Spring Boot Actuator `HealthIndicator` for Elasticsearch. It checks the health of an Elasticsearch cluster by using the Elasticsearch `RestClient`.

1.  **`ElasticsearchRestClientHealthIndicator` 类:**

    *   `public class ElasticsearchRestClientHealthIndicator extends AbstractHealthIndicator`: 定义一个名为 `ElasticsearchRestClientHealthIndicator` 的类，它继承了 `AbstractHealthIndicator`。 `AbstractHealthIndicator` 是 Spring Boot Actuator 提供的一个抽象类，用于创建自定义的健康指示器。
    *   `private static final String RED_STATUS = "red";`: 定义一个静态常量 `RED_STATUS`，其值为 "red"。 Elasticsearch 集群健康状态为 "red" 通常表示存在问题。
    *   `private final RestClient client;`: 定义一个 `RestClient` 类型的私有 final 变量 `client`。 `RestClient` 是 Elasticsearch 提供的用于与 Elasticsearch 集群交互的客户端。
    *   `private final JsonParser jsonParser;`: 定义一个 `JsonParser` 类型的私有 final 变量 `jsonParser`。 `JsonParser` 用于解析 Elasticsearch 返回的 JSON 响应。
    *   `public ElasticsearchRestClientHealthIndicator(RestClient client)`:  构造函数，接收一个 `RestClient` 对象作为参数。 在构造函数中，初始化了 `client` 变量和 `jsonParser` 变量。`super("Elasticsearch health check failed");`  调用父类的构造器，设置了默认的健康检查失败消息。`this.jsonParser = JsonParserFactory.getJsonParser();` 通过`JsonParserFactory`获取一个 JSON 解析器实例。

2.  **`doHealthCheck` 方法:**

    *   `@Override protected void doHealthCheck(Health.Builder builder) throws Exception`: 重写了 `AbstractHealthIndicator` 的 `doHealthCheck` 方法。 此方法执行实际的健康检查逻辑。
    *   `Response response = this.client.performRequest(new Request("GET", "/_cluster/health/"));`:  使用 `RestClient` 发送一个 HTTP GET 请求到 Elasticsearch 集群的 `/_cluster/health/` 端点。此端点返回集群的健康信息。
    *   `StatusLine statusLine = response.getStatusLine();`: 从响应中获取 `StatusLine` 对象，其中包含 HTTP 状态码和原因短语。
    *   `if (statusLine.getStatusCode() != HttpStatus.SC_OK)`: 检查 HTTP 状态码是否为 `HttpStatus.SC_OK` (200)。 如果状态码不是 200，则表示请求失败。
    *   `builder.down(); builder.withDetail("statusCode", statusLine.getStatusCode()); builder.withDetail("reasonPhrase", statusLine.getReasonPhrase()); return;`:  如果请求失败，则将健康状态设置为 "down"，并添加状态码和原因短语作为详细信息。
    *   `try (InputStream inputStream = response.getEntity().getContent())`: 获取响应的内容流。 使用 try-with-resources 语句确保流在使用后被关闭。
    *   `doHealthCheck(builder, StreamUtils.copyToString(inputStream, StandardCharsets.UTF_8));`:  调用私有的 `doHealthCheck` 方法来处理响应内容。

3.  **`doHealthCheck` (private) 方法:**

    *   `private void doHealthCheck(Health.Builder builder, String json)`:  私有方法，接收一个 `Health.Builder` 对象和一个 JSON 字符串作为参数。
    *   `Map<String, Object> response = this.jsonParser.parseMap(json);`: 使用 `JsonParser` 将 JSON 字符串解析为一个 `Map` 对象。
    *   `String status = (String) response.get("status");`: 从 `Map` 对象中获取名为 "status" 的值。
    *   `if (RED_STATUS.equals(status))`: 检查 "status" 值是否等于 `RED_STATUS` ("red")。
    *   `builder.outOfService();`:  如果状态为 "red"，则将健康状态设置为 "outOfService"。 这表示集群正在运行但无法正常提供服务。
    *   `else { builder.up(); }`: 否则，将健康状态设置为 "up"。
    *   `builder.withDetails(response);`:  将整个响应 `Map` 对象作为详细信息添加到健康状态中。

**工作原理 (How it Works):**

该健康指示器通过以下步骤检查 Elasticsearch 集群的健康状况：

1.  创建一个 Elasticsearch `RestClient` 对象。
2.  发送一个 HTTP GET 请求到 Elasticsearch 集群的 `/_cluster/health/` 端点。
3.  检查 HTTP 状态码。 如果状态码不是 200，则认为集群已关闭。
4.  如果状态码是 200，则解析响应的 JSON 内容。
5.  检查响应中的 "status" 字段。 如果状态为 "red"，则认为集群已停止服务。
6.  根据集群的状态，设置健康状态为 "up"、"down" 或 "outOfService"，并将详细信息添加到健康状态中。

**使用示例 (Usage Example):**

To use this `HealthIndicator` in a Spring Boot application, you need to:

1.  Configure an Elasticsearch `RestClient` bean.
2.  Spring Boot will automatically detect the `ElasticsearchRestClientHealthIndicator` bean and include it in the health endpoint (`/actuator/health`).

**配置示例 (Configuration Example - Spring Boot):**

```java
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ElasticsearchConfig {

    @Bean(destroyMethod = "close")
    public RestClient restClient() {
        RestClientBuilder builder = RestClient.builder(
                // Add your Elasticsearch hosts here
                new org.apache.http.HttpHost("localhost", 9200, "http"));
        return builder.build();
    }
}
```

**中文解释:**

这段代码创建了一个 Spring Boot 配置类，其中定义了一个名为 `restClient` 的 Bean。 这个 Bean 创建了一个 `RestClient` 对象，用于连接到 Elasticsearch 集群。 你需要在 `new org.apache.http.HttpHost("localhost", 9200, "http")` 中替换为你实际的 Elasticsearch 主机地址、端口和协议。  `@Bean(destroyMethod = "close")`  确保在应用程序关闭时，`RestClient` 对象会被正确关闭，释放资源。

**简单 Demo:**

假设你有一个运行在 `localhost:9200` 的 Elasticsearch 集群.

1.  添加上述配置类到你的 Spring Boot 项目中.
2.  添加 `org.springframework.boot:spring-boot-starter-actuator` 依赖到你的 `pom.xml` 或者 `build.gradle` 文件中.

现在，当你访问 `/actuator/health` 端点时，你会看到 Elasticsearch 健康状态的信息。 例如:

```json
{
  "status": "UP",
  "details": {
    "elasticsearch": {
      "status": "UP",
      "details": {
        "cluster_name": "elasticsearch",
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
        "active_shards_percent_as_number": 100.0
      }
    }
  }
}
```

如果 Elasticsearch 集群的状态是 "red",  那么 `status` 将会是 "OUT_OF_SERVICE".
