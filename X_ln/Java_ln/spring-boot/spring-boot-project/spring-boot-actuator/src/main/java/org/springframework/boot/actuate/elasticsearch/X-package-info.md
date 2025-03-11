Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\elasticsearch\package-info.java`

好的，我们重新开始。

/**
 * Copyright 2012-2019 the original author or authors.
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
 * See the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Actuator support for Elasticsearch.
 */
package org.springframework.boot.actuate.elasticsearch;

为了更好地理解和扩展 Spring Boot Actuator 对 Elasticsearch 的支持，我们可以从以下几个方面入手，并提供一些示例代码片段。

**1. 健康状况指标 (Health Indicator)**

健康状况指标用于检查 Elasticsearch 集群的健康状况。例如，检查集群是否可用，节点是否正常运行。

```java
package org.springframework.boot.actuate.elasticsearch;

import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.cluster.health.ClusterHealthRequest;
import org.elasticsearch.client.cluster.health.ClusterHealthResponse;
import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;
import org.springframework.util.Assert;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Elasticsearch health indicator.
 * 基于 Elasticsearch 集群的健康状况提供健康信息。
 */
public class ElasticsearchHealthIndicator extends AbstractHealthIndicator {

    private final RestHighLevelClient client;

    public ElasticsearchHealthIndicator(RestHighLevelClient client) {
        Assert.notNull(client, "Client must not be null");
        this.client = client;
    }

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        try {
            ClusterHealthRequest request = new ClusterHealthRequest();
            ClusterHealthResponse response = this.client.cluster().health(request, RequestOptions.DEFAULT);

            Status status = (response.getStatus().name().equals("GREEN") || response.getStatus().name().equals("YELLOW")) ? Status.UP : Status.DOWN;
            builder.status(status);

            Map<String, Object> details = new HashMap<>();
            details.put("cluster_name", response.getClusterName());
            details.put("status", response.getStatus());
            details.put("number_of_nodes", response.getNumberOfNodes());
            details.put("number_of_data_nodes", response.getNumberOfDataNodes());
            details.put("active_primary_shards", response.getActivePrimaryShards());
            details.put("active_shards", response.getActiveShards());
            details.put("relocating_shards", response.getRelocatingShards());
            details.put("initializing_shards", response.getInitializingShards());
            details.put("unassigned_shards", response.getUnassignedShards());

            builder.withDetails(details);

        } catch (IOException ex) {
            builder.down(ex);
        }
    }
}
```

**描述:** 这个 `ElasticsearchHealthIndicator` 类通过 `RestHighLevelClient` 与 Elasticsearch 集群通信，检索集群健康状况信息，并将其转换为 Spring Boot Actuator 的 `Health` 对象。`doHealthCheck` 方法是核心，它执行健康检查逻辑。

**示例用法 (中文):**

```java
// 在 Spring Boot 配置类中配置 Elasticsearch 连接
@Configuration
public class ElasticsearchConfig {

    @Bean
    public RestHighLevelClient client() {
        // 配置 Elasticsearch 连接信息，例如 host 和 port
        RestClientBuilder builder = RestClient.builder(new HttpHost("localhost", 9200, "http"));
        RestHighLevelClient client = new RestHighLevelClient(builder);
        return client;
    }

    @Bean
    public ElasticsearchHealthIndicator elasticsearchHealthIndicator(RestHighLevelClient client) {
        return new ElasticsearchHealthIndicator(client);
    }
}

// 现在访问 /actuator/health 端点，会包含 Elasticsearch 的健康信息。
```

**2. 信息指标 (Info Indicator)**

信息指标用于暴露 Elasticsearch 集群的元数据信息，例如版本号、插件列表等。

```java
package org.springframework.boot.actuate.elasticsearch;

import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.info.InfoRequest;
import org.elasticsearch.client.info.InfoResponse;
import org.springframework.boot.actuate.info.Info;
import org.springframework.boot.actuate.info.InfoContributor;
import org.springframework.util.Assert;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Elasticsearch info contributor.
 * 提供 Elasticsearch 集群的信息，例如版本号。
 */
public class ElasticsearchInfoContributor implements InfoContributor {

    private final RestHighLevelClient client;

    public ElasticsearchInfoContributor(RestHighLevelClient client) {
        Assert.notNull(client, "Client must not be null");
        this.client = client;
    }

    @Override
    public void contribute(Info.Builder builder) {
        try {
            InfoRequest request = new InfoRequest();
            InfoResponse response = this.client.info(request, RequestOptions.DEFAULT);

            Map<String, Object> details = new HashMap<>();
            details.put("cluster_name", response.getClusterName());
            details.put("version", response.getVersion().getNumber());
            details.put("lucene_version", response.getVersion().getLuceneVersion());

            builder.withDetail("elasticsearch", details);

        } catch (IOException ex) {
            builder.withDetail("elasticsearch", "Unable to retrieve info: " + ex.getMessage());
        }
    }
}
```

**描述:**  `ElasticsearchInfoContributor` 类使用 `RestHighLevelClient` 获取 Elasticsearch 集群的信息，并添加到 Actuator 的 `Info` 对象中。 `contribute` 方法用于添加信息。

**示例用法 (中文):**

```java
@Configuration
public class ElasticsearchConfig {

    // ... (client() bean 定义)

    @Bean
    public ElasticsearchInfoContributor elasticsearchInfoContributor(RestHighLevelClient client) {
        return new ElasticsearchInfoContributor(client);
    }
}

// 现在访问 /actuator/info 端点，会包含 Elasticsearch 的信息。
```

**3. 自定义指标 (Custom Metrics)**

除了健康状况和信息指标，还可以创建自定义指标来监控 Elasticsearch 集群的特定方面。 例如，监控索引的大小、查询延迟等。  这通常涉及到使用 Elasticsearch 的 Stats API。

```java
// (示例，需要更完善的实现)
package org.springframework.boot.actuate.elasticsearch;

import io.micrometer.core.instrument.MeterRegistry;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.indices.GetIndexRequest;
import org.elasticsearch.client.indices.GetIndexResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.util.Arrays;

@Component
public class ElasticsearchMetrics {

    @Autowired
    private RestHighLevelClient client;

    @Autowired
    private MeterRegistry registry;

    @PostConstruct
    public void bindToRegistry() {
        // 监控特定索引的大小 (示例)
        String[] indices = {"my_index"}; // 你需要监控的索引名称

        Arrays.stream(indices).forEach(indexName -> {
            registry.gauge("elasticsearch.index.size",
                    () -> {
                        try {
                            GetIndexRequest request = new GetIndexRequest(indexName);
                            GetIndexResponse response = client.indices().get(request, RequestOptions.DEFAULT);
                            // TODO: 从 response 中获取索引大小的信息并返回
                            // 这需要更复杂的逻辑，可能涉及到 _stats API 的使用。
                            return 0.0; // 替换为实际的索引大小
                        } catch (IOException e) {
                            e.printStackTrace();
                            return Double.NaN; // 表示无法获取
                        }
                    });
        });
    }
}
```

**描述:**  `ElasticsearchMetrics` 类使用 Micrometer 来注册自定义指标。`bindToRegistry` 方法在 bean 初始化后执行，它注册了一个 gauge 指标，用于监控 `my_index` 索引的大小。  **注意：** 这只是一个示例，你需要根据 Elasticsearch 的 Stats API 完善获取索引大小的逻辑。

**重要提示 (中文):**

*   **依赖:** 确保你的项目中包含了 Elasticsearch 的 Java High Level REST Client (`org.elasticsearch.client:elasticsearch-rest-high-level-client`) 和 Micrometer (`io.micrometer:micrometer-core`) 的依赖。
*   **配置:**  正确配置 `RestHighLevelClient` 连接到你的 Elasticsearch 集群。
*   **安全性:**  生产环境中，需要考虑 Elasticsearch 的安全性，例如身份验证和授权。
*   **异常处理:**  在与 Elasticsearch 交互时，妥善处理 `IOException` 异常。
*   **Micrometer:**  Micrometer 是一个指标收集的 Facade， 可以方便的将指标暴露给 Prometheus, Graphite 等监控系统.

这些示例提供了一个起点，可以帮助你开始使用 Spring Boot Actuator 监控 Elasticsearch 集群。你可以根据自己的需求定制这些指标，并添加更多的监控功能。 记住，清晰的监控对于维护 Elasticsearch 集群的健康和性能至关重要。
