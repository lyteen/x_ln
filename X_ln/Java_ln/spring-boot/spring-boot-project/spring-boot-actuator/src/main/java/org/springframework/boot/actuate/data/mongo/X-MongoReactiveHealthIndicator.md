Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\data\mongo\MongoReactiveHealthIndicator.java`

好的，让我们重新开始，并提供更智能的 MongoDB Reactive Health Indicator 代码，同时附带中文描述。

**1.  更加健壮的 MongoDB Reactive Health Indicator:**

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

package org.springframework.boot.actuate.data.mongo;

import com.mongodb.MongoException;
import org.bson.Document;
import reactor.core.publisher.Mono;

import org.springframework.boot.actuate.health.AbstractReactiveHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.ReactiveHealthIndicator;
import org.springframework.data.mongodb.core.ReactiveMongoTemplate;
import org.springframework.util.Assert;

import java.time.Duration;

/**
 * A {@link ReactiveHealthIndicator} for Mongo.
 *
 * @author Yulin Qin
 * @author Stephane Nicoll
 * @since 2.0.0
 */
public class MongoReactiveHealthIndicator extends AbstractReactiveHealthIndicator {

    private final ReactiveMongoTemplate reactiveMongoTemplate;

    public MongoReactiveHealthIndicator(ReactiveMongoTemplate reactiveMongoTemplate) {
        super("MongoDB health check failed"); // 更具体的默认错误信息
        Assert.notNull(reactiveMongoTemplate, "ReactiveMongoTemplate must not be null");
        this.reactiveMongoTemplate = reactiveMongoTemplate;
    }

    @Override
    protected Mono<Health> doHealthCheck(Health.Builder builder) {
        return reactiveMongoTemplate.executeCommand("{ hello: 1 }")
                .timeout(Duration.ofSeconds(5)) // 添加超时，防止无限等待
                .map(document -> up(builder, document))
                .onErrorResume(MongoException.class, ex -> down(builder, ex)) // 处理 MongoDB 异常
                .onErrorResume(Exception.class, ex -> unknown(builder, ex)); // 处理其他异常，更加健壮
    }

    private Health up(Health.Builder builder, Document document) {
        return builder.up()
                .withDetail("version", document.getString("version")) // 添加版本信息
                .withDetail("maxWireVersion", document.getInteger("maxWireVersion"))
                .build();
    }

    private Mono<Health> down(Health.Builder builder, MongoException ex) {
        return Mono.just(builder.down(ex).build());
    }

    private Mono<Health> unknown(Health.Builder builder, Exception ex) {
        return Mono.just(builder.unknown(ex).build());
    }
}
```

**代码描述 (中文):**

这个 `MongoReactiveHealthIndicator` 类是一个 Spring Boot Actuator 的健康指示器，用于检查 MongoDB 数据库的健康状况。

*   **构造函数:** 接受一个 `ReactiveMongoTemplate` 对象作为参数，用于执行 MongoDB 操作。
*   **`doHealthCheck` 方法:**  是健康检查的核心方法。 它执行以下步骤：
    1.  使用 `reactiveMongoTemplate.executeCommand("{ hello: 1 }")` 发送一个 "hello" 命令到 MongoDB 服务器。这是一个简单的命令，用于检查服务器是否可用。
    2.  **超时处理:**  `timeout(Duration.ofSeconds(5))`  设置了 5 秒的超时时间。如果 MongoDB 服务器在 5 秒内没有响应，则会抛出一个异常，防止健康检查无限期地挂起。
    3.  **成功处理:**  `map(document -> up(builder, document))` 如果命令执行成功，则调用 `up` 方法构建一个 "UP" 状态的 Health 对象，并包含 MongoDB 的版本信息和最大线协议版本 (maxWireVersion)。
    4.  **MongoDB 异常处理:**  `onErrorResume(MongoException.class, ex -> down(builder, ex))`  如果发生了 `MongoException`（例如连接错误、认证失败等），则调用 `down` 方法构建一个 "DOWN" 状态的 Health 对象，并包含异常信息。
    5.  **其他异常处理:**  `onErrorResume(Exception.class, ex -> unknown(builder, ex))`  处理其他类型的异常，并将健康状态设置为 "UNKNOWN"。  这使得健康检查器更加健壮，可以处理各种意外情况。
*   **`up` 方法:** 构建一个 "UP" 状态的 Health 对象，包含 MongoDB 的详细信息，例如版本号和最大线协议版本。
*   **`down` 方法:** 构建一个 "DOWN" 状态的 Health 对象，包含 MongoDB 异常信息。
*   **`unknown` 方法:** 构建一个 "UNKNOWN" 状态的 Health 对象，包含未知的异常信息。

**2.  简单演示 (中文):**

假设你有一个 Spring Boot 应用程序，并且配置了 MongoDB 连接。 为了使用这个健康指示器，你需要：

1.  **添加依赖:**  确保你的 `pom.xml` 或 `build.gradle` 文件中包含 Spring Boot Actuator 和 Spring Data MongoDB 的依赖。

2.  **配置 `ReactiveMongoTemplate`:**  确保你的应用程序中配置了一个 `ReactiveMongoTemplate` Bean。 Spring Data MongoDB 会自动配置它，只要你在 `application.properties` 或 `application.yml` 中正确配置了 MongoDB 连接信息。 例如:

    ```properties
    spring.data.mongodb.uri=mongodb://user:password@localhost:27017/mydatabase
    ```

3.  **健康检查端点:**  Spring Boot Actuator 默认会暴露 `/actuator/health` 端点。 你可以通过访问该端点来查看应用程序的健康状况，包括 MongoDB 的健康状况。

**访问 `/actuator/health` 端点后，你可能会看到类似以下的 JSON 响应 (例子):**

**MongoDB 健康状态正常:**

```json
{
  "status": "UP",
  "components": {
    "mongo": {
      "status": "UP",
      "details": {
        "version": "4.4.6",
        "maxWireVersion": 9
      }
    }
  }
}
```

**MongoDB 健康状态异常:**

```json
{
  "status": "DOWN",
  "components": {
    "mongo": {
      "status": "DOWN",
      "details": {
        "error": "com.mongodb.MongoTimeoutException: Timeout waiting for server"
      }
    }
  }
}
```

**总结 (中文):**

这个改进后的 `MongoReactiveHealthIndicator` 提供了更健壮的 MongoDB 健康检查，包括超时处理、MongoDB 异常处理和未知异常处理。 它还包含 MongoDB 的版本信息和最大线协议版本，方便你了解 MongoDB 的运行状态。 通过 Spring Boot Actuator 暴露的 `/actuator/health` 端点，你可以轻松地监控你的 MongoDB 数据库的健康状况。  这种方式能够帮助你更早地发现并解决 MongoDB 相关的问题，保证你的应用程序的稳定运行。
