Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\audit\AuditEventsEndpoint.java`

Okay, let's break down the Spring Boot Actuator `AuditEventsEndpoint` class, explaining each part with Chinese annotations and usage examples.

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

package org.springframework.boot.actuate.audit;

import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.List;

import org.springframework.boot.actuate.endpoint.OperationResponseBody;
import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.lang.Nullable;
import org.springframework.util.Assert;

/**
 * {@link Endpoint @Endpoint} to expose audit events.
 *
 * @author Andy Wilkinson
 * @since 2.0.0
 */
@Endpoint(id = "auditevents") // 声明这是一个 Actuator 端点，ID 为 "auditevents" (声明这是一个可以被 Actuator 暴露的端点，其ID为auditevents，可以通过/actuator/auditevents访问)
public class AuditEventsEndpoint {

	private final AuditEventRepository auditEventRepository; // 审计事件存储库 (审计事件仓库，用于存储和检索审计事件)

	public AuditEventsEndpoint(AuditEventRepository auditEventRepository) {
		Assert.notNull(auditEventRepository, "'auditEventRepository' must not be null"); // 确保 auditEventRepository 不为空 (断言auditEventRepository不能为空，如果为空则抛出异常)
		this.auditEventRepository = auditEventRepository;
	}

	@ReadOperation // 声明这是一个读取操作 (声明这是一个可以通过GET请求访问的读取操作)
	public AuditEventsDescriptor events(@Nullable String principal, @Nullable OffsetDateTime after,
			@Nullable String type) {
		List<AuditEvent> events = this.auditEventRepository.find(principal, getInstant(after), type); // 从存储库查找审计事件 (从仓库中根据条件查找审计事件)
		return new AuditEventsDescriptor(events); // 创建并返回一个 AuditEventsDescriptor (创建并返回审计事件描述符)
	}

	private Instant getInstant(OffsetDateTime offsetDateTime) {
		return (offsetDateTime != null) ? offsetDateTime.toInstant() : null; // 将 OffsetDateTime 转换为 Instant (如果 OffsetDateTime 不为空，则转换为 Instant 类型)
	}

	/**
	 * Description of an application's {@link AuditEvent audit events}.
	 */
	public static final class AuditEventsDescriptor implements OperationResponseBody { // 审计事件的描述符 (审计事件的描述类)

		private final List<AuditEvent> events; // 审计事件列表 (审计事件列表)

		private AuditEventsDescriptor(List<AuditEvent> events) {
			this.events = events; // 初始化审计事件列表 (初始化审计事件列表)
		}

		public List<AuditEvent> getEvents() {
			return this.events; // 返回审计事件列表 (返回审计事件列表)
		}

	}

}
```

**Explanation (解释):**

*   **`@Endpoint(id = "auditevents")`**:  This annotation marks the `AuditEventsEndpoint` class as a Spring Boot Actuator endpoint. The `id` attribute specifies the URL path for accessing this endpoint (e.g., `/actuator/auditevents`).  This means that you can access this endpoint via HTTP at the URL `/actuator/auditevents`.

    *   中文解释:  `@Endpoint` 注解将 `AuditEventsEndpoint` 类标记为 Spring Boot Actuator 端点。`id` 属性指定访问此端点的 URL 路径（例如，`/actuator/auditevents`）。 这意味着您可以通过 HTTP 在 URL `/actuator/auditevents` 访问此端点。

*   **`AuditEventRepository auditEventRepository`**:  This field holds a reference to an `AuditEventRepository` interface implementation. The repository is responsible for storing and retrieving audit events.

    *   中文解释:  此字段保存对 `AuditEventRepository` 接口实现的引用。 存储库负责存储和检索审计事件。

*   **`@ReadOperation`**: This annotation marks the `events` method as a read operation, making it accessible via an HTTP GET request.

    *   中文解释:  `@ReadOperation` 注解将 `events` 方法标记为读取操作，使其可以通过 HTTP GET 请求访问。

*   **`events(@Nullable String principal, @Nullable OffsetDateTime after, @Nullable String type)`**: This method handles the retrieval of audit events. It accepts optional parameters to filter the events by principal (user), date/time after which events should be retrieved, and event type.

    *   中文解释:  `events(@Nullable String principal, @Nullable OffsetDateTime after, @Nullable String type)` 方法处理审计事件的检索。 它接受可选参数以按负责人（用户）、应检索事件的日期/时间之后和事件类型来过滤事件。

*   **`AuditEventRepository.find(principal, getInstant(after), type)`**: This line calls the `find` method of the `AuditEventRepository` to retrieve the audit events based on the provided filtering criteria.

    *   中文解释:  此行调用 `AuditEventRepository` 的 `find` 方法，以根据提供的过滤条件检索审计事件。

*   **`AuditEventsDescriptor`**:  A simple class that encapsulates a list of `AuditEvent` objects for the response.

    *   中文解释:  一个简单的类，封装了用于响应的 `AuditEvent` 对象列表。

**How to Use (如何使用):**

1.  **Add the Actuator Dependency:** Make sure you have the Spring Boot Actuator dependency in your `pom.xml` or `build.gradle`.

    *   Maven:

    ```xml
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
    ```

    *   Gradle:

    ```gradle
    implementation 'org.springframework.boot:spring-boot-starter-actuator'
    ```

2.  **Implement `AuditEventRepository`:**  You need to provide an implementation of the `AuditEventRepository` interface.  This could be a simple in-memory implementation, or one that stores events in a database.  Here's a simple example:

    ```java
    import org.springframework.boot.actuate.audit.AuditEvent;
    import org.springframework.boot.actuate.audit.AuditEventRepository;
    import org.springframework.stereotype.Component;

    import java.time.Instant;
    import java.util.ArrayList;
    import java.util.List;

    @Component
    public class InMemoryAuditEventRepository implements AuditEventRepository {

        private final List<AuditEvent> events = new ArrayList<>();

        @Override
        public List<AuditEvent> find(String principal, Instant after, String type) {
            List<AuditEvent> filteredEvents = new ArrayList<>();
            for (AuditEvent event : events) {
                if ((principal == null || event.getPrincipal().equals(principal)) &&
                    (after == null || event.getTimestamp().isAfter(after)) &&
                    (type == null || event.getType().equals(type))) {
                    filteredEvents.add(event);
                }
            }
            return filteredEvents;
        }

        public void add(AuditEvent event) {
            events.add(event);
        }
    }
    ```

3.  **Configure Management Endpoints (Optional):** In your `application.properties` or `application.yml`, you can configure which Actuator endpoints are exposed.  To expose the `auditevents` endpoint, add the following:

    ```properties
    management.endpoints.web.exposure.include=auditevents,health,info
    ```

    or

    ```yaml
    management:
      endpoints:
        web:
          exposure:
            include: auditevents,health,info
    ```

4.  **Use the Endpoint:** Now you can access the audit events endpoint via HTTP. For example:

    *   `GET /actuator/auditevents`:  Gets all audit events.
    *   `GET /actuator/auditevents?principal=user1`: Gets audit events for user "user1".
    *   `GET /actuator/auditevents?type=AUTHENTICATION_SUCCESS`: Gets all successful authentication events.
    *   `GET /actuator/auditevents?after=2024-01-01T00:00:00Z`: Gets all audit events after January 1, 2024.

**Example Usage with a Custom Audit Event:**

```java
import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.Map;

@Service
public class MyService {

    private final ApplicationEventPublisher eventPublisher;

    public MyService(ApplicationEventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void doSomething(String user, String data) {
        // Perform some action
        System.out.println("Doing something with data: " + data + " for user: " + user);

        // Publish a custom audit event
        AuditEvent auditEvent = new AuditEvent(
            user,  // Principal (user performing the action)
            "MY_CUSTOM_EVENT", // Type of event
            Map.of("data", data, "action", "doSomething") // Event data
        );

        eventPublisher.publishEvent(auditEvent);
    }
}
```

In your controller or other component:

```java
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MyController {

    private final MyService myService;

    public MyController(MyService myService) {
        this.myService = myService;
    }

    @GetMapping("/doit")
    public String doIt(@RequestParam String user, @RequestParam String data) {
        myService.doSomething(user, data);
        return "OK";
    }
}
```

Now, after calling `/doit?user=testuser&data=someData`, you should be able to retrieve the `MY_CUSTOM_EVENT` audit event via the `/actuator/auditevents` endpoint.

**Chinese Summary (中文总结):**

`AuditEventsEndpoint` 是 Spring Boot Actuator 提供的一个端点，用于暴露应用程序的审计事件。 它可以根据用户、时间和事件类型来过滤审计事件。 要使用它，需要添加 Actuator 依赖，实现 `AuditEventRepository` 接口，配置管理端点，然后可以通过 HTTP 访问 `/actuator/auditevents` 端点来检索审计事件。 应用程序可以通过 `ApplicationEventPublisher` 发布自定义的审计事件。
