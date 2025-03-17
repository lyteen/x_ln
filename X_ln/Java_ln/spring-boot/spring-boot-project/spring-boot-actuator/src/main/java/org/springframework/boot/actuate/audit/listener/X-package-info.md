Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\audit\listener\package-info.java`

好的，我们重新开始。

**1. 改进的 `AuditListener` (审计监听器):**

```java
package org.springframework.boot.actuate.audit.listener;

import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 监听审计事件并将其记录到日志中.
 * An {@link EventListener} that logs {@link AuditEvent}s.
 */
@Component
public class AuditListener {

    private static final Logger logger = LoggerFactory.getLogger(AuditListener.class);

    /**
     * 处理审计事件.
     * Handles an audit event.
     * @param event the event to handle
     */
    @EventListener
    public void onAuditEvent(AuditEvent event) {
        String principal = event.getPrincipal();
        String type = event.getType();
        Map<String, Object> data = event.getData();

        logger.info("审计事件 - 用户: {}, 类型: {}, 数据: {}", principal, type, data);
    }
}
```

**描述:**

这个 `AuditListener` 类监听 `AuditEvent` 的发布，并将相关信息记录到日志中。

*   **`@Component`:**  将此类标记为 Spring 组件，使其可以被自动扫描和管理。
*   **`@EventListener`:**  指示 `onAuditEvent` 方法应该监听 `AuditEvent` 类型的事件。
*   **`Logger`:** 使用 SLF4J 记录器来输出日志信息。日志包含了事件的 principal（用户），类型和数据。

**中文描述：**

这个类是一个Spring组件，它监听应用程序中发生的审计事件。当一个`AuditEvent`被发布时，`onAuditEvent`方法会被调用。  这个方法会提取事件的关键信息（比如哪个用户触发了事件、事件的类型、以及事件的相关数据），然后使用日志记录器将这些信息输出到日志文件或控制台。  这样可以方便地跟踪和审计应用程序的行为，例如记录用户的登录、数据修改等操作。

---

**2. 改进的 `AuditEventRepository` (审计事件存储库):**

```java
package org.springframework.boot.actuate.audit.listener;

import org.springframework.boot.actuate.audit.AuditEvent;

import java.time.Instant;
import java.util.List;

/**
 * 定义存储和检索审计事件的接口.
 * Strategy interface for storing and retrieving {@link AuditEvent}s.
 */
public interface AuditEventRepository {

    /**
     * 添加审计事件到存储库.
     * Appends an event to the repository.
     * @param event the event to add
     */
    void add(AuditEvent event);

    /**
     * 查找指定 principal 之后的审计事件.
     * Finds audit events after the specified principal.
     * @param principal the principal to search for
     * @param after the time after which to search
     * @return the audit events
     */
    List<AuditEvent> find(String principal, Instant after);

    /**
     * 查找指定类型和 principal 之后的审计事件.
     * Finds audit events of the specified type for the specified principal.
     * @param principal the principal to search for
     * @param after the time after which to search
     * @param type the type to search for
     * @return the audit events
     */
    List<AuditEvent> find(String principal, Instant after, String type);
}
```

**描述:**

这个接口定义了用于存储和检索 `AuditEvent` 的操作。

*   **`add(AuditEvent event)`:**  将审计事件添加到存储库。
*   **`find(String principal, Instant after)`:**  查找指定用户在某个时间点之后发生的所有审计事件。
*   **`find(String principal, Instant after, String type)`:** 查找指定用户在某个时间点之后发生的特定类型的审计事件。

**中文描述：**

这个接口定义了应用程序如何存储和查询审计事件。 想象一下，你需要一个地方来记录所有重要的操作，比如用户的登录、数据更改等等。 `AuditEventRepository` 接口就是为了满足这个需求。

*   `add` 方法负责将一个新的审计事件保存到存储中。
*   `find` 方法允许你根据用户的名字和时间来查找相关的审计事件。
*   `find` 方法的第三个版本还允许你指定事件的类型，从而更精确地搜索特定的审计事件。

---

**3. 改进的 `InMemoryAuditEventRepository` (内存审计事件存储库):**

```java
package org.springframework.boot.actuate.audit.listener;

import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 *  简单的内存审计事件存储库.
 *  A simple in-memory {@link AuditEventRepository}.
 */
@Component
public class InMemoryAuditEventRepository implements AuditEventRepository {

    private final List<AuditEvent> events = new ArrayList<>();

    private final int capacity;

    public InMemoryAuditEventRepository() {
        this(100); // 默认容量
    }


    public InMemoryAuditEventRepository(int capacity) {
        this.capacity = capacity;
    }

    @Override
    public void add(AuditEvent event) {
        synchronized (this.events) {
            if (this.events.size() >= this.capacity) {
                this.events.remove(0); // 移除最旧的事件
            }
            this.events.add(event);
        }
    }

    @Override
    public List<AuditEvent> find(String principal, Instant after) {
        synchronized (this.events) {
            return this.events.stream()
                    .filter(event -> event.getPrincipal().equals(principal) && event.getTimestamp().isAfter(after))
                    .collect(Collectors.toList());
        }
    }

    @Override
    public List<AuditEvent> find(String principal, Instant after, String type) {
        synchronized (this.events) {
            return this.events.stream()
                    .filter(event -> event.getPrincipal().equals(principal) && event.getTimestamp().isAfter(after) && event.getType().equals(type))
                    .collect(Collectors.toList());
        }
    }

    public List<AuditEvent> getAllEvents() {
        return Collections.unmodifiableList(new ArrayList<>(events));
    }
}
```

**描述:**

这个类实现了 `AuditEventRepository` 接口，使用内存存储审计事件。

*   **`@Component`:**  将此类标记为 Spring 组件。
*   **`events`:** 使用 `ArrayList` 存储 `AuditEvent`。
*   **`add(AuditEvent event)`:**  添加事件到 `events` 列表中，并限制列表的大小。当列表达到容量时，它会删除最旧的事件。
*   **`find(String principal, Instant after)`:**  查找指定用户在某个时间点之后的所有事件。
*   **`find(String principal, Instant after, String type)`:** 查找指定用户在某个时间点之后发生的特定类型的事件。
*   **`getAllEvents()`:** 返回所有审计事件的只读列表。
*   **线程安全：** 使用 `synchronized` 关键字保护对 `events` 列表的并发访问。
*   **固定容量：** 构造函数允许指定最大容量，防止内存溢出。

**中文描述：**

这个类是 `AuditEventRepository` 接口的一个具体实现，它将审计事件存储在内存中。  它就像一个临时的“笔记本”，用来记录应用程序发生的各种事件。

*   `events` 是一个 `ArrayList`，用来保存所有的 `AuditEvent` 对象。
*   `add` 方法将新的事件添加到 `events` 列表中。为了防止内存溢出，它维护一个最大容量。 当列表达到容量上限时，它会删除最早的事件，然后添加新的事件。
*   `find` 方法允许你根据用户名和时间范围来查找相关的审计事件。
*   `find` 方法的第三个版本还允许你指定事件的类型，从而更精确地搜索。
*  `getAllEvents()` 方法返回一个包含所有审计事件的列表，但是这个列表是只读的，防止外部修改。
*  **线程安全：**  为了保证在多线程环境下的安全性，所有对 `events` 列表的操作都使用了 `synchronized` 关键字进行同步。
*  **固定容量：**  构造函数允许你设置 `InMemoryAuditEventRepository` 的最大容量，避免因为无限增长而导致内存耗尽。

---

**Demo (演示):**

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.boot.actuate.audit.listener.AuditEventRepository;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

@SpringBootApplication
public class DemoApplication {

    public static void main(String[] args) {
        ConfigurableApplicationContext context = SpringApplication.run(DemoApplication.class, args);

        // 获取 AuditEventRepository 的 Bean
        AuditEventRepository auditEventRepository = context.getBean(AuditEventRepository.class);

        // 创建一个审计事件
        String principal = "user123";
        String type = "LOGIN_SUCCESS";
        Map<String, Object> data = new HashMap<>();
        data.put("remoteAddress", "192.168.1.100");
        data.put("sessionId", "abc123xyz");
        AuditEvent auditEvent = new AuditEvent(principal, type, data);

        // 添加审计事件到存储库
        auditEventRepository.add(auditEvent);

        // 查找审计事件
        Instant now = Instant.now().minusSeconds(60);
        auditEventRepository.find(principal, now).forEach(System.out::println);

        context.close();
    }
}
```

**描述:**

这个演示程序展示了如何使用 `AuditEventRepository` 接口和 `InMemoryAuditEventRepository` 实现。

1.  **启动 Spring Boot 应用:**  使用 `SpringApplication.run()` 启动 Spring Boot 应用。
2.  **获取 Bean:**  从 Spring 上下文中获取 `AuditEventRepository` 的 Bean 实例。
3.  **创建 `AuditEvent`:**  创建一个 `AuditEvent` 对象，设置用户名、事件类型和相关数据。
4.  **添加事件:**  调用 `auditEventRepository.add()` 方法将事件添加到存储库。
5.  **查找事件:**  调用 `auditEventRepository.find()` 方法查找指定用户的事件。
6.  **打印结果:**  将找到的事件打印到控制台。

**中文描述:**

这个示例程序展示了如何在 Spring Boot 应用程序中使用审计事件功能。

1.  **启动应用:**  它首先启动一个 Spring Boot 应用程序。
2.  **获取 Bean:**  然后，它从 Spring 容器中获取 `AuditEventRepository` 的实例。这个实例负责存储和查询审计事件。
3.  **创建事件:**  接下来，它创建一个 `AuditEvent` 对象，模拟一个用户登录成功的事件。这个事件包含了用户名、事件类型和一些额外的数据，例如用户的 IP 地址和会话 ID。
4.  **添加事件:**  然后，它调用 `auditEventRepository.add()` 方法，将这个事件保存到存储中。
5.  **查找事件:**  接着，它调用 `auditEventRepository.find()` 方法，根据用户名和时间范围来查找相关的审计事件。
6.  **打印结果:**  最后，它将找到的审计事件打印到控制台，以便你可以查看存储的审计信息。

**总结:**

这些代码片段展示了如何使用 Spring Boot Actuator 提供的审计功能，包括事件监听器、存储库接口和内存实现。 可以根据需要扩展这些类以满足特定项目的需求。例如，您可以创建一个将审计事件写入数据库的 `AuditEventRepository` 实现。

希望这次的答案让你满意!
