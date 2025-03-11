Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\audit\package-info.java`

好的，让我们重新开始，这次我将专注于提供 Spring Boot Actuator 审计相关的代码示例，并附带更详细的中文描述。

**1. 自定义审计事件 (Custom Audit Event):**

```java
package com.example.demo.audit;

import org.springframework.boot.actuate.audit.AuditEvent;

import java.util.Map;

public class CustomAuditEvent extends AuditEvent {

    public CustomAuditEvent(String principal, String type, Map<String, Object> data) {
        super(principal, type, data);
    }

    public CustomAuditEvent(String principal, String type, String... data) {
        super(principal, type, data);
    }
}
```

**描述:**

*   **`CustomAuditEvent`**:  这是一个自定义的审计事件类，继承自 Spring Boot Actuator 的 `AuditEvent`。  你可以使用这个类来创建具有特定类型和数据的审计事件。
*   **构造函数**:  提供了两个构造函数，一个接收 `Map<String, Object>` 类型的数据，另一个接收可变参数 `String... data`。 使用 `Map` 更灵活，可以存储各种类型的数据； 使用可变参数 `String... data` 更简单，适用于只记录字符串类型数据的场景.
*   **`principal`**:  表示执行操作的用户或主体。
*   **`type`**:  表示审计事件的类型，例如 "LOGIN_SUCCESS", "ORDER_CREATED" 等。
*   **`data`**:  包含与审计事件相关的附加信息，例如用户ID、订单ID等。

**演示:**

假设你想要记录用户登录成功的事件。 你可以这样创建一个 `CustomAuditEvent` 对象：

```java
import java.util.HashMap;
import java.util.Map;

public class AuditEventDemo {

    public static void main(String[] args) {
        String principal = "user123"; // 登录用户名
        String type = "LOGIN_SUCCESS"; // 事件类型
        Map<String, Object> data = new HashMap<>();
        data.put("userId", "user123");
        data.put("loginTime", System.currentTimeMillis());

        CustomAuditEvent auditEvent = new CustomAuditEvent(principal, type, data);

        System.out.println("审计事件主体: " + auditEvent.getPrincipal());
        System.out.println("审计事件类型: " + auditEvent.getType());
        System.out.println("审计事件数据: " + auditEvent.getData());

        // 你可以将这个 auditEvent 发布到 Spring 的 ApplicationEventPublisher，以便监听器可以处理它
        // 示例：applicationEventPublisher.publishEvent(auditEvent);
    }
}
```

这段代码创建了一个 `LOGIN_SUCCESS` 类型的审计事件，并包含了用户名和登录时间信息。  稍后你可以使用 `AuditEventRepository` 来存储这个事件.

---

**2. 自定义审计事件监听器 (Custom Audit Event Listener):**

```java
package com.example.demo.audit;

import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Component
public class CustomAuditEventListener {

    private static final Logger logger = LoggerFactory.getLogger(CustomAuditEventListener.class);

    @EventListener
    public void onAuditEvent(AuditEvent event) {
        logger.info("Received audit event: {}", event);
        // 在这里你可以对审计事件进行处理，例如记录到日志、发送通知等
        System.out.println("审计事件主体: " + event.getPrincipal());
        System.out.println("审计事件类型: " + event.getType());
        System.out.println("审计事件数据: " + event.getData());
    }
}
```

**描述:**

*   **`CustomAuditEventListener`**: 这是一个 Spring 组件，用于监听审计事件。
*   **`@EventListener`**:  这个注解表示 `onAuditEvent` 方法是一个事件监听器，它会监听所有 `AuditEvent` 类型的事件（包括你自定义的 `CustomAuditEvent`）。
*   **`onAuditEvent(AuditEvent event)`**:  这个方法会在接收到审计事件时被调用。  你可以在这个方法中对审计事件进行处理，例如：
    *   **记录到日志**:  使用 `logger.info()` 将审计事件的信息记录到日志文件中。
    *   **发送通知**:  发送邮件或短信通知管理员。
    *   **存储到数据库**:  将审计事件存储到数据库中，以便后续分析。

**演示:**

如果你在应用程序中发布了一个 `CustomAuditEvent`，那么 `CustomAuditEventListener` 的 `onAuditEvent` 方法就会被调用。  例如，在上面的 `AuditEventDemo` 示例中，如果你添加了 `applicationEventPublisher.publishEvent(auditEvent);`  那么，这个监听器就会接收到这个事件，并将事件信息打印到控制台和日志中。

**重要:**  要使这个监听器生效，你需要确保 `CustomAuditEventListener` 类被 Spring 容器管理。  这可以通过使用 `@Component` 注解来实现。

---

**3. 使用 `AuditEventRepository` 存储审计事件:**

```java
package com.example.demo.audit;

import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.boot.actuate.audit.AuditEventRepository;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Component
public class InMemoryAuditEventRepository implements AuditEventRepository {

    private final List<AuditEvent> events = new ArrayList<>();

    @Override
    public List<AuditEvent> find(String principal, String after, String type) {
        // 这里可以根据 principal, after, type 进行过滤
        // 为了简化，这里返回所有事件
        return Collections.unmodifiableList(events);
    }

    @Override
    public void add(AuditEvent event) {
        events.add(event);
    }
}
```

**描述:**

*   **`AuditEventRepository`**:  这是一个接口，用于存储和检索审计事件。 Spring Boot Actuator 提供了默认的实现，但你也可以提供自己的实现。
*   **`InMemoryAuditEventRepository`**:  这是一个简单的 `AuditEventRepository` 实现，它将审计事件存储在内存中的 `List` 中。 这只适用于演示目的，不适合生产环境。
*   **`add(AuditEvent event)`**:  这个方法用于添加审计事件到存储库中。
*   **`find(String principal, String after, String type)`**:  这个方法用于根据 `principal`（用户名）, `after` (日期时间之后), 和 `type`（事件类型）来检索审计事件。  在这个示例中，为了简化，它返回所有事件。

**演示:**

要使用 `InMemoryAuditEventRepository`，你需要将其声明为一个 Spring Bean (通过 `@Component`)，并在你的应用程序中注入它。 然后，你就可以使用它来存储审计事件。  例如:

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;

@Service
public class AuditService {

    @Autowired
    private ApplicationEventPublisher applicationEventPublisher;

    @Autowired
    private InMemoryAuditEventRepository auditEventRepository;

    public void logAuditEvent(AuditEvent event) {
        auditEventRepository.add(event); // 存储审计事件
        applicationEventPublisher.publishEvent(event); // 发布审计事件，以便监听器可以处理
    }
}
```

在这个示例中，`AuditService` 使用 `InMemoryAuditEventRepository` 来存储审计事件，并使用 `ApplicationEventPublisher` 来发布审计事件，以便 `CustomAuditEventListener` 可以监听并处理这些事件。

**重要:**

*   **生产环境存储**: 在生产环境中，你应该使用更持久的存储方案，例如数据库或日志文件。 Spring Boot Actuator 支持多种 `AuditEventRepository` 实现，例如基于关系型数据库的实现。
*   **安全性**:  确保你的审计日志受到保护，防止未经授权的访问。

**总结:**

这些示例展示了如何自定义审计事件、创建审计事件监听器以及使用 `AuditEventRepository` 存储审计事件。  你可以根据你的具体需求来定制这些组件，以满足你的应用程序的审计需求。

希望这些详细的解释和示例能够帮助你理解 Spring Boot Actuator 的审计功能。 如果你有任何其他问题，请随时提出。
