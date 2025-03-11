Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\availability\package-info.java`

好的，我们重新开始。我将为您提供更详细、更模块化的代码片段，并附带中文解释和简单的演示。这次我会更侧重于代码的可读性和实用性。

**1. 自定义就绪状态探测器 (Custom Readiness State Probe):**

```java
package org.springframework.boot.actuate.availability;

import org.springframework.boot.availability.AvailabilityChangeEvent;
import org.springframework.boot.availability.ReadinessState;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 自定义就绪状态探测器，用于演示如何控制应用程序的就绪状态。
 * Custom Readiness State Probe to demonstrate controlling the application's readiness state.
 */
@Component
public class CustomReadinessStateProbe {

    private final AtomicBoolean ready = new AtomicBoolean(false); // 初始状态为未就绪 Initial state is not ready

    /**
     * 设置应用程序为就绪状态。
     * Sets the application to the ready state.
     */
    public void setReady() {
        this.ready.set(true);
        AvailabilityChangeEvent.publish(this.getClass(), ReadinessState.ACCEPTING_TRAFFIC); // 发布就绪事件 Publish readiness event
    }

    /**
     * 设置应用程序为未就绪状态。
     * Sets the application to the not ready state.
     */
    public void setNotReady() {
        this.ready.set(false);
        AvailabilityChangeEvent.publish(this.getClass(), ReadinessState.REFUSING_TRAFFIC); // 发布未就绪事件 Publish not ready event
    }

    /**
     * 监听就绪状态变化事件。
     * Listens for readiness state change events.
     * @param event 就绪状态变化事件
     */
    @EventListener
    public void onAvailabilityChange(AvailabilityChangeEvent<ReadinessState> event) {
        ReadinessState state = event.getState();
        System.out.println("就绪状态变化: " + state); // 打印状态变化 Print state change
        // 这里可以根据状态执行相应的操作，例如更新监控指标
        // You can perform corresponding actions based on the state, such as updating monitoring metrics.
    }
}
```

**描述:**

这段代码创建了一个名为 `CustomReadinessStateProbe` 的组件，它允许你手动设置应用程序的就绪状态。它包含两个方法：`setReady()` 和 `setNotReady()`，分别用于将应用程序设置为就绪和未就绪状态。

*   **`@Component`**:  将该类标记为 Spring 组件，使其能够被自动扫描和管理。
*   **`AtomicBoolean ready`**:  使用 `AtomicBoolean` 来保证线程安全地更新就绪状态。
*   **`AvailabilityChangeEvent.publish()`**:  使用 `AvailabilityChangeEvent` 发布事件，通知 Spring Boot 应用程序就绪状态发生了变化。
*   **`@EventListener`**:  使用 `@EventListener` 注解来监听 `AvailabilityChangeEvent<ReadinessState>` 事件。

**如何使用 (使用示例):**

1.  **在你的 Spring Boot 应用程序中注入 `CustomReadinessStateProbe`：**

    ```java
    @Autowired
    private CustomReadinessStateProbe readinessStateProbe;
    ```

2.  **在适当的时候调用 `setReady()` 或 `setNotReady()` 方法：**

    ```java
    // 例如，在应用程序启动后，设置应用程序为就绪状态
    @PostConstruct
    public void initialize() {
        readinessStateProbe.setReady();
    }

    // 例如，在发生错误时，设置应用程序为未就绪状态
    public void handleError() {
        readinessStateProbe.setNotReady();
    }
    ```

    你可以通过 REST API、定时任务或其他任何适当的方式来调用这些方法。

**中文解释:**

这段代码的核心思想是，提供一种手动控制应用程序就绪状态的机制。就绪状态是指应用程序是否准备好接收流量。通过 `setReady()` 和 `setNotReady()` 方法，你可以告诉 Spring Boot 应用程序是否应该接受新的请求。 这对于在部署、维护或发生故障时控制流量非常有用。`AvailabilityChangeEvent` 负责通知 Spring Boot，而 `@EventListener` 允许你监听这些状态变化并执行相应的操作。

---

**2. 自定义存活状态探测器 (Custom Liveness State Probe):**

```java
package org.springframework.boot.actuate.availability;

import org.springframework.boot.availability.AvailabilityChangeEvent;
import org.springframework.boot.availability.LivenessState;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 自定义存活状态探测器，用于演示如何控制应用程序的存活状态。
 * Custom Liveness State Probe to demonstrate controlling the application's liveness state.
 */
@Component
public class CustomLivenessStateProbe {

    private final AtomicBoolean alive = new AtomicBoolean(true); // 初始状态为存活 Initial state is alive

    /**
     * 设置应用程序为存活状态。
     * Sets the application to the alive state.
     */
    public void setAlive() {
        this.alive.set(true);
        AvailabilityChangeEvent.publish(this.getClass(), LivenessState.CORRECT); // 发布存活事件 Publish liveness event
    }

    /**
     * 设置应用程序为死亡状态。
     * Sets the application to the dead state.
     */
    public void setDead() {
        this.alive.set(false);
        AvailabilityChangeEvent.publish(this.getClass(), LivenessState.BROKEN); // 发布死亡事件 Publish dead event
    }

    /**
     * 监听存活状态变化事件。
     * Listens for liveness state change events.
     * @param event 存活状态变化事件
     */
    @EventListener
    public void onAvailabilityChange(AvailabilityChangeEvent<LivenessState> event) {
        LivenessState state = event.getState();
        System.out.println("存活状态变化: " + state); // 打印状态变化 Print state change
        // 这里可以根据状态执行相应的操作，例如重启应用程序
        // You can perform corresponding actions based on the state, such as restarting the application.
    }
}
```

**描述:**

这段代码与 `CustomReadinessStateProbe` 类似，但它控制的是应用程序的存活状态 (liveness)。存活状态是指应用程序是否正在运行且健康。 如果应用程序的存活状态为 `BROKEN`，则意味着应用程序可能已经崩溃或进入了无法恢复的状态，需要重启。

*   **`LivenessState.CORRECT`**: 表示应用程序正常运行。
*   **`LivenessState.BROKEN`**:  表示应用程序已损坏或无法正常运行。

**如何使用 (使用示例):**

与 `CustomReadinessStateProbe` 的使用方法类似，你需要注入 `CustomLivenessStateProbe`，并在适当的时候调用 `setAlive()` 或 `setDead()` 方法。

**中文解释:**

存活状态探测对于监控应用程序的健康状况至关重要。如果应用程序的存活状态变为 `BROKEN`，运维人员可以通过监控系统自动重启应用程序，以保证服务的可用性。  例如，如果你的应用程序遇到了内存泄漏或死锁等问题，你可以调用 `setDead()` 方法，让 Kubernetes 等容器编排系统自动重启你的应用程序。

---

**3.  Spring Boot Actuator 配置 (Application.properties):**

```properties
# 开启健康检查端点 Enable health check endpoint
management.endpoints.web.exposure.include=health, liveness, readiness, custom

# 自定义健康检查端点路径 (可选) Customize health check endpoint path (optional)
management.endpoints.web.base-path=/actuator

# 启用就绪状态组 Enable readiness state group
management.health.readiness.enabled=true

# 启用存活状态组 Enable liveness state group
management.health.liveness.enabled=true
```

**描述:**

这些配置告诉 Spring Boot Actuator 暴露 `health`、`liveness`、`readiness` 和 `custom` 端点，并且启用就绪和存活状态组。  你可以通过访问这些端点来查看应用程序的健康状况和状态。

*   **`management.endpoints.web.exposure.include`**:  指定要暴露的 Actuator 端点。
*   **`management.endpoints.web.base-path`**:  设置 Actuator 端点的基本路径。
*   **`management.health.readiness.enabled`**:  启用就绪状态组。
*   **`management.health.liveness.enabled`**:  启用存活状态组。

**如何使用:**

将这些配置添加到你的 `application.properties` 或 `application.yml` 文件中。 然后，启动你的 Spring Boot 应用程序。 你可以通过访问 `/actuator/health`、`/actuator/liveness` 和 `/actuator/readiness` 端点来查看应用程序的健康状况和状态。

**中文解释:**

Actuator 是 Spring Boot 提供的一个监控和管理应用程序的框架。 通过配置 Actuator，你可以轻松地暴露应用程序的健康状况、状态、指标等信息。  这些信息可以被监控系统用来监控应用程序的运行状况，并在出现问题时发出警报。

**Summary (总结):**

通过以上代码示例，你就可以自定义 Spring Boot 应用程序的就绪和存活状态探测器，并使用 Actuator 暴露这些状态信息。这可以帮助你更好地监控和管理你的应用程序，并提高其可用性和可靠性。  记住，这只是一个简单的示例，你可以根据你的实际需求进行更复杂的定制。 例如，你可以将就绪和存活状态与数据库连接、消息队列或其他外部服务的状态关联起来。

希望这些改进后的代码示例和解释对您有所帮助！
