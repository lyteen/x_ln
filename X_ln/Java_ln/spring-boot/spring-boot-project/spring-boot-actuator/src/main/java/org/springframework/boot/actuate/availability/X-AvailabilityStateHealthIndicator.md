Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\availability\AvailabilityStateHealthIndicator.java`

好的，我们重新开始，针对 `AvailabilityStateHealthIndicator` 类进行分析并提供更智能的改进建议。

**目标：** 旨在改进 `AvailabilityStateHealthIndicator` 类，使其更健壮、更灵活，并更好地适应不同的应用场景。 具体改进点如下：

*   **更灵活的状态映射：** 允许更复杂的条件状态映射，例如，基于状态属性进行映射。
*   **异步状态检查：** 支持异步的状态检查，避免阻塞健康检查线程。
*   **可配置的错误处理：** 提供可配置的错误处理机制，以便在状态检查失败时能够优雅地降级。
*   **自定义健康信息：** 允许在健康信息中包含更多关于状态的细节。

下面将逐步实现这些改进，并提供详细的解释和示例。

**1. 更灵活的状态映射**

当前的 `statusMappings` 仅支持 `AvailabilityState` 到 `Status` 的直接映射。 为了更灵活，我们可以引入一个 `Predicate<AvailabilityState>` 来判断是否应用某个 `Status`。

```java
package org.springframework.boot.actuate.availability;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health.Builder;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.availability.ApplicationAvailability;
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.util.Assert;

/**
 * A {@link HealthIndicator} that checks a specific {@link AvailabilityState} of the
 * application.
 *
 * @author Phillip Webb
 * @author Brian Clozel
 * @since 2.3.0
 */
public class AvailabilityStateHealthIndicator extends AbstractHealthIndicator {

    private final ApplicationAvailability applicationAvailability;

    private final Class<? extends AvailabilityState> stateType;

    private final Map<Predicate<AvailabilityState>, Status> statusMappings = new HashMap<>();

    /**
     * Create a new {@link AvailabilityStateHealthIndicator} instance.
     * @param <S> the availability state type
     * @param applicationAvailability the application availability
     * @param stateType the availability state type
     * @param statusMappings consumer used to set up the status mappings
     */
    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings) {
        Assert.notNull(applicationAvailability, "'applicationAvailability' must not be null");
        Assert.notNull(stateType, "'stateType' must not be null");
        Assert.notNull(statusMappings, "'statusMappings' must not be null");
        this.applicationAvailability = applicationAvailability;
        this.stateType = stateType;
        statusMappings.accept(this.statusMappings::put);
        // Removed assertAllEnumsMapped, as Predicate allows for more complex logic.
    }

    @Override
    protected void doHealthCheck(Builder builder) throws Exception {
        AvailabilityState state = getState(this.applicationAvailability);
        Status status = this.statusMappings.entrySet().stream()
                .filter(entry -> entry.getKey().test(state))
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse(this.statusMappings.get(alwaysTrue())); // Default status

        Assert.state(status != null, () -> "No mapping provided for state: " + state);
        builder.status(status);
    }

    private Predicate<AvailabilityState> alwaysTrue() {
        return state -> true;
    }

    /**
     * Return the current availability state. Subclasses can override this method if a
     * different retrieval mechanism is needed.
     * @param applicationAvailability the application availability
     * @return the current availability state
     */
    protected AvailabilityState getState(ApplicationAvailability applicationAvailability) {
        return applicationAvailability.getState(this.stateType);
    }

    /**
     * Callback used to add status mappings.
     *
     * @param <S> the availability state type
     */
    public interface StatusMappings<S extends AvailabilityState> {

        /**
         * Add the status that should be used if no explicit mapping is defined.
         * @param status the default status
         */
        default void addDefaultStatus(Status status) {
            add(state -> true, status);
        }

        /**
         * Add a new status mapping .
         * @param predicate the predicate for the availability state
         * @param status the mapped status
         */
        void add(Predicate<S> predicate, Status status);

        @SuppressWarnings("unchecked")
        default void add(S availabilityState, Status status) {
            add(state -> state.equals(availabilityState), status);
        }
    }
}
```

**代码解释:**

*   `statusMappings` 的类型从 `Map<AvailabilityState, Status>` 变为 `Map<Predicate<AvailabilityState>, Status>`。
*   `StatusMappings` 接口中的 `add` 方法现在接受一个 `Predicate<S>` 参数，用于定义状态匹配的条件。
*   `doHealthCheck` 方法使用 Stream API 查找第一个匹配状态的 `Status`。
*   添加了一个 `alwaysTrue()` 方法，用于表示默认的状态映射。
*   移除了 `assertAllEnumsMapped` 方法，因为 Predicate 允许更复杂的逻辑，不再需要确保所有枚举都被映射。

**示例用法:**

```java
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.boot.availability.ReadinessState;
import org.springframework.boot.actuate.health.Status;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.boot.availability.ApplicationAvailability;

@Configuration
public class AvailabilityConfiguration {

    @Bean
    public AvailabilityStateHealthIndicator readinessHealthIndicator(ApplicationAvailability applicationAvailability) {
        return new AvailabilityStateHealthIndicator(applicationAvailability, ReadinessState.class, mappings -> {
            mappings.add(ReadinessState.ACCEPTING_TRAFFIC, Status.UP);
            mappings.add(ReadinessState.REFUSING_TRAFFIC, Status.DOWN);
            mappings.add(state -> state instanceof ReadinessState && state.name().startsWith("ACCEPTING"), Status.UP); // 示例：匹配名称以 "ACCEPTING" 开头的状态
            mappings.addDefaultStatus(Status.UNKNOWN);
        });
    }
}
```

**描述:**  通过使用 `Predicate`，我们可以定义更灵活的状态映射规则。 在这个例子中，我们添加了一个规则，该规则匹配所有名称以 "ACCEPTING" 开头的 `ReadinessState`。 这允许我们基于状态属性（例如，状态名称）进行映射，而不仅仅是基于状态本身。

**2. 异步状态检查（Async State Check）**

为了防止健康检查阻塞主线程，特别是当状态检查涉及网络调用或耗时操作时，我们可以引入异步状态检查机制。

```java
package org.springframework.boot.actuate.availability;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health.Builder;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.availability.ApplicationAvailability;
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.util.Assert;

/**
 * A {@link HealthIndicator} that checks a specific {@link AvailabilityState} of the
 * application.
 *
 * @author Phillip Webb
 * @author Brian Clozel
 * @since 2.3.0
 */
public class AvailabilityStateHealthIndicator extends AbstractHealthIndicator {

    private final ApplicationAvailability applicationAvailability;

    private final Class<? extends AvailabilityState> stateType;

    private final Map<Predicate<AvailabilityState>, Status> statusMappings = new HashMap<>();

    private final Executor executor;

    /**
     * Create a new {@link AvailabilityStateHealthIndicator} instance.
     * @param <S> the availability state type
     * @param applicationAvailability the application availability
     * @param stateType the availability state type
     * @param statusMappings consumer used to set up the status mappings
     */
    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings) {
        this(applicationAvailability, stateType, statusMappings, Executors.newVirtualThreadPerTaskExecutor()); // 使用虚拟线程池作为默认执行器
    }


    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings, Executor executor) {
        Assert.notNull(applicationAvailability, "'applicationAvailability' must not be null");
        Assert.notNull(stateType, "'stateType' must not be null");
        Assert.notNull(statusMappings, "'statusMappings' must not be null");
        Assert.notNull(executor, "'executor' must not be null");
        this.applicationAvailability = applicationAvailability;
        this.stateType = stateType;
        this.executor = executor;
        statusMappings.accept(this.statusMappings::put);
    }


    @Override
    protected void doHealthCheck(Builder builder) throws Exception {
        CompletableFuture<AvailabilityState> future = CompletableFuture.supplyAsync(() -> getState(this.applicationAvailability), this.executor);

        try {
            AvailabilityState state = future.get(); // 获取异步结果，可以设置超时时间
            Status status = this.statusMappings.entrySet().stream()
                    .filter(entry -> entry.getKey().test(state))
                    .map(Map.Entry::getValue)
                    .findFirst()
                    .orElse(this.statusMappings.get(alwaysTrue())); // Default status

            Assert.state(status != null, () -> "No mapping provided for state: " + state);
            builder.status(status);
        } catch (Exception e) {
            builder.down(e); // 处理异步状态检查中的异常
        }
    }

    private Predicate<AvailabilityState> alwaysTrue() {
        return state -> true;
    }

    /**
     * Return the current availability state. Subclasses can override this method if a
     * different retrieval mechanism is needed.
     * @param applicationAvailability the application availability
     * @return the current availability state
     */
    protected AvailabilityState getState(ApplicationAvailability applicationAvailability) {
        return applicationAvailability.getState(this.stateType);
    }

    /**
     * Callback used to add status mappings.
     *
     * @param <S> the availability state type
     */
    public interface StatusMappings<S extends AvailabilityState> {

        /**
         * Add the status that should be used if no explicit mapping is defined.
         * @param status the default status
         */
        default void addDefaultStatus(Status status) {
            add(state -> true, status);
        }

        /**
         * Add a new status mapping .
         * @param predicate the predicate for the availability state
         * @param status the mapped status
         */
        void add(Predicate<S> predicate, Status status);

        @SuppressWarnings("unchecked")
        default void add(S availabilityState, Status status) {
            add(state -> state.equals(availabilityState), status);
        }
    }
}
```

**代码解释:**

*   添加了一个 `Executor` 类型的字段，用于执行异步状态检查。
*   提供了一个新的构造函数，允许配置 `Executor`。
*   `doHealthCheck` 方法现在使用 `CompletableFuture.supplyAsync` 异步地获取状态。
*   使用 `future.get()` 获取异步结果，可以设置超时时间。
*   添加了 `try-catch` 块，用于处理异步状态检查中的异常。

**示例用法:**

```java
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.boot.availability.ReadinessState;
import org.springframework.boot.actuate.health.Status;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.boot.availability.ApplicationAvailability;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Configuration
public class AvailabilityConfiguration {

    @Bean
    public AvailabilityStateHealthIndicator readinessHealthIndicator(ApplicationAvailability applicationAvailability) {
        ExecutorService executor = Executors.newFixedThreadPool(10); // 创建一个固定大小的线程池
        return new AvailabilityStateHealthIndicator(applicationAvailability, ReadinessState.class, mappings -> {
            mappings.add(ReadinessState.ACCEPTING_TRAFFIC, Status.UP);
            mappings.add(ReadinessState.REFUSING_TRAFFIC, Status.DOWN);
            mappings.addDefaultStatus(Status.UNKNOWN);
        }, executor);
    }
}
```

**描述:**  通过使用 `CompletableFuture` 和 `Executor`，我们可以异步地执行状态检查。 这可以防止健康检查阻塞主线程，提高应用程序的响应性。  在这个例子中，我们创建了一个固定大小的线程池来执行异步任务。 也可以使用 Spring 的 `TaskExecutor`。  默认情况下使用 `Executors.newVirtualThreadPerTaskExecutor()` 创建虚拟线程池。

**3. 可配置的错误处理（Configurable Error Handling）**

在状态检查失败时，我们可能希望能够优雅地降级，而不是直接抛出异常。 我们可以通过引入一个 `ErrorHandler` 接口来实现可配置的错误处理。

```java
package org.springframework.boot.actuate.availability;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.function.Function;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health.Builder;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.availability.ApplicationAvailability;
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.util.Assert;

/**
 * A {@link HealthIndicator} that checks a specific {@link AvailabilityState} of the
 * application.
 *
 * @author Phillip Webb
 * @author Brian Clozel
 * @since 2.3.0
 */
public class AvailabilityStateHealthIndicator extends AbstractHealthIndicator {

    private final ApplicationAvailability applicationAvailability;

    private final Class<? extends AvailabilityState> stateType;

    private final Map<Predicate<AvailabilityState>, Status> statusMappings = new HashMap<>();

    private final Executor executor;

    private final Function<Throwable, Status> errorHandler;

    /**
     * Create a new {@link AvailabilityStateHealthIndicator} instance.
     * @param <S> the availability state type
     * @param applicationAvailability the application availability
     * @param stateType the availability state type
     * @param statusMappings consumer used to set up the status mappings
     */
    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings) {
        this(applicationAvailability, stateType, statusMappings, Executors.newVirtualThreadPerTaskExecutor(), throwable -> Status.DOWN);
    }


    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings, Executor executor) {
        this(applicationAvailability, stateType, statusMappings, executor, throwable -> Status.DOWN);
    }

    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings, Executor executor, Function<Throwable, Status> errorHandler) {
        Assert.notNull(applicationAvailability, "'applicationAvailability' must not be null");
        Assert.notNull(stateType, "'stateType' must not be null");
        Assert.notNull(statusMappings, "'statusMappings' must not be null");
        Assert.notNull(executor, "'executor' must not be null");
        Assert.notNull(errorHandler, "'errorHandler' must not be null");

        this.applicationAvailability = applicationAvailability;
        this.stateType = stateType;
        this.executor = executor;
        this.errorHandler = errorHandler;
        statusMappings.accept(this.statusMappings::put);
    }


    @Override
    protected void doHealthCheck(Builder builder) throws Exception {
        CompletableFuture<AvailabilityState> future = CompletableFuture.supplyAsync(() -> getState(this.applicationAvailability), this.executor);

        try {
            AvailabilityState state = future.get(); // 获取异步结果，可以设置超时时间
            Status status = this.statusMappings.entrySet().stream()
                    .filter(entry -> entry.getKey().test(state))
                    .map(Map.Entry::getValue)
                    .findFirst()
                    .orElse(this.statusMappings.get(alwaysTrue())); // Default status

            Assert.state(status != null, () -> "No mapping provided for state: " + state);
            builder.status(status);
        } catch (Exception e) {
            Status errorStatus = this.errorHandler.apply(e);
            builder.status(errorStatus); // 使用错误处理程序来设置状态
            builder.withException(e);
        }
    }

    private Predicate<AvailabilityState> alwaysTrue() {
        return state -> true;
    }

    /**
     * Return the current availability state. Subclasses can override this method if a
     * different retrieval mechanism is needed.
     * @param applicationAvailability the application availability
     * @return the current availability state
     */
    protected AvailabilityState getState(ApplicationAvailability applicationAvailability) {
        return applicationAvailability.getState(this.stateType);
    }

    /**
     * Callback used to add status mappings.
     *
     * @param <S> the availability state type
     */
    public interface StatusMappings<S extends AvailabilityState> {

        /**
         * Add the status that should be used if no explicit mapping is defined.
         * @param status the default status
         */
        default void addDefaultStatus(Status status) {
            add(state -> true, status);
        }

        /**
         * Add a new status mapping .
         * @param predicate the predicate for the availability state
         * @param status the mapped status
         */
        void add(Predicate<S> predicate, Status status);

        @SuppressWarnings("unchecked")
        default void add(S availabilityState, Status status) {
            add(state -> state.equals(availabilityState), status);
        }
    }
}
```

**代码解释:**

*   添加了一个 `errorHandler` 类型的字段，用于处理状态检查中的异常。 `errorHandler` 是一个 `Function<Throwable, Status>`，它接受一个 `Throwable` 对象并返回一个 `Status` 对象。
*   提供了一个新的构造函数，允许配置 `errorHandler`。
*   在 `doHealthCheck` 方法的 `catch` 块中，使用 `errorHandler.apply(e)` 获取错误状态，并将其设置到 `builder` 中。

**示例用法:**

```java
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.boot.availability.ReadinessState;
import org.springframework.boot.actuate.health.Status;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.boot.availability.ApplicationAvailability;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Configuration
public class AvailabilityConfiguration {

    @Bean
    public AvailabilityStateHealthIndicator readinessHealthIndicator(ApplicationAvailability applicationAvailability) {
        ExecutorService executor = Executors.newFixedThreadPool(10); // 创建一个固定大小的线程池

        return new AvailabilityStateHealthIndicator(applicationAvailability, ReadinessState.class, mappings -> {
            mappings.add(ReadinessState.ACCEPTING_TRAFFIC, Status.UP);
            mappings.add(ReadinessState.REFUSING_TRAFFIC, Status.DOWN);
            mappings.addDefaultStatus(Status.UNKNOWN);
        }, executor, throwable -> {
            //  根据异常类型返回不同的状态
            if (throwable instanceof  java.util.concurrent.TimeoutException) {
                return Status.WARN; // 超时警告
            } else {
                return Status.DOWN; // 其他错误
            }
        });
    }
}
```

**描述:**  通过使用 `ErrorHandler`，我们可以根据不同的异常类型返回不同的状态。 在这个例子中，如果状态检查超时，则返回 `Status.WARN`，否则返回 `Status.DOWN`。 这允许我们更细粒度地控制健康检查的行为。

**4. 自定义健康信息 (Custom Health Information)**

有时，我们可能希望在健康信息中包含更多关于状态的细节。 我们可以通过将状态信息添加到 `Builder` 对象来实现这一点。

```java
package org.springframework.boot.actuate.availability;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.function.Function;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health.Builder;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.availability.ApplicationAvailability;
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.util.Assert;

/**
 * A {@link HealthIndicator} that checks a specific {@link AvailabilityState} of the
 * application.
 *
 * @author Phillip Webb
 * @author Brian Clozel
 * @since 2.3.0
 */
public class AvailabilityStateHealthIndicator extends AbstractHealthIndicator {

    private final ApplicationAvailability applicationAvailability;

    private final Class<? extends AvailabilityState> stateType;

    private final Map<Predicate<AvailabilityState>, Status> statusMappings = new HashMap<>();

    private final Executor executor;

    private final Function<Throwable, Status> errorHandler;

    private final boolean includeStateDetails;

    /**
     * Create a new {@link AvailabilityStateHealthIndicator} instance.
     * @param <S> the availability state type
     * @param applicationAvailability the application availability
     * @param stateType the availability state type
     * @param statusMappings consumer used to set up the status mappings
     */
    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings) {
        this(applicationAvailability, stateType, statusMappings, Executors.newVirtualThreadPerTaskExecutor(), throwable -> Status.DOWN, false);
    }


    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings, Executor executor) {
        this(applicationAvailability, stateType, statusMappings, executor, throwable -> Status.DOWN, false);
    }

    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings, Executor executor, Function<Throwable, Status> errorHandler) {
         this(applicationAvailability, stateType, statusMappings, executor, errorHandler, false);
    }


    public <S extends AvailabilityState> AvailabilityStateHealthIndicator(
            ApplicationAvailability applicationAvailability, Class<S> stateType,
            Consumer<StatusMappings<S>> statusMappings, Executor executor, Function<Throwable, Status> errorHandler, boolean includeStateDetails) {
        Assert.notNull(applicationAvailability, "'applicationAvailability' must not be null");
        Assert.notNull(stateType, "'stateType' must not be null");
        Assert.notNull(statusMappings, "'statusMappings' must not be null");
        Assert.notNull(executor, "'executor' must not be null");
        Assert.notNull(errorHandler, "'errorHandler' must not be null");

        this.applicationAvailability = applicationAvailability;
        this.stateType = stateType;
        this.executor = executor;
        this.errorHandler = errorHandler;
        this.includeStateDetails = includeStateDetails;
        statusMappings.accept(this.statusMappings::put);
    }


    @Override
    protected void doHealthCheck(Builder builder) throws Exception {
        CompletableFuture<AvailabilityState> future = CompletableFuture.supplyAsync(() -> getState(this.applicationAvailability), this.executor);

        try {
            AvailabilityState state = future.get(); // 获取异步结果，可以设置超时时间
            Status status = this.statusMappings.entrySet().stream()
                    .filter(entry -> entry.getKey().test(state))
                    .map(Map.Entry::getValue)
                    .findFirst()
                    .orElse(this.statusMappings.get(alwaysTrue())); // Default status

            Assert.state(status != null, () -> "No mapping provided for state: " + state);
            builder.status(status);

            if (this.includeStateDetails) {
                builder.withDetail("state", state.getClass().getSimpleName() + "." + state.name());  // 添加状态名称到健康信息
            }

        } catch (Exception e) {
            Status errorStatus = this.errorHandler.apply(e);
            builder.status(errorStatus); // 使用错误处理程序来设置状态
            builder.withException(e);
        }
    }

    private Predicate<AvailabilityState> alwaysTrue() {
        return state -> true;
    }

    /**
     * Return the current availability state. Subclasses can override this method if a
     * different retrieval mechanism is needed.
     * @param applicationAvailability the application availability
     * @return the current availability state
     */
    protected AvailabilityState getState(ApplicationAvailability applicationAvailability) {
        return applicationAvailability.getState(this.stateType);
    }

    /**
     * Callback used to add status mappings.
     *
     * @param <S> the availability state type
     */
    public interface StatusMappings<S extends AvailabilityState> {

        /**
         * Add the status that should be used if no explicit mapping is defined.
         * @param status the default status
         */
        default void addDefaultStatus(Status status) {
            add(state -> true, status);
        }

        /**
         * Add a new status mapping .
         * @param predicate the predicate for the availability state
         * @param status the mapped status
         */
        void add(Predicate<S> predicate, Status status);

        @SuppressWarnings("unchecked")
        default void add(S availabilityState, Status status) {
            add(state -> state.equals(availabilityState), status);
        }
    }
}
```

**代码解释:**

*   添加了一个 `includeStateDetails` 类型的字段，用于控制是否在健康信息中包含状态细节。
*   提供了一个新的构造函数，允许配置 `includeStateDetails`。
*   在 `doHealthCheck` 方法中，如果 `includeStateDetails` 为 `true`，则将状态名称添加到 `builder` 对象中。

**示例用法:**

```java
import org.springframework.boot.availability.AvailabilityState;
import org.springframework.boot.availability.ReadinessState;
import org.springframework.boot.actuate.health.Status;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.boot.availability.ApplicationAvailability;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Configuration
public class AvailabilityConfiguration {

    @Bean
    public AvailabilityStateHealthIndicator readinessHealthIndicator(ApplicationAvailability applicationAvailability) {
        ExecutorService executor = Executors.newFixedThreadPool(10); // 创建一个固定大小的线程池

        return new AvailabilityStateHealthIndicator(applicationAvailability, ReadinessState.class, mappings -> {
            mappings.add(ReadinessState.ACCEPTING_TRAFFIC, Status.UP);
            mappings.add(ReadinessState.REFUSING_TRAFFIC, Status.DOWN);
            mappings.addDefaultStatus(Status.UNKNOWN);
        }, executor, throwable -> Status.DOWN, true); // 包含状态细节
    }
}
```

**描述:**  通过设置 `includeStateDetails` 为 `true`，我们可以将状态名称添加到健康信息中。 这可以帮助我们更好地了解应用程序的当前状态。

**总结**

通过上述改进，`AvailabilityStateHealthIndicator` 类变得更加健壮、灵活，并且能够更好地适应不同的应用场景。  主要改进点包括：

*   **更灵活的状态映射：** 允许基于状态属性进行映射。
*   **异步状态检查：** 支持异步状态检查，避免阻塞健康检查线程。
*   **可配置的错误处理：** 提供可配置的错误处理机制，以便在状态检查失败时能够优雅地降级。
*   **自定义健康信息：** 允许在健康信息中包含更多关于状态的细节。

这些改进使得 `AvailabilityStateHealthIndicator` 类更加强大，并且能够更好地满足实际应用的需求。

**中文描述**

这段代码主要是为了提高Spring Boot Actuator中 `AvailabilityStateHealthIndicator` 类的灵活性和健壮性。 主要做了以下几点改进：

1.  **更灵活的状态映射 (状态映射更灵活)**:
    *   以前只能直接把某种应用状态 (`AvailabilityState`) 对应到一个健康状态 (`Status`)。
    *   现在，可以通过一个条件 (`Predicate`) 来判断是否应用某个健康状态。 这样就可以根据状态的属性 (比如状态名称) 来决定健康状态，而不仅仅是根据状态本身。
    *   **例子**: 可以设置一个规则，只要状态名称以 "ACCEPTING" 开头，就认为服务是健康的。

2.  **异步状态检查 (异步状态检查)**:
    *   如果检查应用状态需要耗时操作 (比如网络请求)， 可能会阻塞健康检查的线程，影响应用的响应速度。
    *   现在，状态检查可以在一个独立的线程中异步执行，不会阻塞主线程。
    *   **例子**: 可以创建一个线程池来专门执行状态检查任务。

3.  **可配置的错误处理 (可配置的错误处理)**:
    *   如果状态检查失败，以前会直接抛出异常。
    *   现在，可以配置一个错误处理函数 (`ErrorHandler`)，根据不同的异常类型返回不同的健康状态。 这样可以更优雅地处理错误，避免应用崩溃。
    *   **例子**: 如果状态检查超时，可以返回一个警告状态 (`WARN`)，而不是直接返回错误状态 (`DOWN`)。

4.  **自定义健康信息 (自定义健康信息)**:
    *   可以在健康信息中包含更多关于应用状态的细节，比如状态的名称。
    *   **例子**:  可以在健康信息中显示当前状态是 "ReadinessState.ACCEPTING_TRAFFIC"， 方便开发者了解应用的状态。

总而言之，这些改进让 `AvailabilityStateHealthIndicator` 类更加强大和灵活，可以更好地适应各种不同的应用场景，并提供更准确、更详细的健康信息。
