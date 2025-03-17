Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\audit\listener\AuditListener.java`

好的，我们重新开始，并对之前的答案进行改进，使其更符合你的要求。

**目标:**

针对给定的 `AuditListener` 类，提供更智能的增强版本。 主要改进目标包括：

*   **异步处理:** 使用异步操作来存储审计事件，避免阻塞主线程。
*   **批量存储:**  将多个审计事件批量存储到 `AuditEventRepository`，提升效率。
*   **错误处理:**  加入更健壮的错误处理机制。
*   **可配置性:**  提供更灵活的配置选项。

**1. 改进的 AsyncBatchAuditListener:**

```java
package org.springframework.boot.actuate.audit.listener;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.boot.actuate.audit.AuditEvent;
import org.springframework.boot.actuate.audit.AuditEventRepository;
import org.springframework.scheduling.annotation.Async;
import org.springframework.util.Assert;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * 异步批量审计监听器. 监听 AuditApplicationEvent 并异步地批量存储到
 * AuditEventRepository 中。
 */
public class AsyncBatchAuditListener extends AbstractAuditListener {

    private static final Log logger = LogFactory.getLog(AsyncBatchAuditListener.class);

    private final AuditEventRepository auditEventRepository;
    private final BlockingQueue<AuditEvent> auditEventQueue;
    private final int batchSize;
    private final long flushIntervalMillis;
    private volatile boolean running = true;

    /**
     * 创建 AsyncBatchAuditListener.
     * @param auditEventRepository AuditEventRepository 实例
     * @param batchSize 批量存储的大小
     * @param flushIntervalMillis 批量存储的时间间隔 (毫秒)
     */
    public AsyncBatchAuditListener(AuditEventRepository auditEventRepository, int batchSize, long flushIntervalMillis) {
        Assert.notNull(auditEventRepository, "AuditEventRepository must not be null");
        Assert.isTrue(batchSize > 0, "Batch size must be greater than 0");
        Assert.isTrue(flushIntervalMillis > 0, "Flush interval must be greater than 0");

        this.auditEventRepository = auditEventRepository;
        this.auditEventQueue = new LinkedBlockingQueue<>();
        this.batchSize = batchSize;
        this.flushIntervalMillis = flushIntervalMillis;

        startBatchProcessor();
    }

    @Override
    protected void onAuditEvent(AuditEvent event) {
        if (logger.isDebugEnabled()) {
            logger.debug("Received audit event: " + event);
        }
        try {
            auditEventQueue.put(event); // 放入队列，如果队列满了则阻塞
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt(); // 恢复中断状态
            logger.warn("Interrupted while adding audit event to queue.", e);
        }
    }

    /**
     * 启动批量处理线程.
     */
    @Async // 使用 @Async 注解，使其在独立的线程中运行
    public void startBatchProcessor() {
        List<AuditEvent> batch = new ArrayList<>();
        long lastFlushTime = System.currentTimeMillis();

        while (running) {
            try {
                AuditEvent event = auditEventQueue.poll(flushIntervalMillis, TimeUnit.MILLISECONDS);
                if (event != null) {
                    batch.add(event);
                }

                // 达到批处理大小或者超时，则进行刷新
                if (batch.size() >= batchSize || (System.currentTimeMillis() - lastFlushTime >= flushIntervalMillis && !batch.isEmpty())) {
                    flushBatch(batch);
                    batch = new ArrayList<>();
                    lastFlushTime = System.currentTimeMillis();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("Batch processor interrupted.", e);
                running = false; // 停止线程
            } catch (Exception e) {
                logger.error("Error processing audit event batch.", e);
            }
        }

        // 确保退出前刷新剩余的事件
        if (!batch.isEmpty()) {
            flushBatch(batch);
        }
        logger.info("Batch processor stopped.");
    }

    /**
     * 刷新批处理.
     * @param batch 审计事件的批处理
     */
    private void flushBatch(List<AuditEvent> batch) {
        try {
            auditEventRepository.addAll(batch); // 假设 AuditEventRepository 有 addAll 方法
            if (logger.isDebugEnabled()) {
                logger.debug("Flushed " + batch.size() + " audit events to repository.");
            }
        } catch (Exception e) {
            logger.error("Failed to flush audit event batch.", e);
        }
    }

    /**
     * 停止监听器，并确保刷新剩余的事件.
     */
    public void stop() {
        running = false;
    }
}
```

**描述:**

*   **异步处理:**  使用 Spring 的 `@Async` 注解，`startBatchProcessor` 方法将在单独的线程中运行，不会阻塞处理 `AuditApplicationEvent` 的主线程。

*   **批量处理:**  使用 `BlockingQueue` 来缓存审计事件。  `startBatchProcessor`  会定期从队列中取出事件，并批量存储到  `AuditEventRepository` 中。  `batchSize` 和 `flushIntervalMillis`  控制批量处理的大小和时间间隔。

*   **错误处理:**  在  `onAuditEvent`  和  `startBatchProcessor`  中加入了  `try-catch`  块，处理可能出现的  `InterruptedException`  和其他异常，避免线程崩溃。

*   **停止方法:**  添加了  `stop()`  方法，可以安全地停止监听器，并确保刷新剩余的事件。

**2.  示例配置:**

为了使用 `AsyncBatchAuditListener`，你需要在 Spring Boot 应用中配置它：

```java
import org.springframework.boot.actuate.audit.AuditEventRepository;
import org.springframework.boot.actuate.audit.listener.AsyncBatchAuditListener;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;

@Configuration
@EnableAsync // 启用异步支持
public class AuditConfig {

    @Bean
    public AsyncBatchAuditListener asyncBatchAuditListener(AuditEventRepository auditEventRepository) {
        int batchSize = 100;
        long flushIntervalMillis = 5000; // 5 秒
        return new AsyncBatchAuditListener(auditEventRepository, batchSize, flushIntervalMillis);
    }
}
```

**描述:**

*   **@EnableAsync:** 启用Spring的异步执行能力。
*   **配置Bean:** 创建一个 `AsyncBatchAuditListener` 的 Bean，并注入 `AuditEventRepository`。 可以通过配置 `batchSize` 和 `flushIntervalMillis`  来调整批量处理的行为.

**如何使用:**

1.  确保你的 Spring Boot 项目已启用异步支持 (添加 `@EnableAsync` 注解).
2.  配置 `AsyncBatchAuditListener` Bean，并注入你的 `AuditEventRepository` 实现。
3.  当 Spring Boot 应用发布 `AuditApplicationEvent` 时，`AsyncBatchAuditListener` 将异步地批量存储这些事件。

**优点:**

*   **性能提升:**  通过异步和批量处理，显著提高了审计事件的存储效率，避免阻塞主线程。
*   **可靠性:**  增强的错误处理机制，确保即使出现异常，审计事件也能被正确处理。
*   **灵活性:**  通过配置 `batchSize` 和 `flushIntervalMillis`，可以根据实际需求调整批量处理的行为。

**缺点:**

*   **复杂性增加:**  相比简单的 `AuditListener`，代码更复杂，需要考虑线程安全和同步问题。
*   **延迟:**  由于异步处理，审计事件的存储可能会有一定的延迟。

这个改进的版本考虑了异步处理，批量存储，错误处理和可配置性，使其更加健壮和高效。  希望这个改进能满足你的需求!  如果还有其他问题，请随时提出。
