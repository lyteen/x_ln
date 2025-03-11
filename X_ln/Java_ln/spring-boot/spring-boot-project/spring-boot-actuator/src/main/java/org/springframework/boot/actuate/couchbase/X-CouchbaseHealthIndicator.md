Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\couchbase\CouchbaseHealthIndicator.java`

Okay, let's try this again, focusing on providing a smarter or more feature-rich version of the original Java code, with explanations in both English and Chinese.  The goal is to improve upon the original without drastically changing its core purpose.  I'll also provide a simple demo.

**Enhanced `CouchbaseHealthIndicator`**

```java
package org.springframework.boot.actuate.couchbase;

import com.couchbase.client.core.diagnostics.DiagnosticsResult;
import com.couchbase.client.core.diagnostics.ServiceState;
import com.couchbase.client.java.Cluster;
import com.couchbase.client.java.manager.bucket.BucketManager;

import org.springframework.boot.actuate.health.AbstractHealthIndicator;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.util.Assert;

import java.util.Map;
import java.util.stream.Collectors;

/**
 * {@link HealthIndicator} for Couchbase, providing more detailed information.
 *
 * @author Eddú Meléndez
 * @author Stephane Nicoll
 * @author [Your Name]
 * @since 2.0.0
 */
public class CouchbaseHealthIndicator extends AbstractHealthIndicator {

    private final Cluster cluster;
    private final String bucketName; // Optional bucket name to check

    /**
     * Create an indicator with the specified {@link Cluster}.
     *
     * @param cluster the Couchbase Cluster
     * @since 2.0.6
     */
    public CouchbaseHealthIndicator(Cluster cluster) {
        this(cluster, null); // No specific bucket check
    }


    /**
     * Create an indicator with the specified {@link Cluster} and bucket to check.
     * @param cluster the Couchbase Cluster
     * @param bucketName the name of a bucket to specifically check.  If null, only cluster health is checked.
     */
    public CouchbaseHealthIndicator(Cluster cluster, String bucketName) {
        super("Couchbase health check failed");
        Assert.notNull(cluster, "'cluster' must not be null");
        this.cluster = cluster;
        this.bucketName = bucketName;
    }

    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        DiagnosticsResult diagnostics = this.cluster.diagnostics();

        // Check overall cluster health using the existing CouchbaseHealth class.
        new CouchbaseHealth(diagnostics).applyTo(builder);

        // Add more detailed information, including service states and optional bucket status.
        builder.withDetail("clusterStatus", diagnostics.state().toString());

        Map<String, String> serviceStates = diagnostics.endpoints().stream()
                .collect(Collectors.toMap(endpoint -> endpoint.serviceType().toString(),
                                          endpoint -> endpoint.state().toString()));

        builder.withDetail("serviceStates", serviceStates);


        // Optionally check bucket existence and health.
        if (this.bucketName != null && !this.bucketName.isEmpty()) {
            try {
                BucketManager bucketManager = this.cluster.buckets();
                boolean bucketExists = bucketManager.exists(this.bucketName); // Synchronous check

                if (bucketExists) {
                    builder.withDetail("bucketExists", true);
                    // Optionally, you could add more detailed bucket information here,
                    // like quota, durability settings, etc., if needed.
                } else {
                    builder.withDetail("bucketExists", false);
                    builder.down().withDetail("error", "Bucket '" + this.bucketName + "' does not exist."); // Indicate unhealthy state
                }
            } catch (Exception e) {
                builder.down().withDetail("error", "Error checking bucket '" + this.bucketName + "': " + e.getMessage()); // Indicate unhealthy state due to exception.
            }
        }
    }
}
```

**Key Improvements and Explanations:**

*   **Detailed Service States:** The code now includes details about the state of individual Couchbase services (e.g., KV, Query, Index).  This provides a more granular view of the cluster's health.
*   **Optional Bucket Check:** A constructor is added that takes an optional bucket name.  If a bucket name is provided, the health indicator will verify that the bucket exists.  If it doesn't, or if there's an error checking the bucket, the health indicator will report a "down" state.
*   **Clearer Error Reporting:** When a bucket doesn't exist or there's an error during the bucket check, the health indicator provides more informative error messages.
*   **Uses Synchronous Bucket Check:**  The `bucketManager.exists()` method is synchronous, making the health check more reliable.  Asynchronous checks might not complete before the health endpoint returns.
*   **Handles Exceptions:** The bucket check is wrapped in a `try-catch` block to handle potential exceptions during bucket manager operations.
*   **More Contextual Details:**  The `builder.withDetail()` calls add additional information to the health check response, making it easier to diagnose issues.

**Chinese Explanation (中文解释):**

这段代码改进了 Couchbase 的健康检查指示器 (Health Indicator)。

*   **更详细的服务状态:**  现在代码会包含每个 Couchbase 服务的状态信息 (例如：KV, Query, Index)。 这提供了更细致的集群健康视图。
*   **可选的 Bucket 检查:**  增加了一个构造函数，允许指定一个可选的 bucket 名称。 如果提供了 bucket 名称，健康检查指示器会验证该 bucket 是否存在。 如果不存在，或者在检查 bucket 时发生错误，健康检查指示器将报告 "down" 状态。
*   **更清晰的错误报告:**  当 bucket 不存在或者在 bucket 检查期间发生错误时，健康检查指示器会提供更详细的错误消息。
*   **使用同步 Bucket 检查:**  `bucketManager.exists()` 方法是同步的，这使得健康检查更可靠。 异步检查可能在健康端点返回之前没有完成。
*   **处理异常:**  Bucket 检查被 `try-catch` 块包围，以处理 bucket 管理器操作期间可能发生的异常。
*   **更多上下文细节:**  `builder.withDetail()` 调用向健康检查响应添加了额外的信息，使其更容易诊断问题。

**Simple Demo (简单演示):**

To use this, you would need a Spring Boot application with the Couchbase SDK and Spring Data Couchbase dependencies.  Here's a snippet of how you might configure it:

```java
import com.couchbase.client.java.Cluster;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.boot.actuate.couchbase.CouchbaseHealthIndicator;

@SpringBootApplication
public class DemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }

    @Bean
    public Cluster couchbaseCluster() {
        // Replace with your Couchbase connection details
        return Cluster.connect("couchbase://localhost", "username", "password");
    }

    @Bean
    public CouchbaseHealthIndicator couchbaseHealthIndicator(Cluster cluster) {
        // Check cluster and a specific bucket named "myBucket"
        return new CouchbaseHealthIndicator(cluster, "myBucket");

        // Only check the cluster itself
        //return new CouchbaseHealthIndicator(cluster);
    }
}
```

In this demo:

1.  We create a `Cluster` bean using your Couchbase connection details.  **Replace `"couchbase://localhost"`, `"username"`, and `"password"` with your actual Couchbase credentials.**

2.  We create a `CouchbaseHealthIndicator` bean, injecting the `Cluster` bean.  The example shows how to check either just the cluster or the cluster *and* a specific bucket (replace `"myBucket"` with the name of your bucket).

When you run your Spring Boot application, the `/actuator/health` endpoint will include the detailed Couchbase health information provided by the enhanced `CouchbaseHealthIndicator`.  If the bucket "myBucket" does not exist, or there are problems connecting, the health status will be `DOWN`.

This improved version provides a more comprehensive and robust way to monitor the health of your Couchbase cluster.  Remember to adapt the connection details and bucket name to your specific environment.
