Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\dockerTest\java\org\springframework\boot\actuate\neo4j\Neo4jReactiveHealthIndicatorIntegrationTests.java`


```java
/*
 * Copyright 2012-2024 the original author or authors.
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

package org.springframework.boot.actuate.neo4j;

import java.time.Duration;

import org.junit.jupiter.api.Test;
import org.testcontainers.containers.Neo4jContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.autoconfigure.neo4j.Neo4jAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testsupport.container.TestImage;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for {@link Neo4jReactiveHealthIndicator}.
 *
 * @author Phillip Webb
 */
@SpringBootTest
@Testcontainers(disabledWithoutDocker = true)
class Neo4jReactiveHealthIndicatorIntegrationTests {

	// gh-33428

	@Container
	private static final Neo4jContainer<?> neo4jServer = TestImage.container(Neo4jContainer.class);

	@DynamicPropertySource
	static void neo4jProperties(DynamicPropertyRegistry registry) {
		registry.add("spring.neo4j.uri", neo4jServer::getBoltUrl);
		registry.add("spring.neo4j.authentication.username", () -> "neo4j");
		registry.add("spring.neo4j.authentication.password", neo4jServer::getAdminPassword);
	}

	@Autowired
	private Neo4jReactiveHealthIndicator healthIndicator;

	@Test
	void health() {
		Health health = this.healthIndicator.getHealth(true).block(Duration.ofSeconds(20));
		assertThat(health.getStatus()).isEqualTo(Status.UP);
		assertThat(health.getDetails()).containsEntry("edition", "community");
	}

	@Configuration(proxyBeanMethods = false)
	@ImportAutoConfiguration(Neo4jAutoConfiguration.class)
	@Import(Neo4jReactiveHealthIndicator.class)
	static class TestConfiguration {

	}

}
```

**总览 (Overview):**

This Java code is an integration test for the `Neo4jReactiveHealthIndicator`. It checks if the Neo4j database is up and running, and retrieves some details about it. It uses Spring Boot's Actuator framework to expose the health status. The test uses Testcontainers to spin up a real Neo4j instance for testing.

这段 Java 代码是一个集成测试，用于测试 `Neo4jReactiveHealthIndicator`。 它检查 Neo4j 数据库是否正常运行，并检索有关它的一些详细信息。 它使用 Spring Boot 的 Actuator 框架来暴露健康状态。 该测试使用 Testcontainers 启动一个真实的 Neo4j 实例进行测试。

**主要组成部分 (Key Components):**

1. **`@SpringBootTest`**:

   ```java
   @SpringBootTest
   ```

   *   **描述 (Description):** This annotation tells Spring Boot to run this class as a Spring Boot test. It sets up the Spring application context for testing.
   *   **解释 (Explanation in Chinese):**  这个注解告诉 Spring Boot 将这个类作为一个 Spring Boot 测试来运行。 它会设置 Spring 应用程序上下文以进行测试。
   *   **使用 (Usage):**  Used to bootstrap the entire Spring Boot application context for integration testing.  This allows you to test how different components work together.

2.  **`@Testcontainers`**:

    ```java
    @Testcontainers(disabledWithoutDocker = true)
    ```

    *   **描述 (Description):**  Enables Testcontainers integration.  `disabledWithoutDocker = true` means the test will only run if Docker is available.
    *   **解释 (Explanation in Chinese):** 启用 Testcontainers 集成。 `disabledWithoutDocker = true` 表示只有在 Docker 可用的情况下才会运行测试。
    *   **使用 (Usage):**  This is crucial for integration tests that require external services like databases.  It starts a Docker container with the Neo4j database.

3.  **`@Container`**:

    ```java
    @Container
    private static final Neo4jContainer<?> neo4jServer = TestImage.container(Neo4jContainer.class);
    ```

    *   **描述 (Description):** Declares a Testcontainers container to be managed during the test lifecycle. A `Neo4jContainer` is created, which will run a Neo4j database in a Docker container. `TestImage.container` is a utility to handle compatibility across different Testcontainers versions and potentially preconfigure the image.
    *   **解释 (Explanation in Chinese):** 声明一个 Testcontainers 容器，以便在测试生命周期中进行管理。 创建一个 `Neo4jContainer`，它将在 Docker 容器中运行 Neo4j 数据库。`TestImage.container` 是一个实用程序，用于处理不同 Testcontainers 版本之间的兼容性，并可能预配置图像。
    *   **使用 (Usage):**  Defines the Neo4j container that will be used for the test. Testcontainers handles starting and stopping the container.

4.  **`@DynamicPropertySource`**:

    ```java
    @DynamicPropertySource
    static void neo4jProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.neo4j.uri", neo4jServer::getBoltUrl);
        registry.add("spring.neo4j.authentication.username", () -> "neo4j");
        registry.add("spring.neo4j.authentication.password", neo4jServer::getAdminPassword);
    }
    ```

    *   **描述 (Description):**  This annotation allows you to dynamically set Spring properties for the test.  It's used here to configure the Neo4j connection using the dynamically allocated port from the Testcontainers Neo4j instance.
    *   **解释 (Explanation in Chinese):** 此注解允许您为测试动态设置 Spring 属性。 这里用于使用 Testcontainers Neo4j 实例动态分配的端口来配置 Neo4j 连接。
    *   **使用 (Usage):**  Sets the `spring.neo4j.uri`, `spring.neo4j.authentication.username`, and `spring.neo4j.authentication.password` properties so that Spring can connect to the Neo4j instance running in the Docker container.

5.  **`@Autowired`**:

    ```java
    @Autowired
    private Neo4jReactiveHealthIndicator healthIndicator;
    ```

    *   **描述 (Description):** Injects an instance of `Neo4jReactiveHealthIndicator` into the test class. Spring will automatically find and provide an instance of this bean.
    *   **解释 (Explanation in Chinese):**  将 `Neo4jReactiveHealthIndicator` 的实例注入到测试类中。 Spring 将自动查找并提供此 bean 的实例。
    *   **使用 (Usage):**  The `healthIndicator` is used to get the health status of the Neo4j database.

6.  **`@Test`**:

    ```java
    @Test
    void health() {
        Health health = this.healthIndicator.getHealth(true).block(Duration.ofSeconds(20));
        assertThat(health.getStatus()).isEqualTo(Status.UP);
        assertThat(health.getDetails()).containsEntry("edition", "community");
    }
    ```

    *   **描述 (Description):**  Marks the `health()` method as a JUnit test method.  This method gets the health status from the `Neo4jReactiveHealthIndicator`, checks if the status is "UP", and verifies that the details contain the "edition" key with the value "community".
    *   **解释 (Explanation in Chinese):**  将 `health()` 方法标记为 JUnit 测试方法。 此方法从 `Neo4jReactiveHealthIndicator` 获取健康状态，检查状态是否为“UP”，并验证详细信息是否包含键“edition”，值为“community”。
    *   **使用 (Usage):**  This is the actual test logic. It retrieves the health information and asserts that it's what is expected.  `block(Duration.ofSeconds(20))` waits for the reactive health check to complete, up to 20 seconds.

7.  **`@Configuration`, `@ImportAutoConfiguration`, `@Import`**:

    ```java
    @Configuration(proxyBeanMethods = false)
    @ImportAutoConfiguration(Neo4jAutoConfiguration.class)
    @Import(Neo4jReactiveHealthIndicator.class)
    static class TestConfiguration {

    }
    ```

    *   **描述 (Description):**  This inner class defines a Spring configuration for the test.  `@ImportAutoConfiguration(Neo4jAutoConfiguration.class)` imports the necessary Neo4j auto-configuration. `@Import(Neo4jReactiveHealthIndicator.class)` explicitly imports the `Neo4jReactiveHealthIndicator` class so it can be autowired. `proxyBeanMethods = false` optimizes the configuration class (minor performance improvement).
    *   **解释 (Explanation in Chinese):**  这个内部类定义了测试的 Spring 配置。 `@ImportAutoConfiguration(Neo4jAutoConfiguration.class)` 导入必要的 Neo4j 自动配置。 `@Import(Neo4jReactiveHealthIndicator.class)` 显式导入 `Neo4jReactiveHealthIndicator` 类，以便可以自动装配。 `proxyBeanMethods = false` 优化配置类（轻微的性能提升）。
    *   **使用 (Usage):** Sets up the Spring context with the beans needed for the test, specifically ensuring that Neo4j is configured and the health indicator is available.

**简而言之 (In Summary):**

This test uses Testcontainers to create a real Neo4j database instance within a Docker container. It then configures Spring to connect to this database and uses the `Neo4jReactiveHealthIndicator` to check its health. The test asserts that the database is up and running and verifies some of its properties.  This is a robust way to test the integration between your application and a Neo4j database.

这个测试使用 Testcontainers 在 Docker 容器中创建一个真实的 Neo4j 数据库实例。 然后，它配置 Spring 以连接到此数据库，并使用 `Neo4jReactiveHealthIndicator` 检查其健康状况。 该测试断言数据库已启动并正在运行，并验证其某些属性。 这是测试应用程序与 Neo4j 数据库之间集成的可靠方法。

**Example Usage Scenario:**

Imagine you're building a Spring Boot application that uses Neo4j to store graph data.  You want to make sure that your application can connect to Neo4j and that Neo4j is running correctly.  This integration test provides a way to automatically verify this. Every time you run your tests, it will spin up a Neo4j instance, run the health check, and then shut down the instance.  This gives you confidence that your application will work correctly when deployed.

想象一下，您正在构建一个使用 Neo4j 存储图形数据的 Spring Boot 应用程序。 您想确保您的应用程序可以连接到 Neo4j 并且 Neo4j 运行正常。 这个集成测试提供了一种自动验证这一点的方法。 每次运行测试时，它都会启动一个 Neo4j 实例，运行健康检查，然后关闭该实例。 这使您确信您的应用程序在部署时能够正常工作。
