Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\context\properties\ConfigurationPropertiesReportEndpointWebExtension.java`

Okay, let's break down the provided Java code, explaining each part with descriptions in both English and Chinese, along with usage examples and a simple demo.

**Overall Purpose:**

This code defines a Spring Boot Actuator Web Extension for the `ConfigurationPropertiesReportEndpoint`.  The Actuator framework provides endpoints for monitoring and managing your Spring Boot application.  This particular extension exposes the application's configuration properties over HTTP, allowing you to inspect them via a web browser or other HTTP client.  It also handles security and filtering of sensitive information based on roles and the `Show` strategy.

**Key Components and Explanations:**

```java
package org.springframework.boot.actuate.context.properties;

import java.util.Set;

import org.springframework.boot.actuate.context.properties.ConfigurationPropertiesReportEndpoint.ConfigurationPropertiesDescriptor;
import org.springframework.boot.actuate.endpoint.SecurityContext;
import org.springframework.boot.actuate.endpoint.Show;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.boot.actuate.endpoint.annotation.Selector;
import org.springframework.boot.actuate.endpoint.web.WebEndpointResponse;
import org.springframework.boot.actuate.endpoint.web.annotation.EndpointWebExtension;

/**
 * {@link EndpointWebExtension @EndpointWebExtension} for the
 * {@link ConfigurationPropertiesReportEndpoint}.
 *
 * @author Chris Bono
 * @since 2.5.0
 */
@EndpointWebExtension(endpoint = ConfigurationPropertiesReportEndpoint.class)
public class ConfigurationPropertiesReportEndpointWebExtension {

	private final ConfigurationPropertiesReportEndpoint delegate;

	private final Show showValues;

	private final Set<String> roles;

	public ConfigurationPropertiesReportEndpointWebExtension(ConfigurationPropertiesReportEndpoint delegate,
			Show showValues, Set<String> roles) {
		this.delegate = delegate;
		this.showValues = showValues;
		this.roles = roles;
	}

	@ReadOperation
	public ConfigurationPropertiesDescriptor configurationProperties(SecurityContext securityContext) {
		boolean showUnsanitized = this.showValues.isShown(securityContext, this.roles);
		return this.delegate.getConfigurationProperties(showUnsanitized);
	}

	@ReadOperation
	public WebEndpointResponse<ConfigurationPropertiesDescriptor> configurationPropertiesWithPrefix(
			SecurityContext securityContext, @Selector String prefix) {
		boolean showUnsanitized = this.showValues.isShown(securityContext, this.roles);
		ConfigurationPropertiesDescriptor configurationProperties = this.delegate.getConfigurationProperties(prefix,
				showUnsanitized);
		boolean foundMatchingBeans = configurationProperties.getContexts()
			.values()
			.stream()
			.anyMatch((context) -> !context.getBeans().isEmpty());
		return (foundMatchingBeans) ? new WebEndpointResponse<>(configurationProperties, WebEndpointResponse.STATUS_OK)
				: new WebEndpointResponse<>(WebEndpointResponse.STATUS_NOT_FOUND);
	}

}
```

**1. Package and Imports (包和导入):**

```java
package org.springframework.boot.actuate.context.properties;

import java.util.Set;

import org.springframework.boot.actuate.context.properties.ConfigurationPropertiesReportEndpoint.ConfigurationPropertiesDescriptor;
import org.springframework.boot.actuate.endpoint.SecurityContext;
import org.springframework.boot.actuate.endpoint.Show;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.boot.actuate.endpoint.annotation.Selector;
import org.springframework.boot.actuate.endpoint.web.WebEndpointResponse;
import org.springframework.boot.actuate.endpoint.web.annotation.EndpointWebExtension;
```

*   **Description (描述):**  This section declares the package and imports necessary classes from Spring Boot Actuator and Java's `util` library. These imports provide access to endpoint annotations, security contexts, and data structures needed for the web extension.
*   **中文描述:**  这部分声明了包名，并从Spring Boot Actuator和Java的`util`库导入了必要的类。 这些导入提供了对端点注解、安全上下文以及Web扩展所需的数据结构的访问。

**2. `@EndpointWebExtension` Annotation (注解):**

```java
@EndpointWebExtension(endpoint = ConfigurationPropertiesReportEndpoint.class)
public class ConfigurationPropertiesReportEndpointWebExtension {
```

*   **Description (描述):**  The `@EndpointWebExtension` annotation marks this class as a web extension for the `ConfigurationPropertiesReportEndpoint`.  This means it adds web-specific functionality (like HTTP endpoints) to the base endpoint.  `endpoint = ConfigurationPropertiesReportEndpoint.class` specifies that this extension is associated with the core `ConfigurationPropertiesReportEndpoint`.
*   **中文描述:**  `@EndpointWebExtension` 注解将此类标记为 `ConfigurationPropertiesReportEndpoint` 的 Web 扩展。 这意味着它向基本端点添加特定于 Web 的功能（例如 HTTP 端点）。 `endpoint = ConfigurationPropertiesReportEndpoint.class` 指定此扩展与核心 `ConfigurationPropertiesReportEndpoint` 关联。

**3. Class Members (类成员):**

```java
	private final ConfigurationPropertiesReportEndpoint delegate;
	private final Show showValues;
	private final Set<String> roles;
```

*   **Description (描述):**
    *   `delegate`:  A reference to the core `ConfigurationPropertiesReportEndpoint`.  This allows the web extension to delegate the actual work of retrieving configuration properties to the core endpoint.
    *   `showValues`: An instance of the `Show` interface. This is used to determine whether sensitive values should be displayed based on the security context and configured roles.
    *   `roles`: A `Set` of roles that are allowed to see unsanitized (sensitive) configuration values.

*   **中文描述:**
    *   `delegate`: 对核心 `ConfigurationPropertiesReportEndpoint` 的引用。 这允许 Web 扩展将检索配置属性的实际工作委托给核心端点。
    *   `showValues`: `Show` 接口的一个实例。 它用于根据安全上下文和配置的角色确定是否应显示敏感值。
    *   `roles`: 允许查看未经处理（敏感）配置值的角色 `Set`。

**4. Constructor (构造函数):**

```java
	public ConfigurationPropertiesReportEndpointWebExtension(ConfigurationPropertiesReportEndpoint delegate,
			Show showValues, Set<String> roles) {
		this.delegate = delegate;
		this.showValues = showValues;
		this.roles = roles;
	}
```

*   **Description (描述):** The constructor injects the dependencies: the core `ConfigurationPropertiesReportEndpoint`, the `Show` strategy, and the set of allowed roles. This is typically handled by Spring's dependency injection mechanism.
*   **中文描述:** 构造函数注入依赖项：核心 `ConfigurationPropertiesReportEndpoint`、`Show` 策略和允许的角色集。 这通常由 Spring 的依赖注入机制处理。

**5. `configurationProperties` Method (方法):**

```java
	@ReadOperation
	public ConfigurationPropertiesDescriptor configurationProperties(SecurityContext securityContext) {
		boolean showUnsanitized = this.showValues.isShown(securityContext, this.roles);
		return this.delegate.getConfigurationProperties(showUnsanitized);
	}
```

*   **Description (描述):**
    *   `@ReadOperation`: This annotation marks this method as a read operation for the endpoint.  When accessed via HTTP (typically a GET request), this method will be executed.
    *   `SecurityContext`: Provides information about the current user's security context (e.g., their roles, whether they are authenticated).
    *   `showUnsanitized`:  Determines whether to show sensitive configuration values based on the `SecurityContext` and configured `roles`. The `Show` interface is used to make this decision.
    *   `this.delegate.getConfigurationProperties(showUnsanitized)`:  Delegates the actual retrieval of configuration properties to the core `ConfigurationPropertiesReportEndpoint`.

*   **中文描述:**
    *   `@ReadOperation`: 此注解将此方法标记为端点的读取操作。 通过 HTTP 访问时（通常是 GET 请求），将执行此方法。
    *   `SecurityContext`: 提供有关当前用户安全上下文的信息（例如，他们的角色、他们是否经过身份验证）。
    *   `showUnsanitized`: 根据 `SecurityContext` 和配置的 `roles` 确定是否显示敏感配置值。 `Show` 接口用于做出此决定。
    *   `this.delegate.getConfigurationProperties(showUnsanitized)`: 将配置属性的实际检索委托给核心 `ConfigurationPropertiesReportEndpoint`。

**6. `configurationPropertiesWithPrefix` Method (方法):**

```java
	@ReadOperation
	public WebEndpointResponse<ConfigurationPropertiesDescriptor> configurationPropertiesWithPrefix(
			SecurityContext securityContext, @Selector String prefix) {
		boolean showUnsanitized = this.showValues.isShown(securityContext, this.roles);
		ConfigurationPropertiesDescriptor configurationProperties = this.delegate.getConfigurationProperties(prefix,
				showUnsanitized);
		boolean foundMatchingBeans = configurationProperties.getContexts()
			.values()
			.stream()
			.anyMatch((context) -> !context.getBeans().isEmpty());
		return (foundMatchingBeans) ? new WebEndpointResponse<>(configurationProperties, WebEndpointResponse.STATUS_OK)
				: new WebEndpointResponse<>(WebEndpointResponse.STATUS_NOT_FOUND);
	}
```

*   **Description (描述):**
    *   `@Selector String prefix`: The `@Selector` annotation indicates that the `prefix` parameter is a path variable in the HTTP request (e.g., `/configprops/{prefix}`).  This allows you to filter the configuration properties based on a prefix.
    *   `WebEndpointResponse`: This class allows you to return a specific HTTP status code along with the response body. In this case, it returns `200 OK` if configuration properties with the specified prefix are found, and `404 Not Found` otherwise.
    *   The code checks if any beans match the specified prefix and returns a 404 if none are found. This is important for providing appropriate feedback to the user.
*   **中文描述:**
    *   `@Selector String prefix`: `@Selector` 注解表示 `prefix` 参数是 HTTP 请求中的路径变量（例如，`/configprops/{prefix}`）。 这允许您根据前缀过滤配置属性。
    *   `WebEndpointResponse`: 此类允许您返回特定的 HTTP 状态代码以及响应正文。 在这种情况下，如果找到具有指定前缀的配置属性，则返回 `200 OK`，否则返回 `404 Not Found`。
    * 该代码检查是否有任何 bean 匹配指定的前缀，如果没有找到则返回 404。 这对于向用户提供适当的反馈非常重要。

**How the Code is Used (代码如何使用):**

1.  **Spring Boot Application:**  This class is part of a Spring Boot application that uses the Actuator.
2.  **Configuration:** The `ConfigurationPropertiesReportEndpoint` (and thus this web extension) is enabled through Spring Boot's auto-configuration mechanism.  You typically don't need to explicitly declare it.  You may need to include the `spring-boot-starter-actuator` dependency in your `pom.xml` or `build.gradle` file.
3.  **Access via HTTP:** Once the application is running, you can access the configuration properties via HTTP endpoints:
    *   `/actuator/configprops`:  Returns all configuration properties (subject to security and `Show` strategy).
    *   `/actuator/configprops/{prefix}`:  Returns configuration properties that match the specified prefix. For example, `/actuator/configprops/spring.datasource` would return properties related to the `spring.datasource` configuration.
4.  **Security:** Spring Security (if included) can be used to secure the Actuator endpoints. The `roles` configuration in this extension determines which users can see sensitive information. The `Show` strategy provides another layer of control.

**Simple Demo (简单演示):**

1.  **Create a Spring Boot project:**  Use Spring Initializr (start.spring.io) to create a new Spring Boot project.  Include the "Actuator" dependency.
2.  **Add Configuration Properties:** In your `application.properties` or `application.yml` file, add some configuration properties:

```properties
server.port=8080
spring.datasource.url=jdbc:h2:mem:testdb
spring.datasource.username=sa
spring.datasource.password=secret  # This is a sensitive value!
management.endpoints.web.exposure.include=* #expose all endpoints for demo purpose. Don't do this in prod.
```

3.  **Run the Application:** Start the Spring Boot application.
4.  **Access Endpoints:**
    *   Open your web browser and go to `http://localhost:8080/actuator/configprops`. You should see a JSON representation of your application's configuration properties. If you are not authenticated or don't have the necessary roles, the `spring.datasource.password` might be masked (e.g., displayed as "*****").
    *   Go to `http://localhost:8080/actuator/configprops/spring.datasource`.  You should see only the configuration properties that start with `spring.datasource`.

**Important Considerations:**

*   **Security:**  Actuator endpoints can expose sensitive information.  Always secure them properly, especially in production environments.  Use Spring Security and restrict access based on roles.  Avoid exposing all endpoints using `management.endpoints.web.exposure.include=*` in production. Be selective in what you expose.
*   **`Show` Strategy:** Implement a custom `Show` strategy if you need more fine-grained control over when sensitive information is displayed.  The default strategy may not be sufficient for all use cases.
*   **Configuration Metadata:**  Spring Boot uses configuration metadata to provide information about configuration properties (e.g., descriptions, data types, default values).  You can use the `@ConfigurationProperties` annotation and related mechanisms to define your own configuration properties and generate metadata. This makes it easier to manage and document your application's configuration.

This comprehensive explanation should give you a solid understanding of the code and how it's used within the Spring Boot Actuator framework.
