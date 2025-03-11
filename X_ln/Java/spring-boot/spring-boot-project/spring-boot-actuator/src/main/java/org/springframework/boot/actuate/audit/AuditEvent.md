Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\audit\AuditEvent.java`

Okay, let's start fresh and break down this Java code.

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

import java.io.Serializable;
import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;

import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.ApplicationEventPublisherAware;
import org.springframework.util.Assert;

/**
 * A value object representing an audit event: at a particular time, a particular user or
 * agent carried out an action of a particular type. This object records the details of
 * such an event.
 * <p>
 * Users can inject a {@link AuditEventRepository} to publish their own events or
 * alternatively use Spring's {@link ApplicationEventPublisher} (usually obtained by
 * implementing {@link ApplicationEventPublisherAware}) to publish AuditApplicationEvents
 * (wrappers for AuditEvent).
 *
 * @author Dave Syer
 * @since 1.0.0
 * @see AuditEventRepository
 */
@JsonInclude(Include.NON_EMPTY)
public class AuditEvent implements Serializable {

	private final Instant timestamp;

	private final String principal;

	private final String type;

	private final Map<String, Object> data;

	/**
	 * Create a new audit event for the current time.
	 * @param principal the user principal responsible
	 * @param type the event type
	 * @param data the event data
	 */
	public AuditEvent(String principal, String type, Map<String, Object> data) {
		this(Instant.now(), principal, type, data);
	}

	/**
	 * Create a new audit event for the current time from data provided as name-value
	 * pairs.
	 * @param principal the user principal responsible
	 * @param type the event type
	 * @param data the event data in the form 'key=value' or simply 'key'
	 */
	public AuditEvent(String principal, String type, String... data) {
		this(Instant.now(), principal, type, convert(data));
	}

	/**
	 * Create a new audit event.
	 * @param timestamp the date/time of the event
	 * @param principal the user principal responsible
	 * @param type the event type
	 * @param data the event data
	 */
	public AuditEvent(Instant timestamp, String principal, String type, Map<String, Object> data) {
		Assert.notNull(timestamp, "'timestamp' must not be null");
		Assert.notNull(type, "'type' must not be null");
		this.timestamp = timestamp;
		this.principal = (principal != null) ? principal : "";
		this.type = type;
		this.data = Collections.unmodifiableMap(data);
	}

	private static Map<String, Object> convert(String[] data) {
		Map<String, Object> result = new HashMap<>();
		for (String entry : data) {
			int index = entry.indexOf('=');
			if (index != -1) {
				result.put(entry.substring(0, index), entry.substring(index + 1));
			}
			else {
				result.put(entry, null);
			}
		}
		return result;
	}

	/**
	 * Returns the date/time that the event was logged.
	 * @return the timestamp
	 */
	public Instant getTimestamp() {
		return this.timestamp;
	}

	/**
	 * Returns the user principal responsible for the event or an empty String if the
	 * principal is not available.
	 * @return the principal
	 */
	public String getPrincipal() {
		return this.principal;
	}

	/**
	 * Returns the type of event.
	 * @return the event type
	 */
	public String getType() {
		return this.type;
	}

	/**
	 * Returns the event data.
	 * @return the event data
	 */
	public Map<String, Object> getData() {
		return this.data;
	}

	@Override
	public String toString() {
		return "AuditEvent [timestamp=" + this.timestamp + ", principal=" + this.principal + ", type=" + this.type
				+ ", data=" + this.data + "]";
	}

}
```

**1. Class Declaration and Imports (类声明和导入)**

```java
package org.springframework.boot.actuate.audit;

import java.io.Serializable;
import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;

import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.ApplicationEventPublisherAware;
import org.springframework.util.Assert;

@JsonInclude(Include.NON_EMPTY)
public class AuditEvent implements Serializable {
  // ... class body ...
}
```

*   **`package org.springframework.boot.actuate.audit;`**:  This line specifies the package that this class belongs to. 包声明，表明类属于哪个包。
*   **`import ...;`**: These lines import necessary classes from the Java standard library and the Spring Framework. 导入依赖的类。
*   **`@JsonInclude(Include.NON_EMPTY)`**: This annotation from Jackson library indicates that properties with `null` or empty values will be excluded when serializing this object to JSON. Jackson注解，序列化为JSON时忽略空值。
*   **`public class AuditEvent implements Serializable`**: Defines the `AuditEvent` class, which implements the `Serializable` interface, meaning it can be converted to a stream of bytes and back.  这是一个公共类，实现了 `Serializable` 接口，可以被序列化。

**2. Instance Variables (实例变量)**

```java
private final Instant timestamp;
private final String principal;
private final String type;
private final Map<String, Object> data;
```

*   **`private final Instant timestamp;`**:  Stores the timestamp of the audit event. `Instant` is from `java.time` and represents a point in time. 审计事件发生的时间戳。
*   **`private final String principal;`**: Stores the user or system responsible for the event (e.g., username). 负责触发事件的用户或系统的标识。
*   **`private final String type;`**:  Stores the type of audit event (e.g., "login", "access_denied"). 审计事件的类型，比如 "登录" 或 "权限拒绝"。
*   **`private final Map<String, Object> data;`**: Stores additional information about the event as key-value pairs. 存储审计事件的额外信息。

**3. Constructors (构造函数)**

```java
public AuditEvent(String principal, String type, Map<String, Object> data) {
  this(Instant.now(), principal, type, data);
}

public AuditEvent(String principal, String type, String... data) {
  this(Instant.now(), principal, type, convert(data));
}

public AuditEvent(Instant timestamp, String principal, String type, Map<String, Object> data) {
  Assert.notNull(timestamp, "'timestamp' must not be null");
  Assert.notNull(type, "'type' must not be null");
  this.timestamp = timestamp;
  this.principal = (principal != null) ? principal : "";
  this.type = type;
  this.data = Collections.unmodifiableMap(data);
}
```

*   The class provides three constructors, allowing the `AuditEvent` to be created with varying degrees of detail.  类提供了多个构造函数，方便创建 `AuditEvent` 对象。
*   The first two constructors automatically set the timestamp to the current time using `Instant.now()`. 前两个构造函数自动设置时间戳为当前时间。
*   The third constructor allows specifying the timestamp explicitly. 第三个构造函数允许显式指定时间戳。
*   `Assert.notNull()`:  Spring's utility method to ensure that `timestamp` and `type` are not null.  使用 Spring 的 `Assert` 来确保 `timestamp` 和 `type` 不为 `null`。
*   `(principal != null) ? principal : ""`: Handles null `principal` values by setting it to an empty string.  如果 `principal` 为 `null`，则设置为空字符串。
*   `Collections.unmodifiableMap(data)`:  Creates an unmodifiable copy of the data map to prevent external modification of the `AuditEvent`'s internal state.  创建数据 Map 的不可修改副本，防止外部修改 `AuditEvent` 的内部状态。

**4. `convert` Method (转换方法)**

```java
private static Map<String, Object> convert(String[] data) {
  Map<String, Object> result = new HashMap<>();
  for (String entry : data) {
    int index = entry.indexOf('=');
    if (index != -1) {
      result.put(entry.substring(0, index), entry.substring(index + 1));
    } else {
      result.put(entry, null);
    }
  }
  return result;
}
```

*   This private method converts an array of strings (in the form "key=value" or "key") into a `Map<String, Object>`. 这个私有方法将字符串数组（"key=value" 或 "key" 格式）转换为 `Map<String, Object>`。
*   If a string contains "=", it's split into a key-value pair.  如果字符串包含 "=", 则将其拆分为键值对。
*   If a string doesn't contain "=", the entire string is used as the key, and the value is set to `null`.  如果字符串不包含 "=", 则整个字符串用作键，值设置为 `null`。

**5. Getter Methods (Getter 方法)**

```java
public Instant getTimestamp() {
  return this.timestamp;
}

public String getPrincipal() {
  return this.principal;
}

public String getType() {
  return this.type;
}

public Map<String, Object> getData() {
  return this.data;
}
```

*   These are standard getter methods to access the private instance variables. 标准的 getter 方法，用于访问私有实例变量。

**6. `toString` Method**

```java
@Override
public String toString() {
  return "AuditEvent [timestamp=" + this.timestamp + ", principal=" + this.principal + ", type=" + this.type + ", data=" + this.data + "]";
}
```

*   Overrides the `toString` method to provide a human-readable representation of the `AuditEvent` object.  重写 `toString` 方法，提供 `AuditEvent` 对象的可读表示。

**How it's used (如何使用)**

This `AuditEvent` class is used within a Spring Boot application to represent and record events for auditing purposes.  这个 `AuditEvent` 类在 Spring Boot 应用程序中用于表示和记录审计事件。

1.  **Creation:** Audit events are created when specific actions occur in the application (e.g., user login, data modification).  当应用程序中发生特定操作时（例如，用户登录、数据修改），会创建审计事件。

2.  **Publishing:** These events can be published using either:  这些事件可以通过以下方式发布：

    *   An `AuditEventRepository`:  A dedicated repository for storing audit events.  专门用于存储审计事件的仓库。
    *   The `ApplicationEventPublisher`:  A Spring mechanism for publishing application events.  Spring 提供的发布应用程序事件的机制。  Using this method means wrapping the `AuditEvent` in an `AuditApplicationEvent`.

3.  **Storage and Analysis:**  The audit events are then typically stored in a database or log file for later analysis and security monitoring.  然后，审计事件通常存储在数据库或日志文件中，以供以后分析和安全监控。

**Simple Demo (简单示例)**

```java
import org.springframework.boot.actuate.audit.AuditEvent;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class AuditEventDemo {

    public static void main(String[] args) {
        // Create an audit event with current time
        AuditEvent loginEvent = new AuditEvent("user123", "LOGIN_SUCCESS", Map.of("ipAddress", "192.168.1.1"));
        System.out.println("Login Event: " + loginEvent);

        //Create an audit event with specified time
        Instant eventTime = Instant.now().minusSeconds(60);
        Map<String, Object> data = new HashMap<>();
        data.put("resourceId", "resource456");
        data.put("operation", "READ");
        AuditEvent accessEvent = new AuditEvent(eventTime, "admin", "RESOURCE_ACCESS", data);
        System.out.println("Access Event: " + accessEvent);

        // Create an audit event with key-value pairs
        AuditEvent failedLogin = new AuditEvent("guest", "LOGIN_FAILED", "username=guest", "reason=invalid password");
        System.out.println("Failed Login: " + failedLogin);
    }
}
```

This example demonstrates how to create `AuditEvent` objects with different constructors and print them to the console. In a real application, you would typically publish these events using `AuditEventRepository` or `ApplicationEventPublisher`.

这个例子演示了如何使用不同的构造函数创建 `AuditEvent` 对象，并将它们打印到控制台。 在实际应用程序中，您通常会使用 `AuditEventRepository` 或 `ApplicationEventPublisher` 发布这些事件。
