Lan: `java` From`Java/spring-boot/spring-boot-project\spring-boot-actuator\src\main\java\org\springframework\boot\actuate\context\properties\ConfigurationPropertiesReportEndpoint.java`

好的，我们重新开始，对这段代码的关键部分进行解释，并生成小的代码片段，附带简短的中文描述。 同时，也会描述代码的用途，并提供简单的示例。

**总览 (Overview)**

这段 Java 代码定义了一个 Spring Boot Actuator 端点，用于公开应用程序中带有 `@ConfigurationProperties` 注解的 Bean 的配置属性。 重要的是，它还处理了敏感信息的屏蔽，并提供了显示或隐藏属性值的配置选项。

**1. `ConfigurationPropertiesReportEndpoint` 类**

```java
package org.springframework.boot.actuate.context.properties;

import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.context.ApplicationContextAware;

@Endpoint(id = "configprops")
public class ConfigurationPropertiesReportEndpoint implements ApplicationContextAware {

    // ... (其他成员变量和方法) ...
}
```

**描述:**
- 这是一个 Spring Boot Actuator 端点，通过 `@Endpoint(id = "configprops")` 注解声明。
- `ApplicationContextAware` 接口允许该类访问 Spring 应用程序上下文。
- 端点的 ID 是 "configprops"，可以通过 `/actuator/configprops` 访问（具体路径取决于 Actuator 的配置）。

**用途:**
- 公开应用程序的配置属性。

**示例:**
- 访问 `/actuator/configprops` 可以查看应用程序的配置属性信息。

**2. 构造函数和依赖注入**

```java
private final Sanitizer sanitizer;
private final Show showValues;
private ApplicationContext context;
private ObjectMapper objectMapper;

public ConfigurationPropertiesReportEndpoint(Iterable<SanitizingFunction> sanitizingFunctions, Show showValues) {
    this.sanitizer = new Sanitizer(sanitizingFunctions);
    this.showValues = showValues;
}

@Override
public void setApplicationContext(ApplicationContext context) {
    this.context = context;
}
```

**描述:**
- `Sanitizer` 用于清理敏感信息。
- `Show` 用于控制属性值的显示。
- `ApplicationContext` 提供对 Spring 应用程序上下文的访问。
- 构造函数使用依赖注入来获取 `SanitizingFunction` 和 `Show` 的实例。
- `setApplicationContext` 方法实现了 `ApplicationContextAware` 接口，允许 Spring 容器注入 `ApplicationContext`。

**用途:**
- `Sanitizer` 对属性值进行脱敏处理，保护敏感信息不被泄露。
- `Show` 决定是否显示属性的真实值，或者只显示已脱敏的值。
- `ApplicationContext` 用于访问 Spring 容器中的其他 Bean。

**3.  `@ReadOperation` 方法：公开配置属性**

```java
@ReadOperation
public ConfigurationPropertiesDescriptor configurationProperties() {
    boolean showUnsanitized = this.showValues.isShown(true);
    return getConfigurationProperties(showUnsanitized);
}

@ReadOperation
public ConfigurationPropertiesDescriptor configurationPropertiesWithPrefix(@Selector String prefix) {
    boolean showUnsanitized = this.showValues.isShown(true);
    return getConfigurationProperties(prefix, showUnsanitized);
}
```

**描述:**
- `@ReadOperation` 注解表明这些方法是 Actuator 端点的读取操作，可以通过 HTTP GET 请求访问。
- `configurationProperties()` 方法返回所有配置属性。
- `configurationPropertiesWithPrefix(@Selector String prefix)` 方法返回指定前缀的配置属性。
- `this.showValues.isShown(true)` 决定是否显示未脱敏的属性值。

**用途:**
- 允许用户通过 HTTP 请求获取应用程序的配置属性。
- 可以选择只获取特定前缀的属性，方便过滤。

**示例:**
- 访问 `/actuator/configprops` 获取所有配置属性。
- 访问 `/actuator/configprops/{prefix}` 获取指定前缀的配置属性，例如 `/actuator/configprops/spring.datasource`。

**4.  `getConfigurationProperties` 方法：核心逻辑**

```java
private ConfigurationPropertiesDescriptor getConfigurationProperties(ApplicationContext context,
        Predicate<ConfigurationPropertiesBean> beanFilterPredicate, boolean showUnsanitized) {
    ObjectMapper mapper = getObjectMapper();
    Map<String, ContextConfigurationPropertiesDescriptor> contexts = new HashMap<>();
    ApplicationContext target = context;

    while (target != null) {
        contexts.put(target.getId(), describeBeans(mapper, target, beanFilterPredicate, showUnsanitized));
        target = target.getParent();
    }
    return new ConfigurationPropertiesDescriptor(contexts);
}
```

**描述:**
- 遍历应用程序上下文及其父上下文，获取所有带有 `@ConfigurationProperties` 注解的 Bean。
- 使用 `describeBeans` 方法描述每个 Bean 的属性。
- 将结果封装到 `ConfigurationPropertiesDescriptor` 对象中。

**用途:**
- 收集所有配置属性，并进行整理。
- 支持多级上下文，例如父子上下文的配置属性会合并在一起。

**5. `describeBeans` 方法：描述 Bean 属性**

```java
private ContextConfigurationPropertiesDescriptor describeBeans(ObjectMapper mapper, ApplicationContext context,
        Predicate<ConfigurationPropertiesBean> beanFilterPredicate, boolean showUnsanitized) {
    Map<String, ConfigurationPropertiesBean> beans = ConfigurationPropertiesBean.getAll(context);
    Map<String, ConfigurationPropertiesBeanDescriptor> descriptors = beans.values()
            .stream()
            .filter(beanFilterPredicate)
            .collect(Collectors.toMap(ConfigurationPropertiesBean::getName,
                    (bean) -> describeBean(mapper, bean, showUnsanitized)));
    return new ContextConfigurationPropertiesDescriptor(descriptors,
            (context.getParent() != null) ? context.getParent().getId() : null);
}
```

**描述:**
- 获取应用程序上下文中所有 `ConfigurationPropertiesBean`。
- 使用 `beanFilterPredicate` 过滤 Bean。
- 使用 `describeBean` 方法描述每个 Bean 的属性。
- 将结果封装到 `ContextConfigurationPropertiesDescriptor` 对象中。

**用途:**
- 获取指定上下文中的配置属性 Bean，并提取其描述信息。

**6. `describeBean` 方法：描述单个 Bean**

```java
private ConfigurationPropertiesBeanDescriptor describeBean(ObjectMapper mapper, ConfigurationPropertiesBean bean,
        boolean showUnsanitized) {
    String prefix = bean.getAnnotation().prefix();
    Map<String, Object> serialized = safeSerialize(mapper, bean.getInstance(), prefix);
    Map<String, Object> properties = sanitize(prefix, serialized, showUnsanitized);
    Map<String, Object> inputs = getInputs(prefix, serialized, showUnsanitized);
    return new ConfigurationPropertiesBeanDescriptor(prefix, properties, inputs);
}
```

**描述:**
- 获取 Bean 的前缀。
- 使用 `safeSerialize` 方法将 Bean 序列化为 Map。
- 使用 `sanitize` 方法清理敏感信息。
- 使用 `getInputs` 方法获取输入的源信息 (origin)。
- 将结果封装到 `ConfigurationPropertiesBeanDescriptor` 对象中。

**用途:**
- 描述单个配置属性 Bean 的信息，包括属性值、脱敏状态以及来源信息。

**7. `safeSerialize` 方法：安全序列化**

```java
@SuppressWarnings({ "unchecked" })
private Map<String, Object> safeSerialize(ObjectMapper mapper, Object bean, String prefix) {
    try {
        return new HashMap<>(mapper.convertValue(bean, Map.class));
    }
    catch (Exception ex) {
        return new HashMap<>(Collections.singletonMap("error", "Cannot serialize '" + prefix + "'"));
    }
}
```

**描述:**
- 使用 Jackson `ObjectMapper` 将 Bean 序列化为 Map。
- 捕获序列化过程中可能发生的异常，避免程序崩溃。

**用途:**
- 将配置属性 Bean 转换为 Map，方便后续处理。
- 确保序列化过程的安全性，避免因序列化失败导致程序中断。

**8. `sanitize` 方法：清理敏感信息**

```java
@SuppressWarnings("unchecked")
private Map<String, Object> sanitize(String prefix, Map<String, Object> map, boolean showUnsanitized) {
    map.forEach((key, value) -> {
        String qualifiedKey = getQualifiedKey(prefix, key);
        if (value instanceof Map) {
            map.put(key, sanitize(qualifiedKey, (Map<String, Object>) value, showUnsanitized));
        }
        else if (value instanceof List) {
            map.put(key, sanitize(qualifiedKey, (List<Object>) value, showUnsanitized));
        }
        else {
            map.put(key, sanitizeWithPropertySourceIfPresent(qualifiedKey, value, showUnsanitized));
        }
    });
    return map;
}
```

**描述:**
- 递归遍历 Map，对每个属性值进行清理。
- 如果属性值是 Map 或 List，则递归调用 `sanitize` 方法。
- 如果属性值是基本类型，则调用 `sanitizeWithPropertySourceIfPresent` 方法。

**用途:**
- 对配置属性进行递归脱敏，确保所有敏感信息都被处理。

**9. `sanitizeWithPropertySourceIfPresent` 方法：使用 PropertySource 进行清理**

```java
private Object sanitizeWithPropertySourceIfPresent(String qualifiedKey, Object value, boolean showUnsanitized) {
    ConfigurationPropertyName currentName = getCurrentName(qualifiedKey);
    ConfigurationProperty candidate = getCandidate(currentName);
    PropertySource<?> propertySource = getPropertySource(candidate);
    if (propertySource != null) {
        SanitizableData data = new SanitizableData(propertySource, qualifiedKey, value);
        return this.sanitizer.sanitize(data, showUnsanitized);
    }
    SanitizableData data = new SanitizableData(null, qualifiedKey, value);
    return this.sanitizer.sanitize(data, showUnsanitized);
}
```

**描述:**
- 获取属性的 `PropertySource`。
- 如果 `PropertySource` 存在，则使用 `Sanitizer` 对属性值进行清理。
- 否则，直接使用 `Sanitizer` 进行清理。

**用途:**
- 利用 `PropertySource` 信息进行更精确的脱敏处理。
- 允许用户自定义 `SanitizingFunction`，实现特定的脱敏策略。

**10. `getInputs` 方法：获取输入的源信息**

```java
@SuppressWarnings("unchecked")
private Map<String, Object> getInputs(String prefix, Map<String, Object> map, boolean showUnsanitized) {
    Map<String, Object> augmented = new LinkedHashMap<>(map);
    map.forEach((key, value) -> {
        String qualifiedKey = getQualifiedKey(prefix, key);
        if (value instanceof Map) {
            augmented.put(key, getInputs(qualifiedKey, (Map<String, Object>) value, showUnsanitized));
        }
        else if (value instanceof List) {
            augmented.put(key, getInputs(qualifiedKey, (List<Object>) value, showUnsanitized));
        }
        else {
            augmented.put(key, applyInput(qualifiedKey, showUnsanitized));
        }
    });
    return augmented;
}
```

**描述:**
- 递归遍历 Map，为每个属性添加源信息。
- 如果属性值是 Map 或 List，则递归调用 `getInputs` 方法。
- 如果属性值是基本类型，则调用 `applyInput` 方法。

**用途:**
- 为配置属性添加元数据，方便用户了解属性的来源。

**11. `applyInput` 方法：应用源信息**

```java
private Map<String, Object> applyInput(String qualifiedKey, boolean showUnsanitized) {
    ConfigurationPropertyName currentName = getCurrentName(qualifiedKey);
    ConfigurationProperty candidate = getCandidate(currentName);
    PropertySource<?> propertySource = getPropertySource(candidate);
    if (propertySource != null) {
        Object value = stringifyIfNecessary(candidate.getValue());
        SanitizableData data = new SanitizableData(propertySource, currentName.toString(), value);
        return getInput(candidate, this.sanitizer.sanitize(data, showUnsanitized));
    }
    return Collections.emptyMap();
}
```

**描述:**
- 获取属性的 `ConfigurationProperty` 和 `PropertySource`。
- 使用 `getInput` 方法创建包含源信息的 Map。

**用途:**
- 获取配置属性的来源 (origin) 信息，例如配置文件名、行号等。

**12. `getInput` 方法：创建包含源信息的 Map**

```java
private Map<String, Object> getInput(ConfigurationProperty candidate, Object sanitizedValue) {
    Map<String, Object> input = new LinkedHashMap<>();
    Origin origin = Origin.from(candidate);
    List<Origin> originParents = Origin.parentsFrom(candidate);
    input.put("value", sanitizedValue);
    input.put("origin", (origin != null) ? origin.toString() : "none");
    if (!originParents.isEmpty()) {
        input.put("originParents", originParents.stream().map(Object::toString).toArray(String[]::new));
    }
    return input;
}
```

**描述:**
- 从 `ConfigurationProperty` 中提取 `Origin` 信息。
- 创建包含 `value` 和 `origin` 的 Map。

**用途:**
- 创建描述配置属性来源信息的 Map。
- `origin` 包含了属性的来源，例如配置文件名、行号等。

**13. Jackson 配置：`configureJsonMapper` 方法**

```java
protected void configureJsonMapper(JsonMapper.Builder builder) {
    builder.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
    builder.configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false);
    builder.configure(SerializationFeature.WRITE_DURATIONS_AS_TIMESTAMPS, false);
    builder.configure(MapperFeature.USE_STD_BEAN_NAMING, true);
    builder.serializationInclusion(Include.NON_NULL);
    applyConfigurationPropertiesFilter(builder);
    applySerializationModifier(builder);
    builder.addModule(new JavaTimeModule());
    builder.addModule(new ConfigurationPropertiesModule());
}
```

**描述:**
- 配置 Jackson `ObjectMapper` 的行为，例如：
  - 忽略空 Bean。
  - 禁用日期和时间戳序列化。
  - 使用标准的 Bean 命名规则。
  - 忽略 null 值。
- 应用自定义的过滤器和序列化器修改器。
- 注册 JavaTimeModule 和 ConfigurationPropertiesModule。

**用途:**
- 控制配置属性序列化为 JSON 的方式。
- 可以自定义序列化行为，例如过滤属性、格式化日期等。

**14.  过滤器和序列化器修改器**

- `ConfigurationPropertiesAnnotationIntrospector`：扩展 JacksonAnnotationIntrospector 以抑制 CGLIB 生成的 Bean 属性。
- `ConfigurationPropertiesPropertyFilter`：用于序列化 `@ConfigurationProperties` Bean 的 `SimpleBeanPropertyFilter`。 此过滤器隐藏：名称以 '$$' 开头的属性；自引用的属性；以及在检索其值时抛出异常的属性。
- `GenericSerializerModifier`：一个 `BeanSerializerModifier`，只返回相关的配置属性。

**总结:**

这段代码实现了一个 Spring Boot Actuator 端点，用于公开应用程序的配置属性。 它具有以下关键功能：

- **属性清理:** 可以清理敏感信息，例如密码和密钥。
- **源信息:** 可以显示属性的来源，方便用户了解属性的配置位置。
- **可配置性:** 可以通过配置控制属性值的显示。
- **安全性:** 可以避免因序列化失败导致程序中断。

这段代码的核心思想是利用 Spring Boot 的 Actuator 框架和 Jackson 库，将配置属性 Bean 转换为 JSON 格式，并进行必要的处理，最终通过 HTTP 请求返回给用户。 通过这种方式，用户可以方便地查看和管理应用程序的配置。
