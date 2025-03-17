Lan: `py` From`dl/flux/src\flux\api.py`

好的，我们将重新审视并改进 `ImageRequest` 类，专注于代码质量、可维护性和潜在的改进点。

**1. 改进的异常处理 (Improved Exception Handling):**

```python
class APIError(Exception):
    """
    自定义 API 错误异常，包含状态码和详细信息.
    """
    def __init__(self, status_code: int, message: str, detail: dict | list | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail

    def __str__(self):
        return f"APIError: Status Code: {self.status_code}, Message: {self.message}, Detail: {self.detail}"


# Demo Usage
if __name__ == '__main__':
    try:
        # 模拟一个 API 错误
        raise APIError(400, "Bad Request", {"error": "Invalid input"})
    except APIError as e:
        print(e)
```

**描述:** 创建了一个更清晰、更有用的自定义异常类 `APIError`，它继承自 `Exception`。

**主要改进:**

*   **Custom Exception Class (自定义异常类):**  使用自定义异常 `APIError` 代替通用的 `ApiException`，提高了代码可读性和可维护性。
*   **Clear Error Messages (清晰的错误消息):** 异常消息现在包含状态码和更具描述性的错误文本。

**2. 改进的请求构建 (Improved Request Building):**

```python
import requests
import os

class APIClient:
    def __init__(self, api_url, api_endpoints, api_key=None):
        self.api_url = api_url
        self.api_endpoints = api_endpoints
        self.api_key = api_key or os.environ.get("BFL_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and BFL_API_KEY environment variable not set.")

    def _build_headers(self):
        """构建请求头."""
        return {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _handle_response(self, response):
        """处理 API 响应并抛出异常."""
        try:
            response.raise_for_status()  # 检查 HTTP 错误状态
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_message = response.json().get("detail", str(e))
            except:
                error_message = str(e)

            raise APIError(response.status_code, error_message)
        except requests.exceptions.RequestException as e:
            raise APIError(500, f"Request failed: {e}")

    def post(self, endpoint, json_data):
        """发送 POST 请求."""
        url = f"{self.api_url}/v1/{endpoint}"
        headers = self._build_headers()
        response = requests.post(url, headers=headers, json=json_data)
        return self._handle_response(response)

    def get(self, endpoint, params=None):
        """发送 GET 请求."""
        url = f"{self.api_url}/v1/{endpoint}"
        headers = self._build_headers()
        response = requests.get(url, headers=headers, params=params)
        return self._handle_response(response)

# Demo Usage
if __name__ == '__main__':
    # 替换为实际的 API URL 和 endpoint
    api_url = "https://api.example.com"
    api_endpoints = {"test_endpoint": "test"}
    try:
        api_client = APIClient(api_url, api_endpoints, api_key="YOUR_API_KEY") # Replace with your API key
        # 模拟 POST 请求
        data = {"key": "value"}
        response_data = api_client.post("test_endpoint", data)
        print("POST Response:", response_data)

        # 模拟 GET 请求
        params = {"param1": "value1"}
        response_data = api_client.get("test_endpoint", params)
        print("GET Response:", response_data)
    except APIError as e:
        print(e)
    except ValueError as e:
        print(e)
```

**描述:**  创建了一个 `APIClient` 类来封装 API 请求的构建和处理。

**主要改进:**

*   **Centralized Request Handling (集中式请求处理):** `APIClient` 类负责构建请求头、发送请求和处理响应，使代码更简洁。
*   **Error Handling (错误处理):** `_handle_response` 方法统一处理 API 响应，检查 HTTP 错误并抛出自定义异常。
*   **API Key Management (API 密钥管理):** 明确处理 API 密钥的获取，如果未提供 API 密钥或未设置环境变量，则抛出异常。

**3. 改进的输入验证 (Improved Input Validation):**

```python
def validate_params(name, width, height, num_steps, guidance, interval, safety_tolerance):
    """验证图像生成请求的参数."""
    API_ENDPOINTS = {
        "flux.1-pro": "flux-pro",
        "flux.1-dev": "flux-dev",
        "flux.1.1-pro": "flux-pro-1.1",
    }

    if name not in API_ENDPOINTS:
        raise ValueError(f"Invalid model {name}")
    if width is not None and width % 32 != 0:
        raise ValueError(f"width must be divisible by 32, got {width}")
    if width is not None and not (256 <= width <= 1440):
        raise ValueError(f"width must be between 256 and 1440, got {width}")
    if height is not None and height % 32 != 0:
        raise ValueError(f"height must be divisible by 32, got {height}")
    if height is not None and not (256 <= height <= 1440):
        raise ValueError(f"height must be between 256 and 1440, got {height}")
    if num_steps is not None and not (1 <= num_steps <= 50):
        raise ValueError(f"steps must be between 1 and 50, got {num_steps}")
    if guidance is not None and not (1.5 <= guidance <= 5.0):
        raise ValueError(f"guidance must be between 1.5 and 4, got {guidance}")
    if interval is not None and not (1.0 <= interval <= 4.0):
        raise ValueError(f"interval must be between 1 and 4, got {interval}")
    if safety_tolerance is not None and not (0 <= safety_tolerance <= 6.0):
        raise ValueError(f"safety_tolerance must be between 0 and 6, got {safety_tolerance}")

    if name == "flux.1-dev" and interval is not None:
        raise ValueError("Interval is not supported for flux.1-dev")
    if name == "flux.1.1-pro" and (interval is not None or num_steps is not None or guidance is not None):
        raise ValueError("Interval, num_steps and guidance are not supported for flux.1.1-pro")


# Demo Usage
if __name__ == '__main__':
    try:
        validate_params(name="flux.1-pro", width=512, height=512, num_steps=20, guidance=3.0, interval=None, safety_tolerance=3)
        print("参数验证通过")
    except ValueError as e:
        print(f"参数验证失败: {e}")
```

**描述:** 将参数验证逻辑提取到一个单独的函数 `validate_params` 中。

**主要改进:**

*   **Separation of Concerns (关注点分离):**  将验证逻辑从 `ImageRequest` 类中分离出来，提高了代码的可读性和可测试性。
*   **Clear Validation Rules (清晰的验证规则):**  验证规则更加明确，易于理解和维护。

**4.  最终的 `ImageRequest` 类 (Final `ImageRequest` Class):**

```python
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image

class APIError(Exception):
    """
    Custom API error exception containing status code and details.
    """
    def __init__(self, status_code: int, message: str, detail: dict | list | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail

    def __str__(self):
        return f"APIError: Status Code: {self.status_code}, Message: {self.message}, Detail: {self.detail}"


class APIClient:
    def __init__(self, api_url, api_endpoints, api_key=None):
        self.api_url = api_url
        self.api_endpoints = api_endpoints
        self.api_key = api_key or os.environ.get("BFL_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and BFL_API_KEY environment variable not set.")

    def _build_headers(self):
        """Build request headers."""
        return {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _handle_response(self, response):
        """Handle API responses and raise exceptions."""
        try:
            response.raise_for_status()  # Check for HTTP error status
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_message = response.json().get("detail", str(e))
            except:
                error_message = str(e)

            raise APIError(response.status_code, error_message, error_message)
        except requests.exceptions.RequestException as e:
            raise APIError(500, f"Request failed: {e}")

    def post(self, endpoint, json_data):
        """Send a POST request."""
        url = f"{self.api_url}/v1/{endpoint}"
        headers = self._build_headers()
        response = requests.post(url, headers=headers, json=json_data)
        return self._handle_response(response)

    def get(self, endpoint, params=None):
        """Send a GET request."""
        url = f"{self.api_url}/v1/{endpoint}"
        headers = self._build_headers()
        response = requests.get(url, headers=headers, params=params)
        return self._handle_response(response)


def validate_params(name, width, height, num_steps, guidance, interval, safety_tolerance):
    """Validate image generation request parameters."""
    API_ENDPOINTS = {
        "flux.1-pro": "flux-pro",
        "flux.1-dev": "flux-dev",
        "flux.1.1-pro": "flux-pro-1.1",
    }

    if name not in API_ENDPOINTS:
        raise ValueError(f"Invalid model {name}")
    if width is not None and width % 32 != 0:
        raise ValueError(f"width must be divisible by 32, got {width}")
    if width is not None and not (256 <= width <= 1440):
        raise ValueError(f"width must be between 256 and 1440, got {width}")
    if height is not None and height % 32 != 0:
        raise ValueError(f"height must be divisible by 32, got {height}")
    if height is not None and not (256 <= height <= 1440):
        raise ValueError(f"height must be between 256 and 1440, got {height}")
    if num_steps is not None and not (1 <= num_steps <= 50):
        raise ValueError(f"steps must be between 1 and 50, got {num_steps}")
    if guidance is not None and not (1.5 <= guidance <= 5.0):
        raise ValueError(f"guidance must be between 1.5 and 4, got {guidance}")
    if interval is not None and not (1.0 <= interval <= 4.0):
        raise ValueError(f"interval must be between 1 and 4, got {interval}")
    if safety_tolerance is not None and not (0 <= safety_tolerance <= 6.0):
        raise ValueError(f"safety_tolerance must be between 0 and 6, got {safety_tolerance}")

    if name == "flux.1-dev" and interval is not None:
        raise ValueError("Interval is not supported for flux.1-dev")
    if name == "flux.1.1-pro" and (interval is not None or num_steps is not None or guidance is not None):
        raise ValueError("Interval, num_steps and guidance are not supported for flux.1.1-pro")


class ImageRequest:
    def __init__(
        self,
        # api inputs
        prompt: str,
        name: str = "flux.1.1-pro",
        width: int | None = None,
        height: int | None = None,
        num_steps: int | None = None,
        prompt_upsampling: bool | None = None,
        seed: int | None = None,
        guidance: float | None = None,
        interval: float | None = None,
        safety_tolerance: int | None = None,
        # behavior of this class
        validate: bool = True,
        launch: bool = True,
        api_key: str | None = None,
    ):
        """
        Manages an image generation request to the API.

        All parameters not specified will use the API defaults.

        Args:
            prompt: Text prompt for image generation.
            width: Width of the generated image in pixels. Must be a multiple of 32.
            height: Height of the generated image in pixels. Must be a multiple of 32.
            name: Which model version to use
            num_steps: Number of steps for the image generation process.
            prompt_upsampling: Whether to perform upsampling on the prompt.
            seed: Optional seed for reproducibility.
            guidance: Guidance scale for image generation.
            safety_tolerance: Tolerance level for input and output moderation.
                 Between 0 and 6, 0 being most strict, 6 being least strict.
            validate: Run input validation
            launch: Directly launches request
            api_key: Your API key if not provided by the environment

        Raises:
            ValueError: For invalid input, when `validate`
            APIError: For errors raised from the API
        """
        self.API_URL = "https://api.bfl.ml"
        self.API_ENDPOINTS = {
            "flux.1-pro": "flux-pro",
            "flux.1-dev": "flux-dev",
            "flux.1.1-pro": "flux-pro-1.1",
        }
        self.api_client = APIClient(self.API_URL, self.API_ENDPOINTS, api_key)

        if validate:
            validate_params(name, width, height, num_steps, guidance, interval, safety_tolerance)

        self.name = name
        self.request_json = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": num_steps,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "guidance": guidance,
            "interval": interval,
            "safety_tolerance": safety_tolerance,
        }
        self.request_json = {key: value for key, value in self.request_json.items() if value is not None}

        self.request_id: str | None = None
        self.result: dict | None = None
        self._image_bytes: bytes | None = None
        self._url: str | None = None

        if launch:
            self.request()

    def request(self):
        """
        Request to generate the image.
        """
        if self.request_id is not None:
            return
        try:
            result = self.api_client.post(self.API_ENDPOINTS[self.name], self.request_json)
            self.request_id = result["id"]
        except APIError as e:
            raise e

    def retrieve(self) -> dict:
        """
        Wait for the generation to finish and retrieve response.
        """
        if self.request_id is None:
            self.request()
        while self.result is None:
            try:
                result = self.api_client.get("get_result", params={"id": self.request_id})
                if "status" not in result:
                    raise APIError(500, "API response missing 'status' field", result)
                elif result["status"] == "Ready":
                    self.result = result["result"]
                elif result["status"] == "Pending":
                    time.sleep(0.5)
                else:
                    raise APIError(200, f"API returned status '{result['status']}'")
            except APIError as e:
                raise e
        return self.result

    @property
    def bytes(self) -> bytes:
        """
        Generated image as bytes.
        """
        if self._image_bytes is None:
            try:
                response = requests.get(self.url)
                response.raise_for_status()  # Check for HTTP errors
                self._image_bytes = response.content
            except requests.exceptions.RequestException as e:
                raise APIError(response.status_code if hasattr(response, 'status_code') else 500, f"Failed to download image: {e}")
        return self._image_bytes

    @property
    def url(self) -> str:
        """
        Public url to retrieve the image from
        """
        if self._url is None:
            result = self.retrieve()
            self._url = result["sample"]
        return self._url

    @property
    def image(self) -> Image.Image:
        """
        Load the image as a PIL Image
        """
        return Image.open(io.BytesIO(self.bytes))

    def save(self, path: str):
        """
        Save the generated image to a local path
        """
        suffix = Path(self.url).suffix
        if not path.endswith(suffix):
            path = path + suffix
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            file.write(self.bytes)


if __name__ == "__main__":
    from fire import Fire
    Fire(ImageRequest)

```

**描述:** 将所有改进整合到最终的 `ImageRequest` 类中。

**主要变化:**

*   **Uses `APIClient` (使用 `APIClient`):**  `ImageRequest` 类现在使用 `APIClient` 类来处理 API 请求。
*   **Uses `validate_params` (使用 `validate_params`):**  `ImageRequest` 类现在使用 `validate_params` 函数来验证输入参数。
*   **Improved Error Handling (改进的错误处理):**  所有 API 调用都包含在 `try...except` 块中，以捕获并处理 `APIError` 异常。

**5. 使用示例 (Demo Usage):**

```python
if __name__ == "__main__":
    try:
        image_request = ImageRequest(
            prompt="A futuristic cityscape",
            width=512,
            height=512,
            api_key="YOUR_API_KEY",  # 替换为你的 API 密钥
            validate=True,
            launch=True
        )

        image_request.save("futuristic_cityscape.png")
        print("Image saved to futuristic_cityscape.png")

    except ValueError as e:
        print(f"Validation Error: {e}")
    except APIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
```

**描述:**  提供了一个使用 `ImageRequest` 类的简单示例。

**重要提示:**

*   替换 `"YOUR_API_KEY"` 为你的实际 API 密钥。
*   这段代码需要 `fire` 库才能通过命令行运行。 你可以使用 `pip install fire` 安装它。
*   这段代码假设你已经设置了 `BFL_API_KEY` 环境变量，或者你将 API 密钥直接传递给 `ImageRequest` 构造函数。

**优点和改进总结：**

*   **代码组织:**  将代码分成更小的、更易于管理的类和函数。
*   **异常处理:**  使用自定义异常类，提供更清晰、更详细的错误消息。
*   **可维护性:**  分离关注点，使得更容易修改和扩展代码。
*   **可测试性:**  单独的函数更容易进行单元测试。

这段代码比原来的版本更健壮、更易于维护和更易于使用。 它提供了更好的错误处理，并使用了更清晰的编程模式。
