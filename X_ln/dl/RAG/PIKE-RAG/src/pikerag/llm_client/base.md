Lan: `py` From`dl/RAG/PIKE-RAG/src\pikerag\llm_client\base.py`

Okay, I'll restart and provide a breakdown of the `BaseLLMClient` code, explaining the key parts with code snippets and Chinese descriptions, along with usage examples.

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pickledb import PickleDB

from pikerag.utils.logger import Logger


class BaseLLMClient(object):
    NAME = "BaseLLMClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        self._cache_auto_dump: bool = auto_dump
        self._cache: PickleDB = None
        if location is not None:
            self.update_cache_location(location)

        self._max_attempt: int = max_attempt
        assert max_attempt >= 1, f"max_attempt should be no less than 1 (but {max_attempt} was given)!"

        self._exponential_backoff_factor: int = exponential_backoff_factor
        self._unit_wait_time: int = unit_wait_time
        if self._exponential_backoff_factor is None:
            assert self._unit_wait_time > 0, (
                f"unit_wait_time should be positive (but {unit_wait_time} was given) "
                f"if exponential backoff is disabled ({exponential_backoff_factor} was given)!"
            )
        else:
            assert exponential_backoff_factor > 1, (
                "To enable the exponential backoff mode, the factor should be greater than 1 "
                f"(but {exponential_backoff_factor} was given)!"
            )

        self.logger = logger

    def warning(self, warning_message: str) -> None:
        if self.logger is not None:
            self.logger.info(msg=warning_message)
        else:
            print(warning_message)
        return

    def debug(self, debug_message: str) -> None:
        if self.logger is not None:
            self.logger.debug(msg=debug_message)
        return

    def _wait(self, num_attempt: int, wait_time: Optional[int] = None) -> None:
        if wait_time is None:
            if self._exponential_backoff_factor is None:
                wait_time = self._unit_wait_time * num_attempt
            else:
                wait_time = self._exponential_backoff_factor ** num_attempt

        time.sleep(wait_time)
        return

    def _generate_cache_key(self, messages: List[dict], llm_config: dict) -> str:
        assert isinstance(messages, List) and len(messages) > 0

        if isinstance(messages[0], Dict):
            return json.dumps((messages, llm_config))

        else:
            raise ValueError(f"Messages with unsupported type: {type(messages[0])}")

    def _save_cache(self, messages: List[dict], llm_config: dict, content: str) -> None:
        if self._cache is None:
            return

        key = self._generate_cache_key(messages, llm_config)
        self._cache.set(key, content)
        return

    def _get_cache(self, messages: List[dict], llm_config: dict) -> Union[str, Literal[False]]:
        if self._cache is None:
            return False

        key = self._generate_cache_key(messages, llm_config)
        value = self._cache.get(key)
        return value

    def _remove_cache(self, messages: List[dict], llm_config: dict) -> None:
        if self._cache is None:
            return

        key = self._generate_cache_key(messages, llm_config)
        self._cache.remove(key)
        return

    def generate_content_with_messages(self, messages: List[dict], **llm_config) -> str:
        # TODO: utilize self.llm_config if None provided in call.
        # TODO: add functions to get tokens, logprobs.
        content = self._get_cache(messages, llm_config)

        if content is False or content is None or content == "":
            if self.logger is not None:
                self.logger.debug(msg=f"{datetime.now()} create completion...", tag=self.NAME)
                start_time = time.time()

            response = self._get_response_with_messages(messages, **llm_config)

            if self.logger is not None:
                time_used = time.time() - start_time
                result = "receive response" if response is not None else "request failed"
                self.logger.debug(msg=f"{datetime.now()} {result}, time spent: {time_used} s.", tag=self.NAME)

            if response is None:
                self.warning("None returned as response")
                if messages is not None and len(messages) >= 1:
                    self.debug(f"  -- Last message: {messages[-1]}")
                content = ""
            else:
                content = self._get_content_from_response(response, messages=messages)

            self._save_cache(messages, llm_config, content)

        return content

    @abstractmethod
    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_content_from_response(self, response: Any, messages: List[dict] = None) -> str:
        raise NotImplementedError

    def update_cache_location(self, new_location: str) -> None:
        if self._cache is not None:
            self._cache.save()

        assert new_location is not None, f"A valid cache location must be provided"

        self._cache_location = new_location
        self._cache = PickleDB(location=self._cache_location)

    def close(self):
        """Close the active memory, connections, ...
        The client would not be usable after this operation."""
        self._cache.save()
```

**1. `BaseLLMClient` Class - 基础 LLM 客户端类**

```python
class BaseLLMClient(object):
    NAME = "BaseLLMClient"

    def __init__(
        self, location: str = None, auto_dump: bool = True, logger: Logger = None,
        max_attempt: int = 5, exponential_backoff_factor: int = None, unit_wait_time: int = 60, **kwargs,
    ) -> None:
        self._cache_auto_dump: bool = auto_dump
        self._cache: PickleDB = None
        if location is not None:
            self.update_cache_location(location)

        self._max_attempt: int = max_attempt
        assert max_attempt >= 1, f"max_attempt should be no less than 1 (but {max_attempt} was given)!"

        self._exponential_backoff_factor: int = exponential_backoff_factor
        self._unit_wait_time: int = unit_wait_time
        if self._exponential_backoff_factor is None:
            assert self._unit