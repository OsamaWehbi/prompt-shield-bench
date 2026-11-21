"""A small OpenAI-compatible chat completions client using only the standard library."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit


class ClientError(RuntimeError):
    pass


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
        retries: int = 2,
        temperature: float = 0.0,
    ) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ClientError(f"base URL must be http or https, got {base_url!r}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")  # noqa: S310
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310  # nosec B310
                    payload = json.loads(resp.read())
                return payload["choices"][0]["message"]
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    detail = exc.read()[:300].decode("utf-8", "replace")
                    raise ClientError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(min(8.0, 1.5 * 2**attempt))
        raise ClientError(f"model endpoint failed after {self.retries + 1} attempts: {last}")
