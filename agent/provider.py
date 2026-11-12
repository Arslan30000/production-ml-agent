"""Small OpenAI-compatible Chat Completions client for local tool calling."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


class ChatProvider:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: int = 45):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or
                         "https://api.groq.com/openai/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY in the environment before using the LLM agent")
        if not self.base_url.startswith("https://") and not self.base_url.startswith("http://127.0.0.1:"):
            raise ValueError("Use an HTTPS provider URL or a local HTTP provider")

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = json.dumps({"model": self.model, "messages": messages, "tools": tools,
                              "tool_choice": "auto", "parallel_tool_calls": False,
                              "max_completion_tokens": 1600}).encode("utf-8")
        request = Request(self.base_url + "/chat/completions", data=payload, method="POST",
                          headers={"Authorization": "Bearer " + self.api_key,
                                   "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("missing message")
            return message
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
            # Provider response bodies can contain prompts and tokens; avoid exposing them.
            raise ProviderError("LLM provider unavailable or returned an invalid response") from exc
