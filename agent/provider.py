"""Small OpenAI-compatible Chat Completions client for local tool calling."""

import json
import os
import re
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
        except HTTPError as exc:
            # Read only structured error identifiers; provider messages can echo prompts or secrets.
            try:
                error = json.load(exc).get("error", {})
                code = error.get("code") or error.get("type")
                code = code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,60}", code) else None
            except (ValueError, TypeError, AttributeError):
                code = None
            suffix = f", code={code}" if code else ""
            retry = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code == 429 and retry and re.fullmatch(r"[0-9]{1,5}", retry):
                suffix += f", retry after {retry}s"
            raise ProviderError(f"LLM provider HTTP {exc.code}{suffix}") from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderError("LLM provider connection unavailable or timed out") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("LLM provider returned an invalid response") from exc

