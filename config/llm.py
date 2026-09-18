"""LLM endpoint settings, read from environment variables.

Any OpenAI-compatible chat-completions endpoint is supported:

- LLM_BASE_URL:  e.g. http://192.168.18.200:1234/v1 (OpenAI-compatible root)
- LLM_API_KEY:   optional bearer token (local servers usually need none)
- LLM_MODEL:     model name sent in the request body
- LLM_TIMEOUT:   per-request timeout in seconds

Reasoning (thinking) models — e.g. vLLM-served Qwen3 — are handled in
models/blm.py, which sends chat_template_kwargs={"enable_thinking": false}
so the model answers directly instead of burning its token budget
reasoning. Set LLM_ENABLE_THINKING=1 to force thinking back on.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout: float

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "http://192.168.18.200:1234/v1").rstrip("/"),
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", "Qwen3.8_mtp-27B"),
            timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"
