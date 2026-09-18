"""Minimal OpenAI-compatible chat-completions client (no SDK dependency)."""

import requests

from config.llm import LLMSettings


def chat_completion(
    messages: list[dict],
    settings: LLMSettings | None = None,
    temperature: float = 0.8,
    max_tokens: int = 200,
    extra_body: dict | None = None,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the message content.

    Args:
        messages:    OpenAI-style message list ([{"role": ..., "content": ...}]).
        settings:    LLM endpoint settings; defaults to LLMSettings.from_env().
        temperature: Sampling temperature (kept > 0 so repeated samples disperse).
        max_tokens:  Response token cap.
        extra_body:  Extra top-level payload fields (e.g. vLLM chat_template_kwargs).

    Returns:
        The assistant message content as a string.

    Raises:
        RuntimeError: On HTTP errors, timeout, or malformed responses.
    """
    if settings is None:
        settings = LLMSettings.from_env()
    if not settings.is_configured:
        raise RuntimeError(
            "LLM endpoint not configured. Set LLM_BASE_URL and LLM_MODEL (see config/llm.py)."
        )

    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)

    try:
        resp = requests.post(settings.chat_url, json=payload, headers=headers, timeout=settings.timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"LLM request failed ({settings.chat_url}): {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        body = resp.json()
        return str(body["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Malformed LLM response: {e}") from e
