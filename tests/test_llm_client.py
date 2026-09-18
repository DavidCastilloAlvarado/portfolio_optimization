"""Tests for models/llm_client.py (HTTP mocked)."""

import pytest
import requests
from config.llm import LLMSettings
from models.llm_client import chat_completion


class FakeResp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _settings(api_key=""):
    return LLMSettings(base_url="http://test/v1", api_key=api_key, model="test-model", timeout=5)


def _ok_response(content="42.5"):
    return FakeResp(200, {"choices": [{"message": {"content": content}}]})


class TestChatCompletion:
    def test_success(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return _ok_response()

        monkeypatch.setattr("models.llm_client.requests.post", fake_post)
        out = chat_completion([{"role": "user", "content": "hi"}], settings=_settings(), temperature=0.7, max_tokens=99)
        assert out == "42.5"
        assert calls[0]["url"] == "http://test/v1/chat/completions"
        assert calls[0]["json"]["model"] == "test-model"
        assert calls[0]["json"]["temperature"] == 0.7
        assert calls[0]["json"]["max_tokens"] == 99
        assert calls[0]["timeout"] == 5
        assert "Authorization" not in calls[0]["headers"]

    def test_auth_header_when_key_set(self, monkeypatch):
        def fake_post(url, json=None, headers=None, timeout=None):
            assert headers.get("Authorization") == "Bearer sekrit"
            return _ok_response()

        monkeypatch.setattr("models.llm_client.requests.post", fake_post)
        chat_completion([{"role": "user", "content": "hi"}], settings=_settings(api_key="sekrit"))

    def test_http_error_raises(self, monkeypatch):
        monkeypatch.setattr("models.llm_client.requests.post",
                            lambda *a, **k: FakeResp(500, text="boom"))
        with pytest.raises(RuntimeError, match="HTTP 500"):
            chat_completion([{"role": "user", "content": "hi"}], settings=_settings())

    def test_timeout_raises(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError("timed out")

        monkeypatch.setattr("models.llm_client.requests.post", boom)
        with pytest.raises(RuntimeError, match="LLM request failed"):
            chat_completion([{"role": "user", "content": "hi"}], settings=_settings())

    def test_malformed_body_raises(self, monkeypatch):
        monkeypatch.setattr("models.llm_client.requests.post",
                            lambda *a, **k: FakeResp(200, {"nope": 1}))
        with pytest.raises(RuntimeError, match="Malformed LLM response"):
            chat_completion([{"role": "user", "content": "hi"}], settings=_settings())

    def test_not_configured_raises(self):
        with pytest.raises(RuntimeError, match="not configured"):
            chat_completion([{"role": "user", "content": "hi"}],
                            settings=LLMSettings(base_url="", api_key="", model="", timeout=5))


class TestLLMSettings:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://lan:1234/v1/")
        monkeypatch.setenv("LLM_API_KEY", "abc")
        monkeypatch.setenv("LLM_MODEL", "some-model")
        monkeypatch.setenv("LLM_TIMEOUT", "33")
        s = LLMSettings.from_env()
        assert s.base_url == "http://lan:1234/v1"
        assert s.chat_url == "http://lan:1234/v1/chat/completions"
        assert s.api_key == "abc"
        assert s.model == "some-model"
        assert s.timeout == 33.0
        assert s.is_configured

    def test_from_env_defaults(self, monkeypatch):
        for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT"):
            monkeypatch.delenv(var, raising=False)
        s = LLMSettings.from_env()
        assert s.base_url == "http://192.168.18.200:1234/v1"
        assert s.model == "Qwen3.8_mtp-27B"
        assert s.api_key == ""
        assert s.is_configured
