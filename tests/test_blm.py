"""Tests for models/blm.py (LLM-Black-Litterman; LLM + HTTP mocked)."""

import numpy as np
import pandas as pd
import pytest
from config.defaults import Config
from config.llm import LLMSettings
import models.blm as blm
import core.data.cache as cache_mod


SETTINGS = LLMSettings(base_url="http://test/v1", api_key="", model="test-model", timeout=5)


@pytest.fixture
def ret_frame():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-01", periods=61)
    a = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 61))
    b = 100 * np.cumprod(1 + rng.normal(0.002, 0.02, 61))
    prices = pd.DataFrame({"AAA": a, "BBB": b}, index=dates)
    return prices.pct_change().dropna()


@pytest.fixture
def cov_frame(ret_frame):
    return np.array(ret_frame.cov())


class TestEquilibriumPrior:
    def test_equal_weights(self):
        cov = np.array([[1e-4, 0.0], [0.0, 4e-4]])
        tau = 0.0385
        pi = blm.build_equilibrium_prior(cov, tau)
        assert pi[0] == pytest.approx(tau * 0.5e-4)
        assert pi[1] == pytest.approx(tau * 2e-4)

    def test_custom_weights_normalized(self):
        cov = np.eye(2) * 1e-4
        pi = blm.build_equilibrium_prior(cov, 1.0, weights=np.array([3.0, 1.0]))
        assert pi[0] == pytest.approx(0.75 * 1e-4)
        assert pi[1] == pytest.approx(0.25 * 1e-4)

    def test_inverse_vol_weights(self):
        cov = np.diag([1e-4, 4e-4])
        w = blm.inverse_vol_weights(cov)
        assert w[0] == pytest.approx(2 / 3)
        assert w[1] == pytest.approx(1 / 3)
        assert w.sum() == pytest.approx(1.0)

    def test_inverse_vol_zero_volatility_is_finite(self):
        w = blm.inverse_vol_weights(np.diag([0.0, 4e-4]))
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0)


class TestBlackLittermanPosterior:
    def test_views_equal_prior_returns_prior(self):
        cov = np.array([[1e-4, 1e-5], [1e-5, 2e-4]])
        tau = 0.05
        prior = blm.build_equilibrium_prior(cov, tau)
        P = np.eye(2)
        q = prior.copy()
        omega = np.diag(tau * np.diag(cov))
        post = blm.black_litterman_posterior(prior, cov, P, q, omega, tau)
        assert np.allclose(post, prior, atol=1e-12)

    def test_positive_view_pulls_mean_up(self):
        cov = np.array([[1e-4, 0.0], [0.0, 2e-4]])
        tau = 0.05
        prior = blm.build_equilibrium_prior(cov, tau)
        P = np.array([[1.0, 0.0]])
        q = np.array([0.01])
        omega = np.array([[1e-6]])
        post = blm.black_litterman_posterior(prior, cov, P, q, omega, tau)
        assert post[0] > prior[0]
        assert np.all(np.isfinite(post))

    def test_overconfident_view_is_bounded(self):
        cov = np.array([[1e-4, 0.0], [0.0, 2e-4]])
        tau = 0.05
        prior = blm.build_equilibrium_prior(cov, tau)
        P = np.array([[1.0, 0.0]])
        q = np.array([0.5])
        omega = np.array([[1e-18]])
        post = blm.black_litterman_posterior(prior, cov, P, q, omega, tau)
        assert np.all(np.isfinite(post))
        assert post[0] < q[0]


class TestParseViewResponse:
    @pytest.mark.parametrize("text,expected", [
        ('{"predicted_avg_daily_return_pct": 0.15}', 0.15),
        ('The forecast is {"predicted_avg_daily_return_pct": -0.05}.', -0.05),
        ('0.42', 0.42),
        ('{"value": 2}', 2.0),
        ('forecast over 14 days is +0.2', 0.2),
        ('the horizon is 14 days', None),
        ('{"value": NaN}', None),
        ('I expect steady growth', None),
    ])
    def test_parsing(self, text, expected):
        assert blm._parse_view_response(text) == pytest.approx(expected) if expected is not None \
            else blm._parse_view_response(text) is None


class TestGenerateViews:
    def test_mean_omega_and_cache(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        responses = {
            "AAA": ['{"predicted_avg_daily_return_pct": 0.10}',
                    '{"predicted_avg_daily_return_pct": 0.30}',
                    '{"predicted_avg_daily_return_pct": -0.10}'],
            "BBB": ["no numbers here"],
        }
        idx = {"AAA": 0, "BBB": 0}
        calls = {"n": 0}

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            calls["n"] += 1
            content = prompt[1]["content"]
            for name in ("AAA", "BBB"):
                if name in content:
                    v = responses[name][min(idx[name], len(responses[name]) - 1)]
                    idx[name] += 1
                    return v
            raise AssertionError("unexpected prompt")

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=3, blm_history_days=30)

        views, from_cache = blm.generate_views(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert from_cache is False
        assert set(views.keys()) == {"AAA"}
        v = views["AAA"]
        assert v["q"] == pytest.approx(0.10 / 100)
        assert v["n_ok"] == 3
        assert v["omega"] == pytest.approx(4e-6, rel=1e-6)
        cache_files = list(tmp_path.glob("llm_views_*.json"))
        assert len(cache_files) == 1

        def no_more_calls(*a, **k):
            raise AssertionError("should have been served from cache")

        monkeypatch.setattr(blm, "chat_completion", no_more_calls)
        views2, from_cache2 = blm.generate_views(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert from_cache2 is True
        assert set(views2.keys()) == {"AAA"}
        assert views2["AAA"]["q"] == pytest.approx(0.10 / 100)

    def test_each_sample_sends_a_distinct_prompt(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        prompts: list[list[str]] = []

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            prompts.append([m["content"] for m in prompt])
            return '{"predicted_avg_daily_return_pct": 0.10}'

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=3, blm_history_days=30)

        blm.generate_views(ret_frame, ["AAA"], cov_frame, cfg, settings=SETTINGS)

        assert len(prompts) == 3
        assert len({p[1] for p in prompts}) == 3
        assert {p[0] for p in prompts} == {blm.SYSTEM_PROMPT.format(horizon=cfg.blm_horizon_days)}
        for k, user in enumerate((p[1] for p in prompts), start=1):
            assert user.splitlines()[0].startswith(f"This is the sample {k} of 3.")
            assert "Asset: " in user

    def test_omega_floor_applied_on_zero_variance(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            return '{"predicted_avg_daily_return_pct": 0.20}'

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=3)
        views, _ = blm.generate_views(ret_frame, ["AAA"], cov_frame, cfg, settings=SETTINGS)
        floor = blm.OMEGA_FLOOR_FRACTION * cfg.llm_tau * cov_frame[0, 0]
        assert views["AAA"]["omega"] == pytest.approx(floor)

    def test_skips_short_history(self, monkeypatch, ret_frame, cov_frame, tmp_path):
        short = ret_frame.tail(5)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        called = []

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            called.append(prompt)
            return '{"predicted_avg_daily_return_pct": 0.1}'

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        cfg = Config(llm_view_engine="direct", llm_samples=2)
        views, _ = blm.generate_views(short, ["AAA"], cov_frame, cfg, settings=SETTINGS)
        assert views == {}
        assert called == []

    def test_unreachable_llm_returns_empty(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(blm, "chat_completion", boom)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=2)
        views, _ = blm.generate_views(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert views == {}


class TestThinkingHandling:
    def test_extra_body_sent_by_default(self, monkeypatch):
        monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
        monkeypatch.setitem(blm._EXTRABODY_SUPPORT, "supports", None)
        captured = []

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            captured.append(extra_body)
            return '{"predicted_avg_daily_return_pct": 0.1}'

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        blm._call_for_sample([{"role": "user", "content": "x"}], SETTINGS)
        assert captured == [{"chat_template_kwargs": {"enable_thinking": False}}]

    def test_env_forces_thinking_on(self, monkeypatch):
        monkeypatch.setenv("LLM_ENABLE_THINKING", "1")
        assert blm._view_extra_body() is None

    def test_fallback_when_endpoint_rejects_extra_body(self, monkeypatch):
        monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
        monkeypatch.setitem(blm._EXTRABODY_SUPPORT, "supports", None)
        calls = []

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            calls.append(extra_body)
            if extra_body is not None:
                raise RuntimeError("HTTP 400: unsupported parameter")
            return '{"predicted_avg_daily_return_pct": 0.2}'

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        out = blm._call_for_sample([{"role": "user", "content": "x"}], SETTINGS)
        assert out == '{"predicted_avg_daily_return_pct": 0.2}'
        assert calls[0] is not None
        assert calls[1] is None
        assert blm._EXTRABODY_SUPPORT["supports"] is False


class TestRunBlm:
    def test_degraded_when_no_views(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def boom(*a, **k):
            raise RuntimeError("llm down")

        monkeypatch.setattr(blm, "chat_completion", boom)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=2)
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert out["degraded"] is True
        assert out["view_names"] == []
        prior = blm.build_equilibrium_prior(cov_frame, cfg.llm_tau)
        assert np.allclose(out["posterior_mean"], prior)

    def test_partial_views(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            if "AAA" in prompt[1]["content"]:
                return '{"predicted_avg_daily_return_pct": 0.30}'
            return "garbage"

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=2)
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert out["degraded"] is False
        assert out["view_names"] == ["AAA"]
        assert out["posterior_mean"][0] > out["prior_mean"][0]
        assert np.all(np.isfinite(out["posterior_mean"]))

    def test_inverse_vol_prior_used(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def boom(*a, **k):
            raise RuntimeError("llm down")

        monkeypatch.setattr(blm, "chat_completion", boom)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=2, blm_prior="inverse_vol")
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        expected = blm.build_equilibrium_prior(cov_frame, cfg.llm_tau, blm.inverse_vol_weights(cov_frame))
        assert np.allclose(out["prior_mean"], expected)


class TestEngineDispatch:
    def test_agent_engine_used_when_configured(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        calls = {}

        def fake_agent_views(returns, names, cov, cfg, settings=None, mcp_settings=None):
            calls["agent"] = True
            return {
                "AAA": {
                    "q": 0.002, "omega": 1e-5,
                    "samples": [0.002], "n_ok": 1,
                    "details": [{"rationale": "web says up", "sources": ["http://x"], "confidence": 0.5}],
                },
            }, False

        monkeypatch.setenv("WEBSEARCH_MCP_COMMAND", "npx")
        monkeypatch.setattr("models.llm_agent.generate_agent_views", fake_agent_views, raising=False)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=1)
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert calls.get("agent") is True
        assert out["degraded"] is False
        assert out["view_names"] == ["AAA"]

    def test_agent_failure_falls_back_to_direct(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def boom(*a, **k):
            raise RuntimeError("mcp server down")

        monkeypatch.setenv("WEBSEARCH_MCP_COMMAND", "npx")
        monkeypatch.setattr("models.llm_agent.generate_agent_views", boom, raising=False)

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            if "BBB" in prompt[1]["content"]:
                return '{"predicted_avg_daily_return_pct": 0.25}'
            return "garbage"

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=2)
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert out["degraded"] is False
        assert out["view_names"] == ["BBB"]

    def test_unconfigured_mcp_falls_back_to_direct(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setenv("WEBSEARCH_MCP_COMMAND", "")

        def boom(*a, **k):
            raise AssertionError("agent must not run without an MCP command")

        monkeypatch.setattr("models.llm_agent.generate_agent_views", boom, raising=False)

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            if "BBB" in prompt[1]["content"]:
                return '{"predicted_avg_daily_return_pct": 0.25}'
            return "garbage"

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=2)
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert out["degraded"] is False
        assert out["view_names"] == ["BBB"]

    def test_direct_engine_skips_agent(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        def boom(*a, **k):
            raise AssertionError("agent must not run in direct mode")

        monkeypatch.setattr("models.llm_agent.generate_agent_views", boom, raising=False)

        def fake_chat(prompt, settings=None, temperature=0.8, max_tokens=160, extra_body=None):
            if "BBB" in prompt[1]["content"]:
                return '{"predicted_avg_daily_return_pct": 0.25}'
            return "garbage"

        monkeypatch.setattr(blm, "chat_completion", fake_chat)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="direct", llm_samples=2)
        out = blm.run_blm(ret_frame, ["AAA", "BBB"], cov_frame, cfg, settings=SETTINGS)
        assert out["degraded"] is False
        assert out["view_names"] == ["BBB"]
