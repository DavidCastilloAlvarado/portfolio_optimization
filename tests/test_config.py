"""Tests for config/defaults.py."""

import pytest
from config.defaults import Config


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.resample == "none"
        assert cfg.days == 720
        assert cfg.min_variance is False
        assert cfg.monto_usd == 10000.0
        assert cfg.monthly_delta == 300.0
        assert cfg.sim_days == 252
        assert cfg.risk_free_annual_perc == 5.0
        assert cfg.w_limits == (0.02, 0.12)
        assert cfg.w_limits_per_ticker == {}
        assert cfg.strategy == "sharpe"
        assert cfg.kelly_fraction == 0.5
        assert cfg.llm_samples == 5
        assert cfg.llm_tau == pytest.approx(1.0 / 26.0)
        assert cfg.blm_prior == "equal"
        assert cfg.blm_horizon_days == 14
        assert cfg.blm_history_days == 56
        assert cfg.llm_view_engine == "agent"
        assert cfg.llm_reasoning_effort == "medium"
        assert cfg.llm_agent_websearch is True
        assert cfg.llm_agent_max_tokens == 32768
        assert cfg.llm_agent_recursion_limit == 40

    def test_risk_free_computation(self):
        cfg = Config()
        expected = (1 + 5 / 100) ** (1 / 252) - 1
        assert cfg.risk_free == pytest.approx(expected)

    def test_get_resample_none(self):
        assert Config(resample="none").get_resample() is None

    def test_get_resample_week(self):
        assert Config(resample="week").get_resample() == "week"

    def test_get_resample_month(self):
        assert Config(resample="month").get_resample() == "month"

    def test_from_dict_basic(self):
        data = {
            "resample": "week",
            "days": "365",
            "shares": "AAPL, GOOG",
            "w_limits": "0.05,0.20",
            "min_variance": "true",
            "monto_usd": "5000",
            "monthly_delta": "500",
            "sim_days": "126",
            "risk_free_annual_perc": "3.5",
        }
        cfg = Config.from_dict(data)
        assert cfg.resample == "week"
        assert cfg.days == 365
        assert cfg.shares == ["AAPL", "GOOG"]
        assert cfg.w_limits == (0.05, 0.20)
        assert cfg.min_variance is True
        assert cfg.monto_usd == 5000.0
        assert cfg.monthly_delta == 500.0
        assert cfg.sim_days == 126
        assert cfg.risk_free_annual_perc == 3.5

    def test_from_dict_empty(self):
        cfg = Config.from_dict({})
        assert cfg.shares == []
        assert cfg.w_limits == (0.02, 0.12)
        assert cfg.w_limits_per_ticker == {}

    def test_from_dict_per_ticker(self):
        data = {
            "shares": "QQQM,SCHD",
            "w_limits": "0.02,0.12",
            "w_limits_per_ticker": "qqqm:0.02,0.12;SCHD:0.05,",
        }
        cfg = Config.from_dict(data)
        assert cfg.w_limits_per_ticker == {"QQQM": (0.02, 0.12), "SCHD": (0.05, None)}

    def test_from_dict_per_ticker_invalid_side(self):
        cfg = Config.from_dict({"w_limits_per_ticker": "QQQM:0.02,abc;XLU:1.5"})
        assert cfg.w_limits_per_ticker == {"QQQM": (0.02, None), "XLU": (1.5, None)}

    def test_from_dict_invalid_global_limits(self):
        cfg = Config.from_dict({"w_limits": "0.05"})
        assert cfg.w_limits == (0.02, 0.12)


class TestStrategyResolution:
    def test_resolve_defaults_to_sharpe(self):
        assert Config().resolve_strategy() == "sharpe"

    def test_resolve_min_variance_flag(self):
        assert Config(min_variance=True).resolve_strategy() == "min_variance"

    def test_resolve_strategy_field(self):
        assert Config(strategy="kelly").resolve_strategy() == "kelly"
        assert Config(strategy="min_variance").resolve_strategy() == "min_variance"

    def test_resolve_strategy_overrides_flag(self):
        assert Config(min_variance=True, strategy="kelly").resolve_strategy() == "kelly"

    def test_resolve_unknown_strategy_falls_back(self):
        assert Config(strategy="bogus").resolve_strategy() == "sharpe"

    def test_from_dict_kelly(self):
        data = {
            "shares": "AAPL,GOOG",
            "strategy": "kelly",
            "kelly_fraction": "0.25",
        }
        cfg = Config.from_dict(data)
        assert cfg.strategy == "kelly"
        assert cfg.kelly_fraction == pytest.approx(0.25)
        assert cfg.resolve_strategy() == "kelly"

    def test_from_dict_kelly_fraction_clamped(self):
        assert Config.from_dict({"strategy": "kelly", "kelly_fraction": "5"}).kelly_fraction == 1.0
        assert Config.from_dict({"strategy": "kelly", "kelly_fraction": "abc"}).kelly_fraction == 0.5
        assert Config.from_dict({}).kelly_fraction == 0.5

    def test_from_dict_legacy_min_variance_maps_to_strategy(self):
        cfg = Config.from_dict({"min_variance": "on"})
        assert cfg.min_variance is True
        assert cfg.strategy == "min_variance"
        assert cfg.resolve_strategy() == "min_variance"

    def test_from_dict_invalid_strategy_ignored(self):
        cfg = Config.from_dict({"strategy": "quantum"})
        assert cfg.strategy == "sharpe"
        assert cfg.resolve_strategy() == "sharpe"

    def test_resolve_llm_bl(self):
        assert Config(strategy="llm_bl").resolve_strategy() == "llm_bl"

    def test_from_dict_llm_bl(self):
        data = {
            "strategy": "llm_bl",
            "llm_samples": "10",
            "llm_tau": "0.05",
            "blm_prior": "inverse_vol",
            "blm_horizon_days": "7",
            "blm_history_days": "120",
        }
        cfg = Config.from_dict(data)
        assert cfg.strategy == "llm_bl"
        assert cfg.llm_samples == 10
        assert cfg.llm_tau == pytest.approx(0.05)
        assert cfg.blm_prior == "inverse_vol"
        assert cfg.blm_horizon_days == 7
        assert cfg.blm_history_days == 120

    def test_from_dict_llm_clamped(self):
        cfg = Config.from_dict({"strategy": "llm_bl", "llm_samples": "9999", "llm_tau": "5",
                                "blm_horizon_days": "0", "blm_prior": "bogus"})
        assert cfg.llm_samples == 50
        assert cfg.llm_tau == 1.0
        assert cfg.blm_horizon_days == 1
        assert cfg.blm_prior == "equal"

    def test_from_dict_agent_engine_fields(self):
        data = {
            "strategy": "llm_bl",
            "llm_view_engine": "direct",
            "llm_reasoning_effort": "high",
            "llm_agent_websearch": "off",
            "llm_agent_max_tokens": "2048",
            "llm_agent_recursion_limit": "60",
        }
        cfg = Config.from_dict(data)
        assert cfg.llm_view_engine == "direct"
        assert cfg.llm_reasoning_effort == "high"
        assert cfg.llm_agent_websearch is False
        assert cfg.llm_agent_max_tokens == 2048
        assert cfg.llm_agent_recursion_limit == 60

    def test_from_dict_agent_engine_defaults(self):
        cfg = Config.from_dict({"strategy": "llm_bl"})
        assert cfg.llm_view_engine == "agent"
        assert cfg.llm_reasoning_effort == "medium"
        assert cfg.llm_agent_websearch is True
        assert cfg.llm_agent_max_tokens == 32768
        assert cfg.llm_agent_recursion_limit == 40

    def test_from_dict_agent_engine_invalid(self):
        cfg = Config.from_dict({
            "strategy": "llm_bl",
            "llm_view_engine": "quantum",
            "llm_reasoning_effort": "ultra",
            "llm_agent_max_tokens": "999999",
            "llm_agent_recursion_limit": "999999",
        })
        assert cfg.llm_view_engine == "direct"
        assert cfg.llm_reasoning_effort == "medium"
        assert cfg.llm_agent_max_tokens == 32768
        assert cfg.llm_agent_recursion_limit == 200

    def test_from_dict_websearch_true_variants(self):
        for raw in (True, "true", "on", "1", 1):
            cfg = Config.from_dict({"strategy": "llm_bl", "llm_agent_websearch": raw})
            assert cfg.llm_agent_websearch is True
    def test_from_dict_selection_fields(self):
        cfg = Config.from_dict({
            "selection_top_n": "14",
            "selection_sector_quant_weight": "0.75",
            "selection_name_llm_weight": "0.3",
            "selection_event_risk_days": "7",
            "selection_recursion_limit": "120",
            "selection_verdict_concurrency": "4",
        })
        assert cfg.selection_top_n == 14
        assert cfg.selection_sector_quant_weight == pytest.approx(0.75)
        assert cfg.selection_name_llm_weight == pytest.approx(0.3)
        assert cfg.selection_event_risk_days == 7
        assert cfg.selection_recursion_limit == 120
        assert cfg.selection_verdict_concurrency == 4

    def test_from_dict_selection_defaults(self):
        cfg = Config.from_dict({})
        assert cfg.selection_top_n == 10
        assert cfg.selection_sector_quant_weight == pytest.approx(0.6)
        assert cfg.selection_name_llm_weight == pytest.approx(0.5)
        assert cfg.selection_event_risk_days == 5
        assert cfg.selection_recursion_limit == 160
        assert cfg.selection_verdict_concurrency == 2

    def test_from_dict_selection_clamped(self):
        cfg = Config.from_dict({
            "selection_top_n": "999",
            "selection_sector_quant_weight": "5",
            "selection_name_llm_weight": "-3",
            "selection_event_risk_days": "999",
            "selection_recursion_limit": "1",
            "selection_verdict_concurrency": "99",
        })
        assert cfg.selection_top_n == 16
        assert cfg.selection_sector_quant_weight == pytest.approx(1.0)
        assert cfg.selection_name_llm_weight == pytest.approx(0.0)
        assert cfg.selection_event_risk_days == 30
        assert cfg.selection_recursion_limit == 20
        assert cfg.selection_verdict_concurrency == 8
