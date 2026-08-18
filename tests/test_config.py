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

    def test_risk_free_computation(self):
        cfg = Config()
        expected = (1 + 5 / 100) ** (1 / 365) - 1
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
