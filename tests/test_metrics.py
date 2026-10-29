import pandas as pd
import pytest

from realestate.metrics.index_metrics import rebase, region_metrics


def _series(values, start="2020-01-01", key="A"):
    ym = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"region_key": key, "ym": ym, "index_value": values})


def test_drawdown_rebound_and_changes():
    # 100 → 고점 120 → 바닥 90 → 현재 99
    values = [100, 110, 120, 110, 100, 90, 95, 99]
    r = region_metrics(_series(values), lookback_months=None).iloc[0]
    assert r["peak"] == 120 and r["peak_ym"] == pd.Timestamp("2020-03-01")
    assert r["drawdown_pct"] == pytest.approx((99 / 120 - 1) * 100)
    assert r["trough"] == 90 and r["rebound_pct"] == pytest.approx(10.0)
    assert r["months_since_peak"] == 5
    assert r["chg_3m_pct"] == pytest.approx((99 / 100 - 1) * 100)
    assert pd.isna(r["chg_12m_pct"])  # 12개월 전 값이 없음
    assert r["short_history"]


def test_lookback_limits_peak_window():
    values = [200] + [100] * 70 + [90]
    r = region_metrics(_series(values), lookback_months=60).iloc[0]
    assert r["peak"] == 100  # 60개월 밖의 200은 무시


def test_asof_ignores_later_months():
    values = [100, 120, 80, 130]
    r = region_metrics(_series(values), lookback_months=None, asof=pd.Timestamp("2020-03-01")).iloc[0]
    assert r["latest"] == 80 and r["peak"] == 120


def test_peak_now_means_zero_drawdown():
    r = region_metrics(_series([90, 95, 100]), lookback_months=None).iloc[0]
    assert r["drawdown_pct"] == 0 and r["rebound_pct"] == 0


def test_rebase():
    df = pd.concat([_series([50, 100], key="A"), _series([200, 100], key="B")])
    out = rebase(df, pd.Timestamp("2020-01-01"))
    assert out[out["region_key"] == "A"]["index_value"].tolist() == [100, 200]
    assert out[out["region_key"] == "B"]["index_value"].tolist() == [100, 50]
