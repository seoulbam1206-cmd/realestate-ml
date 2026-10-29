"""지수 시계열 지표: 고점 대비 하락률, 기간별 변화율, 바닥 대비 반등률.

입력은 세로형 지수 표(region_key, ym, index_value) 한 종류(source)만.
asof 를 주면 그 달까지의 값만 써서 계산한다 (이후 백테스트에서 그대로 재사용).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANGE_HORIZONS = (3, 6, 12)
MIN_HISTORY_MONTHS = 24  # 이보다 짧은 시계열은 순위에서 기본 제외


def _change(s: pd.Series, latest_ym: pd.Timestamp, months: int) -> float:
    past = s.get(latest_ym - pd.DateOffset(months=months))
    if past is None or pd.isna(past) or past == 0:
        return np.nan
    return (s[latest_ym] / past - 1) * 100


def region_metrics(
    index: pd.DataFrame,
    lookback_months: int | None = 60,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """지역별 지표 한 줄씩.

    - peak: 최근 lookback_months 안의 최고값 (None 이면 전체 기간)
    - drawdown_pct: 최신값 ÷ 고점 − 1 (%). 0 이면 지금이 고점
    - rebound_pct: 고점 이후 최저점 대비 최신값 상승률 (%)
    """
    df = index if asof is None else index[index["ym"] <= asof]
    out = []
    for key, g in df.groupby("region_key", sort=False):
        s = g.set_index("ym")["index_value"].sort_index()
        latest_ym = s.index[-1]
        window = (
            s if lookback_months is None else s[s.index > latest_ym - pd.DateOffset(months=lookback_months)]
        )
        peak_ym = window.idxmax()
        peak = window[peak_ym]
        after_peak = window[window.index >= peak_ym]
        trough_ym = after_peak.idxmin()
        row = {
            "region_key": key,
            "latest_ym": latest_ym,
            "latest": s.iloc[-1],
            "peak": peak,
            "peak_ym": peak_ym,
            "drawdown_pct": (s.iloc[-1] / peak - 1) * 100,
            "months_since_peak": (latest_ym.year - peak_ym.year) * 12 + latest_ym.month - peak_ym.month,
            "trough": after_peak[trough_ym],
            "trough_ym": trough_ym,
            "rebound_pct": (s.iloc[-1] / after_peak[trough_ym] - 1) * 100,
            "n_months": len(s),
            "short_history": len(s) < MIN_HISTORY_MONTHS,
        }
        for h in CHANGE_HORIZONS:
            row[f"chg_{h}m_pct"] = _change(s, latest_ym, h)
        out.append(row)
    return pd.DataFrame(out)


def rebase(index: pd.DataFrame, base_ym: pd.Timestamp) -> pd.DataFrame:
    """base_ym 값을 100 으로 다시 맞춘다. 그 달 값이 없는 지역은 빠진다."""
    base = index[index["ym"] == base_ym].set_index("region_key")["index_value"]
    out = index[index["region_key"].isin(base.index)].copy()
    out["index_value"] = out["index_value"] / out["region_key"].map(base) * 100
    return out
