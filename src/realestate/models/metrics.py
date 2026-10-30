"""시세 모델 평가 지표 (PLAN.md MVP 3-6). 모든 모델·기준선을 같은 거래로 채점한다.

- MAE: 평균 |예측 거래가 − 실제 거래가| (만원)
- MAPE: 평균 |예측 − 실제| ÷ 실제 × 100 (%)
- MdAPE: 위 비율의 중앙값, within10: 오차율 ≤ 10% 비율
- 기간 합산은 거래 단위 (월별 값의 평균 아님)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    err = np.abs(p - a)
    ape = err / a * 100
    return {
        "n": int(len(a)),
        "mae": float(err.mean()),
        "mape": float(ape.mean()),
        "mdape": float(np.median(ape)),
        "within10": float((ape <= 10).mean() * 100),
    }


def metrics_by(
    df: pd.DataFrame, group: str | list[str], preds: dict[str, str], actual: str = "deal_amount"
) -> pd.DataFrame:
    """그룹별 × 예측별 지표. preds = {이름: 예측 칸}."""
    rows = []
    for key, g in df.groupby(group, observed=True, sort=True):
        for name, col in preds.items():
            rows.append({"group": key, "model": name, **price_metrics(g[actual], g[col])})
    return pd.DataFrame(rows)


def block_bootstrap_diff(
    df: pd.DataFrame, pred_a: str, pred_b: str, actual: str = "deal_amount", block: str = "ym",
    n_boot: int = 1000, seed: int = 0,
) -> dict[str, tuple[float, float, float]]:  # fmt: skip
    """(B − A) 의 MAE·MAPE 차이와 95% 구간. 월(block) 단위로 다시 뽑아 같은 달 거래끼리의 상관을 유지."""
    a = np.abs(df[pred_a].to_numpy(float) - df[actual].to_numpy(float))
    b = np.abs(df[pred_b].to_numpy(float) - df[actual].to_numpy(float))
    act = df[actual].to_numpy(float)
    blocks = pd.DataFrame({"blk": df[block].to_numpy(), "da": b - a, "dp": (b - a) / act * 100})
    sums = blocks.groupby("blk").agg(n=("da", "size"), da=("da", "sum"), dp=("dp", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    n, da, dp = (sums[c].to_numpy()[idx].sum(axis=1) for c in ("n", "da", "dp"))
    out = {}
    for name, total, boot in (("mae", sums["da"].sum(), da / n), ("mape", sums["dp"].sum(), dp / n)):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = (float(total / sums["n"].sum()), float(lo), float(hi))
    return out
