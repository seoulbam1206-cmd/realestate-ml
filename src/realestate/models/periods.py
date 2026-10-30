"""시세 모델 기간 구분과 채점 행 (PLAN.md MVP 3-2). 기준선·LightGBM·XGBoost 가 모두 이것을 쓴다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from realestate.config import DATA_FEATURES

FEATURE_DIR = DATA_FEATURES / "trade_features"
PERIODS = {  # 이름: (첫 달, 끝 달) — 계약 월 T 기준
    "tuning": ("202301", "202312"),
    "eval": ("202401", "202509"),
    "check": ("202510", "202608"),
}


def period_of(ym: str) -> str | None:
    for name, (a, b) in PERIODS.items():
        if a <= ym <= b:
            return name
    return None


def load_scoring_rows(columns: list[str] | None = None, feature_dir: Path = FEATURE_DIR) -> pd.DataFrame:
    """튜닝·평가·확인 기간의 채점 행: 지금 기준으로 해제되지 않은 거래 (실제로 성립한 거래만 채점)."""
    first, last = min(a for a, _ in PERIODS.values()), max(b for _, b in PERIODS.values())
    files = sorted(p for p in feature_dir.glob("ym=*.parquet") if first <= p.stem[3:] <= last)
    df = pd.concat((pd.read_parquet(p, columns=columns) for p in files), ignore_index=True)
    df = df[~df["is_canceled"]].copy()
    df["period"] = df["ym"].map(period_of)
    return df
