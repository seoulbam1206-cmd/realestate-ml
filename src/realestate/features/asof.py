"""시점(as-of) 규칙: 계약 월 M 인 거래의 피처가 쓸 수 있는 정보의 끝 (PLAN.md MVP 3-3, 9-1).

기준일 C(M) = (M−1월 말일) − 30일. 실거래는 계약 후 30일 안에 신고하므로, M 월 초(= M−1월 말일)에
공개됐다고 확신할 수 있는 것은 C(M) 까지 계약된 거래뿐이다. 해제 신고도 30일 안이라 해제일 ≤ C(M) 만 반영.
- 월 집계: M−2월까지 (그 달도 C(M) 이후 계약분은 빠짐. 2월이 끼면 1일분)
- R-ONE: 조사 지수 M−1월까지, 실거래가격지수 M−2월까지 (발표 지연)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

REPORT_DAYS = 30

# 피처로 쓰면 안 되는 칸: 계약 뒤에 생기거나 바뀌는 정보, 정답 자체
FORBIDDEN = frozenset(
    {
        "rgst_date", "cdeal_day", "cdeal_type", "is_canceled", "apt_dong", "first_seen_at", "last_seen_at",
        "missing_count", "source", "source_file", "deal_amount", "price_per_m2", "ppm2", "lp", "trade_key",
    }
)  # fmt: skip


class AsOfViolation(AssertionError):
    """피처가 기준일 이후 정보를 썼다."""


def _period(ym: str) -> pd.Period:
    return pd.Period(f"{ym[:4]}-{ym[4:]}", freq="M")


def _ym(p: pd.Period) -> str:
    return f"{p.year}{p.month:02d}"


def info_date(ym: str) -> date:
    """M 월 예측의 정보 기준일 = M−1월 말일."""
    return (_period(ym) - 1).end_time.date()


def cutoff_date(ym: str) -> date:
    return info_date(ym) - timedelta(days=REPORT_DAYS)


def last_agg_month(ym: str) -> str:
    return _ym(_period(ym) - 2)


def last_survey_month(ym: str) -> str:
    return _ym(_period(ym) - 1)


def last_real_month(ym: str) -> str:
    return _ym(_period(ym) - 2)


def training_rows(trades: pd.DataFrame, d: date) -> pd.DataFrame:
    """기준일 D 에 학습할 수 있는 행: 계약일 ≤ D − 30일, 그때까지 공개된 해제(해제일 ≤ D − 30일)는 뺌."""
    c = pd.Timestamp(d - timedelta(days=REPORT_DAYS))
    known_cancel = trades["cdeal_day"].notna() & (trades["cdeal_day"] <= c)
    return trades[(trades["deal_date"] <= c) & ~known_cancel]


def audit(meta: pd.DataFrame, features: Iterable[str]) -> None:
    """피처 생성 기록(meta)과 피처 목록을 검사. 어기면 AsOfViolation (학습·리포트 전에 반드시 호출)."""
    bad = sorted(set(features) & FORBIDDEN)
    if bad:
        raise AsOfViolation(f"금지 칸이 피처에 있음: {bad}")
    problems = []
    for r in meta.itertuples(index=False):
        checks = [
            ("hist_max_deal_date", r.hist_max_deal_date, cutoff_date(r.ym)),
            ("hist_max_cancel_day", r.hist_max_cancel_day, cutoff_date(r.ym)),
            ("agg_max_month", r.agg_max_month, last_agg_month(r.ym)),
            ("survey_max_month", r.survey_max_month, last_survey_month(r.ym)),
            ("real_max_month", r.real_max_month, last_real_month(r.ym)),
        ]
        for name, used, limit in checks:
            if used is not None and not pd.isna(used) and _as_comparable(used) > _as_comparable(limit):
                problems.append(f"{r.ym}: {name}={used} > 허용 {limit}")
    if problems:
        raise AsOfViolation("; ".join(problems[:10]))


def _as_comparable(v):
    if isinstance(v, str):
        return v
    return pd.Timestamp(v).date()
