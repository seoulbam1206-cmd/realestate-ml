"""신고 지연(시점) 규칙 자동 검사 (PLAN.md MVP 3-3).

계약 월 M 인 거래의 피처는 기준일 C(M) = (M−1월 말일) − 30일 까지 공개된 정보만 써야 한다.
- 거래 이력: 계약일 ≤ C(M)
- 해제: 해제일 ≤ C(M) 일 때만 해제로 봄
- 월 집계: M−2월까지 (그 달도 C(M) 이후 계약분은 빠짐)
- R-ONE 조사 지수 M−1월까지, 실거래가격지수 M−2월까지

핵심 검사는 "바꿔도 같아야 한다": C(M) 이후 정보를 아무리 바꿔도 M 월 거래의 피처·기준선은 그대로여야 한다.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from realestate.features import asof
from realestate.features.trade_features import FEATURES, build_features

M = "202403"  # C(M) = 2024-02-29 − 30일 = 2024-01-30
C = date(2024, 1, 30)
# 두 경우: M−1월이 29일(기준일이 M−2월 안) / 31일(기준일이 M−1월 1일 → M−1월 거래 하루분이 h 에 들어옴)
CASES = [("202403", date(2024, 1, 30)), ("202402", date(2024, 1, 1))]


def _trades(seed: int = 0) -> pd.DataFrame:
    """단지 4개(동 2개, 시군구 2개) × 2023.1~2024.6 매달 몇 건씩."""
    rng = np.random.default_rng(seed)
    rows = []
    complexes = [
        ("cA", "1111010100", 84.9), ("cB", "1111010100", 59.8),
        ("cC", "1111010200", 84.9), ("cD", "1114010100", 114.0),
    ]  # fmt: skip
    k = 0
    for ym in pd.period_range("2023-01", "2024-06", freq="M"):
        for cx, bjd_cd, area in complexes:
            for _ in range(int(rng.integers(2, 6))):
                day = int(rng.integers(1, ym.days_in_month + 1))
                ppm2 = float(rng.normal(1000 if bjd_cd.startswith("11110") else 700, 40))
                rows.append(
                    {
                        "trade_key": f"t{k}", "sgg_cd": bjd_cd[:5], "bjd_cd": bjd_cd, "complex_key": cx,
                        "exclu_use_ar": area, "floor": int(rng.integers(1, 25)), "build_year": 2005,
                        "deal_date": pd.Timestamp(ym.year, ym.month, day), "price_per_m2": ppm2,
                        "deal_amount": int(ppm2 * area), "cdeal_day": pd.NaT, "is_canceled": False,
                        "dealing_gbn": "중개거래", "buyer_gbn": "개인", "seller_gbn": "개인",
                    }
                )  # fmt: skip
                k += 1
    return pd.DataFrame(rows)


def _rone() -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.period_range("2022-01", "2024-06", freq="M")
    rows = [{"sgg_cd": s, "ym": f"{p.year}{p.month:02d}", "value": 90 + i * 0.5}
            for s in ("11110", "11140") for i, p in enumerate(months)]  # fmt: skip
    survey = pd.DataFrame(rows)
    return survey, survey.assign(value=survey["value"] * 1.1)


def _features_for_m(
    trades: pd.DataFrame, survey: pd.DataFrame, real: pd.DataFrame, m: str = M
) -> pd.DataFrame:
    feats, meta = build_features(trades, [m], survey, real)
    asof.audit(meta, FEATURES)
    cols = ["trade_key", *FEATURES, "b0_ppm2", "b0_level", "b1_ppm2"]
    return feats[cols].sort_values("trade_key").reset_index(drop=True)


def test_cutoff_dates():
    assert asof.cutoff_date("202403") == date(2024, 1, 30)  # 2월 말(29일) − 30일
    assert asof.cutoff_date("202405") == date(2024, 3, 31)  # 4월 말(30일) − 30일
    assert asof.cutoff_date("202402") == date(2024, 1, 1)  # 1월 말(31일) − 30일
    assert asof.last_agg_month("202403") == "202401"
    assert asof.last_survey_month("202403") == "202402" and asof.last_real_month("202403") == "202401"


@pytest.mark.parametrize(("m", "c"), CASES)
def test_features_ignore_everything_after_cutoff(m, c):
    """C(M) 이후 계약된 거래의 가격·건수, C(M) 이후 해제, 늦은 R-ONE 값을 바꿔도 M 월 피처는 같다."""
    trades = _trades()
    survey, real = _rone()
    base = _features_for_m(trades, survey, real, m)
    assert len(base) and base["b0_ppm2"].notna().all()

    late = trades["deal_date"].dt.date > c
    month_start = pd.Timestamp(f"{m[:4]}-{m[4:]}-01")
    changed = trades.copy()
    changed.loc[late, "price_per_m2"] *= 3  # 공개 전 거래 가격 바꾸기 (M 월 자신 포함)
    changed.loc[late, "deal_amount"] *= 3
    changed.loc[late & (changed["deal_date"] < month_start), "floor"] = 99
    extra = trades[late & (trades["deal_date"] < month_start)].assign(  # 공개 전 거래 추가
        trade_key=lambda d: d["trade_key"] + "x", price_per_m2=5000.0
    )
    changed = pd.concat([changed, extra], ignore_index=True)
    public = changed.index[~late.reindex(changed.index, fill_value=True)]
    hist = public[:20].append(public[-15:])  # 오래된 거래 + 해제율 창(M−4~M−2)에 드는 최근 거래
    changed.loc[hist, "cdeal_day"] = pd.Timestamp(c + timedelta(days=1))  # C 다음 날 해제 → 당시엔 정상
    changed.loc[hist, "is_canceled"] = True
    survey2, real2 = survey.copy(), real.copy()
    survey2.loc[survey2["ym"] >= m, "value"] = 1  # M 월 이후 조사 지수
    real2.loc[real2["ym"] >= asof.last_survey_month(m), "value"] = 1  # M−1 월 이후 실거래가격지수

    after = _features_for_m(changed, survey2, real2, m)
    after = after[after["trade_key"].isin(base["trade_key"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(base, after, check_exact=False, rtol=1e-6)


def test_cancel_known_before_cutoff_is_used():
    """반대 방향: C(M) 이전에 공개된 해제는 반영돼야 한다 (규칙이 아무것도 안 보는 게 아님을 확인)."""
    trades = _trades()
    survey, real = _rone()
    base = _features_for_m(trades, survey, real)
    hist = trades.index[
        (trades["deal_date"].dt.date <= C - timedelta(days=40)) & (trades["complex_key"] == "cA")
    ]
    changed = trades.copy()
    changed.loc[hist, "cdeal_day"] = changed.loc[hist, "deal_date"] + pd.Timedelta(days=5)
    changed.loc[hist, "is_canceled"] = True
    after = _features_for_m(changed, survey, real)
    assert not np.allclose(base["cx36_n"].fillna(-1), after["cx36_n"].fillna(-1))


@pytest.mark.parametrize(("m", "c"), CASES)
def test_meta_records_sources_within_cutoff(m, c):
    feats, meta = build_features(_trades(), [m], *_rone())
    row = meta.set_index("ym").loc[m]
    assert row["hist_max_deal_date"] <= c
    assert row["agg_max_month"] <= asof.last_agg_month(m)
    assert row["survey_max_month"] <= asof.last_survey_month(m) and row[
        "real_max_month"
    ] <= asof.last_real_month(m)
    assert set(feats["ym"]) == {m}
    asof.audit(meta, FEATURES)


def test_audit_catches_violations():
    _, meta = build_features(_trades(), [M], *_rone())
    asof.audit(meta, FEATURES)
    for col, bad in [
        ("hist_max_deal_date", C + timedelta(days=1)),
        ("agg_max_month", "202402"),
        ("survey_max_month", "202403"),
        ("real_max_month", "202402"),
    ]:
        with pytest.raises(asof.AsOfViolation):
            asof.audit(meta.assign(**{col: bad}), FEATURES)
    for forbidden in ["rgst_date", "cdeal_day", "is_canceled", "apt_dong", "first_seen_at", "price_per_m2"]:
        with pytest.raises(asof.AsOfViolation):
            asof.audit(meta, [*FEATURES, forbidden])


def test_feature_list_has_no_forbidden_columns():
    assert not set(FEATURES) & asof.FORBIDDEN


def test_training_rows_only_public_labels():
    """기준일 D 모델의 학습 행: 계약일 ≤ D − 30일. D − 30일 이후 공개된 해제는 몰랐으므로 학습에 남김."""
    trades = _trades()
    d = date(2023, 12, 31)
    trades.loc[0, ["cdeal_day", "is_canceled"]] = [pd.Timestamp(2023, 1, 20), True]  # 일찍 공개된 해제 → 빠짐
    trades.loc[1, ["cdeal_day", "is_canceled"]] = [pd.Timestamp(2023, 12, 25), True]  # 늦게 공개 → 남음
    rows = asof.training_rows(trades, d)
    assert rows["deal_date"].max().date() <= d - timedelta(days=30)
    assert "t0" not in set(rows["trade_key"]) and "t1" in set(rows["trade_key"])


def test_rone_by_sgg_picks_smallest_region_with_history():
    from realestate.features.trade_features import rone_by_sgg

    def series(key, start):
        months = pd.date_range(start, "2026-08-01", freq="MS")
        return pd.DataFrame({"source": "survey", "region_key": key, "ym": months, "index_value": 100.0})

    idx = pd.concat([series("인천", "2003-11-01"), series("인천>제물포구", "2026-06-01"),
                     series("인천>연수구", "2003-11-01"), series("전국", "2003-11-01")])  # fmt: skip
    rmap = pd.DataFrame({"region_key": ["인천", "인천", "인천>제물포구", "인천>연수구", "전국", "전국"],
                         "sgg_cd": ["28125", "28185", "28125", "28185", "28125", "28185"]})  # fmt: skip
    out = rone_by_sgg(idx, rmap, "survey").drop_duplicates("sgg_cd").set_index("sgg_cd")["region_key"]
    assert out.to_dict() == {"28125": "인천", "28185": "인천>연수구"}
