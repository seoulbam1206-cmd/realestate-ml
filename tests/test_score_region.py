"""매매 점수 시점 규칙·등급·정답 검사 (PLAN.md MVP 3-10, 3-11).

핵심 검사는 피처와 같음: "바꿔도 같아야 한다" — 기준일 C(T) 이후 정보를 바꿔도 T 월 점수 요소는 그대로.
정답은 그 반대: T 이전 거래를 바꿔도 정답은 그대로 (정답 창이 점수 자료와 겹치지 않음).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from realestate.score import region

T = "202403"  # C(T) = 2024-01-30
C = pd.Timestamp(2024, 1, 30)


def _trades(seed: int = 0) -> pd.DataFrame:
    """시군구 2개 × 단지 3개 × 2022.1~2025.6 매달 몇 건, 중개·직거래 섞음."""
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for ym in pd.period_range("2022-01", "2025-06", freq="M"):
        for sgg in ("11110", "11140"):
            for cx in ("a", "b", "c"):
                for _ in range(int(rng.integers(2, 5))):
                    day = int(rng.integers(1, ym.days_in_month + 1))
                    lp = float(rng.normal(7.0 if sgg == "11110" else 6.5, 0.05))
                    rows.append(
                        {
                            "trade_key": f"t{k}", "ym": f"{ym.year}{ym.month:02d}", "sgg_cd": sgg,
                            "complex_key": f"{sgg}{cx}", "band": 3, "deal_date": pd.Timestamp(ym.year, ym.month, day),
                            "cdeal_day": pd.NaT, "dealing_gbn": "중개거래" if k % 4 else "직거래", "lp": lp,
                            "res": float(rng.normal(0, 0.05)),
                        }
                    )  # fmt: skip
                    k += 1
    return pd.DataFrame(rows)


def _survey() -> pd.DataFrame:
    months = pd.period_range("2018-01", "2025-12", freq="M")
    return pd.DataFrame(
        [{"sgg_cd": s, "ym": f"{p.year}{p.month:02d}", "value": 100 + 10 * np.sin(i / 7) + (s == "11140")}
         for s in ("11110", "11140") for i, p in enumerate(months)]
    )  # fmt: skip


def _el(trades: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    return region.elements(trades, survey, T).drop(columns="max_deal_date").set_index("sgg_cd")


def test_cutoff_is_report_lag():
    assert region.asof.cutoff_date(T) == date(2024, 1, 30)


def test_future_trades_do_not_change_elements():
    tr, sv = _trades(), _survey()
    base = _el(tr, sv)
    late = tr["deal_date"] > C
    changed = tr.copy()
    changed.loc[late, "res"] = 9.0  # 공개 전 거래의 가격
    changed.loc[late, "dealing_gbn"] = "중개거래"
    extra = tr[late].head(50).assign(trade_key=lambda d: d["trade_key"] + "x")  # 공개 전 거래 추가
    changed = pd.concat([changed, extra], ignore_index=True)
    pd.testing.assert_frame_equal(_el(changed, sv), base)


def test_cancel_after_cutoff_is_unknown():
    tr, sv = _trades(), _survey()
    base = _el(tr, sv)
    recent = tr[(tr["deal_date"] <= C) & (tr["ym"] >= "202311")].index[:20]
    late = tr.copy()
    late.loc[recent, "cdeal_day"] = C + pd.Timedelta(days=1)  # C(T) 다음 날 해제 → 그때는 정상 거래
    pd.testing.assert_frame_equal(_el(late, sv), base)
    known = tr.copy()
    known.loc[recent, "cdeal_day"] = C  # C(T) 에 공개된 해제는 반영
    assert not _el(known, sv)["cancel_ratio"].equals(base["cancel_ratio"])


def test_survey_after_t_minus_1_is_unused():
    tr, sv = _trades(), _survey()
    base = _el(tr, sv)
    changed = sv.copy()
    changed.loc[changed["ym"] >= T, "value"] = 1.0  # T 월 이후 조사 지수
    pd.testing.assert_frame_equal(_el(tr, changed), base)
    earlier = sv.copy()
    earlier.loc[earlier["ym"] == region.shift(T, -1), "value"] *= 0.5
    assert not _el(tr, earlier)["A"].equals(base["A"])


def test_month_agg_stops_at_t_minus_2():
    tr, sv = _trades(), _survey()
    base = _el(tr, sv)
    # T−1월 거래 (2024.2) 는 계약일이 모두 C(T) 이후 → 거래량·B 에 안 들어감
    assert (tr.loc[tr["ym"] == region.shift(T, -1), "deal_date"] > C).all()
    b = tr[(tr["ym"] >= "202311") & (tr["ym"] <= "202401") & (tr["deal_date"] <= C)]
    assert base.loc["11110", "v_recent"] == (b["sgg_cd"] == "11110").sum()


def test_b_uses_brokered_median_with_shrink():
    tr, sv = _trades(), _survey()
    el = _el(tr, sv)
    w = tr[(tr["ym"] >= "202311") & (tr["ym"] <= "202401") & (tr["deal_date"] <= C)]
    w = w[(w["sgg_cd"] == "11110") & (w["dealing_gbn"] == "중개거래")]
    n = len(w)
    assert el.loc["11110", "B"] == pytest.approx(-w["res"].median() * n / (n + region.SHRINK_K))
    direct = tr.copy()
    direct.loc[direct["dealing_gbn"] == "직거래", "res"] = -5.0  # 직거래는 B 에 안 씀
    assert _el(direct, sv).loc["11110", "B"] == pytest.approx(el.loc["11110", "B"])


def test_b_missing_when_few_trades():
    tr, sv = _trades(), _survey()
    tr.loc[(tr["sgg_cd"] == "11140") & (tr["ym"] >= "202311"), "res"] = np.nan  # 예측 없음
    el = _el(tr, sv)
    assert np.isnan(el.loc["11140", "B"])


def _grid(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {
        "t": [T] * n,
        "n12": [100] * n,
        "vol_ratio": [1.0] * n,
        "down3": [False] * n,
        "cancel_ratio": [0.02] * n,
    }
    return pd.DataFrame({**base, **cols})


def test_percentile_and_grades():
    x = np.arange(11, dtype=float)
    el = _grid(A=x, B=x, C=x, D=x)
    out = region.score(el, {"A": 0.3, "B": 0.3, "C": 0.25, "D": 0.15})
    assert out["P_A"].tolist() == pytest.approx(list(x * 10))
    assert out["score"].tolist() == pytest.approx(list(x * 10))
    assert out["grade"].tolist() == [region.NOT] * 5 + [region.CONSIDER] * 2 + [region.REC] * 4


def test_high_score_with_weak_trend_is_consider():
    x = np.arange(11, dtype=float)
    el = _grid(A=x, B=x, C=-x, D=x)  # 점수는 높아도 추세 백분위 < 40
    out = region.score(el, {"A": 0.4, "B": 0.4, "C": 0.0, "D": 0.2})
    assert out.loc[10, "score"] == pytest.approx(100)
    assert out.loc[10, "grade"] == region.CONSIDER


def test_rules_override_score():
    x = np.arange(4, dtype=float) + 10
    el = _grid(A=x, B=x, C=[np.log(0.96)] * 3 + [0.0], D=x)
    el.loc[0, "n12"] = 29  # R1
    el.loc[1, ["vol_ratio", "down3"]] = [0.5, True]  # R2 (C ≤ ln 0.97)
    el.loc[2, "cancel_ratio"] = 0.151  # R3
    el.loc[3, "D"] = np.nan  # 요소 결측 → R1
    out = region.score(el, {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    assert (out["grade"] == region.NEVER).all()
    assert out[["r1", "r2", "r3"]].values.tolist() == [
        [True, False, False], [False, True, False], [False, False, True], [True, False, False]
    ]  # fmt: skip


def test_labels_do_not_use_trades_before_t():
    tr, sv = _trades(), _survey()
    final = tr[["sgg_cd", "complex_key", "band", "ym", "lp"]]
    base = region.labels(final, sv, T, 6, "202512")
    changed = final.copy()
    changed.loc[changed["ym"] < T, "lp"] += 1.0  # 점수가 쓰는 기간의 가격
    pd.testing.assert_frame_equal(region.labels(changed, sv, T, 6, "202512"), base)


def test_labels_matched_pairs_and_excess(monkeypatch):
    rows = []
    for sgg, cx, p0, p1 in (("11110", "a", 7.0, 7.1), ("11110", "b", 6.0, 6.0), ("11140", "c", 6.5, 6.3)):
        for ym, lp in (("202403", p0), ("202404", p0), ("202409", p1), ("202410", p1)):
            rows.append({"sgg_cd": sgg, "complex_key": cx, "band": 3, "ym": ym, "lp": lp})
    rows.append({"sgg_cd": "11140", "complex_key": "d", "band": 3, "ym": "202403", "lp": 6.0})  # 짝 없음
    monkeypatch.setattr(region, "PAIR_MIN", 1)
    lab = region.labels(pd.DataFrame(rows), _survey(), T, 6, "202512").set_index("sgg_cd")
    assert lab.loc["11110", "R6"] == pytest.approx(0.05)  # (0.1·2 + 0·2) ÷ 4
    assert lab.loc["11140", "pairs6"] == 1
    nat = (0.1 * 2 + 0 * 2 - 0.2 * 2) / 6
    assert lab.loc["11140", "X6"] == pytest.approx(-0.2 - nat)


def test_labels_need_complete_window(monkeypatch):
    monkeypatch.setattr(region, "PAIR_MIN", 1)  # 가짜 자료는 시군구마다 단지 3개
    tr, sv = _trades(), _survey()
    final = tr[["sgg_cd", "complex_key", "band", "ym", "lp"]]
    lab = region.labels(final, sv, T, 12, "202604")  # W1 끝 = 2025.5 ≤ 2026.4
    assert lab["R12"].notna().all()
    lab = region.labels(final, sv, "202509", 12, "202609")  # W1 끝 = 2026.11 > 2026.9
    assert lab["R12"].isna().all()
