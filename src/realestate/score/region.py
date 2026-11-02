"""시군구 매매 점수·등급과 정답 (PLAN.md MVP 3-10, 3-11). 기준월 T = T 월 초에 매기는 점수.

- 시점: 실거래는 기준일 C(T) = (T−1월 말일) − 30일 까지 계약된 것, 해제는 해제일 ≤ C(T) 만 반영.
  월 집계는 T−2월까지, R-ONE 조사 지수는 T−1월까지 (asof 와 같은 규칙)
- A 고점 대비 하락률 · C 3개월 추세: 조사 지수 (시군구, 없으면 시도 — trade_features.rone_by_sgg)
- B 모델 예측가 대비 저평가: 중개거래 잔차 ln(실제가) − ln(예측가) 중위값을 건수로 0 쪽으로 당김
- D 거래량 변화: log(최근 3개월 ÷ 전년 같은 기간)
- 등급: R2·R3 (매매 절대 불가) 먼저, R1 은 등급 없이 "데이터 부족" (13장 조정 1회), 그다음 점수
- 정답: 같은 단지·면적대의 W0(T~T+2) → W1(T+h~T+h+2) 가격 변화. 지금 기준 해제 아닌 거래로 계산 (평가용)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realestate.features import asof

SHRINK_K = 30  # B 를 n ÷ (n + 30) 만큼만 반영
B_MIN_N = 10
N12_MIN = 30  # R1: 12개월 정상 거래
VOL_DROP = 0.5  # R2: 거래량 절반 이하
DROP_3M = float(np.log(0.97))  # R2: 3개월 −3% 이하
CANCEL_MAX = 0.15  # R3
GRADE_HI, GRADE_MID, C_MIN = 70, 50, 40
REC, CONSIDER, NOT, NEVER = "매매 추천", "매매 고려", "매매 고려 대상 아님", "매매 절대 불가"
GRADES = [REC, CONSIDER, NOT, NEVER]
NODATA = "데이터 부족"  # 등급 아님: R1 (판단 불가). 4등급 밖에 따로 표시
RULE_VERSION = "2"  # 1: R1~R3 → 절대 불가. 2: R1 → 데이터 부족 (PLAN 13장 튜닝 후 조정 1회, 2026-11-02)
ELEMENTS = ["A", "B", "C", "D"]
PAIR_MIN = 10  # 정답: 시군구 짝 최소
CLIP = 0.5


def shift(ym: str, k: int) -> str:
    p = pd.Period(f"{ym[:4]}-{ym[4:]}", freq="M") + k
    return f"{p.year}{p.month:02d}"


def elements(trades: pd.DataFrame, survey: pd.DataFrame, t: str) -> pd.DataFrame:
    """기준월 T 의 시군구별 요소 원값과 규칙 판정 값 (sgg_cd 인덱스).

    trades: trade_key, ym, deal_date, cdeal_day, sgg_cd, dealing_gbn, res (res = ln 실제가 − ln 예측가, 없으면 NaN)
    survey: sgg_cd, ym, value (조사 지수)
    """
    c = pd.Timestamp(asof.cutoff_date(t))
    seen = trades[trades["deal_date"] <= c]
    known_cancel = seen["cdeal_day"].notna() & (seen["cdeal_day"] <= c)
    h = seen[~known_cancel]

    def window(df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
        return df[(df["ym"] >= shift(t, a)) & (df["ym"] <= shift(t, b))]

    out = pd.DataFrame({"n12": window(h, -13, -2).groupby("sgg_cd").size()})
    out["v_recent"] = window(h, -4, -2).groupby("sgg_cd").size()
    out["v_prev"] = window(h, -16, -14).groupby("sgg_cd").size()
    out[["v_recent", "v_prev"]] = out[["v_recent", "v_prev"]].fillna(0)
    out["vol_ratio"] = out["v_recent"] / out["v_prev"].where(out["v_prev"] > 0)
    out["D"] = np.log(out["vol_ratio"].where(out["vol_ratio"] > 0))
    w = window(seen, -4, -2)
    out["cancel_ratio"] = (w["cdeal_day"].notna() & (w["cdeal_day"] <= c)).groupby(w["sgg_cd"]).mean()

    b = window(h, -4, -2)
    b = (
        b[(b["dealing_gbn"] == "중개거래") & b["res"].notna()]
        .groupby("sgg_cd")["res"]
        .agg(["median", "size"])
    )
    out["b_n"] = b["size"].reindex(out.index).fillna(0).astype(int)
    out["b_med"] = b["median"]
    out["B"] = (-out["b_med"] * out["b_n"] / (out["b_n"] + SHRINK_K)).where(out["b_n"] >= B_MIN_N)

    piv = survey[survey["ym"] <= asof.last_survey_month(t)].pivot(
        index="sgg_cd", columns="ym", values="value"
    )
    col = lambda k: piv[shift(t, -k)] if shift(t, -k) in piv else pd.Series(np.nan, index=piv.index)  # noqa: E731
    i1 = col(1)
    peak = piv[[m for m in (shift(t, -k) for k in range(60, 0, -1)) if m in piv]].max(axis=1)
    out["A"] = (1 - i1 / peak).reindex(out.index)
    out["C"] = (np.log(i1) - np.log(col(4))).reindex(out.index)
    down = (col(1) < col(2)) & (col(2) < col(3)) & (col(3) < col(4))
    out["down3"] = down.reindex(out.index).fillna(False).astype(bool)
    out["max_deal_date"] = h.groupby("sgg_cd")["deal_date"].max()  # 시점 검사용 기록
    out.index.name = "sgg_cd"
    return out.reset_index().assign(t=t)


def percentile(x: pd.Series) -> pd.Series:
    """그 달 값이 있는 시군구 안에서 0~100 (같은 값은 평균 순위)."""
    n = x.notna().sum()
    return 100 * (x.rank(method="average") - 1) / max(n - 1, 1)


def score(el: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """el: elements() 를 여러 달 이어 붙인 것. 달마다 백분위 → 가중합 → 등급."""
    out = el.copy()
    for e in ELEMENTS:
        out[f"P_{e}"] = out.groupby("t")[e].transform(percentile)
    p = out[[f"P_{e}" for e in ELEMENTS]]
    out["score"] = (p * [weights[e] for e in ELEMENTS]).sum(axis=1, min_count=4).where(p.notna().all(axis=1))
    out["r1"] = (out["n12"] < N12_MIN) | out["score"].isna()
    out["r2"] = (out["vol_ratio"] <= VOL_DROP) & out["down3"] & (out["C"] <= DROP_3M)
    out["r3"] = out["cancel_ratio"] > CANCEL_MAX
    out["grade"] = np.select(
        [
            out["r2"] | out["r3"],  # R1 과 겹쳐도 절대 불가 (R2·R3 는 B 없이 판정 가능)
            out["r1"],
            (out["score"] >= GRADE_HI) & (out["P_C"] >= C_MIN),
            out["score"] >= GRADE_MID,
        ],
        [NEVER, NODATA, REC, CONSIDER],
        NOT,
    )
    return out


def labels(final: pd.DataFrame, survey: pd.DataFrame, t: str, h: int, last_month: str) -> pd.DataFrame:
    """정답 (sgg_cd 별): R 같은 단지 수익률, X 전국 대비 초과, pairs 짝 수, s_ret 조사 지수 수익률, mdd 12개월 최대 하락.

    final: 지금 기준 해제 아닌 거래 (sgg_cd, complex_key, band, ym, lp). last_month: 정답에 쓸 수 있는 마지막 계약 월
    """
    out = pd.DataFrame(columns=["sgg_cd", f"R{h}", f"X{h}", f"pairs{h}"])
    if shift(t, h + 2) <= last_month:

        def med(a: int, b: int) -> pd.DataFrame:
            w = final[(final["ym"] >= shift(t, a)) & (final["ym"] <= shift(t, b))]
            return w.groupby(["sgg_cd", "complex_key", "band"], observed=True)["lp"].agg(["median", "size"])

        j = (
            med(0, 2)
            .merge(med(h, h + 2), left_index=True, right_index=True, suffixes=("0", "1"))
            .reset_index()
        )
        j["r"] = (j["median1"] - j["median0"]).clip(-CLIP, CLIP)
        j["w"] = np.minimum(j["size0"], j["size1"])
        j["wr"] = j["w"] * j["r"]
        g = j.groupby("sgg_cd").agg(wr=("wr", "sum"), w=("w", "sum"), pairs=("r", "size"))
        nat = j["wr"].sum() / j["w"].sum()
        r = (g["wr"] / g["w"]).where(g["pairs"] >= PAIR_MIN)
        out = pd.DataFrame({f"R{h}": r, f"X{h}": r - nat, f"pairs{h}": g["pairs"]})
        out.index.name = "sgg_cd"
        out = out.reset_index()
    piv = survey.pivot(index="sgg_cd", columns="ym", values="value")
    # 조사 지수도 last_month 를 넘는 달은 안 읽음 (튜닝 정답이 평가 기간 지수를 쓰지 않게)
    get = lambda k: (  # noqa: E731
        piv[shift(t, k)]
        if shift(t, k) in piv and shift(t, k) <= last_month
        else pd.Series(np.nan, index=piv.index)
    )
    s = pd.DataFrame({f"s_ret{h}": np.log(get(h + 1)) - np.log(get(1))})
    if h == 12:
        s["mdd"] = pd.concat([np.log(get(k)) - np.log(get(-1)) for k in range(12)], axis=1).min(
            axis=1, skipna=False
        )
    s.index.name = "sgg_cd"
    return out.merge(s.reset_index(), on="sgg_cd", how="outer").assign(t=t)
