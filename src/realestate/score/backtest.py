"""매매 점수·등급 검증 지표 (PLAN.md MVP 3-11). 모두 시군구 × 기준월 단위.

- IC: 매월 Spearman(점수, 초과 수익률) → 평균, 양수 달 비율
- 등급별: 시군구-월 수, 초과 수익률 평균·중앙, 적중률(> 0), 수익률, 조사 지수 수익률·최대 하락
- 기준선: 같은 달 안에서 등급 섞기 (무작위), 다른 점수로 같은 등급 수만큼 자르기 ("A 만" 등)
- 우연인지: 원형 블록 부트스트랩 (블록 = 연속 h 개월, 정답 기간이 겹치는 달끼리 묶음)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realestate.score.region import CONSIDER, GRADES, NEVER, NOT, REC

IC_MIN_N = 20  # IC 를 계산하는 달의 최소 시군구 수


def monthly_ic(df: pd.DataFrame, col: str, x: str) -> pd.Series:
    """기준월 → Spearman(col, x). 값이 둘 다 있는 시군구가 IC_MIN_N 미만인 달은 빠짐."""
    d = df[["t", col, x]].dropna()
    out = {}
    for t, g in d.groupby("t", sort=True):
        if len(g) >= IC_MIN_N:
            out[t] = g[col].rank().corr(g[x].rank())
    return pd.Series(out, dtype=float)


def grade_table(df: pd.DataFrame, h: int, grade: str = "grade") -> pd.DataFrame:
    x = f"X{h}"
    d = df[df[x].notna()]
    rows = []
    for g in GRADES:
        part = d[d[grade] == g]
        row = {
            "grade": g, "n": len(part), "x_mean": part[x].mean(), "x_median": part[x].median(),
            "hit": (part[x] > 0).mean() if len(part) else np.nan, "r_mean": part[f"R{h}"].mean(),
            "s_ret_mean": part[f"s_ret{h}"].mean(),
        }  # fmt: skip
        if "mdd" in part:
            row |= {"mdd_mean": part["mdd"].mean(), "mdd_p10": part["mdd"].quantile(0.1)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("grade")


def spread(df: pd.DataFrame, x: str, grade: str = "grade") -> float:
    """추천 평균 − 고려 대상 아님 평균 (시군구-월 단위로 모음)."""
    return float(df.loc[df[grade] == REC, x].mean() - df.loc[df[grade] == NOT, x].mean())


def matched_grades(df: pd.DataFrame, col: str) -> pd.Series:
    """다른 점수(col)로 매긴 등급: 달마다 절대 불가는 그대로 두고, 나머지를 col 순서로 원래 등급 수만큼 자름."""
    out = df["grade"].copy()
    for _, g in df.groupby("t"):
        ok = g[g["grade"] != NEVER]
        n_rec, n_con = (ok["grade"] == REC).sum(), (ok["grade"] == CONSIDER).sum()
        order = ok[col].rank(method="first", ascending=False)
        out.loc[ok.index] = np.where(order <= n_rec, REC, np.where(order <= n_rec + n_con, CONSIDER, NOT))
    return out


def random_spreads(df: pd.DataFrame, x: str, n: int = 1000, seed: int = 0) -> np.ndarray:
    """같은 달 안에서 등급을 섞었을 때 (등급별 수 유지) 추천 − 고려 대상 아님 차이의 분포."""
    d = df[df[x].notna()]
    rng = np.random.default_rng(seed)
    groups = [(g["grade"].to_numpy(), g[x].to_numpy()) for _, g in d.groupby("t")]
    out = np.empty(n)
    for i in range(n):
        s_rec = n_rec = s_not = n_not = 0.0
        for gr, xs in groups:
            p = rng.permutation(gr)
            s_rec, n_rec = s_rec + xs[p == REC].sum(), n_rec + (p == REC).sum()
            s_not, n_not = s_not + xs[p == NOT].sum(), n_not + (p == NOT).sum()
        out[i] = s_rec / n_rec - s_not / n_not
    return out


def block_months(months: list[str], h: int, n: int, seed: int = 0) -> list[np.ndarray]:
    """원형 블록 부트스트랩: 연속 h 개월 블록을 뽑아 원래 길이만큼 이어 붙인 달 위치 목록 n 개."""
    m = len(months)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(m / h))
    return [
        np.concatenate([(s + np.arange(h)) % m for s in rng.integers(0, m, size=k)])[:m] for _ in range(n)
    ]


def bootstrap(df: pd.DataFrame, h: int, alt: str = "P_A", n: int = 1000, seed: int = 0) -> dict:
    """IC 평균, 추천 − 고려 대상 아님 차이, IC(점수) − IC(alt) 의 (값, 2.5%, 97.5%)."""
    x = f"X{h}"
    d = df[df[x].notna()]
    ic, ic_alt = monthly_ic(d, "score", x), monthly_ic(d, alt, x)
    months = sorted(set(ic.index) & set(ic_alt.index))
    per = (
        d.assign(rec=lambda v: v["grade"] == REC, nt=lambda v: v["grade"] == NOT)
        .groupby("t")
        .apply(
            lambda g: pd.Series(
                {
                    "s_rec": g.loc[g["rec"], x].sum(),
                    "n_rec": g["rec"].sum(),
                    "s_not": g.loc[g["nt"], x].sum(),
                    "n_not": g["nt"].sum(),
                }  # fmt: skip
            ),
            include_groups=False,
        )
        .reindex(months)
    )
    ic_v, alt_v, p = ic.reindex(months).to_numpy(), ic_alt.reindex(months).to_numpy(), per.to_numpy()

    def stats(idx: np.ndarray) -> tuple[float, float, float]:
        s = p[idx].sum(axis=0)
        return ic_v[idx].mean(), s[0] / s[1] - s[2] / s[3], (ic_v[idx] - alt_v[idx]).mean()

    full = stats(np.arange(len(months)))
    boot = np.array([stats(i) for i in block_months(months, h, n, seed)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    names = ("ic_mean", "spread", f"ic_minus_{alt}")
    return {k: (float(full[i]), float(lo[i]), float(hi[i])) for i, k in enumerate(names)} | {
        "months": len(months)
    }
