"""화면용 공개 집계표 → data/public/*.parquet (개별 거래 행 없음). TiDB 업로드는 scripts/publish.py.

| 표 | 단위 | 내용 |
|---|---|---|
| region_name | 시군구 | 시도·시군구 이름, 권역 (수도권 = 서울·인천·경기) |
| agg_region_month | 시군구 × 계약월 | 정상 거래 수, 해제 수, 중위 거래가(만원), ㎡당가 p25/중위/p75. 거래 < 5건이면 가격 칸은 비움 |
| region_score | 시군구 (최신 기준월 하나) | 4요소 원값·백분위, 점수, 등급(데이터 부족 포함), 걸린 규칙 |
| bt_ic | 기간 × 정답 기간 × 권역 × 값 | 매월 Spearman IC 평균, 양수 달 비율, 95% 구간 (평가만) |
| bt_grade | 기간 × 정답 기간 × 등급 | 건수, 정답 있는 건수, 초과 수익 평균·중앙, 적중률, 12개월 최대 하락 |
| bt_baseline | 정답 기간 × 기준선 | 추천 − 고려 대상 아님 차이 (4요소 · "A 만" · "B 만" · 무작위 섞기) |
| bt_meta | 키-값 | 설정·판정·MLflow run |

- 점수·등급은 config/score.json 고정 설정 그대로 (eval 다시 안 함). 부트스트랩·섞기는 eval·사후 분석과 같은 seed 라 리포트 숫자와 같음
- 실거래 집계: 계약월 2020.1 ~ 2026.10 (피처표 마지막 달). 최근 2개월은 신고 기한(계약 후 30일) 때문에 덜 잡혔을 수 있어
  화면에서 점선으로 구분. 지금 기준 해제 아닌 거래

실행: uv run python scripts/build_public.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import duckdb
import mlflow
import numpy as np
import pandas as pd
from score_backtest import CAPITAL, CONFIG, LAST_LABEL_MONTH, PERIODS, load, months_of

from realestate.config import BJD_RAW, ROOT
from realestate.ingest import bjd
from realestate.models import tracking
from realestate.models.periods import FEATURE_DIR
from realestate.score import backtest as bt
from realestate.score import region

OUT = ROOT / "data" / "public"
PRICE_MIN_N = 5  # 거래가 이보다 적은 시군구-월은 가격 칸을 비움 (한두 건이면 개별 거래가가 그대로 드러남)
RULE_NAMES = {"r2": "R2 거래 급감 + 하락 지속", "r3": "R3 해제 급증"}
VALUES = {"score": "4요소 점수", "P_A": "A 고점 대비 하락률", "P_B": "B 모델 저평가", "P_C": "C 3개월 추세",
          "P_D": "D 거래량 변화"}  # fmt: skip
AREAS = {"전국": None, "수도권": True, "지방": False}
N_BOOT = 1000


def ym_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str) + "01", format="%Y%m%d").dt.date


def region_names(sgg: list[str]) -> pd.DataFrame:
    codes = bjd.load(BJD_RAW)
    act = codes[codes["is_active"]]
    top = act[act["bjd_cd"].str.endswith("00000000")]
    sido = dict(zip(top["bjd_cd"].str[:2], top["full_nm"], strict=True))
    sido.setdefault("36", "세종특별자치시")
    lvl = act[act["bjd_cd"].str.endswith("00000")].drop_duplicates("sgg_cd").set_index("sgg_cd")["full_nm"]
    rows = []
    for c in sgg:
        s_nm = sido[c[:2]]
        full = lvl.get(c, s_nm)
        name = full[len(s_nm) :].strip() or s_nm
        rows.append({"sgg_cd": c, "sido_cd": c[:2], "sido_nm": s_nm, "sgg_nm": name,
                     "zone": "수도권" if c[:2] in CAPITAL else "지방"})  # fmt: skip
    return pd.DataFrame(rows)


def agg_month() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    df = con.sql(
        f"""
        SELECT sgg_cd, ym,
               count(*) FILTER (WHERE NOT is_canceled) AS trade_cnt,
               count(*) FILTER (WHERE is_canceled) AS cancel_cnt,
               median(deal_amount) FILTER (WHERE NOT is_canceled) AS median_price,
               quantile_cont(ppm2, 0.25) FILTER (WHERE NOT is_canceled) AS p25_ppm2,
               median(ppm2) FILTER (WHERE NOT is_canceled) AS median_ppm2,
               quantile_cont(ppm2, 0.75) FILTER (WHERE NOT is_canceled) AS p75_ppm2
        FROM read_parquet('{(FEATURE_DIR / "ym=*.parquet").as_posix()}', union_by_name = true)
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()
    few = df["trade_cnt"] < PRICE_MIN_N
    df.loc[few, ["median_price", "p25_ppm2", "median_ppm2", "p75_ppm2"]] = np.nan
    df["ym"] = ym_date(df["ym"])
    return df


def latest_score(cfg: dict) -> pd.DataFrame:
    t = months_of("live")[-1]
    sc = region.score(load("live"), cfg["weights"])
    sc = sc[sc["t"] == t].copy()
    sc["rule_flags"] = sc.apply(lambda r: ",".join(k.upper() for k in RULE_NAMES if r[k]), axis=1)
    out = pd.DataFrame({
        "asof_ym": ym_date(sc["t"]), "sgg_cd": sc["sgg_cd"], "grade": sc["grade"], "score": sc["score"],
        "p_a": sc["P_A"], "p_b": sc["P_B"], "p_c": sc["P_C"], "p_d": sc["P_D"],
        "drawdown_pct": sc["A"] * 100, "underval_pct": sc["B"] * 100, "trend_3m_pct": sc["C"] * 100,
        "volume_ratio": sc["vol_ratio"], "trade_12m": sc["n12"], "cancel_ratio": sc["cancel_ratio"],
        "r1": sc["r1"], "r2": sc["r2"], "r3": sc["r3"], "rule_flags": sc["rule_flags"],
        "sido_level_index": sc["sido_level"],
        "model_version": f"score {cfg['weights_hash']} · rule {cfg['rule_version']} · lgbm {cfg['lgbm_params_hash']}",
    })  # fmt: skip
    return out.reset_index(drop=True)


def ic_rows(res: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for p, sc in res.items():
        cap = sc["sgg_cd"].str[:2].isin(CAPITAL)
        for h in (6, 12):
            for area, flag in AREAS.items():
                d = sc if flag is None else sc[cap == flag]
                for col, name in VALUES.items():
                    ic = bt.monthly_ic(d, col, f"X{h}")
                    if ic.empty:
                        continue
                    lo = hi = np.nan
                    if p == "eval":  # 95% 구간은 평가만 (리포트와 같은 방법·seed)
                        v = ic.to_numpy()
                        boot = [v[i].mean() for i in bt.block_months(list(ic.index), h, N_BOOT, seed=0)]
                        lo, hi = np.percentile(boot, [2.5, 97.5])
                    rows.append({"period": p, "horizon_m": h, "zone": area, "value_key": col, "value_name": name,
                                 "ic_mean": ic.mean(), "ci_lo": lo, "ci_hi": hi, "pos_share": (ic > 0).mean(),
                                 "months": len(ic)})  # fmt: skip
    return pd.DataFrame(rows)


def grade_rows(res: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for p, sc in res.items():
        for h in (6, 12):
            t = bt.grade_table(sc, h)
            for g, r in t.iterrows():
                rows.append({"period": p, "horizon_m": h, "grade": g, "n_all": int((sc["grade"] == g).sum()),
                             "n": int(r["n"]), "x_mean": r["x_mean"], "x_median": r["x_median"], "hit": r["hit"],
                             "mdd_mean": r.get("mdd_mean", np.nan) if h == 12 else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def baseline_rows(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.assign(grade_A=bt.matched_grades(ev, "P_A"), grade_B=bt.matched_grades(ev, "P_B"))
    rows = []
    for h in (6, 12):
        x = f"X{h}"
        d = ev[ev[x].notna()]
        b = bt.bootstrap(ev, h, "P_A")
        r = bt.random_spreads(ev, x)
        rows += [
            {"horizon_m": h, "name": "4요소 점수", "spread": b["spread"][0], "lo": b["spread"][1], "hi": b["spread"][2]},
            {"horizon_m": h, "name": '"A 만" (같은 등급 수로 자름)', "spread": bt.spread(d, x, "grade_A"), "lo": np.nan, "hi": np.nan},
            {"horizon_m": h, "name": '"B 만" (같은 등급 수로 자름)', "spread": bt.spread(d, x, "grade_B"), "lo": np.nan, "hi": np.nan},
            {"horizon_m": h, "name": "무작위 섞기 1,000회 (95% 범위)", "spread": float(np.median(r)),
             "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))},
        ]  # fmt: skip
    return pd.DataFrame(rows)


def meta(cfg: dict, ev: pd.DataFrame) -> pd.DataFrame:
    tracking.use_experiment("region_score_backtest")
    run = mlflow.search_runs(
        filter_string=f"tags.kind = 'score_eval' and tags.weights_hash = '{cfg['weights_hash']}'",
        output_format="pandas",
    )
    assert len(run) == 1, f"평가 run 이 {len(run)}개"
    b6 = bt.bootstrap(ev, 6, "P_A")
    items = {
        "weights": json.dumps(cfg["weights"]), "cand": cfg["cand"], "weights_hash": cfg["weights_hash"],
        "rule_version": cfg["rule_version"], "lgbm_params_hash": cfg["lgbm_params_hash"],
        "fixed_at": cfg["fixed_at"], "eval_run_id": run["run_id"].iloc[0], "verdict": run["tags.verdict"].iloc[0],
        "tuning_period": "-".join(PERIODS["tuning"]), "eval_period": "-".join(PERIODS["eval"]),
        "check_period": "-".join(PERIODS["check"]), "label_last_month": LAST_LABEL_MONTH,
        "eval_ic6": f"{b6['ic_mean'][0]:.3f}", "built_at": f"{datetime.now():%Y-%m-%d %H:%M}",
    }  # fmt: skip
    return pd.DataFrame({"k": list(items), "v": [str(v) for v in items.values()]})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    res = {p: region.score(load(p), cfg["weights"]) for p in ("tuning", "eval", "check")}
    agg = agg_month()
    tables = {
        "region_name": region_names(sorted(agg["sgg_cd"].unique())),
        "agg_region_month": agg,
        "region_score": latest_score(cfg),
        "bt_ic": ic_rows(res),
        "bt_grade": grade_rows(res),
        "bt_baseline": baseline_rows(res["eval"]),
        "bt_meta": meta(cfg, res["eval"]),
    }
    assert set(tables["region_score"]["sgg_cd"]) <= set(tables["region_name"]["sgg_cd"])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"\n{name}: {len(df):,}행 × {df.shape[1]}열")
        print("  " + ", ".join(f"{c}({t})" for c, t in df.dtypes.astype(str).items()))


if __name__ == "__main__":
    main()
