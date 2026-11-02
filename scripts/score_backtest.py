"""매매 점수·등급 백테스트 (PLAN.md MVP 3-10, 3-11)

- build: 기준월 T 마다 요소 원값 + 정답 → data/score/T=YYYYMM.parquet (하나씩 저장, 있으면 건너뜀).
        달마다 시점 검사: 쓴 거래의 최대 계약일 ≤ C(T), B 창의 중개거래는 모두 예측이 있어야 함 (어기면 멈춤)
- tune: 튜닝 T 2022.2~2023.4 에서 가중치 후보 5개 → 6개월 IC 평균 최대 → config/score.json 고정
- eval: 고정 설정으로 평가 T 2024.1~2025.9 (+ 확인 2025.10~2026.8 참고) 한 번 → MLflow region_score_backtest,
        reports/score_backtest.md. 같은 설정의 평가가 이미 있으면 거부

실행: uv run python scripts/score_backtest.py build | tune | eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime

import duckdb
import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa

from realestate.config import BJD_RAW, REFERENCE_CLEAN, RONE_CLEAN, ROOT
from realestate.features import asof
from realestate.features.trade_features import rone_by_sgg
from realestate.ingest import bjd
from realestate.models import tracking
from realestate.models.periods import FEATURE_DIR
from realestate.score import backtest as bt
from realestate.score import region

OUT = ROOT / "data" / "score"
CONFIG = ROOT / "config" / "score.json"
LGBM_CONFIG = ROOT / "config" / "lgbm.json"
TUNING_CSV = ROOT / "reports" / "score_tuning.csv"
REPORT = ROOT / "reports" / "score_backtest.md"
PERIODS = {"tuning": ("202202", "202304"), "eval": ("202401", "202509"), "check": ("202510", "202608")}
LAST_LABEL_MONTH = "202609"  # 신고 기한(10.30) 이 지난 마지막 계약 월. 10.29~30 수집분
LABEL_END = {"tuning": "202312", "eval": LAST_LABEL_MONTH, "check": LAST_LABEL_MONTH}
CANDIDATES = {
    "s0": {"A": 0.30, "B": 0.30, "C": 0.25, "D": 0.15},
    "s1": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    "s2": {"A": 0.40, "B": 0.20, "C": 0.25, "D": 0.15},
    "s3": {"A": 0.20, "B": 0.40, "C": 0.25, "D": 0.15},
    "s4": {"A": 0.25, "B": 0.25, "C": 0.40, "D": 0.10},
}
TIE = 0.01  # 1위와 s0 의 6개월 IC 평균 차이가 이 안이면 s0
CAPITAL = {"11", "28", "41"}  # 서울·인천·경기


def months_of(period: str) -> list[str]:
    a, b = PERIODS[period]
    return [
        f"{p.year}{p.month:02d}" for p in pd.period_range(f"{a[:4]}-{a[4:]}", f"{b[:4]}-{b[4:]}", freq="M")
    ]


def period_of(t: str) -> str:
    return next(k for k, (a, b) in PERIODS.items() if a <= t <= b)


def weights_hash(w: dict) -> str:
    return hashlib.sha1(repr(sorted(w.items())).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- build


def load_trades(pred_dir) -> pd.DataFrame:
    """피처표의 모든 거래 + 모델 잔차 (채점 행 예측 + 지금 기준 해제된 거래 예측). 문자열은 arrow 로 (메모리)."""
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    tbl = con.sql(
        f"""
        SELECT f.ym, f.deal_date, f.cdeal_day, f.is_canceled, f.sgg_cd, f.dealing_gbn, f.complex_key, f.band, f.lp,
               ln(f.deal_amount) - ln(p.pred_lgbm) AS res
        FROM read_parquet('{(FEATURE_DIR / "ym=*.parquet").as_posix()}', union_by_name = true) f
        LEFT JOIN read_parquet('{(pred_dir / "D=*.parquet").as_posix()}') p USING (trade_key, ym)
        """
    ).to_arrow_table()  # duckdb 2: .arrow() 는 RecordBatchReader
    return tbl.to_pandas(types_mapper=lambda t: pd.ArrowDtype(t) if pa.types.is_string(t) else None)


def load_survey() -> pd.DataFrame:
    rone = pd.read_parquet(RONE_CLEAN / "rone_index.parquet")
    rmap = pd.read_parquet(REFERENCE_CLEAN / "region_map.parquet")
    s = rone_by_sgg(rone, rmap, "survey")
    rtype = pd.read_parquet(RONE_CLEAN / "rone_region.parquet").set_index("region_key")["region_type"]
    return s.assign(sido_level=s["region_key"].map(rtype).eq("시도"))


def check_month(sub: pd.DataFrame, el: pd.DataFrame, t: str) -> None:
    c = pd.Timestamp(asof.cutoff_date(t))
    if el["max_deal_date"].max() > c:
        raise asof.AsOfViolation(
            f"T={t}: 쓴 거래의 최대 계약일 {el['max_deal_date'].max()} > C(T) {c.date()}"
        )
    w = sub[(sub["ym"] >= region.shift(t, -4)) & (sub["ym"] <= region.shift(t, -2)) & (sub["deal_date"] <= c)]
    w = w[~(w["cdeal_day"].notna() & (w["cdeal_day"] <= c)) & (w["dealing_gbn"] == "중개거래")]
    missing = int(w["res"].isna().sum())
    if missing:
        raise SystemExit(
            f"T={t}: B 창의 중개거래 {missing:,}건에 예측 없음 (train_lgbm.py prep / canceled 먼저)"
        )


def build() -> None:
    lgbm_cfg = json.loads(LGBM_CONFIG.read_text(encoding="utf-8"))
    pred_dir = ROOT / "data" / "models" / "lgbm" / lgbm_cfg["params_hash"]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [t for p in PERIODS for t in months_of(p) if not (OUT / f"T={t}.parquet").exists()]
    if not todo:
        print("모든 기준월이 이미 있음")
        return
    t0 = time.perf_counter()
    trades = load_trades(pred_dir)
    survey = load_survey()
    sido_level = survey.drop_duplicates("sgg_cd").set_index("sgg_cd")["sido_level"]
    sv = survey[["sgg_cd", "ym", "value"]]
    final = trades.loc[~trades["is_canceled"], ["sgg_cd", "complex_key", "band", "ym", "lp"]]
    print(
        f"거래 {len(trades):,}행, 조사 지수 시군구 {sv['sgg_cd'].nunique()}개 읽음 ({time.perf_counter() - t0:.0f}초)"
    )
    for t in todo:
        sub = trades[(trades["ym"] >= region.shift(t, -16)) & (trades["ym"] <= t)]
        el = region.elements(sub, sv, t)
        check_month(sub, el, t)
        end = LABEL_END[period_of(t)]
        f = final[(final["ym"] >= t) & (final["ym"] <= region.shift(t, 14))]
        lab = region.labels(f, sv, t, 6, end).merge(
            region.labels(f, sv, t, 12, end).drop(columns="t"), on="sgg_cd", how="outer"
        )
        df = el.merge(lab.drop(columns="t"), on="sgg_cd", how="left")
        df["sido_level"] = df["sgg_cd"].map(sido_level).fillna(True).astype(bool)
        df["max_deal_date"] = df["max_deal_date"].astype("datetime64[us]")
        df.to_parquet(OUT / f"T={t}.parquet", index=False)
        print(
            f"T={t}: 시군구 {len(df)}, B 있음 {df['B'].notna().sum()}, 정답 6개월 {df['X6'].notna().sum()} · "
            f"12개월 {df['X12'].notna().sum()} ({time.perf_counter() - t0:.0f}초)"
        )


def load(period: str) -> pd.DataFrame:
    files = [OUT / f"T={t}.parquet" for t in months_of(period)]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"{period}: 기준월 파일 없음 {missing[:3]}… (build 먼저)")
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


# ---------------------------------------------------------------- tune


def summarize(sc: pd.DataFrame) -> dict:
    out = {}
    for h in (6, 12):
        x = f"X{h}"
        ic = bt.monthly_ic(sc, "score", x)
        out |= {f"ic{h}_mean": ic.mean(), f"ic{h}_pos": (ic > 0).mean(), f"ic{h}_months": len(ic),
                f"spread{h}": bt.spread(sc[sc[x].notna()], x)}  # fmt: skip
    share = sc["grade"].value_counts(normalize=True)
    out |= {f"share_{g}": float(share.get(g, 0)) for g in region.GRADES}
    out |= {f"rule_{r}": float(sc[r].mean()) for r in ("r1", "r2", "r3")}
    return out


def tune(force: bool) -> None:
    if CONFIG.exists() and not force:
        raise SystemExit(f"{CONFIG.relative_to(ROOT)} 가 이미 고정됨 (다시 하려면 --force)")
    el = load("tuning")
    rows = []
    for c, w in CANDIDATES.items():
        rows.append({"cand": c, **w, **summarize(region.score(el, w))})
    res = pd.DataFrame(rows).set_index("cand")
    res.to_csv(TUNING_CSV)
    best = res["ic6_mean"].idxmax()
    pick = "s0" if res.loc[best, "ic6_mean"] - res.loc["s0", "ic6_mean"] <= TIE else best
    w = CANDIDATES[pick]
    lgbm_hash = json.loads(LGBM_CONFIG.read_text(encoding="utf-8"))["params_hash"]
    CONFIG.write_text(
        json.dumps(
            {"cand": pick, "weights": w, "weights_hash": weights_hash(w), "lgbm_params_hash": lgbm_hash,
             "rule": f"튜닝 T {PERIODS['tuning'][0]}~{PERIODS['tuning'][1]} 6개월 IC 평균 최대 (s0 와 {TIE} 이내면 s0)",
             "fixed_at": f"{datetime.now():%Y-%m-%d %H:%M}",
             "tuning_ic6_mean": round(float(res.loc[pick, "ic6_mean"]), 4)},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    tracking.use_experiment("region_score_backtest")
    tags = tracking.common_tags("tuning")
    for c, r in res.iterrows():
        with mlflow.start_run(run_name=f"score_{c}"):
            mlflow.set_tags({**tags, "kind": "score_candidate", "cand": c, "weights_hash": weights_hash(CANDIDATES[c]),
                             "lgbm_params_hash": lgbm_hash, "chosen": str(c == pick)})  # fmt: skip
            mlflow.log_params(CANDIDATES[c])
            mlflow.log_metrics({k: float(v) for k, v in r.items() if k not in CANDIDATES[c] and pd.notna(v)})
    print(res.round(4).to_string())
    print(f"고정: {pick} → {CONFIG.relative_to(ROOT)}")


# ---------------------------------------------------------------- eval


def pct(v: float) -> str:
    return "-" if pd.isna(v) else f"{v * 100:+.2f}"


def grade_lines(sc: pd.DataFrame, h: int, grade: str = "grade") -> list[str]:
    t = bt.grade_table(sc, h, grade)
    head = "| 등급 | 시군구-월 | 초과 평균 (%) | 초과 중앙 (%) | 적중률 (%) | 같은 단지 수익률 (%) | 조사 지수 수익률 (%) |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    if h == 12:
        head, sep = head + " 12개월 최대 하락 평균 (%) | 하위 10% (%) |", sep + "---:|---:|"
    lines = [head, sep]
    for g, r in t.iterrows():
        hit = "-" if pd.isna(r["hit"]) else f"{r['hit'] * 100:.1f}"
        line = (
            f"| {g} | {int(r['n']):,} | {pct(r['x_mean'])} | {pct(r['x_median'])} | {hit} | "
            f"{pct(r['r_mean'])} | {pct(r['s_ret_mean'])} |"
        )
        if h == 12:
            line += f" {pct(r['mdd_mean'])} | {pct(r['mdd_p10'])} |"
        lines.append(line)
    return lines


def group_lines(sc: pd.DataFrame, key: pd.Series, title: str) -> list[str]:
    d = sc.assign(_k=key)[sc["X6"].notna()]
    lines = [f"### {title} (6개월)", "", "| 구분 | 시군구-월 | 추천 수 | 추천 초과 (%) | 고려 대상 아님 초과 (%) | 차이 (%p) |",
             "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for k, g in d.groupby("_k", sort=True):
        rec, nt = g.loc[g["grade"] == region.REC, "X6"], g.loc[g["grade"] == region.NOT, "X6"]
        diff = rec.mean() - nt.mean()
        lines.append(
            f"| {k} | {len(g):,} | {len(rec):,} | {pct(rec.mean())} | {pct(nt.mean())} | {pct(diff)} |"
        )
    return [*lines, ""]


def evaluate(force: bool) -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    tracking.use_experiment("region_score_backtest")
    prev = mlflow.search_runs(
        filter_string=f"tags.kind = 'score_eval' and tags.weights_hash = '{cfg['weights_hash']}'",
        output_format="pandas",
    )
    if len(prev) and not force:
        raise SystemExit(f"이 설정({cfg['weights_hash']})의 평가가 이미 기록됨: run {prev['run_id'].iloc[0]}")
    w = cfg["weights"]
    res = {p: region.score(load(p), w) for p in ("tuning", "eval", "check")}
    for sc in res.values():
        sc["grade_A"] = bt.matched_grades(sc, "P_A")
        sc["grade_B"] = bt.matched_grades(sc, "P_B")
    ev = res["eval"]
    metrics, boots, rand = {}, {}, {}
    for p, sc in res.items():
        for h in (6, 12):
            x = f"X{h}"
            if sc[x].notna().sum() == 0:
                continue
            for col in ("score", "P_A", "P_B"):
                ic = bt.monthly_ic(sc, col, x)
                metrics[f"{p}_ic{h}_{col}"] = ic.mean()
                metrics[f"{p}_ic{h}_{col}_pos"] = (ic > 0).mean()
            d = sc[sc[x].notna()]
            metrics[f"{p}_spread{h}"] = bt.spread(d, x)
            metrics[f"{p}_spread{h}_A"] = bt.spread(d, x, "grade_A")
            metrics[f"{p}_spread{h}_B"] = bt.spread(d, x, "grade_B")
    for h in (6, 12):
        boots[h] = bt.bootstrap(ev, h, "P_A")
        rand[h] = bt.random_spreads(ev, f"X{h}")
    b6 = boots[6]
    verdict = {
        "① IC 평균 > 0, 95% 하한 > 0": b6["ic_mean"][0] > 0 and b6["ic_mean"][1] > 0,
        "② 추천 − 고려 대상 아님 > 0, 95% 하한 > 0": b6["spread"][0] > 0 and b6["spread"][1] > 0,
        '③ IC 평균 > "A 만" IC 평균': metrics["eval_ic6_score"] > metrics["eval_ic6_P_A"],
    }
    ok = all(verdict.values())
    write_report(cfg, res, metrics, boots, rand, verdict, ok)

    with mlflow.start_run(run_name=f"score_eval_{cfg['cand']}"):
        mlflow.set_tags({**tracking.common_tags("eval"), "kind": "score_eval", "weights_hash": cfg["weights_hash"],
                         "lgbm_params_hash": cfg["lgbm_params_hash"], "verdict": "pass" if ok else "fail"})  # fmt: skip
        mlflow.log_params(w)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        for h, b in boots.items():
            for k, v in b.items():
                if k != "months":
                    mlflow.log_metrics({f"eval_boot{h}_{k}": v[0], f"eval_boot{h}_{k}_lo": v[1],
                                        f"eval_boot{h}_{k}_hi": v[2]})  # fmt: skip
        mlflow.log_artifact(str(REPORT))
    print("\n".join(REPORT.read_text(encoding="utf-8").splitlines()[:60]))


def write_report(cfg, res, metrics, boots, rand, verdict, ok) -> None:
    ev = res["eval"]
    tune = pd.read_csv(TUNING_CSV, index_col="cand")
    codes = bjd.load(BJD_RAW)
    top = codes[codes["is_active"] & codes["bjd_cd"].str.endswith("00000000")]
    sido = dict(zip(top["bjd_cd"].str[:2], top["full_nm"], strict=True))
    sido.setdefault("36", "세종특별자치시")
    f2 = lambda v: "-" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
    lines = [
        "# 매매 점수·등급 백테스트",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/score_backtest.py` (build → tune → eval)",
        "- 시군구 × 기준월 T (T 월 초 점수). 점수 자료는 C(T) = (T−1월 말일) − 30일 까지 공개된 거래, 조사 지수 T−1월까지",
        "- 요소: A 고점 대비 하락률 · C 3개월 추세 (R-ONE 조사 지수), B 모델 예측가 대비 저평가 (중개거래 잔차 중위값, "
        f"n÷(n+{region.SHRINK_K})), D 거래량 변화 → 달마다 전국 백분위 → 가중합",
        f"- 시세 모델: LightGBM `config/lgbm.json` (params_hash `{cfg['lgbm_params_hash']}`), 그 거래를 학습에 안 쓴 "
        "walk-forward 예측. 나중에 해제된 거래도 예측 포함 (그때는 정상 거래로 보였으므로)",
        "- 정답: 같은 단지·면적대의 W0 (계약월 T~T+2) → W1 (T+h~T+h+2) 중위 log ㎡당가 변화, 짝 수 가중, "
        f"짝 {region.PAIR_MIN}개 미만이면 없음. 초과 = 시군구 − 전국. 지금 기준 해제 아닌 거래로 계산 (평가용)",
        f"- 정답 데이터 끝: 계약월 {LAST_LABEL_MONTH} (2026-10-29~30 수집, 신고 기한 10.30 직전이라 9월 말 며칠분은 덜 들어왔을 수 있음)",
        "- 수익률은 log 차이 × 100 (%). 리포트에는 집계만",
        "",
        "## 1. 튜닝 (T 2022.2~2023.4, 정답은 2023.12 안에서 끝남)",
        "",
        f"선택 규칙 (튜닝 전에 정함): 6개월 IC 평균 최대, s0 와 {TIE} 이내면 s0 → **{cfg['cand']}** "
        f"(고정 {cfg['fixed_at']}, `config/score.json`)",
        "",
        "| 후보 | wA | wB | wC | wD | IC 6개월 | 양수 달 (%) | IC 12개월 | 추천−아님 6개월 (%p) | 추천 (%) | 고려 (%) | 아님 (%) | 절대 불가 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, r in tune.iterrows():
        mark = " **(선택)**" if c == cfg["cand"] else ""
        lines.append(
            f"| {c}{mark} | {r['A']:.2f} | {r['B']:.2f} | {r['C']:.2f} | {r['D']:.2f} | {f2(r['ic6_mean'])} | "
            f"{r['ic6_pos'] * 100:.0f} | {f2(r['ic12_mean'])} | {pct(r['spread6'])} | "
            + " | ".join(f"{r[f'share_{g}'] * 100:.1f}" for g in region.GRADES)
            + " |"
        )
    lines += ["", "튜닝 기간 등급별 (선택 설정)", "", *grade_lines(res["tuning"], 6), ""]

    lines += [
        "## 2. 평가 (T 2024.1~2025.9, 설정 고정 뒤 한 번)",
        "",
        "### 판정 (평가 6개월, 결과 보기 전에 정한 규칙)",
        "",
    ]
    lines += [f"- {k}: {'맞음' if v else '아님'}" for k, v in verdict.items()]
    lines += ["", f"**{'등급이 의미 있었음' if ok else '판정 규칙을 다 맞추지 못함'}**", ""]
    lines += ["### 순위 상관 IC (매월 Spearman, 평균 / 양수였던 달 %)", "",
              "| 기간 | 정답 | 점수 | \"A 만\" | \"B 만\" |", "|---|---|---:|---:|---:|"]  # fmt: skip
    for p, label in (("tuning", "튜닝"), ("eval", "**평가**"), ("check", "확인(참고)")):
        for h in (6, 12):
            if f"{p}_ic{h}_score" not in metrics:
                continue
            cells = [
                f"{f2(metrics[f'{p}_ic{h}_{c}'])} / {metrics[f'{p}_ic{h}_{c}_pos'] * 100:.0f}"
                for c in ("score", "P_A", "P_B")
            ]
            lines.append(f"| {label} | {h}개월 | " + " | ".join(cells) + " |")
    lines += ["", "### 우연인지 (평가, 원형 블록 부트스트랩 1,000회, 블록 = h 개월, 95% 구간)", ""]
    for h, b in boots.items():
        r = rand[h]
        lines += [
            f"- {h}개월 ({b['months']}개월): IC 평균 {f2(b['ic_mean'][0])} ({f2(b['ic_mean'][1])} ~ {f2(b['ic_mean'][2])}), "
            f"추천 − 고려 대상 아님 {pct(b['spread'][0])}%p ({pct(b['spread'][1])} ~ {pct(b['spread'][2])}), "
            f'IC − "A 만" {f2(b["ic_minus_P_A"][0])} ({f2(b["ic_minus_P_A"][1])} ~ {f2(b["ic_minus_P_A"][2])})',
            f"  - 무작위 등급 (같은 달 안에서 섞기 1,000회): 추천 − 아님 95% 범위 {pct(np.percentile(r, 2.5))} ~ "
            f"{pct(np.percentile(r, 97.5))}%p, 실제 값 이상인 비율 {(r >= b['spread'][0]).mean() * 100:.1f}%",
        ]
    for h in (6, 12):
        lines += ["", f"### 평가 등급별 {h}개월", "", *grade_lines(ev, h)]
    lines += ["", "### 기준선: 같은 등급 수로 자른 다른 점수 (평가 6개월)", "",
              "| 점수 | 추천 − 고려 대상 아님 (%p) |", "|---|---:|",
              f"| 4요소 점수 | {pct(metrics['eval_spread6'])} |", f"| \"A 만\" (고점 대비 하락률) | {pct(metrics['eval_spread6_A'])} |",
              f"| \"B 만\" (모델 저평가) | {pct(metrics['eval_spread6_B'])} |", ""]  # fmt: skip

    lines += ["## 3. 평가 나눠 보기", ""]
    lines += group_lines(ev, np.where(ev["sgg_cd"].str[:2].isin(CAPITAL), "수도권", "지방"), "수도권 / 지방")
    lines += group_lines(
        ev, np.where(ev["sido_level"], "시도 지수 (A·C 가 시도 값)", "시군구 지수"), "조사 지수 단위"
    )
    lines += group_lines(ev, ev["t"].str[:4], "연도")
    lines += group_lines(ev, ev["sgg_cd"].str[:2].map(sido), "시도")
    r = ev[["r1", "r2", "r3"]].mean() * 100
    lines += ["### 절대 불가 규칙 해당 비율 (평가, 시군구-월 %)", "",
              f"R1 데이터 부족 {r['r1']:.1f} · R2 거래 급감 + 하락 지속 {r['r2']:.1f} · R3 해제 급증 {r['r3']:.1f}", ""]  # fmt: skip

    ck = res["check"]
    lines += ["## 4. 확인 (T 2025.10~2026.8, 참고. 6개월 정답은 2026.1 까지)", "", *grade_lines(ck, 6), ""]
    lines += [
        "## 5. 한계",
        "",
        "- 튜닝 기간이 짧음 (6개월 정답 15개월, 12개월 9개월). 2023년 예측은 c05 를 고른 튜닝 폴드와 겹침",
        "- 시군구 79개는 조사 지수가 시도 값이라 같은 시도 안에서 A·C 가 같음",
        "- 정답은 같은 단지 비교 수익률로 거래비용·세금을 빼지 않음. 지역 단위 결과는 개별 물건과 다름",
        "- 30일을 넘긴 지연 신고는 일부 섞일 수 있음 (9-1)",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "tune", "eval"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "tune":
        tune(args.force)
    else:
        evaluate(args.force)


if __name__ == "__main__":
    main()
