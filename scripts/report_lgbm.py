"""LightGBM vs 기준선 B0·B1 → reports/lgbm.md (PLAN.md MVP 3-6). scripts/train_lgbm.py report 가 부른다.

- 모든 모델을 같은 채점 행(지금 기준 해제되지 않은 거래)으로 채점, 기간 합산은 거래 단위
- 리포트에는 집계만 (개별 거래 없음)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from train_lgbm import (
    BACKTEST_PERIODS,
    CANDIDATES,
    OUT,
    TUNING_CSV,
    TUNING_FOLDS,
    backtest_months,
    load_backtest,
    scoring_rows,
    tuning_summary,
)

from realestate.config import ROOT
from realestate.models import lgbm
from realestate.models.metrics import block_bootstrap_diff, price_metrics
from realestate.models.periods import PERIODS

PREDS = {"B0 지역 중위가": "pred_b0", "B1 단지 중위가": "pred_b1", "LightGBM": "pred_lgbm"}
HEAD = [
    "| 구분 | 거래 수 | MAE (만원) | MAPE (%) | MdAPE (%) | ±10% 이내 (%) |",
    "|---|---:|---:|---:|---:|---:|",
]


def _row(label: str, m: dict) -> str:
    return f"| {label} | {m['n']:,} | {m['mae']:,.0f} | {m['mape']:.2f} | {m['mdape']:.2f} | {m['within10']:.1f} |"


def bias(actual: pd.Series, pred: pd.Series) -> float:
    """평균 부호 오차율 (%): + 면 비싸게, − 면 싸게 예측."""
    a = np.asarray(actual, float)
    return float(((np.asarray(pred, float) - a) / a * 100).mean())


def breakdown(df: pd.DataFrame, col: str, title: str, note: str = "") -> list[str]:
    names = list(PREDS)
    lines = [f"### {title}", ""] + ([note, ""] if note else [])
    lines += [
        "| 구분 | 거래 수 | " + " | ".join(f"{n.split()[0]} MAE" for n in names) + " | "
        + " | ".join(f"{n.split()[0]} MAPE" for n in names) + " | "
        + " | ".join(f"{n.split()[0]} 편향" for n in names) + " |",
        "|---|---:|" + "---:|" * (3 * len(names)),
    ]  # fmt: skip
    for g, part in df.groupby(col, observed=True, sort=True):
        ms = [price_metrics(part["deal_amount"], part[c]) for c in PREDS.values()]
        bs = [bias(part["deal_amount"], part[c]) for c in PREDS.values()]
        lines.append(
            f"| {g} | {len(part):,} | " + " | ".join(f"{m['mae']:,.0f}" for m in ms) + " | "
            + " | ".join(f"{m['mape']:.2f}" for m in ms) + " | " + " | ".join(f"{b:+.2f}" for b in bs) + " |"
        )  # fmt: skip
    return [*lines, ""]


def write_report(cfg: dict, periods: tuple[str, ...] = BACKTEST_PERIODS) -> None:
    tune = pd.read_csv(TUNING_CSV, dtype={"t": str})
    summ = tuning_summary(tune)
    df, runs = load_backtest(cfg, periods)
    ev = df[df["period"] == "eval"].copy()

    lines = [
        "# LightGBM 시세 모델 vs 기준선 B0·B1",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/train_lgbm.py` (tune → backtest → report)",
        "- 타깃 log ㎡당가, 예측 거래가 = exp(예측) × 전용면적. 피처 29개 (`trade_features.FEATURES`, 시점 검사 통과)",
        "- 분기 말 기준일 D 에 학습한 모델로 다음 분기 3개월을 예측. 학습 행: 2021.1 ~, 계약일 ≤ D − 30일, 그때까지 공개된 해제 제외",
        "- 트리 수: 마지막 완결 2개월로 조기 종료 → 같은 트리 수로 학습 행 전부에 재학습",
        "- 채점 행은 `reports/baselines.md` 와 같음 (지금 기준 해제되지 않은 거래). 기간 합산은 거래 단위",
        "",
        "## 1. 튜닝 (T 2023.3 / 2023.7 / 2023.11, 2023년까지 데이터만)",
        "",
        f"공통: 학습률 {lgbm.BASE_PARAMS['learning_rate']}, feature_fraction {lgbm.BASE_PARAMS['feature_fraction']}, "
        f"L2 {lgbm.BASE_PARAMS['lambda_l2']}, 255구간, 확장 창(2021.1~). 학습 창은 튜닝에서 가릴 수 없어 확장 창 고정 "
        "(학습이 2021.1 부터라 폴드 학습 기간이 23~33개월 → 최근 36개월 창과 같음). 선택 규칙: **3폴드 평균 MAPE 최소** (튜닝 전에 정함)",
        "",
        "| 후보 | 손실 | 잎 수 | 최소 잎 데이터 | 평균 MAE | 평균 MAPE | 평균 MdAPE | 평균 트리 수 | 평균 학습 초 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, s in summ.iterrows():
        p = CANDIDATES[c]
        mark = " **(선택)**" if c == cfg["cand"] else ""
        lines.append(
            f"| {c}{mark} | {p['objective']} | {p['num_leaves']} | {p['min_data_in_leaf']} | {s['mae']:,.0f} | "
            f"{s['mape']:.2f} | {s['mdape']:.2f} | {s['best_iter']:,.0f} | {s['sec']:,.0f} |"
        )
    lines += ["", "### 튜닝 폴드에서 기준선과 비교 (같은 채점 행)", "", *HEAD]
    chosen = tune[tune["cand"] == cfg["cand"]].set_index("t")
    for t in TUNING_FOLDS:
        part = scoring_rows(t, t)
        lines.append(_row(f"T {t} · B0", price_metrics(part["deal_amount"], part["b0_ppm2"] * part["area"])))
        lines.append(_row(f"T {t} · B1", price_metrics(part["deal_amount"], part["b1_ppm2"] * part["area"])))
        lines.append(_row(f"T {t} · LightGBM", chosen.loc[t].to_dict() | {"n": int(chosen.loc[t, "n"])}))

    lines += [
        "",
        "## 2. 평가 (설정 고정 뒤 1회 실행)",
        "",
        f"고정 설정 `config/lgbm.json` ({cfg['cand']}, params_hash `{cfg['params_hash']}`, 고정 {cfg['fixed_at']})",
        *(
            []
            if "check" in periods
            else [
                "",
                "확인 기간(T 2025.10~2026.8, 기준일 4개)은 아직 실행 안 함 → 이 리포트는 평가 기준일 7개만",
            ]
        ),
        "",
        *HEAD,
    ]
    labels = {"eval": "**평가**", "check": "확인(참고)"}
    for period in periods:
        label = labels[period]
        a, b = PERIODS[period]
        part = df[df["period"] == period]
        for name, col in PREDS.items():
            lines.append(_row(f"{label} {a}~{b} · {name}", price_metrics(part["deal_amount"], part[col])))

    lg, b0, b1 = (price_metrics(ev["deal_amount"], ev[c]) for c in ("pred_lgbm", "pred_b0", "pred_b1"))
    passed = lg["mae"] < b0["mae"] and lg["mape"] < b0["mape"]
    lines += ["", "### 평가 기간 차이 (LightGBM − 기준선, 월 단위 블록 부트스트랩 1,000회, 95% 구간)", ""]
    for base in ("b0", "b1"):
        boot = block_bootstrap_diff(ev, f"pred_{base}", "pred_lgbm")
        lines.append(
            f"- vs {base.upper()}: MAE {boot['mae'][0]:+,.0f}만원 ({boot['mae'][1]:+,.0f} ~ {boot['mae'][2]:+,.0f}), "
            f"MAPE {boot['mape'][0]:+.2f}%p ({boot['mape'][1]:+.2f} ~ {boot['mape'][2]:+.2f})"
        )
    lines += [
        "",
        f"**완료 기준 (B0 보다 MAE·MAPE 둘 다 낮음): {'통과' if passed else '미달'}** — "
        f"MAE {lg['mae']:,.0f} vs {b0['mae']:,.0f}, MAPE {lg['mape']:.2f} vs {b0['mape']:.2f}. "
        f"B1 대비: MAE {lg['mae'] - b1['mae']:+,.0f}만원, MAPE {lg['mape'] - b1['mape']:+.2f}%p",
        "",
        "## 3. 평가 기간 나눠 보기",
        "",
        "편향 = 평균 (예측 − 실제) ÷ 실제 × 100. + 면 비싸게, − 면 싸게 예측",
        "",
    ]
    ev["price_q"] = pd.qcut(ev["deal_amount"], 5, labels=["1 (싼 20%)", "2", "3", "4", "5 (비싼 20%)"])
    lines += breakdown(ev.assign(dealing_gbn=ev["dealing_gbn"].fillna("(없음)")), "dealing_gbn", "거래유형")
    lines += breakdown(
        ev, "price_q", "실제 거래가 5분위",
        "실제 가격으로 나눈 구간이라 싼 구간은 예측이 비싸게, 비싼 구간은 싸게 나오는 쪽으로 기움 (평균 회귀). 모델 간 비교로 볼 것",
    )  # fmt: skip
    q1 = ev[ev["price_q"] == "1 (싼 20%)"]
    lines += breakdown(
        q1.assign(dealing_gbn=q1["dealing_gbn"].fillna("(없음)")),
        "dealing_gbn",
        "가장 싼 20% 안에서 거래유형별",
    )
    lines += breakdown(ev, "b1_level", "단지 이력 유무 (complex = 단지 12개월 이력 있음, b0 = 없음)")
    lines += breakdown(df, "ym", "월별 (평가 + 확인)" if "check" in periods else "월별 (평가)")

    lines += [
        "## 4. 실측 시간·메모리 (기준일 하나 = 프로세스 하나, 4스레드)",
        "",
        "| 기준일 D | 예측 월 | 학습 행 | 트리 수 | 학습 초 | 전체 초 | 최대 메모리 (MB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in runs.itertuples():
        lines.append(
            f"| {r.anchor} | {r.months[0]}~{r.months[-1]} | {r.n_train:,} | {r.best_iter:,} | {r.sec_total:,.0f} | "
            f"{r.sec_wall:,.0f} | {r.peak_mb:,.0f} |"
        )
    lgbm_out = OUT / cfg["params_hash"]
    imp = pd.concat(
        pd.read_csv(lgbm_out / f"D={d:%Y%m%d}_importance.csv").assign(
            share=lambda x: x["gain"] / x["gain"].sum()
        )
        for d in backtest_months(periods)
    )
    top = imp.groupby("feature")["share"].mean().sort_values(ascending=False).head(12)
    lines += [
        "",
        f"## 5. 피처 중요도 (gain 비중, 기준일 {len(runs)}개 평균, 상위 12)",
        "",
        "| 피처 | 비중 (%) |",
        "|---|---:|",
        *[f"| {f} | {v * 100:.1f} |" for f, v in top.items()],
        "",
        f"기준일별 예측·중요도 파일: `{lgbm_out.relative_to(ROOT).as_posix()}/` (git 제외). MLflow 실험 `price_tuning`, `price_backtest`",
        "",
    ]
    (ROOT / "reports" / "lgbm.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[: lines.index("## 3. 평가 기간 나눠 보기")]))
