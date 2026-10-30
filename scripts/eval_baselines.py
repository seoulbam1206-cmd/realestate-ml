"""기준선 B0(지역 중위가)·B1(단지 중위가) 채점 → reports/baselines.md (PLAN.md MVP 3-5, 3-6)

- 피처표(scripts/build_features.py)의 b0_ppm2·b1_ppm2 × 전용면적 = 예측 거래가 (만원)
- 채점 행: 튜닝 T 2023 / 평가 T 2024.1~2025.9 / 확인 T 2025.10~2026.8, 지금 기준 해제되지 않은 거래
- 리포트에는 집계만 (개별 거래 없음)

실행: uv run python scripts/eval_baselines.py
"""

from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd

from realestate.config import BJD_RAW, ROOT
from realestate.features import asof
from realestate.features.trade_features import FEATURES
from realestate.ingest import bjd
from realestate.models.metrics import block_bootstrap_diff, metrics_by, price_metrics
from realestate.models.periods import FEATURE_DIR, PERIODS, load_scoring_rows

REPORTS = ROOT / "reports"
PREDS = {"B0 지역 중위가": "pred_b0", "B1 단지 중위가": "pred_b1"}
PERIOD_NAMES = {"tuning": "튜닝", "eval": "**평가**", "check": "확인(참고)"}
BAND_NAMES = {1: "≤40㎡", 2: "40~60㎡", 3: "60~85㎡", 4: "85~102㎡", 5: "102~135㎡", 6: ">135㎡"}


def _fmt_row(label: str, m: dict) -> str:
    return f"| {label} | {m['n']:,} | {m['mae']:,.0f} | {m['mape']:.2f} | {m['mdape']:.2f} | {m['within10']:.1f} |"


HEAD = [
    "| 구분 | 거래 수 | MAE (만원) | MAPE (%) | MdAPE (%) | ±10% 이내 (%) |",
    "|---|---:|---:|---:|---:|---:|",
]


def breakdown(df: pd.DataFrame, col: str, title: str, names: dict | None = None) -> list[str]:
    t = metrics_by(df, col, PREDS)
    lines = [
        f"### {title}",
        "",
        "| 구분 | 거래 수 | B0 MAE | B1 MAE | B0 MAPE | B1 MAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g, part in t.groupby("group", sort=True):
        b0, b1 = (part.set_index("model").loc[k] for k in PREDS)
        label = (names or {}).get(g, g)
        lines.append(
            f"| {label} | {int(b0['n']):,} | {b0['mae']:,.0f} | {b1['mae']:,.0f} | {b0['mape']:.2f} | {b1['mape']:.2f} |"
        )
    return [*lines, ""]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    meta = pd.read_parquet(FEATURE_DIR / "_meta.parquet")
    asof.audit(meta, FEATURES)  # 시점 규칙을 어긴 피처표면 여기서 멈춤
    cols = ["trade_key", "ym", "deal_amount", "area", "band", "sido_cd", "dealing_gbn", "is_canceled",
            "b0_ppm2", "b0_level", "b1_ppm2", "b1_level"]  # fmt: skip
    df = load_scoring_rows(cols)
    df["pred_b0"] = df["b0_ppm2"] * df["area"]
    df["pred_b1"] = df["b1_ppm2"] * df["area"]
    codes = bjd.load(BJD_RAW)
    top = codes[codes["is_active"] & codes["bjd_cd"].str.endswith("00000000")]
    sido = dict(zip(top["bjd_cd"].str[:2], top["full_nm"], strict=True))
    sido.setdefault("36", "세종특별자치시")  # 세종은 시도 행 없이 시군구(36110)만 있음

    lines = [
        "# 기준선 B0·B1 성적",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/eval_baselines.py` · 피처표 `scripts/build_features.py`",
        "- B0 지역 중위가: 법정동 M−4~M−2월 해제 제외 ㎡당 중위가 × 전용면적 (5건 미만이면 시군구 → 시도 → 전국)",
        "- B1 단지 중위가: 단지×면적대 최근 12개월 ㎡당 중위가 × 그 뒤 시군구 지수 변화 × 전용면적 (단지 이력 없으면 B0)",
        "- 시점 규칙: 계약 월 M 의 예측은 C(M) = (M−1월 말일) − 30일 까지 공개된 거래·해제만 사용 (PLAN.md MVP 3-3, 자동 검사 통과)",
        "- 채점 행: 지금 기준으로 해제되지 않은 거래. 기간 합산은 거래 단위",
        "",
        "## 기간별 성적",
        "",
        *HEAD,
    ]
    for p, (a, b) in PERIODS.items():
        part = df[df["period"] == p]
        for name, col in PREDS.items():
            lines.append(
                _fmt_row(f"{PERIOD_NAMES[p]} {a}~{b} · {name}", price_metrics(part["deal_amount"], part[col]))
            )

    ev = df[df["period"] == "eval"].copy()
    boot = block_bootstrap_diff(ev, "pred_b0", "pred_b1")
    b0 = price_metrics(ev["deal_amount"], ev["pred_b0"])
    lines += [
        "",
        "## 평가 기간: B1 − B0 차이 (월 단위 블록 부트스트랩 1,000회, 95% 구간)",
        "",
        f"- MAE: {boot['mae'][0]:+,.0f}만원 ({boot['mae'][1]:+,.0f} ~ {boot['mae'][2]:+,.0f})",
        f"- MAPE: {boot['mape'][0]:+.2f}%p ({boot['mape'][1]:+.2f} ~ {boot['mape'][2]:+.2f})",
        "",
        "## LightGBM 완료 기준 (평가 기간에 B0 보다 둘 다 낮아야 함)",
        "",
        f"- MAE < **{b0['mae']:,.0f}만원**, MAPE < **{b0['mape']:.2f}%**",
        "",
        "## 평가 기간 나눠 보기",
        "",
    ]
    ev["price_q"] = pd.qcut(ev["deal_amount"], 5, labels=["1 (싼 20%)", "2", "3", "4", "5 (비싼 20%)"])
    lines += breakdown(ev, "sido_cd", "시도", sido)
    lines += breakdown(ev, "price_q", "실제 거래가 5분위")
    lines += breakdown(ev, "band", "면적대", BAND_NAMES)
    lines += breakdown(ev.assign(dealing_gbn=ev["dealing_gbn"].fillna("(없음)")), "dealing_gbn", "거래유형")
    lines += breakdown(ev, "b1_level", "B1 이 단지 이력을 썼는지 (complex = 썼음, b0 = 이력 없어 B0 사용)")
    lines += breakdown(ev, "b0_level", "B0 가 어느 단위로 계산됐는지")

    monthly = metrics_by(df[df["period"].isin(["eval", "check"])], "ym", PREDS)
    lines += [
        "## 월별 (평가 + 확인)",
        "",
        "| 월 | 거래 수 | B0 MAE | B1 MAE | B0 MAPE | B1 MAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ym, part in monthly.groupby("group"):
        m0, m1 = (part.set_index("model").loc[k] for k in PREDS)
        tag = " (확인)" if ym >= PERIODS["check"][0] else ""
        lines.append(
            f"| {ym}{tag} | {int(m0['n']):,} | {m0['mae']:,.0f} | {m1['mae']:,.0f} | {m0['mape']:.2f} | {m1['mape']:.2f} |"
        )
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "baselines.md").write_text("\n".join([*lines, ""]), encoding="utf-8")

    summary = metrics_by(df, "period", PREDS)
    print(summary.round(2).to_string(index=False))
    print(f"B1 − B0 (평가): MAE {boot['mae']}, MAPE {boot['mape']} → reports/baselines.md")


if __name__ == "__main__":
    main()
