"""CSV·API 겹침 검증 리포트: 같은 달을 두 경로로 받았을 때 같은 거래가 같은 키로 들어오는지 확인.

대상: CSV 정리본이 있고 API 정리본도 있는 달 (매일 이어받기의 점검(rotate)이 2~12개월 전을 API 로 다시 받음).
비교 단위: (시군구, 월). API 파일 안의 행 중 이번에 API 로 받은 것(source=api)만 쓴다
          (CSV 에서 넘어와 아직 확정 전인 행(missing_count=1)은 API 쪽에 없는 것으로 센다).

보는 것
- 건수: CSV / API / 양쪽 다(trade_key 같음) / CSV 에만 / API 에만
- trade_key 는 법정동·지번·단지명·면적·층·동·계약일·금액으로 만들므로, 키가 같으면 그 칸들은 같다
  → 키가 같은 거래끼리 나머지 칸(해제 여부·해제일·거래유형·매수/매도자·등기일)이 다른지 센다
- CSV 에만 / API 에만 있는 거래: 단지명·금액만 바뀐 같은 거래인지 (지번·면적·층·계약일 같음) 따로 센다
- 결과: reports/overlap_YYYYMM.md (요약) + reports/overlap_YYYYMM_sgg.csv (시군구별 건수, 개별 거래 없음)

실행: uv run python scripts/report_overlap.py [--months 202608]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from realestate.config import APT_TRADE_CLEAN, BJD_RAW, ROOT
from realestate.ingest import bjd

REPORTS = ROOT / "reports"
COMPARE_COLS = ["is_canceled", "cdeal_day", "dealing_gbn", "buyer_gbn", "seller_gbn", "rgst_date"]
LOOSE_KEY = ["bjd_cd", "bonbun", "bubun", "exclu_use_ar", "floor", "deal_date"]  # 단지명·금액 제외


def overlap_months() -> list[str]:
    return sorted(
        d.name[3:]
        for d in APT_TRADE_CLEAN.glob("ym=*")
        if (d / "source=csv.parquet").exists() and any(d.glob("source=api_sgg=*.parquet"))
    )


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def compare_month(ym: str, sgg_names: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    d = APT_TRADE_CLEAN / f"ym={ym}"
    csv_all = pd.read_parquet(d / "source=csv.parquet")
    rows, field_diff, loose_pairs = [], dict.fromkeys(COMPARE_COLS, 0), 0
    csv_download = csv_all["first_seen_at"].max()
    for path in sorted(d.glob("source=api_sgg=*.parquet")):
        sgg = path.stem.split("sgg=")[-1]
        api = pd.read_parquet(path)
        api = api[api["source"] == "api"]
        csv = csv_all[csv_all["sgg_cd"] == sgg]
        m = csv.merge(api, on="trade_key", suffixes=("_c", "_a"))
        for c in COMPARE_COLS:
            field_diff[c] += int((~_same(m[f"{c}_c"], m[f"{c}_a"])).sum())
        csv_only = csv[~csv["trade_key"].isin(api["trade_key"])]
        api_only = api[~api["trade_key"].isin(csv["trade_key"])]
        loose = csv_only[LOOSE_KEY].merge(api_only[LOOSE_KEY].drop_duplicates(), on=LOOSE_KEY)
        loose_pairs += len(loose)
        rows.append(
            {
                "sgg_cd": sgg,
                "sgg_nm": sgg_names.get(sgg, ""),
                "csv": len(csv),
                "api": len(api),
                "both": len(m),
                "csv_only": len(csv_only),
                "api_only": len(api_only),
                "csv_canceled": int(csv["is_canceled"].sum()),
                "api_canceled": int(api["is_canceled"].sum()),
                "canceled_after_csv": int((~m["is_canceled_c"] & m["is_canceled_a"]).sum()),
                "api_fetched_at": api["last_seen_at"].max() if len(api) else pd.NaT,
            }
        )
    by_sgg = pd.DataFrame(rows)
    csv_sggs = set(csv_all["sgg_cd"].dropna())
    meta = {
        "ym": ym,
        "csv_download": csv_download,
        "n_sgg_api": len(by_sgg),
        "csv_total": len(csv_all),
        "csv_not_in_api_sgg": int((~csv_all["sgg_cd"].isin(by_sgg["sgg_cd"])).sum()),
        "csv_sgg_without_api": sorted(csv_sggs - set(by_sgg["sgg_cd"])),
        "csv_unmatched_bjd": int(csv_all["bjd_cd"].isna().sum()),
        "field_diff": field_diff,
        "loose_pairs": loose_pairs,
    }
    return by_sgg, meta


def write_report(by_sgg: pd.DataFrame, meta: dict, out_dir: Path) -> Path:
    ym = meta["ym"]
    t = by_sgg[
        ["csv", "api", "both", "csv_only", "api_only", "csv_canceled", "api_canceled", "canceled_after_csv"]
    ].sum()
    match_rate = t["both"] / t["csv"] if t["csv"] else float("nan")
    exact = by_sgg[(by_sgg["csv_only"] == 0) & (by_sgg["api_only"] == 0)]
    diff = by_sgg[(by_sgg["csv_only"] > 0) | (by_sgg["api_only"] > 0)].sort_values(
        ["csv_only", "api_only"], ascending=False
    )
    fetched = by_sgg["api_fetched_at"].dropna()
    full = meta["csv_not_in_api_sgg"] == 0
    lines = [
        f"# CSV·API 겹침 검증 — {ym[:4]}.{ym[4:]}",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/report_overlap.py`",
        f"- CSV 다운로드: {meta['csv_download']:%Y-%m-%d %H:%M} · API 수집: "
        + (f"{fetched.min():%Y-%m-%d %H:%M} ~ {fetched.max():%Y-%m-%d %H:%M}" if len(fetched) else "-"),
        f"- 비교한 시군구: {meta['n_sgg_api']}개"
        + (
            ""
            if full
            else f" (API 아직 안 받은 시군구 {len(meta['csv_sgg_without_api'])}개, CSV {meta['csv_not_in_api_sgg']:,}건은 비교 제외)"
        ),
        "",
        "## 요약",
        "",
        "| 항목 | 건수 |",
        "|---|---:|",
        f"| CSV 거래 | {t['csv']:,} |",
        f"| API 거래 | {t['api']:,} |",
        f"| 양쪽 다 (trade_key 일치) | {t['both']:,} ({match_rate:.2%} of CSV) |",
        f"| CSV 에만 | {t['csv_only']:,} |",
        f"| API 에만 | {t['api_only']:,} |",
        f"| └ 단지명·금액만 다른 짝 (같은 지번·면적·층·계약일) | {meta['loose_pairs']:,} |",
        f"| 해제: CSV / API | {t['csv_canceled']:,} / {t['api_canceled']:,} |",
        f"| CSV 받은 뒤 해제된 거래 | {t['canceled_after_csv']:,} |",
        f"| 건수까지 완전히 같은 시군구 | {len(exact)} / {len(by_sgg)} |",
        "",
        "## 키가 같은 거래끼리 다른 칸",
        "",
        "| 칸 | 다른 건수 |",
        "|---|---:|",
        *[f"| `{c}` | {n:,} |" for c, n in meta["field_diff"].items()],
        "",
        "## 차이가 있는 시군구 (상위 20)",
        "",
    ]
    if len(diff):
        lines += [
            "| 시군구 | CSV | API | CSV에만 | API에만 | 받은 뒤 해제 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in diff.head(20).iterrows():
            lines.append(
                f"| {r['sgg_nm']} ({r['sgg_cd']}) | {r['csv']:,} | {r['api']:,} | {r['csv_only']:,} | "
                f"{r['api_only']:,} | {r['canceled_after_csv']:,} |"
            )
    else:
        lines.append("없음 — 모든 시군구에서 CSV 와 API 의 거래 목록이 같음")
    lines += [
        "",
        "## 판단",
        "",
        *judge(t, meta, match_rate),
        "",
        "키 규칙: PLAN.md 6-1-2 (2026-10-30 이 리포트로 동(apt_dong)을 키에서 뺌. 빼기 전 2026.8 일치율 98.72%)",
        "",
        f"시군구별 건수: `reports/overlap_{ym}_sgg.csv` (개별 거래 없음)",
        "",
    ]
    out = out_dir / f"overlap_{ym}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    by_sgg.drop(columns="api_fetched_at").to_csv(
        out_dir / f"overlap_{ym}_sgg.csv", index=False, encoding="utf-8"
    )
    return out


def judge(t: pd.Series, meta: dict, match_rate: float) -> list[str]:
    out = []
    if match_rate >= 0.999:
        out.append(
            f"- CSV 거래의 {match_rate:.2%} 가 API 에서 같은 trade_key 로 나옴 → **두 자료를 같은 표에 합쳐도 됨** "
            "(달 단위로 한쪽만 읽는 규칙 `realestate.ingest.trades` 유지)"
        )
    else:
        out.append(f"- 일치율 {match_rate:.2%} — 99.9% 미만. 차이가 큰 시군구부터 원인 확인 필요")
    if t["api_only"]:
        out.append(
            f"- API 에만 {t['api_only']:,}건: CSV 다운로드 뒤 늦게 신고됐거나, 내용이 고쳐져 키가 바뀐 거래 "
            f"(그중 단지명·금액만 다른 짝 {meta['loose_pairs']:,}건)"
        )
    if t["csv_only"]:
        out.append(
            f"- CSV 에만 {t['csv_only']:,}건: API 에서 빠졌거나 키가 바뀐 거래. 정리본에는 missing_count=1 로 남아 있고 "
            "다음 점검(14일 뒤)에도 없으면 removed 로 확정"
        )
    if t["canceled_after_csv"]:
        out.append(f"- CSV 받은 뒤 해제로 바뀐 거래 {t['canceled_after_csv']:,}건 → API 로 다시 받는 이유")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", help="기본: CSV·API 가 둘 다 있는 모든 달")
    args = parser.parse_args()
    months = args.months or overlap_months()
    if not months:
        print("CSV·API 가 겹치는 달이 없습니다. 매일 이어받기(점검)가 2~12개월 전을 받은 뒤 다시 실행하세요.")
        return
    codes = bjd.load(BJD_RAW)
    sgg = codes[codes["is_active"] & codes["bjd_cd"].str.endswith("00000")]
    sgg_names = dict(zip(sgg["sgg_cd"], sgg["full_nm"], strict=True))
    REPORTS.mkdir(exist_ok=True)
    for ym in months:
        by_sgg, meta = compare_month(ym, sgg_names)
        out = write_report(by_sgg, meta, REPORTS)
        t = by_sgg[["csv", "both", "csv_only", "api_only"]].sum()
        print(
            f"{ym}: 시군구 {len(by_sgg)}개, CSV {t['csv']:,} / 일치 {t['both']:,} / CSV에만 {t['csv_only']:,} / "
            f"API에만 {t['api_only']:,} → {out.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
