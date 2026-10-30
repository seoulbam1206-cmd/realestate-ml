"""시점 안전 피처표 생성 → data/features/trade_features/ym=YYYYMM.parquet (+ _meta.parquet)

- 정리본은 `realestate.ingest.trades` 규칙(달 단위 CSV 또는 API)으로 읽음
- 계약 월마다 기준일 C(M) = (M−1월 말일) − 30일 까지 공개된 정보만 사용 (PLAN.md MVP 3-3)
- 끝나면 asof.audit 으로 기록을 검사, 어기면 실패 (종료 코드 1)
- R-ONE 지역 → 시군구 연결은 data/clean/reference/region_map.parquet (scripts/report_rone_vs_trade.py 가 만듦)

실행: uv run python scripts/build_features.py [--from 202001] [--to 202610]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

import pandas as pd

from realestate.config import BJD_RAW, DATA_FEATURES, REFERENCE_CLEAN, RONE_CLEAN
from realestate.features import asof
from realestate.features.trade_features import FEATURES, iter_months, rone_by_sgg
from realestate.ingest import bjd, trades

OUT = DATA_FEATURES / "trade_features"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="ym_from", default="202001")
    parser.add_argument("--to", dest="ym_to", default=f"{date.today():%Y%m}")
    args = parser.parse_args()

    lawds = bjd.lawd_codes(bjd.load(BJD_RAW))
    src = trades.month_sources(lawds)
    files = [p for f in src["files"] for p in f]
    months = [ym for ym in src["ym"] if args.ym_from <= ym <= args.ym_to]
    rone = pd.read_parquet(RONE_CLEAN / "rone_index.parquet")
    rmap = pd.read_parquet(REFERENCE_CLEAN / "region_map.parquet")
    survey, real = rone_by_sgg(rone, rmap, "survey"), rone_by_sgg(rone, rmap, "real")
    print(
        f"정리본 {len(src)}개월 → 피처 {months[0]}~{months[-1]} ({len(months)}개월). "
        f"R-ONE 연결 시군구: 조사 {survey['sgg_cd'].nunique()}, 실거래가격지수 {real['sgg_cd'].nunique()}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    metas, t0 = [], time.perf_counter()
    for df, meta in iter_months(files, months, survey, real):
        df.to_parquet(OUT / f"ym={meta['ym']}.parquet", index=False, compression="zstd")
        meta["seconds"] = round(time.perf_counter() - t0, 1)
        metas.append(meta)
        if meta["ym"].endswith("12") or meta["ym"] == months[-1]:
            print(
                f"  {meta['ym']}: {meta['rows']:,}행, 이력 {meta['hist_rows']:,}행, 누적 {meta['seconds']}초"
            )
    meta = pd.DataFrame(metas)
    meta.to_parquet(OUT / "_meta.parquet", index=False)
    try:
        asof.audit(meta, FEATURES)
    except asof.AsOfViolation as e:
        print(f"시점 규칙 위반: {e}")
        return 1
    print(f"시점 검사 통과. {meta['rows'].sum():,}행, {time.perf_counter() - t0:.0f}초 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
