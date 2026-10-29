"""국토부 실거래 API 시험 호출 → 응답을 CSV 정리본과 비교. 수집기(재시도·하루 상한)를 그대로 쓴다.

기본: 1페이지만 (호출 1회). --all-pages 면 totalCount 만큼 페이지를 끝까지 받는다.
정리본(clean)에는 쓰지 않는다. 원본은 data/raw/molit/apt_trade/ 에 저장, 결과는 ingest_log(job_type=probe).

실행:
  uv run python scripts/probe_molit_api.py --lawd 11110 --ym 202512
  uv run python scripts/probe_molit_api.py --lawd 41463 --ym 202606 --all-pages
  uv run python scripts/probe_molit_api.py --lawd 28110 --ym 202512 --compare-sgg 28125 28155   # 옛 코드
  uv run python scripts/probe_molit_api.py --from-raw <저장된 .xml.gz> ...                         # 호출 없이 분석
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from realestate.config import APT_TRADE_CLEAN, REFERENCE_CLEAN, sources
from realestate.ingest import log as ingest_log
from realestate.ingest import molit_api, molit_csv
from realestate.ingest.molit_collect import SOURCE, Collector, FatalApiError, QuotaStop, RetryExhausted


def _loose_key(df: pd.DataFrame) -> pd.Series:
    """시군구 코드 없이 비교하는 키 (옛 코드 ↔ 새 코드 비교용): 읍면동 끝 토막 + 지번 + 단지 + 면적·층·날짜·금액."""
    umd = df["umd_nm"].str.split(" ").str[-1]
    parts = [umd, df["bonbun"], df["bubun"], molit_csv.normalize_apt_name(df["apt_nm"]),
             df["exclu_use_ar"].round(2), df["floor"], df["deal_date"].dt.strftime("%Y%m%d"), df["deal_amount"]]  # fmt: skip
    return pd.concat(parts, axis=1).astype(str).agg("|".join, axis=1)


def report(
    items: list[dict[str, str]], total: int | None, pages: int, ym: str, compare_sgg: list[str]
) -> None:
    print(f"totalCount={total}, 받은 item={len(items)}, 페이지 {pages}")
    if not items:
        return
    raw = pd.DataFrame(items).fillna("")
    print(f"  sggCd 값: {dict(Counter(raw['sggCd']))}")
    if pages > 1:
        print(f"  페이지 사이 완전히 같은 item: {raw.duplicated().sum()}건")
    canceled = raw[(raw["cdealType"].str.strip() != "") | (raw["cdealDay"].str.strip() != "")]
    print(f"  해제 표시 {len(canceled)}건, cdealType 값 {dict(Counter(canceled['cdealType']))}, "
          f"cdealDay 예 {canceled['cdealDay'].head(3).tolist()}")  # fmt: skip

    csv_path = APT_TRADE_CLEAN / f"ym={ym}" / "source=csv.parquet"
    if not csv_path.exists():
        print(f"  (CSV 정리본 {ym} 없음 → 비교 생략)")
        return
    codes = pd.read_parquet(REFERENCE_CLEAN / "bjd_code.parquet")
    api = molit_api.to_clean(items, datetime.now(), dict(zip(codes["bjd_cd"], codes["full_nm"], strict=True)))
    csv = pd.read_parquet(csv_path)
    csv = csv[csv["sgg_cd"].isin(compare_sgg)]
    print(f"\nCSV(시군구 {','.join(compare_sgg)}, {ym}) {len(csv)}건 vs API {len(api)}건")
    strict = api.merge(csv, on="trade_key", suffixes=("_api", "_csv"))
    print(
        f"  trade_key 일치 {len(strict)}건 (API에만 {len(api) - len(strict)}, CSV에만 {len(csv) - len(strict)})"
    )
    a, c = Counter(_loose_key(api)), Counter(_loose_key(csv))
    print(f"  시군구 코드 빼고 비교: 일치 {sum((a & c).values())}건")
    if len(strict):
        same_cancel = (strict["is_canceled_api"] == strict["is_canceled_csv"]).sum()
        both = strict[strict["is_canceled_api"] & strict["is_canceled_csv"]]
        same_day = (both["cdeal_day_api"] == both["cdeal_day_csv"]).sum()
        print(
            f"  해제 여부 같음 {same_cancel}/{len(strict)}, 둘 다 해제인 {len(both)}건 중 해제일 같음 {same_day}건"
        )
        print(f"  법정동 코드 같음 {(strict['bjd_cd_api'] == strict['bjd_cd_csv']).sum()}/{len(strict)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = sources()["molit_apt_trade"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--lawd", default="11110")
    parser.add_argument("--ym", default="202512")
    parser.add_argument("--all-pages", action="store_true")
    parser.add_argument("--compare-sgg", nargs="+", help="비교할 CSV 시군구 코드 (기본: --lawd)")
    parser.add_argument("--cap", type=int, default=cfg["backfill_daily_cap"])
    parser.add_argument("--from-raw", type=Path, nargs="+", help="호출하지 않고 저장된 응답(.xml.gz)만 분석")
    args = parser.parse_args()
    compare = args.compare_sgg or [args.lawd]

    if args.from_raw:
        items, total = [], None
        for f in args.from_raw:
            _, _, total, its = molit_api.parse_xml(gzip.decompress(f.read_bytes()))
            items += its
        report(items, total, len(args.from_raw), args.ym, compare)
        return

    common = dict(
        run_id=ingest_log.new_run_id(), source=SOURCE, job_type="probe",
        target_key=f"sgg={args.lawd},ym={args.ym}", started_at=datetime.now(),
    )  # fmt: skip
    with ingest_log.connect() as con:
        col = Collector(con, cap=args.cap, max_retries=cfg["max_retries"])
        try:
            res = col.collect_month(args.lawd, args.ym, max_pages=None if args.all_pages else 1)
        except (QuotaStop, FatalApiError, RetryExhausted) as e:
            p = e.partial
            status = "quota_stop" if isinstance(e, QuotaStop) else "fail"
            ingest_log.write(con, status=status, error_msg=str(e)[:500], api_calls=p.api_calls,
                             retry_count=p.retries, result_code=p.result_code, **common)  # fmt: skip
            print(f"멈춤/실패: {e}")
            return
        ingest_log.write(
            con, status="success", http_status=res.http_status, result_code=res.result_code,
            result_msg=res.result_msg, pages=res.pages, api_calls=res.api_calls, retry_count=res.retries,
            rows_fetched=len(res.items),
            rows_canceled=sum(1 for i in res.items if i.get("cdealType", "").strip() or i.get("cdealDay", "").strip()),
            source_file=",".join(f.name for f in res.raw_files), **common,
        )  # fmt: skip
        print(f"resultCode={res.result_code} {res.result_msg}, 이번 호출 {res.api_calls}회 (재시도 {res.retries}), "
              f"오늘 누계 {col.used_today()}/{col.cap}")  # fmt: skip
        for f in res.raw_files:
            print(f"원본 저장: {f.relative_to(f.parents[3])}")
    report(res.items, res.total_count, res.pages, args.ym, compare)


if __name__ == "__main__":
    main()
