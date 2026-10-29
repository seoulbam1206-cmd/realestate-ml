"""국토부 실거래 API 1회 시험 호출 → 응답 항목을 CSV 컬럼과 비교.

기본: 서울 종로구(11110) 2025년 12월 1페이지. 호출은 정확히 1번만 한다.
실행:
  uv run python scripts/probe_molit_api.py [--lawd 11110] [--ym 202512]
  uv run python scripts/probe_molit_api.py --from-raw <저장된 .xml.gz>   # 호출 없이 저장된 응답만 분석
"""

from __future__ import annotations

import argparse
import gzip
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from realestate.config import APT_TRADE_CLEAN, sources
from realestate.ingest import log as ingest_log
from realestate.ingest import molit_api

SOURCE = "molit_api"


def report(res: molit_api.ApiResult, lawd: str, ym: str) -> None:
    print(f"HTTP {res.http_status} / resultCode={res.result_code} / resultMsg={res.result_msg}")
    print(f"totalCount={res.total_count}, 받은 item={len(res.items)}")
    if not res.items:
        return

    fields = list(dict.fromkeys(k for it in res.items for k in it))
    print(f"\n응답 항목 {len(fields)}개:")
    for f in fields:
        vals = Counter(it.get(f, "") for it in res.items)
        sample = [v for v, _ in vals.most_common(4) if v][:3]
        print(f"  {f:<20} 빈값 {vals.get('', 0):>3}/{len(res.items)}  예: {sample}")

    csv_path = APT_TRADE_CLEAN / f"ym={ym}" / "source=csv.parquet"
    if not csv_path.exists():
        return
    csv = pd.read_parquet(csv_path)
    csv = csv[csv["sgg_cd"] == lawd]
    api = pd.DataFrame(res.items)
    api_key = (
        api["umdNm"].str.strip() + "|" + api["dealDay"].str.zfill(2) + "|"
        + api["dealAmount"].str.replace(",", "").str.strip() + "|" + api["floor"].str.strip() + "|"
        + api["excluUseAr"].astype(float).round(2).astype(str)
    )  # fmt: skip
    csv_key = (
        csv["umd_nm"] + "|" + csv["deal_date"].dt.strftime("%d") + "|" + csv["deal_amount"].astype(str) + "|"
        + csv["floor"].astype(str) + "|" + csv["exclu_use_ar"].round(2).astype(str)
    )  # fmt: skip
    a, c = Counter(api_key), Counter(csv_key)
    print(f"\nCSV(시군구 {lawd}, {ym}) {len(csv)}건 vs API {len(api)}건, 내용 일치 {sum((a & c).values())}건")
    print(f"  API에만: {sum((a - c).values())}건, CSV에만: {sum((c - a).values())}건")
    canceled_api = (api["cdealType"].str.strip() != "").sum() if "cdealType" in api else "항목 없음"
    print(f"  해제: CSV {int(csv['is_canceled'].sum())}건, API cdealType 값 있음 {canceled_api}건")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lawd", default="11110")
    parser.add_argument("--ym", default="202512")
    parser.add_argument("--from-raw", type=Path, help="호출하지 않고 저장된 응답(.xml.gz)만 분석")
    args = parser.parse_args()
    cfg = sources()["molit_apt_trade"]

    if args.from_raw:
        raw = gzip.decompress(args.from_raw.read_bytes())
        code, msg, total, items = molit_api.parse_xml(raw)
        report(molit_api.ApiResult(None, code, msg, total, items, raw), args.lawd, args.ym)
        return

    common = dict(
        run_id=ingest_log.new_run_id(), source=SOURCE, job_type="probe",
        target_key=f"sgg={args.lawd},ym={args.ym}", started_at=datetime.now(), pages=1, api_calls=1,
    )  # fmt: skip
    with ingest_log.connect() as con:
        used = ingest_log.add_api_calls(con, SOURCE, 1, cfg["daily_limit"])
        try:
            res = molit_api.fetch(args.lawd, args.ym, page=1)
        except RuntimeError as e:
            ingest_log.write(con, status="fail", error_msg=str(e)[:500], **common)
            print(f"호출 실패: {e}")
            return
        raw_path = molit_api.save_raw(res.raw, args.lawd, args.ym, 1)
        ok = res.result_code in molit_api.OK_CODES
        ingest_log.write(
            con, status="success" if ok else "fail", http_status=res.http_status, result_code=res.result_code,
            result_msg=res.result_msg, rows_fetched=len(res.items),
            rows_canceled=sum(1 for i in res.items if i.get("cdealType", "").strip()),
            source_file=raw_path.name, **common,
        )  # fmt: skip
    print(f"오늘 호출 누계 {used}/{cfg['daily_limit']}, 원본 저장: {raw_path}")
    report(res, args.lawd, args.ym)


if __name__ == "__main__":
    main()
