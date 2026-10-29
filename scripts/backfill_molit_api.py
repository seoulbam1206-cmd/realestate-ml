"""국토부 실거래 API 백필: 시군구 × 월 단위로 받아 data/clean/molit/apt_trade/ym=YYYYMM/source=api_sgg=XXXXX.parquet

- 대상 월: config/sources.yaml 의 backfill_months (기본 2026.10, 2026.9. 2026.8 이전은 CSV)
- 시군구: 법정동 코드표에서 지금 존재하는 시군구 (일반구가 있는 시는 구 코드)
- 이미 성공한 (시군구, 월)은 건너뜀 → 멈춘 뒤 다시 실행하면 이어서 받음 (--force 로 다시)
- 하루 상한(backfill_daily_cap, 기본 800회)에 닿으면 멈추고 quota_stop 으로 기록
- 진행 상황: uv run python scripts/ingest_status.py

실행: uv run python scripts/backfill_molit_api.py [--months 202610 202609] [--cap 800] [--limit-jobs N] [--force]
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from realestate.config import BJD_RAW, sources
from realestate.ingest import bjd, molit_api
from realestate.ingest import log as ingest_log
from realestate.ingest.molit_collect import (
    SOURCE,
    Collector,
    FatalApiError,
    QuotaStop,
    save_clean,
)

JOB = "backfill"


def main() -> None:
    cfg = sources()["molit_apt_trade"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=cfg["backfill_months"])
    parser.add_argument("--cap", type=int, default=cfg["backfill_daily_cap"], help="오늘 전체 호출 상한")
    parser.add_argument("--limit-jobs", type=int, help="이번 실행에서 처리할 (시군구, 월) 최대 개수 (시험용)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    codes = bjd.load(BJD_RAW)
    lawds = bjd.lawd_codes(codes)
    full_nm = dict(zip(codes["bjd_cd"], codes["full_nm"], strict=True))
    run_id = ingest_log.new_run_id()
    targets = [(ym, sgg) for ym in sorted(args.months, reverse=True) for sgg in lawds]

    with ingest_log.connect() as con:
        done = set() if args.force else ingest_log.done_targets(con, SOURCE, JOB)
        todo = [(ym, sgg) for ym, sgg in targets if f"sgg={sgg},ym={ym}" not in done]
        if args.limit_jobs:
            todo = todo[: args.limit_jobs]
        col = Collector(con, cap=args.cap, max_retries=cfg["max_retries"])
        print(
            f"대상 {len(targets)}개 (시군구 {len(lawds)} × {len(args.months)}개월), 남은 작업 {len(todo)}개"
        )
        print(f"오늘 호출 {col.used_today()}회 사용 / 상한 {col.cap}회")

        n_ok = n_fail = consecutive_fail = 0
        stop_reason = None
        for i, (ym, sgg) in enumerate(todo, 1):
            common = dict(
                run_id=run_id, source=SOURCE, job_type=JOB, target_key=f"sgg={sgg},ym={ym}",
                started_at=datetime.now(), period_from=pd.Timestamp(f"{ym}01").date(),
            )  # fmt: skip
            res = None
            try:
                res = col.collect_month(sgg, ym)
                df = molit_api.to_clean(res.items, res.fetched_at, full_nm)
                outside = (df["deal_date"].dt.strftime("%Y%m") != ym).sum() if len(df) else 0
                counts = save_clean(df, sgg, ym)
            except QuotaStop as e:
                stop_reason = str(e)
                ingest_log.write(con, status="quota_stop", error_msg=stop_reason, **_res_fields(e), **common)
                break
            except FatalApiError as e:
                stop_reason = f"키·권한 문제로 중단: {e}"
                ingest_log.write(con, status="fail", error_msg=str(e)[:500], **_res_fields(e), **common)
                break
            except Exception as e:  # noqa: BLE001  재시도 소진·파싱 오류 등은 이 작업만 fail, 다음으로
                n_fail += 1
                consecutive_fail += 1
                ingest_log.write(
                    con,
                    status="fail",
                    error_msg=str(e)[:500],
                    **_res_fields(getattr(e, "partial", res)),
                    **common,
                )
                print(f"실패   {ym} {sgg}  {e}")
                if consecutive_fail >= cfg["max_consecutive_fail"]:
                    stop_reason = f"연속 {consecutive_fail}회 실패로 중단"
                    break
                continue

            consecutive_fail = 0
            n_ok += 1
            warnings = []
            if res.total_count != len(res.items):
                warnings.append(f"totalCount {res.total_count} ≠ 받은 {len(res.items)}건")
            if outside:
                warnings.append(f"요청 월 밖 계약일 {outside}건")
            ingest_log.write(
                con, status="success", rows_fetched=len(df), rows_unmatched=int((~df["bjd_cd"].isin(full_nm)).sum()),
                source_file=res.raw_files[-1].name if res.raw_files else None,
                warning="; ".join(warnings) or None, **counts, **_res_fields(res), **common,
            )  # fmt: skip
            if i % 25 == 0 or res.pages > 1:
                print(
                    f"진행   {i}/{len(todo)}  {ym} {sgg}  {len(df):,}건 {res.pages}페이지  오늘 {col.used_today()}회"
                )

        print(f"\n이번 실행: 성공 {n_ok}, 실패 {n_fail}, 오늘 호출 {col.used_today()}회 / 상한 {col.cap}회")
        if stop_reason:
            print(f"멈춤: {stop_reason}")
        done = ingest_log.done_targets(con, SOURCE, JOB)
        left = sum(f"sgg={sgg},ym={ym}" not in done for ym, sgg in targets)
        print(f"남은 작업 {left}개" + (" → 내일 같은 명령으로 이어서 실행" if left else " → 백필 완료"))


def _res_fields(res) -> dict:
    res = getattr(res, "partial", res)  # 예외면 그 안의 부분 결과
    if res is None:
        return {}
    return dict(
        http_status=res.http_status, result_code=res.result_code, result_msg=res.result_msg,
        pages=res.pages, api_calls=res.api_calls, retry_count=res.retries,
    )  # fmt: skip


if __name__ == "__main__":
    main()
