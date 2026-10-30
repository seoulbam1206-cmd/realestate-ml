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

from realestate.config import BJD_RAW, sources
from realestate.ingest import bjd
from realestate.ingest import log as ingest_log
from realestate.ingest.molit_collect import SOURCE, Collector
from realestate.ingest.molit_jobs import Job, run_jobs

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
    targets = [Job(JOB, ym, sgg) for ym in sorted(args.months, reverse=True) for sgg in lawds]

    with ingest_log.connect() as con:
        done = set() if args.force else ingest_log.done_targets(con, SOURCE, JOB)
        todo = [j for j in targets if j.target_key not in done]
        if args.limit_jobs:
            todo = todo[: args.limit_jobs]
        col = Collector(con, cap=args.cap, max_retries=cfg["max_retries"])
        print(
            f"대상 {len(targets)}개 (시군구 {len(lawds)} × {len(args.months)}개월), 남은 작업 {len(todo)}개"
        )
        print(f"오늘 호출 {col.used_today()}회 사용 / 상한 {col.cap}회")

        out = run_jobs(con, col, todo, run_id, full_nm, cfg["max_consecutive_fail"])

        print(
            f"\n이번 실행: 성공 {out.n_ok}, 실패 {out.n_fail}, 오늘 호출 {col.used_today()}회 / 상한 {col.cap}회"
        )
        if out.stop_reason:
            print(f"멈춤: {out.stop_reason}")
        done = ingest_log.done_targets(con, SOURCE, JOB)
        left = sum(j.target_key not in done for j in targets)
        print(f"남은 작업 {left}개" + (" → 내일 같은 명령으로 이어서 실행" if left else " → 백필 완료"))


if __name__ == "__main__":
    main()
