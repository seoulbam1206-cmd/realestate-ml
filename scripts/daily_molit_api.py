"""국토부 실거래 API 매일 이어받기 (PLAN.md 6-1-4). Windows 작업 스케줄러가 하루 한 번 실행.

- 이번 달 + 지난달: 매일 전체 시군구 다시 받기 (job_type=daily)
- 2~12개월 전: 마지막으로 받은 지 14일 넘은 (시군구, 월)부터 남은 호출로 받기 (job_type=rotate)
  → 각 달이 2주에 한 번 갱신. 하루 못 끝낸 몫·실패한 작업은 다음 날 자동으로 다시 대상이 됨
- 하루 상한 daily_cap(800회, 오늘 전체 사용량 기준)에 닿으면 호출 전에 멈춤 (quota_stop, 정상 종료)
- (시군구, 월) 통째로 교체, 이전 묶음과 비교해 new/canceled/removed 를 apt_trade_change 에 기록
- 같은 날 다시 실행해도 오늘 성공한 daily 작업은 건너뜀

종료 코드: 0 정상(상한 멈춤 포함), 1 실패한 작업 있음(다음 날 다시 시도), 2 키·권한 문제(직접 확인 필요)

실행: uv run python scripts/daily_molit_api.py [--dry-run] [--cap 800] [--limit-jobs N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, datetime

from realestate.config import BJD_RAW, sources
from realestate.ingest import bjd
from realestate.ingest import log as ingest_log
from realestate.ingest.molit_collect import SOURCE, Collector
from realestate.ingest.molit_jobs import API_JOB_TYPES, plan_daily, run_jobs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = sources()["molit_apt_trade"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=cfg["daily_cap"], help="오늘 전체 호출 상한")
    parser.add_argument("--limit-jobs", type=int, help="이번 실행에서 처리할 작업 최대 개수 (시험용)")
    parser.add_argument("--dry-run", action="store_true", help="호출하지 않고 오늘 할 일만 보여줌")
    args = parser.parse_args()

    codes = bjd.load(BJD_RAW)
    lawds = bjd.lawd_codes(codes)
    full_nm = dict(zip(codes["bjd_cd"], codes["full_nm"], strict=True))
    today = date.today()
    rot_from, rot_to = cfg["rotate_months"]

    def plan(con):
        return plan_daily(
            today, lawds, ingest_log.last_success(con, SOURCE, API_JOB_TYPES),
            rot_from, rot_to, cfg["rotate_every_days"],
        )  # fmt: skip

    print(f"===== 매일 이어받기 {datetime.now():%Y-%m-%d %H:%M:%S} =====")
    with ingest_log.connect() as con:
        jobs = plan(con)
        if args.limit_jobs:
            jobs = jobs[: args.limit_jobs]
        col = Collector(con, cap=args.cap, max_retries=cfg["max_retries"])
        by_type = Counter(j.job_type for j in jobs)
        rot_months = Counter(j.ym for j in jobs if j.job_type == "rotate")
        print(f"시군구 {len(lawds)}개. 오늘 할 일: 매일 {by_type['daily']}개, 점검 {by_type['rotate']}개")
        if rot_months:
            print(
                "  점검 대상 (월: 작업 수) "
                + ", ".join(f"{ym}: {n}" for ym, n in sorted(rot_months.items(), reverse=True))
            )
        print(f"오늘 호출 {col.used_today()}회 사용 / 상한 {col.cap}회")
        if args.dry_run:
            return 0

        run_id = ingest_log.new_run_id()
        out = run_jobs(con, col, jobs, run_id, full_nm, cfg["max_consecutive_fail"])

        print(
            f"\n이번 실행: 성공 {out.n_ok}, 실패 {out.n_fail}, 받은 거래 {out.rows:,}건 "
            f"(새로 생김 {out.rows_new:,}, 해제로 바뀜 {out.rows_canceled:,}, 없어짐 확정 {out.rows_removed:,})"
        )
        print(f"오늘 호출 {col.used_today()}회 / 상한 {col.cap}회")
        if out.stop_reason:
            print(f"멈춤: {out.stop_reason}")
        left = Counter(j.job_type for j in plan(con))
        print(
            f"남은 할 일: 매일 {left['daily']}개, 점검 {left['rotate']}개"
            + (" → 다음 실행 때 이어서" if sum(left.values()) else "")
        )
    if out.fatal:
        return 2
    return 1 if out.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
