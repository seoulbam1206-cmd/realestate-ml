"""ingest_log 로 수집 진행 상황 보기: 하루 호출 수, 월별 진행률, 최근 실패.

실행: uv run python scripts/ingest_status.py [--months 202610 202609]
"""

from __future__ import annotations

import argparse
import sys

from realestate.config import BJD_RAW, sources
from realestate.ingest import bjd
from realestate.ingest import log as ingest_log

SOURCE = "molit_api"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = sources()["molit_apt_trade"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=cfg["backfill_months"])
    args = parser.parse_args()
    n_sgg = len(bjd.lawd_codes(bjd.load(BJD_RAW)))

    with ingest_log.connect() as con:
        print("■ 하루 호출 수 (api_quota)")
        print(
            con.execute(
                "SELECT day AS 날짜, source AS 출처, calls_used AS 사용, daily_limit AS 한도 "
                "FROM api_quota ORDER BY day DESC LIMIT 14"
            )
            .df()
            .to_string(index=False)
        )
        print(f"  백필 상한: {cfg['backfill_daily_cap']}회/일 (오늘 전체 사용량 기준)")

        print(f"\n■ API 백필 진행 (시군구 {n_sgg}개 × 월)")
        # 같은 (시군구, 월)을 여러 번 시도했으면 마지막 상태로 센다
        prog = con.execute(
            """
            WITH last AS (
                SELECT *, row_number() OVER (PARTITION BY target_key ORDER BY finished_at DESC) AS rn
                FROM ingest_log WHERE source = ? AND job_type = 'backfill'
            ), ok AS (
                SELECT DISTINCT target_key FROM ingest_log
                WHERE source = ? AND job_type = 'backfill' AND status = 'success'
            )
            SELECT regexp_extract(l.target_key, 'ym=(\\d+)', 1) AS 월,
                   count(ok.target_key) AS 완료,
                   count(*) FILTER (WHERE ok.target_key IS NULL AND l.status = 'fail') AS 실패,
                   count(*) FILTER (WHERE ok.target_key IS NULL AND l.status = 'quota_stop') AS 한도멈춤,
                   coalesce(sum(l.rows_fetched) FILTER (WHERE l.status = 'success'), 0)::BIGINT AS 건수,
                   coalesce(sum(l.rows_canceled) FILTER (WHERE l.status = 'success'), 0)::BIGINT AS 해제
            FROM last l LEFT JOIN ok USING (target_key)
            WHERE l.rn = 1 GROUP BY 1 ORDER BY 1 DESC
            """,
            [SOURCE, SOURCE],
        ).df()
        done = dict(zip(prog["월"], prog["완료"], strict=True))
        for ym in sorted(args.months, reverse=True):
            if ym not in done:
                print(f"  {ym}: 완료 0/{n_sgg} (아직 시작 안 함)")
        for _, r in prog.iterrows():
            print(
                f"  {r['월']}: 완료 {r['완료']}/{n_sgg} ({r['완료'] / n_sgg:.0%}), 실패 {r['실패']}, "
                f"한도멈춤 {r['한도멈춤']}, 받은 거래 {r['건수']:,}건 (해제 {r['해제']:,})"
            )

        print("\n■ 실행별 요약 (API)")
        print(
            con.execute(
                """
                SELECT run_id, job_type, min(started_at)::TIMESTAMP(0) AS 시작, count(*) AS 작업,
                       count(*) FILTER (WHERE status = 'success') AS 성공,
                       count(*) FILTER (WHERE status = 'fail') AS 실패,
                       count(*) FILTER (WHERE status = 'quota_stop') AS 한도멈춤,
                       sum(api_calls)::BIGINT AS 호출, sum(retry_count)::BIGINT AS 재시도, sum(rows_fetched)::BIGINT AS 건수
                FROM ingest_log WHERE source = ? GROUP BY 1, 2 ORDER BY 3
                """,
                [SOURCE],
            )
            .df()
            .to_string(index=False)
        )

        fails = con.execute(
            """
            SELECT finished_at::TIMESTAMP(0) AS 시각, job_type, target_key, status, result_code, error_msg
            FROM ingest_log WHERE source = ? AND status IN ('fail', 'quota_stop')
            ORDER BY finished_at DESC LIMIT 10
            """,
            [SOURCE],
        ).df()
        if len(fails):
            print("\n■ 최근 실패·멈춤")
            print(fails.to_string(index=False))

        csv = con.execute(
            """
            WITH last AS (
                SELECT target_key, rows_fetched,
                       row_number() OVER (PARTITION BY target_key ORDER BY finished_at DESC) AS rn
                FROM ingest_log WHERE source = 'molit_csv' AND status = 'success'
            )
            SELECT count(*), sum(rows_fetched), min(target_key), max(target_key) FROM last WHERE rn = 1
            """
        ).fetchone()
        print(f"\n■ CSV 정리본: {csv[0]}개월 ({csv[2][3:]}~{csv[3][3:]}), {csv[1]:,}건" if csv[0] else "")


if __name__ == "__main__":
    main()
