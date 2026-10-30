"""국토부 실거래 API 작업: 무엇을 받을지 정하고(plan_daily), 차례로 받아 저장·기록한다(run_jobs).

작업 하나 = (시군구, 월). target_key 는 `sgg=XXXXX,ym=YYYYMM` 로 백필·매일·점검이 같다.

매일 이어받기 (PLAN.md 6-1-4)
- daily  : 이번 달 + 지난달 × 전체 시군구. 오늘 이미 성공한 것은 건너뜀
- rotate : 2~12개월 전 × 전체 시군구. 마지막 성공이 rotate_every_days(14일) 이전이거나 없으면 대상.
           안 받은 것 → 오래전에 받은 것 순, 같으면 최근 달부터. 하루에 못 끝낸 몫은 다음 날로 넘어감
- 상한(daily_cap)은 Collector 가 호출 전에 확인 → 닿으면 quota_stop 으로 남기고 멈춤
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd

from realestate.ingest import log as ingest_log
from realestate.ingest import molit_api
from realestate.ingest.molit_collect import SOURCE, Collector, FatalApiError, QuotaStop, save_clean

API_JOB_TYPES = ("backfill", "daily", "rotate", "catchup")


@dataclass(frozen=True)
class Job:
    job_type: str
    ym: str
    sgg: str

    @property
    def target_key(self) -> str:
        return f"sgg={self.sgg},ym={self.ym}"


@dataclass
class RunSummary:
    n_ok: int = 0
    n_fail: int = 0
    stop_reason: str | None = None
    fatal: bool = False  # 키·권한 문제 → 스케줄러에 실패로 알려야 함
    rows: int = 0
    rows_new: int = 0
    rows_canceled: int = 0
    rows_removed: int = 0


def shift_ym(ym: str, months: int) -> str:
    p = pd.Period(f"{ym[:4]}-{ym[4:]}", freq="M") + months
    return f"{p.year}{p.month:02d}"


def plan_daily(
    today: date,
    lawds: list[str],
    last_ok: dict[str, datetime],
    rotate_from: int = 2,
    rotate_to: int = 12,
    rotate_every_days: int = 14,
) -> list[Job]:
    this_ym = f"{today:%Y%m}"
    jobs = [
        Job("daily", ym, sgg)
        for ym in (this_ym, shift_ym(this_ym, -1))
        for sgg in lawds
        if (t := last_ok.get(f"sgg={sgg},ym={ym}")) is None or t.date() < today
    ]
    due_before = today - timedelta(days=rotate_every_days - 1)  # 14일 주기면 13일 전까지 받은 것은 아직
    rotate = []
    for k in range(rotate_from, rotate_to + 1):
        ym = shift_ym(this_ym, -k)
        for sgg in lawds:
            t = last_ok.get(f"sgg={sgg},ym={ym}")
            if t is None or t.date() < due_before:
                rotate.append((t or datetime.min, -int(ym), sgg, Job("rotate", ym, sgg)))
    return jobs + [j for *_, j in sorted(rotate)]


def run_jobs(
    con: duckdb.DuckDBPyConnection,
    col: Collector,
    jobs: list[Job],
    run_id: str,
    full_nm: dict[str, str],
    max_consecutive_fail: int,
    log: Callable[[str], None] = print,
) -> RunSummary:
    out = RunSummary()
    consecutive_fail = 0
    for i, job in enumerate(jobs, 1):
        common = dict(
            run_id=run_id, source=SOURCE, job_type=job.job_type, target_key=job.target_key,
            started_at=datetime.now(), period_from=pd.Timestamp(f"{job.ym}01").date(),
        )  # fmt: skip
        res = None
        try:
            res = col.collect_month(job.sgg, job.ym)
            df = molit_api.to_clean(res.items, res.fetched_at, full_nm)
            outside = (df["deal_date"].dt.strftime("%Y%m") != job.ym).sum() if len(df) else 0
            counts, changes = save_clean(df, job.sgg, job.ym)
        except QuotaStop as e:
            out.stop_reason = str(e)
            ingest_log.write(con, status="quota_stop", error_msg=out.stop_reason, **_res_fields(e), **common)
            break
        except FatalApiError as e:
            out.stop_reason, out.fatal = f"키·권한 문제로 중단: {e}", True
            ingest_log.write(con, status="fail", error_msg=str(e)[:500], **_res_fields(e), **common)
            break
        except Exception as e:  # noqa: BLE001  재시도 소진·파싱 오류 등은 이 작업만 fail, 다음으로
            out.n_fail += 1
            consecutive_fail += 1
            ingest_log.write(
                con,
                status="fail",
                error_msg=str(e)[:500],
                **_res_fields(getattr(e, "partial", res)),
                **common,
            )
            log(f"실패   {job.job_type} {job.ym} {job.sgg}  {e}")
            if consecutive_fail >= max_consecutive_fail:
                out.stop_reason = f"연속 {consecutive_fail}회 실패로 중단"
                break
            continue

        consecutive_fail = 0
        out.n_ok += 1
        out.rows += len(df)
        out.rows_new += counts["rows_new"]
        out.rows_canceled += counts["rows_canceled"]
        out.rows_removed += counts["rows_removed"]
        ingest_log.write_changes(con, changes, job.sgg, job.ym, run_id)
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
        if i % 50 == 0 or res.pages > 1:
            log(
                f"진행   {i}/{len(jobs)}  {job.job_type} {job.ym} {job.sgg}  {len(df):,}건 {res.pages}페이지  "
                f"오늘 {col.used_today()}회"
            )
    return out


def _res_fields(res) -> dict:
    res = getattr(res, "partial", res)  # 예외면 그 안의 부분 결과
    if res is None:
        return {}
    return dict(
        http_status=res.http_status, result_code=res.result_code, result_msg=res.result_msg,
        pages=res.pages, api_calls=res.api_calls, retry_count=res.retries,
    )  # fmt: skip
