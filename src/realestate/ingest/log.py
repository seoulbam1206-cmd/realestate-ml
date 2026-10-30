"""ingest_log: 어떤 자료를 언제 몇 행 받았는지 기록한다 (PLAN.md 5-4).

로컬 DuckDB 파일(data/clean/ingest.duckdb)에 쌓는다. 키·요청 URL은 저장하지 않는다.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from realestate.config import INGEST_DB

STATUSES = ("pending", "success", "fail", "quota_stop", "skipped")

_DDL = """
CREATE TABLE IF NOT EXISTS ingest_log (
    run_id        VARCHAR NOT NULL,
    source        VARCHAR NOT NULL,      -- molit_csv / molit_api / onbid_xxx / rone
    job_type      VARCHAR NOT NULL,      -- csv_import / probe / backfill / daily / catchup / manual
    target_key    VARCHAR NOT NULL,      -- 예: ym=202512  /  sgg=11110,ym=202512
    status        VARCHAR NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP,
    source_file   VARCHAR,
    file_sha256   VARCHAR,
    period_from   DATE,
    period_to     DATE,
    http_status   INTEGER,
    result_code   VARCHAR,
    result_msg    VARCHAR,
    pages         INTEGER,
    rows_fetched  INTEGER,
    rows_new      INTEGER,
    rows_canceled INTEGER,
    rows_removed  INTEGER,
    rows_unmatched INTEGER,
    retry_count   INTEGER DEFAULT 0,
    api_calls     INTEGER DEFAULT 0,
    warning       VARCHAR,
    error_msg     VARCHAR
);
CREATE TABLE IF NOT EXISTS api_quota (
    day         DATE NOT NULL,
    source      VARCHAR NOT NULL,
    calls_used  INTEGER NOT NULL,
    daily_limit INTEGER NOT NULL,
    PRIMARY KEY (day, source)
);
-- 거래 단위 변화 기록 (PLAN.md 5-2 apt_trade_change). 이전 묶음이 있을 때만 남긴다
CREATE TABLE IF NOT EXISTS apt_trade_change (
    trade_key   VARCHAR NOT NULL,
    sgg_cd      VARCHAR NOT NULL,
    ym          VARCHAR NOT NULL,
    change_type VARCHAR NOT NULL,      -- new / canceled / removed
    detected_at TIMESTAMP NOT NULL,
    run_id      VARCHAR NOT NULL
);
ALTER TABLE ingest_log ADD COLUMN IF NOT EXISTS rows_missing INTEGER;  -- 이번에 안 보인 거래 (확정 전)
"""


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


@contextmanager
def connect(path: Path = INGEST_DB) -> Iterator[duckdb.DuckDBPyConnection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute(_DDL)
        yield con
    finally:
        con.close()


def write(con: duckdb.DuckDBPyConnection, **fields) -> None:
    if fields.get("status") not in STATUSES:
        raise ValueError(f"알 수 없는 status: {fields.get('status')}")
    fields.setdefault("finished_at", datetime.now())
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    con.execute(f"INSERT INTO ingest_log ({cols}) VALUES ({marks})", list(fields.values()))


def add_api_calls(con: duckdb.DuckDBPyConnection, source: str, calls: int, daily_limit: int) -> int:
    """오늘 사용한 호출 수를 더하고, 더한 뒤의 합계를 돌려준다."""
    today = datetime.now().date()
    con.execute(
        "INSERT INTO api_quota VALUES (?, ?, ?, ?) "
        "ON CONFLICT (day, source) DO UPDATE SET calls_used = calls_used + excluded.calls_used",
        [today, source, calls, daily_limit],
    )
    return con.execute(
        "SELECT calls_used FROM api_quota WHERE day = ? AND source = ?", [today, source]
    ).fetchone()[0]


def calls_today(con: duckdb.DuckDBPyConnection, source: str) -> int:
    row = con.execute(
        "SELECT calls_used FROM api_quota WHERE day = ? AND source = ?", [datetime.now().date(), source]
    ).fetchone()
    return row[0] if row else 0


def done_targets(con: duckdb.DuckDBPyConnection, source: str, job_type: str) -> set[str]:
    rows = con.execute(
        "SELECT DISTINCT target_key FROM ingest_log WHERE source = ? AND job_type = ? AND status = 'success'",
        [source, job_type],
    ).fetchall()
    return {r[0] for r in rows}


def last_success(
    con: duckdb.DuckDBPyConnection, source: str, job_types: tuple[str, ...]
) -> dict[str, datetime]:
    """target_key → 마지막 성공 시각 (여러 job_type 을 합쳐서)."""
    marks = ", ".join("?" for _ in job_types)
    rows = con.execute(
        f"SELECT target_key, max(finished_at) FROM ingest_log "
        f"WHERE source = ? AND status = 'success' AND job_type IN ({marks}) GROUP BY 1",
        [source, *job_types],
    ).fetchall()
    return dict(rows)


def write_changes(
    con: duckdb.DuckDBPyConnection, changes: pd.DataFrame, sgg_cd: str, ym: str, run_id: str
) -> None:
    if changes.empty:
        return
    df = changes[["trade_key", "change_type"]].assign(
        sgg_cd=sgg_cd, ym=ym, detected_at=datetime.now(), run_id=run_id
    )
    with con.cursor() as cur:  # 같은 연결의 앞선 INSERT 결과가 남아 있어도 막히지 않게 별도 커서
        cur.register("_changes", df)
        cur.execute(
            "INSERT INTO apt_trade_change "
            "SELECT trade_key, sgg_cd, ym, change_type, detected_at, run_id FROM _changes"
        )


def imported_hashes(con: duckdb.DuckDBPyConnection, source: str) -> set[str]:
    rows = con.execute(
        "SELECT file_sha256 FROM ingest_log WHERE source = ? AND status = 'success' AND file_sha256 IS NOT NULL",
        [source],
    ).fetchall()
    return {r[0] for r in rows}


def summary(con: duckdb.DuckDBPyConnection, source: str | None = None) -> pd.DataFrame:
    where = "WHERE source = ?" if source else ""
    return con.execute(
        f"""
        SELECT source, job_type, target_key, status, period_from, rows_fetched, rows_canceled,
               rows_unmatched, api_calls, finished_at, warning
        FROM ingest_log {where}
        ORDER BY finished_at
        """,
        [source] if source else [],
    ).df()
