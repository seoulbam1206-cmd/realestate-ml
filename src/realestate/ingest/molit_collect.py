"""국토부 실거래 API 수집기: (시군구, 월) 하나를 페이지 끝까지 받는다. 재시도, 하루 상한에서 멈춤.

- 호출하기 **전에** api_quota 의 오늘 사용량을 보고, 상한(cap)에 닿았으면 호출하지 않고 QuotaStop
- 실패한 호출도 한도에서 빠진다고 보고 재시도 1회 = 호출 1회로 센다
- 결과 코드 분류는 molit_api.*_CODES (PLAN.md 6-1)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from realestate.config import APT_TRADE_CLEAN, sources
from realestate.ingest import log as ingest_log
from realestate.ingest import molit_api

SOURCE = "molit_api"
BACKOFF_SECONDS = (2, 5, 15)


class QuotaStop(Exception):
    """오늘 더 부르면 안 됨 (우리 상한 도달 또는 서버가 한도 초과라고 답함)."""


class FatalApiError(Exception):
    """키·권한·요청 형식 문제. 재시도해도 안 되므로 실행 전체를 멈춘다."""


class RetryExhausted(Exception):
    """재시도를 다 써도 실패. 이 작업만 fail 로 두고 다음 작업으로 간다."""


@dataclass
class MonthResult:
    lawd_cd: str
    deal_ymd: str
    items: list[dict[str, str]] = field(default_factory=list)
    total_count: int = 0
    pages: int = 0
    api_calls: int = 0
    retries: int = 0
    http_status: int | None = None
    result_code: str | None = None
    result_msg: str | None = None
    raw_files: list[Path] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)


class Collector:
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        cap: int,
        fetch: Callable[..., molit_api.ApiResult] = molit_api.fetch,
        sleep: Callable[[float], None] = time.sleep,
        save_raw: Callable[..., Path] | None = molit_api.save_raw,
        max_retries: int = 3,
    ):
        cfg = sources()["molit_apt_trade"]
        self.con = con
        self.cap = min(cap, cfg["daily_limit"])
        self.daily_limit = cfg["daily_limit"]
        self.num_rows = cfg["num_of_rows"]
        self.fetch, self.sleep, self.save_raw, self.max_retries = fetch, sleep, save_raw, max_retries

    def used_today(self) -> int:
        return ingest_log.calls_today(self.con, SOURCE)

    def _call(self, out: MonthResult, page: int) -> molit_api.ApiResult:
        attempt = 0
        while True:
            used = self.used_today()
            if used >= self.cap:
                raise QuotaStop(f"오늘 {used}회 사용 → 상한 {self.cap}회에 닿아 멈춤")
            ingest_log.add_api_calls(self.con, SOURCE, 1, self.daily_limit)
            out.api_calls += 1
            attempt += 1
            try:
                res = self.fetch(out.lawd_cd, out.deal_ymd, page=page, num_rows=self.num_rows)
            except RuntimeError as e:  # 네트워크 오류 (메시지는 molit_api 에서 키 마스킹됨)
                problem = str(e)
            else:
                out.http_status, out.result_code, out.result_msg = (
                    res.http_status,
                    res.result_code,
                    res.result_msg,
                )
                code = res.result_code
                if code in molit_api.OK_CODES | molit_api.NO_DATA_CODES and res.http_status == 200:
                    return res
                if code in molit_api.QUOTA_CODES or res.http_status == 429:
                    raise QuotaStop(f"서버가 한도 초과로 응답 (HTTP {res.http_status}, code={code})")
                if code in molit_api.FATAL_CODES or res.http_status in (401, 403):
                    raise FatalApiError(f"HTTP {res.http_status}, code={code}, {res.result_msg}")
                problem = f"HTTP {res.http_status}, code={code}, {res.result_msg}"
            if attempt > self.max_retries:
                raise RetryExhausted(f"{attempt}회 시도 모두 실패: {problem}")
            out.retries += 1
            self.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])

    def collect_month(self, lawd_cd: str, deal_ymd: str, max_pages: int | None = None) -> MonthResult:
        """1페이지로 totalCount 를 보고, 남은 페이지를 차례로 받는다. 중간에 멈추면 예외 (받은 건 버림)."""
        out = MonthResult(lawd_cd, deal_ymd)
        page = 1
        while True:
            try:
                res = self._call(out, page)
            except (QuotaStop, FatalApiError, RetryExhausted) as e:
                e.partial = out  # 로그에 이 작업이 쓴 호출 수를 남기려고
                raise
            out.pages = page
            if self.save_raw:
                out.raw_files.append(self.save_raw(res.raw, lawd_cd, deal_ymd, page))
            out.items += res.items
            out.total_count = res.total_count or 0
            n_pages = max(1, math.ceil(out.total_count / self.num_rows))
            if page >= n_pages or not res.items or (max_pages and page >= max_pages):
                return out
            page += 1


def clean_path(lawd_cd: str, deal_ymd: str) -> Path:
    # CSV 정리본(ym=YYYYMM/source=csv.parquet) 옆에 시군구별로 둔다 → (시군구, 월) 통째로 교체
    return APT_TRADE_CLEAN / f"ym={deal_ymd}" / f"source=api_sgg={lawd_cd}.parquet"


MISSING_CONFIRM = 2  # 이전엔 있던 거래가 이만큼 연속으로 안 보이면 없어진 것으로 확정 (PLAN.md 6-1-4 ④)


@lru_cache(maxsize=2)
def _csv_month(deal_ymd: str) -> pd.DataFrame | None:
    path = APT_TRADE_CLEAN / f"ym={deal_ymd}" / "source=csv.parquet"
    return pd.read_parquet(path) if path.exists() else None


def previous_rows(lawd_cd: str, deal_ymd: str) -> pd.DataFrame | None:
    """비교 기준이 되는 이전 묶음: API 정리본이 있으면 그것, 처음 받는 달이면 같은 시군구의 CSV 행."""
    path = clean_path(lawd_cd, deal_ymd)
    if path.exists():
        return pd.read_parquet(path)
    csv = _csv_month(deal_ymd)
    return None if csv is None else csv[csv["sgg_cd"] == lawd_cd]


def merge_with_previous(
    new: pd.DataFrame, old: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """새 묶음 + 이전 묶음 → (저장할 표, 건수, 바뀐 거래 목록).

    - 전에 본 거래는 처음 본 시각(first_seen_at)을 유지
    - 이번에 안 보인 거래는 바로 지우지 않고 missing_count 를 1 올려 남긴다. MISSING_CONFIRM 번 연속이면 뺀다
    - 바뀐 거래: new(새로 생김) / canceled(해제로 바뀜) / removed(없어짐 확정). 이전 묶음이 없으면 기록하지 않음
    """
    new = new.copy()
    new["missing_count"] = 0
    if old is None or old.empty:
        counts = {"rows_new": len(new), "rows_removed": 0, "rows_missing": 0,
                  "rows_canceled": int(new["is_canceled"].sum())}  # fmt: skip
        return new, counts, pd.DataFrame(columns=["trade_key", "change_type"])
    if len(new):
        first = old.set_index("trade_key")["first_seen_at"]
        new["first_seen_at"] = new["trade_key"].map(first).fillna(new["first_seen_at"])

    new_keys = set(new["trade_key"])
    gone = old[~old["trade_key"].isin(new_keys)].copy()
    prev_missing = gone["missing_count"] if "missing_count" in gone else pd.Series(0, index=gone.index)
    gone["missing_count"] = prev_missing.fillna(0).astype(int) + 1
    confirmed = gone[gone["missing_count"] >= MISSING_CONFIRM]
    carry = gone[gone["missing_count"] < MISSING_CONFIRM]

    was_canceled = set(old.loc[old["is_canceled"], "trade_key"])
    added = new_keys - set(old["trade_key"])
    canceled = set(new.loc[new["is_canceled"], "trade_key"]) - was_canceled
    changes = pd.DataFrame(
        [(k, "new") for k in sorted(added)]
        + [(k, "canceled") for k in sorted(canceled)]
        + [(k, "removed") for k in confirmed["trade_key"]],
        columns=["trade_key", "change_type"],
    )
    out = pd.concat([new, carry], ignore_index=True) if len(carry) else new
    out["missing_count"] = out["missing_count"].astype("int8")
    counts = {"rows_new": len(added), "rows_removed": len(confirmed), "rows_missing": len(carry),
              "rows_canceled": len(canceled)}  # fmt: skip
    return out, counts, changes


def save_clean(df: pd.DataFrame, lawd_cd: str, deal_ymd: str) -> tuple[dict[str, int], pd.DataFrame]:
    out, counts, changes = merge_with_previous(df, previous_rows(lawd_cd, deal_ymd))
    path = clean_path(lawd_cd, deal_ymd)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return counts, changes
