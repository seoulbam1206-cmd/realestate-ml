"""실거래 정리본 읽기: 한 달에 CSV 와 API 가 둘 다 있으면 어느 쪽을 쓸지 정한다.

정리본 모양: data/clean/molit/apt_trade/ym=YYYYMM/
    source=csv.parquet            공개시스템 CSV (전국 한 파일)
    source=api_sgg=XXXXX.parquet  API (시군구별, 매일 이어받기·점검으로 교체됨)

규칙 (달 단위, 섞지 않음 → 같은 거래가 두 번 들어가지 않음)
- 그 달 API 가 전체 시군구를 다 갖고 있고, 가장 오래전에 받은 API 시군구도 CSV 다운로드보다 나중이면 → API
  (API 쪽이 해제·늦은 신고가 더 반영된 최신 상태. CSV 를 새로 받아 다시 넣으면 CSV 가 다시 이김)
  시각은 파일 안 값으로 비교: CSV 는 first_seen_at(다운로드 시각), API 는 last_seen_at(받은 시각).
  파일 수정 시각은 쓰지 않음 (키 재계산처럼 내용 그대로 다시 쓰면 바뀌므로, 2026-10-30)
- 그 밖에는 CSV. CSV 가 없으면 있는 API 파일 (빠진 시군구가 있으면 api_partial)
- 시군구 코드가 바뀐 지역(인천 2026.7 등)은 API 가 새 코드로 과거 달도 주므로, 달 단위 선택이 안전하다
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from realestate.config import APT_TRADE_CLEAN

CSV_FILE = "source=csv.parquet"


def _api_sgg(path: Path) -> str:
    return path.stem.split("sgg=")[-1]


def _time(path: Path, col: str, agg: str) -> pd.Timestamp | None:
    """파일 안 시각 칸의 max/min (빈 파일·칸 없음이면 None)."""
    if col not in pq.read_schema(path).names:
        return None
    v = getattr(pc, agg)(pq.read_table(path, columns=[col])[col]).as_py()
    return None if v is None else pd.Timestamp(v)


def api_newer_than_csv(api: list[Path], csv: Path) -> bool:
    csv_time = _time(csv, "first_seen_at", "max")
    fetched = [t for p in api if (t := _time(p, "last_seen_at", "max")) is not None]
    if csv_time is None or not fetched:
        return bool(fetched)
    return min(fetched) > csv_time


def month_sources(lawds: list[str], root: Path = APT_TRADE_CLEAN) -> pd.DataFrame:
    """달마다: CSV 유무, API 시군구 수, 고른 출처(use), 읽을 파일 목록(files)."""
    want = set(lawds)
    rows = []
    for d in sorted(root.glob("ym=*")):
        csv = d / CSV_FILE
        api = [p for p in d.glob("source=api_sgg=*.parquet") if _api_sgg(p) in want]
        api_complete = {_api_sgg(p) for p in api} >= want
        if api_complete and (not csv.exists() or api_newer_than_csv(api, csv)):
            use, files = "api", sorted(api)
        elif csv.exists():
            use, files = "csv", [csv]
        else:
            use, files = ("api" if api_complete else "api_partial"), sorted(api)
        rows.append(
            {"ym": d.name[3:], "has_csv": csv.exists(), "n_api": len(api), "use": use, "files": files}
        )
    return pd.DataFrame(rows, columns=["ym", "has_csv", "n_api", "use", "files"])


def trade_files(
    lawds: list[str], months: list[str] | None = None, root: Path = APT_TRADE_CLEAN
) -> list[Path]:
    src = month_sources(lawds, root)
    if months is not None:
        src = src[src["ym"].isin(months)]
    return [p for files in src["files"] for p in files]
